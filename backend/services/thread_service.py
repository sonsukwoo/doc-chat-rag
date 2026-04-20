"""thread 메타데이터 서비스 계층."""

from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any, Literal, TypedDict

from backend.app_db import (
    ChatRepository,
    CheckpointRepository,
    DocumentRepository,
    app_db_connection,
)
from backend.document_store import (
    build_document_paths,
    create_document_record,
    load_document_record,
    save_uploaded_pdf,
    sanitize_document_id,
    update_document_stage_record,
)
from backend.thread_identity import (
    build_thread_id,
    ensure_thread_metadata,
    resolve_thread_collection_name,
    THREAD_COLLECTION_NAME_METADATA_KEY,
)
from backend.stage3_indexing.config import (
    STAGE3_QDRANT_API_KEY,
    STAGE3_QDRANT_TIMEOUT,
    STAGE3_QDRANT_URL,
)
from backend.stage3_indexing.qdrant import QdrantRestClient


ThreadRetrievalMode = Literal["dense", "hybrid"]


class ThreadPayload(TypedDict, total=False):
    """프론트와 API가 공통으로 읽는 thread 응답 스키마."""

    thread_id: str
    thread_name: str
    collection_name: str
    description: str | None
    default_retrieval_mode: str
    metadata: dict[str, Any]
    active_document_ids: list[str]
    document_count: int
    created_at: str | None
    updated_at: str | None
    archived_at: str | None


class ThreadDocumentPayload(TypedDict, total=False):
    """thread 패널에서 노출할 문서 요약 스키마."""

    document_id: str
    original_filename: str
    uploaded_at: str | None
    stages: dict[str, Any]
    source_pdf_path: str


class ThreadDeletionPayload(TypedDict, total=False):
    """thread 영구 삭제 결과 요약."""

    thread_id: str
    deleted_document_ids: list[str]
    deleted_checkpoint_rows: dict[str, int]
    deleted_collection_name: str | None
    cleanup_warnings: list[str]


def _normalize_pdf_filename(original_filename: str) -> str:
    normalized_filename = str(original_filename or "").strip() or "uploaded.pdf"
    if not normalized_filename.lower().endswith(".pdf"):
        raise ValueError("only PDF upload is supported")
    return normalized_filename


def _build_thread_document_id(thread_id: str, original_filename: str) -> str:
    stem = Path(original_filename or "uploaded.pdf").stem
    return sanitize_document_id(f"{thread_id}__{stem}")


def _serialize_thread_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    public_metadata = dict(metadata or {})
    public_metadata.pop(THREAD_COLLECTION_NAME_METADATA_KEY, None)
    return public_metadata


def _normalize_thread_metadata_input(metadata: dict[str, Any] | None) -> dict[str, Any]:
    normalized_metadata = dict(metadata or {})
    normalized_metadata.pop(THREAD_COLLECTION_NAME_METADATA_KEY, None)
    return normalized_metadata


def _resolve_thread_collection_name(thread_row: dict[str, Any]) -> str:
    thread_id = str(thread_row.get("thread_id") or "")
    metadata = dict(thread_row.get("metadata") or {})
    return resolve_thread_collection_name(thread_id, metadata=metadata)


def _serialize_thread(
    thread_row: dict[str, Any],
    *,
    active_document_ids: list[str],
) -> ThreadPayload:
    thread_id = str(thread_row.get("thread_id") or "")
    metadata = dict(thread_row.get("metadata") or {})
    return {
        "thread_id": thread_id,
        "thread_name": str(thread_row.get("thread_name") or ""),
        "collection_name": resolve_thread_collection_name(
            thread_id,
            metadata=metadata,
        ),
        "description": (
            str(thread_row.get("description"))
            if thread_row.get("description") not in (None, "")
            else None
        ),
        "default_retrieval_mode": str(
            thread_row.get("default_retrieval_mode") or "dense"
        ),
        "metadata": _serialize_thread_metadata(metadata),
        "active_document_ids": active_document_ids,
        "document_count": len(active_document_ids),
        "created_at": (
            thread_row.get("created_at").isoformat()
            if thread_row.get("created_at") is not None
            else None
        ),
        "updated_at": (
            thread_row.get("updated_at").isoformat()
            if thread_row.get("updated_at") is not None
            else None
        ),
        "archived_at": (
            thread_row.get("archived_at").isoformat()
            if thread_row.get("archived_at") is not None
            else None
        ),
    }


def _serialize_document_record(document_id: str) -> ThreadDocumentPayload | None:
    try:
        record = load_document_record(document_id)
    except FileNotFoundError:
        return None

    paths = build_document_paths(document_id)
    return {
        "document_id": str(record.get("document_id") or document_id),
        "original_filename": str(record.get("original_filename") or ""),
        "uploaded_at": str(record.get("uploaded_at") or "") or None,
        "stages": dict(record.get("stages") or {}),
        "source_pdf_path": str(paths.source_pdf),
    }


def _cleanup_document_storage_roots(storage_roots: list[str]) -> list[str]:
    warnings: list[str] = []
    normalized_roots = sorted(
        {str(item).strip() for item in storage_roots if str(item).strip()}
    )
    for raw_root in normalized_roots:
        target_path = Path(raw_root).expanduser()
        if not target_path.exists():
            continue
        try:
            shutil.rmtree(target_path)
        except Exception as exc:
            warnings.append(f"문서 산출물 삭제 실패: {target_path} ({exc})")
    return warnings


def _cleanup_thread_collection(collection_name: str | None) -> list[str]:
    resolved_collection_name = str(collection_name or "").strip()
    if not resolved_collection_name or not STAGE3_QDRANT_URL:
        return []

    client = QdrantRestClient(
        base_url=STAGE3_QDRANT_URL,
        api_key=STAGE3_QDRANT_API_KEY,
        timeout=STAGE3_QDRANT_TIMEOUT,
    )
    try:
        client.delete_collection(resolved_collection_name)
    except Exception as exc:
        return [f"Qdrant 컬렉션 삭제 실패: {resolved_collection_name} ({exc})"]
    finally:
        client.close()
    return []


def list_threads(*, include_archived: bool = False) -> list[ThreadPayload]:
    """저장된 thread 목록을 반환한다."""
    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        document_repository = DocumentRepository(connection)
        threads = chat_repository.list_threads(include_archived=include_archived)
        return [
            _serialize_thread(
                thread,
                active_document_ids=document_repository.list_active_document_ids(
                    str(thread.get("thread_id") or "")
                ),
            )
            for thread in threads
        ]


def get_thread_detail(
    thread_id: str,
    *,
    include_archived: bool = False,
) -> ThreadPayload | None:
    """thread 상세 정보를 반환한다."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        document_repository = DocumentRepository(connection)
        thread = chat_repository.get_thread(normalized_thread_id)
        if thread is None:
            return None
        if thread.get("archived_at") is not None and not include_archived:
            return None
        return _serialize_thread(
            thread,
            active_document_ids=document_repository.list_active_document_ids(
                normalized_thread_id
            ),
        )


def create_thread(
    *,
    thread_name: str,
    description: str | None = None,
    default_retrieval_mode: ThreadRetrievalMode = "dense",
    metadata: dict[str, Any] | None = None,
) -> ThreadPayload:
    """새 thread를 생성한다."""
    normalized_thread_name = str(thread_name or "").strip()
    if not normalized_thread_name:
        raise ValueError("thread_name is required")

    thread_id = build_thread_id(normalized_thread_name)
    resolved_metadata = _normalize_thread_metadata_input(metadata)
    resolved_metadata.setdefault("lifecycle_status", "draft")
    resolved_metadata = ensure_thread_metadata(thread_id, resolved_metadata)

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        chat_repository.upsert_thread(
            thread_id=thread_id,
            thread_name=normalized_thread_name,
            description=str(description or "").strip() or None,
            default_retrieval_mode=default_retrieval_mode,
            metadata=resolved_metadata,
        )
        connection.commit()

    created = get_thread_detail(thread_id, include_archived=True)
    if created is None:
        raise RuntimeError("failed to load created thread")
    return created


def update_thread(
    thread_id: str,
    *,
    thread_name: str | None = None,
    description: str | None = None,
    default_retrieval_mode: ThreadRetrievalMode | None = None,
    metadata: dict[str, Any] | None = None,
) -> ThreadPayload | None:
    """기존 thread 메타데이터를 갱신한다."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        current = chat_repository.get_thread(normalized_thread_id)
        if current is None or current.get("archived_at") is not None:
            return None

        merged_metadata = dict(current.get("metadata") or {})
        if metadata is not None:
            merged_metadata.update(_normalize_thread_metadata_input(metadata))
        merged_metadata = ensure_thread_metadata(normalized_thread_id, merged_metadata)

        chat_repository.upsert_thread(
            thread_id=normalized_thread_id,
            thread_name=str(thread_name or current.get("thread_name") or "").strip()
            or normalized_thread_id,
            description=(
                str(description).strip()
                if description is not None
                else current.get("description")
            ),
            default_retrieval_mode=(
                default_retrieval_mode
                or str(current.get("default_retrieval_mode") or "dense")
            ),
            metadata=merged_metadata,
            last_user_message_at=current.get("last_user_message_at"),
        )
        connection.commit()

    return get_thread_detail(normalized_thread_id, include_archived=False)


def archive_thread(thread_id: str) -> ThreadPayload | None:
    """thread를 soft-delete 용도로 archive 처리한다."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        thread = chat_repository.get_thread(normalized_thread_id)
        if thread is None:
            return None
        chat_repository.archive_thread(normalized_thread_id)
        connection.commit()

    return get_thread_detail(normalized_thread_id, include_archived=True)


def delete_thread_permanently(thread_id: str) -> ThreadDeletionPayload | None:
    """thread와 연관 문서/체크포인트를 영구 삭제한다."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    deleted_document_ids: list[str] = []
    deleted_storage_roots: list[str] = []
    cleanup_warnings: list[str] = []
    deleted_checkpoint_rows: dict[str, int] = {}
    deleted_collection_name: str | None = None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        document_repository = DocumentRepository(connection)
        checkpoint_repository = CheckpointRepository(connection)

        thread = chat_repository.get_thread(normalized_thread_id)
        if thread is None:
            return None

        document_links = document_repository.list_thread_document_links(normalized_thread_id)
        deletable_document_ids = [
            str(item.get("document_id") or "").strip()
            for item in document_links
            if int(item.get("linked_thread_count") or 0) <= 1
        ]
        retained_document_ids = [
            str(item.get("document_id") or "").strip()
            for item in document_links
            if int(item.get("linked_thread_count") or 0) > 1
        ]

        deleted_document_rows = document_repository.delete_documents(deletable_document_ids)
        deleted_document_ids = [
            str(row.get("document_id") or "").strip()
            for row in deleted_document_rows
            if str(row.get("document_id") or "").strip()
        ]
        deleted_storage_roots = [
            str(row.get("storage_root") or "").strip()
            for row in deleted_document_rows
            if str(row.get("storage_root") or "").strip()
        ]

        deleted_checkpoint_rows = checkpoint_repository.delete_thread_checkpoints(
            normalized_thread_id
        )
        chat_repository.delete_thread(normalized_thread_id)
        connection.commit()

        deleted_collection_name = _resolve_thread_collection_name(thread)
        if retained_document_ids:
            cleanup_warnings.append(
                "다른 채팅방과 연결된 문서는 유지되었습니다: "
                + ", ".join(sorted(retained_document_ids))
            )

    cleanup_warnings.extend(_cleanup_document_storage_roots(deleted_storage_roots))
    cleanup_warnings.extend(_cleanup_thread_collection(deleted_collection_name))

    return {
        "thread_id": normalized_thread_id,
        "deleted_document_ids": deleted_document_ids,
        "deleted_checkpoint_rows": deleted_checkpoint_rows,
        "deleted_collection_name": deleted_collection_name,
        "cleanup_warnings": cleanup_warnings,
    }


def list_thread_document_records(thread_id: str) -> list[ThreadDocumentPayload]:
    """현재 thread에 연결된 문서 메타데이터를 반환한다."""
    thread = get_thread_detail(thread_id)
    if thread is None:
        return []
    records: list[ThreadDocumentPayload] = []
    for document_id in thread.get("active_document_ids") or []:
        record = _serialize_document_record(str(document_id))
        if record is not None:
            records.append(record)
    return records


def upload_document_to_thread(
    *,
    thread_id: str,
    original_filename: str,
    content: bytes,
) -> dict[str, Any]:
    """기존 thread에 문서를 업로드하고 연결한다."""
    normalized_thread_id = str(thread_id or "").strip()
    normalized_filename = _normalize_pdf_filename(original_filename)
    if not normalized_thread_id:
        raise ValueError("thread_id is required")

    thread = get_thread_detail(normalized_thread_id)
    if thread is None:
        raise FileNotFoundError("thread not found")

    resolved_original_filename = str(original_filename or "").strip() or normalized_filename
    document_id = _build_thread_document_id(normalized_thread_id, normalized_filename)
    paths = build_document_paths(document_id)
    if paths.root.exists():
        shutil.rmtree(paths.root)

    record = create_document_record(
        original_filename=resolved_original_filename,
        normalized_filename=normalized_filename,
        document_id=document_id,
    )
    saved_path = save_uploaded_pdf(document_id=document_id, content=content)
    updated_record = update_document_stage_record(
        document_id=document_id,
        stage="upload",
        status="uploaded",
        outputs={"source_pdf_path": str(saved_path)},
    )

    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        document_repository.upsert_document(
            document_id=document_id,
            original_filename=resolved_original_filename,
            normalized_filename=normalized_filename,
            storage_root=str(paths.root),
            source_pdf_path=str(saved_path),
            metadata={"thread_id": normalized_thread_id},
        )
        document_repository.attach_document_to_thread(
            thread_id=normalized_thread_id,
            document_id=document_id,
            slot_key=normalized_filename,
        )
        connection.commit()

    return {
        "thread": get_thread_detail(normalized_thread_id),
        "document": updated_record or record,
        "paths": {"source_pdf_path": str(saved_path)},
    }


def create_thread_with_document(
    *,
    thread_name: str,
    original_filename: str,
    content: bytes,
    description: str | None = None,
    default_retrieval_mode: ThreadRetrievalMode = "dense",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """draft thread를 생성하고 첫 문서를 바로 업로드한다."""
    normalized_filename = _normalize_pdf_filename(original_filename)
    thread = create_thread(
        thread_name=thread_name,
        description=description,
        default_retrieval_mode=default_retrieval_mode,
        metadata=metadata,
    )
    try:
        upload_result = upload_document_to_thread(
            thread_id=thread["thread_id"],
            original_filename=normalized_filename,
            content=content,
        )
    except Exception:
        archive_thread(thread["thread_id"])
        raise
    return {
        "thread": upload_result["thread"],
        "document": upload_result["document"],
        "paths": upload_result["paths"],
    }
