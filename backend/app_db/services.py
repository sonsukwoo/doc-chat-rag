"""Application Postgres service layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from backend.thread_identity import build_thread_collection_name

from .repositories import ChatRepository, DocumentRepository, app_db_connection


class ThreadRuntimeContext(TypedDict, total=False):
    """thread 기반 검색/대화에 필요한 최소 런타임 컨텍스트."""

    thread_id: str
    thread_name: str
    collection_name: str
    default_retrieval_mode: str
    active_document_ids: list[str]


class ExpandedContextBlockPayload(TypedDict, total=False):
    """child chunk를 상위 parent 문맥으로 확장한 결과."""

    document_id: str
    parent_id: str
    section_title: str | None
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    matched_chunk_ids: list[str]
    window_chunk_ids: list[str]
    context_text: str
    expansion_mode: str


class VisualAssetPayload(TypedDict, total=False):
    """표/이미지 원본 asset 메타데이터."""

    asset_ref: str
    document_id: str
    chunk_id: str
    asset_kind: str
    relative_path: str
    asset_stage: str
    page: int | None
    caption: str | None
    summary_text: str | None
    heading_path: list[str]
    pages: list[int]


def _normalize_document_ids(active_document_ids: list[str] | None) -> list[str]:
    return [
        str(item).strip()
        for item in active_document_ids or []
        if str(item).strip()
    ]


def _split_qualified_ref(value: str) -> tuple[str | None, str]:
    normalized = str(value or "").strip()
    if not normalized:
        return (None, "")
    if ":" not in normalized:
        return (None, normalized)
    document_id, chunk_or_asset_id = normalized.split(":", 1)
    return (document_id.strip() or None, chunk_or_asset_id.strip())


def _match_chunk_ref(
    *,
    document_id: str,
    chunk_id: str,
    requested_refs: list[str],
) -> bool:
    for requested_ref in requested_refs:
        requested_document_id, requested_chunk_id = _split_qualified_ref(requested_ref)
        if requested_chunk_id != chunk_id:
            continue
        if requested_document_id is None or requested_document_id == document_id:
            return True
    return False


def _slice_window_chunk_ids(
    *,
    child_chunk_ids: list[str],
    matched_chunk_ids: list[str],
    window_size: int,
) -> list[str]:
    if not child_chunk_ids or not matched_chunk_ids:
        return []
    matched_positions = [
        index
        for index, child_chunk_id in enumerate(child_chunk_ids)
        if child_chunk_id in matched_chunk_ids
    ]
    if not matched_positions:
        return []
    start = max(0, min(matched_positions) - max(0, window_size))
    end = min(len(child_chunk_ids), max(matched_positions) + max(0, window_size) + 1)
    return child_chunk_ids[start:end]


def load_thread_runtime_context(thread_id: str) -> ThreadRuntimeContext | None:
    """thread 메타데이터와 현재 연결된 문서 ID 목록을 함께 읽는다."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        document_repository = DocumentRepository(connection)
        thread = chat_repository.get_thread(normalized_thread_id)
        if thread is None:
            return None
        active_document_ids = document_repository.list_active_document_ids(
            normalized_thread_id
        )
        return {
            "thread_id": normalized_thread_id,
            "thread_name": str(thread.get("thread_name") or normalized_thread_id),
            "collection_name": build_thread_collection_name(normalized_thread_id),
            "default_retrieval_mode": str(
                thread.get("default_retrieval_mode") or "dense"
            ).strip()
            or "dense",
            "active_document_ids": active_document_ids,
        }


def try_load_thread_runtime_context(thread_id: str) -> ThreadRuntimeContext | None:
    """DB 연결 실패까지 포함해 안전하게 thread 컨텍스트를 읽는다."""
    try:
        return load_thread_runtime_context(thread_id)
    except Exception:
        return None


def sync_document_runtime_metadata(
    *,
    thread_id: str,
    document_id: str,
    original_filename: str,
    normalized_filename: str,
    storage_root: str | Path,
    source_pdf_path: str | None,
    parents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """stage3 이후 document/thread/parent/asset 메타데이터를 Postgres와 동기화한다."""
    normalized_storage_root = str(Path(storage_root).expanduser().resolve())
    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        document_repository.upsert_document(
            document_id=document_id,
            original_filename=original_filename,
            normalized_filename=normalized_filename,
            storage_root=normalized_storage_root,
            source_pdf_path=source_pdf_path,
            metadata={},
        )
        document_repository.attach_document_to_thread(
            thread_id=thread_id,
            document_id=document_id,
            slot_key=normalized_filename,
        )
        document_repository.replace_document_chunks(
            document_id=document_id,
            chunks=chunks,
        )
        document_repository.replace_document_parents(
            document_id=document_id,
            parents=parents,
        )
        document_repository.replace_document_assets(
            document_id=document_id,
            chunks=chunks,
        )


def load_expanded_context_blocks(
    *,
    thread_id: str | None,
    active_document_ids: list[str],
    chunk_ids: list[str],
    window_size: int = 1,
) -> list[ExpandedContextBlockPayload]:
    """현재 thread의 child chunk ids를 parent 문맥 블록으로 확장한다."""
    del thread_id
    normalized_document_ids = _normalize_document_ids(active_document_ids)
    normalized_chunk_refs = [str(item).strip() for item in chunk_ids if str(item).strip()]
    if not normalized_document_ids or not normalized_chunk_refs:
        return []

    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        parent_rows = document_repository.list_document_parents(normalized_document_ids)
        chunk_rows = document_repository.list_document_chunks(normalized_document_ids)

    chunk_text_lookup: dict[tuple[str, str], str] = {}
    chunk_ids_by_parent: dict[tuple[str, str], list[str]] = {}
    for row in chunk_rows:
        document_id = str(row.get("document_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        parent_id = str(row.get("parent_id") or "").strip()
        if not document_id or not chunk_id:
            continue
        chunk_text_lookup[(document_id, chunk_id)] = str(row.get("text") or "").strip()
        if parent_id:
            chunk_ids_by_parent.setdefault((document_id, parent_id), []).append(chunk_id)

    context_blocks: list[ExpandedContextBlockPayload] = []
    for row in parent_rows:
        document_id = str(row.get("document_id") or "").strip()
        parent_id = str(row.get("parent_id") or "").strip()
        child_chunk_ids = [
            str(item).strip()
            for item in row.get("chunk_ids") or []
            if str(item).strip()
        ]
        matched_chunk_ids = [
            child_chunk_id
            for child_chunk_id in child_chunk_ids
            if _match_chunk_ref(
                document_id=document_id,
                chunk_id=child_chunk_id,
                requested_refs=normalized_chunk_refs,
            )
        ]
        if not matched_chunk_ids:
            continue

        stored_chunk_ids = chunk_ids_by_parent.get((document_id, parent_id), [])
        ordered_chunk_ids = stored_chunk_ids or child_chunk_ids
        window_chunk_ids = _slice_window_chunk_ids(
            child_chunk_ids=ordered_chunk_ids,
            matched_chunk_ids=matched_chunk_ids,
            window_size=window_size,
        )
        selected_texts = [
            chunk_text_lookup.get((document_id, selected_chunk_id), "").strip()
            for selected_chunk_id in window_chunk_ids
        ]
        context_text = "\n\n".join(text for text in selected_texts if text).strip()
        if not context_text:
            context_text = str(row.get("body_text") or "").strip()
        expansion_mode = (
            "postgres_window"
            if len(window_chunk_ids) > 1
            else "postgres_child"
        )
        if not window_chunk_ids:
            expansion_mode = "postgres_parent_fallback"

        context_blocks.append(
            {
                "document_id": document_id,
                "parent_id": parent_id,
                "section_title": row.get("section_title"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "heading_path": [
                    str(item)
                    for item in row.get("heading_path") or []
                    if str(item)
                ],
                "matched_chunk_ids": matched_chunk_ids,
                "window_chunk_ids": window_chunk_ids,
                "context_text": context_text,
                "expansion_mode": expansion_mode,
            }
        )

    return context_blocks


def load_visual_assets(
    *,
    thread_id: str | None,
    active_document_ids: list[str],
    asset_refs: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> list[VisualAssetPayload]:
    """현재 thread 범위에서 visual asset 메타데이터를 읽는다."""
    del thread_id
    normalized_document_ids = _normalize_document_ids(active_document_ids)
    normalized_asset_refs = [str(item).strip() for item in asset_refs or [] if str(item).strip()]
    normalized_chunk_refs = [str(item).strip() for item in chunk_ids or [] if str(item).strip()]
    if not normalized_document_ids:
        return []

    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        asset_rows = document_repository.list_document_assets(normalized_document_ids)

    visual_assets: list[VisualAssetPayload] = []
    for row in asset_rows:
        document_id = str(row.get("document_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        asset_ref = str(row.get("asset_ref") or "").strip()

        if normalized_asset_refs and asset_ref not in normalized_asset_refs:
            continue
        if normalized_chunk_refs and not _match_chunk_ref(
            document_id=document_id,
            chunk_id=chunk_id,
            requested_refs=normalized_chunk_refs,
        ):
            continue
        if not normalized_asset_refs and not normalized_chunk_refs:
            continue

        metadata = dict(row.get("metadata") or {})
        visual_assets.append(
            {
                "asset_ref": asset_ref,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "asset_kind": str(row.get("asset_kind") or "").strip(),
                "relative_path": str(row.get("relative_path") or "").strip(),
                "asset_stage": "stage2",
                "page": row.get("page"),
                "caption": row.get("caption"),
                "summary_text": row.get("summary_text"),
                "heading_path": [
                    str(item)
                    for item in metadata.get("heading_path") or []
                    if str(item)
                ],
                "pages": [
                    int(item)
                    for item in metadata.get("pages") or []
                    if isinstance(item, int) or str(item).isdigit()
                ],
            }
        )

    return visual_assets
