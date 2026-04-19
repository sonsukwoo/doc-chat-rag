"""Application Postgres service layer."""

from __future__ import annotations

from pathlib import Path
from typing import Any, TypedDict

from .repositories import ChatRepository, DocumentRepository, app_db_connection


class RoomRuntimeContext(TypedDict, total=False):
    """room 기반 검색/대화에 필요한 최소 런타임 컨텍스트."""

    room_id: str
    room_name: str
    collection_name: str
    default_retrieval_mode: str
    active_document_ids: list[str]


def load_room_runtime_context(room_id: str) -> RoomRuntimeContext | None:
    """room 메타데이터와 현재 연결된 문서 ID 목록을 함께 읽는다."""
    normalized_room_id = str(room_id or "").strip()
    if not normalized_room_id:
        return None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        document_repository = DocumentRepository(connection)
        room = chat_repository.get_room(normalized_room_id)
        if room is None:
            return None
        active_document_ids = document_repository.list_active_document_ids(
            normalized_room_id
        )
        return {
            "room_id": normalized_room_id,
            "room_name": str(room.get("room_name") or normalized_room_id),
            "collection_name": str(room.get("collection_name") or "").strip(),
            "default_retrieval_mode": str(
                room.get("default_retrieval_mode") or "dense"
            ).strip()
            or "dense",
            "active_document_ids": active_document_ids,
        }


def try_load_room_runtime_context(room_id: str) -> RoomRuntimeContext | None:
    """DB 연결 실패까지 포함해 안전하게 room 컨텍스트를 읽는다."""
    try:
        return load_room_runtime_context(room_id)
    except Exception:
        return None


def sync_document_runtime_metadata(
    *,
    room_id: str,
    document_id: str,
    original_filename: str,
    normalized_filename: str,
    storage_root: str | Path,
    source_pdf_path: str | None,
    parents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
) -> None:
    """stage3 이후 document/room/parent/asset 메타데이터를 Postgres와 동기화한다."""
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
        document_repository.attach_document_to_room(
            room_id=room_id,
            document_id=document_id,
            slot_key=normalized_filename,
        )
        document_repository.replace_document_parents(
            document_id=document_id,
            parents=parents,
        )
        document_repository.replace_document_assets(
            document_id=document_id,
            chunks=chunks,
        )

