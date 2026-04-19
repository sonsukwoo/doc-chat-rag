"""Application Postgres repository layer.

room/thread/document 메타데이터는 여기서만 직접 SQL을 다룬다.
상위 서비스는 repository를 조합해서 의미 단위 작업만 수행한다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from psycopg import Connection, connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import APP_CHAT_SCHEMA, APP_DOC_SCHEMA, build_app_uri


def _chat_table(name: str) -> str:
    return f'"{APP_CHAT_SCHEMA}"."{name}"'


def _doc_table(name: str) -> str:
    return f'"{APP_DOC_SCHEMA}"."{name}"'


@contextmanager
def app_db_connection() -> Iterator[Connection[Any]]:
    """애플리케이션 메타데이터용 Postgres 연결을 연다."""
    with connect(build_app_uri(), row_factory=dict_row) as connection:
        yield connection


def _normalize_jsonb(value: dict[str, Any] | list[Any] | None) -> Jsonb:
    return Jsonb(value if value is not None else {})


class ChatRepository:
    """room/thread 메타데이터 접근 전용 repository."""

    def __init__(self, connection: Connection[Any]) -> None:
        self.connection = connection

    def get_room(self, room_id: str) -> dict[str, Any] | None:
        query = f"""
            SELECT
                room_id,
                room_name,
                collection_name,
                description,
                default_retrieval_mode,
                metadata,
                created_at,
                updated_at,
                archived_at
            FROM {_chat_table("rooms")}
            WHERE room_id = %s
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (room_id,))
            return cursor.fetchone()

    def upsert_room(
        self,
        *,
        room_id: str,
        room_name: str,
        collection_name: str,
        description: str | None = None,
        default_retrieval_mode: str = "dense",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        query = f"""
            INSERT INTO {_chat_table("rooms")} (
                room_id,
                room_name,
                collection_name,
                description,
                default_retrieval_mode,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (room_id) DO UPDATE
            SET room_name = EXCLUDED.room_name,
                collection_name = EXCLUDED.collection_name,
                description = EXCLUDED.description,
                default_retrieval_mode = EXCLUDED.default_retrieval_mode,
                metadata = EXCLUDED.metadata,
                updated_at = NOW(),
                archived_at = NULL
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    room_id,
                    room_name,
                    collection_name,
                    description,
                    default_retrieval_mode,
                    _normalize_jsonb(metadata),
                ),
            )

    def upsert_thread(
        self,
        *,
        thread_id: str,
        room_id: str,
        title: str | None = None,
        status: str = "active",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        query = f"""
            INSERT INTO {_chat_table("threads")} (
                thread_id,
                room_id,
                title,
                status,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE
            SET room_id = EXCLUDED.room_id,
                title = EXCLUDED.title,
                status = EXCLUDED.status,
                metadata = EXCLUDED.metadata,
                updated_at = NOW()
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    thread_id,
                    room_id,
                    title,
                    status,
                    _normalize_jsonb(metadata),
                ),
            )


class DocumentRepository:
    """document 및 room-document 연결 메타데이터 접근 전용 repository."""

    def __init__(self, connection: Connection[Any]) -> None:
        self.connection = connection

    def upsert_document(
        self,
        *,
        document_id: str,
        original_filename: str,
        normalized_filename: str,
        storage_root: str,
        source_pdf_path: str | None = None,
        source_kind: str = "upload",
        lifecycle_status: str = "active",
        file_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        query = f"""
            INSERT INTO {_doc_table("documents")} (
                document_id,
                original_filename,
                normalized_filename,
                file_hash,
                storage_root,
                source_pdf_path,
                source_kind,
                lifecycle_status,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE
            SET original_filename = EXCLUDED.original_filename,
                normalized_filename = EXCLUDED.normalized_filename,
                file_hash = EXCLUDED.file_hash,
                storage_root = EXCLUDED.storage_root,
                source_pdf_path = EXCLUDED.source_pdf_path,
                source_kind = EXCLUDED.source_kind,
                lifecycle_status = EXCLUDED.lifecycle_status,
                metadata = EXCLUDED.metadata,
                updated_at = NOW(),
                deleted_at = NULL
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    document_id,
                    original_filename,
                    normalized_filename,
                    file_hash,
                    storage_root,
                    source_pdf_path,
                    source_kind,
                    lifecycle_status,
                    _normalize_jsonb(metadata),
                ),
            )

    def attach_document_to_room(
        self,
        *,
        room_id: str,
        document_id: str,
        slot_key: str,
    ) -> None:
        query = f"""
            INSERT INTO {_doc_table("room_documents")} (
                room_id,
                document_id,
                slot_key,
                is_active,
                detached_at
            )
            VALUES (%s, %s, %s, TRUE, NULL)
            ON CONFLICT (room_id, document_id) DO UPDATE
            SET slot_key = EXCLUDED.slot_key,
                is_active = TRUE,
                detached_at = NULL,
                attached_at = NOW()
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (room_id, document_id, slot_key))

    def list_active_document_ids(self, room_id: str) -> list[str]:
        query = f"""
            SELECT document_id
            FROM {_doc_table("room_documents")}
            WHERE room_id = %s
              AND is_active = TRUE
              AND detached_at IS NULL
            ORDER BY attached_at ASC, document_id ASC
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (room_id,))
            return [
                str(row["document_id"])
                for row in cursor.fetchall()
                if row.get("document_id")
            ]

    def replace_document_parents(
        self,
        *,
        document_id: str,
        parents: Sequence[dict[str, Any]],
    ) -> None:
        delete_query = f"DELETE FROM {_doc_table('document_parents')} WHERE document_id = %s"
        insert_query = f"""
            INSERT INTO {_doc_table("document_parents")} (
                parent_id,
                document_id,
                section_title,
                page_start,
                page_end,
                heading_path,
                chunk_ids,
                body_text,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self.connection.cursor() as cursor:
            cursor.execute(delete_query, (document_id,))
            for parent in parents:
                cursor.execute(
                    insert_query,
                    (
                        str(parent.get("parent_id") or ""),
                        document_id,
                        parent.get("section_title"),
                        parent.get("page_start"),
                        parent.get("page_end"),
                        Jsonb(list(parent.get("heading_path") or [])),
                        Jsonb(list(parent.get("child_chunk_ids") or [])),
                        str(parent.get("text") or ""),
                        _normalize_jsonb(parent.get("metadata")),
                    ),
                )

    def replace_document_assets(
        self,
        *,
        document_id: str,
        chunks: Sequence[dict[str, Any]],
    ) -> None:
        delete_query = f"DELETE FROM {_doc_table('document_assets')} WHERE document_id = %s"
        insert_query = f"""
            INSERT INTO {_doc_table("document_assets")} (
                asset_ref,
                document_id,
                chunk_id,
                asset_kind,
                relative_path,
                page,
                caption,
                summary_text,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """

        with self.connection.cursor() as cursor:
            cursor.execute(delete_query, (document_id,))
            for chunk in chunks:
                metadata = dict(chunk.get("metadata") or {})
                asset_relative_path = str(metadata.get("image_path") or "").strip()
                if not asset_relative_path:
                    continue
                chunk_id = str(chunk.get("chunk_id") or "").strip()
                if not chunk_id:
                    continue
                chunk_type = str(chunk.get("chunk_type") or "").strip() or "asset"
                pages = [
                    int(page)
                    for page in chunk.get("pages") or []
                    if isinstance(page, int) or str(page).isdigit()
                ]
                caption = str(metadata.get("caption") or "").strip() or None
                summary_text = (
                    str(metadata.get("summary_text") or "").strip() or None
                )
                cursor.execute(
                    insert_query,
                    (
                        f"{document_id}:{chunk_id}",
                        document_id,
                        chunk_id,
                        chunk_type,
                        asset_relative_path,
                        pages[0] if pages else None,
                        caption,
                        summary_text,
                        _normalize_jsonb(
                            {
                                "pages": pages,
                                "heading_path": list(chunk.get("heading_path") or []),
                            }
                        ),
                    ),
                )
