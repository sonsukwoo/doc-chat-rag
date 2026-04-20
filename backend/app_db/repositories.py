"""Application Postgres repository layer.

thread/document 메타데이터는 여기서만 직접 SQL을 다룬다.
상위 서비스는 repository를 조합해서 의미 단위 작업만 수행한다.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from typing import Any

from psycopg import Connection, connect
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from .config import (
    APP_CHAT_SCHEMA,
    APP_CHECKPOINT_SCHEMA,
    APP_DOC_SCHEMA,
    build_app_uri,
)


def _chat_table(name: str) -> str:
    return f'"{APP_CHAT_SCHEMA}"."{name}"'


def _doc_table(name: str) -> str:
    return f'"{APP_DOC_SCHEMA}"."{name}"'


def _checkpoint_table(name: str) -> str:
    return f'"{APP_CHECKPOINT_SCHEMA}"."{name}"'


@contextmanager
def app_db_connection() -> Iterator[Connection[Any]]:
    """애플리케이션 메타데이터용 Postgres 연결을 연다."""
    with connect(build_app_uri(), row_factory=dict_row) as connection:
        yield connection


def _normalize_jsonb(value: dict[str, Any] | list[Any] | None) -> Jsonb:
    return Jsonb(value if value is not None else {})


class ChatRepository:
    """최상위 thread 메타데이터 접근 전용 repository."""

    def __init__(self, connection: Connection[Any]) -> None:
        self.connection = connection

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        query = f"""
            SELECT
                thread_id,
                thread_name,
                description,
                default_retrieval_mode,
                metadata,
                created_at,
                updated_at,
                archived_at,
                last_user_message_at
            FROM {_chat_table("threads")}
            WHERE thread_id = %s
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (thread_id,))
            return cursor.fetchone()

    def list_threads(self, *, include_archived: bool = False) -> list[dict[str, Any]]:
        query = f"""
            SELECT
                thread_id,
                thread_name,
                description,
                default_retrieval_mode,
                metadata,
                created_at,
                updated_at,
                archived_at,
                last_user_message_at
            FROM {_chat_table("threads")}
        """
        params: tuple[object, ...] = ()
        if not include_archived:
            query += "\nWHERE archived_at IS NULL"
        query += "\nORDER BY updated_at DESC, thread_id ASC"
        with self.connection.cursor() as cursor:
            cursor.execute(query, params)
            return list(cursor.fetchall())

    def upsert_thread(
        self,
        *,
        thread_id: str,
        thread_name: str,
        description: str | None = None,
        default_retrieval_mode: str = "dense",
        metadata: dict[str, Any] | None = None,
        last_user_message_at: Any | None = None,
    ) -> None:
        query = f"""
            INSERT INTO {_chat_table("threads")} (
                thread_id,
                thread_name,
                description,
                default_retrieval_mode,
                metadata,
                last_user_message_at
            )
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (thread_id) DO UPDATE
            SET thread_name = EXCLUDED.thread_name,
                description = EXCLUDED.description,
                default_retrieval_mode = EXCLUDED.default_retrieval_mode,
                metadata = EXCLUDED.metadata,
                last_user_message_at = COALESCE(
                    EXCLUDED.last_user_message_at,
                    {_chat_table("threads")}.last_user_message_at
                ),
                updated_at = NOW(),
                archived_at = NULL
        """
        with self.connection.cursor() as cursor:
            cursor.execute(
                query,
                (
                    thread_id,
                    thread_name,
                    description,
                    default_retrieval_mode,
                    _normalize_jsonb(metadata),
                    last_user_message_at,
                ),
            )

    def archive_thread(self, thread_id: str) -> bool:
        query = f"""
            UPDATE {_chat_table("threads")}
            SET archived_at = NOW(),
                updated_at = NOW()
            WHERE thread_id = %s
              AND archived_at IS NULL
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (thread_id,))
            return cursor.rowcount > 0

    def delete_thread(self, thread_id: str) -> bool:
        query = f"DELETE FROM {_chat_table('threads')} WHERE thread_id = %s"
        with self.connection.cursor() as cursor:
            cursor.execute(query, (thread_id,))
            return cursor.rowcount > 0


class CheckpointRepository:
    """LangGraph checkpointer 스키마 접근 전용 repository."""

    def __init__(self, connection: Connection[Any]) -> None:
        self.connection = connection

    def delete_thread_checkpoints(self, thread_id: str) -> dict[str, int]:
        deleted_rows: dict[str, int] = {}
        with self.connection.cursor() as cursor:
            for table_name in ("checkpoint_writes", "checkpoint_blobs", "checkpoints"):
                cursor.execute(
                    f"DELETE FROM {_checkpoint_table(table_name)} WHERE thread_id = %s",
                    (thread_id,),
                )
                deleted_rows[table_name] = cursor.rowcount
        return deleted_rows


class DocumentRepository:
    """document 및 thread-document 연결 메타데이터 접근 전용 repository."""

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

    def attach_document_to_thread(
        self,
        *,
        thread_id: str,
        document_id: str,
        slot_key: str,
    ) -> None:
        query = f"""
            INSERT INTO {_doc_table("thread_documents")} (
                thread_id,
                document_id,
                slot_key,
                is_active,
                detached_at
            )
            VALUES (%s, %s, %s, TRUE, NULL)
            ON CONFLICT (thread_id, document_id) DO UPDATE
            SET slot_key = EXCLUDED.slot_key,
                is_active = TRUE,
                detached_at = NULL,
                attached_at = NOW()
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (thread_id, document_id, slot_key))

    def list_active_document_ids(self, thread_id: str) -> list[str]:
        query = f"""
            SELECT document_id
            FROM {_doc_table("thread_documents")}
            WHERE thread_id = %s
              AND is_active = TRUE
              AND detached_at IS NULL
            ORDER BY attached_at ASC, document_id ASC
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (thread_id,))
            return [
                str(row["document_id"])
                for row in cursor.fetchall()
                if row.get("document_id")
            ]

    def list_thread_document_links(self, thread_id: str) -> list[dict[str, Any]]:
        query = f"""
            SELECT
                td.document_id,
                d.storage_root,
                COUNT(DISTINCT td_all.thread_id) FILTER (
                    WHERE td_all.is_active = TRUE
                      AND td_all.detached_at IS NULL
                ) AS linked_thread_count
            FROM {_doc_table("thread_documents")} td
            INNER JOIN {_doc_table("documents")} d
                ON d.document_id = td.document_id
            LEFT JOIN {_doc_table("thread_documents")} td_all
                ON td_all.document_id = td.document_id
            WHERE td.thread_id = %s
              AND td.is_active = TRUE
              AND td.detached_at IS NULL
            GROUP BY td.document_id, d.storage_root
            ORDER BY td.document_id ASC
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (thread_id,))
            return list(cursor.fetchall())

    def delete_documents(self, document_ids: Sequence[str]) -> list[dict[str, Any]]:
        normalized_document_ids = [
            str(item).strip() for item in document_ids if str(item).strip()
        ]
        if not normalized_document_ids:
            return []

        query = f"""
            DELETE FROM {_doc_table("documents")}
            WHERE document_id = ANY(%s)
            RETURNING document_id, storage_root
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (normalized_document_ids,))
            return list(cursor.fetchall())

    def list_document_parents(
        self,
        document_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """여러 문서에 속한 parent 문맥 블록을 한 번에 읽는다."""
        normalized_document_ids = [str(item).strip() for item in document_ids if str(item).strip()]
        if not normalized_document_ids:
            return []

        query = f"""
            SELECT
                parent_id,
                document_id,
                section_title,
                page_start,
                page_end,
                heading_path,
                chunk_ids,
                body_text,
                metadata
            FROM {_doc_table("document_parents")}
            WHERE document_id = ANY(%s)
            ORDER BY document_id ASC, page_start ASC NULLS LAST, parent_id ASC
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (normalized_document_ids,))
            return list(cursor.fetchall())

    def list_document_chunks(
        self,
        document_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """여러 문서에 속한 child chunk 본문과 순서를 한 번에 읽는다."""
        normalized_document_ids = [str(item).strip() for item in document_ids if str(item).strip()]
        if not normalized_document_ids:
            return []

        query = f"""
            SELECT
                document_id,
                chunk_id,
                parent_id,
                chunk_index,
                chunk_type,
                page_start,
                page_end,
                pages,
                heading_path,
                text,
                metadata
            FROM {_doc_table("document_chunks")}
            WHERE document_id = ANY(%s)
            ORDER BY document_id ASC, chunk_index ASC, chunk_id ASC
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (normalized_document_ids,))
            return list(cursor.fetchall())

    def list_document_assets(
        self,
        document_ids: Sequence[str],
    ) -> list[dict[str, Any]]:
        """여러 문서에 속한 visual asset 메타데이터를 한 번에 읽는다."""
        normalized_document_ids = [str(item).strip() for item in document_ids if str(item).strip()]
        if not normalized_document_ids:
            return []

        query = f"""
            SELECT
                asset_ref,
                document_id,
                chunk_id,
                asset_kind,
                relative_path,
                page,
                caption,
                summary_text,
                metadata
            FROM {_doc_table("document_assets")}
            WHERE document_id = ANY(%s)
            ORDER BY document_id ASC, page ASC NULLS LAST, asset_ref ASC
        """
        with self.connection.cursor() as cursor:
            cursor.execute(query, (normalized_document_ids,))
            return list(cursor.fetchall())

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

    def replace_document_chunks(
        self,
        *,
        document_id: str,
        chunks: Sequence[dict[str, Any]],
    ) -> None:
        delete_query = f"DELETE FROM {_doc_table('document_chunks')} WHERE document_id = %s"
        insert_query = f"""
            INSERT INTO {_doc_table("document_chunks")} (
                document_id,
                chunk_id,
                parent_id,
                chunk_index,
                chunk_type,
                page_start,
                page_end,
                pages,
                heading_path,
                text,
                metadata
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        with self.connection.cursor() as cursor:
            cursor.execute(delete_query, (document_id,))
            for index, chunk in enumerate(chunks, start=1):
                pages = [
                    int(page)
                    for page in chunk.get("pages") or []
                    if isinstance(page, int) or str(page).isdigit()
                ]
                cursor.execute(
                    insert_query,
                    (
                        document_id,
                        str(chunk.get("chunk_id") or ""),
                        str(chunk.get("parent_id") or "") or None,
                        index,
                        str(chunk.get("chunk_type") or ""),
                        pages[0] if pages else None,
                        pages[-1] if pages else None,
                        Jsonb(pages),
                        Jsonb(list(chunk.get("heading_path") or [])),
                        str(chunk.get("text") or ""),
                        _normalize_jsonb(chunk.get("metadata")),
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
