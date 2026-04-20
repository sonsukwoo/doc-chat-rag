"""Application Postgres DDL builders."""

from __future__ import annotations

from .config import (
    APP_CHAT_SCHEMA,
    APP_CHECKPOINT_SCHEMA,
    APP_DOC_SCHEMA,
    APP_PIPELINE_SCHEMA,
)


def _ident(name: str) -> str:
    return f'"{name}"'


def build_schema_ddl() -> list[str]:
    """애플리케이션 메타 스키마와 테이블 DDL을 순서대로 반환한다."""
    chat = _ident(APP_CHAT_SCHEMA)
    doc = _ident(APP_DOC_SCHEMA)
    pipeline = _ident(APP_PIPELINE_SCHEMA)
    checkpoint = _ident(APP_CHECKPOINT_SCHEMA)

    return [
        f"CREATE SCHEMA IF NOT EXISTS {chat};",
        f"CREATE SCHEMA IF NOT EXISTS {doc};",
        f"CREATE SCHEMA IF NOT EXISTS {pipeline};",
        f"CREATE SCHEMA IF NOT EXISTS {checkpoint};",
        f"""
        CREATE TABLE IF NOT EXISTS {chat}.threads (
            thread_id TEXT PRIMARY KEY,
            thread_name TEXT NOT NULL,
            description TEXT,
            default_retrieval_mode TEXT NOT NULL DEFAULT 'dense',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            archived_at TIMESTAMPTZ,
            last_user_message_at TIMESTAMPTZ
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {doc}.documents (
            document_id TEXT PRIMARY KEY,
            original_filename TEXT NOT NULL,
            normalized_filename TEXT NOT NULL,
            file_hash TEXT,
            storage_root TEXT NOT NULL,
            source_pdf_path TEXT,
            source_kind TEXT NOT NULL DEFAULT 'upload',
            lifecycle_status TEXT NOT NULL DEFAULT 'active',
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            deleted_at TIMESTAMPTZ
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {doc}.thread_documents (
            thread_id TEXT NOT NULL REFERENCES {chat}.threads(thread_id) ON DELETE CASCADE,
            document_id TEXT NOT NULL REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            slot_key TEXT NOT NULL,
            is_active BOOLEAN NOT NULL DEFAULT TRUE,
            attached_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            detached_at TIMESTAMPTZ,
            PRIMARY KEY (thread_id, document_id)
        );
        """,
        f"""
        CREATE UNIQUE INDEX IF NOT EXISTS idx_thread_documents_active_slot
        ON {doc}.thread_documents(thread_id, slot_key)
        WHERE is_active = TRUE AND detached_at IS NULL;
        """,
        f"CREATE INDEX IF NOT EXISTS idx_thread_documents_document_id ON {doc}.thread_documents(document_id);",
        f"""
        CREATE TABLE IF NOT EXISTS {doc}.document_parents (
            document_id TEXT NOT NULL REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            parent_id TEXT NOT NULL,
            section_title TEXT,
            page_start INTEGER,
            page_end INTEGER,
            heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,
            chunk_ids JSONB NOT NULL DEFAULT '[]'::jsonb,
            body_text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (document_id, parent_id)
        );
        """,
        f"CREATE INDEX IF NOT EXISTS idx_document_parents_document_id ON {doc}.document_parents(document_id);",
        f"""
        CREATE TABLE IF NOT EXISTS {doc}.document_chunks (
            document_id TEXT NOT NULL REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            chunk_id TEXT NOT NULL,
            parent_id TEXT,
            chunk_index INTEGER NOT NULL,
            chunk_type TEXT NOT NULL,
            page_start INTEGER,
            page_end INTEGER,
            pages JSONB NOT NULL DEFAULT '[]'::jsonb,
            heading_path JSONB NOT NULL DEFAULT '[]'::jsonb,
            text TEXT NOT NULL,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (document_id, chunk_id)
        );
        """,
        f"CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id ON {doc}.document_chunks(document_id);",
        f"CREATE INDEX IF NOT EXISTS idx_document_chunks_parent_order ON {doc}.document_chunks(document_id, parent_id, chunk_index);",
        f"""
        CREATE TABLE IF NOT EXISTS {doc}.document_assets (
            asset_ref TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            chunk_id TEXT,
            asset_kind TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            page INTEGER,
            caption TEXT,
            summary_text TEXT,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        f"CREATE INDEX IF NOT EXISTS idx_document_assets_document_id ON {doc}.document_assets(document_id);",
        f"""
        CREATE TABLE IF NOT EXISTS {doc}.document_review_decisions (
            document_id TEXT PRIMARY KEY REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            element_decisions JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            exact_text_drop JSONB NOT NULL DEFAULT '[]'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {pipeline}.document_stage_status (
            document_id TEXT NOT NULL REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            stage_name TEXT NOT NULL,
            status TEXT NOT NULL,
            error TEXT,
            outputs JSONB NOT NULL DEFAULT '{{}}'::jsonb,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            PRIMARY KEY (document_id, stage_name)
        );
        """,
        f"""
        CREATE TABLE IF NOT EXISTS {pipeline}.document_stage_runs (
            run_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES {doc}.documents(document_id) ON DELETE CASCADE,
            stage_name TEXT NOT NULL,
            status TEXT NOT NULL,
            started_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
            finished_at TIMESTAMPTZ,
            error TEXT,
            metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
        );
        """,
        f"CREATE INDEX IF NOT EXISTS idx_document_stage_runs_document_id ON {pipeline}.document_stage_runs(document_id);",
    ]
