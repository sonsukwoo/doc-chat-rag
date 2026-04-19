"""Application Postgres configuration."""

from __future__ import annotations

import os
import re
from pathlib import Path
from urllib.parse import quote, urlencode

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _env_int(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value in (None, ""):
        return default
    return int(raw_value)


def validate_identifier(value: str, *, field_name: str) -> str:
    """SQL identifier로 안전하게 쓸 수 있는지 검증한다."""
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    if not _IDENTIFIER_PATTERN.match(normalized):
        raise ValueError(f"invalid SQL identifier for {field_name}: {normalized}")
    return normalized


DB_HOST = os.getenv("DB_HOST", "").strip()
DB_PORT = _env_int("DB_PORT", 5432)
DB_USER = os.getenv("DB_USER", "").strip()
DB_PASSWORD = os.getenv("DB_PASSWORD", "").strip()
DB_ADMIN_DATABASE = validate_identifier(
    os.getenv("DB_ADMIN_DATABASE", "postgres"),
    field_name="DB_ADMIN_DATABASE",
)
APP_DATABASE_NAME = validate_identifier(
    os.getenv("DB_NAME", "rag_chat_app"),
    field_name="DB_NAME",
)

APP_CHAT_SCHEMA = validate_identifier(
    os.getenv("APP_CHAT_SCHEMA", "app_chat"),
    field_name="APP_CHAT_SCHEMA",
)
APP_DOC_SCHEMA = validate_identifier(
    os.getenv("APP_DOC_SCHEMA", "app_doc"),
    field_name="APP_DOC_SCHEMA",
)
APP_PIPELINE_SCHEMA = validate_identifier(
    os.getenv("APP_PIPELINE_SCHEMA", "app_pipeline"),
    field_name="APP_PIPELINE_SCHEMA",
)
APP_CHECKPOINT_SCHEMA = validate_identifier(
    os.getenv("APP_CHECKPOINT_SCHEMA", "app_checkpoint"),
    field_name="APP_CHECKPOINT_SCHEMA",
)


def _validate_connection_settings() -> None:
    missing = [
        name
        for name, value in (
            ("DB_HOST", DB_HOST),
            ("DB_USER", DB_USER),
            ("DB_PASSWORD", DB_PASSWORD),
        )
        if not value
    ]
    if missing:
        raise ValueError(f"missing database env vars: {', '.join(missing)}")


def build_postgres_uri(
    *,
    database: str,
    search_path: str | None = None,
) -> str:
    """database와 optional search_path를 반영한 Postgres URI를 만든다."""
    _validate_connection_settings()
    resolved_database = validate_identifier(database, field_name="database")
    user = quote(DB_USER, safe="")
    password = quote(DB_PASSWORD, safe="")
    base_uri = f"postgresql://{user}:{password}@{DB_HOST}:{DB_PORT}/{resolved_database}"
    if not search_path:
        return base_uri
    query = urlencode({"options": f"-csearch_path={search_path},public"})
    return f"{base_uri}?{query}"


def build_admin_uri() -> str:
    """DB 생성/검증에 사용할 admin 연결 URI."""
    return build_postgres_uri(database=DB_ADMIN_DATABASE)


def build_app_uri() -> str:
    """애플리케이션 메타데이터용 기본 DB 연결 URI."""
    return build_postgres_uri(database=APP_DATABASE_NAME)


def build_checkpoint_uri() -> str:
    """LangGraph checkpointer 전용 search_path를 가진 URI."""
    return build_postgres_uri(
        database=APP_DATABASE_NAME,
        search_path=APP_CHECKPOINT_SCHEMA,
    )
