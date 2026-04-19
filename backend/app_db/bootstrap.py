"""Application Postgres bootstrap routines."""

from __future__ import annotations

from typing import Any, TypedDict

from psycopg import connect
from psycopg.rows import dict_row
from psycopg.sql import Identifier, SQL

from .config import (
    APP_CHAT_SCHEMA,
    APP_CHECKPOINT_SCHEMA,
    APP_DATABASE_NAME,
    APP_DOC_SCHEMA,
    APP_PIPELINE_SCHEMA,
    build_admin_uri,
    build_app_uri,
    build_checkpoint_uri,
)
from .ddl import build_schema_ddl


class BootstrapResult(TypedDict):
    """DB bootstrap 결과 요약."""

    database_name: str
    database_created: bool
    schemas: list[str]
    checkpoint_schema_initialized: bool


def ensure_database_exists() -> bool:
    """애플리케이션용 DB가 없으면 생성한다."""
    admin_uri = build_admin_uri()
    with connect(admin_uri, autocommit=True, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 AS exists FROM pg_database WHERE datname = %s",
                (APP_DATABASE_NAME,),
            )
            if cur.fetchone():
                return False
            cur.execute(
                SQL("CREATE DATABASE {}").format(Identifier(APP_DATABASE_NAME))
            )
            return True


def ensure_application_schemas() -> None:
    """애플리케이션 메타 스키마와 테이블을 생성한다."""
    app_uri = build_app_uri()
    with connect(app_uri, autocommit=True, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            for statement in build_schema_ddl():
                cur.execute(statement)


def ensure_checkpoint_schema() -> None:
    """LangGraph checkpointer 전용 테이블을 생성한다."""
    from langgraph.checkpoint.postgres import PostgresSaver

    checkpoint_uri = build_checkpoint_uri()
    with PostgresSaver.from_conn_string(checkpoint_uri) as checkpointer:
        checkpointer.setup()


def bootstrap_application_database() -> BootstrapResult:
    """DB, 앱 스키마, LangGraph 체크포인터 테이블을 한 번에 준비한다."""
    database_created = ensure_database_exists()
    ensure_application_schemas()
    ensure_checkpoint_schema()
    return {
        "database_name": APP_DATABASE_NAME,
        "database_created": database_created,
        "schemas": [
            APP_CHAT_SCHEMA,
            APP_DOC_SCHEMA,
            APP_PIPELINE_SCHEMA,
            APP_CHECKPOINT_SCHEMA,
        ],
        "checkpoint_schema_initialized": True,
    }
