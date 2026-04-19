"""Stage-5 chatbot checkpointer helpers."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.checkpoint.memory import InMemorySaver

from backend.app_db import build_checkpoint_uri

from .config import STAGE5_CHECKPOINTER_BACKEND, STAGE5_POSTGRES_URI


@contextmanager
def stage5_checkpointer_context(
    *,
    backend: str | None = None,
    postgres_uri: str | None = None,
) -> Iterator[object]:
    """stage5 graph가 사용할 checkpointer를 컨텍스트로 제공한다."""
    resolved_backend = str(backend or STAGE5_CHECKPOINTER_BACKEND).strip().lower()
    resolved_postgres_uri = str(postgres_uri or STAGE5_POSTGRES_URI).strip()

    if resolved_backend == "memory":
        yield InMemorySaver()
        return

    if resolved_backend != "postgres":
        raise ValueError(f"unsupported stage5 checkpointer backend: {resolved_backend}")

    if not resolved_postgres_uri:
        resolved_postgres_uri = build_checkpoint_uri()

    try:
        from langgraph.checkpoint.postgres import PostgresSaver
    except ImportError as exc:
        raise RuntimeError(
            "missing langgraph postgres checkpoint dependency; install "
            "'langgraph-checkpoint-postgres' before enabling postgres checkpointer"
        ) from exc

    with PostgresSaver.from_conn_string(resolved_postgres_uri) as checkpointer:
        checkpointer.setup()
        yield checkpointer
