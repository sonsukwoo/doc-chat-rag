"""thread 식별자와 Qdrant 컬렉션명을 일관되게 생성하는 helper."""

from __future__ import annotations

import os
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)

DEFAULT_THREAD_COLLECTION_BASE = (
    os.getenv("STAGE3_QDRANT_COLLECTION_NAME", "rag_chat_hybrid").strip()
    or "rag_chat_hybrid"
)
THREAD_COLLECTION_NAME_METADATA_KEY = "qdrant_collection_name"


def sanitize_thread_name(value: str) -> str:
    """thread 이름을 id/collection suffix로 안전하게 정규화한다."""
    allowed: list[str] = []
    for char in str(value or "").strip().lower():
        if char.isalnum():
            allowed.append(char)
        elif char in {"-", "_"}:
            allowed.append(char)
        elif char.isspace():
            allowed.append("-")
    normalized = "".join(allowed).strip("-_")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    return normalized or "thread"


def build_thread_id(thread_name: str) -> str:
    """사람이 보는 thread 이름으로부터 고유 thread id를 만든다."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"thread_{sanitize_thread_name(thread_name)}_{timestamp}_{uuid4().hex[:6]}"


def build_thread_collection_name(thread_id: str) -> str:
    """thread_id에 종속된 단일 Qdrant collection 이름을 계산한다."""
    base_name = sanitize_thread_name(DEFAULT_THREAD_COLLECTION_BASE)
    thread_suffix = sanitize_thread_name(thread_id)
    return f"{base_name}_{thread_suffix}"[:180]


def resolve_thread_collection_name(
    thread_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    explicit_name: str | None = None,
) -> str:
    """저장된 metadata 또는 기본 규칙에서 thread collection 이름을 결정한다."""
    resolved_explicit_name = str(explicit_name or "").strip()
    if resolved_explicit_name:
        return resolved_explicit_name

    resolved_metadata_name = str(
        (metadata or {}).get(THREAD_COLLECTION_NAME_METADATA_KEY) or ""
    ).strip()
    if resolved_metadata_name:
        return resolved_metadata_name

    return build_thread_collection_name(thread_id)


def ensure_thread_metadata(
    thread_id: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """thread 생성/갱신 시 내부 런타임 metadata를 빠짐없이 채운다."""
    resolved_metadata = dict(metadata or {})
    resolved_metadata.setdefault(
        THREAD_COLLECTION_NAME_METADATA_KEY,
        build_thread_collection_name(thread_id),
    )
    return resolved_metadata
