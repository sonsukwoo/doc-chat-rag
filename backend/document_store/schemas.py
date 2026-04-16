"""문서 저장소 메타데이터 스키마."""

from __future__ import annotations

from typing import Literal, TypedDict


DocumentStageStatus = Literal[
    "not_started",
    "uploaded",
    "running",
    "completed",
    "failed",
]


class DocumentStageRecord(TypedDict, total=False):
    """각 stage 실행 상태와 산출물 요약."""

    status: DocumentStageStatus
    updated_at: str
    error: str | None
    outputs: dict[str, str]


class DocumentRecord(TypedDict, total=False):
    """문서 단위 저장소 메타데이터."""

    document_id: str
    original_filename: str
    uploaded_at: str
    stages: dict[str, DocumentStageRecord]
