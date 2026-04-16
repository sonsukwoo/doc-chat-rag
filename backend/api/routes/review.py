"""Review overlay 라우터."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from backend.document_store import build_document_paths, load_document_record
from backend.review_overlay import (
    apply_review_overlay,
    build_review_source,
    load_review_decisions,
    save_review_decisions,
)


router = APIRouter(prefix="/documents", tags=["review"])


class ReviewElementDecisionBody(BaseModel):
    dropped: bool | None = Field(default=None)
    category_override: str | None = Field(default=None)


class ReviewDecisionsBody(BaseModel):
    element_decisions: dict[str, ReviewElementDecisionBody] = Field(default_factory=dict)
    exact_text_drop: list[str] = Field(default_factory=list)


def _ensure_document_exists(document_id: str) -> None:
    try:
        load_document_record(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


def _load_json_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"artifact not found: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"invalid json artifact: {path.name}")
    return payload


@router.get("/{document_id}/review/source")
def get_review_source(document_id: str) -> dict[str, Any]:
    """review UI가 사용할 source payload를 반환한다."""
    _ensure_document_exists(document_id)
    return build_review_source(document_id)


@router.get("/{document_id}/review/decisions")
def get_review_decisions(document_id: str) -> dict[str, Any]:
    """저장된 review decisions를 반환한다."""
    _ensure_document_exists(document_id)
    return load_review_decisions(document_id)


@router.post("/{document_id}/review/decisions")
def post_review_decisions(
    document_id: str,
    body: ReviewDecisionsBody,
) -> dict[str, Any]:
    """review decisions를 저장한다."""
    _ensure_document_exists(document_id)
    saved = save_review_decisions(
        document_id,
        element_decisions={
            key: value.model_dump()
            for key, value in body.element_decisions.items()
        },
        exact_text_drop=body.exact_text_drop,
    )
    return {"status": "saved", "review_decisions": saved}


@router.post("/{document_id}/review/apply")
def post_review_apply(document_id: str) -> dict[str, Any]:
    """saved review decisions를 반영해 reviewed 산출물을 생성한다."""
    _ensure_document_exists(document_id)
    return {"status": "completed", "result": apply_review_overlay(document_id)}


@router.get("/{document_id}/review/result")
def get_review_result(document_id: str) -> dict[str, Any]:
    """review overlay가 반영된 reviewed_cleaned.json을 반환한다."""
    _ensure_document_exists(document_id)
    paths = build_document_paths(document_id)
    return _load_json_file(paths.reviewed_cleaned_json)


def _resolve_stage_asset_path(document_id: str, stage: str, relative_path: str) -> Path:
    paths = build_document_paths(document_id)
    root_map = {
        "source": paths.source_dir,
        "stage1": paths.stage1_dir,
        "stage2": paths.stage2_dir,
        "review": paths.review_dir,
        "stage3": paths.stage3_dir,
        "stage4": paths.stage4_dir,
    }
    root = root_map.get(stage)
    if root is None:
        raise HTTPException(status_code=400, detail=f"unsupported stage: {stage}")

    resolved = (root / relative_path).resolve()
    try:
        resolved.relative_to(root.resolve())
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid asset path") from exc
    if not resolved.exists():
        raise HTTPException(status_code=404, detail="asset not found")
    return resolved


@router.get("/{document_id}/assets/{stage}/{relative_path:path}")
def get_stage_asset(document_id: str, stage: str, relative_path: str) -> FileResponse:
    """stage 디렉터리 아래 산출물/이미지 파일을 그대로 내려준다."""
    _ensure_document_exists(document_id)
    path = _resolve_stage_asset_path(document_id, stage, relative_path)
    return FileResponse(path)


@router.get("/{document_id}/review/preview")
def get_review_preview(document_id: str) -> FileResponse:
    """reviewed preview HTML 파일을 내려준다."""
    _ensure_document_exists(document_id)
    paths = build_document_paths(document_id)
    if not paths.reviewed_preview_html.exists():
        raise HTTPException(status_code=404, detail="reviewed preview not found")
    return FileResponse(paths.reviewed_preview_html, media_type="text/html")
