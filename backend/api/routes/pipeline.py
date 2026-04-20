"""문서별 stage 실행 라우터."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.document_store import build_document_paths, load_document_record
from backend.services import (
    run_stage1_for_document,
    run_stage2_for_document,
)


router = APIRouter(prefix="/documents", tags=["pipeline"])


def _ensure_document_exists(document_id: str) -> None:
    try:
        load_document_record(document_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{document_id}/stage1")
def run_stage1(document_id: str) -> dict:
    """업로드된 원본 PDF를 raw.json으로 변환한다."""
    _ensure_document_exists(document_id)
    result = run_stage1_for_document(document_id)
    return {"status": "completed", "result": result}


@router.post("/{document_id}/stage2")
def run_stage2(document_id: str) -> dict:
    """raw.json을 cleaned 산출물로 변환한다."""
    _ensure_document_exists(document_id)
    result = run_stage2_for_document(document_id)
    return {
        "status": "completed",
        "result": {
            "output_paths": result.get("output_paths"),
            "cleaned_element_count": len(result.get("cleaned_elements", [])),
            "logs": result.get("logs", []),
        },
    }


@router.post("/{document_id}/stage3")
def run_stage3(document_id: str) -> dict:
    """문서 단독 stage3 실행은 더 이상 지원하지 않는다."""
    _ensure_document_exists(document_id)
    paths = build_document_paths(document_id)
    if not paths.stage2_cleaned_json.exists() and not paths.reviewed_cleaned_json.exists():
        raise HTTPException(
            status_code=400,
            detail="stage2 result is missing; run stage2 before stage3",
        )

    raise HTTPException(
        status_code=400,
        detail=(
            "thread-scoped stage3 only supports the finalize-review flow. "
            "Use /threads/{thread_id}/documents/{document_id}/finalize-review instead."
        ),
    )
