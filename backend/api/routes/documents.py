"""문서 업로드/조회 라우터."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from backend.document_store import (
    build_document_paths,
    create_document_record,
    list_document_records,
    load_document_record,
    save_uploaded_pdf,
    update_document_stage_record,
)


router = APIRouter(prefix="/documents", tags=["documents"])


def _not_found(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def _load_json_file(path: Path) -> dict:
    if not path.exists():
        raise _not_found(f"artifact not found: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail=f"invalid json artifact: {path.name}")
    return payload


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    """원본 PDF를 업로드하고 문서 ID를 발급한다."""
    original_filename = file.filename or "uploaded.pdf"
    if not original_filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="only PDF upload is supported")

    record = create_document_record(original_filename=original_filename)
    document_id = str(record["document_id"])
    payload = await file.read()
    paths = build_document_paths(document_id)
    saved_path = save_uploaded_pdf(document_id=document_id, content=payload)
    updated_record = update_document_stage_record(
        document_id=document_id,
        stage="upload",
        status="uploaded",
        outputs={"source_pdf_path": str(saved_path)},
    )
    return {
        "document": updated_record,
        "paths": {
            "source_pdf_path": str(paths.source_pdf),
        },
    }


@router.get("")
def list_documents() -> dict:
    """등록된 문서 목록을 반환한다."""
    return {"documents": list_document_records()}


@router.get("/{document_id}")
def get_document(document_id: str) -> dict:
    """문서 메타데이터와 주요 경로를 반환한다."""
    try:
        record = load_document_record(document_id)
    except FileNotFoundError as exc:
        raise _not_found(str(exc)) from exc

    paths = build_document_paths(document_id)
    return {
        "document": record,
        "paths": {
            "source_pdf_path": str(paths.source_pdf),
            "stage1_raw_json_path": str(paths.stage1_raw_json),
            "stage2_cleaned_json_path": str(paths.stage2_cleaned_json),
            "reviewed_cleaned_json_path": str(paths.reviewed_cleaned_json),
            "stage3_chunks_json_path": str(paths.stage3_chunks_json),
        },
    }


@router.get("/{document_id}/stage2/cleaned")
def get_stage2_cleaned_json(document_id: str) -> dict:
    """stage2 cleaned.json 내용을 그대로 반환한다."""
    paths = build_document_paths(document_id)
    return _load_json_file(paths.stage2_cleaned_json)


@router.get("/{document_id}/stage2/preview")
def get_stage2_preview_html(document_id: str) -> FileResponse:
    """stage2 preview.html 파일을 바로 내려준다."""
    paths = build_document_paths(document_id)
    if not paths.stage2_preview_html.exists():
        raise _not_found("stage2 preview not found")
    return FileResponse(paths.stage2_preview_html, media_type="text/html")
