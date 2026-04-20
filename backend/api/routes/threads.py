"""채팅 스레드(thread) CRUD 라우터."""

from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from pydantic import BaseModel, Field

from backend.services import (
    bootstrap_thread_for_review,
    create_thread,
    create_thread_with_document,
    delete_thread_permanently,
    finalize_thread_document_review,
    get_thread_detail,
    list_thread_document_records,
    list_threads,
    prepare_uploaded_thread_document_for_review,
    update_thread,
    upload_document_to_thread,
    upload_thread_document_for_review,
)


router = APIRouter(prefix="/threads", tags=["threads"])


class CreateThreadBody(BaseModel):
    thread_name: str = Field(min_length=1)
    description: str | None = None
    default_retrieval_mode: Literal["dense", "hybrid"] = "dense"
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateThreadBody(BaseModel):
    thread_name: str | None = None
    description: str | None = None
    default_retrieval_mode: Literal["dense", "hybrid"] | None = None
    metadata: dict[str, Any] | None = None


@router.get("")
def get_threads(include_archived: bool = False) -> dict[str, Any]:
    """thread 목록을 반환한다."""
    return {"threads": list_threads(include_archived=include_archived)}


@router.post("", status_code=201)
def post_thread(body: CreateThreadBody) -> dict[str, Any]:
    """새 thread를 생성한다."""
    try:
        thread = create_thread(
            thread_name=body.thread_name,
            description=body.description,
            default_retrieval_mode=body.default_retrieval_mode,
            metadata=body.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"thread": thread}


@router.post("/with-document", status_code=201)
async def post_thread_with_document(
    thread_name: str = Form(...),
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    default_retrieval_mode: Literal["dense", "hybrid"] = Form(default="dense"),
) -> dict[str, Any]:
    """draft thread를 만들고 첫 문서를 바로 업로드한다."""
    original_filename = file.filename or "uploaded.pdf"
    payload = await file.read()
    try:
        result = create_thread_with_document(
            thread_name=thread_name,
            original_filename=original_filename,
            content=payload,
            description=description,
            default_retrieval_mode=default_retrieval_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return result


@router.post("/bootstrap", status_code=201)
async def post_thread_bootstrap(
    thread_name: str = Form(...),
    file: UploadFile = File(...),
    description: str | None = Form(default=None),
    default_retrieval_mode: Literal["dense", "hybrid"] = Form(default="dense"),
) -> dict[str, Any]:
    """thread 생성 후 stage1/stage2까지 실행해 review 단계로 보낸다."""
    original_filename = file.filename or "uploaded.pdf"
    payload = await file.read()
    try:
        return bootstrap_thread_for_review(
            thread_name=thread_name,
            original_filename=original_filename,
            content=payload,
            description=description,
            default_retrieval_mode=default_retrieval_mode,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.get("/{thread_id}")
def get_thread(thread_id: str, include_archived: bool = False) -> dict[str, Any]:
    """thread 상세를 반환한다."""
    thread = get_thread_detail(thread_id, include_archived=include_archived)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"thread": thread}


@router.get("/{thread_id}/documents")
def get_thread_documents(thread_id: str) -> dict[str, Any]:
    """현재 thread에 연결된 문서 목록을 반환한다."""
    thread = get_thread_detail(thread_id)
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"documents": list_thread_document_records(thread_id)}


@router.post("/{thread_id}/documents/upload")
async def post_thread_document_upload(
    thread_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """기존 thread에 문서를 업로드하고 연결한다."""
    original_filename = file.filename or "uploaded.pdf"
    payload = await file.read()
    try:
        return upload_document_to_thread(
            thread_id=thread_id,
            original_filename=original_filename,
            content=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{thread_id}/documents/process-upload")
async def post_thread_document_process_upload(
    thread_id: str,
    file: UploadFile = File(...),
) -> dict[str, Any]:
    """기존 thread에 문서를 업로드하고 stage1/stage2까지 실행한다."""
    original_filename = file.filename or "uploaded.pdf"
    payload = await file.read()
    try:
        return upload_thread_document_for_review(
            thread_id=thread_id,
            original_filename=original_filename,
            content=payload,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{thread_id}/documents/{document_id}/prepare-review")
def post_thread_document_prepare_review(
    thread_id: str,
    document_id: str,
) -> dict[str, Any]:
    """이미 업로드된 thread 문서를 stage1/stage2까지 실행한다."""
    try:
        return prepare_uploaded_thread_document_for_review(
            thread_id=thread_id,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/{thread_id}/documents/{document_id}/finalize-review")
def post_thread_document_finalize_review(
    thread_id: str,
    document_id: str,
) -> dict[str, Any]:
    """review overlay 반영 후 stage3 indexing까지 완료한다."""
    try:
        return finalize_thread_document_review(
            thread_id=thread_id,
            document_id=document_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.patch("/{thread_id}")
def patch_thread(thread_id: str, body: UpdateThreadBody) -> dict[str, Any]:
    """thread 메타데이터를 수정한다."""
    thread = update_thread(
        thread_id,
        thread_name=body.thread_name,
        description=body.description,
        default_retrieval_mode=body.default_retrieval_mode,
        metadata=body.metadata,
    )
    if thread is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"thread": thread}


@router.delete("/{thread_id}")
def delete_thread(thread_id: str) -> dict[str, Any]:
    """thread와 연관 런타임 데이터를 영구 삭제한다."""
    result = delete_thread_permanently(thread_id)
    if result is None:
        raise HTTPException(status_code=404, detail="thread not found")
    return {"status": "deleted", **result}
