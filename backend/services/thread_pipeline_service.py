"""thread 기반 업로드-전처리-검수-인덱싱 오케스트레이션 서비스."""

from __future__ import annotations

from typing import Any, TypedDict

from backend.document_store import load_document_record
from backend.review_overlay import apply_review_overlay

from .pipeline_runner import (
    run_stage1_for_document,
    run_stage2_for_document,
    run_stage3_for_document,
)
from .thread_service import (
    ThreadPayload,
    create_thread_with_document,
    get_thread_detail,
    update_thread,
    upload_document_to_thread,
)


class ThreadDocumentPipelinePayload(TypedDict, total=False):
    """프론트가 thread 문서 진행 상태를 읽을 때 쓰는 최소 응답 스키마."""

    thread: ThreadPayload
    document: dict[str, Any]
    stage_status: dict[str, str]
    review: dict[str, Any]
    indexing: dict[str, Any]
    next_step: str


def _ensure_thread_document_membership(
    thread_id: str,
    document_id: str,
) -> ThreadPayload:
    """해당 문서가 thread에 실제 연결되어 있는지 검증한다."""
    thread = get_thread_detail(thread_id)
    if thread is None:
        raise FileNotFoundError("thread not found")

    active_document_ids = {
        str(item).strip() for item in thread.get("active_document_ids") or []
    }
    if str(document_id).strip() not in active_document_ids:
        raise FileNotFoundError("document is not attached to the thread")
    return thread


def _build_review_links(document_id: str) -> dict[str, str]:
    return {
        "source_url": f"/documents/{document_id}/review/source",
        "decisions_url": f"/documents/{document_id}/review/decisions",
        "apply_url": f"/documents/{document_id}/review/apply",
        "preview_url": f"/documents/{document_id}/review/preview",
    }


def _build_review_ready_payload(
    *,
    thread_id: str,
    document_id: str,
) -> ThreadDocumentPipelinePayload:
    """stage2 완료 후 review 단계로 진입한 상태 payload를 만든다."""
    thread = get_thread_detail(thread_id)
    if thread is None:
        raise FileNotFoundError("thread not found")

    document_record = load_document_record(document_id)
    update_thread(
        thread_id,
        metadata={"lifecycle_status": "review_pending"},
    )

    return {
        "thread": get_thread_detail(thread_id) or thread,
        "document": document_record,
        "stage_status": {
            "upload": "completed",
            "stage1": "completed",
            "stage2": "completed",
            "review": "pending",
            "stage3": "not_started",
        },
        "review": _build_review_links(document_id),
        "next_step": "review",
    }


def prepare_uploaded_thread_document_for_review(
    *,
    thread_id: str,
    document_id: str,
) -> ThreadDocumentPipelinePayload:
    """이미 업로드된 thread 문서를 stage1/stage2까지 실행해 review 단계로 보낸다."""
    thread = _ensure_thread_document_membership(thread_id, document_id)
    run_stage1_for_document(document_id)
    run_stage2_for_document(document_id)
    return _build_review_ready_payload(
        thread_id=str(thread.get("thread_id") or thread_id),
        document_id=document_id,
    )


def upload_thread_document_for_review(
    *,
    thread_id: str,
    original_filename: str,
    content: bytes,
) -> ThreadDocumentPipelinePayload:
    """기존 thread에 문서를 업로드한 뒤 stage1/stage2까지 실행한다."""
    upload_result = upload_document_to_thread(
        thread_id=thread_id,
        original_filename=original_filename,
        content=content,
    )
    document = dict(upload_result.get("document") or {})
    document_id = str(document.get("document_id") or "").strip()
    if not document_id:
        raise RuntimeError("uploaded document_id is missing")
    return prepare_uploaded_thread_document_for_review(
        thread_id=thread_id,
        document_id=document_id,
    )


def bootstrap_thread_for_review(
    *,
    thread_name: str,
    original_filename: str,
    content: bytes,
    description: str | None = None,
    default_retrieval_mode: str = "dense",
    metadata: dict[str, Any] | None = None,
) -> ThreadDocumentPipelinePayload:
    """thread 생성과 첫 문서 업로드 후 stage1/stage2까지 한 번에 실행한다."""
    upload_result = create_thread_with_document(
        thread_name=thread_name,
        original_filename=original_filename,
        content=content,
        description=description,
        default_retrieval_mode=default_retrieval_mode,
        metadata=metadata,
    )
    thread = dict(upload_result.get("thread") or {})
    document = dict(upload_result.get("document") or {})
    thread_id = str(thread.get("thread_id") or "").strip()
    document_id = str(document.get("document_id") or "").strip()
    if not thread_id or not document_id:
        raise RuntimeError("thread bootstrap payload is incomplete")

    return prepare_uploaded_thread_document_for_review(
        thread_id=thread_id,
        document_id=document_id,
    )


def finalize_thread_document_review(
    *,
    thread_id: str,
    document_id: str,
) -> ThreadDocumentPipelinePayload:
    """saved review overlay를 반영한 뒤 stage3 indexing까지 완료한다."""
    thread = _ensure_thread_document_membership(thread_id, document_id)
    collection_name = str(thread.get("collection_name") or "").strip() or None
    review_result = apply_review_overlay(document_id)
    stage3_result = run_stage3_for_document(
        document_id,
        thread_id=thread_id,
        collection_name=collection_name,
    )
    update_thread(
        thread_id,
        metadata={"lifecycle_status": "ready"},
    )

    return {
        "thread": get_thread_detail(thread_id) or thread,
        "document": load_document_record(document_id),
        "stage_status": {
            "upload": "completed",
            "stage1": "completed",
            "stage2": "completed",
            "review": "completed",
            "stage3": "completed",
        },
        "review": {
            **review_result,
            **_build_review_links(document_id),
        },
        "indexing": stage3_result,
        "next_step": "chat_ready",
    }
