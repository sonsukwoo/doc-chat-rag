"""Stage-5 chatbot input/output schemas."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import AnyMessage


class ChatbotCitationPayload(TypedDict, total=False):
    """최종 답변에 첨부할 citation 최소 단위."""

    document_id: str
    chunk_id: str
    parent_id: str | None
    page: int | None
    section_title: str | None
    asset_ref: str | None
    asset_relative_path: str | None


class ChatbotInterruptPayload(TypedDict, total=False):
    """사용자 clarification이 필요할 때 프론트로 보낼 payload."""

    kind: Literal["clarification"]
    question: str
    reason: str
    options: list[str]


class ChatbotVisualAssetPayload(TypedDict, total=False):
    """검색 결과와 함께 프론트에 노출할 visual asset 메타데이터."""

    asset_ref: str
    document_id: str
    chunk_id: str
    asset_kind: str
    relative_path: str
    asset_stage: str
    page: int | None
    caption: str | None
    summary_text: str | None
    heading_path: list[str]
    pages: list[int]


class ChatbotEvidenceChunkPayload(TypedDict, total=False):
    """최종 답변 작성에 사용된 텍스트 근거 청크 요약."""

    document_id: str
    chunk_id: str
    parent_id: str | None
    page: int | None
    section_title: str | None
    chunk_type: str | None
    text_excerpt: str


class ChatbotToolTracePayload(TypedDict, total=False):
    """툴 실행 내역을 프론트에 보여주기 위한 디버그 정보."""

    name: str
    label: str
    status: str | None
    query: str | None
    document_ids: list[str]
    chunk_ids: list[str]
    asset_ref: str | None
    retrieved_count: int | None
    retrieval_mode: str | None
    rerank_requested: bool | None
    rerank_applied: bool | None
    rerank_error: str | None
    mmr_requested: bool | None
    mmr_applied: bool | None
    per_document_search_used: bool | None
    score_threshold_applied: float | None
    score_fallback_applied: bool | None
    top_k: int | None
    fetch_k: int | None
    block_count: int | None
    message: str | None


class ChatbotDebugTracePayload(TypedDict, total=False):
    """질문 분류/문서 선택/툴 호출 흐름을 보여주기 위한 trace."""

    model: str
    query_kind: str
    selection_type: str
    selection_source: str
    answer_strategy: str
    selection_reason: str
    selected_document_ids: list[str]
    selected_document_queries: dict[str, str]
    thread_default_retrieval_mode: str | None
    retrieval_mode: str | None
    executed_retrieval_mode: str | None
    logs: list[str]
    tool_calls: list[ChatbotToolTracePayload]


class Stage5Input(TypedDict, total=False):
    """stage5 챗봇 진입 시 외부에서 전달하는 입력."""

    thread_id: str
    thread_name: str | None
    user_id: str | None
    user_message: str
    messages: list[AnyMessage]
    thread_default_retrieval_mode: str
    active_document_ids: list[str]
    document_profiles: list[dict[str, Any]]
    collection_name: str | None
    conversation_summary: str | None
    user_facts: dict[str, str]
    metadata: dict[str, Any]
    allow_web_search: bool


class Stage5Output(TypedDict, total=False):
    """stage5 챗봇 응답 정규화 출력."""

    status: Literal["completed", "interrupted"]
    thread_id: str
    final_answer: str | None
    citations: list[ChatbotCitationPayload]
    visual_assets: list[ChatbotVisualAssetPayload]
    visual_asset_refs: list[str]
    evidence_chunks: list[ChatbotEvidenceChunkPayload]
    interrupt: ChatbotInterruptPayload | None
    retrieval_mode: str
    logs: list[str]
    debug_trace: ChatbotDebugTracePayload | None
