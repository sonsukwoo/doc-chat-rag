"""Stage-5 chatbot graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from backend.stage4_retrieval.schemas import RetrievedChunkPayload

from .schemas import (
    ChatbotCitationPayload,
    ChatbotDebugTracePayload,
    ChatbotEvidenceChunkPayload,
    ChatbotInterruptPayload,
)


class QueryAnalysisPayload(TypedDict, total=False):
    """질문 해석 결과."""

    query_text: str
    query_kind: Literal[
        "general",
        "document_scoped",
        "document_grounded",
        "conversation_memory",
        "open_domain_unrelated",
        "ambiguous",
        "lexical",
        "smalltalk",
    ]
    needs_clarification: bool
    reason: str
    selected_document_ids: list[str]
    selected_document_queries: dict[str, str]
    selection_type: Literal[
        "single_document",
        "multi_document",
        "comparison",
        "thread_wide",
        "open_domain",
        "clarification_needed",
    ]
    selection_source: Literal["deterministic", "llm", "fallback"]
    answer_strategy: Literal["profile_only", "retrieve_chunks"] | None
    retrieval_mode_hint: Literal["dense", "hybrid"] | None
    use_per_document_search: bool
    matched_profile_topics: list[str]
    document_match_score: float


class RetrievalPolicyPayload(TypedDict, total=False):
    """검색 정책 결정 결과."""

    mode: Literal["dense", "hybrid"]
    use_rerank: bool
    enable_mmr: bool
    use_web_search: bool
    top_k: int
    score_threshold: float | None
    use_context_window: bool
    context_window_size: int


class GroundingDecisionPayload(TypedDict, total=False):
    """검색 근거가 충분한지에 대한 판단 결과."""

    action: Literal["answer", "retrieve_deeper", "clarify"]
    clarification_question: str | None


class ChatbotState(TypedDict, total=False):
    """stage5 챗봇 전체 공유 상태."""

    messages: Annotated[list[AnyMessage], add_messages]
    thread_id: str
    thread_name: str | None
    user_id: str | None
    user_message: str
    thread_default_retrieval_mode: str
    active_document_ids: list[str]
    retrieval_document_ids: list[str]
    retrieval_document_queries: dict[str, str]
    use_per_document_search: bool
    document_profiles: list[dict[str, object]]
    collection_name: str | None
    conversation_summary: str | None
    user_facts: dict[str, str]
    log_cursor: int
    query_analysis: QueryAnalysisPayload
    retrieval_policy: RetrievalPolicyPayload
    retrieval_hits: list[RetrievedChunkPayload]
    expanded_context_blocks: list[str]
    grounding_decision: GroundingDecisionPayload
    citations: list[ChatbotCitationPayload]
    evidence_chunks: list[ChatbotEvidenceChunkPayload]
    needs_clarification: bool
    clarification_payload: ChatbotInterruptPayload | None
    clarification_response: str | None
    answer_draft: str | None
    final_answer: str | None
    debug_trace: ChatbotDebugTracePayload | None
    logs: Annotated[list[str], operator.add]
