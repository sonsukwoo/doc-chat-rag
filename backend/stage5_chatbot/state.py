"""Stage-5 chatbot graph state."""

from __future__ import annotations

import operator
from typing import Annotated, Literal, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from backend.stage4_retrieval.schemas import RetrievedChunkPayload

from .schemas import ChatbotCitationPayload, ChatbotInterruptPayload


class QueryAnalysisPayload(TypedDict, total=False):
    """질문 해석 결과."""

    query_text: str
    query_kind: Literal["general", "document_scoped", "ambiguous", "lexical"]
    needs_clarification: bool
    reason: str


class RetrievalPolicyPayload(TypedDict, total=False):
    """검색 정책 결정 결과."""

    mode: Literal["dense", "hybrid"]
    use_rerank: bool
    use_web_search: bool
    top_k: int


class GroundingDecisionPayload(TypedDict, total=False):
    """검색 근거가 충분한지에 대한 판단 결과."""

    enough_evidence: bool
    needs_deeper_retrieval: bool
    needs_clarification: bool
    clarification_question: str | None
    missing_aspects: list[str]


class ChatbotState(TypedDict, total=False):
    """stage5 챗봇 전체 공유 상태."""

    messages: Annotated[list[AnyMessage], add_messages]
    room_id: str
    thread_id: str
    user_id: str | None
    user_message: str
    active_document_ids: list[str]
    collection_name: str | None
    query_analysis: QueryAnalysisPayload
    retrieval_policy: RetrievalPolicyPayload
    retrieval_hits: list[RetrievedChunkPayload]
    expanded_context_blocks: list[str]
    grounding_decision: GroundingDecisionPayload
    citations: Annotated[list[ChatbotCitationPayload], operator.add]
    needs_clarification: bool
    clarification_payload: ChatbotInterruptPayload | None
    clarification_response: str | None
    answer_draft: str | None
    final_answer: str | None
    logs: Annotated[list[str], operator.add]
