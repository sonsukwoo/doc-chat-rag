"""Stage-5 chatbot structured output models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class DocumentSelectionResult(BaseModel):
    """문서 프로파일만 보고 질문 대상 문서를 고르는 구조화 응답."""

    query_type: Literal[
        "single_document",
        "multi_document",
        "comparison",
        "thread_wide",
        "conversation_memory",
        "open_domain",
    ] = Field(
        description=(
            "질문 유형. single_document, multi_document, comparison, "
            "thread_wide, conversation_memory, open_domain 중 하나."
        )
    )
    selected_document_ids: list[str] = Field(
        default_factory=list,
        description="현재 스레드 문서 중 검색 대상으로 선택한 document_id 목록.",
    )
    per_document_queries: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "문서별 검색 질의를 따로 쓰는 편이 좋을 때만 document_id별 질의를 제공."
        ),
    )
    retrieval_mode: Literal["dense", "hybrid"] | None = Field(
        default=None,
        description=(
            "문서 검색이 필요할 때 권장 검색 모드. 확실하지 않으면 null."
        ),
    )
    answer_strategy: Literal[
        "profile_only",
        "retrieve_chunks",
        "conversation_memory",
        "direct",
    ] | None = Field(
        default=None,
        description=(
            "문서 프로파일만으로 답할 수 있으면 profile_only, "
            "실제 문서 청크 검색이 필요하면 retrieve_chunks, "
            "대화 메모를 기반으로 답해야 하면 conversation_memory, "
            "문서와 무관한 일반 답변이면 direct, "
            "확실하지 않으면 null."
        ),
    )
    clarification_question: str | None = Field(
        default=None,
        description=(
            "현재는 사용하지 않는 필드. 특별한 질문이 없으면 null."
        ),
    )


class FinalAnswerResult(BaseModel):
    """최종 grounded answer 생성 시 사용하는 구조화 응답."""

    answer: str = Field(
        description="제공된 문맥만 바탕으로 작성한 최종 한국어 답변 본문."
    )
    grounded: bool = Field(
        description="제공된 문맥만으로 질문에 답할 수 있으면 true, 부족하면 false."
    )


class GroundingDecisionResult(BaseModel):
    """grounding 이후 그래프가 취할 다음 action을 결정하는 구조화 응답."""

    action: Literal["answer", "retrieve_deeper", "clarify"] = Field(
        description=(
            "현재 근거만으로 충분하면 answer, "
            "추가 검색이 필요하면 retrieve_deeper, "
            "질문 범위가 모호하면 clarify."
        )
    )
    clarification_question: str | None = Field(
        default=None,
        description="action이 clarify일 때만 사용자에게 되물을 짧은 한국어 질문.",
    )


GroundingCheckResult = GroundingDecisionResult
