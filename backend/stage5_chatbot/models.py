"""Stage-5 chatbot structured output models."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class IntentClassificationResult(BaseModel):
    """초기 질문 분류에서 큰 방향만 결정하는 구조화 응답."""

    answer_strategy: Literal[
        "direct",
        "retrieve_chunks",
        "conversation_memory",
        "memory_only",
    ] = Field(
        description=(
            "문서와 무관한 일반 질문이면 direct, "
            "실제 문서 청크 검색이 필요하면 retrieve_chunks. "
            "conversation_memory/memory_only는 LLM 응답 호환용이며 "
            "노드에서 direct+memory_only로 정규화된다."
        )
    )
    memory_mode: Literal[
        "none",
        "memory_only",
        "resolve_for_retrieval",
    ] = Field(
        description=(
            "이전 대화 기억이 불필요하면 none, "
            "이전 대화만으로 답할 수 있으면 memory_only, "
            "이전 대화 기억이 필요하지만 최종 답은 다시 검색해야 하면 "
            "resolve_for_retrieval."
        )
    )
    reason: str | None = Field(
        default=None,
        description="현재 분류를 선택한 짧은 이유. 없으면 null.",
    )


class RetrievalTask(BaseModel):
    """질문을 검색 단위 task로 분해한 결과."""

    task_id: str = Field(
        description="task 식별자. 예: task-1"
    )
    subquery: str = Field(
        description=(
            "해당 task에 사용할 검색 질의. "
            "사용자 원문 표현을 최대한 유지하고 프로파일 제목/토픽을 덧붙이지 않는다."
        )
    )
    user_question: str | None = Field(
        default=None,
        description=(
            "사용자가 실제로 물은 하위 질문 원문. 없으면 subquery와 동일하게 처리한다."
        ),
    )
    search_query: str | None = Field(
        default=None,
        description=(
            "검색 엔진에 넣을 질의. 원문 표현 보존이 기본이며, "
            "문서 요약/개요처럼 검색 전략이 별도인 경우 null이어도 된다."
        ),
    )
    task_type: Literal[
        "fact_lookup",
        "exact_keyword",
        "document_summary",
        "comparison",
        "procedure",
        "figure_table",
        "conversation_memory",
        "general",
    ] | None = Field(
        default=None,
        description=(
            "하위 질문 유형. 문서 전체 요약/개요/설명은 document_summary를 사용한다."
        ),
    )
    retrieval_strategy: Literal[
        "vector_search",
        "hybrid_search",
        "document_overview",
        "balanced_multi_document",
        "asset_lookup",
        "conversation_only",
        "no_retrieval",
    ] | None = Field(
        default=None,
        description=(
            "실행 검색 전략. 문서 전체 요약/개요는 document_overview, "
            "정확 키워드/표/그림/함수명은 hybrid_search를 우선 고려한다."
        ),
    )
    document_ids: list[str] = Field(
        default_factory=list,
        description="이 task를 검색할 document_id 목록.",
    )


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
    retrieval_tasks: list[RetrievalTask] = Field(
        default_factory=list,
        description=(
            "실제 검색에 사용할 retrieval task 목록. "
            "질문이 하나면 1개 task, 여러 독립 질문이면 여러 task를 반환한다."
        ),
    )
    retrieval_mode: Literal["dense", "hybrid"] | None = Field(
        default=None,
        description=(
            "문서 검색이 필요할 때 권장 검색 모드. 확실하지 않으면 null."
        ),
    )
    answer_strategy: Literal[
        "retrieve_chunks",
        "conversation_memory",
        "direct",
    ] | None = Field(
        default=None,
        description=(
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
