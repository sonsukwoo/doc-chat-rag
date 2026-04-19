"""Stage-5 chatbot structured output models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class FinalAnswerResult(BaseModel):
    """최종 grounded answer 생성 시 사용하는 구조화 응답."""

    answer: str = Field(
        description="제공된 문맥만 바탕으로 작성한 최종 한국어 답변 본문."
    )
    grounded: bool = Field(
        description="제공된 문맥만으로 질문에 답할 수 있으면 true, 부족하면 false."
    )


class GroundingCheckResult(BaseModel):
    """retrieval 근거가 현재 질문에 충분한지 판단하는 구조화 응답."""

    enough_evidence: bool = Field(
        description="현재 검색 근거만으로 질문에 답할 수 있으면 true."
    )
    needs_deeper_retrieval: bool = Field(
        description="현재 근거가 일부 관련되지만 부족해 추가 검색이 필요하면 true."
    )
    needs_clarification: bool = Field(
        description="질문 범위가 모호하거나 대상이 불명확해 사용자 확인이 필요하면 true."
    )
    clarification_question: str | None = Field(
        default=None,
        description="사용자에게 되물어야 한다면 보여줄 짧은 한국어 질문. 아니면 null.",
    )
    missing_aspects: list[str] = Field(
        default_factory=list,
        description="현재 부족한 정보 항목을 짧게 나열. 없으면 빈 리스트.",
    )
