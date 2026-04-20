"""Stage-5 chatbot prompt helpers."""

from __future__ import annotations


def build_stage5_agent_system_prompt(
    *,
    active_document_ids: list[str],
    retrieval_mode: str,
) -> str:
    """tool-calling agent용 시스템 프롬프트를 구성한다."""
    joined_document_ids = ", ".join(active_document_ids) if active_document_ids else "없음"
    return (
        "당신은 thread-scoped 문서 QA 에이전트입니다.\n"
        "현재 스레드에 연결된 문서 범위 안에서만 답하세요.\n"
        f"현재 연결 문서: {joined_document_ids}\n"
        f"기본 retrieval 모드: {retrieval_mode}\n"
        "문서 내용 질문이면 먼저 search_thread_knowledge를 사용해 근거를 확인하세요.\n"
        "검색 결과만으로 맥락이 부족하면 expand_context_window를 사용하세요.\n"
        "표나 이미지 근거가 보이면 load_visual_asset를 사용하세요. asset_ref는 document_id:chunk_id 형식입니다.\n"
        "문서 목록 질문이면 list_thread_documents를 사용하세요.\n"
        "근거가 불충분하면 추측하지 말고, 문서 범위가 모호하면 사용자에게 다시 물어보세요.\n"
        "답변은 짧고 정확하게 작성하고, 가능한 경우 페이지/섹션/표/이미지 참조를 언급하세요."
    )


def build_stage5_answer_system_prompt() -> str:
    """검색 근거를 바탕으로 최종 답변을 작성할 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 문서 근거 기반 답변 작성기입니다.\n"
        "제공된 검색 결과만 사용해 답하세요.\n"
        "근거에 없는 내용은 추측하지 말고, 찾지 못했다고 명시하세요.\n"
        "핵심 답변을 먼저 말하고, 필요하면 페이지/섹션/표/이미지 참조를 함께 언급하세요."
    )


def build_stage5_answer_user_prompt(
    *,
    query_text: str,
    context_blocks: list[str],
) -> str:
    """최종 답변 생성에 사용할 사용자 프롬프트를 구성한다."""
    context_text = "\n\n".join(context_blocks).strip() or "근거 없음"
    return (
        f"사용자 질문:\n{query_text}\n\n"
        f"검색 근거:\n{context_text}\n\n"
        "위 근거만 사용해 답변하세요."
    )


def build_stage5_grounding_system_prompt() -> str:
    """현재 retrieval 근거가 질문에 충분한지 판단할 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 문서 QA grounding 판정기입니다.\n"
        "질문, 현재 답변 초안, 검색 근거를 보고 세 가지를 판단하세요.\n"
        "1. 현재 근거만으로 답변 가능한가\n"
        "2. 추가 검색이 필요한가\n"
        "3. 사용자에게 질문을 다시 물어야 하는가\n"
        "근거가 있더라도 질문에 직접 답하지 못하면 enough_evidence를 true로 두지 마세요.\n"
        "질문 범위가 모호할 때만 needs_clarification을 true로 두세요.\n"
        "충분히 답할 수 있으면 needs_deeper_retrieval과 needs_clarification은 false여야 합니다."
    )


def build_stage5_grounding_user_prompt(
    *,
    query_text: str,
    answer_draft: str | None,
    context_blocks: list[str],
) -> str:
    """grounding check에 사용할 사용자 프롬프트를 구성한다."""
    context_text = "\n\n".join(context_blocks).strip() or "근거 없음"
    normalized_answer_draft = answer_draft.strip() if answer_draft else "(없음)"
    return (
        f"사용자 질문:\n{query_text}\n\n"
        f"현재 답변 초안:\n{normalized_answer_draft}\n\n"
        f"검색 근거:\n{context_text}\n\n"
        "위 정보를 보고 현재 답변 가능 여부를 판단하세요."
    )
