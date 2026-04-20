"""Stage-5 chatbot prompt helpers."""

from __future__ import annotations

from backend.stage5_chatbot.document_selection import extract_numeric_filename_aliases


def _render_document_profile_line(profile: dict[str, object]) -> str:
    title = str(profile.get("title") or "").strip()
    document_type = str(profile.get("document_type") or "").strip()
    main_topics = [
        str(item).strip()
        for item in profile.get("main_topics") or []
        if str(item).strip()
    ]
    keywords = [
        str(item).strip()
        for item in profile.get("keywords") or []
        if str(item).strip()
    ]
    short_summary = str(profile.get("short_summary") or "").strip()
    parts = [
        title,
        document_type,
        ", ".join(main_topics[:3]) if main_topics else "",
        ", ".join(keywords[:4]) if keywords else "",
        short_summary,
    ]
    return " / ".join(part for part in parts if part)


def build_stage5_agent_system_prompt(
    *,
    active_document_ids: list[str],
    retrieval_mode: str,
    document_profiles: list[dict[str, object]] | None = None,
) -> str:
    """tool-calling agent용 시스템 프롬프트를 구성한다."""
    joined_document_ids = ", ".join(active_document_ids) if active_document_ids else "없음"
    profile_lines: list[str] = []
    for profile in document_profiles or []:
        compact = _render_document_profile_line(profile)
        if compact:
            profile_lines.append(f"- {compact}")
    joined_profiles = "\n".join(profile_lines) if profile_lines else "- (프로파일 없음)"
    return (
        "당신은 thread-scoped 문서 QA 에이전트입니다.\n"
        "현재 스레드에 연결된 문서 범위 안에서만 답하세요.\n"
        f"현재 답변 대상 문서: {joined_document_ids}\n"
        f"문서 프로파일:\n{joined_profiles}\n"
        f"기본 retrieval 모드: {retrieval_mode}\n"
        "문서 내용 질문이면 먼저 search_thread_knowledge를 사용해 근거를 확인하세요.\n"
        "검색 결과만으로 맥락이 부족하면 expand_context_window를 사용하세요.\n"
        "표나 이미지 근거가 보이면 load_visual_asset를 사용하세요. asset_ref는 document_id:chunk_id 형식입니다.\n"
        "문서 목록 질문이면 list_thread_documents를 사용하세요.\n"
        "근거가 불충분하면 추측하지 말고, 문서 범위가 모호하면 사용자에게 다시 물어보세요.\n"
        "답변은 짧고 정확하게 작성하고, 가능한 경우 페이지/섹션/표/이미지 참조를 언급하세요."
    )


def build_stage5_document_selection_system_prompt() -> str:
    """문서 프로파일만 보고 질문 대상 문서를 고를 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 thread-scoped 문서 선택기입니다.\n"
        "문서 원문 청크는 아직 보지 못했고, 제공된 문서 프로파일과 대화 메모만으로 판단해야 합니다.\n"
        "반드시 현재 후보 문서 목록에 있는 document_id만 선택하세요.\n"
        "질문이 한 문서만 가리키면 single_document를 선택하세요.\n"
        "질문이 여러 문서를 함께 다루면 multi_document 또는 comparison을 선택하세요.\n"
        "문서 전체를 넓게 훑어야 하면 thread_wide를 선택하세요.\n"
        "질문이 이전 대화, 직전 답변, 지금까지의 질문/대화를 묻는 경우 conversation_memory를 선택하세요.\n"
        "문서와 무관한 일반 질문이면 open_domain을 선택하세요.\n"
        "answer_strategy는 direct, conversation_memory, profile_only, retrieve_chunks 중 하나로 고르세요.\n"
        "profile_only는 오직 문서 제목, 문서 종류, main_topics, short_summary만으로 답할 수 있을 때만 선택하세요.\n"
        "사용자가 이전 대화, 지금까지, 방금, 앞서, 내가 물어본, 우리가 이야기한, 직전 답변, 최근 질문처럼 대화 이력을 묻는 경우 절대 profile_only를 선택하지 말고 conversation_memory를 선택하세요.\n"
        "코드, 함수, 인자, 페이지, 표, 그림, 절차, 수치처럼 원문 근거가 필요하면 retrieve_chunks를 선택하세요.\n"
        "직전 assistant가 문서 지정을 물었고, 사용자가 '1번 문서야', '2.pdf야', '첫 번째 문서'처럼 짧게 답한 경우에는 그 답을 문서 지정으로 해석해 single_document를 우선 선택하세요.\n"
        "여러 문서가 연결된 상태에서 표/그림/페이지/섹션 번호만 있고 문서가 특정되지 않으면 thread_wide로 두고 retrieve_chunks를 선택하세요.\n"
        "여러 문서가 연결된 상태에서 질문이 'Table 14', 'Figure 3', 'p.12', '3장'처럼 참조 번호만 말하고 문서 별칭/제목/파일명이 없으면 절대 특정 문서를 단정하지 말고 thread_wide 검색으로 넘기세요.\n"
        "예시 1: '1번 문서 요약해줘' -> single_document + profile_only\n"
        "예시 2: '2.pdf 핵심 주제 설명' -> single_document + profile_only\n"
        "예시 3: 'create_agent 인자 알려줘' -> single_document 또는 multi_document + retrieve_chunks\n"
        "예시 4: '지금까지 내가 질문한 것들 요약해줘' -> conversation_memory + conversation_memory\n"
        "예시 5: 'Figure 4 설명해줘'처럼 문서 지정 없는 참조 질의 -> thread_wide + retrieve_chunks\n"
        "문서별로 다른 검색 질의를 쓰는 편이 좋을 때만 per_document_queries를 채우고, 그렇지 않으면 비워 두세요.\n"
        "retrieval_mode는 기본적으로 null로 두고, 정확 키워드, 함수명, 클래스명, snake_case, CamelCase, 페이지/표/그림처럼 정확 매칭 성격이 강할 때 hybrid를 우선 고려하세요."
    )


def build_stage5_document_selection_user_prompt(
    *,
    thread_name: str | None,
    query_text: str,
    document_profiles: list[dict[str, object]],
    conversation_summary: str | None = None,
    recent_dialog_lines: list[str] | None = None,
) -> str:
    """문서 선택 LLM에 전달할 사용자 프롬프트를 구성한다."""
    header = f"스레드 이름: {thread_name}\n" if str(thread_name or "").strip() else ""
    profile_lines: list[str] = []
    for index, profile in enumerate(document_profiles, start=1):
        document_id = str(profile.get("document_id") or "").strip()
        original_filename = str(profile.get("original_filename") or "").strip() or "-"
        compact = _render_document_profile_line(profile) or "(프로파일 없음)"
        alias_candidates = [f"{index}번 문서"]
        alias_candidates.extend(extract_numeric_filename_aliases(original_filename))
        aliases = ", ".join(dict.fromkeys(alias_candidates))
        profile_lines.append(
            f"- document_id={document_id} | filename={original_filename} | aliases={aliases} | {compact}"
        )
    joined_profiles = "\n".join(profile_lines) if profile_lines else "- (프로파일 없음)"
    memory_parts: list[str] = []
    normalized_summary = str(conversation_summary or "").strip()
    if normalized_summary:
        memory_parts.append(f"대화 요약:\n{normalized_summary}")
    recent_lines = [str(item).strip() for item in recent_dialog_lines or [] if str(item).strip()]
    if recent_lines:
        memory_parts.append("최근 대화:\n" + "\n".join(f"- {line}" for line in recent_lines))
    joined_memory = "\n\n".join(memory_parts).strip() or "(대화 메모 없음)"
    return (
        f"{header}"
        f"사용자 질문:\n{query_text}\n\n"
        f"대화 메모:\n{joined_memory}\n\n"
        f"후보 문서 프로파일:\n{joined_profiles}\n\n"
        "후보 문서 중 필요한 문서를 고르고, 필요하면 문서별 검색 질의를 따로 제안하세요.\n"
        "대화 이력 기반 질문이면 conversation_memory를, 문서와 무관한 일반 질문이면 direct를 고르세요.\n"
        "단일 문서 또는 여러 문서의 설명/요약/비교처럼 프로파일만으로 답할 수 있으면 answer_strategy를 profile_only로 두세요.\n"
        "정확한 키워드, 함수명, 클래스명, 코드 식별자, 페이지/표/그림, 절차 같은 질문은 answer_strategy를 retrieve_chunks로 두고, hybrid가 유리하면 retrieval_mode를 hybrid로 지정하세요.\n"
        "질문이 표/그림/페이지 같은 참조를 포함하지만 어떤 문서인지 분명하지 않으면 특정 문서로 좁히지 말고 thread_wide 검색으로 넘기세요.\n"
        "특히 여러 문서가 연결된 상태에서 번호가 붙은 table/figure/page 참조만 있고 문서명이 없으면 추정하지 말고 thread_wide와 retrieve_chunks를 반환하세요."
        "반대로 최근 대화에서 assistant가 문서 선택을 요청했고, 이번 사용자 발화가 문서 별칭이나 파일명만 짧게 답한 경우에는 그 답을 문서 지정으로 해석해 single_document를 선택하세요."
    )


def build_stage5_profile_answer_system_prompt() -> str:
    """문서 프로파일만으로 설명/요약 답변을 생성할 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 문서 프로파일 기반 요약/설명 작성기입니다.\n"
        "제공된 문서 프로파일만 사용해 답하세요.\n"
        "여러 문서가 있으면 문서별로 구분해서 설명하세요.\n"
        "프로파일에 없는 세부 구현, 코드, 페이지, 정확한 수치, 절차는 추측하지 마세요.\n"
        "상세 정보가 부족하면 프로파일 기준의 개요만 답한다고 명시하세요."
    )


def build_stage5_profile_answer_user_prompt(
    *,
    query_text: str,
    document_profiles: list[dict[str, object]],
) -> str:
    """프로파일 기반 설명/요약 답변에 사용할 사용자 프롬프트를 구성한다."""
    profile_lines: list[str] = []
    for profile in document_profiles:
        document_id = str(profile.get("document_id") or "").strip() or "-"
        original_filename = str(profile.get("original_filename") or "").strip() or "-"
        title = str(profile.get("title") or "").strip() or "-"
        document_type = str(profile.get("document_type") or "").strip() or "-"
        main_topics = ", ".join(
            str(item).strip()
            for item in profile.get("main_topics") or []
            if str(item).strip()
        ) or "-"
        keywords = ", ".join(
            str(item).strip()
            for item in profile.get("keywords") or []
            if str(item).strip()
        ) or "-"
        short_summary = str(profile.get("short_summary") or "").strip() or "-"
        profile_lines.append(
            "\n".join(
                [
                    f"document_id: {document_id}",
                    f"filename: {original_filename}",
                    f"title: {title}",
                    f"document_type: {document_type}",
                    f"main_topics: {main_topics}",
                    f"keywords: {keywords}",
                    f"short_summary: {short_summary}",
                ]
            )
        )
    joined_profiles = "\n\n".join(profile_lines).strip() or "(프로파일 없음)"
    return (
        f"사용자 질문:\n{query_text}\n\n"
        f"선택 문서 프로파일:\n{joined_profiles}\n\n"
        "위 프로파일만 사용해 답변하세요."
    )


def build_stage5_answer_system_prompt() -> str:
    """검색 근거를 바탕으로 최종 답변을 작성할 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 문서 근거 기반 답변 작성기입니다.\n"
        "제공된 검색 결과만 사용해 답하세요.\n"
        "근거에 없는 내용은 추측하지 말고, 찾지 못했다고 명시하세요.\n"
        "여러 문서 근거가 섞여 있으면 document_id, filename, title 표기를 기준으로 문서를 구분하세요.\n"
        "서로 다른 문서의 내용을 같은 문서 설명으로 합치지 마세요.\n"
        "핵심 답변을 먼저 말하고, 필요하면 페이지/섹션/표/이미지 참조를 함께 언급하세요."
    )


def build_stage5_memory_system_prompt() -> str:
    """대화 메모리/사용자 facts만으로 답할 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 스레드 대화 메모리를 바탕으로 답하는 비서입니다.\n"
        "제공된 사용자 facts, 이전 대화 요약, 최근 대화만 사용하세요.\n"
        "기억에 없는 내용은 추측하지 말고 모른다고 답하세요.\n"
        "문서 검색이나 외부 정보를 끌어오지 마세요."
    )


def build_stage5_general_response_system_prompt() -> str:
    """문서와 무관한 일반 질문에 답할 때 쓰는 시스템 프롬프트."""
    return (
        "당신은 간결하게 답하는 일반 비서입니다.\n"
        "현재 질문은 문서 검색 없이 일반 답변으로 처리합니다.\n"
        "답변은 짧고 직접적으로 작성하세요."
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
        "질문, 현재 답변 초안, 검색 근거를 보고 다음 action 하나만 선택하세요.\n"
        "answer: 현재 근거만으로 질문에 직접 답할 수 있음\n"
        "retrieve_deeper: 현재 근거가 일부 관련되지만 부족해 추가 검색이 필요함\n"
        "clarify: 질문 대상 문서나 범위가 모호해 사용자 확인이 필요함\n"
        "근거가 질문에 직접 답하지 못하면 answer를 선택하지 마세요.\n"
        "검색 근거가 비어 있거나, 질문과 직접 맞는 청크가 보이지 않거나, 엇나간 청크만 보이면 answer보다 clarify를 우선 고려하세요.\n"
        "clarify는 질문 범위가 모호할 때만 선택하세요.\n"
        "여러 문서가 연결된 상태에서 표/그림/페이지/섹션 번호만 있고 문서가 특정되지 않으면 clarify를 우선 고려하세요.\n"
        "이미 추가 검색을 한 번 더 시도했는데도 근거가 비어 있거나 부족하면 retrieve_deeper를 반복하지 말고 clarify를 우선 고려하세요.\n"
        "clarify일 때만 clarification_question을 채우고, 나머지 action에서는 null로 두세요."
    )


def build_stage5_grounding_user_prompt(
    *,
    query_text: str,
    answer_draft: str | None,
    context_blocks: list[str],
    selection_type: str | None,
    available_document_count: int,
    selected_document_count: int,
    deep_retrieval_attempted: bool,
) -> str:
    """grounding check에 사용할 사용자 프롬프트를 구성한다."""
    context_text = "\n\n".join(context_blocks).strip() or "근거 없음"
    normalized_answer_draft = answer_draft.strip() if answer_draft else "(없음)"
    return (
        f"사용자 질문:\n{query_text}\n\n"
        f"선택 유형: {selection_type or '-'}\n"
        f"현재 스레드 문서 수: {available_document_count}\n"
        f"현재 검색 대상 문서 수: {selected_document_count}\n"
        f"추가 검색 재시도 여부: {'yes' if deep_retrieval_attempted else 'no'}\n\n"
        f"현재 답변 초안:\n{normalized_answer_draft}\n\n"
        f"검색 근거:\n{context_text}\n\n"
        "위 정보를 보고 answer, retrieve_deeper, clarify 중 하나를 선택하세요."
    )
