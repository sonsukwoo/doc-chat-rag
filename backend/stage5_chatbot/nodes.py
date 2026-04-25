"""Stage-5 chatbot graph nodes."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately, trim_messages
from langgraph.types import interrupt

from backend.stage4_retrieval import (
    search_thread_knowledge as default_search_thread_knowledge,
)
from backend.thread_identity import build_thread_collection_name

from .config import (
    STAGE5_CONTEXT_WINDOW_SIZE,
    STAGE5_CONTEXT_WINDOW_MAX_HITS,
    STAGE5_DEEP_RETRIEVAL_FETCH_K,
    STAGE5_DEEP_RETRIEVAL_TOP_K,
    STAGE5_DEFAULT_RETRIEVAL_MODE,
    STAGE5_DEFAULT_TOP_K,
    STAGE5_AGENT_MODEL,
    STAGE5_ENABLE_CONTEXT_WINDOW,
    STAGE5_HISTORY_MAX_TOKENS,
    STAGE5_MULTI_DOC_PER_DOCUMENT_TOP_K,
    STAGE5_SUMMARY_MAX_LINES,
)
from .document_selection import (
    extract_explicit_document_ids as _shared_extract_explicit_document_ids,
    extract_numeric_filename_aliases as _shared_extract_numeric_filename_aliases,
    iter_ordered_document_profiles as _shared_iter_ordered_document_profiles,
    normalize_match_text as _shared_normalize_match_text,
    tokenize_match_terms as _shared_tokenize_match_terms,
)
from .models import (
    DocumentSelectionResult,
    FinalAnswerResult,
    GroundingDecisionResult,
    IntentClassificationResult,
    RetrievalTask,
)
from .prompts import (
    build_stage5_agent_system_prompt,
    build_stage5_answer_system_prompt,
    build_stage5_answer_user_prompt,
    build_stage5_document_selection_system_prompt,
    build_stage5_document_selection_user_prompt,
    build_stage5_general_response_system_prompt,
    build_stage5_grounding_system_prompt,
    build_stage5_grounding_user_prompt,
    build_stage5_intent_system_prompt,
    build_stage5_intent_user_prompt,
    build_stage5_memory_system_prompt,
)
from .state import (
    ChatbotState,
    FollowupReferencePayload,
    GroundingDecisionPayload,
    IntentAnalysisPayload,
    QueryAnalysisPayload,
    RetrievalTaskPayload,
    RetrievalPolicyPayload,
)

_normalize_match_text = _shared_normalize_match_text
_tokenize_match_terms = _shared_tokenize_match_terms
_extract_explicit_document_ids = _shared_extract_explicit_document_ids
_extract_numeric_filename_aliases = _shared_extract_numeric_filename_aliases


DOCUMENT_REFERENCE_MARKERS = (
    "문서",
    "논문",
    "자료",
    "pdf",
    "본문",
    "이 글",
    "이 문서",
    "이 논문",
    "이 자료",
    "섹션",
    "페이지",
    "표",
    "그림",
    "chapter",
    "section",
    "page",
    "table",
    "figure",
)
LEXICAL_REFERENCE_MARKERS = ("table", "figure", "section", "appendix", "page", "표", "그림", "섹션", "페이지")
CONVERSATION_MEMORY_MARKERS = (
    "내 이름",
    "제 이름",
    "내 별명",
    "제 별명",
    "내 닉네임",
    "제 닉네임",
    "내가 뭐라고",
    "제가 뭐라고",
    "방금 뭐라고",
    "아까 뭐라고",
    "이전 대화",
    "지난 대화",
    "직전에",
    "방금 한 질문",
    "직전 답변",
    "이전 답변",
)
DOCUMENT_LIST_MARKERS = (
    "어떤 문서",
    "문서 목록",
    "문서 리스트",
    "문서들이",
    "문서가 들어",
    "들어있",
    "연결된 문서",
    "첨부된 문서",
    "몇 개 문서",
    "몇개 문서",
)
THANKS_MARKERS = ("고마워", "감사", "thanks", "thank you")
MULTI_DOCUMENT_SCOPE_MARKERS = (
    "각 문서",
    "각각",
    "비교",
    "차이",
    "둘 다",
    "두 문서",
    "모든 문서",
    "전체 문서",
)
COMPARISON_SCOPE_MARKERS = (
    "비교",
    "차이",
    "공통점",
    "차이점",
    "어떻게 다르",
)
INSUFFICIENT_ANSWER_MARKERS = (
    "찾지 못",
    "없습니다",
    "없음을 알려",
    "명시되어 있지",
    "확인할 수 없",
    "구체적인 설명이나 예제가 포함되어 있지",
)
STRONG_DEICTIC_DOCUMENT_MARKERS = (
    "이 문서",
    "저 문서",
    "해당 문서",
    "이 논문",
    "저 논문",
    "이 자료",
    "저 자료",
    "이거",
)
TECHNICAL_QUERY_MARKERS = (
    "인자",
    "파라미터",
    "함수",
    "메서드",
    "클래스",
    "코드",
    "구현",
    "예시",
    "옵션",
    "설정",
    "agent",
    "middleware",
    "checkpointer",
    "parameter",
    "argument",
    "function",
    "method",
    "class",
)
RETRIEVAL_TASK_TYPES = {
    "fact_lookup",
    "exact_keyword",
    "document_summary",
    "comparison",
    "procedure",
    "figure_table",
    "conversation_memory",
    "general",
}
RETRIEVAL_STRATEGIES = {
    "vector_search",
    "hybrid_search",
    "document_overview",
    "balanced_multi_document",
    "asset_lookup",
    "conversation_only",
    "no_retrieval",
}


def _utc_now_iso() -> str:
    """UI와 직렬화에서 재사용할 메시지 생성 시각을 UTC ISO 형식으로 만든다."""
    return datetime.now(timezone.utc).isoformat()


def _build_thread_chat_metadata(
    *,
    created_at: str | None = None,
    kind: str | None = None,
    citations: list[dict[str, Any]] | None = None,
    evidence_chunks: list[dict[str, Any]] | None = None,
    visual_asset_refs: list[str] | None = None,
    retrieval_mode: str | None = None,
    debug_trace: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """체크포인터에 남길 UI용 메시지 메타데이터를 통일된 형태로 만든다."""
    payload: dict[str, Any] = {
        "created_at": created_at or _utc_now_iso(),
    }
    if kind is not None:
        payload["kind"] = kind
    if citations is not None:
        payload["citations"] = citations
    if evidence_chunks is not None:
        payload["evidence_chunks"] = evidence_chunks
    if visual_asset_refs is not None:
        payload["visual_asset_refs"] = visual_asset_refs
    if retrieval_mode is not None:
        payload["retrieval_mode"] = retrieval_mode
    if debug_trace is not None:
        payload["debug_trace"] = debug_trace
    return {"thread_chat": payload}


def _format_interrupt_content(payload: dict[str, Any]) -> str:
    parts = [
        str(payload.get("question") or "").strip(),
        str(payload.get("reason") or "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def _build_interrupt_history_message(payload: dict[str, Any]) -> AIMessage | None:
    interrupt_content = _format_interrupt_content(payload)
    if not interrupt_content:
        return None
    return AIMessage(
        content=interrupt_content,
        name="stage5_clarification",
        additional_kwargs=_build_thread_chat_metadata(kind="interrupt"),
    )


def _has_matching_interrupt_history(
    messages: list[Any],
    payload: dict[str, Any],
) -> bool:
    interrupt_content = _format_interrupt_content(payload)
    if not interrupt_content:
        return False
    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            break
        if not isinstance(message, AIMessage):
            continue
        metadata = dict(getattr(message, "additional_kwargs", {}) or {}).get(
            "thread_chat"
        )
        if not isinstance(metadata, dict):
            break
        if str(metadata.get("kind") or "").strip() != "interrupt":
            break
        return str(message.content or "").strip() == interrupt_content
    return False
FOLLOW_UP_DOCUMENT_MARKERS = (
    "그중",
    "그럼",
    "그거",
    "그건",
    "그 기능",
    "그 인자",
    "그 파라미터",
    "그 부분",
    "이어서",
    "그러면",
)
FOLLOW_UP_REFERENCE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("figure", re.compile(r"\bfigure\s*([a-z0-9][a-z0-9._-]*)", re.IGNORECASE)),
    ("table", re.compile(r"\btable\s*([a-z0-9][a-z0-9._-]*)", re.IGNORECASE)),
    ("section", re.compile(r"\bsection\s*([a-z0-9][a-z0-9._-]*)", re.IGNORECASE)),
    ("figure", re.compile(r"그림\s*([0-9A-Za-z가-힣._-]+)")),
    ("table", re.compile(r"표\s*([0-9A-Za-z가-힣._-]+)")),
    ("section", re.compile(r"섹션\s*([0-9A-Za-z가-힣._-]+)")),
)
GENERIC_FOLLOW_UP_REFERENCE_MARKERS: dict[str, tuple[str, ...]] = {
    "figure": ("그 그림", "이 그림", "해당 그림", "그 figure", "이 figure"),
    "table": ("그 표", "이 표", "해당 표", "그 table", "이 table"),
    "section": ("그 섹션", "이 섹션", "해당 섹션", "그 section", "이 section"),
}


def _get_latest_user_text(state: ChatbotState) -> str:
    resumed_query = str(state.get("user_message") or "").strip()
    clarification_response = str(state.get("clarification_response") or "").strip()
    if resumed_query and clarification_response:
        return resumed_query
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage) and isinstance(message.content, str):
            return message.content.strip()
    return resumed_query


def _get_latest_ai_message(state: ChatbotState) -> AIMessage | None:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _get_current_turn_messages(state: ChatbotState) -> list[Any]:
    messages = list(state.get("messages") or [])
    last_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_human_index = index
            break
    if last_human_index < 0:
        return messages
    return messages[last_human_index:]


def _iter_tool_messages(
    state: ChatbotState,
    *,
    tool_name: str | None = None,
    current_turn_only: bool = False,
) -> list[ToolMessage]:
    tool_messages: list[ToolMessage] = []
    source_messages = (
        _get_current_turn_messages(state)
        if current_turn_only
        else list(state.get("messages") or [])
    )
    for message in source_messages:
        if not isinstance(message, ToolMessage):
            continue
        if tool_name and getattr(message, "name", None) != tool_name:
            continue
        tool_messages.append(message)
    return tool_messages


def _parse_tool_message_json(message: ToolMessage) -> dict[str, Any] | None:
    if not isinstance(message.content, str):
        return None
    try:
        payload = json.loads(message.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _normalize_string_list(values: Any) -> list[str]:
    return [str(item).strip() for item in values or [] if str(item).strip()]


def _build_recent_dialog_lines(
    state: ChatbotState,
    *,
    limit: int = 6,
) -> list[str]:
    dialog_messages = [
        message
        for message in state.get("messages") or []
        if isinstance(message, (HumanMessage, AIMessage))
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    if not dialog_messages:
        return []

    lines: list[str] = []
    for message in dialog_messages[-limit:]:
        role = "사용자" if isinstance(message, HumanMessage) else "assistant"
        lines.append(f"{role}: {_truncate_text(message.content, limit=96)}")
    return lines


def _normalize_optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return None


def _normalize_optional_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


def _get_tool_label(tool_name: str) -> str:
    labels = {
        "search_thread_knowledge": "문서 검색",
        "expand_context_window": "문맥 확장",
        "load_visual_asset": "시각 자료 로드",
        "list_thread_documents": "문서 목록 조회",
        "web_search": "웹 검색",
    }
    return labels.get(tool_name, tool_name)


def _build_tool_trace_payload(message: ToolMessage) -> dict[str, Any]:
    tool_name = str(getattr(message, "name", None) or "").strip() or "tool"
    payload = _parse_tool_message_json(message) or {}
    trace: dict[str, Any] = {
        "name": tool_name,
        "label": _get_tool_label(tool_name),
        "status": str(payload.get("status") or "").strip() or None,
    }

    if tool_name == "search_thread_knowledge":
        trace["query"] = str(payload.get("query") or "").strip() or None
        trace["document_ids"] = _normalize_string_list(
            payload.get("active_document_ids") or payload.get("document_ids")
        )
        trace["retrieval_task_count"] = _normalize_optional_int(
            payload.get("retrieval_task_count")
        )
        trace["retrieved_count"] = _normalize_optional_int(payload.get("retrieved_count"))
        trace["retrieval_mode"] = str(payload.get("retrieval_mode") or "").strip() or None
        trace["rerank_requested"] = (
            bool(payload.get("rerank_requested"))
            if "rerank_requested" in payload
            else None
        )
        trace["rerank_applied"] = (
            bool(payload.get("rerank_applied"))
            if "rerank_applied" in payload
            else None
        )
        trace["rerank_error"] = str(payload.get("rerank_error") or "").strip() or None
        trace["mmr_requested"] = (
            bool(payload.get("mmr_requested"))
            if "mmr_requested" in payload
            else None
        )
        trace["mmr_applied"] = (
            bool(payload.get("mmr_applied"))
            if "mmr_applied" in payload
            else None
        )
        trace["task_search_used"] = (
            bool(payload.get("task_search_used"))
            if "task_search_used" in payload
            else None
        )
        trace["overview_search_used"] = (
            bool(payload.get("overview_search_used"))
            if "overview_search_used" in payload
            else None
        )
        trace["per_document_search_used"] = (
            bool(payload.get("per_document_search_used"))
            if "per_document_search_used" in payload
            else None
        )
        trace["score_threshold_applied"] = _normalize_optional_float(
            payload.get("score_threshold_applied")
        )
        trace["score_fallback_applied"] = (
            bool(payload.get("score_fallback_applied"))
            if "score_fallback_applied" in payload
            else None
        )
        trace["top_k"] = _normalize_optional_int(payload.get("top_k"))
        trace["fetch_k"] = _normalize_optional_int(payload.get("fetch_k"))
        summary_parts: list[str] = []
        if trace.get("overview_search_used"):
            summary_parts.append("문서 대표 블록을 가져오는 개요 경로를 사용했습니다.")
        if trace.get("task_search_used"):
            summary_parts.append("질문 task별로 독립 검색 후 병합했습니다.")
        if trace.get("per_document_search_used"):
            summary_parts.append("문서별로 독립 검색 후 병합했습니다.")
        if trace.get("rerank_applied"):
            summary_parts.append("리랭크를 적용했습니다.")
        elif trace.get("rerank_requested"):
            rerank_error = str(trace.get("rerank_error") or "").strip()
            if rerank_error:
                summary_parts.append(
                    f"리랭크가 요청됐지만 적용되지 않았습니다 ({rerank_error})."
                )
            else:
                summary_parts.append("리랭크가 요청됐지만 이번 결과에는 적용되지 않았습니다.")
        if trace.get("mmr_applied"):
            summary_parts.append("MMR을 적용했습니다.")
        if trace.get("score_fallback_applied"):
            summary_parts.append("score threshold fallback이 발생했습니다.")
        trace["message"] = " ".join(summary_parts) or None
        return trace

    if tool_name == "expand_context_window":
        trace["chunk_ids"] = _normalize_string_list(payload.get("chunk_ids"))
        blocks = payload.get("blocks") or []
        trace["block_count"] = len(blocks) if isinstance(blocks, list) else None
        return trace

    if tool_name == "load_visual_asset":
        trace["asset_ref"] = str(payload.get("asset_ref") or "").strip() or None
        asset = payload.get("asset")
        if isinstance(asset, dict):
            trace["document_ids"] = _normalize_string_list([asset.get("document_id")])
        return trace

    if tool_name == "list_thread_documents":
        trace["document_ids"] = _normalize_string_list(payload.get("document_ids"))
        return trace

    if tool_name == "web_search":
        trace["query"] = str(payload.get("query") or "").strip() or None
        trace["message"] = str(payload.get("message") or "").strip() or None
        return trace

    if isinstance(payload, dict):
        trace["message"] = str(payload.get("message") or "").strip() or None
    return trace


def _build_tool_traces(state: ChatbotState) -> list[dict[str, Any]]:
    traces: list[dict[str, Any]] = []
    for message in _iter_tool_messages(state, current_turn_only=True):
        traces.append(_build_tool_trace_payload(message))
    return traces


def _build_debug_trace(
    state: ChatbotState,
    *,
    final_log_entries: list[str] | None = None,
) -> dict[str, Any]:
    query_analysis = dict(state.get("query_analysis") or {})
    retrieval_policy = dict(state.get("retrieval_policy") or {})
    latest_search_payload = _extract_latest_search_payload(state) or {}
    log_cursor = int(state.get("log_cursor") or 0)
    current_logs = [
        str(item).strip()
        for item in list(state.get("logs") or [])[log_cursor:]
        if str(item).strip()
    ]
    for entry in final_log_entries or []:
        normalized_entry = str(entry).strip()
        if normalized_entry:
            current_logs.append(normalized_entry)

    return {
        "model": STAGE5_AGENT_MODEL,
        "query_kind": str(query_analysis.get("query_kind") or "").strip(),
        "selection_type": str(query_analysis.get("selection_type") or "").strip(),
        "selection_source": str(query_analysis.get("selection_source") or "").strip(),
        "answer_strategy": str(query_analysis.get("answer_strategy") or "").strip(),
        "selection_reason": str(query_analysis.get("reason") or "").strip(),
        "selected_document_ids": _normalize_string_list(
            query_analysis.get("selected_document_ids")
        ),
        "selected_document_queries": {
            str(document_id).strip(): str(query_text).strip()
            for document_id, query_text in dict(
                query_analysis.get("selected_document_queries") or {}
            ).items()
            if str(document_id).strip() and str(query_text).strip()
        },
        "retrieval_tasks": _normalize_retrieval_tasks(
            fallback_document_ids=_normalize_string_list(
                query_analysis.get("selected_document_ids")
            ),
            retrieval_tasks=list(query_analysis.get("retrieval_tasks") or []),
        ),
        "thread_default_retrieval_mode": (
            str(state.get("thread_default_retrieval_mode") or "").strip() or None
        ),
        "retrieval_mode": str(retrieval_policy.get("mode") or "").strip() or None,
        "executed_retrieval_mode": (
            str(latest_search_payload.get("retrieval_mode") or "").strip() or None
        ),
        "logs": current_logs,
        "tool_calls": _build_tool_traces(state),
    }


def _extract_latest_search_payload(state: ChatbotState) -> dict[str, Any] | None:
    return _merge_current_search_payloads(state)


def _extract_current_search_payloads(state: ChatbotState) -> list[dict[str, Any]]:
    payloads: list[dict[str, Any]] = []
    for message in reversed(
        _iter_tool_messages(
            state,
            tool_name="search_thread_knowledge",
            current_turn_only=True,
        )
    ):
        payload = _parse_tool_message_json(message)
        if payload is not None:
            payloads.append(payload)
    payloads.reverse()
    return payloads


def _merge_current_search_payloads(state: ChatbotState) -> dict[str, Any] | None:
    payloads = _extract_current_search_payloads(state)
    if not payloads:
        return None
    merged = dict(payloads[-1])
    merged_hits: list[dict[str, Any]] = []
    seen_hit_keys: set[tuple[str, str]] = set()
    for payload in payloads:
        for hit in list(payload.get("retrievals") or []):
            if not isinstance(hit, dict):
                continue
            document_id = str(hit.get("document_id") or "").strip()
            chunk_id = str(hit.get("chunk_id") or "").strip()
            if not document_id or not chunk_id:
                continue
            key = (document_id, chunk_id)
            if key in seen_hit_keys:
                continue
            seen_hit_keys.add(key)
            merged_hits.append(dict(hit))

    active_document_ids: list[str] = []
    seen_document_ids: set[str] = set()
    for payload in payloads:
        for document_id in payload.get("active_document_ids") or payload.get("document_ids") or []:
            normalized_id = str(document_id).strip()
            if not normalized_id or normalized_id in seen_document_ids:
                continue
            seen_document_ids.add(normalized_id)
            active_document_ids.append(normalized_id)

    queries: list[str] = []
    seen_queries: set[str] = set()
    for payload in payloads:
        query = str(payload.get("query") or "").strip()
        if not query or query in seen_queries:
            continue
        seen_queries.add(query)
        queries.append(query)

    merged["query"] = " | ".join(queries) if queries else merged.get("query")
    merged["active_document_ids"] = active_document_ids
    merged["retrievals"] = merged_hits
    merged["retrieved_count"] = len(merged_hits)
    merged["retrieval_task_count"] = max(
        [
            int(payload.get("retrieval_task_count") or 0)
            for payload in payloads
        ]
        or [0]
    )
    merged["task_search_used"] = any(bool(payload.get("task_search_used")) for payload in payloads)
    merged["overview_search_used"] = any(
        bool(payload.get("overview_search_used")) for payload in payloads
    )
    merged["per_document_search_used"] = any(
        bool(payload.get("per_document_search_used")) for payload in payloads
    )
    merged["rerank_applied"] = any(bool(payload.get("rerank_applied")) for payload in payloads)
    merged["mmr_applied"] = any(bool(payload.get("mmr_applied")) for payload in payloads)
    return merged


def _extract_previous_search_payload(state: ChatbotState) -> dict[str, Any] | None:
    for message in reversed(
        _iter_tool_messages(
            state,
            tool_name="search_thread_knowledge",
            current_turn_only=False,
        )
    ):
        payload = _parse_tool_message_json(message)
        if payload is not None:
            return payload
    return None


def _normalize_followup_references(
    references: list[FollowupReferencePayload | dict[str, Any]] | None,
) -> list[FollowupReferencePayload]:
    normalized: list[FollowupReferencePayload] = []
    seen_keys: set[tuple[str, str, str, str]] = set()
    for reference in references or []:
        if not isinstance(reference, dict):
            continue
        kind = str(reference.get("kind") or "").strip().lower()
        document_id = str(reference.get("document_id") or "").strip()
        chunk_id = str(reference.get("chunk_id") or "").strip()
        label = str(reference.get("label") or "").strip()
        if kind not in {"figure", "table", "section"} or not document_id:
            continue
        normalized_label = _normalize_match_text(label)
        dedupe_key = (kind, normalized_label, document_id, chunk_id)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        page = reference.get("page")
        normalized.append(
            {
                "kind": kind,  # type: ignore[typeddict-item]
                "label": label,
                "document_id": document_id,
                "chunk_id": chunk_id or None,
                "page": int(page) if isinstance(page, (int, float)) else None,
                "asset_ref": str(reference.get("asset_ref") or "").strip() or None,
                "section_title": (
                    str(reference.get("section_title") or "").strip() or None
                ),
            }
        )
    return normalized


def _extract_reference_entities_from_hits(
    retrieval_hits: list[dict[str, Any]],
) -> list[FollowupReferencePayload]:
    references: list[FollowupReferencePayload] = []
    for hit in retrieval_hits:
        document_id = str(hit.get("document_id") or "").strip()
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not document_id:
            continue
        asset_ref = (
            f"{document_id}:{chunk_id}"
            if document_id and chunk_id and hit.get("asset_relative_path")
            else None
        )
        page = hit.get("primary_page")
        source_texts = [
            str(hit.get("section_title") or "").strip(),
            str(hit.get("caption") or "").strip(),
        ]
        for source_text in source_texts:
            if not source_text:
                continue
            for kind, pattern in FOLLOW_UP_REFERENCE_PATTERNS:
                match = pattern.search(source_text)
                if match is None:
                    continue
                references.append(
                    {
                        "kind": kind,  # type: ignore[typeddict-item]
                        "label": match.group(1).strip(),
                        "document_id": document_id,
                        "chunk_id": chunk_id or None,
                        "page": int(page) if isinstance(page, (int, float)) else None,
                        "asset_ref": asset_ref,
                        "section_title": source_text,
                    }
                )
                break
    return _normalize_followup_references(references)


def _extract_requested_followup_targets(
    query_text: str,
) -> list[tuple[str, str | None]]:
    targets: list[tuple[str, str | None]] = []
    for kind, pattern in FOLLOW_UP_REFERENCE_PATTERNS:
        for match in pattern.finditer(query_text):
            label = _normalize_match_text(match.group(1))
            if label:
                targets.append((kind, label))
    lowered = _normalize_match_text(query_text)
    for kind, markers in GENERIC_FOLLOW_UP_REFERENCE_MARKERS.items():
        if any(_normalize_match_text(marker) in lowered for marker in markers):
            targets.append((kind, None))
    deduped: list[tuple[str, str | None]] = []
    for item in targets:
        if item not in deduped:
            deduped.append(item)
    return deduped


def _resolve_structured_followup_document_scope(
    state: ChatbotState,
    query_text: str,
    *,
    force: bool,
) -> tuple[list[str], str | None]:
    last_document_ids = _normalize_string_list(state.get("last_resolved_document_ids"))
    last_references = _normalize_followup_references(
        list(state.get("last_referenced_entities") or [])
    )
    lowered = _normalize_match_text(query_text)
    requested_targets = _extract_requested_followup_targets(query_text)

    for kind, label in requested_targets:
        matched_document_ids: list[str] = []
        for reference in last_references:
            if str(reference.get("kind") or "").strip() != kind:
                continue
            normalized_label = _normalize_match_text(str(reference.get("label") or ""))
            if label is not None and normalized_label != label:
                continue
            document_id = str(reference.get("document_id") or "").strip()
            if document_id and document_id not in matched_document_ids:
                matched_document_ids.append(document_id)
        if matched_document_ids:
            return (
                matched_document_ids,
                "이전 대화에서 참조한 표/그림/섹션 기준 후속 질문입니다.",
            )

    if not (
        force
        or any(marker in lowered for marker in FOLLOW_UP_DOCUMENT_MARKERS)
        or any(marker in lowered for marker in STRONG_DEICTIC_DOCUMENT_MARKERS)
        or requested_targets
    ):
        return [], None

    if last_document_ids:
        return last_document_ids, "직전 retrieval task 문서 범위를 이어받은 후속 질문입니다."
    return [], None


def _build_followup_state_updates(
    state: ChatbotState,
    *,
    retrieval_hits: list[dict[str, Any]] | None = None,
    retrieval_document_ids: list[str] | None = None,
    retrieval_tasks: list[RetrievalTaskPayload] | None = None,
    visual_asset_refs: list[str] | None = None,
) -> dict[str, Any]:
    resolved_document_ids = _normalize_string_list(
        retrieval_document_ids or _get_retrieval_document_ids(state)
    )
    normalized_tasks = _normalize_retrieval_tasks(
        fallback_document_ids=resolved_document_ids,
        retrieval_tasks=(
            retrieval_tasks
            if retrieval_tasks is not None
            else _get_retrieval_tasks(state)
        ),
    )
    normalized_visual_asset_refs = _normalize_string_list(
        visual_asset_refs
        if visual_asset_refs is not None
        else [
            f"{str(hit.get('document_id') or '').strip()}:{str(hit.get('chunk_id') or '').strip()}"
            for hit in retrieval_hits or []
            if str(hit.get("asset_relative_path") or "").strip()
            and str(hit.get("document_id") or "").strip()
            and str(hit.get("chunk_id") or "").strip()
        ]
    )
    normalized_references = _extract_reference_entities_from_hits(
        list(retrieval_hits or [])
    )
    return {
        "last_resolved_document_ids": resolved_document_ids,
        "last_retrieval_tasks": normalized_tasks,
        "last_referenced_entities": normalized_references,
        "last_visual_asset_refs": normalized_visual_asset_refs,
    }


def _extract_latest_context_window_blocks(
    state: ChatbotState,
) -> list[dict[str, Any]] | None:
    for message in reversed(
        _iter_tool_messages(
            state,
            tool_name="expand_context_window",
            current_turn_only=True,
        )
    ):
        payload = _parse_tool_message_json(message)
        if payload is None:
            continue
        blocks = payload.get("blocks")
        if isinstance(blocks, list):
            return [dict(item) for item in blocks if isinstance(item, dict)]
    return None


def _truncate_text(value: str, limit: int = 240) -> str:
    normalized = " ".join(str(value or "").split()).strip()
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1].rstrip()}…"


def _build_document_profile_lookup(
    state: ChatbotState,
) -> dict[str, dict[str, object]]:
    return {
        str(profile.get("document_id") or "").strip(): dict(profile)
        for profile in _iter_ordered_document_profiles(state)
        if str(profile.get("document_id") or "").strip()
    }


def _append_document_profile_lines(
    lines: list[str],
    *,
    document_id: str,
    profile_lookup: dict[str, dict[str, object]] | None,
) -> None:
    if not document_id or not profile_lookup:
        return

    profile = dict(profile_lookup.get(document_id) or {})
    if not profile:
        return

    original_filename = str(profile.get("original_filename") or "").strip()
    title = str(profile.get("title") or "").strip()
    document_order = profile.get("document_order")
    filename_aliases = _extract_numeric_filename_aliases(original_filename)

    if original_filename:
        lines.append(f"filename: {original_filename}")
    if title:
        lines.append(f"title: {title}")
    if isinstance(document_order, int):
        lines.append(f"document_order: {document_order}")
    if filename_aliases:
        lines.append(f"filename_aliases: {', '.join(filename_aliases)}")


def _build_context_blocks(
    retrieval_hits: list[dict[str, Any]],
    *,
    profile_lookup: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    context_blocks: list[str] = []
    for index, hit in enumerate(retrieval_hits, start=1):
        document_id = str(hit.get("document_id") or "")
        chunk_id = str(hit.get("chunk_id") or "")
        lines = [
            f"[근거 {index}]",
            f"document_id: {document_id}",
            f"chunk_id: {chunk_id}",
        ]
        _append_document_profile_lines(
            lines,
            document_id=document_id,
            profile_lookup=profile_lookup,
        )
        if hit.get("section_title"):
            lines.append(f"section_title: {str(hit['section_title'])}")
        if hit.get("primary_page") is not None:
            lines.append(f"page: {int(hit['primary_page'])}")
        if hit.get("asset_relative_path"):
            lines.append(f"asset_ref: {document_id}:{chunk_id}")
        if hit.get("caption"):
            lines.append(f"caption: {str(hit['caption'])}")
        lines.append(f"text: {str(hit.get('text') or '').strip()}")
        context_blocks.append("\n".join(lines))
    return context_blocks


def _render_expanded_context_blocks(
    blocks: list[dict[str, Any]],
    *,
    profile_lookup: dict[str, dict[str, object]] | None = None,
) -> list[str]:
    rendered: list[str] = []
    for index, block in enumerate(blocks, start=1):
        document_id = str(block.get("document_id") or "").strip()
        parent_id = str(block.get("parent_id") or "").strip()
        section_title = str(block.get("section_title") or "").strip()
        matched_chunk_ids = [
            str(item).strip()
            for item in block.get("matched_chunk_ids") or []
            if str(item).strip()
        ]
        window_chunk_ids = [
            str(item).strip()
            for item in block.get("window_chunk_ids") or []
            if str(item).strip()
        ]
        lines = [
            f"[확장 근거 {index}]",
            f"document_id: {document_id}",
            f"parent_id: {parent_id}",
        ]
        _append_document_profile_lines(
            lines,
            document_id=document_id,
            profile_lookup=profile_lookup,
        )
        if section_title:
            lines.append(f"section_title: {section_title}")
        if block.get("page_start") is not None:
            if block.get("page_end") is not None and block.get("page_end") != block.get("page_start"):
                lines.append(
                    f"pages: {block.get('page_start')}-{block.get('page_end')}"
                )
            else:
                lines.append(f"page: {block.get('page_start')}")
        if matched_chunk_ids:
            lines.append(f"matched_chunk_ids: {', '.join(matched_chunk_ids)}")
        if window_chunk_ids:
            lines.append(f"context_chunk_ids: {', '.join(window_chunk_ids)}")
        lines.append(f"text: {str(block.get('context_text') or '').strip()}")
        rendered.append("\n".join(lines))
    return rendered


def _build_citations(retrieval_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    seen_keys: set[tuple[str, str, int | None, str | None, str | None]] = set()
    for hit in retrieval_hits:
        document_id = str(hit.get("document_id") or "")
        chunk_id = str(hit.get("chunk_id") or "")
        citation = {
            "document_id": document_id,
            "chunk_id": chunk_id,
            "parent_id": hit.get("parent_id"),
            "page": hit.get("primary_page"),
            "section_title": hit.get("section_title"),
            "asset_ref": (
                f"{document_id}:{chunk_id}"
                if hit.get("asset_relative_path")
                else None
            ),
            "asset_relative_path": hit.get("asset_relative_path"),
        }
        dedupe_key = (
            document_id,
            chunk_id,
            citation.get("page"),
            str(citation.get("section_title") or "") or None,
            str(citation.get("asset_ref") or "") or None,
        )
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        citations.append(citation)
    return citations


def _build_evidence_chunks(
    retrieval_hits: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    evidence_chunks: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    for hit in retrieval_hits:
        document_id = str(hit.get("document_id") or "").strip()
        chunk_id = str(hit.get("chunk_id") or "").strip()
        if not document_id or not chunk_id:
            continue
        dedupe_key = (document_id, chunk_id)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        evidence_chunks.append(
            {
                "document_id": document_id,
                "chunk_id": chunk_id,
                "parent_id": hit.get("parent_id"),
                "page": hit.get("primary_page"),
                "section_title": hit.get("section_title"),
                "chunk_type": hit.get("chunk_type"),
                "text_excerpt": _truncate_text(str(hit.get("text") or "")),
            }
        )
    return evidence_chunks


def _select_inline_visual_asset_refs(
    retrieval_hits: list[dict[str, Any]],
    *,
    candidate_hit_limit: int = 3,
    asset_limit: int = 1,
) -> list[str]:
    asset_refs: list[str] = []
    for hit in retrieval_hits[: max(1, candidate_hit_limit)]:
        document_id = str(hit.get("document_id") or "").strip()
        chunk_id = str(hit.get("chunk_id") or "").strip()
        asset_relative_path = str(hit.get("asset_relative_path") or "").strip()
        if not document_id or not chunk_id or not asset_relative_path:
            continue
        asset_ref = f"{document_id}:{chunk_id}"
        if asset_ref in asset_refs:
            continue
        asset_refs.append(asset_ref)
        if len(asset_refs) >= asset_limit:
            break
    return asset_refs


def _build_interrupt_metadata(
    state: ChatbotState,
    *,
    retrieval_hits: list[dict[str, Any]],
    final_log_entry: str,
) -> dict[str, Any]:
    return {
        "citations": _build_citations(retrieval_hits),
        "evidence_chunks": _build_evidence_chunks(retrieval_hits),
        "visual_asset_refs": _select_inline_visual_asset_refs(retrieval_hits),
        "debug_trace": _build_debug_trace(
            state,
            final_log_entries=[final_log_entry],
        ),
    }


def _get_active_document_ids(state: ChatbotState) -> list[str]:
    return [
        str(item).strip()
        for item in state.get("active_document_ids") or []
        if str(item).strip()
    ]


def _get_retrieval_document_ids(state: ChatbotState) -> list[str]:
    selected_document_ids = [
        str(item).strip()
        for item in state.get("retrieval_document_ids") or []
        if str(item).strip()
    ]
    if selected_document_ids:
        return selected_document_ids
    return _get_active_document_ids(state)


def _normalize_retrieval_tasks(
    *,
    fallback_document_ids: list[str],
    retrieval_tasks: list[RetrievalTask | RetrievalTaskPayload | dict[str, Any]] | None,
) -> list[RetrievalTaskPayload]:
    normalized_fallback_document_ids = [
        str(document_id).strip()
        for document_id in fallback_document_ids
        if str(document_id).strip()
    ]
    allowed_document_ids = set(normalized_fallback_document_ids)
    normalized_tasks: list[RetrievalTaskPayload] = []
    seen_signatures: set[tuple[str, tuple[str, ...]]] = set()

    for index, task in enumerate(retrieval_tasks or [], start=1):
        if isinstance(task, RetrievalTask):
            raw_task = task.model_dump()
        else:
            raw_task = dict(task or {})
        subquery = str(
            raw_task.get("search_query")
            or raw_task.get("subquery")
            or raw_task.get("user_question")
            or ""
        ).strip()
        if not subquery:
            continue
        document_ids = [
            str(document_id).strip()
            for document_id in (
                raw_task.get("document_ids")
                or raw_task.get("target_document_ids")
                or []
            )
            if str(document_id).strip()
            and (
                not allowed_document_ids
                or str(document_id).strip() in allowed_document_ids
            )
        ]
        if not document_ids:
            document_ids = list(normalized_fallback_document_ids)
        if not document_ids:
            continue
        task_type = str(raw_task.get("task_type") or "").strip()
        if task_type not in RETRIEVAL_TASK_TYPES:
            task_type = ""
        retrieval_strategy = str(raw_task.get("retrieval_strategy") or "").strip()
        if retrieval_strategy not in RETRIEVAL_STRATEGIES:
            retrieval_strategy = ""
        signature = (subquery, tuple(document_ids), task_type, retrieval_strategy)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        task_id = str(raw_task.get("task_id") or "").strip() or f"task-{index}"
        normalized_task: RetrievalTaskPayload = {
            "task_id": task_id,
            "subquery": subquery,
            "document_ids": document_ids,
        }
        user_question = str(raw_task.get("user_question") or "").strip()
        search_query = str(raw_task.get("search_query") or "").strip()
        if user_question:
            normalized_task["user_question"] = user_question
        if search_query:
            normalized_task["search_query"] = search_query
        if task_type:
            normalized_task["task_type"] = task_type  # type: ignore[typeddict-item]
        if retrieval_strategy:
            normalized_task["retrieval_strategy"] = retrieval_strategy  # type: ignore[typeddict-item]
        normalized_tasks.append(normalized_task)

    return normalized_tasks


def _build_default_retrieval_tasks(
    *,
    query_text: str,
    selection_type: str,
    selected_document_ids: list[str],
) -> list[RetrievalTaskPayload]:
    normalized_document_ids = [
        str(document_id).strip()
        for document_id in selected_document_ids
        if str(document_id).strip()
    ]
    normalized_query_text = str(query_text or "").strip()
    if not normalized_query_text or not normalized_document_ids:
        return []
    if selection_type in {"multi_document", "comparison", "thread_wide"} and len(
        normalized_document_ids
    ) > 1:
        return [
            {
                "task_id": f"task-{index}",
                "subquery": normalized_query_text,
                "document_ids": [document_id],
            }
            for index, document_id in enumerate(normalized_document_ids, start=1)
        ]
    return [
        {
            "task_id": "task-1",
            "subquery": normalized_query_text,
            "document_ids": normalized_document_ids,
        }
    ]


def _build_raw_query_tasks_per_document(
    *,
    query_text: str,
    document_ids: list[str],
) -> list[RetrievalTaskPayload]:
    """문서 미지정 광역 검색에서 질의 재작성 없이 문서별 균형 후보를 모은다."""
    normalized_query_text = str(query_text or "").strip()
    if not normalized_query_text:
        return []
    normalized_document_ids = [
        str(document_id).strip()
        for document_id in document_ids
        if str(document_id).strip()
    ]
    return [
        {
            "task_id": f"task-{index}",
            "subquery": normalized_query_text,
            "document_ids": [document_id],
        }
        for index, document_id in enumerate(normalized_document_ids, start=1)
    ]


def _should_use_balanced_raw_document_search(
    *,
    answer_strategy: str,
    explicit_document_ids: list[str],
    active_document_ids: list[str],
    selection_type: str,
    selected_document_ids: list[str],
    retrieval_tasks: list[RetrievalTaskPayload],
) -> bool:
    """문서 미지정 다중 질문은 프로파일 추정 대신 문서별 균형 검색으로 보낸다."""
    if answer_strategy != "retrieve_chunks":
        return False
    if retrieval_tasks and all(
        _is_document_overview_task(task) for task in retrieval_tasks
    ):
        return False
    if explicit_document_ids:
        return False
    if len(active_document_ids) <= 1:
        return False
    if selection_type in {"comparison", "multi_document", "thread_wide"}:
        return True
    if len(retrieval_tasks) > 1:
        return True
    return not (selection_type == "single_document" and len(selected_document_ids) == 1)


def _build_tasks_from_document_queries(
    *,
    fallback_document_ids: list[str],
    document_queries: dict[str, str] | None,
) -> list[RetrievalTaskPayload]:
    normalized_fallback_document_ids = {
        str(document_id).strip()
        for document_id in fallback_document_ids
        if str(document_id).strip()
    }
    tasks: list[RetrievalTaskPayload] = []
    for index, (document_id, query_text) in enumerate(
        dict(document_queries or {}).items(),
        start=1,
    ):
        normalized_document_id = str(document_id).strip()
        normalized_query = str(query_text).strip()
        if (
            not normalized_document_id
            or not normalized_query
            or (
                normalized_fallback_document_ids
                and normalized_document_id not in normalized_fallback_document_ids
            )
        ):
            continue
        tasks.append(
            {
                "task_id": f"task-{index}",
                "subquery": normalized_query,
                "document_ids": [normalized_document_id],
            }
        )
    return tasks


def _build_document_query_map_from_tasks(
    retrieval_tasks: list[RetrievalTaskPayload] | None,
) -> dict[str, str]:
    document_queries: dict[str, str] = {}
    for task in retrieval_tasks or []:
        subquery = str(task.get("subquery") or "").strip()
        document_ids = [
            str(document_id).strip()
            for document_id in task.get("document_ids") or []
            if str(document_id).strip()
        ]
        if not subquery or len(document_ids) != 1:
            continue
        document_id = document_ids[0]
        if document_id not in document_queries:
            document_queries[document_id] = subquery
    return document_queries


def _is_document_overview_task(task: RetrievalTaskPayload) -> bool:
    task_type = str(task.get("task_type") or "").strip()
    retrieval_strategy = str(task.get("retrieval_strategy") or "").strip()
    return task_type == "document_summary" or retrieval_strategy == "document_overview"


def _split_overview_and_search_tasks(
    retrieval_tasks: list[RetrievalTaskPayload],
) -> tuple[list[RetrievalTaskPayload], list[RetrievalTaskPayload]]:
    overview_tasks: list[RetrievalTaskPayload] = []
    search_tasks: list[RetrievalTaskPayload] = []
    for task in retrieval_tasks:
        if _is_document_overview_task(task):
            overview_tasks.append(task)
        else:
            search_tasks.append(task)
    return overview_tasks, search_tasks


def _collect_document_ids_from_tasks(
    retrieval_tasks: list[RetrievalTaskPayload] | None,
) -> list[str]:
    ordered_ids: list[str] = []
    seen_ids: set[str] = set()
    for task in retrieval_tasks or []:
        for document_id in task.get("document_ids") or []:
            normalized_document_id = str(document_id).strip()
            if not normalized_document_id or normalized_document_id in seen_ids:
                continue
            seen_ids.add(normalized_document_id)
            ordered_ids.append(normalized_document_id)
    return ordered_ids


def _get_retrieval_tasks(state: ChatbotState) -> list[RetrievalTaskPayload]:
    fallback_document_ids = _get_retrieval_document_ids(state)
    return _normalize_retrieval_tasks(
        fallback_document_ids=fallback_document_ids,
        retrieval_tasks=list(state.get("retrieval_tasks") or []),
    )


def _get_retrieval_document_queries(state: ChatbotState) -> dict[str, str]:
    explicit_queries = {
        str(document_id).strip(): str(query_text).strip()
        for document_id, query_text in dict(
            state.get("retrieval_document_queries") or {}
        ).items()
        if str(document_id).strip() and str(query_text).strip()
    }
    if explicit_queries:
        return explicit_queries
    return _build_document_query_map_from_tasks(_get_retrieval_tasks(state))


def _iter_ordered_document_profiles(
    state: ChatbotState,
) -> list[dict[str, object]]:
    return _shared_iter_ordered_document_profiles(
        _get_active_document_ids(state),
        state.get("document_profiles") or [],
    )


def _score_document_profile(
    query_text: str,
    profile: dict[str, object],
) -> tuple[float, list[str]]:
    normalized_query = _normalize_match_text(query_text)
    query_terms = _tokenize_match_terms(query_text)
    if not query_terms:
        return 0.0, []

    matched_topics: list[str] = []
    profile_terms: set[str] = set()
    candidates = [
        str(profile.get("title") or "").strip(),
        str(profile.get("document_type") or "").strip(),
        str(profile.get("short_summary") or "").strip(),
        str(profile.get("original_filename") or "").strip(),
        *[
            str(item).strip()
            for item in profile.get("main_topics") or []
            if str(item).strip()
        ],
        *[
            str(item).strip()
            for item in profile.get("keywords") or []
            if str(item).strip()
        ],
        *[
            str(item).strip()
            for item in profile.get("section_titles") or []
            if str(item).strip()
        ],
    ]
    for candidate in candidates:
        if not candidate:
            continue
        candidate_normalized = _normalize_match_text(candidate)
        if candidate_normalized and candidate_normalized in normalized_query:
            matched_topics.append(candidate)
        profile_terms.update(_tokenize_match_terms(candidate))

    overlap = sorted(query_terms & profile_terms)
    score = float(len(overlap) + len(set(matched_topics)))
    score /= max(len(query_terms), 1)

    if not matched_topics:
        matched_topics = overlap[:4]
    else:
        deduped: list[str] = []
        for item in matched_topics:
            if item not in deduped:
                deduped.append(item)
        matched_topics = deduped[:4]
    return score, matched_topics


def _has_technical_query_signal(query_text: str) -> bool:
    normalized_query = _normalize_match_text(query_text)
    if any(marker in normalized_query for marker in TECHNICAL_QUERY_MARKERS):
        return True

    raw_query = str(query_text or "").strip()
    if re.search(r"\b[a-z][a-z0-9]*_[a-z0-9_]+\b", raw_query):
        return True
    if re.search(r"\b[A-Za-z]+[A-Z][A-Za-z0-9]+\b", raw_query):
        return True
    if re.search(r"`[^`]+`", raw_query):
        return True
    return False


def _query_matches_thread_scope(
    state: ChatbotState,
    query_text: str,
) -> tuple[bool, list[str]]:
    thread_name = str(state.get("thread_name") or "").strip()
    if not thread_name:
        return False, []

    normalized_query = _normalize_match_text(query_text)
    normalized_thread_name = _normalize_match_text(thread_name)
    if not normalized_thread_name or len(normalized_thread_name) < 2:
        return False, []

    if normalized_thread_name in normalized_query:
        return True, [thread_name]

    thread_terms = _tokenize_match_terms(thread_name)
    matched_terms = sorted(
        term for term in thread_terms if len(term) >= 2 and term in normalized_query
    )
    return bool(matched_terms), matched_terms


def _is_comparison_query(query_text: str) -> bool:
    lowered = _normalize_match_text(query_text)
    return any(marker in lowered for marker in COMPARISON_SCOPE_MARKERS)


def _has_rich_document_profiles(state: ChatbotState) -> bool:
    for profile in _iter_ordered_document_profiles(state):
        if any(
            str(item).strip()
            for item in (
                list(profile.get("main_topics") or [])
                + list(profile.get("keywords") or [])
                + list(profile.get("section_titles") or [])
            )
        ):
            return True
        original_filename = str(profile.get("original_filename") or "").strip()
        title = str(profile.get("title") or "").strip()
        document_type = str(profile.get("document_type") or "").strip()
        short_summary = str(profile.get("short_summary") or "").strip()
        normalized_summary = _normalize_match_text(short_summary)
        trivial_summaries = {
            _normalize_match_text(original_filename),
            _normalize_match_text(title),
            _normalize_match_text(f"{title} / {document_type}"),
            _normalize_match_text(f"{original_filename} / {document_type}"),
        }
        if normalized_summary and normalized_summary not in trivial_summaries:
            return True
    return False


def _infer_selection_type(
    *,
    query_text: str,
    selected_document_ids: list[str],
    selection_confident: bool,
    has_multi_document_scope: bool,
) -> str:
    if not selected_document_ids:
        return "thread_wide"
    if len(selected_document_ids) == 1 and selection_confident:
        return "single_document"
    if len(selected_document_ids) > 1:
        if _is_comparison_query(query_text):
            return "comparison"
        if selection_confident or has_multi_document_scope:
            return "multi_document"
        return "thread_wide"
    return "thread_wide"


def _should_invoke_llm_document_selection(
    state: ChatbotState,
    query_analysis: QueryAnalysisPayload,
) -> bool:
    active_document_ids = _get_active_document_ids(state)
    if len(active_document_ids) <= 1:
        return False
    if not _has_rich_document_profiles(state):
        return False
    query_kind = str(query_analysis.get("query_kind") or "").strip()
    if query_kind == "smalltalk":
        return False
    return True


def _answer_draft_signals_insufficient(answer_draft: str) -> bool:
    normalized_answer = _normalize_match_text(answer_draft)
    return any(
        marker in normalized_answer
        for marker in (
            _normalize_match_text(item)
            for item in INSUFFICIENT_ANSWER_MARKERS
        )
    )


def _apply_llm_document_selection(
    *,
    state: ChatbotState,
    llm: Any,
    query_text: str,
    base_query_analysis: QueryAnalysisPayload,
) -> QueryAnalysisPayload:
    structured_llm = llm.with_structured_output(
        DocumentSelectionResult,
        method="function_calling",
    )
    ordered_profiles = _iter_ordered_document_profiles(state)
    explicit_document_ids = _extract_explicit_document_ids(query_text, ordered_profiles)
    active_document_ids = _get_active_document_ids(state)
    result = structured_llm.invoke(
        [
            SystemMessage(content=build_stage5_document_selection_system_prompt()),
            HumanMessage(
                content=build_stage5_document_selection_user_prompt(
                    thread_name=str(state.get("thread_name") or "").strip() or None,
                    query_text=query_text,
                    document_profiles=ordered_profiles,
                    conversation_summary=(
                        str(state.get("conversation_summary") or "").strip() or None
                    ),
                    recent_dialog_lines=_build_recent_dialog_lines(state),
                )
            ),
        ]
    )

    active_document_id_set = set(active_document_ids)
    selected_document_ids = [
        document_id
        for document_id in result.selected_document_ids
        if document_id in active_document_id_set
    ]
    selection_type = str(result.query_type or "").strip() or str(
        base_query_analysis.get("selection_type") or "thread_wide"
    )
    if explicit_document_ids:
        explicit_document_id_set = set(explicit_document_ids)
        selected_document_ids = [
            document_id
            for document_id in active_document_ids
            if document_id in explicit_document_id_set
        ]
        if selected_document_ids:
            if len(selected_document_ids) == 1:
                selection_type = "single_document"
            elif _is_comparison_query(query_text):
                selection_type = "comparison"
            else:
                selection_type = "multi_document"
    if selection_type == "single_document" and not selected_document_ids:
        selection_type = "thread_wide"
    if selection_type == "thread_wide":
        selected_document_ids = _get_active_document_ids(state)

    normalized_retrieval_tasks = _normalize_retrieval_tasks(
        fallback_document_ids=selected_document_ids,
        retrieval_tasks=result.retrieval_tasks,
    )
    if not normalized_retrieval_tasks:
        normalized_retrieval_tasks = _build_default_retrieval_tasks(
            query_text=query_text,
            selection_type=selection_type,
            selected_document_ids=selected_document_ids,
        )
    answer_strategy = str(result.answer_strategy or "").strip() or str(
        base_query_analysis.get("answer_strategy") or "retrieve_chunks"
    )
    if answer_strategy not in {
        "retrieve_chunks",
        "conversation_memory",
        "direct",
    }:
        answer_strategy = "retrieve_chunks"
    should_use_balanced_raw_scope = _should_use_balanced_raw_document_search(
        answer_strategy=answer_strategy,
        explicit_document_ids=explicit_document_ids,
        active_document_ids=active_document_ids,
        selection_type=selection_type,
        selected_document_ids=selected_document_ids,
        retrieval_tasks=normalized_retrieval_tasks,
    )
    if should_use_balanced_raw_scope:
        selected_document_ids = list(active_document_ids)
        selection_type = "thread_wide"
        normalized_retrieval_tasks = _build_raw_query_tasks_per_document(
            query_text=query_text,
            document_ids=active_document_ids,
        )
    normalized_document_queries = _build_document_query_map_from_tasks(
        normalized_retrieval_tasks
    )
    use_per_document_search = (
        len(normalized_retrieval_tasks) > 1
        and all(len(task.get("document_ids") or []) == 1 for task in normalized_retrieval_tasks)
    )
    retrieval_mode_hint = (
        str(result.retrieval_mode or "").strip().lower()
        or str(base_query_analysis.get("retrieval_mode_hint") or "").strip().lower()
        or None
    )
    if retrieval_mode_hint not in {"dense", "hybrid"}:
        retrieval_mode_hint = None
    query_kind = str(base_query_analysis.get("query_kind") or "document_grounded")
    needs_clarification = False
    reason = str(base_query_analysis.get("reason") or "").strip()
    clarification_question = (
        str(result.clarification_question or "").strip() or None
    )

    if selection_type == "conversation_memory" or answer_strategy == "conversation_memory":
        query_kind = "conversation_memory"
        selected_document_ids = []
        normalized_retrieval_tasks = []
        normalized_document_queries = {}
        use_per_document_search = False
        retrieval_mode_hint = None
        answer_strategy = "conversation_memory"
        selection_type = "conversation_memory"
        reason = reason or "현재 스레드 대화 메모를 기준으로 답해야 하는 질문입니다."
    elif selection_type == "open_domain" or answer_strategy == "direct":
        query_kind = "open_domain_unrelated"
        selected_document_ids = []
        normalized_retrieval_tasks = []
        normalized_document_queries = {}
        use_per_document_search = False
        retrieval_mode_hint = None
        answer_strategy = "direct"
        selection_type = "open_domain"
        reason = reason or "문서 프로파일 기준으로는 일반 질문으로 판단했습니다."
    else:
        if query_kind not in {"lexical", "document_grounded"}:
            query_kind = "document_grounded"
        if selection_type == "thread_wide":
            reason = reason or "현재 스레드 문서를 넓게 함께 검색해야 하는 질문입니다."
        elif selection_type == "single_document":
            reason = reason or "문서 프로파일 기준으로 단일 문서를 선택했습니다."
        elif selection_type == "comparison":
            reason = reason or "여러 문서를 비교해야 하는 질문입니다."
        else:
            reason = reason or "여러 문서를 함께 다뤄야 하는 질문입니다."

    return {
        **base_query_analysis,
        "query_kind": query_kind,
        "needs_clarification": needs_clarification,
        "reason": reason,
        "selected_document_ids": selected_document_ids,
        "retrieval_tasks": normalized_retrieval_tasks,
        "selected_document_queries": normalized_document_queries,
        "selection_type": selection_type,
        "selection_source": "llm",
        "answer_strategy": answer_strategy,
        "retrieval_mode_hint": retrieval_mode_hint,
        "use_per_document_search": use_per_document_search,
        "document_match_score": float(
            base_query_analysis.get("document_match_score") or 0.0
        ),
        "clarification_question": clarification_question,
    }


def _select_document_candidates(
    state: ChatbotState,
    query_text: str,
) -> tuple[list[str], list[str], float, bool]:
    active_document_ids = _get_active_document_ids(state)
    ordered_profiles = _iter_ordered_document_profiles(state)
    if not active_document_ids:
        return [], [], 0.0, False

    explicit_document_ids = _extract_explicit_document_ids(query_text, ordered_profiles)
    if explicit_document_ids:
        selected_ids = [
            document_id
            for document_id in active_document_ids
            if document_id in explicit_document_ids
        ]
        return selected_ids, [], 1.0, True

    lowered_query = _normalize_match_text(query_text)
    if any(marker in lowered_query for marker in MULTI_DOCUMENT_SCOPE_MARKERS):
        return active_document_ids, [], 0.0, True

    scored_profiles: list[tuple[str, float, list[str]]] = []
    for profile in ordered_profiles:
        document_id = str(profile.get("document_id") or "").strip()
        if not document_id:
            continue
        score, matched_topics = _score_document_profile(query_text, profile)
        scored_profiles.append((document_id, score, matched_topics))

    if not scored_profiles:
        return active_document_ids, [], 0.0, False

    scored_profiles.sort(key=lambda item: item[1], reverse=True)
    top_document_id, top_score, top_topics = scored_profiles[0]
    second_score = scored_profiles[1][1] if len(scored_profiles) > 1 else 0.0
    confident_top_match = (
        top_score >= 0.2
        and (len(scored_profiles) == 1 or top_score >= second_score + 0.15)
    )
    if confident_top_match:
        return [top_document_id], top_topics, top_score, True

    return active_document_ids, top_topics, top_score, False


def _should_force_clarification(
    state: ChatbotState,
    query_text: str,
    *,
    selected_document_ids: list[str],
) -> tuple[bool, str]:
    active_document_ids = _get_active_document_ids(state)
    if not active_document_ids:
        return True, "현재 스레드에 연결된 문서가 없습니다."

    return False, ""


def _is_smalltalk_query(query_text: str) -> bool:
    normalized = " ".join(str(query_text or "").strip().lower().split())
    if not normalized:
        return False

    exact_matches = {
        "안녕",
        "안녕하세요",
        "반가워",
        "반갑습니다",
        "고마워",
        "감사합니다",
        "hi",
        "hello",
        "thanks",
        "thank you",
    }
    if normalized in exact_matches:
        return True

    smalltalk_prefixes = (
        "안녕",
        "안녕하세요",
        "hi ",
        "hello ",
        "thanks ",
        "thank you ",
    )
    return any(normalized.startswith(prefix) for prefix in smalltalk_prefixes)


def _is_conversation_memory_query(query_text: str) -> bool:
    lowered = _normalize_match_text(query_text)
    return any(marker in lowered for marker in CONVERSATION_MEMORY_MARKERS)


def _is_document_list_query(query_text: str) -> bool:
    lowered = _normalize_match_text(query_text)
    return "문서" in lowered and any(marker in lowered for marker in DOCUMENT_LIST_MARKERS)


def _build_document_list_response(state: ChatbotState) -> str:
    profiles = _iter_ordered_document_profiles(state)
    if not profiles:
        active_document_ids = _get_active_document_ids(state)
        if not active_document_ids:
            return "현재 이 채팅방에 연결된 문서가 없습니다."
        return "현재 이 채팅방에 연결된 문서 ID는 다음과 같습니다.\n\n" + "\n".join(
            f"{index}. {document_id}"
            for index, document_id in enumerate(active_document_ids, start=1)
        )

    lines = ["현재 이 채팅방에 연결된 문서는 다음과 같습니다."]
    for index, profile in enumerate(profiles, start=1):
        filename = str(profile.get("original_filename") or "").strip()
        title = str(profile.get("title") or "").strip()
        document_type = str(profile.get("document_type") or "").strip()
        label = filename or title or str(profile.get("document_id") or "").strip()
        details = " · ".join(item for item in [document_type, title if filename else ""] if item)
        lines.append(f"{index}. {label}" + (f" ({details})" if details else ""))
    return "\n".join(lines)


def _extract_user_facts_from_message(message_text: str) -> dict[str, str]:
    normalized_text = str(message_text or "").strip()
    if not normalized_text:
        return {}

    extracted: dict[str, str] = {}
    name_patterns = (
        r"(?:내 이름은|제 이름은)\s*['\"]?([0-9A-Za-z가-힣_-]{1,32})",
        r"(?:나는|저는)\s*['\"]?([0-9A-Za-z가-힣_-]{1,32})(?:라고|입니다|이에요|예요)",
    )
    for pattern in name_patterns:
        matched = re.search(pattern, normalized_text)
        if matched:
            extracted["name"] = matched.group(1).strip()
            break
    nickname_patterns = (
        r"(?:내 별명은|제 별명은)\s*['\"]?([0-9A-Za-z가-힣_-]{1,32})",
        r"(?:내 닉네임은|제 닉네임은)\s*['\"]?([0-9A-Za-z가-힣_-]{1,32})",
    )
    for pattern in nickname_patterns:
        matched = re.search(pattern, normalized_text)
        if matched:
            extracted["nickname"] = matched.group(1).strip()
            break
    return extracted


def _trim_summary_line(message: Any) -> str | None:
    if not isinstance(message, (HumanMessage, AIMessage)):
        return None
    if not isinstance(message.content, str):
        return None
    content = _truncate_text(message.content, limit=96)
    if not content:
        return None
    role = "사용자" if isinstance(message, HumanMessage) else "assistant"
    return f"{role}: {content}"


def _build_conversation_summary(
    existing_summary: str | None,
    older_dialog_messages: list[Any],
) -> str | None:
    existing_lines = [
        line.strip()
        for line in str(existing_summary or "").splitlines()
        if line.strip()
    ]
    new_lines = [
        line
        for line in (_trim_summary_line(message) for message in older_dialog_messages)
        if line
    ]
    merged: list[str] = []
    for line in [*existing_lines, *new_lines]:
        if line and line not in merged:
            merged.append(line)
    if not merged:
        return None
    return "\n".join(merged[-STAGE5_SUMMARY_MAX_LINES :])


def _build_model_input_messages(state: ChatbotState) -> list[Any]:
    messages = list(state.get("messages") or [])
    if not messages:
        return []

    last_human_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if isinstance(messages[index], HumanMessage):
            last_human_index = index
            break

    if last_human_index < 0:
        previous_messages: list[Any] = messages
        current_turn_messages: list[Any] = []
    else:
        previous_messages = messages[:last_human_index]
        current_turn_messages = messages[last_human_index:]

    previous_dialog_messages = [
        message
        for message in previous_messages
        if isinstance(message, (HumanMessage, AIMessage))
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    if not previous_dialog_messages:
        return current_turn_messages

    trimmed_previous = trim_messages(
        previous_dialog_messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=STAGE5_HISTORY_MAX_TOKENS,
        start_on=HumanMessage,
        end_on=(HumanMessage, AIMessage),
    )
    return [*trimmed_previous, *current_turn_messages]


def _build_memory_context_text(state: ChatbotState) -> str:
    parts: list[str] = []
    user_facts = dict(state.get("user_facts") or {})
    if user_facts:
        fact_lines = [
            f"- {key}: {value}"
            for key, value in sorted(user_facts.items())
            if str(value).strip()
        ]
        if fact_lines:
            parts.append("사용자 facts:\n" + "\n".join(fact_lines))
    conversation_summary = str(state.get("conversation_summary") or "").strip()
    if conversation_summary:
        parts.append("이전 대화 요약:\n" + conversation_summary)
    return "\n\n".join(parts).strip()


def _build_missing_evidence_clarification_payload(
    state: ChatbotState,
    *,
    query_text: str,
    selected_document_count: int,
) -> dict[str, Any]:
    profile_labels: list[str] = []
    for profile in _iter_ordered_document_profiles(state):
        original_filename = str(profile.get("original_filename") or "").strip()
        title = str(profile.get("title") or "").strip()
        if original_filename:
            profile_labels.append(original_filename)
        elif title:
            profile_labels.append(title)
        if len(profile_labels) >= 3:
            break

    joined_labels = ", ".join(profile_labels)
    if len(_get_active_document_ids(state)) > 1 and selected_document_count != 1:
        question = "현재 검색된 청크만으로는 어느 문서를 봐야 할지 확정하기 어렵습니다."
        if joined_labels:
            question += f" 기준 문서를 지정해주세요. 예: {joined_labels}"
        else:
            question += " 기준 문서를 지정해주세요."
    elif len(_get_active_document_ids(state)) > 1:
        question = (
            "현재 선택된 문서 청크에서 바로 답할 근거를 찾지 못했습니다. "
            "다른 문서를 말한 것이라면 문서를 지정하거나, 페이지/표 제목/키워드를 더 알려주세요."
        )
    else:
        question = (
            "현재 검색된 문서 청크에서 바로 답할 근거를 찾지 못했습니다. "
            "페이지, 표 제목, 섹션명, 키워드처럼 조금 더 구체적으로 말씀해주세요."
        )

    return {
        "kind": "clarification",
        "question": question,
        "reason": (
            "현재 검색된 청크가 질문과 직접 맞지 않거나, 답변에 필요한 근거가 부족합니다."
        ),
        "options": _get_active_document_ids(state),
        "query_text": query_text,
    }


def _build_query_analysis_from_intent(
    *,
    query_text: str,
    intent_analysis: IntentAnalysisPayload,
) -> QueryAnalysisPayload:
    answer_strategy = str(intent_analysis.get("answer_strategy") or "").strip()
    memory_mode = str(intent_analysis.get("memory_mode") or "").strip()
    reason = str(intent_analysis.get("reason") or "").strip()

    if memory_mode == "memory_only":
        return {
            "answer_strategy": "conversation_memory",
            "query_text": query_text,
            "query_kind": "conversation_memory",
            "needs_clarification": False,
            "reason": reason or "이전 대화 메모만으로 답할 수 있는 질문입니다.",
            "selected_document_ids": [],
            "retrieval_tasks": [],
            "selected_document_queries": {},
            "selection_type": "conversation_memory",
            "selection_source": "llm",
            "retrieval_mode_hint": None,
            "use_per_document_search": False,
            "matched_profile_topics": [],
            "document_match_score": 0.0,
            "clarification_question": None,
        }

    if answer_strategy == "direct":
        query_kind: QueryAnalysisPayload["query_kind"] = (
            "smalltalk" if _is_smalltalk_query(query_text) else "open_domain_unrelated"
        )
        return {
            "answer_strategy": "direct",
            "query_text": query_text,
            "query_kind": query_kind,
            "needs_clarification": False,
            "reason": reason or "문서 검색 없이 바로 답할 수 있는 질문입니다.",
            "selected_document_ids": [],
            "retrieval_tasks": [],
            "selected_document_queries": {},
            "selection_type": "open_domain",
            "selection_source": "llm",
            "retrieval_mode_hint": None,
            "use_per_document_search": False,
            "matched_profile_topics": [],
            "document_match_score": 0.0,
            "clarification_question": None,
        }

    return {
        "answer_strategy": "retrieve_chunks",
        "query_text": query_text,
        "query_kind": "document_grounded",
        "needs_clarification": False,
        "reason": reason
        or (
            "이전 대화를 참고해 검색 대상을 해석한 뒤 다시 검색해야 하는 질문입니다."
            if memory_mode == "resolve_for_retrieval"
            else "문서 검색이 필요한 질문입니다."
        ),
        "selected_document_ids": [],
        "retrieval_tasks": [],
        "selected_document_queries": {},
        "selection_type": "thread_wide",
        "selection_source": "llm",
        "retrieval_mode_hint": None,
        "use_per_document_search": False,
        "matched_profile_topics": [],
        "document_match_score": 0.0,
        "clarification_question": None,
    }


def build_classify_intent_node(*, llm: Any):
    """질문의 큰 방향만 먼저 분류하는 intent 노드를 생성한다."""

    structured_llm = llm.with_structured_output(
        IntentClassificationResult,
        method="function_calling",
    )

    def classify_intent(state: ChatbotState) -> dict[str, Any]:
        query_text = _get_latest_user_text(state)

        if _is_smalltalk_query(query_text):
            intent_analysis: IntentAnalysisPayload = {
                "query_text": query_text,
                "answer_strategy": "direct",
                "memory_mode": "none",
                "reason": "simple conversational query",
            }
        elif _is_document_list_query(query_text):
            intent_analysis = {
                "query_text": query_text,
                "answer_strategy": "direct",
                "memory_mode": "none",
                "reason": "current thread document list request",
            }
        elif _is_conversation_memory_query(query_text):
            intent_analysis = {
                "query_text": query_text,
                "answer_strategy": "direct",
                "memory_mode": "memory_only",
                "reason": "query refers to prior conversation memory",
            }
        else:
            result = structured_llm.invoke(
                [
                    SystemMessage(content=build_stage5_intent_system_prompt()),
                    HumanMessage(
                        content=build_stage5_intent_user_prompt(
                            query_text=query_text,
                            document_profiles=_iter_ordered_document_profiles(state),
                            conversation_summary=(
                                str(state.get("conversation_summary") or "").strip()
                                or None
                            ),
                            recent_dialog_lines=_build_recent_dialog_lines(state),
                        )
                    ),
                ]
            )
            answer_strategy = str(result.answer_strategy or "").strip()
            memory_mode = str(result.memory_mode or "").strip()
            if answer_strategy in {"conversation_memory", "memory_only"}:
                memory_mode = "memory_only"
                answer_strategy = "direct"
            if answer_strategy not in {"direct", "retrieve_chunks"}:
                answer_strategy = "retrieve_chunks"
            if memory_mode not in {"none", "memory_only", "resolve_for_retrieval"}:
                memory_mode = "none"
            if memory_mode == "memory_only":
                answer_strategy = "direct"
            elif memory_mode == "resolve_for_retrieval":
                answer_strategy = "retrieve_chunks"
            intent_analysis = {
                "query_text": query_text,
                "answer_strategy": answer_strategy,  # type: ignore[typeddict-item]
                "memory_mode": memory_mode,  # type: ignore[typeddict-item]
                "reason": str(result.reason or "").strip()
                or "llm intent classification result",
            }

        query_analysis = _build_query_analysis_from_intent(
            query_text=query_text,
            intent_analysis=intent_analysis,
        )
        return {
            "intent_analysis": intent_analysis,
            "query_analysis": query_analysis,
            "logs": [
                "classify_intent:"
                f"{intent_analysis.get('answer_strategy') or 'retrieve_chunks'}:"
                f"{intent_analysis.get('memory_mode') or 'none'}"
            ],
        }

    return classify_intent


def _build_query_analysis(
    state: ChatbotState,
    query_text: str,
    *,
    memory_mode: str = "none",
) -> QueryAnalysisPayload:
    lowered = _normalize_match_text(query_text)
    thread_scope_matched, matched_thread_terms = _query_matches_thread_scope(
        state,
        query_text,
    )
    has_technical_signal = _has_technical_query_signal(query_text)
    (
        selected_document_ids,
        matched_profile_topics,
        document_match_score,
        selection_confident,
    ) = (
        _select_document_candidates(
            state,
            query_text,
        )
    )
    structured_followup_document_ids, structured_followup_reason = (
        _resolve_structured_followup_document_scope(
            state,
            query_text,
            force=memory_mode == "resolve_for_retrieval",
        )
    )
    previous_search_payload = _extract_previous_search_payload(state) or {}
    previous_document_ids = [
        str(item).strip()
        for item in previous_search_payload.get("active_document_ids") or []
        if str(item).strip()
    ]
    followup_document_scope = False
    followup_scope_reason = structured_followup_reason or ""
    if structured_followup_document_ids:
        selected_document_ids = structured_followup_document_ids
        document_match_score = max(document_match_score, 0.5)
        selection_confident = True
        followup_document_scope = True
    elif bool(previous_document_ids) and any(
        marker in lowered for marker in FOLLOW_UP_DOCUMENT_MARKERS
    ):
        selected_document_ids = previous_document_ids
        document_match_score = max(document_match_score, 0.5)
        selection_confident = True
        followup_document_scope = True
        followup_scope_reason = "직전 문서 검색 대상을 이어받은 후속 질문입니다."
    has_multi_document_scope = any(
        marker in lowered for marker in MULTI_DOCUMENT_SCOPE_MARKERS
    )
    selection_type = _infer_selection_type(
        query_text=query_text,
        selected_document_ids=selected_document_ids,
        selection_confident=selection_confident,
        has_multi_document_scope=has_multi_document_scope,
    )
    use_per_document_search = (
        selection_type in {"multi_document", "comparison"}
        and len(selected_document_ids) > 1
    )

    if _is_smalltalk_query(query_text):
        query_kind: QueryAnalysisPayload["query_kind"] = "smalltalk"
        needs_clarification = False
        reason = "simple conversational query"
        selection_type = "open_domain"
        use_per_document_search = False
        answer_strategy: QueryAnalysisPayload["answer_strategy"] = "direct"
    elif _is_conversation_memory_query(query_text):
        query_kind = "conversation_memory"
        needs_clarification = False
        reason = "query refers to prior conversation memory"
        selection_type = "conversation_memory"
        use_per_document_search = False
        answer_strategy = "conversation_memory"
    else:
        has_lexical_marker = any(token in lowered for token in LEXICAL_REFERENCE_MARKERS)
        has_document_marker = any(
            token in lowered for token in DOCUMENT_REFERENCE_MARKERS
        )
        if has_lexical_marker:
            query_kind = "lexical"
        elif (
            has_document_marker
            or matched_profile_topics
            or followup_document_scope
            or (thread_scope_matched and has_technical_signal)
            or document_match_score >= 0.2
            or selection_confident
            or has_multi_document_scope
        ):
            query_kind = "document_grounded"
        else:
            query_kind = "open_domain_unrelated"

        needs_clarification = False
        reason = ""
        if query_kind in {"lexical", "document_grounded"}:
            needs_clarification, reason = _should_force_clarification(
                state,
                query_text,
                selected_document_ids=selected_document_ids,
            )
            if needs_clarification:
                query_kind = "ambiguous"
            elif selection_confident and len(selected_document_ids) == 1:
                reason = "질문에서 단일 대상 문서를 식별했습니다."
            elif has_multi_document_scope:
                reason = "질문이 여러 문서를 함께 다루고 있습니다."
            elif followup_document_scope:
                reason = (
                    followup_scope_reason
                    or "직전 문서 검색 대상을 이어받은 후속 질문입니다."
                )
            elif matched_profile_topics:
                reason = "query overlaps with current document profile"
            elif thread_scope_matched and has_technical_signal:
                reason = (
                    "질문이 현재 스레드 주제와 기술 질의를 함께 포함하고 있습니다."
                )
            elif has_document_marker:
                reason = "query explicitly references the current document"
            else:
                reason = "query requires grounded document retrieval"
        else:
            reason = "query is not related to the current document scope"
            selection_type = "open_domain"
            use_per_document_search = False
        answer_strategy = (
            "direct"
            if query_kind == "open_domain_unrelated"
            else "retrieve_chunks"
        )

    return {
        "answer_strategy": answer_strategy,
        "query_text": query_text,
        "query_kind": query_kind,
        "needs_clarification": needs_clarification,
        "reason": reason,
        "selected_document_ids": selected_document_ids,
        "retrieval_tasks": [],
        "selected_document_queries": {},
        "selection_type": selection_type,
        "selection_source": "deterministic",
        "retrieval_mode_hint": None,
        "use_per_document_search": use_per_document_search,
        "matched_profile_topics": [
            *matched_profile_topics,
            *[
                term
                for term in matched_thread_terms
                if term not in matched_profile_topics
            ],
        ],
        "document_match_score": document_match_score,
        "clarification_question": None,
    }


def _build_retrieval_policy(
    state: ChatbotState,
    query_analysis: QueryAnalysisPayload,
) -> RetrievalPolicyPayload:
    query_kind = query_analysis.get("query_kind") or "general"
    selected_document_ids = _normalize_string_list(
        query_analysis.get("selected_document_ids")
    )
    retrieval_tasks = _normalize_retrieval_tasks(
        fallback_document_ids=selected_document_ids,
        retrieval_tasks=list(query_analysis.get("retrieval_tasks") or []),
    )
    multi_document_search = len(selected_document_ids) > 1 or len(retrieval_tasks) > 1
    base_mode = STAGE5_DEFAULT_RETRIEVAL_MODE
    retrieval_mode_hint = str(
        query_analysis.get("retrieval_mode_hint") or ""
    ).strip().lower()
    hinted_mode = retrieval_mode_hint if retrieval_mode_hint in {"dense", "hybrid"} else None
    task_types = {
        str(task.get("task_type") or "").strip()
        for task in retrieval_tasks
        if str(task.get("task_type") or "").strip()
    }
    task_strategies = {
        str(task.get("retrieval_strategy") or "").strip()
        for task in retrieval_tasks
        if str(task.get("retrieval_strategy") or "").strip()
    }
    strategy_mode = (
        "hybrid"
        if (
            "hybrid_search" in task_strategies
            or "asset_lookup" in task_strategies
            or bool(task_types & {"exact_keyword", "figure_table", "procedure"})
        )
        else None
    )
    resolved_mode = hinted_mode or strategy_mode or base_mode
    resolved_use_rerank = query_kind in {"document_grounded", "lexical"} or multi_document_search
    if task_strategies == {"document_overview"}:
        resolved_use_rerank = False
    return {
        "mode": resolved_mode,  # type: ignore[typeddict-item]
        "use_rerank": resolved_use_rerank,
        "enable_mmr": False,
        "use_web_search": False,
        "top_k": STAGE5_DEFAULT_TOP_K,
        "score_threshold": None,
        "use_context_window": bool(STAGE5_ENABLE_CONTEXT_WINDOW),
        "context_window_size": max(1, STAGE5_CONTEXT_WINDOW_SIZE),
    }


def load_request_context(state: ChatbotState) -> dict[str, Any]:
    """외부 입력을 챗봇 state 기본 구조에 맞게 정리한다."""
    normalized_user_message = str(state.get("user_message") or "").strip()
    unresolved_clarification_payload = dict(state.get("clarification_payload") or {})
    should_persist_unresolved_interrupt = bool(
        state.get("needs_clarification")
        and unresolved_clarification_payload
        and not _has_matching_interrupt_history(
            list(state.get("messages") or []),
            unresolved_clarification_payload,
        )
    )
    updates: dict[str, Any] = {
        "logs": ["load_request_context"],
        "log_cursor": len(list(state.get("logs") or [])),
        "retrieval_hits": [],
        "expanded_context_blocks": [],
        "citations": [],
        "evidence_chunks": [],
        "visual_asset_refs": [],
        "answer_draft": None,
        "final_answer": None,
        "debug_trace": None,
        "grounding_decision": {
            "action": "retrieve_deeper",
            "clarification_question": None,
        },
        "clarification_payload": None,
        "clarification_response": None,
        "needs_clarification": False,
        "deep_retrieval_attempted": False,
        "retrieval_document_ids": _get_active_document_ids(state),
        "retrieval_tasks": [],
        "retrieval_document_queries": {},
        "use_per_document_search": False,
    }

    current_messages = list(state.get("messages") or [])
    last_message = current_messages[-1] if current_messages else None
    appended_messages: list[Any] = list(current_messages)
    new_messages: list[Any] = []
    if should_persist_unresolved_interrupt:
        interrupt_message = _build_interrupt_history_message(
            unresolved_clarification_payload
        )
        if interrupt_message is not None:
            new_messages.append(interrupt_message)
            appended_messages.append(interrupt_message)
    if normalized_user_message and not (
        isinstance(last_message, HumanMessage)
        and isinstance(last_message.content, str)
        and last_message.content.strip() == normalized_user_message
    ):
        created_at = _utc_now_iso()
        user_message = HumanMessage(
            content=normalized_user_message,
            additional_kwargs=_build_thread_chat_metadata(created_at=created_at),
        )
        new_messages.append(user_message)
        appended_messages.append(user_message)
    if new_messages:
        updates["messages"] = new_messages

    merged_user_facts = dict(state.get("user_facts") or {})
    merged_user_facts.update(_extract_user_facts_from_message(normalized_user_message))
    updates["user_facts"] = merged_user_facts

    dialog_messages = [
        message
        for message in appended_messages
        if isinstance(message, (HumanMessage, AIMessage))
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    trimmed_dialog_messages = trim_messages(
        dialog_messages,
        strategy="last",
        token_counter=count_tokens_approximately,
        max_tokens=STAGE5_HISTORY_MAX_TOKENS,
        start_on=HumanMessage,
        end_on=(HumanMessage, AIMessage),
    ) if dialog_messages else []
    trimmed_ids = {id(message) for message in trimmed_dialog_messages}
    older_dialog_messages = [
        message for message in dialog_messages if id(message) not in trimmed_ids
    ]
    updated_summary = _build_conversation_summary(
        str(state.get("conversation_summary") or "").strip() or None,
        older_dialog_messages,
    )
    updates["conversation_summary"] = updated_summary
    return updates


def build_plan_retrieval_node(*, llm: Any):
    """검색이 필요한 질문에 대해 문서 선택과 retrieval 계획을 세운다."""

    def plan_retrieval(state: ChatbotState) -> dict[str, Any]:
        query_text = _get_latest_user_text(state)
        intent_analysis = dict(state.get("intent_analysis") or {})
        memory_mode = str(intent_analysis.get("memory_mode") or "").strip()
        query_analysis = _build_query_analysis(
            state,
            query_text,
            memory_mode=memory_mode,
        )
        preserved_answer_strategy = str(
            intent_analysis.get("answer_strategy")
            or query_analysis.get("answer_strategy")
            or ""
        ).strip()
        if str(intent_analysis.get("memory_mode") or "").strip() == "resolve_for_retrieval":
            query_analysis["reason"] = (
                str(query_analysis.get("reason") or "").strip()
                or "이전 대화를 참고해 검색 대상을 해석한 뒤 다시 검색해야 하는 질문입니다."
            )
        if _should_invoke_llm_document_selection(state, query_analysis):
            query_analysis = _apply_llm_document_selection(
                state=state,
                llm=llm,
                query_text=query_text,
                base_query_analysis=query_analysis,
            )
        selected_document_ids = [
            str(item).strip()
            for item in query_analysis.get("selected_document_ids") or []
            if str(item).strip()
        ]
        retrieval_tasks = _normalize_retrieval_tasks(
            fallback_document_ids=selected_document_ids,
            retrieval_tasks=list(query_analysis.get("retrieval_tasks") or []),
        )
        if not retrieval_tasks and str(
            query_analysis.get("answer_strategy") or ""
        ).strip() == "retrieve_chunks":
            retrieval_tasks = _build_default_retrieval_tasks(
                query_text=query_text,
                selection_type=str(
                    query_analysis.get("selection_type") or "thread_wide"
                ).strip()
                or "thread_wide",
                selected_document_ids=(
                    selected_document_ids or _get_active_document_ids(state)
                ),
            )
        explicit_document_ids = _extract_explicit_document_ids(
            query_text,
            _iter_ordered_document_profiles(state),
        )
        if _should_use_balanced_raw_document_search(
            answer_strategy=str(query_analysis.get("answer_strategy") or "").strip(),
            explicit_document_ids=explicit_document_ids,
            active_document_ids=_get_active_document_ids(state),
            selection_type=(
                str(query_analysis.get("selection_type") or "thread_wide").strip()
                or "thread_wide"
            ),
            selected_document_ids=selected_document_ids,
            retrieval_tasks=retrieval_tasks,
        ):
            selected_document_ids = _get_active_document_ids(state)
            retrieval_tasks = _build_raw_query_tasks_per_document(
                query_text=query_text,
                document_ids=selected_document_ids,
            )
            query_analysis = {
                **query_analysis,
                "selection_type": "thread_wide",
                "selected_document_ids": selected_document_ids,
                "retrieval_tasks": retrieval_tasks,
                "reason": (
                    "문서명이 명시되지 않은 다중/광역 질문이라 전체 문서를 균형 검색합니다."
                ),
            }
        retrieval_document_ids = _collect_document_ids_from_tasks(retrieval_tasks)
        if not retrieval_document_ids:
            retrieval_document_ids = selected_document_ids or _get_active_document_ids(
                state
            )
        retrieval_document_queries = _build_document_query_map_from_tasks(
            retrieval_tasks
        )
        use_per_document_search = (
            len(retrieval_tasks) > 1
            and all(len(task.get("document_ids") or []) == 1 for task in retrieval_tasks)
        )
        query_analysis = {
            **query_analysis,
            "selected_document_ids": retrieval_document_ids,
            "retrieval_tasks": retrieval_tasks,
            "selected_document_queries": retrieval_document_queries,
            "use_per_document_search": use_per_document_search,
        }
        retrieval_policy = _build_retrieval_policy(state, query_analysis)
        return {
            "query_analysis": query_analysis,
            "retrieval_policy": retrieval_policy,
            "retrieval_document_ids": retrieval_document_ids,
            "retrieval_tasks": retrieval_tasks,
            "retrieval_document_queries": retrieval_document_queries,
            "use_per_document_search": use_per_document_search,
            "needs_clarification": bool(query_analysis.get("needs_clarification")),
            "logs": [
                "plan_retrieval:"
                f"{query_analysis.get('query_kind') or 'general'}:"
                f"{query_analysis.get('selection_source') or 'deterministic'}"
            ],
        }

    return plan_retrieval


def build_classify_query_node(*, llm: Any):
    """기존 명칭 호환용 alias."""
    return build_plan_retrieval_node(llm=llm)


def clarify_if_needed(state: ChatbotState) -> dict[str, Any]:
    """문서 범위가 모호하면 사용자 clarification을 요청한다."""
    query_analysis = state.get("query_analysis") or {}
    options = _get_active_document_ids(state)
    ordered_profiles = _shared_iter_ordered_document_profiles(
        options,
        state.get("document_profiles") or [],
    )
    payload = dict(state.get("clarification_payload") or {})
    if not payload:
        clarification_question = (
            str(query_analysis.get("clarification_question") or "").strip() or None
        )
        payload = {
            "kind": "clarification",
            "question": clarification_question or "어떤 문서를 기준으로 답할까요?",
            "reason": str(query_analysis.get("reason") or "질문 범위를 확정해야 합니다."),
            "options": options,
        }
    payload_options = [str(item) for item in payload.get("options") or [] if str(item)]
    response = interrupt(payload)
    normalized_response = str(response or "").strip() or None
    updated_document_ids = options
    resumed_messages: list[Any] = []
    updated_user_message = state.get("user_message")
    explicit_document_ids: list[str] = []
    if normalized_response:
        explicit_document_ids = _extract_explicit_document_ids(
            normalized_response,
            ordered_profiles,
        )
    if normalized_response and not _has_matching_interrupt_history(
        list(state.get("messages") or []),
        payload,
    ):
        interrupt_message = _build_interrupt_history_message(payload)
        if interrupt_message is not None:
            resumed_messages.append(interrupt_message)
    if normalized_response and normalized_response in payload_options:
        updated_document_ids = [normalized_response]
    elif len(explicit_document_ids) == 1:
        updated_document_ids = explicit_document_ids
    elif normalized_response:
        # clarification 응답이 문서 지정이 아니면, 기존 질문에 무조건 덧붙이지 않고
        # 새 질문으로 해석해 다음 classify 단계가 새 턴처럼 처리하게 둔다.
        updated_user_message = normalized_response
    if normalized_response:
        resumed_messages = [
            *resumed_messages,
            HumanMessage(
                content=normalized_response,
                additional_kwargs=_build_thread_chat_metadata(),
            ),
        ]
    return {
        "clarification_payload": payload,
        "clarification_response": normalized_response,
        "retrieval_document_ids": updated_document_ids,
        "retrieval_tasks": [],
        "retrieval_document_queries": {},
        "use_per_document_search": False,
        "needs_clarification": False,
        "messages": resumed_messages,
        "user_message": updated_user_message,
        "logs": ["clarify_if_needed:resume"],
    }


def build_agent_llm_node(
    *,
    llm: Any,
    tools: list[Any],
):
    """tool-calling agent node를 생성한다."""
    bound_llm = llm.bind_tools(tools)

    def agent_llm(state: ChatbotState) -> dict[str, Any]:
        retrieval_policy = dict(state.get("retrieval_policy") or {})
        retrieval_document_ids = _get_retrieval_document_ids(state)
        retrieval_document_id_set = set(retrieval_document_ids)
        system_prompt = build_stage5_agent_system_prompt(
            active_document_ids=retrieval_document_ids,
            retrieval_mode=str(
                retrieval_policy.get("mode") or STAGE5_DEFAULT_RETRIEVAL_MODE
            ),
            document_profiles=[
                dict(profile)
                for profile in state.get("document_profiles") or []
                if isinstance(profile, dict)
                and (
                    not retrieval_document_id_set
                    or str(profile.get("document_id") or "").strip()
                    in retrieval_document_id_set
                )
            ],
        )
        prompt_messages: list[Any] = [SystemMessage(content=system_prompt)]
        memory_context = _build_memory_context_text(state)
        if memory_context:
            prompt_messages.append(
                SystemMessage(content=f"이전 대화 메모:\n{memory_context}")
            )
        prompt_messages.extend(_build_model_input_messages(state))
        response = bound_llm.invoke(prompt_messages)
        answer_strategy = str(
            (state.get("query_analysis") or {}).get("answer_strategy") or ""
        ).strip()
        current_turn_has_tool_result = bool(
            _iter_tool_messages(state, current_turn_only=True)
        )
        if (
            answer_strategy == "retrieve_chunks"
            and not current_turn_has_tool_result
            and not getattr(response, "tool_calls", None)
        ):
            forced_query = str(
                (state.get("query_analysis") or {}).get("query_text")
                or _get_latest_user_text(state)
                or state.get("user_message")
                or ""
            ).strip()
            if forced_query:
                forced_tool_call = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_thread_knowledge",
                            "args": {"query": forced_query},
                            "id": "forced-search-thread-knowledge",
                            "type": "tool_call",
                        }
                    ],
                )
                return {
                    "answer_draft": None,
                    "messages": [forced_tool_call],
                    "logs": ["agent_llm:forced_search"],
                }
        answer_draft = None
        if isinstance(response.content, str):
            answer_draft = response.content.strip() or None
        return {
            "answer_draft": answer_draft,
            "messages": [response],
            "logs": [
                "agent_llm:tool_call"
                if getattr(response, "tool_calls", None)
                else "agent_llm:answer"
            ],
        }

    return agent_llm


def build_run_retrieval_node(
    *,
    retrieval_runner: Any = default_search_thread_knowledge,
    overview_loader: Any | None = None,
):
    """계획된 retrieval task를 LLM tool-call 재작성 없이 실행한다."""
    resolved_retrieval_runner = retrieval_runner or default_search_thread_knowledge

    def run_retrieval(state: ChatbotState) -> dict[str, Any]:
        query_analysis = dict(state.get("query_analysis") or {})
        retrieval_policy = dict(state.get("retrieval_policy") or {})
        query_text = str(
            query_analysis.get("query_text") or _get_latest_user_text(state)
        ).strip()
        retrieval_document_ids = _get_retrieval_document_ids(state)
        retrieval_tasks = _get_retrieval_tasks(state)
        retrieval_document_queries = _get_retrieval_document_queries(state)
        overview_tasks, search_tasks = _split_overview_and_search_tasks(
            retrieval_tasks
        )
        overview_document_ids = _collect_document_ids_from_tasks(overview_tasks)
        search_document_ids = _collect_document_ids_from_tasks(search_tasks)
        if overview_tasks and not overview_document_ids:
            overview_document_ids = list(retrieval_document_ids)
        if search_tasks and not search_document_ids:
            search_document_ids = list(retrieval_document_ids)
        use_per_document_search = bool(state.get("use_per_document_search")) and len(
            search_document_ids or retrieval_document_ids
        ) > 1
        thread_id = str(state.get("thread_id") or "").strip() or None
        collection_name = str(state.get("collection_name") or "").strip() or None
        if collection_name is None and thread_id:
            collection_name = build_thread_collection_name(thread_id)

        base_top_k = int(retrieval_policy.get("top_k") or STAGE5_DEFAULT_TOP_K)
        search_unit_count = max(
            1,
            len(retrieval_tasks),
            len(retrieval_document_ids) if len(retrieval_document_ids) > 1 else 1,
        )
        top_k = (
            base_top_k
            if search_unit_count <= 1
            else min(16, max(12, search_unit_count * 4))
        )
        fetch_k = max(top_k * 2, 20)
        mode = (
            str(retrieval_policy.get("mode") or STAGE5_DEFAULT_RETRIEVAL_MODE).strip()
            or STAGE5_DEFAULT_RETRIEVAL_MODE
        )

        overview_hits: list[dict[str, Any]] = []
        if overview_tasks and callable(overview_loader):
            try:
                overview_hits = [
                    dict(hit)
                    for hit in overview_loader(
                        thread_id=thread_id,
                        active_document_ids=overview_document_ids,
                        max_blocks_per_document=(4 if len(overview_document_ids) > 1 else 8),
                    )
                    or []
                    if isinstance(hit, dict)
                ]
            except Exception:
                overview_hits = []

        if search_tasks:
            runner_document_ids = search_document_ids or retrieval_document_ids
            runner_document_queries = _build_document_query_map_from_tasks(search_tasks)
            result = resolved_retrieval_runner(
                query=query_text,
                thread_id=thread_id,
                active_document_ids=runner_document_ids,
                retrieval_tasks=search_tasks,
                document_queries=runner_document_queries or retrieval_document_queries,
                collection_name=collection_name,
                retrieval_mode=mode,
                top_k=top_k,
                fetch_k=fetch_k,
                dense_fetch_k=fetch_k,
                bm25_fetch_k=fetch_k,
                use_per_document_search=use_per_document_search,
                per_document_top_k=(
                    STAGE5_MULTI_DOC_PER_DOCUMENT_TOP_K
                    if (use_per_document_search or len(search_tasks) > 1)
                    else None
                ),
                enable_rerank=bool(retrieval_policy.get("use_rerank", True)),
                enable_mmr=bool(retrieval_policy.get("enable_mmr", False)),
                score_threshold=retrieval_policy.get("score_threshold"),
            )
        else:
            result = {
                "status": "completed",
                "query": query_text,
                "thread_id": thread_id,
                "active_document_ids": overview_document_ids or retrieval_document_ids,
                "collection_name": collection_name,
                "retrieval_mode": "overview" if overview_tasks else mode,
                "top_k": len(overview_hits),
                "fetch_k": len(overview_hits),
                "dense_fetch_k": 0,
                "bm25_fetch_k": 0,
                "retrievals": [],
                "retrieved_count": 0,
                "rerank_applied": False,
                "rerank_error": None,
                "mmr_applied": False,
                "task_search_used": bool(overview_tasks),
                "per_document_search_used": len(overview_document_ids) > 1,
                "retrieval_task_count": len(retrieval_tasks),
                "retrieval_tasks": retrieval_tasks,
            }
        result = dict(result or {})
        result["rerank_requested"] = bool(retrieval_policy.get("use_rerank", True))
        result["mmr_requested"] = bool(retrieval_policy.get("enable_mmr", False))
        result["overview_search_used"] = bool(overview_tasks)
        result["retrieval_task_count"] = len(retrieval_tasks)
        result["retrieval_tasks"] = retrieval_tasks
        if overview_hits:
            existing_hits = [
                dict(hit)
                for hit in list(result.get("retrievals") or [])
                if isinstance(hit, dict)
            ]
            result["retrievals"] = [*overview_hits, *existing_hits]
            result["retrieved_count"] = len(result["retrievals"])
            result["active_document_ids"] = _normalize_string_list(
                [
                    *(result.get("active_document_ids") or []),
                    *overview_document_ids,
                ]
            )
        retrieval_hits = list(result.get("retrievals") or [])
        return {
            "retrieval_hits": retrieval_hits,
            "answer_draft": None,
            "messages": [
                ToolMessage(
                    content=json.dumps(result, ensure_ascii=False),
                    name="search_thread_knowledge",
                    tool_call_id="planned-search-thread-knowledge",
                )
            ],
            "logs": [
                f"run_retrieval:{result.get('retrieval_mode') or mode}:{len(retrieval_hits)}:"
                f"{'tasks' if retrieval_tasks else 'single'}"
            ],
        }

    return run_retrieval


def _build_memory_response(state: ChatbotState, query_text: str) -> str | None:
    lowered = _normalize_match_text(query_text)
    user_facts = dict(state.get("user_facts") or {})
    dialog_messages = [
        message
        for message in state.get("messages") or []
        if isinstance(message, (HumanMessage, AIMessage))
        and isinstance(message.content, str)
        and message.content.strip()
    ]
    prior_dialog_messages = dialog_messages[:-1] if dialog_messages else []

    if "이름" in lowered:
        user_name = str(user_facts.get("name") or "").strip()
        if user_name:
            return f"당신의 이름은 '{user_name}'입니다."
        return "현재 저장된 이름 정보가 없습니다."

    if "별명" in lowered or "닉네임" in lowered:
        nickname = str(user_facts.get("nickname") or "").strip()
        if nickname:
            return f"당신의 별명은 '{nickname}'입니다."
        return "현재 저장된 별명 정보가 없습니다."

    if "직전 답변" in lowered or "이전 답변" in lowered or "방금 뭐라고" in lowered:
        for message in reversed(prior_dialog_messages):
            if isinstance(message, AIMessage) and isinstance(message.content, str):
                content = message.content.strip()
                if content:
                    return f"직전 답변은 다음과 같습니다.\n\n{content}"

    if "방금 한 질문" in lowered or "내가 뭐라고" in lowered or "제가 뭐라고" in lowered:
        for message in reversed(prior_dialog_messages):
            if isinstance(message, HumanMessage) and isinstance(message.content, str):
                content = message.content.strip()
                if content:
                    return f"직전 질문은 다음과 같습니다.\n\n{content}"

    conversation_summary = str(state.get("conversation_summary") or "").strip()
    if conversation_summary:
        return "현재 저장된 대화 요약은 다음과 같습니다.\n\n" + conversation_summary
    return None


def build_direct_response_node(
    *,
    llm: Any,
):
    """문서 검색 없이 답하는 경로를 담당하는 노드를 생성한다."""

    def respond_without_documents(state: ChatbotState) -> dict[str, Any]:
        query_analysis = dict(state.get("query_analysis") or {})
        query_kind = str(query_analysis.get("query_kind") or "general").strip()
        query_text = str(
            query_analysis.get("query_text") or _get_latest_user_text(state)
        ).strip()
        direct_answer: str | None = None

        if _is_document_list_query(query_text):
            direct_answer = _build_document_list_response(state)
        elif query_kind == "smalltalk":
            lowered = _normalize_match_text(query_text)
            if any(marker in lowered for marker in THANKS_MARKERS):
                direct_answer = "네. 필요하시면 이어서 질문해주세요."
            else:
                direct_answer = "안녕하세요. 무엇을 도와드릴까요?"
        elif query_kind == "conversation_memory":
            direct_answer = _build_memory_response(state, query_text)

        if direct_answer is None:
            system_prompt = (
                build_stage5_memory_system_prompt()
                if query_kind == "conversation_memory"
                else build_stage5_general_response_system_prompt()
            )
            prompt_messages: list[Any] = [SystemMessage(content=system_prompt)]
            memory_context = _build_memory_context_text(state)
            if memory_context:
                prompt_messages.append(
                    SystemMessage(content=f"대화 메모:\n{memory_context}")
                )
            prompt_messages.append(HumanMessage(content=query_text))
            response = llm.invoke(prompt_messages)
            if isinstance(response.content, str):
                direct_answer = response.content.strip() or None

        final_answer = direct_answer or "현재 바로 답할 수 있는 정보가 없습니다."
        final_log = f"respond_without_documents:{query_kind}"
        debug_trace = _build_debug_trace(
            state,
            final_log_entries=[final_log],
        )
        return {
            "answer_draft": final_answer,
            "final_answer": final_answer,
            "debug_trace": debug_trace,
            "retrieval_hits": [],
            "expanded_context_blocks": [],
            "grounding_decision": {
                "action": "answer",
                "clarification_question": None,
            },
            "citations": [],
            "evidence_chunks": [],
            "visual_asset_refs": [],
            "messages": [
                AIMessage(
                    content=final_answer,
                    name="stage5_direct_answer",
                    additional_kwargs=_build_thread_chat_metadata(
                        citations=[],
                        evidence_chunks=[],
                        visual_asset_refs=[],
                        retrieval_mode=debug_trace.get("retrieval_mode"),
                        debug_trace=debug_trace,
                    ),
                )
            ],
            "logs": [final_log],
        }

    return respond_without_documents


def _resolve_context_blocks(
    *,
    state: ChatbotState,
    retrieval_hits: list[dict[str, Any]],
    context_window_loader: Any | None,
) -> list[str]:
    if not retrieval_hits:
        return []

    window_hits = retrieval_hits[: max(1, STAGE5_CONTEXT_WINDOW_MAX_HITS)]
    profile_lookup = _build_document_profile_lookup(state)
    if any(
        str(hit.get("expansion_mode") or "").strip() == "document_overview"
        for hit in window_hits
    ):
        return _build_context_blocks(
            retrieval_hits,
            profile_lookup=profile_lookup,
        )

    tool_blocks = _extract_latest_context_window_blocks(state)
    if tool_blocks:
        rendered = _render_expanded_context_blocks(
            tool_blocks,
            profile_lookup=profile_lookup,
        )
        if rendered:
            return rendered

    retrieval_policy = dict(state.get("retrieval_policy") or {})
    if not bool(retrieval_policy.get("use_context_window")):
        return _build_context_blocks(
            retrieval_hits,
            profile_lookup=profile_lookup,
        )

    if callable(context_window_loader):
        try:
            expanded_blocks = context_window_loader(
                thread_id=str(state.get("thread_id") or "").strip() or None,
                active_document_ids=_get_retrieval_document_ids(state),
                chunk_ids=[
                    f"{str(hit.get('document_id') or '').strip()}:{str(hit.get('chunk_id') or '').strip()}"
                    for hit in window_hits
                    if str(hit.get("document_id") or "").strip()
                    and str(hit.get("chunk_id") or "").strip()
                ],
                window_size=int(
                    retrieval_policy.get("context_window_size")
                    or STAGE5_CONTEXT_WINDOW_SIZE
                ),
            )
            rendered = _render_expanded_context_blocks(
                list(expanded_blocks or []),
                profile_lookup=profile_lookup,
            )
            if rendered:
                return rendered
        except Exception:
            pass

    return _build_context_blocks(
        retrieval_hits,
        profile_lookup=profile_lookup,
    )


def build_grounding_check_node(
    *,
    llm: Any,
    context_window_loader: Any | None = None,
):
    """retrieval hit가 있을 때만 구조화된 LLM 판단을 수행하는 grounding 노드를 만든다."""
    structured_grounding_llm = llm.with_structured_output(GroundingDecisionResult)

    def grounding_check(state: ChatbotState) -> dict[str, Any]:
        latest_search_payload = _extract_latest_search_payload(state)
        query_analysis = dict(state.get("query_analysis") or {})
        query_kind = str(query_analysis.get("query_kind") or "general").strip()
        deep_retrieval_attempted = bool(state.get("deep_retrieval_attempted"))
        available_document_count = len(_get_active_document_ids(state))
        selected_document_count = len(_get_retrieval_document_ids(state))
        query_text = str(
            query_analysis.get("query_text") or _get_latest_user_text(state)
        ).strip()
        retrieval_hits = list(
            (latest_search_payload or {}).get("retrievals")
            or state.get("retrieval_hits")
            or []
        )
        expanded_context_blocks = _resolve_context_blocks(
            state=state,
            retrieval_hits=retrieval_hits,
            context_window_loader=context_window_loader,
        )
        has_any_tool_result = bool(_iter_tool_messages(state, current_turn_only=True))
        answer_draft = str(state.get("answer_draft") or "").strip()
        followup_state_updates = _build_followup_state_updates(
            state,
            retrieval_hits=retrieval_hits,
            retrieval_document_ids=_get_retrieval_document_ids(state),
            retrieval_tasks=_get_retrieval_tasks(state),
        )

        if not retrieval_hits:
            if (
                query_kind in {"lexical", "document_grounded", "ambiguous"}
                and (has_any_tool_result or deep_retrieval_attempted)
            ):
                clarification_payload = _build_missing_evidence_clarification_payload(
                    state,
                    query_text=query_text,
                    selected_document_count=selected_document_count,
                )
                interrupt_metadata = _build_interrupt_metadata(
                    state,
                    retrieval_hits=retrieval_hits,
                    final_log_entry="grounding_check:clarify:missing_evidence",
                )
                return {
                    "retrieval_hits": retrieval_hits,
                    "expanded_context_blocks": expanded_context_blocks,
                    "grounding_decision": {
                        "action": "clarify",
                        "clarification_question": clarification_payload["question"],
                    },
                    "clarification_payload": clarification_payload,
                    "clarification_response": None,
                    "needs_clarification": True,
                    **interrupt_metadata,
                    **followup_state_updates,
                    "logs": ["grounding_check:clarify:missing_evidence"],
                }
            if query_kind in {
                "smalltalk",
                "conversation_memory",
                "open_domain_unrelated",
            } and answer_draft:
                decision: GroundingDecisionPayload = {
                    "action": "answer",
                    "clarification_question": None,
                }
                clarification_payload = None
                decision_source = "deterministic"
            else:
                result = structured_grounding_llm.invoke(
                    [
                        SystemMessage(content=build_stage5_grounding_system_prompt()),
                        HumanMessage(
                            content=build_stage5_grounding_user_prompt(
                                query_text=query_text,
                                answer_draft=answer_draft or None,
                                context_blocks=[],
                                selection_type=str(
                                    query_analysis.get("selection_type") or ""
                                ).strip()
                                or None,
                                available_document_count=available_document_count,
                                selected_document_count=selected_document_count,
                                deep_retrieval_attempted=deep_retrieval_attempted,
                            )
                        ),
                    ]
                )
                decision = result.model_dump()
                clarification_payload = None
                if result.action == "clarify":
                    clarification_payload = {
                        "kind": "clarification",
                        "question": (
                            result.clarification_question
                            or str(
                                query_analysis.get("clarification_question") or ""
                            ).strip()
                            or "어떤 문서를 기준으로 답할까요?"
                        ),
                        "reason": "질문 대상 문서나 범위를 먼저 확정해야 합니다.",
                        "options": [],
                    }
                decision_source = "llm"
            interrupt_metadata = (
                _build_interrupt_metadata(
                    state,
                    retrieval_hits=retrieval_hits,
                    final_log_entry=(
                        f"grounding_check:{decision.get('action') or 'answer'}:{decision_source}"
                    ),
                )
                if decision.get("action") == "clarify"
                else {}
            )
            return {
                "retrieval_hits": retrieval_hits,
                "expanded_context_blocks": expanded_context_blocks,
                "grounding_decision": decision,
                "clarification_payload": clarification_payload,
                "clarification_response": None
                if decision.get("action") == "clarify"
                else None,
                "needs_clarification": decision.get("action") == "clarify",
                **interrupt_metadata,
                **followup_state_updates,
                "logs": [f"grounding_check:{decision.get('action') or 'answer'}:{decision_source}"],
            }

        if (
            has_any_tool_result
            and answer_draft
            and query_kind in {"lexical", "document_grounded"}
            and _answer_draft_signals_insufficient(answer_draft)
        ):
            if deep_retrieval_attempted:
                clarification_payload = _build_missing_evidence_clarification_payload(
                    state,
                    query_text=query_text,
                    selected_document_count=selected_document_count,
                )
                interrupt_metadata = _build_interrupt_metadata(
                    state,
                    retrieval_hits=retrieval_hits,
                    final_log_entry="grounding_check:clarify:insufficient_after_deep",
                )
                return {
                    "retrieval_hits": retrieval_hits,
                    "expanded_context_blocks": expanded_context_blocks,
                    "grounding_decision": {
                        "action": "clarify",
                        "clarification_question": clarification_payload["question"],
                    },
                    "clarification_payload": clarification_payload,
                    "clarification_response": None,
                    "needs_clarification": True,
                    **interrupt_metadata,
                    **followup_state_updates,
                    "logs": ["grounding_check:clarify:insufficient_after_deep"],
                }
            return {
                "retrieval_hits": retrieval_hits,
                "expanded_context_blocks": expanded_context_blocks,
                "grounding_decision": {
                    "action": "retrieve_deeper",
                    "clarification_question": None,
                },
                "clarification_payload": None,
                "needs_clarification": False,
                **followup_state_updates,
                "logs": ["grounding_check:retrieve_deeper:deterministic"],
            }

        result = structured_grounding_llm.invoke(
            [
                SystemMessage(content=build_stage5_grounding_system_prompt()),
                HumanMessage(
                    content=build_stage5_grounding_user_prompt(
                        query_text=query_text,
                        answer_draft=answer_draft or None,
                        context_blocks=expanded_context_blocks,
                        selection_type=str(
                            query_analysis.get("selection_type") or ""
                        ).strip()
                        or None,
                        available_document_count=available_document_count,
                        selected_document_count=selected_document_count,
                        deep_retrieval_attempted=deep_retrieval_attempted,
                    )
                ),
            ]
        )
        if result.action == "retrieve_deeper" and deep_retrieval_attempted:
            clarification_payload = _build_missing_evidence_clarification_payload(
                state,
                query_text=query_text,
                selected_document_count=selected_document_count,
            )
            interrupt_metadata = _build_interrupt_metadata(
                state,
                retrieval_hits=retrieval_hits,
                final_log_entry="grounding_check:clarify:deeper_exhausted",
            )
            return {
                "retrieval_hits": retrieval_hits,
                "expanded_context_blocks": expanded_context_blocks,
                "grounding_decision": {
                    "action": "clarify",
                    "clarification_question": clarification_payload["question"],
                },
                "needs_clarification": True,
                "clarification_payload": clarification_payload,
                "clarification_response": None,
                **interrupt_metadata,
                **followup_state_updates,
                "logs": ["grounding_check:clarify:deeper_exhausted"],
            }
        decision = result.model_dump()
        clarification_payload = None
        if result.action == "clarify":
            clarification_payload = {
                "kind": "clarification",
                "question": (
                    result.clarification_question
                    or "질문 범위를 더 구체적으로 알려주세요."
                ),
                "reason": "질문 대상 문서나 범위를 먼저 확정해야 합니다.",
                "options": [],
            }
        interrupt_metadata = (
            _build_interrupt_metadata(
                state,
                retrieval_hits=retrieval_hits,
                final_log_entry=f"grounding_check:{result.action}:llm",
            )
            if result.action == "clarify"
            else {}
        )

        return {
            "retrieval_hits": retrieval_hits,
            "expanded_context_blocks": expanded_context_blocks,
            "grounding_decision": decision,
            "needs_clarification": result.action == "clarify",
            "clarification_payload": clarification_payload,
            "clarification_response": None if result.action == "clarify" else None,
            **interrupt_metadata,
            **followup_state_updates,
            "logs": [f"grounding_check:{result.action}:llm"],
        }

    return grounding_check


def build_fallback_or_retrieve_deeper_node(
    *,
    retrieval_runner: Any = default_search_thread_knowledge,
    context_window_loader: Any | None = None,
):
    """deeper retrieval을 deterministic하게 수행하는 노드를 생성한다."""
    resolved_retrieval_runner = retrieval_runner or default_search_thread_knowledge

    def fallback_or_retrieve_deeper(state: ChatbotState) -> dict[str, Any]:
        query_analysis = dict(state.get("query_analysis") or {})
        retrieval_policy = dict(state.get("retrieval_policy") or {})
        query_text = str(
            query_analysis.get("query_text") or _get_latest_user_text(state)
        ).strip()
        base_mode = (
            str(retrieval_policy.get("mode") or STAGE5_DEFAULT_RETRIEVAL_MODE).strip()
            or STAGE5_DEFAULT_RETRIEVAL_MODE
        )
        active_document_ids = _get_retrieval_document_ids(state)
        retrieval_tasks = _get_retrieval_tasks(state)
        use_per_document_search = bool(state.get("use_per_document_search")) and len(
            active_document_ids
        ) > 1
        retrieval_document_queries = _get_retrieval_document_queries(state)
        base_top_k = int(retrieval_policy.get("top_k") or STAGE5_DEFAULT_TOP_K)
        deep_top_k = max(base_top_k, STAGE5_DEEP_RETRIEVAL_TOP_K)
        deep_fetch_k = max(
            STAGE5_DEEP_RETRIEVAL_FETCH_K,
            deep_top_k * (2 if (use_per_document_search or len(retrieval_tasks) > 1) else 1),
        )
        thread_id = str(state.get("thread_id") or "").strip() or None
        collection_name = str(state.get("collection_name") or "").strip() or None
        if collection_name is None and thread_id:
            collection_name = build_thread_collection_name(thread_id)

        retrieval_query = query_text
        if len(active_document_ids) == 1:
            retrieval_query = (
                retrieval_document_queries.get(active_document_ids[0]) or query_text
            )

        result = resolved_retrieval_runner(
            query=retrieval_query,
            thread_id=thread_id,
            active_document_ids=active_document_ids,
            retrieval_tasks=retrieval_tasks,
            document_queries=retrieval_document_queries,
            collection_name=collection_name,
            retrieval_mode=base_mode,
            top_k=deep_top_k,
            fetch_k=deep_fetch_k,
            dense_fetch_k=deep_fetch_k,
            bm25_fetch_k=deep_fetch_k,
            use_per_document_search=use_per_document_search,
            per_document_top_k=(
                STAGE5_MULTI_DOC_PER_DOCUMENT_TOP_K
                if use_per_document_search
                else None
            ),
            enable_rerank=True,
            enable_mmr=bool(retrieval_policy.get("enable_mmr", False)),
            score_threshold=retrieval_policy.get("score_threshold"),
        )
        retrieval_hits = list(result.get("retrievals") or [])
        if not retrieval_hits:
            return {
                "retrieval_hits": [],
                "expanded_context_blocks": [],
                "deep_retrieval_attempted": True,
                "answer_draft": "현재 연결된 문서에서 관련 근거를 찾지 못했습니다.",
                "logs": [f"fallback_or_retrieve_deeper:empty:{base_mode}"],
            }

        return {
            "retrieval_hits": retrieval_hits,
            "expanded_context_blocks": _resolve_context_blocks(
                state=state,
                retrieval_hits=retrieval_hits,
                context_window_loader=context_window_loader,
            ),
            "deep_retrieval_attempted": True,
            "answer_draft": None,
            "logs": [
                f"fallback_or_retrieve_deeper:retrieved:{base_mode}:{len(retrieval_hits)}"
            ],
        }

    return fallback_or_retrieve_deeper


def build_compose_answer_with_citations_node(
    *,
    llm: Any,
):
    """최종 grounded answer 생성 노드를 생성한다."""
    structured_answer_llm = llm.with_structured_output(FinalAnswerResult)

    def compose_answer_with_citations(state: ChatbotState) -> dict[str, Any]:
        retrieval_hits = list(state.get("retrieval_hits") or [])
        citations = _build_citations(retrieval_hits)
        evidence_chunks = _build_evidence_chunks(retrieval_hits)
        visual_asset_refs = _select_inline_visual_asset_refs(retrieval_hits)
        answer_draft = str(state.get("answer_draft") or "").strip()

        if retrieval_hits:
            query_analysis = dict(state.get("query_analysis") or {})
            query_text = str(
                query_analysis.get("query_text") or _get_latest_user_text(state)
            ).strip()
            response = structured_answer_llm.invoke(
                [
                    SystemMessage(content=build_stage5_answer_system_prompt()),
                    HumanMessage(
                        content=build_stage5_answer_user_prompt(
                            query_text=query_text,
                            context_blocks=list(
                                state.get("expanded_context_blocks") or []
                            ),
                        )
                    ),
                ]
            )
            if response.grounded and response.answer.strip():
                final_answer = response.answer.strip()
            else:
                final_answer = "현재 연결된 문서에서 질문에 답할 수 있는 근거를 찾지 못했습니다."
        elif answer_draft:
            final_answer = answer_draft
        else:
            final_answer = "현재 연결된 문서에서 질문에 답할 수 있는 근거를 찾지 못했습니다."

        updates: dict[str, Any] = {
            "final_answer": final_answer,
            "citations": citations,
            "evidence_chunks": evidence_chunks,
            "visual_asset_refs": visual_asset_refs,
            **_build_followup_state_updates(
                state,
                retrieval_hits=retrieval_hits,
                retrieval_document_ids=_get_retrieval_document_ids(state),
                retrieval_tasks=_get_retrieval_tasks(state),
                visual_asset_refs=visual_asset_refs,
            ),
            "debug_trace": _build_debug_trace(
                state,
                final_log_entries=["compose_answer_with_citations"],
            ),
            "logs": ["compose_answer_with_citations"],
        }
        updates["messages"] = [
            AIMessage(
                content=final_answer,
                name="stage5_final_answer",
                additional_kwargs=_build_thread_chat_metadata(
                    citations=citations,
                    evidence_chunks=evidence_chunks,
                    visual_asset_refs=visual_asset_refs,
                    retrieval_mode=(
                        str(
                            ((_extract_latest_search_payload(state) or {}).get("retrieval_mode"))
                            or ((state.get("retrieval_policy") or {}).get("mode"))
                            or ""
                        ).strip()
                        or None
                    ),
                    debug_trace=updates["debug_trace"],
                ),
            )
        ]
        return updates

    return compose_answer_with_citations


def route_after_classification(state: ChatbotState) -> str:
    """질문 분류 결과에 따라 다음 노드를 고른다."""
    answer_strategy = str(
        (state.get("query_analysis") or {}).get("answer_strategy") or ""
    ).strip()
    if answer_strategy in {"conversation_memory", "direct"}:
        return "respond_without_documents"
    query_kind = str((state.get("query_analysis") or {}).get("query_kind") or "")
    if query_kind in {
        "smalltalk",
        "conversation_memory",
        "open_domain_unrelated",
    }:
        return "respond_without_documents"
    return "run_retrieval"


def route_after_intent_classification(state: ChatbotState) -> str:
    """초기 intent 분류 결과에 따라 바로 답할지 retrieval planning으로 갈지 고른다."""
    answer_strategy = str(
        (state.get("query_analysis") or {}).get("answer_strategy") or ""
    ).strip()
    if answer_strategy in {"conversation_memory", "direct"}:
        return "respond_without_documents"
    return "plan_retrieval"


def route_after_agent(state: ChatbotState) -> str:
    """마지막 AIMessage에 tool call이 있으면 ToolNode로 보낸다."""
    messages = list(state.get("messages") or [])
    if not messages:
        return "grounding_check"
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None) or []
    if tool_calls:
        return "tools"
    return "grounding_check"


def route_after_grounding(state: ChatbotState) -> str:
    """근거 충분 여부에 따라 deeper retrieval 또는 답변 마무리로 보낸다."""
    decision = state.get("grounding_decision") or {}
    action = str(decision.get("action") or "").strip()
    if action == "clarify":
        return "clarify_if_needed"
    if action == "retrieve_deeper":
        if bool(state.get("deep_retrieval_attempted")):
            return "compose_answer_with_citations"
        return "fallback_or_retrieve_deeper"
    return "compose_answer_with_citations"
