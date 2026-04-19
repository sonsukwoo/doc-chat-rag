"""Stage-5 chatbot graph nodes."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.types import interrupt

from backend.stage4_retrieval import search_room_knowledge as default_search_room_knowledge

from .config import STAGE5_DEFAULT_RETRIEVAL_MODE, STAGE5_DEFAULT_TOP_K
from .models import FinalAnswerResult, GroundingCheckResult
from .prompts import (
    build_stage5_agent_system_prompt,
    build_stage5_answer_system_prompt,
    build_stage5_answer_user_prompt,
    build_stage5_grounding_system_prompt,
    build_stage5_grounding_user_prompt,
)
from .state import ChatbotState, GroundingDecisionPayload, QueryAnalysisPayload, RetrievalPolicyPayload


def _get_latest_user_text(state: ChatbotState) -> str:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, HumanMessage):
            content = message.content
            if isinstance(content, str):
                return content.strip()
    return str(state.get("user_message") or "").strip()


def _get_latest_ai_message(state: ChatbotState) -> AIMessage | None:
    for message in reversed(state.get("messages") or []):
        if isinstance(message, AIMessage):
            return message
    return None


def _iter_tool_messages(
    state: ChatbotState,
    *,
    tool_name: str | None = None,
) -> list[ToolMessage]:
    tool_messages: list[ToolMessage] = []
    for message in state.get("messages") or []:
        if not isinstance(message, ToolMessage):
            continue
        if tool_name and getattr(message, "name", None) != tool_name:
            continue
        tool_messages.append(message)
    return tool_messages


def _parse_tool_message_json(message: ToolMessage) -> dict[str, Any] | None:
    content = message.content
    if not isinstance(content, str):
        return None
    try:
        payload = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    return payload


def _extract_latest_search_payload(state: ChatbotState) -> dict[str, Any] | None:
    search_messages = _iter_tool_messages(state, tool_name="search_room_knowledge")
    for message in reversed(search_messages):
        payload = _parse_tool_message_json(message)
        if payload is not None:
            return payload
    return None


def _build_context_blocks(retrieval_hits: list[dict[str, Any]]) -> list[str]:
    context_blocks: list[str] = []
    for index, hit in enumerate(retrieval_hits, start=1):
        lines = [
            f"[근거 {index}]",
            f"document_id: {str(hit.get('document_id') or '')}",
            f"chunk_id: {str(hit.get('chunk_id') or '')}",
        ]
        if hit.get("section_title"):
            lines.append(f"section_title: {str(hit['section_title'])}")
        if hit.get("primary_page") is not None:
            lines.append(f"page: {int(hit['primary_page'])}")
        if hit.get("caption"):
            lines.append(f"caption: {str(hit['caption'])}")
        lines.append(f"text: {str(hit.get('text') or '').strip()}")
        context_blocks.append("\n".join(lines))
    return context_blocks


def _build_citations(retrieval_hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    citations = []
    for hit in retrieval_hits:
        citations.append(
            {
                "document_id": str(hit.get("document_id") or ""),
                "chunk_id": str(hit.get("chunk_id") or ""),
                "parent_id": hit.get("parent_id"),
                "page": hit.get("primary_page"),
                "section_title": hit.get("section_title"),
                "asset_relative_path": hit.get("asset_relative_path"),
            }
        )
    return citations


def _should_force_clarification(state: ChatbotState, query_text: str) -> tuple[bool, str]:
    active_document_ids = [str(item) for item in state.get("active_document_ids") or [] if str(item)]
    if not active_document_ids:
        return True, "현재 채팅방에 연결된 문서가 없습니다."

    if len(active_document_ids) > 1 and query_text:
        vague_markers = ("이 문서", "이거", "여기", "저 문서", "이 논문")
        if any(marker in query_text for marker in vague_markers):
            return True, "여러 문서가 연결된 상태에서 질문 대상 문서가 모호합니다."

    return False, ""


def _build_query_analysis(state: ChatbotState, query_text: str) -> QueryAnalysisPayload:
    needs_clarification, reason = _should_force_clarification(state, query_text)
    query_kind: QueryAnalysisPayload["query_kind"] = "general"
    lowered = query_text.lower()
    if any(token in lowered for token in ("table", "figure", "section", "appendix", "page")):
        query_kind = "lexical"
    if needs_clarification:
        query_kind = "ambiguous"
    return {
        "query_text": query_text,
        "query_kind": query_kind,
        "needs_clarification": needs_clarification,
        "reason": reason,
    }


def _build_retrieval_policy(query_analysis: QueryAnalysisPayload) -> RetrievalPolicyPayload:
    query_kind = query_analysis.get("query_kind") or "general"
    return {
        "mode": "hybrid" if query_kind == "lexical" else STAGE5_DEFAULT_RETRIEVAL_MODE,
        "use_rerank": True,
        "use_web_search": False,
        "top_k": STAGE5_DEFAULT_TOP_K,
    }


def load_request_context(state: ChatbotState) -> dict[str, Any]:
    """외부 입력을 챗봇 state 기본 구조에 맞게 정리한다."""
    updates: dict[str, Any] = {
        "logs": ["load_request_context"],
    }
    if not state.get("messages") and state.get("user_message"):
        updates["messages"] = [HumanMessage(content=str(state["user_message"]).strip())]
    if not state.get("thread_id") and state.get("room_id"):
        updates["thread_id"] = str(state["room_id"])
    return updates


def classify_query(state: ChatbotState) -> dict[str, Any]:
    """질문 성격과 retrieval 기본 정책을 결정한다."""
    query_text = _get_latest_user_text(state)
    query_analysis = _build_query_analysis(state, query_text)
    retrieval_policy = _build_retrieval_policy(query_analysis)
    return {
        "query_analysis": query_analysis,
        "retrieval_policy": retrieval_policy,
        "needs_clarification": bool(query_analysis.get("needs_clarification")),
        "logs": [f"classify_query:{query_analysis.get('query_kind') or 'general'}"],
    }


def clarify_if_needed(state: ChatbotState) -> dict[str, Any]:
    """문서 범위가 모호하면 사용자 clarification을 요청한다."""
    query_analysis = state.get("query_analysis") or {}
    options = [str(item) for item in state.get("active_document_ids") or [] if str(item)]
    payload = dict(state.get("clarification_payload") or {})
    if not payload:
        payload = {
            "kind": "clarification",
            "question": "어떤 문서를 기준으로 답할까요?",
            "reason": str(query_analysis.get("reason") or "질문 범위를 확정해야 합니다."),
            "options": options,
        }
    payload_options = [str(item) for item in payload.get("options") or [] if str(item)]
    response = interrupt(payload)
    normalized_response = str(response or "").strip() or None
    updated_document_ids = options
    resumed_messages: list[Any] = []
    updated_user_message = state.get("user_message")
    if normalized_response and normalized_response in payload_options:
        updated_document_ids = [normalized_response]
    elif normalized_response:
        combined_query = (
            f"{str(state.get('user_message') or _get_latest_user_text(state)).strip()}\n\n"
            f"추가 정보:\n{normalized_response}"
        ).strip()
        resumed_messages = [HumanMessage(content=combined_query)]
        updated_user_message = combined_query
    return {
        "clarification_payload": payload,
        "clarification_response": normalized_response,
        "active_document_ids": updated_document_ids,
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
        system_prompt = build_stage5_agent_system_prompt(
            active_document_ids=[
                str(item)
                for item in state.get("active_document_ids") or []
                if str(item)
            ],
            retrieval_mode=str(
                retrieval_policy.get("mode") or STAGE5_DEFAULT_RETRIEVAL_MODE
            ),
        )
        response = bound_llm.invoke(
            [
                SystemMessage(content=system_prompt),
                *list(state.get("messages") or []),
            ]
        )
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


def build_grounding_check_node(
    *,
    llm: Any,
):
    """retrieval hit가 있을 때만 구조화된 LLM 판단을 수행하는 grounding 노드를 만든다."""
    structured_grounding_llm = llm.with_structured_output(GroundingCheckResult)

    def grounding_check(state: ChatbotState) -> dict[str, Any]:
        latest_search_payload = _extract_latest_search_payload(state)
        retrieval_hits = list(
            (latest_search_payload or {}).get("retrievals")
            or state.get("retrieval_hits")
            or []
        )
        expanded_context_blocks = _build_context_blocks(retrieval_hits)
        has_any_tool_result = bool(_iter_tool_messages(state))
        answer_draft = str(state.get("answer_draft") or "").strip()

        if not retrieval_hits:
            enough_evidence = False
            reason = "grounded retrieval evidence is missing"
            if has_any_tool_result and answer_draft:
                enough_evidence = True
                reason = "non-search tool result answered the request"

            decision: GroundingDecisionPayload = {
                "enough_evidence": enough_evidence,
                "needs_deeper_retrieval": not enough_evidence,
                "needs_clarification": False,
                "clarification_question": None,
                "missing_aspects": [],
            }
            return {
                "retrieval_hits": retrieval_hits,
                "expanded_context_blocks": expanded_context_blocks,
                "grounding_decision": decision,
                "needs_clarification": False,
                "logs": [
                    f"grounding_check:{'enough' if enough_evidence else 'insufficient'}:deterministic"
                ],
            }

        query_analysis = dict(state.get("query_analysis") or {})
        query_text = str(
            query_analysis.get("query_text") or _get_latest_user_text(state)
        ).strip()
        result = structured_grounding_llm.invoke(
            [
                SystemMessage(content=build_stage5_grounding_system_prompt()),
                HumanMessage(
                    content=build_stage5_grounding_user_prompt(
                        query_text=query_text,
                        answer_draft=answer_draft or None,
                        context_blocks=expanded_context_blocks,
                    )
                ),
            ]
        )
        decision = result.model_dump()
        clarification_payload = None
        if result.needs_clarification:
            missing_aspects = [item.strip() for item in result.missing_aspects if item.strip()]
            clarification_reason = (
                f"추가 확인이 필요한 항목: {', '.join(missing_aspects)}"
                if missing_aspects
                else "질문 범위를 더 구체적으로 확인해야 합니다."
            )
            clarification_payload = {
                "kind": "clarification",
                "question": (
                    result.clarification_question
                    or "질문 범위를 더 구체적으로 알려주세요."
                ),
                "reason": clarification_reason,
                "options": [],
            }

        return {
            "retrieval_hits": retrieval_hits,
            "expanded_context_blocks": expanded_context_blocks,
            "grounding_decision": decision,
            "needs_clarification": bool(result.needs_clarification),
            "clarification_payload": clarification_payload,
            "logs": [
                (
                    "grounding_check:"
                    f"{'enough' if result.enough_evidence else 'insufficient'}:"
                    "llm"
                )
            ],
        }

    return grounding_check


def build_fallback_or_retrieve_deeper_node(
    *,
    retrieval_runner: Any = default_search_room_knowledge,
):
    """deeper retrieval을 deterministic하게 수행하는 노드를 생성한다."""
    resolved_retrieval_runner = retrieval_runner or default_search_room_knowledge

    def fallback_or_retrieve_deeper(state: ChatbotState) -> dict[str, Any]:
        query_analysis = dict(state.get("query_analysis") or {})
        retrieval_policy = dict(state.get("retrieval_policy") or {})
        query_text = str(query_analysis.get("query_text") or _get_latest_user_text(state)).strip()
        query_kind = str(query_analysis.get("query_kind") or "general").strip()
        base_mode = str(retrieval_policy.get("mode") or STAGE5_DEFAULT_RETRIEVAL_MODE).strip() or STAGE5_DEFAULT_RETRIEVAL_MODE
        deep_mode = "hybrid" if query_kind == "lexical" else base_mode

        result = resolved_retrieval_runner(
            query=query_text,
            room_id=str(state.get("room_id") or "").strip() or None,
            active_document_ids=[
                str(item)
                for item in state.get("active_document_ids") or []
                if str(item)
            ],
            collection_name=str(state.get("collection_name") or "").strip() or None,
            retrieval_mode=deep_mode,
            top_k=int(retrieval_policy.get("top_k") or STAGE5_DEFAULT_TOP_K),
            enable_rerank=bool(retrieval_policy.get("use_rerank", True)),
            enable_mmr=False,
            score_threshold=None,
        )
        retrieval_hits = list(result.get("retrievals") or [])
        if not retrieval_hits:
            return {
                "retrieval_hits": [],
                "expanded_context_blocks": [],
                "answer_draft": "현재 연결된 문서에서 관련 근거를 찾지 못했습니다.",
                "logs": [f"fallback_or_retrieve_deeper:empty:{deep_mode}"],
            }

        return {
            "retrieval_hits": retrieval_hits,
            "expanded_context_blocks": _build_context_blocks(retrieval_hits),
            "answer_draft": None,
            "logs": [f"fallback_or_retrieve_deeper:retrieved:{deep_mode}:{len(retrieval_hits)}"],
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
        answer_draft = str(state.get("answer_draft") or "").strip()

        if answer_draft and retrieval_hits:
            final_answer = answer_draft
        elif answer_draft and not retrieval_hits:
            final_answer = answer_draft
        elif retrieval_hits:
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
                            context_blocks=list(state.get("expanded_context_blocks") or []),
                        )
                    ),
                ]
            )
            if response.grounded and response.answer.strip():
                final_answer = response.answer.strip()
            else:
                final_answer = "현재 연결된 문서에서 질문에 답할 수 있는 근거를 찾지 못했습니다."
        else:
            final_answer = "현재 연결된 문서에서 질문에 답할 수 있는 근거를 찾지 못했습니다."

        updates: dict[str, Any] = {
            "final_answer": final_answer,
            "citations": citations,
            "logs": ["compose_answer_with_citations"],
        }
        latest_ai_message = _get_latest_ai_message(state)
        latest_ai_text = (
            latest_ai_message.content.strip()
            if latest_ai_message is not None and isinstance(latest_ai_message.content, str)
            else None
        )
        if latest_ai_text != final_answer:
            updates["messages"] = [
                AIMessage(
                    content=final_answer,
                    name="stage5_final_answer",
                )
            ]
        return updates

    return compose_answer_with_citations


def route_after_classification(state: ChatbotState) -> str:
    """clarification 필요 여부에 따라 다음 노드를 고른다."""
    if state.get("needs_clarification"):
        return "clarify_if_needed"
    return "agent_llm"


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
    if bool(decision.get("needs_clarification")) or bool(state.get("needs_clarification")):
        return "clarify_if_needed"
    if not bool(decision.get("enough_evidence")):
        return "fallback_or_retrieve_deeper"
    return "compose_answer_with_citations"
