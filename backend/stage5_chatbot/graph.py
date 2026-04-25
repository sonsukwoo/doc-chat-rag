"""Stage-5 chatbot graph builder."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from .llm import get_agent_model
from .nodes import (
    build_agent_llm_node,
    build_classify_intent_node,
    build_compose_answer_with_citations_node,
    build_direct_response_node,
    build_fallback_or_retrieve_deeper_node,
    build_grounding_check_node,
    build_plan_retrieval_node,
    build_run_retrieval_node,
    clarify_if_needed,
    load_request_context,
    route_after_agent,
    route_after_classification,
    route_after_intent_classification,
    route_after_grounding,
)
from .state import ChatbotState
from .tools import build_stage5_tools


def build_graph(
    *,
    checkpointer: object | None = None,
    tools: list[Any] | None = None,
    llm: Any | None = None,
    retrieval_runner: Any | None = None,
    context_window_loader: Any | None = None,
    overview_loader: Any | None = None,
) -> Any:
    """stage5 챗봇 그래프를 조립한다."""
    builder = StateGraph(ChatbotState)
    resolved_tools = tools or build_stage5_tools()
    resolved_llm = llm or get_agent_model()

    builder.add_node("load_request_context", load_request_context)
    builder.add_node("classify_intent", build_classify_intent_node(llm=resolved_llm))
    builder.add_node("plan_retrieval", build_plan_retrieval_node(llm=resolved_llm))
    builder.add_node("clarify_if_needed", clarify_if_needed)
    builder.add_node(
        "respond_without_documents",
        build_direct_response_node(llm=resolved_llm),
    )
    builder.add_node(
        "agent_llm",
        build_agent_llm_node(llm=resolved_llm, tools=resolved_tools),
    )
    builder.add_node(
        "run_retrieval",
        build_run_retrieval_node(
            retrieval_runner=retrieval_runner,
            overview_loader=overview_loader,
        ),
    )
    builder.add_node("tools", ToolNode(resolved_tools))
    builder.add_node(
        "grounding_check",
        build_grounding_check_node(
            llm=resolved_llm,
            context_window_loader=context_window_loader,
        ),
    )
    builder.add_node(
        "fallback_or_retrieve_deeper",
        build_fallback_or_retrieve_deeper_node(
            retrieval_runner=retrieval_runner,
            context_window_loader=context_window_loader,
        ),
    )
    builder.add_node(
        "compose_answer_with_citations",
        build_compose_answer_with_citations_node(llm=resolved_llm),
    )

    builder.add_edge(START, "load_request_context")
    builder.add_edge("load_request_context", "classify_intent")
    builder.add_conditional_edges(
        "classify_intent",
        route_after_intent_classification,
        [
            "respond_without_documents",
            "plan_retrieval",
        ],
    )
    builder.add_conditional_edges(
        "plan_retrieval",
        route_after_classification,
        [
            "respond_without_documents",
            "agent_llm",
            "run_retrieval",
        ],
    )
    builder.add_edge("clarify_if_needed", "classify_intent")
    builder.add_edge("respond_without_documents", END)
    builder.add_conditional_edges(
        "agent_llm",
        route_after_agent,
        ["tools", "grounding_check"],
    )
    builder.add_edge("tools", "agent_llm")
    builder.add_edge("run_retrieval", "grounding_check")
    builder.add_conditional_edges(
        "grounding_check",
        route_after_grounding,
        ["clarify_if_needed", "fallback_or_retrieve_deeper", "compose_answer_with_citations"],
    )
    builder.add_edge("fallback_or_retrieve_deeper", "grounding_check")
    builder.add_edge("compose_answer_with_citations", END)

    return builder.compile(checkpointer=checkpointer)


def get_agent(
    *,
    checkpointer: object | None = None,
    tools: list[Any] | None = None,
    llm: Any | None = None,
    retrieval_runner: Any | None = None,
    context_window_loader: Any | None = None,
    overview_loader: Any | None = None,
) -> Any:
    """stage5 챗봇 그래프 실행 객체를 반환한다."""
    return build_graph(
        checkpointer=checkpointer,
        tools=tools,
        llm=llm,
        retrieval_runner=retrieval_runner,
        context_window_loader=context_window_loader,
        overview_loader=overview_loader,
    )
