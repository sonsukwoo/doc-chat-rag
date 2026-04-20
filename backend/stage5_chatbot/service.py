"""Stage-5 chatbot service entrypoint."""

from __future__ import annotations

from typing import Any

from langgraph.types import Command

from backend.app_db import (
    load_expanded_context_blocks,
    load_visual_assets,
    try_load_thread_runtime_context,
)
from backend.stage4_retrieval import search_thread_knowledge
from backend.thread_identity import build_thread_collection_name

from .checkpointer import stage5_checkpointer_context
from .graph import build_graph
from .llm import get_agent_model
from .schemas import Stage5Input, Stage5Output
from .tools import build_stage5_tools


def _normalize_interrupt(payload: Any) -> dict[str, Any] | None:
    if payload is None:
        return None
    if isinstance(payload, dict):
        return payload
    value = getattr(payload, "value", None)
    if isinstance(value, dict):
        return value
    return {"kind": "clarification", "question": str(value or payload)}


def _resolve_active_interrupt(result: dict[str, Any]) -> dict[str, Any] | None:
    needs_clarification = bool(result.get("needs_clarification"))
    if not needs_clarification:
        return None

    clarification_payload = _normalize_interrupt(result.get("clarification_payload"))
    if clarification_payload is not None:
        return clarification_payload

    raw_interrupts = list(result.get("__interrupt__") or [])
    if not raw_interrupts:
        clarification_response = str(result.get("clarification_response") or "").strip()
        if clarification_response:
            return None
        return None
    return _normalize_interrupt(raw_interrupts[0])


def _dedupe_asset_refs(citations: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in citations:
        asset_ref = str(item.get("asset_ref") or "").strip()
        if not asset_ref or asset_ref in seen:
            continue
        seen.add(asset_ref)
        ordered.append(asset_ref)
    return ordered


def run_stage5_chatbot(
    inputs: Stage5Input,
    *,
    checkpointer: object | None = None,
    llm: Any | None = None,
    stage4_runner: Any | None = None,
    resume_value: Any | None = None,
) -> Stage5Output:
    """stage5 챗봇 그래프를 1회 실행하고 결과를 정규화한다."""
    thread_id = str(inputs.get("thread_id") or "").strip()
    if not thread_id:
        raise ValueError("thread_id is required")
    thread_context = try_load_thread_runtime_context(thread_id)
    active_document_ids = [
        str(item)
        for item in (
            inputs.get("active_document_ids")
            or (thread_context or {}).get("active_document_ids")
            or []
        )
        if str(item)
    ]
    collection_name = str(
        inputs.get("collection_name")
        or (thread_context or {}).get("collection_name")
        or ""
    ).strip() or None
    if collection_name is None:
        collection_name = build_thread_collection_name(thread_id)
    retrieval_mode = str(
        inputs.get("thread_default_retrieval_mode")
        or (thread_context or {}).get("default_retrieval_mode")
        or "dense"
    ).strip() or "dense"
    document_profiles = [
        dict(item)
        for item in (
            inputs.get("document_profiles")
            or (thread_context or {}).get("document_profiles")
            or []
        )
        if isinstance(item, dict)
    ]
    allow_web_search = bool(inputs.get("allow_web_search"))
    context_window_loader = (
        inputs.get("_context_window_loader") or load_expanded_context_blocks
    )
    visual_asset_loader = inputs.get("_visual_asset_loader") or load_visual_assets

    tools = build_stage5_tools(
        allow_web_search=allow_web_search,
        stage4_runner=stage4_runner
        or (
            lambda **kwargs: search_thread_knowledge(
                query=kwargs["query"],
                thread_id=kwargs.get("thread_id"),
                active_document_ids=kwargs.get("active_document_ids"),
                document_queries=kwargs.get("document_queries"),
                collection_name=kwargs.get("collection_name"),
                retrieval_mode=kwargs.get("retrieval_mode"),
                top_k=kwargs.get("top_k"),
                use_per_document_search=kwargs.get("use_per_document_search"),
                score_threshold=kwargs.get("score_threshold"),
                enable_rerank=kwargs.get("enable_rerank"),
                enable_mmr=kwargs.get("enable_mmr"),
            )
        ),
        context_window_loader=context_window_loader,
        visual_asset_loader=visual_asset_loader,
    )
    config = {"configurable": {"thread_id": thread_id}}
    public_inputs = {
        key: value
        for key, value in dict(inputs).items()
        if not str(key).startswith("_")
    }
    graph_inputs = {
        **public_inputs,
        "thread_id": thread_id,
        "thread_name": str(
            inputs.get("thread_name")
            or (thread_context or {}).get("thread_name")
            or ""
        ).strip()
        or None,
        "thread_default_retrieval_mode": retrieval_mode,
        "active_document_ids": active_document_ids,
        "document_profiles": document_profiles,
        "collection_name": collection_name,
        "allow_web_search": allow_web_search,
    }

    graph_command: Any
    if resume_value is None:
        graph_command = graph_inputs
    else:
        # interrupt 재개 시에는 기존 thread 상태를 그대로 이어받고,
        # 사용자의 응답만 Command(resume=...)로 전달한다.
        graph_command = Command(resume=resume_value)

    if checkpointer is not None:
        graph = build_graph(
            checkpointer=checkpointer,
            tools=tools,
            llm=llm or get_agent_model(),
            retrieval_runner=stage4_runner or search_thread_knowledge,
            context_window_loader=context_window_loader,
        )
        result = graph.invoke(graph_command, config=config)
    else:
        with stage5_checkpointer_context() as managed_checkpointer:
            graph = build_graph(
                checkpointer=managed_checkpointer,
                tools=tools,
                llm=llm or get_agent_model(),
                retrieval_runner=stage4_runner or search_thread_knowledge,
                context_window_loader=context_window_loader,
            )
            result = graph.invoke(graph_command, config=config)

    interrupt_payload = _resolve_active_interrupt(result)

    citations = list(result.get("citations") or [])
    asset_refs = [
        str(item).strip()
        for item in result.get("visual_asset_refs") or []
        if str(item).strip()
    ]
    if not asset_refs:
        asset_refs = _dedupe_asset_refs(citations)
    visual_assets = []
    if asset_refs:
        visual_assets = visual_asset_loader(
            thread_id=thread_id,
            active_document_ids=active_document_ids,
            asset_refs=asset_refs,
        )

    return {
        "status": "interrupted" if interrupt_payload else "completed",
        "thread_id": thread_id,
        "final_answer": result.get("final_answer"),
        "citations": citations,
        "visual_assets": visual_assets,
        "visual_asset_refs": asset_refs,
        "evidence_chunks": list(result.get("evidence_chunks") or []),
        "interrupt": interrupt_payload,
        "retrieval_mode": str(
            (dict(result.get("debug_trace") or {}) or {}).get("executed_retrieval_mode")
            or (dict(result.get("debug_trace") or {}) or {}).get("retrieval_mode")
            or (result.get("retrieval_policy") or {}).get("mode")
            or ""
        ).strip()
        or None,
        "logs": list(result.get("logs") or []),
        "debug_trace": dict(result.get("debug_trace") or {}) or None,
    }
