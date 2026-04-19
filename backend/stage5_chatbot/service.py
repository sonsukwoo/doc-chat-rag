"""Stage-5 chatbot service entrypoint."""

from __future__ import annotations

from typing import Any

from backend.app_db import try_load_room_runtime_context
from backend.stage4_retrieval import search_room_knowledge

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


def run_stage5_chatbot(
    inputs: Stage5Input,
    *,
    checkpointer: object | None = None,
    llm: Any | None = None,
    stage4_runner: Any | None = None,
) -> Stage5Output:
    """stage5 챗봇 그래프를 1회 실행하고 결과를 정규화한다."""
    room_id = str(inputs.get("room_id") or "").strip() or "default-room"
    thread_id = str(inputs.get("thread_id") or room_id).strip() or "default-thread"
    room_context = try_load_room_runtime_context(room_id)
    active_document_ids = [
        str(item)
        for item in (
            inputs.get("active_document_ids")
            or (room_context or {}).get("active_document_ids")
            or []
        )
        if str(item)
    ]
    collection_name = str(
        inputs.get("collection_name")
        or (room_context or {}).get("collection_name")
        or ""
    ).strip() or None
    retrieval_mode = str(
        (room_context or {}).get("default_retrieval_mode") or "dense"
    ).strip() or "dense"
    allow_web_search = bool(inputs.get("allow_web_search"))

    tools = build_stage5_tools(
        allow_web_search=allow_web_search,
        stage4_runner=stage4_runner
        or (lambda *, query, room_id, active_document_ids, collection_name=None, retrieval_mode=None: search_room_knowledge(
            query=query,
            room_id=room_id,
            active_document_ids=active_document_ids,
            collection_name=collection_name,
            retrieval_mode=retrieval_mode,
        )),
    )
    config = {"configurable": {"thread_id": thread_id}}
    graph_inputs = {
        **dict(inputs),
        "room_id": room_id,
        "thread_id": thread_id,
        "active_document_ids": active_document_ids,
        "collection_name": collection_name,
        "allow_web_search": allow_web_search,
    }

    if checkpointer is not None:
        graph = build_graph(
            checkpointer=checkpointer,
            tools=tools,
            llm=llm or get_agent_model(),
            retrieval_runner=stage4_runner or search_room_knowledge,
        )
        result = graph.invoke(graph_inputs, config=config)
    else:
        with stage5_checkpointer_context() as managed_checkpointer:
            graph = build_graph(
                checkpointer=managed_checkpointer,
                tools=tools,
                llm=llm or get_agent_model(),
                retrieval_runner=stage4_runner or search_room_knowledge,
            )
            result = graph.invoke(graph_inputs, config=config)

    interrupt_payload = None
    raw_interrupts = result.get("__interrupt__") or []
    if raw_interrupts:
        interrupt_payload = _normalize_interrupt(raw_interrupts[0])

    return {
        "status": "interrupted" if interrupt_payload else "completed",
        "room_id": room_id,
        "thread_id": thread_id,
        "final_answer": result.get("final_answer"),
        "citations": list(result.get("citations") or []),
        "interrupt": interrupt_payload,
        "retrieval_mode": str(
            (result.get("retrieval_policy") or {}).get("mode") or ""
        ),
        "logs": list(result.get("logs") or []),
    }
