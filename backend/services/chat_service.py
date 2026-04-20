"""thread-scoped stage5 챗봇 실행 서비스."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from backend.app_db import load_visual_assets, try_load_thread_runtime_context
from backend.stage5_chatbot.document_selection import (
    extract_explicit_document_ids,
    iter_ordered_document_profiles,
)
from backend.stage5_chatbot import run_stage5_chatbot
from backend.stage5_chatbot.checkpointer import stage5_checkpointer_context
from backend.stage5_chatbot.schemas import (
    ChatbotCitationPayload,
    ChatbotDebugTracePayload,
    ChatbotEvidenceChunkPayload,
    ChatbotInterruptPayload,
    ChatbotVisualAssetPayload,
)
from backend.stage5_chatbot.schemas import Stage5Output

from .thread_service import ThreadPayload, get_thread_detail


class ThreadChatHistoryMessagePayload(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    kind: Literal["answer", "interrupt"]
    created_at: str | None
    citations: list[ChatbotCitationPayload]
    visual_assets: list[ChatbotVisualAssetPayload]
    evidence_chunks: list[ChatbotEvidenceChunkPayload]
    retrieval_mode: str | None
    debug_trace: ChatbotDebugTracePayload | None


class ThreadChatViewPayload(TypedDict, total=False):
    thread: ThreadPayload
    messages: list[ThreadChatHistoryMessagePayload]
    interrupt: ChatbotInterruptPayload | None
    history_notice: str | None


def _require_thread(thread_id: str) -> ThreadPayload:
    thread = get_thread_detail(thread_id)
    if thread is None:
        raise LookupError("thread not found")
    return thread


def _serialize_visible_messages(
    raw_messages: list[Any],
) -> tuple[list[ThreadChatHistoryMessagePayload], dict[int, list[str]]]:
    visible_messages: list[ThreadChatHistoryMessagePayload] = []
    visual_asset_ref_map: dict[int, list[str]] = {}

    def _normalize_thread_chat_metadata(message: Any) -> dict[str, Any]:
        metadata = dict(getattr(message, "additional_kwargs", {}) or {}).get("thread_chat")
        if not isinstance(metadata, dict):
            return {}
        return metadata

    for message in raw_messages:
        if isinstance(message, HumanMessage):
            content = str(message.content or "").strip()
            if not content:
                continue
            metadata = _normalize_thread_chat_metadata(message)
            visible_messages.append(
                {
                    "role": "user",
                    "content": content,
                    "kind": "answer",
                    "created_at": str(metadata.get("created_at") or "").strip() or None,
                    "citations": [],
                    "visual_assets": [],
                    "evidence_chunks": [],
                    "retrieval_mode": None,
                    "debug_trace": None,
                }
            )
            continue

        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if not content:
                continue
            metadata = _normalize_thread_chat_metadata(message)
            if not metadata:
                # Tool loop 안의 intermediate AI draft는 사용자 히스토리에 노출하지 않는다.
                continue
            message_kind = str(metadata.get("kind") or "answer").strip() or "answer"
            if message_kind not in {"answer", "interrupt"}:
                message_kind = "answer"
            payload: ThreadChatHistoryMessagePayload = {
                "role": "assistant",
                "content": content,
                "kind": message_kind,  # type: ignore[typeddict-item]
                "created_at": str(metadata.get("created_at") or "").strip() or None,
                "citations": list(metadata.get("citations") or []),
                "visual_assets": [],
                "evidence_chunks": list(metadata.get("evidence_chunks") or []),
                "retrieval_mode": str(metadata.get("retrieval_mode") or "").strip()
                or None,
                "debug_trace": (
                    dict(metadata.get("debug_trace") or {}) or None
                ),
            }
            visual_asset_refs = [
                str(item).strip()
                for item in metadata.get("visual_asset_refs") or []
                if str(item).strip()
            ]
            if (
                visible_messages
                and visible_messages[-1]["role"] == "assistant"
                and visible_messages[-1]["kind"] == payload["kind"]
                and visible_messages[-1]["content"] == content
            ):
                previous = visible_messages[-1]
                previous_has_metadata = bool(
                    previous.get("citations")
                    or previous.get("evidence_chunks")
                    or previous.get("debug_trace")
                )
                current_has_metadata = bool(
                    payload["citations"]
                    or payload["evidence_chunks"]
                    or payload["debug_trace"]
                )
                if current_has_metadata or not previous_has_metadata:
                    visible_messages[-1] = payload
                    if visual_asset_refs:
                        visual_asset_ref_map[len(visible_messages) - 1] = visual_asset_refs
                continue
            visible_messages.append(payload)
            if visual_asset_refs:
                visual_asset_ref_map[len(visible_messages) - 1] = visual_asset_refs

    return visible_messages, visual_asset_ref_map


def _normalize_interrupt_payload(value: Any) -> ChatbotInterruptPayload | None:
    if not isinstance(value, dict):
        return None
    question = str(value.get("question") or "").strip()
    reason = str(value.get("reason") or "").strip()
    options = [
        str(item).strip()
        for item in value.get("options") or []
        if str(item).strip()
    ]
    if not question:
        return None
    payload: ChatbotInterruptPayload = {
        "kind": "clarification",
        "question": question,
    }
    if reason:
        payload["reason"] = reason
    if options:
        payload["options"] = options
    return payload


def _resolve_active_interrupt_payload(
    *,
    needs_clarification: bool,
    clarification_response: str | None,
    clarification_payload: Any,
) -> ChatbotInterruptPayload | None:
    if not needs_clarification:
        return None
    normalized_payload = _normalize_interrupt_payload(clarification_payload)
    if normalized_payload is not None:
        return normalized_payload
    if str(clarification_response or "").strip():
        return None
    return None


def _load_visible_message_assets(
    *,
    thread_id: str,
    active_document_ids: list[str],
    visual_asset_ref_map: dict[int, list[str]],
) -> dict[int, list[ChatbotVisualAssetPayload]]:
    ordered_refs: list[str] = []
    for asset_refs in visual_asset_ref_map.values():
        for asset_ref in asset_refs:
            if asset_ref not in ordered_refs:
                ordered_refs.append(asset_ref)
    if not ordered_refs:
        return {}

    loaded_assets = load_visual_assets(
        thread_id=thread_id,
        active_document_ids=active_document_ids,
        asset_refs=ordered_refs,
    )
    assets_by_ref = {
        str(asset.get("asset_ref") or "").strip(): asset
        for asset in loaded_assets
        if str(asset.get("asset_ref") or "").strip()
    }

    return {
        message_index: [
            assets_by_ref[asset_ref]
            for asset_ref in asset_refs
            if asset_ref in assets_by_ref
        ]
        for message_index, asset_refs in visual_asset_ref_map.items()
    }


def _normalize_visual_asset_refs(value: Any) -> list[str]:
    return [
        str(item).strip()
        for item in value or []
        if str(item).strip()
    ]


def _format_interrupt_content(payload: ChatbotInterruptPayload) -> str:
    parts = [
        str(payload.get("question") or "").strip(),
        str(payload.get("reason") or "").strip(),
    ]
    return "\n\n".join(part for part in parts if part)


def _extract_checkpoint_data(checkpoint_tuple: Any) -> dict[str, Any]:
    checkpoint_data = getattr(checkpoint_tuple, "checkpoint", None)
    if checkpoint_data is None and isinstance(checkpoint_tuple, dict):
        checkpoint_data = checkpoint_tuple.get("checkpoint")
    return dict(checkpoint_data or {})


def _extract_pending_interrupt_payload(
    checkpoint_tuple: Any,
) -> ChatbotInterruptPayload | None:
    pending_writes = list(getattr(checkpoint_tuple, "pending_writes", None) or [])
    if not pending_writes and isinstance(checkpoint_tuple, dict):
        pending_writes = list(checkpoint_tuple.get("pending_writes") or [])

    for pending_write in pending_writes:
        if not isinstance(pending_write, tuple) or len(pending_write) < 3:
            continue
        _, channel_name, payloads = pending_write
        if channel_name != "__interrupt__":
            continue
        for payload in payloads or []:
            interrupt_value = getattr(payload, "value", payload)
            normalized = _normalize_interrupt_payload(interrupt_value)
            if normalized is not None:
                return normalized
    return None


def _load_checkpoint_channel_values(thread_id: str) -> dict[str, Any]:
    with stage5_checkpointer_context() as checkpointer:
        checkpoint_tuple = checkpointer.get_tuple(
            {"configurable": {"thread_id": thread_id}}
        )
    checkpoint_data = _extract_checkpoint_data(checkpoint_tuple)
    return dict(checkpoint_data.get("channel_values") or {})


def _looks_like_clarification_followup(
    *,
    thread_id: str,
    message: str,
    clarification_payload: ChatbotInterruptPayload,
) -> bool:
    normalized_message = str(message or "").strip()
    if not normalized_message:
        return False

    normalized_options = {
        str(item).strip()
        for item in clarification_payload.get("options") or []
        if str(item).strip()
    }
    if normalized_message in normalized_options:
        return True

    runtime_context = try_load_thread_runtime_context(thread_id) or {}
    ordered_profiles = iter_ordered_document_profiles(
        runtime_context.get("active_document_ids"),
        runtime_context.get("document_profiles"),
    )
    explicit_document_ids = extract_explicit_document_ids(
        normalized_message,
        ordered_profiles,
    )
    if not explicit_document_ids:
        return False

    normalized_lower = normalized_message.casefold()
    request_markers = (
        "?",
        "설명",
        "요약",
        "정리",
        "알려",
        "말해",
        "비교",
        "찾아",
        "보여",
        "무엇",
        "어디",
        "왜",
        "어떻게",
    )
    if any(marker in normalized_lower for marker in request_markers):
        return False
    return len(normalized_message) <= 40


def load_thread_chat_view(thread_id: str) -> ThreadChatViewPayload:
    """thread 기본 정보와 사람이 읽을 수 있는 채팅 기록을 반환한다."""
    thread = _require_thread(thread_id)
    visible_messages: list[ThreadChatHistoryMessagePayload] = []
    interrupt_payload: ChatbotInterruptPayload | None = None
    history_notice: str | None = None

    try:
        with stage5_checkpointer_context() as checkpointer:
            checkpoint_tuple = checkpointer.get_tuple(
                {"configurable": {"thread_id": thread["thread_id"]}}
            )
    except Exception as exc:
        raise RuntimeError("채팅 기록을 불러오지 못했습니다.") from exc

    checkpoint_data = _extract_checkpoint_data(checkpoint_tuple)
    channel_values = dict(checkpoint_data.get("channel_values") or {})
    raw_messages = list(channel_values.get("messages") or [])
    visible_messages, visual_asset_ref_map = _serialize_visible_messages(raw_messages)

    needs_clarification = bool(channel_values.get("needs_clarification"))
    clarification_response = str(channel_values.get("clarification_response") or "").strip()
    interrupt_payload = _resolve_active_interrupt_payload(
        needs_clarification=needs_clarification,
        clarification_response=clarification_response,
        clarification_payload=channel_values.get("clarification_payload"),
    )
    if interrupt_payload is None and needs_clarification and not clarification_response:
        interrupt_payload = _extract_pending_interrupt_payload(checkpoint_tuple)

    if needs_clarification and not clarification_response and interrupt_payload is not None:
        interrupt_content = _format_interrupt_content(interrupt_payload)
        synthetic_visual_asset_refs = _normalize_visual_asset_refs(
            channel_values.get("visual_asset_refs")
        )
        synthetic_interrupt_message: ThreadChatHistoryMessagePayload = {
            "role": "assistant",
            "content": interrupt_content,
            "kind": "interrupt",
            "created_at": None,
            "citations": list(channel_values.get("citations") or []),
            "visual_assets": [],
            "evidence_chunks": list(channel_values.get("evidence_chunks") or []),
            "retrieval_mode": str(
                channel_values.get("retrieval_mode") or ""
            ).strip()
            or None,
            "debug_trace": (
                dict(channel_values.get("debug_trace") or {}) or None
            ),
        }
        if (
            interrupt_content
            and visible_messages
            and visible_messages[-1]["role"] == "assistant"
            and visible_messages[-1]["kind"] == "interrupt"
            and visible_messages[-1]["content"] == interrupt_content
        ):
            visible_messages[-1] = {
                **visible_messages[-1],
                **synthetic_interrupt_message,
                "created_at": visible_messages[-1]["created_at"],
            }
            if synthetic_visual_asset_refs:
                visual_asset_ref_map[len(visible_messages) - 1] = synthetic_visual_asset_refs
        elif interrupt_content:
            visible_messages.append(
                synthetic_interrupt_message
            )
            if synthetic_visual_asset_refs:
                visual_asset_ref_map[len(visible_messages) - 1] = synthetic_visual_asset_refs

    if visible_messages and visual_asset_ref_map:
        loaded_assets_by_index = _load_visible_message_assets(
            thread_id=thread["thread_id"],
            active_document_ids=list(thread.get("active_document_ids") or []),
            visual_asset_ref_map=visual_asset_ref_map,
        )
        for message_index, visual_assets in loaded_assets_by_index.items():
            if 0 <= message_index < len(visible_messages):
                visible_messages[message_index]["visual_assets"] = visual_assets

    if checkpoint_tuple is None or (
        not raw_messages and not visible_messages and interrupt_payload is None
    ):
        history_notice = "이 채팅방에는 아직 저장된 대화 기록이 없습니다."

    return {
        "thread": thread,
        "messages": visible_messages,
        "interrupt": interrupt_payload,
        "history_notice": history_notice,
    }


def run_thread_chat(
    *,
    thread_id: str,
    message: str,
    allow_web_search: bool = False,
    resume: bool = False,
) -> Stage5Output:
    """thread 범위에서 stage5 챗봇을 실행한다."""
    thread = _require_thread(thread_id)
    normalized_message = str(message or "").strip()
    if not normalized_message:
        raise ValueError("message is required")

    effective_resume = bool(resume)
    if not effective_resume:
        try:
            channel_values = _load_checkpoint_channel_values(thread["thread_id"])
        except Exception:
            channel_values = {}
        needs_clarification = bool(channel_values.get("needs_clarification"))
        clarification_response = str(
            channel_values.get("clarification_response") or ""
        ).strip()
        clarification_payload = _normalize_interrupt_payload(
            channel_values.get("clarification_payload")
        )
        if (
            needs_clarification
            and not clarification_response
            and clarification_payload is not None
            and _looks_like_clarification_followup(
                thread_id=thread["thread_id"],
                message=normalized_message,
                clarification_payload=clarification_payload,
            )
        ):
            effective_resume = True

    return run_stage5_chatbot(
        {
            "thread_id": thread["thread_id"],
            "user_message": normalized_message,
            "allow_web_search": allow_web_search,
        },
        resume_value=normalized_message if effective_resume else None,
    )
