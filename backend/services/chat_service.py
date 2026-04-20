"""thread-scoped stage5 챗봇 실행 서비스."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

from langchain_core.messages import AIMessage, HumanMessage

from backend.stage5_chatbot import run_stage5_chatbot
from backend.stage5_chatbot.checkpointer import stage5_checkpointer_context
from backend.stage5_chatbot.schemas import ChatbotInterruptPayload
from backend.stage5_chatbot.schemas import Stage5Output

from .thread_service import ThreadPayload, get_thread_detail


class ThreadChatHistoryMessagePayload(TypedDict):
    role: Literal["user", "assistant"]
    content: str
    kind: Literal["answer", "interrupt"]


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
) -> list[ThreadChatHistoryMessagePayload]:
    visible_messages: list[ThreadChatHistoryMessagePayload] = []
    for message in raw_messages:
        if isinstance(message, HumanMessage):
            content = str(message.content or "").strip()
            if not content:
                continue
            visible_messages.append(
                {
                    "role": "user",
                    "content": content,
                    "kind": "answer",
                }
            )
            continue

        if isinstance(message, AIMessage):
            content = str(message.content or "").strip()
            if not content:
                continue
            visible_messages.append(
                {
                    "role": "assistant",
                    "content": content,
                    "kind": "answer",
                }
            )

    return visible_messages


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
    visible_messages = _serialize_visible_messages(raw_messages)

    needs_clarification = bool(channel_values.get("needs_clarification"))
    clarification_response = str(channel_values.get("clarification_response") or "").strip()
    interrupt_payload = _normalize_interrupt_payload(
        channel_values.get("clarification_payload")
    )

    if needs_clarification and not clarification_response and interrupt_payload is not None:
        interrupt_content = _format_interrupt_content(interrupt_payload)
        if interrupt_content and (
            not visible_messages
            or visible_messages[-1]["role"] != "assistant"
            or visible_messages[-1]["content"] != interrupt_content
        ):
            visible_messages.append(
                {
                    "role": "assistant",
                    "content": interrupt_content,
                    "kind": "interrupt",
                }
            )

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

    return run_stage5_chatbot(
        {
            "thread_id": thread["thread_id"],
            "user_message": normalized_message,
            "allow_web_search": allow_web_search,
        },
        resume_value=normalized_message if resume else None,
    )
