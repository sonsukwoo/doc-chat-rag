"""thread-scoped 챗봇 라우터."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.services import load_thread_chat_view, run_thread_chat


router = APIRouter(prefix="/threads", tags=["chat"])


class ChatTurnBody(BaseModel):
    message: str = Field(min_length=1)
    allow_web_search: bool = False
    resume: bool = False


@router.get("/{thread_id}/chat")
def get_thread_chat(thread_id: str) -> dict[str, Any]:
    """프론트 채팅 화면이 필요한 thread 기본 정보를 반환한다."""
    try:
        return load_thread_chat_view(thread_id)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@router.post("/{thread_id}/chat")
def post_thread_chat(thread_id: str, body: ChatTurnBody) -> dict[str, Any]:
    """thread 범위 채팅 1턴을 실행한다."""
    try:
        result = run_thread_chat(
            thread_id=thread_id,
            message=body.message,
            allow_web_search=body.allow_web_search,
            resume=body.resume,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"result": result}
