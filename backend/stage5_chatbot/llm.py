"""Stage-5 chatbot model helpers."""

from __future__ import annotations

from functools import lru_cache

from langchain.chat_models import init_chat_model

from .config import STAGE5_AGENT_MODEL


@lru_cache(maxsize=1)
def get_agent_model():
    """stage5 agent와 grounded answer 생성에 공통으로 사용할 chat model."""
    return init_chat_model(STAGE5_AGENT_MODEL, temperature=0)
