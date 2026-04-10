"""Stage-2 preprocessing model and runtime configuration."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"

load_dotenv(ENV_PATH)

DEFAULT_RAW_JSON_PATH = BACKEND_DIR / "outputs" / "2" / "2.json"
OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "openai:gpt-4o-mini")


@lru_cache(maxsize=1)
def get_base_model():
    """2차 전처리에서 공통으로 사용할 기본 OpenAI 모델을 반환한다."""
    return init_chat_model(OPENAI_VLM_MODEL, temperature=0)
