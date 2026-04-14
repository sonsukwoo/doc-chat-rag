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

# stage2를 직접 실행할 때 사용할 기본 raw.json 입력 경로다.
# 현재는 CLI 인자를 따로 받지 않으므로, `python -m backend.stage2` 실행 시 이 값을 사용한다.
DEFAULT_RAW_JSON_PATH = BACKEND_DIR / "outputs" / "1" / "1.json"

# figure 검토, VLM 기반 table summary에 사용할 멀티모달 모델 이름이다.
# `.env`에 없으면 기본값으로 `gpt-4o-mini`를 사용한다.
OPENAI_VLM_MODEL = os.getenv("OPENAI_VLM_MODEL", "openai:gpt-4o-mini")

# document profile 생성, table route, text 기반 table summary에 사용할 텍스트 모델 이름이다.
# `.env`에 없으면 기본값으로 `gpt-4.1-nano`를 사용한다.
OPENAI_TEXT_MODEL = os.getenv("OPENAI_TEXT_MODEL", "openai:gpt-4.1-nano")

# LangGraph retry_policy가 모델 호출 노드를 최대 몇 번까지 재시도할지 정한다.
# 일시적인 API 오류가 나면 이 횟수 안에서 다시 시도하고, 마지막까지 실패하면 fallback으로 내려간다.
MODEL_RETRY_MAX_ATTEMPTS = int(os.getenv("STAGE2_MODEL_RETRY_MAX_ATTEMPTS", "3"))

# 첫 재시도까지 기다리는 초기 간격(초)이다.
# 이후 간격은 LangGraph RetryPolicy의 backoff 정책에 따라 점진적으로 늘어난다.
MODEL_RETRY_INITIAL_INTERVAL = float(
    os.getenv("STAGE2_MODEL_RETRY_INITIAL_INTERVAL", "1.0")
)


@lru_cache(maxsize=1)
def get_base_model():
    """figure 검토와 VLM table summary에 사용할 멀티모달 모델을 반환한다."""
    return init_chat_model(OPENAI_VLM_MODEL, temperature=0)


@lru_cache(maxsize=1)
def get_text_model():
    """document profile, table route, text table summary에 사용할 텍스트 모델을 반환한다."""
    return init_chat_model(OPENAI_TEXT_MODEL, temperature=0)
