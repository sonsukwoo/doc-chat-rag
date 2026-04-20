"""Stage-5 chatbot configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH)


def _env_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


# stage5의 기본 에이전트/답변 생성 모델이다.
# tool-calling과 근거 기반 답변 생성을 모두 수행하므로, nano보다 한 단계 높은 mini를 기본값으로 둔다.
STAGE5_AGENT_MODEL = os.getenv("STAGE5_AGENT_MODEL", "openai:gpt-4.1-mini")

# 채팅은 요청 간 맥락 유지가 핵심이므로 postgres를 기본 체크포인터로 둔다.
STAGE5_CHECKPOINTER_BACKEND = os.getenv(
    "STAGE5_CHECKPOINTER_BACKEND",
    "postgres",
).strip().lower()
STAGE5_POSTGRES_URI = os.getenv("STAGE5_POSTGRES_URI", "").strip()

# retrieval 정책 기본값은 현재 프로젝트 결론에 맞춰 dense로 둔다.
STAGE5_DEFAULT_RETRIEVAL_MODE = os.getenv(
    "STAGE5_DEFAULT_RETRIEVAL_MODE",
    "dense",
).strip().lower()
STAGE5_DEFAULT_TOP_K = int(os.getenv("STAGE5_DEFAULT_TOP_K", "8"))
STAGE5_DEEP_RETRIEVAL_TOP_K = int(
    os.getenv(
        "STAGE5_DEEP_RETRIEVAL_TOP_K",
        str(max(STAGE5_DEFAULT_TOP_K, 10)),
    )
)
STAGE5_DEEP_RETRIEVAL_FETCH_K = int(
    os.getenv(
        "STAGE5_DEEP_RETRIEVAL_FETCH_K",
        str(max(STAGE5_DEEP_RETRIEVAL_TOP_K, 24)),
    )
)
STAGE5_MULTI_DOC_PER_DOCUMENT_TOP_K = int(
    os.getenv("STAGE5_MULTI_DOC_PER_DOCUMENT_TOP_K", "8")
)
STAGE5_HISTORY_MAX_TOKENS = int(os.getenv("STAGE5_HISTORY_MAX_TOKENS", "900"))
STAGE5_SUMMARY_MAX_LINES = int(os.getenv("STAGE5_SUMMARY_MAX_LINES", "8"))
STAGE5_CONTEXT_WINDOW_SIZE = int(os.getenv("STAGE5_CONTEXT_WINDOW_SIZE", "1"))
STAGE5_CONTEXT_WINDOW_MAX_HITS = int(
    os.getenv("STAGE5_CONTEXT_WINDOW_MAX_HITS", "4")
)
STAGE5_ENABLE_WEB_SEARCH = _env_bool("STAGE5_ENABLE_WEB_SEARCH", False)
STAGE5_ENABLE_DEEP_RETRIEVAL = _env_bool("STAGE5_ENABLE_DEEP_RETRIEVAL", True)
STAGE5_ENABLE_CONTEXT_WINDOW = _env_bool("STAGE5_ENABLE_CONTEXT_WINDOW", True)
