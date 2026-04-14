"""Stage-3 chunking runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
OUTPUT_ROOT = BACKEND_DIR / "outputs"


def _read_bool(name: str, default: bool) -> bool:
    """환경변수를 bool 설정값으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}

# stage3를 단독 실행할 때 사용할 기본 cleaned.json 입력 경로다.
# 이후 상위 통합 파이프라인이 생기면 함수 인자로 넘기는 방식으로 교체할 수 있다.
DEFAULT_CLEANED_JSON_PATH = OUTPUT_ROOT / "1" / "cleaned.json"

# stage3 산출물 파일명 규칙이다.
DEFAULT_CHUNKS_JSON_NAME = "chunks.json"
DEFAULT_CHUNKS_JSONL_NAME = "chunks.jsonl"
DEFAULT_CHUNKS_MD_NAME = "chunks.md"

# 구조 기반 text chunk 목표 크기다. 실제 토큰 계산 대신 보수적인 추정값을 사용한다.
STAGE3_TEXT_TARGET_TOKENS = int(os.getenv("STAGE3_TEXT_TARGET_TOKENS", "600"))
STAGE3_TEXT_MAX_TOKENS = int(os.getenv("STAGE3_TEXT_MAX_TOKENS", "900"))
STAGE3_TEXT_MIN_TOKENS = int(os.getenv("STAGE3_TEXT_MIN_TOKENS", "180"))

# text chunk에 붙일 이전 문맥 overlap 크기다.
STAGE3_TEXT_OVERLAP_TOKENS = int(os.getenv("STAGE3_TEXT_OVERLAP_TOKENS", "100"))

# semantic split / merge 사용 여부다.
STAGE3_ENABLE_SEMANTIC = _read_bool("STAGE3_ENABLE_SEMANTIC", True)
STAGE3_ENABLE_SEMANTIC_SPLIT = _read_bool("STAGE3_ENABLE_SEMANTIC_SPLIT", True)
STAGE3_ENABLE_SEMANTIC_MERGE = _read_bool("STAGE3_ENABLE_SEMANTIC_MERGE", True)

# semantic split / merge 판단 기준이다.
STAGE3_SEMANTIC_SPLIT_SIM_THRESHOLD = float(
    os.getenv("STAGE3_SEMANTIC_SPLIT_SIM_THRESHOLD", "0.78")
)
STAGE3_SEMANTIC_MERGE_SIM_THRESHOLD = float(
    os.getenv("STAGE3_SEMANTIC_MERGE_SIM_THRESHOLD", "0.84")
)
STAGE3_SEMANTIC_MERGE_CANDIDATE_MAX_TOKENS = int(
    os.getenv("STAGE3_SEMANTIC_MERGE_CANDIDATE_MAX_TOKENS", "250")
)

# 로컬 Ollama의 OpenAI 호환 embeddings 엔드포인트 설정이다.
STAGE3_EMBEDDING_BASE_URL = os.getenv(
    "STAGE3_EMBEDDING_BASE_URL", "http://localhost:11434/v1"
)
STAGE3_EMBEDDING_API_KEY = os.getenv("STAGE3_EMBEDDING_API_KEY", "ollama")
STAGE3_EMBEDDING_MODEL = os.getenv("STAGE3_EMBEDDING_MODEL", "bge-m3")
STAGE3_EMBEDDING_BATCH_SIZE = int(os.getenv("STAGE3_EMBEDDING_BATCH_SIZE", "16"))
