"""Stage-3 indexing runtime configuration."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = BACKEND_DIR.parent
ENV_PATH = PROJECT_ROOT / ".env"
OUTPUT_ROOT = BACKEND_DIR / "outputs"

load_dotenv(ENV_PATH)


def _read_bool(name: str, default: bool) -> bool:
    """환경변수를 bool 설정값으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


# stage3 인덱싱을 단독 실행할 때 사용할 기본 chunks.json 입력 경로다.
DEFAULT_CHUNKS_JSON_PATH = OUTPUT_ROOT / "1" / "chunks.json"

# chunk 생성 이후 Qdrant 업로드까지 자동으로 이어서 수행할지 결정한다.
STAGE3_ENABLE_INDEXING = _read_bool("STAGE3_ENABLE_INDEXING", True)

# Qdrant 연결 설정이다. URL이 비어 있으면 업로드 단계는 자동으로 건너뛴다.
STAGE3_QDRANT_URL = os.getenv("STAGE3_QDRANT_URL", "").strip()
STAGE3_QDRANT_API_KEY = os.getenv("STAGE3_QDRANT_API_KEY", "").strip()
STAGE3_QDRANT_COLLECTION_NAME = os.getenv(
    "STAGE3_QDRANT_COLLECTION_NAME",
    "rag_chat",
).strip()

# Qdrant upsert 동작 설정이다.
STAGE3_QDRANT_TIMEOUT = float(os.getenv("STAGE3_QDRANT_TIMEOUT", "30.0"))
STAGE3_QDRANT_UPSERT_BATCH_SIZE = int(
    os.getenv("STAGE3_QDRANT_UPSERT_BATCH_SIZE", "64")
)

# 인덱싱 결과를 문서 폴더에 남길 manifest 파일명이다.
DEFAULT_INDEXING_MANIFEST_NAME = "indexing.json"
