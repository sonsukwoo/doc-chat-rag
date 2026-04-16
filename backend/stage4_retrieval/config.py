"""Stage-4 retrieval runtime configuration."""

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


def _read_int(name: str, default: int) -> int:
    """환경변수를 int 설정값으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return int(raw_value.strip())


def _read_bool(name: str, default: bool) -> bool:
    """환경변수를 bool 설정값으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    return raw_value.strip().lower() in {"1", "true", "yes", "on"}


def _read_optional_float(name: str) -> float | None:
    """환경변수가 비어 있으면 None, 값이 있으면 float으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    return float(normalized)


# stage4 단독 실행 시 기본으로 참조할 chunk/parent 입력 경로다.
DEFAULT_CHUNKS_JSON_PATH = OUTPUT_ROOT / "1" / "chunks.json"
DEFAULT_PARENTS_JSON_PATH = OUTPUT_ROOT / "1" / "parents.json"

# 검색 결과 메타데이터를 문서 폴더에 남길 manifest 파일명이다.
DEFAULT_RETRIEVAL_MANIFEST_NAME = "retrieval.json"

# stage4 dense 검색이 사용할 Qdrant 연결 설정이다.
# 별도 stage4 환경변수가 없으면 stage3 인덱싱 설정을 그대로 재사용한다.
STAGE4_QDRANT_URL = os.getenv(
    "STAGE4_QDRANT_URL",
    os.getenv("STAGE3_QDRANT_URL", ""),
).strip()
STAGE4_QDRANT_API_KEY = os.getenv(
    "STAGE4_QDRANT_API_KEY",
    os.getenv("STAGE3_QDRANT_API_KEY", ""),
).strip()
STAGE4_QDRANT_COLLECTION_NAME = os.getenv(
    "STAGE4_QDRANT_COLLECTION_NAME",
    os.getenv("STAGE3_QDRANT_COLLECTION_NAME", "rag_chat"),
).strip()
STAGE4_QDRANT_TIMEOUT = float(
    os.getenv(
        "STAGE4_QDRANT_TIMEOUT",
        os.getenv("STAGE3_QDRANT_TIMEOUT", "30.0"),
    )
)

# 기본 dense retrieval top-k 값이다.
STAGE4_TOP_K = _read_int("STAGE4_TOP_K", 8)

# 최종 top-k를 자르기 전에 넓게 가져올 dense 후보 수다.
# 이후 하이브리드 검색이나 후속 정렬 단계가 붙어도 재사용할 수 있다.
STAGE4_FETCH_K = _read_int("STAGE4_FETCH_K", max(STAGE4_TOP_K, 20))

# 검색 시 현재 문서 범위로만 조회할지 결정한다.
STAGE4_RESTRICT_TO_DOCUMENT = _read_bool("STAGE4_RESTRICT_TO_DOCUMENT", True)

# score threshold를 주고 싶을 때만 사용한다. 비어 있으면 전체 top-k를 그대로 받는다.
STAGE4_SCORE_THRESHOLD = _read_optional_float("STAGE4_SCORE_THRESHOLD")
