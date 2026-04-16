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


def _read_text(name: str, default: str) -> str:
    """환경변수를 문자열 설정값으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    normalized = raw_value.strip()
    return normalized or default


def _read_optional_float(name: str) -> float | None:
    """환경변수가 비어 있으면 None, 값이 있으면 float으로 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None
    return float(normalized)


def _read_optional_float_list(name: str) -> list[float] | None:
    """쉼표 구분 float 목록을 읽는다. 비어 있으면 None을 반환한다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return None
    normalized = raw_value.strip()
    if not normalized:
        return None

    values: list[float] = []
    for token in normalized.split(","):
        stripped = token.strip()
        if not stripped:
            continue
        values.append(float(stripped))
    return values or None


def _read_text_list(name: str, default: list[str]) -> list[str]:
    """쉼표 구분 문자열 목록을 읽는다."""
    raw_value = os.getenv(name)
    if raw_value is None:
        return list(default)
    normalized = raw_value.strip()
    if not normalized:
        return []
    values = [token.strip() for token in normalized.split(",") if token.strip()]
    return values


# stage4 단독 실행 시 기본으로 참조할 chunk/parent 입력 경로다.
DEFAULT_CHUNKS_JSON_PATH = OUTPUT_ROOT / "1" / "stage3" / "chunks.json"
DEFAULT_PARENTS_JSON_PATH = OUTPUT_ROOT / "1" / "stage3" / "parents.json"

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
    os.getenv("STAGE3_QDRANT_COLLECTION_NAME", "rag_chat_hybrid"),
).strip()
STAGE4_QDRANT_TIMEOUT = float(
    os.getenv(
        "STAGE4_QDRANT_TIMEOUT",
        os.getenv("STAGE3_QDRANT_TIMEOUT", "30.0"),
    )
)

# stage4 retrieval 기본 모드다. hybrid를 기본값으로 두고, dense baseline 비교는 옵션으로 남긴다.
STAGE4_RETRIEVAL_MODE = _read_text("STAGE4_RETRIEVAL_MODE", "hybrid")

# hybrid 컬렉션의 named vector 이름이다. stage3 설정을 그대로 재사용한다.
STAGE4_DENSE_VECTOR_NAME = _read_text(
    "STAGE4_DENSE_VECTOR_NAME",
    os.getenv("STAGE3_QDRANT_DENSE_VECTOR_NAME", "dense"),
)
STAGE4_BM25_VECTOR_NAME = _read_text(
    "STAGE4_BM25_VECTOR_NAME",
    os.getenv("STAGE3_QDRANT_BM25_VECTOR_NAME", "bm25"),
)

# 기본 retrieval top-k 값이다.
STAGE4_TOP_K = _read_int("STAGE4_TOP_K", 8)

# 최종 top-k를 자르기 전에 넓게 가져올 dense 후보 수다.
# 이후 하이브리드 검색이나 후속 정렬 단계가 붙어도 재사용할 수 있다.
STAGE4_FETCH_K = _read_int("STAGE4_FETCH_K", max(STAGE4_TOP_K, 20))

# hybrid 모드에서 dense/bm25 브랜치가 각각 prefetch 할 후보 수다.
# 별도 값이 없으면 기존 fetch_k를 그대로 재사용한다.
STAGE4_HYBRID_DENSE_FETCH_K = _read_int(
    "STAGE4_HYBRID_DENSE_FETCH_K",
    STAGE4_FETCH_K,
)
STAGE4_HYBRID_BM25_FETCH_K = _read_int(
    "STAGE4_HYBRID_BM25_FETCH_K",
    STAGE4_FETCH_K,
)

# Qdrant built-in weighted RRF 설정이다. 비어 있으면 equal-weight RRF를 사용한다.
STAGE4_HYBRID_RRF_WEIGHTS = _read_optional_float_list(
    "STAGE4_HYBRID_RRF_WEIGHTS"
)

# 검색 시 현재 문서 범위로만 조회할지 결정한다.
STAGE4_RESTRICT_TO_DOCUMENT = _read_bool("STAGE4_RESTRICT_TO_DOCUMENT", True)

# score threshold를 주고 싶을 때만 사용한다. 비어 있으면 전체 top-k를 그대로 받는다.
STAGE4_SCORE_THRESHOLD = _read_optional_float("STAGE4_SCORE_THRESHOLD")

# BM25 query에 사용할 텍스트 처리 옵션이다.
STAGE4_BM25_TOKENIZER = _read_text(
    "STAGE4_BM25_TOKENIZER",
    os.getenv("STAGE3_BM25_TOKENIZER", "multilingual"),
)
STAGE4_BM25_LANGUAGE = _read_text(
    "STAGE4_BM25_LANGUAGE",
    os.getenv("STAGE3_BM25_LANGUAGE", "none"),
)
STAGE4_BM25_ASCII_FOLDING = _read_bool(
    "STAGE4_BM25_ASCII_FOLDING",
    _read_bool("STAGE3_BM25_ASCII_FOLDING", False),
)

# BM25 브랜치에서 제외할 역할 힌트 목록이다.
STAGE4_BM25_EXCLUDED_ROLE_HINTS = _read_text_list(
    "STAGE4_BM25_EXCLUDED_ROLE_HINTS",
    ["reference_like", "front_matter_like", "title_only"],
)
