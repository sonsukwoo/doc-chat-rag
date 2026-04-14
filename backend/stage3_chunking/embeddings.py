"""Semantic chunking에서 재사용하는 embedding helper."""

from __future__ import annotations

import math
from typing import Iterable

from .config import (
    STAGE3_EMBEDDING_API_KEY,
    STAGE3_EMBEDDING_BASE_URL,
    STAGE3_EMBEDDING_BATCH_SIZE,
    STAGE3_EMBEDDING_MODEL,
    STAGE3_ENABLE_SEMANTIC,
)

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 설치 환경 의존
    OpenAI = None


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """두 dense embedding 사이 코사인 유사도를 계산한다."""
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    dot = sum(left_value * right_value for left_value, right_value in zip(left, right))
    return dot / (left_norm * right_norm)


class SemanticEmbeddingClient:
    """OpenAI 호환 embeddings API를 감싼 best-effort client."""

    def __init__(
        self,
        *,
        enabled: bool = STAGE3_ENABLE_SEMANTIC,
        base_url: str = STAGE3_EMBEDDING_BASE_URL,
        api_key: str = STAGE3_EMBEDDING_API_KEY,
        model: str = STAGE3_EMBEDDING_MODEL,
        batch_size: int = STAGE3_EMBEDDING_BATCH_SIZE,
    ) -> None:
        self.enabled = enabled and OpenAI is not None
        self.model = model
        self.batch_size = max(1, batch_size)
        self.last_error: str | None = None
        self._client = OpenAI(base_url=base_url, api_key=api_key) if self.enabled else None
        self._cache: dict[str, list[float]] = {}

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]] | None:
        """텍스트 목록을 임베딩한다.

        실패하면 예외를 올리지 않고 semantic 기능을 끈 뒤 `None`을 반환한다.
        """
        normalized_texts = [text.strip() for text in texts]
        if not self.enabled or not normalized_texts:
            return None

        missing_texts = [
            text for text in normalized_texts if text and text not in self._cache
        ]
        if missing_texts:
            try:
                for start_index in range(0, len(missing_texts), self.batch_size):
                    batch = missing_texts[start_index : start_index + self.batch_size]
                    response = self._client.embeddings.create(
                        model=self.model,
                        input=batch,
                    )
                    for text, item in zip(batch, response.data):
                        self._cache[text] = list(item.embedding)
            except Exception as exc:  # pragma: no cover - 외부 런타임 의존
                self.enabled = False
                self.last_error = type(exc).__name__
                return None

        embeddings: list[list[float]] = []
        for text in normalized_texts:
            if not text:
                embeddings.append([])
                continue
            cached = self._cache.get(text)
            if cached is None:
                return None
            embeddings.append(cached)
        return embeddings
