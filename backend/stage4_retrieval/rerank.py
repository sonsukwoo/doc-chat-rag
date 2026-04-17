"""LangChain-style reranking helpers for stage-4 retrieval."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def _normalize_device(device: str | None) -> str | None:
    """비어 있는 device 설정은 None으로 정규화한다."""
    if device is None:
        return None
    normalized = device.strip()
    return normalized or None


@lru_cache(maxsize=8)
def get_huggingface_cross_encoder_model(
    *,
    model_name: str,
    device: str | None,
) -> Any:
    """Hugging Face cross-encoder 모델 자체를 캐시한다."""
    try:
        from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    except ImportError as exc:  # pragma: no cover - 설치 환경 의존
        raise RuntimeError("missing_langchain_reranker_dependencies") from exc

    model_kwargs: dict[str, Any] = {}
    normalized_device = _normalize_device(device)
    if normalized_device is not None:
        model_kwargs["device"] = normalized_device

    cross_encoder = HuggingFaceCrossEncoder(
        model_name=model_name,
        model_kwargs=model_kwargs,
    )
    return cross_encoder


def get_huggingface_cross_encoder_reranker(
    *,
    model_name: str,
    top_n: int,
    device: str | None,
) -> Any:
    """Hugging Face cross-encoder 기반 LangChain reranker를 생성한다."""
    try:
        from langchain_classic.retrievers.document_compressors import (
            CrossEncoderReranker,
        )
    except ImportError as exc:  # pragma: no cover - 설치 환경 의존
        raise RuntimeError("missing_langchain_reranker_dependencies") from exc

    cross_encoder = get_huggingface_cross_encoder_model(
        model_name=model_name,
        device=device,
    )
    return CrossEncoderReranker(
        model=cross_encoder,
        top_n=max(1, int(top_n)),
    )


def apply_cross_encoder_reranking(
    *,
    query: str,
    documents: list[Document],
    enabled: bool,
    model_name: str,
    top_n: int,
    device: str | None,
) -> tuple[list[Document], bool, str | None]:
    """Cross-encoder reranker를 적용해 query relevance 기준으로 후보를 다시 정렬한다."""
    if not enabled or len(documents) <= 1:
        return documents, False, None

    try:
        reranker = get_huggingface_cross_encoder_reranker(
            model_name=model_name,
            top_n=top_n,
            device=device,
        )
        reranked_documents = list(
            reranker.compress_documents(
                documents=documents,
                query=query,
            )
        )
    except Exception as exc:  # pragma: no cover - 외부 모델 런타임 의존
        return documents, False, type(exc).__name__

    if not reranked_documents:
        return documents, False, "empty_rerank_result"

    return reranked_documents[:top_n], True, None


def build_cross_encoder_reranker(
    *,
    query: str,
    enabled: bool,
    model_name: str,
    top_n: int,
    device: str | None,
) -> RunnableLambda:
    """문서 payload에 cross-encoder rerank 결과를 반영하는 Runnable 단계다."""

    def _rerank(payload: dict[str, Any]) -> dict[str, Any]:
        documents = list(payload.get("documents") or [])
        reranked_documents, rerank_applied, rerank_error = (
            apply_cross_encoder_reranking(
                query=query,
                documents=documents,
                enabled=enabled,
                model_name=model_name,
                top_n=top_n,
                device=device,
            )
        )
        return {
            **payload,
            "documents": reranked_documents,
            "rerank_applied": rerank_applied,
            "rerank_error": rerank_error,
        }

    return RunnableLambda(_rerank)
