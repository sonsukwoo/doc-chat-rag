"""LangChain-style reranking helpers for stage-4 retrieval."""

from __future__ import annotations

from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda
from langchain_core.vectorstores.utils import maximal_marginal_relevance

from backend.stage3_chunking.embeddings import OpenAIEmbeddingClient


def apply_mmr_reranking(
    *,
    query: str,
    documents: list[Document],
    embedding_client: OpenAIEmbeddingClient,
    top_k: int,
    enabled: bool,
    lambda_mult: float,
) -> tuple[list[Document], bool]:
    """후보 문서가 많을 때 MMR로 중복을 줄여 top-k를 다시 고른다."""
    if not enabled or len(documents) <= 1:
        return documents[:top_k], False

    query_embeddings = embedding_client.embed_texts([query])
    document_embeddings = embedding_client.embed_texts(
        [document.page_content for document in documents]
    )
    if (
        query_embeddings is None
        or not query_embeddings
        or not query_embeddings[0]
        or document_embeddings is None
        or any(not embedding for embedding in document_embeddings)
    ):
        return documents[:top_k], False

    selected_indices = maximal_marginal_relevance(
        query_embedding=np.array(query_embeddings[0], dtype=np.float32),
        embedding_list=np.array(document_embeddings, dtype=np.float32),
        lambda_mult=lambda_mult,
        k=min(top_k, len(documents)),
    )
    if not selected_indices:
        return documents[:top_k], False

    reranked_documents = [documents[index] for index in selected_indices]
    return reranked_documents, True


def build_mmr_reranker(
    *,
    query: str,
    embedding_client: OpenAIEmbeddingClient,
    top_k: int,
    enabled: bool,
    lambda_mult: float,
) -> RunnableLambda:
    """문서 payload에 MMR 재정렬 결과를 반영하는 Runnable 단계다."""

    def _rerank(payload: dict[str, Any]) -> dict[str, Any]:
        documents = list(payload.get("documents") or [])
        reranked_documents, mmr_applied = apply_mmr_reranking(
            query=query,
            documents=documents,
            embedding_client=embedding_client,
            top_k=top_k,
            enabled=enabled,
            lambda_mult=lambda_mult,
        )
        return {
            **payload,
            "documents": reranked_documents,
            "mmr_applied": mmr_applied,
        }

    return RunnableLambda(_rerank)
