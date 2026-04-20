"""Stage-4 thread-aware retrieval service."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from backend.thread_identity import build_thread_collection_name
from backend.stage3_chunking.embeddings import OpenAIEmbeddingClient
from backend.stage3_indexing.qdrant import QdrantRestClient

from .config import (
    STAGE4_BM25_ASCII_FOLDING,
    STAGE4_BM25_EXCLUDED_ROLE_HINTS,
    STAGE4_BM25_LANGUAGE,
    STAGE4_BM25_TOKENIZER,
    STAGE4_BM25_VECTOR_NAME,
    STAGE4_DENSE_VECTOR_NAME,
    STAGE4_ENABLE_MMR,
    STAGE4_ENABLE_RERANK,
    STAGE4_ENABLE_SCORE_FALLBACK,
    STAGE4_FETCH_K,
    STAGE4_HYBRID_BM25_FETCH_K,
    STAGE4_HYBRID_DENSE_FETCH_K,
    STAGE4_HYBRID_RRF_WEIGHTS,
    STAGE4_MMR_LAMBDA_MULT,
    STAGE4_QDRANT_API_KEY,
    STAGE4_QDRANT_COLLECTION_NAME,
    STAGE4_QDRANT_TIMEOUT,
    STAGE4_QDRANT_URL,
    STAGE4_RERANK_DEVICE,
    STAGE4_RERANK_MODEL,
    STAGE4_RETRIEVAL_MODE,
    STAGE4_RESTRICT_TO_DOCUMENT,
    STAGE4_SCORE_THRESHOLD,
    STAGE4_TOP_K,
)
from .postprocess import build_mmr_reranker
from .retriever import build_qdrant_chunk_retriever
from .rerank import build_cross_encoder_reranker


def _to_optional_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_document(document: Document) -> dict[str, Any]:
    metadata = dict(document.metadata or {})
    return {
        "point_id": str(metadata.get("point_id") or ""),
        "document_id": str(metadata.get("document_id") or ""),
        "chunk_id": str(metadata.get("chunk_id") or ""),
        "parent_id": str(metadata.get("parent_id") or "") or None,
        "score": float(metadata.get("score") or 0.0),
        "dense_score": (
            float(metadata.get("dense_score"))
            if metadata.get("dense_score") not in (None, "")
            else None
        ),
        "bm25_score": (
            float(metadata.get("bm25_score"))
            if metadata.get("bm25_score") not in (None, "")
            else None
        ),
        "chunk_type": str(metadata.get("chunk_type") or ""),
        "text": str(document.page_content or ""),
        "section_title": str(metadata.get("section_title") or "") or None,
        "primary_page": _to_optional_int(metadata.get("primary_page")),
        "page_start": _to_optional_int(metadata.get("page_start")),
        "page_end": _to_optional_int(metadata.get("page_end")),
        "has_asset": bool(metadata.get("has_asset")),
        "asset_kind": str(metadata.get("asset_kind") or "") or None,
        "asset_relative_path": str(metadata.get("asset_relative_path") or "") or None,
        "caption": str(metadata.get("caption") or "") or None,
    }


def search_thread_knowledge(
    *,
    query: str,
    thread_id: str | None,
    active_document_ids: list[str] | None = None,
    collection_name: str | None = None,
    retrieval_mode: str | None = None,
    top_k: int | None = None,
    fetch_k: int | None = None,
    dense_fetch_k: int | None = None,
    bm25_fetch_k: int | None = None,
    hybrid_rrf_weights: list[float] | None = None,
    bm25_excluded_role_hints: list[str] | None = None,
    restrict_to_document: bool | None = None,
    score_threshold: float | None = None,
    enable_score_fallback: bool | None = None,
    enable_rerank: bool | None = None,
    rerank_model: str | None = None,
    rerank_device: str | None = None,
    enable_mmr: bool | None = None,
    mmr_lambda_mult: float | None = None,
    embedding_client: OpenAIEmbeddingClient | None = None,
    qdrant_client: QdrantRestClient | None = None,
) -> dict[str, Any]:
    """thread 범위 child chunk 검색만 수행하는 stage5 전용 retrieval entrypoint."""
    normalized_query = str(query or "").strip()
    normalized_thread_id = str(thread_id or "").strip() or None
    normalized_document_ids = [
        str(item).strip()
        for item in active_document_ids or []
        if str(item).strip()
    ]
    if not normalized_query:
        return {
            "status": "skipped",
            "query": normalized_query,
            "thread_id": normalized_thread_id,
            "active_document_ids": normalized_document_ids,
            "retrievals": [],
            "skip_reason": "missing_query",
        }

    resolved_collection_name = str(collection_name or "").strip()
    if not resolved_collection_name and normalized_thread_id:
        resolved_collection_name = build_thread_collection_name(normalized_thread_id)
    if not resolved_collection_name:
        resolved_collection_name = STAGE4_QDRANT_COLLECTION_NAME
    resolved_retrieval_mode = (
        str(retrieval_mode or STAGE4_RETRIEVAL_MODE).strip().lower() or "dense"
    )
    resolved_top_k = int(top_k or STAGE4_TOP_K)
    resolved_dense_fetch_k = int(dense_fetch_k or fetch_k or STAGE4_HYBRID_DENSE_FETCH_K)
    resolved_bm25_fetch_k = int(bm25_fetch_k or fetch_k or STAGE4_HYBRID_BM25_FETCH_K)
    resolved_fetch_k = max(
        resolved_top_k,
        int(fetch_k or STAGE4_FETCH_K),
        resolved_dense_fetch_k,
        resolved_bm25_fetch_k,
    )
    resolved_hybrid_rrf_weights = (
        hybrid_rrf_weights
        if hybrid_rrf_weights is not None
        else STAGE4_HYBRID_RRF_WEIGHTS
    )
    resolved_bm25_excluded_role_hints = list(
        bm25_excluded_role_hints or STAGE4_BM25_EXCLUDED_ROLE_HINTS
    )
    resolved_restrict_to_document = bool(
        STAGE4_RESTRICT_TO_DOCUMENT
        if restrict_to_document is None
        else restrict_to_document
    )
    resolved_score_threshold = (
        STAGE4_SCORE_THRESHOLD if score_threshold is None else score_threshold
    )
    resolved_enable_score_fallback = bool(
        STAGE4_ENABLE_SCORE_FALLBACK
        if enable_score_fallback is None
        else enable_score_fallback
    )
    resolved_enable_rerank = bool(
        STAGE4_ENABLE_RERANK if enable_rerank is None else enable_rerank
    )
    resolved_rerank_model = str(rerank_model or STAGE4_RERANK_MODEL).strip() or STAGE4_RERANK_MODEL
    resolved_rerank_device = str(rerank_device or STAGE4_RERANK_DEVICE).strip() or STAGE4_RERANK_DEVICE
    resolved_enable_mmr = bool(STAGE4_ENABLE_MMR if enable_mmr is None else enable_mmr)
    resolved_mmr_lambda_mult = float(
        STAGE4_MMR_LAMBDA_MULT if mmr_lambda_mult is None else mmr_lambda_mult
    )
    resolved_document_id = normalized_document_ids[0] if len(normalized_document_ids) == 1 else None

    qdrant_configured = bool((qdrant_client is not None or STAGE4_QDRANT_URL) and resolved_collection_name)
    if not qdrant_configured:
        return {
            "status": "skipped",
            "query": normalized_query,
            "thread_id": normalized_thread_id,
            "active_document_ids": normalized_document_ids,
            "retrievals": [],
            "skip_reason": "missing_qdrant_config",
        }

    embedding_client = embedding_client or OpenAIEmbeddingClient(enabled=True)
    owns_qdrant_client = qdrant_client is None
    qdrant_client = qdrant_client or QdrantRestClient(
        base_url=STAGE4_QDRANT_URL,
        api_key=STAGE4_QDRANT_API_KEY,
        timeout=STAGE4_QDRANT_TIMEOUT,
    )

    try:
        retriever = build_qdrant_chunk_retriever(
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            collection_name=resolved_collection_name,
            retrieval_mode=resolved_retrieval_mode,
            fetch_limit=resolved_fetch_k,
            dense_fetch_k=resolved_dense_fetch_k,
            bm25_fetch_k=resolved_bm25_fetch_k,
            dense_vector_name=STAGE4_DENSE_VECTOR_NAME,
            bm25_vector_name=STAGE4_BM25_VECTOR_NAME,
            bm25_options={
                "tokenizer": STAGE4_BM25_TOKENIZER,
                "language": STAGE4_BM25_LANGUAGE,
                "ascii_folding": STAGE4_BM25_ASCII_FOLDING,
            },
            rrf_weights=resolved_hybrid_rrf_weights,
            bm25_excluded_role_hints=resolved_bm25_excluded_role_hints,
            thread_id=normalized_thread_id,
            document_id=resolved_document_id,
            active_document_ids=normalized_document_ids,
            restrict_to_document=resolved_restrict_to_document,
            score_threshold=resolved_score_threshold,
        )
        documents = list(retriever.invoke(normalized_query))
        score_fallback_applied = False
        effective_score_threshold = resolved_score_threshold

        if (
            effective_score_threshold is not None
            and resolved_enable_score_fallback
            and len(documents) < resolved_top_k
        ):
            fallback_retriever = build_qdrant_chunk_retriever(
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                collection_name=resolved_collection_name,
                retrieval_mode=resolved_retrieval_mode,
                fetch_limit=resolved_fetch_k,
                dense_fetch_k=resolved_dense_fetch_k,
                bm25_fetch_k=resolved_bm25_fetch_k,
                dense_vector_name=STAGE4_DENSE_VECTOR_NAME,
                bm25_vector_name=STAGE4_BM25_VECTOR_NAME,
                bm25_options={
                    "tokenizer": STAGE4_BM25_TOKENIZER,
                    "language": STAGE4_BM25_LANGUAGE,
                    "ascii_folding": STAGE4_BM25_ASCII_FOLDING,
                },
                rrf_weights=resolved_hybrid_rrf_weights,
                bm25_excluded_role_hints=resolved_bm25_excluded_role_hints,
                thread_id=normalized_thread_id,
                document_id=resolved_document_id,
                active_document_ids=normalized_document_ids,
                restrict_to_document=resolved_restrict_to_document,
                score_threshold=None,
            )
            documents = list(fallback_retriever.invoke(normalized_query))
            score_fallback_applied = True
            effective_score_threshold = None

        document_pipeline = (
            RunnableLambda(
                lambda retrieved_documents: {
                    "documents": list(retrieved_documents),
                    "rerank_applied": False,
                    "rerank_error": None,
                    "mmr_applied": False,
                }
            )
            | build_cross_encoder_reranker(
                query=normalized_query,
                enabled=resolved_enable_rerank,
                model_name=resolved_rerank_model,
                top_n=resolved_top_k,
                device=resolved_rerank_device,
            )
            | build_mmr_reranker(
                query=normalized_query,
                embedding_client=embedding_client,
                top_k=resolved_top_k,
                enabled=resolved_enable_mmr,
                lambda_mult=resolved_mmr_lambda_mult,
            )
            | RunnableLambda(
                lambda payload: {
                    "retrievals": [
                        _normalize_document(document)
                        for document in list(payload.get("documents") or [])[:resolved_top_k]
                    ],
                    "rerank_applied": bool(payload.get("rerank_applied")),
                    "rerank_error": payload.get("rerank_error"),
                    "mmr_applied": bool(payload.get("mmr_applied")),
                }
            )
        )
        retrieval_result = document_pipeline.invoke(documents)
    finally:
        if owns_qdrant_client:
            qdrant_client.close()

    return {
        "status": "completed",
        "query": normalized_query,
        "thread_id": normalized_thread_id,
        "active_document_ids": normalized_document_ids,
        "collection_name": resolved_collection_name,
        "retrieval_mode": resolved_retrieval_mode,
        "top_k": resolved_top_k,
        "fetch_k": resolved_fetch_k,
        "score_threshold_applied": effective_score_threshold,
        "score_fallback_applied": score_fallback_applied,
        "rerank_applied": bool(retrieval_result.get("rerank_applied")),
        "rerank_error": retrieval_result.get("rerank_error"),
        "mmr_applied": bool(retrieval_result.get("mmr_applied")),
        "retrieved_count": len(list(retrieval_result.get("retrievals") or [])),
        "retrievals": list(retrieval_result.get("retrievals") or []),
        "skip_reason": None,
    }
