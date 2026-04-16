"""Stage-4 retrieval helpers built on top of the shared Qdrant REST client."""

from __future__ import annotations

from backend.stage3_indexing.qdrant import QdrantRestClient


def build_document_filter(document_id: str) -> dict[str, object]:
    """현재 문서 범위로 retrieval을 제한하는 Qdrant filter를 만든다."""
    return {
        "must": [
            {
                "key": "document_id",
                "match": {"value": document_id},
            }
        ]
    }


def build_bm25_filter(
    *,
    document_id: str | None = None,
    restrict_to_document: bool = True,
    excluded_role_hints: list[str] | None = None,
) -> dict[str, object] | None:
    """BM25 브랜치 전용 필터를 만든다."""
    must: list[dict[str, object]] = []
    must_not: list[dict[str, object]] = []

    if restrict_to_document and document_id:
        must.append(
            {
                "key": "document_id",
                "match": {"value": document_id},
            }
        )

    for role_hint in excluded_role_hints or []:
        must_not.append(
            {
                "key": "sparse_role_hints",
                "match": {"value": role_hint},
            }
        )

    if not must and not must_not:
        return None

    query_filter: dict[str, object] = {}
    if must:
        query_filter["must"] = must
    if must_not:
        query_filter["must_not"] = must_not
    return query_filter


def search_dense_chunks(
    *,
    qdrant_client: QdrantRestClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
    dense_vector_name: str = "dense",
    document_id: str | None = None,
    restrict_to_document: bool = True,
    score_threshold: float | None = None,
) -> list[dict[str, object]]:
    """dense query vector로 chunk top-k를 조회한다."""
    query_filter = None
    if restrict_to_document and document_id:
        query_filter = build_document_filter(document_id)

    return qdrant_client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=top_k,
        with_payload=True,
        with_vector=False,
        using=dense_vector_name,
        query_filter=query_filter,
        score_threshold=score_threshold,
    )


def search_hybrid_chunks(
    *,
    qdrant_client: QdrantRestClient,
    collection_name: str,
    query_text: str,
    query_vector: list[float],
    top_k: int,
    dense_fetch_k: int,
    bm25_fetch_k: int,
    dense_vector_name: str,
    bm25_vector_name: str,
    bm25_options: dict[str, object],
    rrf_weights: list[float] | None = None,
    bm25_excluded_role_hints: list[str] | None = None,
    document_id: str | None = None,
    restrict_to_document: bool = True,
    score_threshold: float | None = None,
) -> list[dict[str, object]]:
    """dense prefetch + bm25 prefetch + RRF fusion으로 chunk top-k를 조회한다."""
    query_filter = None
    if restrict_to_document and document_id:
        query_filter = build_document_filter(document_id)

    dense_prefetch: dict[str, object] = {
        "query": query_vector,
        "using": dense_vector_name,
        "limit": dense_fetch_k,
    }
    if query_filter:
        dense_prefetch["filter"] = query_filter
    if score_threshold is not None:
        dense_prefetch["score_threshold"] = score_threshold

    bm25_prefetch: dict[str, object] = {
        "query": {
            "text": query_text,
            "model": "qdrant/bm25",
            "options": bm25_options,
        },
        "using": bm25_vector_name,
        "limit": bm25_fetch_k,
    }
    bm25_filter = build_bm25_filter(
        document_id=document_id,
        restrict_to_document=restrict_to_document,
        excluded_role_hints=bm25_excluded_role_hints,
    )
    if bm25_filter:
        bm25_prefetch["filter"] = bm25_filter

    fusion_query: dict[str, object]
    if rrf_weights:
        fusion_query = {"rrf": {"weights": rrf_weights}}
    else:
        fusion_query = {"fusion": "rrf"}

    return qdrant_client.query_points(
        collection_name=collection_name,
        prefetch=[dense_prefetch, bm25_prefetch],
        query=fusion_query,
        limit=top_k,
        with_payload=True,
        with_vector=False,
    )
