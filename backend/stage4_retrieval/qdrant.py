"""Stage-4 retrieval helpers built on top of the shared Qdrant REST client."""

from __future__ import annotations

from backend.stage3_indexing.qdrant import QdrantRestClient


def build_scope_filter(
    *,
    room_id: str | None = None,
    document_id: str | None = None,
    active_document_ids: list[str] | None = None,
    restrict_to_document: bool = True,
) -> dict[str, object] | None:
    """room/document 범위를 함께 제한하는 Qdrant filter를 만든다."""
    must: list[dict[str, object]] = []

    if room_id:
        must.append(
            {
                "key": "room_id",
                "match": {"value": room_id},
            }
        )

    document_ids = [
        str(item).strip()
        for item in active_document_ids or []
        if str(item).strip()
    ]
    if restrict_to_document:
        if document_ids:
            must.append(
                {
                    "key": "document_id",
                    "match": {"any": document_ids},
                }
            )
        elif document_id:
            must.append(
                {
                    "key": "document_id",
                    "match": {"value": document_id},
                }
            )

    if not must:
        return None
    return {"must": must}


def build_bm25_filter(
    *,
    bm25_vector_name: str,
    room_id: str | None = None,
    document_id: str | None = None,
    active_document_ids: list[str] | None = None,
    restrict_to_document: bool = True,
    excluded_role_hints: list[str] | None = None,
) -> dict[str, object] | None:
    """BM25 브랜치 전용 필터를 만든다."""
    base_filter = build_scope_filter(
        room_id=room_id,
        document_id=document_id,
        active_document_ids=active_document_ids,
        restrict_to_document=restrict_to_document,
    )
    must: list[dict[str, object]] = list((base_filter or {}).get("must") or [])
    must_not: list[dict[str, object]] = []
    must.append(
        {
            "key": "sparse_keep",
            "match": {"value": True},
        }
    )
    must.append({"has_vector": bm25_vector_name})

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
    room_id: str | None = None,
    document_id: str | None = None,
    active_document_ids: list[str] | None = None,
    restrict_to_document: bool = True,
    score_threshold: float | None = None,
) -> list[dict[str, object]]:
    """dense query vector로 chunk top-k를 조회한다."""
    query_filter = build_scope_filter(
        room_id=room_id,
        document_id=document_id,
        active_document_ids=active_document_ids,
        restrict_to_document=restrict_to_document,
    )

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
    room_id: str | None = None,
    document_id: str | None = None,
    active_document_ids: list[str] | None = None,
    restrict_to_document: bool = True,
    score_threshold: float | None = None,
) -> list[dict[str, object]]:
    """dense prefetch + bm25 prefetch + RRF fusion으로 chunk top-k를 조회한다."""
    query_filter = build_scope_filter(
        room_id=room_id,
        document_id=document_id,
        active_document_ids=active_document_ids,
        restrict_to_document=restrict_to_document,
    )

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
        bm25_vector_name=bm25_vector_name,
        room_id=room_id,
        document_id=document_id,
        active_document_ids=active_document_ids,
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
