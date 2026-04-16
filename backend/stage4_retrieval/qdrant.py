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


def search_dense_chunks(
    *,
    qdrant_client: QdrantRestClient,
    collection_name: str,
    query_vector: list[float],
    top_k: int,
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
        query_filter=query_filter,
        score_threshold=score_threshold,
    )
