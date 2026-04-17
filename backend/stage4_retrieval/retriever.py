"""LangChain retriever adapters for stage-4 retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pydantic import ConfigDict

from backend.stage3_chunking.embeddings import OpenAIEmbeddingClient
from backend.stage3_indexing.qdrant import QdrantRestClient

from .qdrant import search_dense_chunks, search_hybrid_chunks


def _to_langchain_document(
    *,
    point: dict[str, Any],
    retrieval_mode: str,
) -> Document:
    """Qdrant point를 LangChain Document로 정규화한다."""
    payload = point.get("payload") or {}
    text = str(payload.get("text") or "")
    score = float(point.get("score") or 0.0)
    metadata = {
        "point_id": str(point.get("id") or ""),
        "document_id": (
            str(payload.get("document_id"))
            if payload.get("document_id") not in (None, "")
            else None
        ),
        "chunk_id": (
            str(payload.get("chunk_id"))
            if payload.get("chunk_id") not in (None, "")
            else None
        ),
        "parent_id": (
            str(payload.get("parent_id"))
            if payload.get("parent_id") not in (None, "")
            else None
        ),
        "score": score,
        "dense_score": score if retrieval_mode == "dense" else None,
        "bm25_score": None,
        "chunk_type": (
            str(payload.get("chunk_type"))
            if payload.get("chunk_type") not in (None, "")
            else None
        ),
        "section_title": (
            str(payload.get("section_title"))
            if payload.get("section_title") not in (None, "")
            else None
        ),
        "primary_page": payload.get("primary_page"),
        "page_start": payload.get("page_start"),
        "page_end": payload.get("page_end"),
        "has_asset": bool(payload.get("has_asset")),
        "asset_kind": (
            str(payload.get("asset_kind"))
            if payload.get("asset_kind") not in (None, "")
            else None
        ),
        "asset_relative_path": (
            str(payload.get("asset_relative_path"))
            if payload.get("asset_relative_path") not in (None, "")
            else None
        ),
        "caption": (
            str(payload.get("caption"))
            if payload.get("caption") not in (None, "")
            else None
        ),
    }
    return Document(page_content=text, metadata=metadata)


class QdrantChunkRetriever(BaseRetriever):
    """Qdrant dense/hybrid 검색을 LangChain BaseRetriever로 감싼다."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # 테스트용 fake client와 실제 client를 모두 받기 위해 Any로 둔다.
    embedding_client: Any
    qdrant_client: Any
    collection_name: str
    retrieval_mode: str = "hybrid"
    fetch_limit: int = 8
    dense_fetch_k: int = 8
    bm25_fetch_k: int = 8
    dense_vector_name: str = "dense"
    bm25_vector_name: str = "bm25"
    bm25_options: dict[str, object]
    rrf_weights: list[float] | None = None
    bm25_excluded_role_hints: list[str] | None = None
    document_id: str | None = None
    restrict_to_document: bool = True
    score_threshold: float | None = None

    def _embed_query(self, query: str) -> list[float]:
        embeddings = self.embedding_client.embed_texts([query])
        if embeddings is None or not embeddings or not embeddings[0]:
            raise RuntimeError(
                "query embedding 생성에 실패했습니다: "
                f"{self.embedding_client.last_error or 'unknown_error'}"
            )
        return embeddings[0]

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> list[Document]:
        """질의 문자열을 받아 LangChain Document 목록으로 반환한다."""
        query_vector = self._embed_query(query)

        if self.retrieval_mode == "dense":
            points = search_dense_chunks(
                qdrant_client=self.qdrant_client,
                collection_name=self.collection_name,
                query_vector=query_vector,
                top_k=max(self.fetch_limit, self.dense_fetch_k),
                dense_vector_name=self.dense_vector_name,
                document_id=self.document_id,
                restrict_to_document=self.restrict_to_document,
                score_threshold=self.score_threshold,
            )
        else:
            points = search_hybrid_chunks(
                qdrant_client=self.qdrant_client,
                collection_name=self.collection_name,
                query_text=query,
                query_vector=query_vector,
                top_k=self.fetch_limit,
                dense_fetch_k=max(self.fetch_limit, self.dense_fetch_k),
                bm25_fetch_k=max(self.fetch_limit, self.bm25_fetch_k),
                dense_vector_name=self.dense_vector_name,
                bm25_vector_name=self.bm25_vector_name,
                bm25_options=self.bm25_options,
                rrf_weights=self.rrf_weights,
                bm25_excluded_role_hints=self.bm25_excluded_role_hints,
                document_id=self.document_id,
                restrict_to_document=self.restrict_to_document,
                score_threshold=self.score_threshold,
            )

        return [
            _to_langchain_document(
                point=point,
                retrieval_mode=self.retrieval_mode,
            )
            for point in points
        ]


def build_qdrant_chunk_retriever(
    *,
    embedding_client: Any,
    qdrant_client: Any,
    collection_name: str,
    retrieval_mode: str,
    fetch_limit: int,
    dense_fetch_k: int,
    bm25_fetch_k: int,
    dense_vector_name: str,
    bm25_vector_name: str,
    bm25_options: dict[str, object],
    rrf_weights: list[float] | None,
    bm25_excluded_role_hints: list[str] | None,
    document_id: str | None,
    restrict_to_document: bool,
    score_threshold: float | None,
) -> QdrantChunkRetriever:
    """pipeline에서 동일한 retriever 설정을 재사용하기 위한 factory다."""
    return QdrantChunkRetriever(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        retrieval_mode=retrieval_mode,
        fetch_limit=fetch_limit,
        dense_fetch_k=dense_fetch_k,
        bm25_fetch_k=bm25_fetch_k,
        dense_vector_name=dense_vector_name,
        bm25_vector_name=bm25_vector_name,
        bm25_options=bm25_options,
        rrf_weights=rrf_weights,
        bm25_excluded_role_hints=bm25_excluded_role_hints,
        document_id=document_id,
        restrict_to_document=restrict_to_document,
        score_threshold=score_threshold,
    )
