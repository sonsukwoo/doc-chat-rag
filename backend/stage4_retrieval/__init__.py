"""Stage-4 retrieval package."""

from .config import DEFAULT_CHUNKS_JSON_PATH, DEFAULT_PARENTS_JSON_PATH
from .pipeline import (
    build_stage4_output_paths,
    main,
    run_stage4_retrieval,
)
from .retriever import QdrantChunkRetriever, build_qdrant_chunk_retriever
from .rerank import build_cross_encoder_reranker
from .service import search_room_knowledge

__all__ = [
    "DEFAULT_CHUNKS_JSON_PATH",
    "DEFAULT_PARENTS_JSON_PATH",
    "QdrantChunkRetriever",
    "build_cross_encoder_reranker",
    "build_qdrant_chunk_retriever",
    "build_stage4_output_paths",
    "run_stage4_retrieval",
    "search_room_knowledge",
    "main",
]
