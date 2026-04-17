"""Stage-4 retrieval input/output schemas."""

from __future__ import annotations

from typing import Literal, TypedDict


class Stage4Input(TypedDict, total=False):
    """stage4 retrieval 진입 시 외부에서 전달하는 최소 입력."""

    query: str
    chunks_json_path: str
    parents_json_path: str
    output_dir: str
    document_id: str
    collection_name: str
    retrieval_mode: str
    top_k: int
    fetch_k: int
    dense_fetch_k: int
    bm25_fetch_k: int
    hybrid_rrf_weights: list[float]
    bm25_excluded_role_hints: list[str]
    restrict_to_document: bool
    score_threshold: float
    enable_score_fallback: bool
    enable_rerank: bool
    rerank_model: str
    rerank_device: str
    enable_mmr: bool
    mmr_lambda_mult: float
    parent_expand_mode: str
    parent_window_size: int


class Stage4OutputPaths(TypedDict):
    """stage4 retrieval이 기록할 산출물 경로 묶음."""

    retrieval_manifest: str


class RetrievedChunkPayload(TypedDict, total=False):
    """향후 dense search 결과를 담을 기본 retrieval 단위."""

    point_id: str
    document_id: str
    chunk_id: str
    parent_id: str | None
    score: float
    dense_score: float | None
    bm25_score: float | None
    chunk_type: str
    text: str
    section_title: str | None
    primary_page: int | None
    page_start: int | None
    page_end: int | None
    has_asset: bool
    asset_kind: str | None
    asset_relative_path: str | None
    caption: str | None
    parent_section_title: str | None
    parent_page_start: int | None
    parent_page_end: int | None
    context_text: str | None
    context_chunk_ids: list[str]
    expansion_mode: str | None


class Stage4Output(TypedDict, total=False):
    """stage4 retrieval 실행 후 외부에서 읽을 출력 메타데이터."""

    query: str
    chunks_json_path: str
    parents_json_path: str | None
    output_dir: str
    document_id: str
    collection_name: str
    retrieval_mode: str
    top_k: int
    fetch_k: int
    dense_fetch_k: int
    bm25_fetch_k: int
    hybrid_rrf_weights: list[float] | None
    bm25_excluded_role_hints: list[str]
    score_threshold_requested: float | None
    score_threshold_applied: float | None
    score_fallback_applied: bool
    rerank_enabled: bool
    rerank_applied: bool
    rerank_model: str
    rerank_device: str
    rerank_error: str | None
    mmr_enabled: bool
    mmr_applied: bool
    mmr_lambda_mult: float
    parent_expand_mode: str
    parent_window_size: int
    chunk_count: int
    parent_count: int
    fetched_count: int
    retrieved_count: int
    qdrant_configured: bool
    document_filter_applied: bool
    retrievals: list[RetrievedChunkPayload]
    output_paths: Stage4OutputPaths
    planned_outputs: Stage4OutputPaths
    status: Literal["completed", "skipped"]
    skip_reason: str | None
