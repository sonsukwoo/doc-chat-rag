"""Stage-3 indexing input/output schemas."""

from __future__ import annotations

from typing import Literal, TypedDict


class Stage3IndexInput(TypedDict, total=False):
    """stage3 indexing 진입 시 외부에서 전달하는 최소 입력."""

    chunks_json_path: str
    output_dir: str
    document_id: str
    thread_id: str
    collection_name: str


class Stage3IndexOutputPaths(TypedDict):
    """stage3 indexing이 기록할 산출물 경로 묶음."""

    indexing_manifest: str


class Stage3IndexOutput(TypedDict, total=False):
    """stage3 indexing 실행 후 외부에서 읽을 출력 메타데이터."""

    chunks_json_path: str
    output_dir: str
    document_id: str
    thread_id: str | None
    collection_name: str
    output_paths: Stage3IndexOutputPaths
    planned_outputs: Stage3IndexOutputPaths
    point_count: int
    vector_size: int
    indexing_mode: str
    dense_vector_name: str
    bm25_vector_name: str
    indexing_enabled: bool
    status: Literal["completed", "skipped"]
    skip_reason: str | None
