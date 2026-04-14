"""Stage-3 chunking input/output schemas."""

from __future__ import annotations

from typing import Any, Literal, TypedDict

ElementPayload = dict[str, Any]


class Stage3Input(TypedDict, total=False):
    """stage3 진입 시 외부에서 전달하는 최소 입력."""

    cleaned_json_path: str
    output_dir: str


class ChunkSourceElement(TypedDict):
    """하나의 chunk가 어떤 element들에서 왔는지 추적하는 메타데이터."""

    element_id: int
    page: int
    category: str


class ChunkPayload(TypedDict, total=False):
    """retrieval과 embedding에 바로 사용할 chunk 기본 스키마."""

    chunk_id: str
    chunk_type: Literal["text", "table", "figure", "mixed"]
    text: str
    pages: list[int]
    heading_path: list[str]
    element_ids: list[int]
    source_elements: list[ChunkSourceElement]
    metadata: dict[str, Any]


class Stage3OutputPaths(TypedDict):
    """stage3가 기록할 산출물 경로 묶음."""

    chunks_json: str
    chunks_jsonl: str
    chunks_md: str


class Stage3ChunkStats(TypedDict):
    """생성된 chunk 분포와 semantic 적용 현황."""

    total_chunks: int
    text_chunks: int
    table_chunks: int
    figure_chunks: int
    semantic_split_chunks: int
    semantic_merge_chunks: int


class Stage3Output(TypedDict, total=False):
    """stage3 실행 후 외부에서 읽을 출력 메타데이터."""

    cleaned_json_path: str
    output_dir: str
    output_paths: Stage3OutputPaths
    planned_outputs: Stage3OutputPaths
    chunk_count: int
    stats: Stage3ChunkStats
    semantic_enabled: bool
    semantic_fallback_reason: str | None
    status: Literal["completed", "completed_with_semantic_fallback"]
