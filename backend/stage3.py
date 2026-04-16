"""Backward-compatible entrypoint for stage-3 chunking + indexing."""

from __future__ import annotations

import json
from typing import Any

from backend.stage3_chunking.pipeline import (
    build_stage3_output_paths,
    prepare_stage3_chunking,
    run_stage3_chunking,
)
from backend.stage3_indexing.pipeline import (
    build_stage3_index_output_paths,
    prepare_stage3_indexing,
    run_stage3_indexing,
)


def run_stage3(inputs: dict[str, Any] | None = None) -> dict[str, Any]:
    """stage3 chunking과 indexing을 순차 실행한다."""
    resolved_inputs = dict(inputs or {})
    chunking_output = run_stage3_chunking(resolved_inputs)
    indexing_output = run_stage3_indexing(
        {
            "chunks_json_path": chunking_output["output_paths"]["chunks_json"],
            "output_dir": chunking_output["output_dir"],
            "document_id": resolved_inputs.get("document_id"),
            "collection_name": resolved_inputs.get("collection_name"),
        }
    )
    return {
        "chunking": chunking_output,
        "indexing": indexing_output,
        "status": (
            "completed"
            if indexing_output["status"] == "completed"
            else "completed_with_indexing_skip"
        ),
    }


def main() -> None:
    """기본 cleaned.json 경로를 기준으로 stage3 전체 파이프라인을 실행한다."""
    response = run_stage3()
    print(json.dumps(response, ensure_ascii=False, indent=2))


__all__ = [
    "build_stage3_output_paths",
    "build_stage3_index_output_paths",
    "prepare_stage3_chunking",
    "prepare_stage3_indexing",
    "run_stage3_chunking",
    "run_stage3_indexing",
    "run_stage3",
    "main",
]


if __name__ == "__main__":
    main()
