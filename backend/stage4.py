"""Backward-compatible entrypoint for stage-4 retrieval."""

from backend.stage4_retrieval import (
    DEFAULT_CHUNKS_JSON_PATH,
    DEFAULT_PARENTS_JSON_PATH,
    build_stage4_output_paths,
    main,
    prepare_stage4_retrieval,
    run_stage4_retrieval,
)

__all__ = [
    "DEFAULT_CHUNKS_JSON_PATH",
    "DEFAULT_PARENTS_JSON_PATH",
    "build_stage4_output_paths",
    "prepare_stage4_retrieval",
    "run_stage4_retrieval",
    "main",
]


if __name__ == "__main__":
    main()
