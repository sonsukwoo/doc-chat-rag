"""Backward-compatible entrypoint for stage-4 retrieval."""

from backend.stage4_retrieval import (
    DEFAULT_CHUNKS_JSON_PATH,
    DEFAULT_PARENTS_JSON_PATH,
    build_stage4_output_paths,
    main,
    run_stage4_retrieval,
    search_thread_knowledge,
)

__all__ = [
    "DEFAULT_CHUNKS_JSON_PATH",
    "DEFAULT_PARENTS_JSON_PATH",
    "build_stage4_output_paths",
    "run_stage4_retrieval",
    "search_thread_knowledge",
    "main",
]


if __name__ == "__main__":
    main()
