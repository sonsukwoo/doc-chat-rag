"""Backward-compatible entrypoint for stage-3 chunking."""

from backend.stage3_chunking.pipeline import (
    build_stage3_output_paths,
    main,
    prepare_stage3_chunking,
    run_stage3_chunking,
)

__all__ = [
    "build_stage3_output_paths",
    "prepare_stage3_chunking",
    "run_stage3_chunking",
    "main",
]


if __name__ == "__main__":
    main()
