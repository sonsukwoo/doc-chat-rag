"""Backward-compatible entrypoint for stage-1 Docling parsing."""

from backend.stage1_parse import (
    INPUT_PDF_PATH,
    OUTPUT_ROOT,
    UpstageStyleDoclingParser,
    main,
    run_stage1_parse,
)

__all__ = [
    "INPUT_PDF_PATH",
    "OUTPUT_ROOT",
    "UpstageStyleDoclingParser",
    "run_stage1_parse",
    "main",
]


if __name__ == "__main__":
    main()
