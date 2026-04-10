"""Backward-compatible entrypoint for stage-2 preprocessing graph."""

from backend.stage2_preprocess.graph import agent, main

__all__ = ["agent", "main"]


if __name__ == "__main__":
    main()
