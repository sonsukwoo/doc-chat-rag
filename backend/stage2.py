"""Backward-compatible entrypoint for stage-2 preprocessing graph."""

from backend.stage2_preprocess.graph import agent, build_graph, get_agent, main

__all__ = ["agent", "build_graph", "get_agent", "main"]


if __name__ == "__main__":
    main()
