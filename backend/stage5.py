"""Backward-compatible entrypoint for stage-5 chatbot."""

from backend.stage5_chatbot import build_graph, get_agent, run_stage5_chatbot

__all__ = [
    "build_graph",
    "get_agent",
    "run_stage5_chatbot",
]
