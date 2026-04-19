"""Stage-5 chatbot tools.

ToolNode에 연결할 툴 집합이다.
room/document 범위는 모델 인자가 아니라 runtime.state에서 읽는다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool


def build_list_room_documents_tool(
) -> BaseTool:
    """현재 room에 연결된 문서 목록 조회 툴을 만든다."""

    @tool("list_room_documents")
    def list_room_documents(runtime: ToolRuntime) -> str:
        """현재 채팅방에 연결된 문서 ID 목록을 반환합니다."""
        state = dict(runtime.state or {})
        document_ids = [
            str(item)
            for item in state.get("active_document_ids") or []
            if str(item)
        ]
        if not document_ids:
            return "현재 채팅방에 연결된 문서가 없습니다."
        return json.dumps(
            {"document_ids": document_ids},
            ensure_ascii=False,
        )

    return list_room_documents


def build_search_room_knowledge_tool(
    *,
    stage4_runner: Callable[..., dict[str, Any]] | None = None,
) -> BaseTool:
    """현재 room 범위 검색 툴을 만든다."""

    @tool("search_room_knowledge")
    def search_room_knowledge(query: str, runtime: ToolRuntime) -> str:
        """현재 채팅방 범위에서 관련 문서를 검색합니다."""
        state = dict(runtime.state or {})
        resolved_room_id = str(state.get("room_id") or "").strip()
        document_ids = [
            str(item)
            for item in state.get("active_document_ids") or []
            if str(item)
        ]
        collection_name = str(state.get("collection_name") or "").strip() or None
        retrieval_policy = dict(state.get("retrieval_policy") or {})
        retrieval_mode = str(retrieval_policy.get("mode") or "dense").strip() or "dense"

        if stage4_runner is None:
            return json.dumps(
                {
                    "status": "scaffold_only",
                    "message": "stage4 retrieval integration pending",
                    "room_id": resolved_room_id,
                    "document_ids": document_ids,
                    "query": query,
                },
                ensure_ascii=False,
            )

        result = stage4_runner(
            query=query,
            room_id=resolved_room_id,
            active_document_ids=document_ids,
            collection_name=collection_name,
            retrieval_mode=retrieval_mode,
        )
        return json.dumps(result, ensure_ascii=False)

    return search_room_knowledge


def build_expand_context_window_tool() -> BaseTool:
    """window/parent 확장 툴 스캐폴드를 만든다."""

    @tool("expand_context_window")
    def expand_context_window(chunk_ids: list[str]) -> str:
        """검색된 child chunk 기준으로 인접 문맥을 확장합니다."""
        return json.dumps(
            {
                "status": "scaffold_only",
                "message": "context expansion integration pending",
                "chunk_ids": chunk_ids,
            },
            ensure_ascii=False,
        )

    return expand_context_window


def build_load_visual_asset_tool() -> BaseTool:
    """표/이미지 asset 조회 툴 스캐폴드를 만든다."""

    @tool("load_visual_asset")
    def load_visual_asset(asset_ref: str) -> str:
        """표/이미지 원본 asset 메타데이터를 불러옵니다."""
        return json.dumps(
            {
                "status": "scaffold_only",
                "message": "visual asset lookup pending",
                "asset_ref": asset_ref,
            },
            ensure_ascii=False,
        )

    return load_visual_asset


def build_web_search_tool(*, enabled: bool = False) -> BaseTool:
    """웹 검색 툴 스캐폴드를 만든다."""

    @tool("web_search")
    def web_search(query: str) -> str:
        """문서 근거가 부족할 때 외부 웹 검색을 수행합니다."""
        if not enabled:
            return json.dumps(
                {
                    "status": "disabled",
                    "message": "web search is disabled",
                    "query": query,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "scaffold_only",
                "message": "web search integration pending",
                "query": query,
            },
            ensure_ascii=False,
        )

    return web_search


def build_stage5_tools(
    *,
    allow_web_search: bool = False,
    stage4_runner: Callable[..., dict[str, Any]] | None = None,
) -> list[BaseTool]:
    """stage5 graph에 연결할 기본 툴 목록을 구성한다."""
    return [
        build_list_room_documents_tool(),
        build_search_room_knowledge_tool(
            stage4_runner=stage4_runner,
        ),
        build_expand_context_window_tool(),
        build_load_visual_asset_tool(),
        build_web_search_tool(enabled=allow_web_search),
    ]
