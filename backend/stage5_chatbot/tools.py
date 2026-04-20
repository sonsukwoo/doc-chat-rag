"""Stage-5 chatbot tools.

ToolNode에 연결할 툴 집합이다.
thread/document 범위는 모델 인자가 아니라 runtime.state에서 읽는다.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from langchain.tools import ToolRuntime, tool
from langchain_core.tools import BaseTool
from backend.thread_identity import build_thread_collection_name

from .document_selection import extract_explicit_document_ids, iter_ordered_document_profiles


def build_list_thread_documents_tool(
) -> BaseTool:
    """현재 thread에 연결된 문서 목록 조회 툴을 만든다."""

    @tool("list_thread_documents")
    def list_thread_documents(runtime: ToolRuntime) -> str:
        """현재 스레드에 연결된 문서 ID 목록을 반환합니다."""
        state = dict(runtime.state or {})
        document_ids = [
            str(item)
            for item in state.get("active_document_ids") or []
            if str(item)
        ]
        if not document_ids:
            return "현재 스레드에 연결된 문서가 없습니다."
        return json.dumps(
            {"document_ids": document_ids},
            ensure_ascii=False,
        )

    return list_thread_documents


def build_search_thread_knowledge_tool(
    *,
    stage4_runner: Callable[..., dict[str, Any]] | None = None,
) -> BaseTool:
    """현재 thread 범위 검색 툴을 만든다."""

    @tool("search_thread_knowledge")
    def search_thread_knowledge(query: str, runtime: ToolRuntime) -> str:
        """현재 스레드 범위에서 관련 문서를 검색합니다."""
        state = dict(runtime.state or {})
        resolved_thread_id = str(state.get("thread_id") or "").strip()
        document_ids = [
            str(item)
            for item in (
                state.get("retrieval_document_ids")
                or state.get("active_document_ids")
                or []
            )
            if str(item)
        ]
        ordered_profiles = iter_ordered_document_profiles(
            state.get("active_document_ids") or document_ids,
            state.get("document_profiles") or [],
        )
        explicit_document_ids = extract_explicit_document_ids(query, ordered_profiles)
        if explicit_document_ids:
            document_ids = [
                document_id
                for document_id in document_ids
                if document_id in explicit_document_ids
            ] or explicit_document_ids
        collection_name = str(state.get("collection_name") or "").strip() or None
        if collection_name is None and resolved_thread_id:
            collection_name = build_thread_collection_name(resolved_thread_id)
        retrieval_policy = dict(state.get("retrieval_policy") or {})
        retrieval_mode = str(retrieval_policy.get("mode") or "dense").strip() or "dense"

        if stage4_runner is None:
            return json.dumps(
                {
                    "status": "scaffold_only",
                    "message": "stage4 retrieval integration pending",
                    "thread_id": resolved_thread_id,
                    "document_ids": document_ids,
                    "query": query,
                },
                ensure_ascii=False,
            )

        result = stage4_runner(
            query=query,
            thread_id=resolved_thread_id,
            active_document_ids=document_ids,
            document_queries=dict(state.get("retrieval_document_queries") or {}),
            collection_name=collection_name,
            retrieval_mode=retrieval_mode,
            top_k=int(retrieval_policy.get("top_k") or 8),
            use_per_document_search=bool(state.get("use_per_document_search")) and len(document_ids) > 1,
            per_document_top_k=8 if len(document_ids) > 1 else None,
            score_threshold=retrieval_policy.get("score_threshold"),
            enable_rerank=bool(retrieval_policy.get("use_rerank", True)),
            enable_mmr=bool(retrieval_policy.get("enable_mmr", False)),
        )
        result = dict(result or {})
        result["rerank_requested"] = bool(retrieval_policy.get("use_rerank", True))
        result["mmr_requested"] = bool(retrieval_policy.get("enable_mmr", False))
        return json.dumps(result, ensure_ascii=False)

    return search_thread_knowledge

def build_expand_context_window_tool(
    *,
    context_window_loader: Callable[..., list[dict[str, Any]]] | None = None,
) -> BaseTool:
    """window/parent 확장 툴을 만든다."""

    @tool("expand_context_window")
    def expand_context_window(chunk_ids: list[str], runtime: ToolRuntime) -> str:
        """검색된 child chunk 기준으로 인접 문맥을 확장합니다."""
        state = dict(runtime.state or {})
        resolved_thread_id = str(state.get("thread_id") or "").strip() or None
        document_ids = [
            str(item)
            for item in (
                state.get("retrieval_document_ids")
                or state.get("active_document_ids")
                or []
            )
            if str(item)
        ]
        window_size = 1
        if callable(context_window_loader):
            blocks = context_window_loader(
                thread_id=resolved_thread_id,
                active_document_ids=document_ids,
                chunk_ids=chunk_ids,
                window_size=window_size,
            )
            return json.dumps(
                {
                    "status": "completed",
                    "chunk_ids": chunk_ids,
                    "blocks": blocks,
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "scaffold_only",
                "message": "context expansion integration pending",
                "chunk_ids": chunk_ids,
            },
            ensure_ascii=False,
        )

    return expand_context_window


def build_load_visual_asset_tool(
    *,
    visual_asset_loader: Callable[..., list[dict[str, Any]]] | None = None,
) -> BaseTool:
    """표/이미지 asset 조회 툴을 만든다."""

    @tool("load_visual_asset")
    def load_visual_asset(asset_ref: str, runtime: ToolRuntime) -> str:
        """표/이미지 원본 asset 메타데이터를 불러옵니다."""
        state = dict(runtime.state or {})
        resolved_thread_id = str(state.get("thread_id") or "").strip() or None
        document_ids = [
            str(item)
            for item in (
                state.get("retrieval_document_ids")
                or state.get("active_document_ids")
                or []
            )
            if str(item)
        ]
        if callable(visual_asset_loader):
            assets = visual_asset_loader(
                thread_id=resolved_thread_id,
                active_document_ids=document_ids,
                asset_refs=[asset_ref],
            )
            return json.dumps(
                {
                    "status": "completed",
                    "asset_ref": asset_ref,
                    "asset": assets[0] if assets else None,
                },
                ensure_ascii=False,
            )
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
    context_window_loader: Callable[..., list[dict[str, Any]]] | None = None,
    visual_asset_loader: Callable[..., list[dict[str, Any]]] | None = None,
) -> list[BaseTool]:
    """stage5 graph에 연결할 기본 툴 목록을 구성한다."""
    return [
        build_list_thread_documents_tool(),
        build_search_thread_knowledge_tool(
            stage4_runner=stage4_runner,
        ),
        build_expand_context_window_tool(
            context_window_loader=context_window_loader,
        ),
        build_load_visual_asset_tool(
            visual_asset_loader=visual_asset_loader,
        ),
        build_web_search_tool(enabled=allow_web_search),
    ]
