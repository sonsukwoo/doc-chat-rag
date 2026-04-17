"""LangChain-style context expansion helpers for stage-4 retrieval."""

from __future__ import annotations

from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda


def build_chunk_lookup(
    chunks_document: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """chunks.json payload를 chunk_id 기준 lookup으로 변환한다."""
    chunk_lookup: dict[str, dict[str, Any]] = {}
    for chunk in chunks_document.get("chunks") or []:
        if not isinstance(chunk, dict):
            continue
        chunk_id = str(chunk.get("chunk_id") or "").strip()
        if not chunk_id:
            continue
        chunk_lookup[chunk_id] = chunk
    return chunk_lookup


def expand_parent_context(
    *,
    documents: list[Document],
    chunk_lookup: dict[str, dict[str, Any]],
    parent_lookup: dict[str, dict[str, Any]],
    expand_mode: str,
    window_size: int,
) -> list[Document]:
    """retrieved child chunk에 answer용 parent/window 문맥을 추가한다."""
    normalized_mode = (expand_mode or "child").strip().lower()
    normalized_window_size = max(0, int(window_size))
    expanded_documents: list[Document] = []

    for document in documents:
        metadata = dict(document.metadata or {})
        chunk_id = str(metadata.get("chunk_id") or "").strip()
        parent_id = str(metadata.get("parent_id") or "").strip()
        context_chunk_ids = [chunk_id] if chunk_id else []
        context_text = document.page_content
        expansion_mode = "child"

        parent = parent_lookup.get(parent_id or "")
        child_chunk_ids = [
            str(item).strip()
            for item in (parent.get("child_chunk_ids") if parent else []) or []
            if str(item).strip()
        ]

        if normalized_mode == "parent" and child_chunk_ids:
            selected_chunk_ids = child_chunk_ids
            selected_texts = [
                str((chunk_lookup.get(selected_id) or {}).get("text") or "").strip()
                for selected_id in selected_chunk_ids
            ]
            joined_text = "\n\n".join(text for text in selected_texts if text)
            if joined_text:
                context_chunk_ids = selected_chunk_ids
                context_text = joined_text
                expansion_mode = "parent"
        elif normalized_mode == "window" and parent and chunk_id in child_chunk_ids:
            center_index = child_chunk_ids.index(chunk_id)
            start_index = max(0, center_index - normalized_window_size)
            end_index = min(
                len(child_chunk_ids),
                center_index + normalized_window_size + 1,
            )
            selected_chunk_ids = child_chunk_ids[start_index:end_index]
            selected_texts = [
                str((chunk_lookup.get(selected_id) or {}).get("text") or "").strip()
                for selected_id in selected_chunk_ids
            ]
            joined_text = "\n\n".join(text for text in selected_texts if text)
            if joined_text:
                context_chunk_ids = selected_chunk_ids
                context_text = joined_text
                expansion_mode = "window" if len(selected_chunk_ids) > 1 else "child"

        metadata["context_text"] = context_text
        metadata["context_chunk_ids"] = context_chunk_ids
        metadata["expansion_mode"] = expansion_mode
        expanded_documents.append(
            Document(page_content=document.page_content, metadata=metadata)
        )

    return expanded_documents


def build_context_expander(
    *,
    chunk_lookup: dict[str, dict[str, Any]],
    parent_lookup: dict[str, dict[str, Any]],
    expand_mode: str,
    window_size: int,
) -> RunnableLambda:
    """문서 목록 payload에 window/parent 문맥을 추가하는 Runnable 단계다."""

    def _expand(payload: dict[str, Any]) -> dict[str, Any]:
        documents = list(payload.get("documents") or [])
        expanded_documents = expand_parent_context(
            documents=documents,
            chunk_lookup=chunk_lookup,
            parent_lookup=parent_lookup,
            expand_mode=expand_mode,
            window_size=window_size,
        )
        return {
            **payload,
            "documents": expanded_documents,
        }

    return RunnableLambda(_expand)
