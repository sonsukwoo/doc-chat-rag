"""Stage-4 dense retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.stage3_chunking.embeddings import OpenAIEmbeddingClient
from backend.stage3_indexing.qdrant import QdrantRestClient

from .config import (
    DEFAULT_CHUNKS_JSON_PATH,
    DEFAULT_PARENTS_JSON_PATH,
    DEFAULT_RETRIEVAL_MANIFEST_NAME,
    STAGE4_FETCH_K,
    STAGE4_QDRANT_COLLECTION_NAME,
    STAGE4_QDRANT_API_KEY,
    STAGE4_QDRANT_TIMEOUT,
    STAGE4_QDRANT_URL,
    STAGE4_RESTRICT_TO_DOCUMENT,
    STAGE4_SCORE_THRESHOLD,
    STAGE4_TOP_K,
)
from .parents import load_parent_lookup
from .qdrant import search_dense_chunks
from .schemas import Stage4Input, Stage4Output, Stage4OutputPaths


def build_stage4_output_paths(
    *,
    chunks_json_path: str | Path,
    output_dir: str | Path | None = None,
) -> Stage4OutputPaths:
    """stage4가 기록할 retrieval 산출물 경로를 계산한다."""
    chunks_path = Path(chunks_json_path).expanduser().resolve()
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else chunks_path.parent.resolve()
    )
    return {
        "retrieval_manifest": str(
            (resolved_output_dir / DEFAULT_RETRIEVAL_MANIFEST_NAME).resolve()
        )
    }


def _load_json_document(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name}은 dict 형태여야 합니다.")
    return payload


def _derive_document_id(
    *,
    chunks_json_path: Path,
    chunks_document: dict[str, Any],
    explicit_document_id: str | None,
) -> str:
    if explicit_document_id:
        return explicit_document_id

    cleaned_json_path = chunks_document.get("cleaned_json_path")
    if cleaned_json_path:
        return Path(str(cleaned_json_path)).expanduser().resolve().parent.name

    return chunks_json_path.parent.name


def _resolve_parents_json_path(
    *,
    explicit_parents_json_path: str | None,
    output_dir: Path,
) -> Path | None:
    if explicit_parents_json_path:
        return Path(explicit_parents_json_path).expanduser().resolve()

    inferred_path = (output_dir / DEFAULT_PARENTS_JSON_PATH.name).resolve()
    if inferred_path.exists():
        return inferred_path
    return None


def _to_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _normalize_retrieval_hit(
    *,
    point: dict[str, Any],
    fallback_document_id: str,
    parent_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = point.get("payload") or {}
    parent_id = str(payload.get("parent_id") or "").strip() or None
    parent = parent_lookup.get(parent_id or "")

    return {
        "point_id": str(point.get("id") or ""),
        "document_id": str(payload.get("document_id") or fallback_document_id),
        "chunk_id": str(payload.get("chunk_id") or ""),
        "parent_id": parent_id,
        "score": float(point.get("score") or 0.0),
        "dense_score": float(point.get("score") or 0.0),
        "chunk_type": str(payload.get("chunk_type") or ""),
        "text": str(payload.get("text") or ""),
        "section_title": (
            str(payload.get("section_title"))
            if payload.get("section_title") not in (None, "")
            else None
        ),
        "primary_page": _to_optional_int(payload.get("primary_page")),
        "page_start": _to_optional_int(payload.get("page_start")),
        "page_end": _to_optional_int(payload.get("page_end")),
        "has_asset": bool(payload.get("has_asset")),
        "asset_kind": (
            str(payload.get("asset_kind"))
            if payload.get("asset_kind") not in (None, "")
            else None
        ),
        "asset_relative_path": (
            str(payload.get("asset_relative_path"))
            if payload.get("asset_relative_path") not in (None, "")
            else None
        ),
        "caption": (
            str(payload.get("caption"))
            if payload.get("caption") not in (None, "")
            else None
        ),
        "parent_section_title": (
            str(parent.get("section_title"))
            if parent and parent.get("section_title") not in (None, "")
            else None
        ),
        "parent_page_start": _to_optional_int(
            parent.get("page_start") if parent else None
        ),
        "parent_page_end": _to_optional_int(
            parent.get("page_end") if parent else None
        ),
    }


def _write_retrieval_manifest(
    output: Stage4Output,
    *,
    output_paths: Stage4OutputPaths,
) -> None:
    manifest_path = Path(output_paths["retrieval_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))


def run_stage4_retrieval(
    inputs: Stage4Input | None = None,
    *,
    embedding_client: OpenAIEmbeddingClient | None = None,
    qdrant_client: QdrantRestClient | None = None,
    persist_manifest: bool = False,
) -> Stage4Output:
    """query embedding 생성 후 dense top-k retrieval을 수행한다."""
    resolved_inputs = dict(inputs or {})

    chunks_json_path = Path(
        resolved_inputs.get("chunks_json_path") or DEFAULT_CHUNKS_JSON_PATH
    ).expanduser().resolve()
    output_dir = (
        Path(resolved_inputs["output_dir"]).expanduser().resolve()
        if resolved_inputs.get("output_dir")
        else chunks_json_path.parent.resolve()
    )
    output_paths = build_stage4_output_paths(
        chunks_json_path=chunks_json_path,
        output_dir=output_dir,
    )

    chunks_document = _load_json_document(chunks_json_path)
    parents_json_path = _resolve_parents_json_path(
        explicit_parents_json_path=resolved_inputs.get("parents_json_path"),
        output_dir=output_dir,
    )
    parents_document = (
        _load_json_document(parents_json_path)
        if parents_json_path is not None and parents_json_path.exists()
        else {"parents": []}
    )
    parent_lookup = load_parent_lookup(parents_json_path)

    document_id = _derive_document_id(
        chunks_json_path=chunks_json_path,
        chunks_document=chunks_document,
        explicit_document_id=resolved_inputs.get("document_id"),
    )
    collection_name = (
        resolved_inputs.get("collection_name")
        or STAGE4_QDRANT_COLLECTION_NAME
    )
    query = str(resolved_inputs.get("query") or "").strip()
    top_k = int(resolved_inputs.get("top_k") or STAGE4_TOP_K)
    fetch_k = int(resolved_inputs.get("fetch_k") or STAGE4_FETCH_K)
    restrict_to_document = bool(
        resolved_inputs.get("restrict_to_document", STAGE4_RESTRICT_TO_DOCUMENT)
    )
    score_threshold = resolved_inputs.get("score_threshold", STAGE4_SCORE_THRESHOLD)
    qdrant_configured = bool(
        (qdrant_client is not None or STAGE4_QDRANT_URL) and collection_name
    )

    base_output: Stage4Output = {
        "query": query,
        "chunks_json_path": str(chunks_json_path),
        "parents_json_path": (
            str(parents_json_path) if parents_json_path is not None else None
        ),
        "output_dir": str(output_dir),
        "document_id": document_id,
        "collection_name": collection_name,
        "top_k": top_k,
        "fetch_k": max(top_k, fetch_k),
        "chunk_count": len(list(chunks_document.get("chunks") or [])),
        "parent_count": len(list(parents_document.get("parents") or [])),
        "fetched_count": 0,
        "retrieved_count": 0,
        "qdrant_configured": qdrant_configured,
        "document_filter_applied": bool(restrict_to_document and document_id),
        "retrievals": [],
        "output_paths": output_paths,
        "planned_outputs": output_paths,
    }

    if not query:
        output: Stage4Output = {
            **base_output,
            "status": "skipped",
            "skip_reason": "missing_query",
        }
        if persist_manifest:
            _write_retrieval_manifest(output, output_paths=output_paths)
        return output

    if not qdrant_configured:
        output = {
            **base_output,
            "status": "skipped",
            "skip_reason": "missing_qdrant_config",
        }
        if persist_manifest:
            _write_retrieval_manifest(output, output_paths=output_paths)
        return output

    embedding_client = embedding_client or OpenAIEmbeddingClient(enabled=True)
    embeddings = embedding_client.embed_texts([query])
    if embeddings is None or not embeddings or not embeddings[0]:
        raise RuntimeError(
            f"query embedding 생성에 실패했습니다: {embedding_client.last_error or 'unknown_error'}"
        )

    created_qdrant_client = False
    if qdrant_client is None:
        qdrant_client = QdrantRestClient(
            base_url=STAGE4_QDRANT_URL,
            api_key=STAGE4_QDRANT_API_KEY,
            timeout=STAGE4_QDRANT_TIMEOUT,
        )
        created_qdrant_client = True

    try:
        points = search_dense_chunks(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_vector=embeddings[0],
            top_k=max(top_k, fetch_k),
            document_id=document_id,
            restrict_to_document=restrict_to_document,
            score_threshold=(
                float(score_threshold)
                if score_threshold is not None
                else None
            ),
        )
    finally:
        if created_qdrant_client:
            qdrant_client.close()

    retrievals = [
        _normalize_retrieval_hit(
            point=point,
            fallback_document_id=document_id,
            parent_lookup=parent_lookup,
        )
        for point in points
    ][:top_k]

    output = {
        **base_output,
        "fetched_count": len(points),
        "retrieved_count": len(retrievals),
        "retrievals": retrievals,
        "status": "completed",
        "skip_reason": None,
    }
    if persist_manifest:
        _write_retrieval_manifest(output, output_paths=output_paths)
    return output


def prepare_stage4_retrieval(inputs: Stage4Input | None = None) -> Stage4Output:
    """기존 stage 패턴과 맞추기 위한 stage4 retrieval 래퍼다."""
    return run_stage4_retrieval(inputs)


def main() -> None:
    """기본 chunks.json 경로를 기준으로 stage4 retrieval 결과를 출력한다."""
    response = run_stage4_retrieval(
        {
            "chunks_json_path": str(DEFAULT_CHUNKS_JSON_PATH),
        },
        persist_manifest=True,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
