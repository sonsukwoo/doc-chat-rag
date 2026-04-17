"""Stage-4 retrieval pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_core.runnables import RunnableLambda

from backend.common import derive_document_id_from_artifact_path
from backend.stage3_chunking.embeddings import OpenAIEmbeddingClient
from backend.stage3_indexing.qdrant import QdrantRestClient

from .config import (
    DEFAULT_CHUNKS_JSON_PATH,
    DEFAULT_PARENTS_JSON_PATH,
    DEFAULT_RETRIEVAL_MANIFEST_NAME,
    STAGE4_BM25_ASCII_FOLDING,
    STAGE4_BM25_LANGUAGE,
    STAGE4_BM25_TOKENIZER,
    STAGE4_BM25_VECTOR_NAME,
    STAGE4_BM25_EXCLUDED_ROLE_HINTS,
    STAGE4_DENSE_VECTOR_NAME,
    STAGE4_ENABLE_MMR,
    STAGE4_ENABLE_SCORE_FALLBACK,
    STAGE4_FETCH_K,
    STAGE4_HYBRID_BM25_FETCH_K,
    STAGE4_HYBRID_DENSE_FETCH_K,
    STAGE4_HYBRID_RRF_WEIGHTS,
    STAGE4_MMR_LAMBDA_MULT,
    STAGE4_PARENT_EXPAND_MODE,
    STAGE4_PARENT_WINDOW_SIZE,
    STAGE4_QDRANT_COLLECTION_NAME,
    STAGE4_QDRANT_API_KEY,
    STAGE4_QDRANT_TIMEOUT,
    STAGE4_QDRANT_URL,
    STAGE4_RETRIEVAL_MODE,
    STAGE4_RESTRICT_TO_DOCUMENT,
    STAGE4_SCORE_THRESHOLD,
    STAGE4_TOP_K,
)
from .context import build_chunk_lookup, build_context_expander
from .parents import load_parent_lookup
from .postprocess import build_mmr_reranker
from .retriever import build_qdrant_chunk_retriever
from .schemas import Stage4Input, Stage4Output, Stage4OutputPaths


def build_stage4_output_paths(
    *,
    chunks_json_path: str | Path,
    output_dir: str | Path | None = None,
) -> Stage4OutputPaths:
    """stage4가 기록할 retrieval 산출물 경로를 계산한다."""
    chunks_path = Path(chunks_json_path).expanduser().resolve()
    default_output_dir = (
        chunks_path.parent.parent / "stage4"
        if chunks_path.parent.name == "stage3"
        else chunks_path.parent.resolve()
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else default_output_dir
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
    sibling_stage3_path = (output_dir.parent / "stage3" / DEFAULT_PARENTS_JSON_PATH.name).resolve()
    if output_dir.name == "stage4" and sibling_stage3_path.exists():
        return sibling_stage3_path
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
    document: Document,
    fallback_document_id: str,
    parent_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    payload = dict(document.metadata or {})
    parent_id = str(payload.get("parent_id") or "").strip() or None
    parent = parent_lookup.get(parent_id or "")

    return {
        "point_id": str(payload.get("point_id") or ""),
        "document_id": str(payload.get("document_id") or fallback_document_id),
        "chunk_id": str(payload.get("chunk_id") or ""),
        "parent_id": parent_id,
        "score": float(payload.get("score") or 0.0),
        "dense_score": (
            float(payload.get("dense_score"))
            if payload.get("dense_score") not in (None, "")
            else None
        ),
        "bm25_score": (
            float(payload.get("bm25_score"))
            if payload.get("bm25_score") not in (None, "")
            else None
        ),
        "chunk_type": str(payload.get("chunk_type") or ""),
        "text": str(document.page_content or ""),
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
        "context_text": (
            str(payload.get("context_text"))
            if payload.get("context_text") not in (None, "")
            else None
        ),
        "context_chunk_ids": [
            str(item)
            for item in payload.get("context_chunk_ids") or []
            if str(item)
        ],
        "expansion_mode": (
            str(payload.get("expansion_mode"))
            if payload.get("expansion_mode") not in (None, "")
            else None
        ),
    }


def _normalize_retrieval_documents(
    *,
    documents: list[Document],
    top_k: int,
    document_id: str,
    parent_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """LangChain Document 목록을 기존 stage4 출력 구조로 되돌린다."""
    retrievals = [
        _normalize_retrieval_hit(
            document=document,
            fallback_document_id=document_id,
            parent_lookup=parent_lookup,
        )
        for document in documents
    ][:top_k]

    return {
        "fetched_count": len(documents),
        "retrieved_count": len(retrievals),
        "retrievals": retrievals,
    }


def _build_retrieval_normalizer(
    *,
    top_k: int,
    document_id: str,
    parent_lookup: dict[str, dict[str, Any]],
) -> RunnableLambda:
    """문서 처리 payload를 stage4 retrieval 출력 구조로 바꾼다."""

    def _normalize(payload: dict[str, Any]) -> dict[str, Any]:
        documents = list(payload.get("documents") or [])
        normalized = _normalize_retrieval_documents(
            documents=documents,
            top_k=top_k,
            document_id=document_id,
            parent_lookup=parent_lookup,
        )
        return {
            **normalized,
            "mmr_applied": bool(payload.get("mmr_applied")),
        }

    return RunnableLambda(_normalize)


def _write_retrieval_manifest(
    output: Stage4Output,
    *,
    output_paths: Stage4OutputPaths,
) -> None:
    manifest_path = Path(output_paths["retrieval_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))


def _build_retriever(
    *,
    embedding_client: OpenAIEmbeddingClient,
    qdrant_client: QdrantRestClient,
    collection_name: str,
    retrieval_mode: str,
    top_k: int,
    effective_fetch_k: int,
    dense_fetch_k: int,
    bm25_fetch_k: int,
    hybrid_rrf_weights: list[float] | None,
    bm25_excluded_role_hints: list[str],
    document_id: str,
    restrict_to_document: bool,
    score_threshold: float | None,
) -> Any:
    """동일 설정으로 Qdrant retriever를 재생성하기 위한 로컬 factory."""
    return build_qdrant_chunk_retriever(
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        retrieval_mode=retrieval_mode,
        fetch_limit=max(top_k, effective_fetch_k),
        dense_fetch_k=max(top_k, dense_fetch_k),
        bm25_fetch_k=max(top_k, bm25_fetch_k),
        dense_vector_name=STAGE4_DENSE_VECTOR_NAME,
        bm25_vector_name=STAGE4_BM25_VECTOR_NAME,
        bm25_options={
            "tokenizer": STAGE4_BM25_TOKENIZER,
            "language": STAGE4_BM25_LANGUAGE,
            "ascii_folding": STAGE4_BM25_ASCII_FOLDING,
        },
        rrf_weights=hybrid_rrf_weights,
        bm25_excluded_role_hints=bm25_excluded_role_hints,
        document_id=document_id,
        restrict_to_document=restrict_to_document,
        score_threshold=score_threshold,
    )


def run_stage4_retrieval(
    inputs: Stage4Input | None = None,
    *,
    embedding_client: OpenAIEmbeddingClient | None = None,
    qdrant_client: QdrantRestClient | None = None,
    persist_manifest: bool = False,
) -> Stage4Output:
    """query retrieval 후 답변용 context expansion까지 포함해 stage4 출력을 만든다."""
    resolved_inputs = dict(inputs or {})

    chunks_json_path = Path(
        resolved_inputs.get("chunks_json_path") or DEFAULT_CHUNKS_JSON_PATH
    ).expanduser().resolve()
    output_dir = (
        Path(resolved_inputs["output_dir"]).expanduser().resolve()
        if resolved_inputs.get("output_dir")
        else (
            chunks_json_path.parent.parent / "stage4"
            if chunks_json_path.parent.name == "stage3"
            else chunks_json_path.parent.resolve()
        )
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
    chunk_lookup = build_chunk_lookup(chunks_document)

    explicit_document_id = resolved_inputs.get("document_id")
    if explicit_document_id:
        document_id = explicit_document_id
    else:
        cleaned_json_path = chunks_document.get("cleaned_json_path")
        document_id = derive_document_id_from_artifact_path(
            cleaned_json_path or chunks_json_path
        )
    collection_name = (
        resolved_inputs.get("collection_name")
        or STAGE4_QDRANT_COLLECTION_NAME
    )
    retrieval_mode = str(
        resolved_inputs.get("retrieval_mode") or STAGE4_RETRIEVAL_MODE
    ).strip().lower() or "hybrid"
    query = str(resolved_inputs.get("query") or "").strip()
    top_k = int(resolved_inputs.get("top_k") or STAGE4_TOP_K)
    shared_fetch_k = int(resolved_inputs.get("fetch_k") or STAGE4_FETCH_K)
    dense_fetch_k = int(
        resolved_inputs.get("dense_fetch_k")
        or resolved_inputs.get("fetch_k")
        or STAGE4_HYBRID_DENSE_FETCH_K
    )
    bm25_fetch_k = int(
        resolved_inputs.get("bm25_fetch_k")
        or resolved_inputs.get("fetch_k")
        or STAGE4_HYBRID_BM25_FETCH_K
    )
    hybrid_rrf_weights = resolved_inputs.get("hybrid_rrf_weights")
    if hybrid_rrf_weights is None:
        hybrid_rrf_weights = STAGE4_HYBRID_RRF_WEIGHTS
    bm25_excluded_role_hints = list(
        resolved_inputs.get("bm25_excluded_role_hints")
        or STAGE4_BM25_EXCLUDED_ROLE_HINTS
    )
    restrict_to_document = bool(
        resolved_inputs.get("restrict_to_document", STAGE4_RESTRICT_TO_DOCUMENT)
    )
    score_threshold = resolved_inputs.get("score_threshold", STAGE4_SCORE_THRESHOLD)
    enable_score_fallback = bool(
        resolved_inputs.get("enable_score_fallback", STAGE4_ENABLE_SCORE_FALLBACK)
    )
    enable_mmr = bool(resolved_inputs.get("enable_mmr", STAGE4_ENABLE_MMR))
    mmr_lambda_mult = float(
        resolved_inputs.get("mmr_lambda_mult", STAGE4_MMR_LAMBDA_MULT)
    )
    parent_expand_mode = str(
        resolved_inputs.get("parent_expand_mode") or STAGE4_PARENT_EXPAND_MODE
    ).strip().lower() or "child"
    parent_window_size = int(
        resolved_inputs.get("parent_window_size") or STAGE4_PARENT_WINDOW_SIZE
    )
    effective_fetch_k = max(top_k, shared_fetch_k, dense_fetch_k, bm25_fetch_k)
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
        "retrieval_mode": retrieval_mode,
        "top_k": top_k,
        "fetch_k": effective_fetch_k,
        "dense_fetch_k": max(top_k, dense_fetch_k),
        "bm25_fetch_k": max(top_k, bm25_fetch_k),
        "hybrid_rrf_weights": hybrid_rrf_weights,
        "bm25_excluded_role_hints": bm25_excluded_role_hints,
        "score_threshold_requested": (
            float(score_threshold)
            if score_threshold is not None
            else None
        ),
        "score_threshold_applied": (
            float(score_threshold)
            if score_threshold is not None
            else None
        ),
        "score_fallback_applied": False,
        "mmr_enabled": enable_mmr,
        "mmr_applied": False,
        "mmr_lambda_mult": mmr_lambda_mult,
        "parent_expand_mode": parent_expand_mode,
        "parent_window_size": parent_window_size,
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

    created_qdrant_client = False
    if qdrant_client is None:
        qdrant_client = QdrantRestClient(
            base_url=STAGE4_QDRANT_URL,
            api_key=STAGE4_QDRANT_API_KEY,
            timeout=STAGE4_QDRANT_TIMEOUT,
        )
        created_qdrant_client = True

    try:
        retriever = _build_retriever(
            embedding_client=embedding_client,
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            retrieval_mode=retrieval_mode,
            top_k=top_k,
            effective_fetch_k=effective_fetch_k,
            dense_fetch_k=dense_fetch_k,
            bm25_fetch_k=bm25_fetch_k,
            hybrid_rrf_weights=hybrid_rrf_weights,
            bm25_excluded_role_hints=bm25_excluded_role_hints,
            document_id=document_id,
            restrict_to_document=restrict_to_document,
            score_threshold=float(score_threshold) if score_threshold is not None else None,
        )
        retrieved_documents = list(retriever.invoke(query))
        threshold_fallback_applied = False
        effective_score_threshold = (
            float(score_threshold) if score_threshold is not None else None
        )

        if (
            effective_score_threshold is not None
            and enable_score_fallback
            and len(retrieved_documents) < top_k
        ):
            fallback_retriever = _build_retriever(
                embedding_client=embedding_client,
                qdrant_client=qdrant_client,
                collection_name=collection_name,
                retrieval_mode=retrieval_mode,
                top_k=top_k,
                effective_fetch_k=effective_fetch_k,
                dense_fetch_k=dense_fetch_k,
                bm25_fetch_k=bm25_fetch_k,
                hybrid_rrf_weights=hybrid_rrf_weights,
                bm25_excluded_role_hints=bm25_excluded_role_hints,
                document_id=document_id,
                restrict_to_document=restrict_to_document,
                score_threshold=None,
            )
            retrieved_documents = list(fallback_retriever.invoke(query))
            threshold_fallback_applied = True
            effective_score_threshold = None

        document_pipeline = (
            RunnableLambda(
                lambda documents: {
                    "documents": list(documents),
                    "mmr_applied": False,
                }
            )
            | build_mmr_reranker(
                query=query,
                embedding_client=embedding_client,
                top_k=top_k,
                enabled=enable_mmr,
                lambda_mult=mmr_lambda_mult,
            )
            | build_context_expander(
                chunk_lookup=chunk_lookup,
                parent_lookup=parent_lookup,
                expand_mode=parent_expand_mode,
                window_size=parent_window_size,
            )
            | _build_retrieval_normalizer(
                top_k=top_k,
                document_id=document_id,
                parent_lookup=parent_lookup,
            )
        )
        retrieval_result = document_pipeline.invoke(retrieved_documents)
    finally:
        if created_qdrant_client:
            qdrant_client.close()

    output = {
        **base_output,
        "score_threshold_applied": effective_score_threshold,
        "score_fallback_applied": threshold_fallback_applied,
        "mmr_applied": bool(retrieval_result["mmr_applied"]),
        "fetched_count": int(retrieval_result["fetched_count"]),
        "retrieved_count": int(retrieval_result["retrieved_count"]),
        "retrievals": list(retrieval_result["retrievals"]),
        "status": "completed",
        "skip_reason": None,
    }
    if persist_manifest:
        _write_retrieval_manifest(output, output_paths=output_paths)
    return output
def main() -> None:
    """기본 chunks.json 경로를 기준으로 stage4 retrieval 결과를 출력한다."""
    response = run_stage4_retrieval(
        {
            "chunks_json_path": str(DEFAULT_CHUNKS_JSON_PATH),
        },
        persist_manifest=True,
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
