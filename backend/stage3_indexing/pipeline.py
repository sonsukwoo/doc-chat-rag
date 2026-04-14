"""Stage-3 chunk indexing pipeline."""

from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any, Iterable

from backend.stage3_chunking.embeddings import OpenAIEmbeddingClient

from .config import (
    DEFAULT_CHUNKS_JSON_PATH,
    DEFAULT_INDEXING_MANIFEST_NAME,
    STAGE3_ENABLE_INDEXING,
    STAGE3_QDRANT_API_KEY,
    STAGE3_QDRANT_COLLECTION_NAME,
    STAGE3_QDRANT_TIMEOUT,
    STAGE3_QDRANT_UPSERT_BATCH_SIZE,
    STAGE3_QDRANT_URL,
)
from .qdrant import QdrantRestClient
from .schemas import (
    Stage3IndexInput,
    Stage3IndexOutput,
    Stage3IndexOutputPaths,
)


def build_stage3_index_output_paths(
    *,
    chunks_json_path: str | Path,
    output_dir: str | Path | None = None,
) -> Stage3IndexOutputPaths:
    """stage3 indexing이 기록할 산출물 경로를 계산한다."""
    chunks_path = Path(chunks_json_path).expanduser().resolve()
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else chunks_path.parent.resolve()
    )
    return {
        "indexing_manifest": str(
            (resolved_output_dir / DEFAULT_INDEXING_MANIFEST_NAME).resolve()
        )
    }


def _load_chunks_document(chunks_json_path: Path) -> dict[str, Any]:
    payload = json.loads(chunks_json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("chunks.json은 dict 형태여야 합니다.")
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


def _iter_batches(items: list[Any], batch_size: int) -> Iterable[list[Any]]:
    for start_index in range(0, len(items), batch_size):
        yield items[start_index : start_index + batch_size]


def _build_section_title(chunk: dict[str, Any]) -> str | None:
    """heading path를 사람이 보기 쉬운 섹션 문자열로 평탄화한다."""
    heading_path = [str(item).strip() for item in chunk.get("heading_path") or [] if str(item).strip()]
    if not heading_path:
        return None
    return " > ".join(heading_path)


def _build_page_fields(chunk: dict[str, Any]) -> dict[str, int | None]:
    """pages 배열을 citation과 UI에 바로 쓰기 쉬운 필드로 변환한다."""
    pages = []
    for value in chunk.get("pages") or []:
        try:
            pages.append(int(value))
        except (TypeError, ValueError):
            continue

    if not pages:
        return {
            "primary_page": None,
            "page_start": None,
            "page_end": None,
        }

    ordered_pages = sorted(set(pages))
    return {
        "primary_page": ordered_pages[0],
        "page_start": ordered_pages[0],
        "page_end": ordered_pages[-1],
    }


def _build_qdrant_payload(
    *,
    document_id: str,
    chunk: dict[str, Any],
) -> dict[str, Any]:
    """검색 결과와 citation/UI에 필요한 최소 payload만 구성한다."""
    metadata = chunk.get("metadata") or {}
    asset_relative_path = str(metadata.get("image_path") or "").strip() or None
    caption = str(metadata.get("caption") or "").strip() or None
    chunk_type = str(chunk.get("chunk_type") or "")
    has_asset = bool(asset_relative_path and chunk_type in {"table", "figure"})
    payload: dict[str, Any] = {
        "document_id": document_id,
        "chunk_id": str(chunk.get("chunk_id") or ""),
        "chunk_type": chunk_type,
        "text": str(chunk.get("text") or ""),
        "section_title": _build_section_title(chunk),
        **_build_page_fields(chunk),
        "has_asset": has_asset,
    }

    if has_asset:
        payload["asset_kind"] = chunk_type
        payload["asset_relative_path"] = asset_relative_path
    if caption:
        payload["caption"] = caption

    return payload


def _build_qdrant_points(
    *,
    document_id: str,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, embeddings):
        chunk_id = str(chunk.get("chunk_id") or "")
        point_id = str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"rag-chat:{document_id}:{chunk_id}",
            )
        )
        points.append(
            {
                "id": point_id,
                "vector": vector,
                "payload": _build_qdrant_payload(
                    document_id=document_id,
                    chunk=chunk,
                ),
            }
        )
    return points


def _write_index_manifest(
    output: Stage3IndexOutput,
    *,
    output_paths: Stage3IndexOutputPaths,
) -> None:
    manifest_path = Path(output_paths["indexing_manifest"])
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2)
    )


def run_stage3_indexing(
    inputs: Stage3IndexInput,
    *,
    embedding_client: OpenAIEmbeddingClient | None = None,
    qdrant_client: QdrantRestClient | None = None,
) -> Stage3IndexOutput:
    """chunks.json을 읽어 dense embedding 생성 후 Qdrant에 upsert 한다."""
    chunks_json_path = Path(
        inputs.get("chunks_json_path") or DEFAULT_CHUNKS_JSON_PATH
    ).expanduser().resolve()
    output_dir = (
        Path(inputs["output_dir"]).expanduser().resolve()
        if inputs.get("output_dir")
        else chunks_json_path.parent.resolve()
    )
    output_paths = build_stage3_index_output_paths(
        chunks_json_path=chunks_json_path,
        output_dir=output_dir,
    )

    collection_name = (
        inputs.get("collection_name")
        or STAGE3_QDRANT_COLLECTION_NAME
    )
    has_qdrant_target = bool(qdrant_client is not None or STAGE3_QDRANT_URL)
    indexing_enabled = bool(
        STAGE3_ENABLE_INDEXING and has_qdrant_target and collection_name
    )

    chunks_document = _load_chunks_document(chunks_json_path)
    chunks = list(chunks_document.get("chunks") or [])
    document_id = _derive_document_id(
        chunks_json_path=chunks_json_path,
        chunks_document=chunks_document,
        explicit_document_id=inputs.get("document_id"),
    )

    if not indexing_enabled:
        output: Stage3IndexOutput = {
            "chunks_json_path": str(chunks_json_path),
            "output_dir": str(output_dir),
            "document_id": document_id,
            "collection_name": collection_name,
            "output_paths": output_paths,
            "planned_outputs": output_paths,
            "point_count": 0,
            "vector_size": 0,
            "indexing_enabled": False,
            "status": "skipped",
            "skip_reason": "indexing_disabled_or_missing_qdrant_config",
        }
        _write_index_manifest(output, output_paths=output_paths)
        return output

    texts = [str(chunk.get("text") or "").strip() for chunk in chunks]
    if not texts:
        output = {
            "chunks_json_path": str(chunks_json_path),
            "output_dir": str(output_dir),
            "document_id": document_id,
            "collection_name": collection_name,
            "output_paths": output_paths,
            "planned_outputs": output_paths,
            "point_count": 0,
            "vector_size": 0,
            "indexing_enabled": True,
            "status": "skipped",
            "skip_reason": "no_chunks_to_index",
        }
        _write_index_manifest(output, output_paths=output_paths)
        return output

    embedding_client = embedding_client or OpenAIEmbeddingClient(enabled=True)
    embeddings = embedding_client.embed_texts(texts)
    if embeddings is None or not embeddings or not embeddings[0]:
        raise RuntimeError(
            f"chunk embedding 생성에 실패했습니다: {embedding_client.last_error or 'unknown_error'}"
        )

    vector_size = len(embeddings[0])
    points = _build_qdrant_points(
        document_id=document_id,
        chunks=chunks,
        embeddings=embeddings,
    )

    owns_client = qdrant_client is None
    qdrant_client = qdrant_client or QdrantRestClient(
        base_url=STAGE3_QDRANT_URL,
        api_key=STAGE3_QDRANT_API_KEY,
        timeout=STAGE3_QDRANT_TIMEOUT,
    )
    try:
        qdrant_client.ensure_dense_collection(
            collection_name=collection_name,
            vector_size=vector_size,
            distance="Cosine",
        )
        for batch in _iter_batches(points, STAGE3_QDRANT_UPSERT_BATCH_SIZE):
            qdrant_client.upsert_points(
                collection_name=collection_name,
                points=batch,
                wait=True,
            )
    finally:
        if owns_client:
            qdrant_client.close()

    output = {
        "chunks_json_path": str(chunks_json_path),
        "output_dir": str(output_dir),
        "document_id": document_id,
        "collection_name": collection_name,
        "output_paths": output_paths,
        "planned_outputs": output_paths,
        "point_count": len(points),
        "vector_size": vector_size,
        "indexing_enabled": True,
        "status": "completed",
        "skip_reason": None,
    }
    _write_index_manifest(output, output_paths=output_paths)
    return output


def prepare_stage3_indexing(
    inputs: Stage3IndexInput,
    *,
    embedding_client: OpenAIEmbeddingClient | None = None,
    qdrant_client: QdrantRestClient | None = None,
) -> Stage3IndexOutput:
    """기존 함수명 패턴을 맞추기 위한 indexing 래퍼다."""
    return run_stage3_indexing(
        inputs,
        embedding_client=embedding_client,
        qdrant_client=qdrant_client,
    )
