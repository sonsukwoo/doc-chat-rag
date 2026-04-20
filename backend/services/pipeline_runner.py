"""문서 저장소 기준으로 stage 파이프라인을 실행한다."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.app_db import (
    sync_document_profile_snapshot,
    sync_document_runtime_metadata,
)
from backend.document_store import (
    build_document_paths,
    get_effective_cleaned_json_path,
    load_document_record,
    update_document_stage_record,
)
from backend.stage1_parse.pipeline import run_stage1_parse
from backend.stage2_preprocess.graph import get_agent
from backend.stage3_indexing.config import (
    STAGE3_QDRANT_API_KEY,
    STAGE3_QDRANT_TIMEOUT,
    STAGE3_QDRANT_URL,
)
from backend.stage3_indexing.qdrant import QdrantRestClient
from backend.stage3 import run_stage3


def _build_stage3_document_filter(
    *,
    thread_id: str | None,
    document_id: str,
) -> dict[str, Any]:
    must: list[dict[str, Any]] = [
        {
            "key": "document_id",
            "match": {"value": document_id},
        }
    ]
    if thread_id:
        must.insert(
            0,
            {
                "key": "thread_id",
                "match": {"value": thread_id},
            },
        )
    return {"must": must}


def _mark_stage3_manifest_failed(
    manifest_path: str | Path | None,
    *,
    error: str,
) -> None:
    raw_manifest_path = str(manifest_path or "").strip()
    if not raw_manifest_path:
        return
    resolved_manifest_path = Path(raw_manifest_path).expanduser()
    if not resolved_manifest_path.exists():
        return

    try:
        payload = json.loads(resolved_manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return
    if not isinstance(payload, dict):
        return

    payload["status"] = "failed"
    payload["sync_status"] = "failed"
    payload["sync_error"] = error
    resolved_manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _cleanup_indexed_document_points(
    *,
    thread_id: str | None,
    collection_name: str | None,
    document_id: str,
) -> str | None:
    resolved_collection_name = str(collection_name or "").strip()
    if not resolved_collection_name or not STAGE3_QDRANT_URL:
        return None

    client = QdrantRestClient(
        base_url=STAGE3_QDRANT_URL,
        api_key=STAGE3_QDRANT_API_KEY,
        timeout=STAGE3_QDRANT_TIMEOUT,
    )
    try:
        client.delete_points_by_filter(
            collection_name=resolved_collection_name,
            query_filter=_build_stage3_document_filter(
                thread_id=thread_id,
                document_id=document_id,
            ),
            wait=True,
        )
        return None
    except Exception as exc:
        return str(exc)
    finally:
        client.close()


def run_stage1_for_document(document_id: str) -> dict[str, Any]:
    """업로드된 원본 PDF를 stage1 raw.json으로 변환한다."""
    paths = build_document_paths(document_id)
    update_document_stage_record(
        document_id=document_id,
        stage="stage1",
        status="running",
    )
    try:
        result = run_stage1_parse(
            pdf_path=paths.source_pdf,
            output_dir=paths.stage1_dir,
            json_name=paths.stage1_raw_json.name,
            copy_source_pdf=False,
        )
        update_document_stage_record(
            document_id=document_id,
            stage="stage1",
            status="completed",
            outputs={
                "raw_json_path": result["json_path"],
            },
        )
        return result
    except Exception as exc:
        update_document_stage_record(
            document_id=document_id,
            stage="stage1",
            status="failed",
            error=str(exc),
        )
        raise


def run_stage2_for_document(document_id: str) -> dict[str, Any]:
    """stage1 raw.json을 기준으로 cleaned 산출물을 만든다."""
    paths = build_document_paths(document_id)
    update_document_stage_record(
        document_id=document_id,
        stage="stage2",
        status="running",
    )
    try:
        result = get_agent().invoke(
            {
                "raw_json_path": str(paths.stage1_raw_json),
                "source_pdf_path": str(paths.source_pdf),
                "output_dir": str(paths.stage2_dir),
            }
        )
        output_paths = result.get("output_paths") or {}
        cleaned_json_path = Path(
            str(output_paths.get("cleaned_json") or paths.stage2_cleaned_json)
        ).expanduser()
        cleaned_payload = json.loads(cleaned_json_path.read_text(encoding="utf-8"))
        document_record = load_document_record(document_id)
        sync_document_profile_snapshot(
            document_id=document_id,
            original_filename=str(
                document_record.get("original_filename") or f"{document_id}.pdf"
            ).strip()
            or f"{document_id}.pdf",
            normalized_filename=str(
                document_record.get("normalized_filename")
                or document_record.get("original_filename")
                or f"{document_id}.pdf"
            ).strip()
            or f"{document_id}.pdf",
            storage_root=paths.root,
            source_pdf_path=str(paths.source_pdf) if paths.source_pdf.exists() else None,
            raw_profile=dict(cleaned_payload.get("document_profile") or {}),
            elements=list(cleaned_payload.get("elements") or []),
            source_stage="stage2",
        )
        update_document_stage_record(
            document_id=document_id,
            stage="stage2",
            status="completed",
            outputs={
                key: str(value)
                for key, value in output_paths.items()
            },
        )
        return result
    except Exception as exc:
        update_document_stage_record(
            document_id=document_id,
            stage="stage2",
            status="failed",
            error=str(exc),
        )
        raise


def run_stage3_for_document(
    document_id: str,
    *,
    thread_id: str | None = None,
    collection_name: str | None = None,
) -> dict[str, Any]:
    """review overlay가 있으면 이를 우선 사용해 chunking/indexing을 수행한다."""
    paths = build_document_paths(document_id)
    document_record = load_document_record(document_id)
    cleaned_json_path = get_effective_cleaned_json_path(paths)
    outputs: dict[str, str] = {}
    stage3_outputs_ready = False
    update_document_stage_record(
        document_id=document_id,
        stage="stage3",
        status="running",
    )
    try:
        result = run_stage3(
            {
                "cleaned_json_path": str(cleaned_json_path),
                "output_dir": str(paths.stage3_dir),
                "document_id": document_id,
                "thread_id": thread_id,
                "collection_name": collection_name,
            }
        )
        chunking_output = result.get("chunking") or {}
        indexing_output = result.get("indexing") or {}
        outputs = {
            **{
                key: str(value)
                for key, value in (chunking_output.get("output_paths") or {}).items()
            },
            **{
                key: str(value)
                for key, value in (indexing_output.get("output_paths") or {}).items()
            },
        }
        stage3_outputs_ready = True
        if thread_id:
            cleaned_payload = json.loads(cleaned_json_path.read_text())
            parents_payload = json.loads(paths.stage3_parents_json.read_text())
            chunks_payload = json.loads(paths.stage3_chunks_json.read_text())
            document_profile = dict(cleaned_payload.get("document_profile") or {})
            document_profile_elements = list(cleaned_payload.get("elements") or [])
            original_filename = str(
                document_record.get("original_filename") or f"{document_id}.pdf"
            ).strip() or f"{document_id}.pdf"
            normalized_filename = str(
                document_record.get("normalized_filename") or original_filename
            ).strip() or original_filename
            sync_document_runtime_metadata(
                thread_id=thread_id,
                document_id=document_id,
                original_filename=original_filename,
                normalized_filename=normalized_filename,
                storage_root=paths.root,
                source_pdf_path=str(paths.source_pdf) if paths.source_pdf.exists() else None,
                parents=list(parents_payload.get("parents") or []),
                chunks=list(chunks_payload.get("chunks") or []),
                metadata={
                    "thread_id": thread_id,
                },
                document_profile=document_profile,
                document_profile_elements=document_profile_elements,
                document_profile_source_stage=(
                    "review" if cleaned_json_path.parent.name == "review" else "stage3"
                ),
            )
        update_document_stage_record(
            document_id=document_id,
            stage="stage3",
            status="completed",
            outputs=outputs,
        )
        return result
    except Exception as exc:
        cleanup_error: str | None = None
        if stage3_outputs_ready:
            manifest_path = outputs.get("indexing_manifest") or str(
                getattr(paths, "stage3_indexing_json", "") or ""
            )
            _mark_stage3_manifest_failed(manifest_path, error=str(exc))
        if stage3_outputs_ready and thread_id:
            cleanup_error = _cleanup_indexed_document_points(
                thread_id=thread_id,
                collection_name=collection_name,
                document_id=document_id,
            )
        failure_message = str(exc)
        if cleanup_error:
            failure_message = (
                f"{failure_message} | qdrant cleanup failed: {cleanup_error}"
            )
        update_document_stage_record(
            document_id=document_id,
            stage="stage3",
            status="failed",
            error=failure_message,
            outputs=outputs or None,
        )
        raise
