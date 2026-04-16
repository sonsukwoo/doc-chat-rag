"""문서 저장소 기준으로 stage 파이프라인을 실행한다."""

from __future__ import annotations

from typing import Any

from backend.document_store import (
    build_document_paths,
    get_effective_cleaned_json_path,
    update_document_stage_record,
)
from backend.stage1_parse.pipeline import run_stage1_parse
from backend.stage2_preprocess.graph import get_agent
from backend.stage3 import run_stage3


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
    collection_name: str | None = None,
) -> dict[str, Any]:
    """review overlay가 있으면 이를 우선 사용해 chunking/indexing을 수행한다."""
    paths = build_document_paths(document_id)
    cleaned_json_path = get_effective_cleaned_json_path(paths)
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
        update_document_stage_record(
            document_id=document_id,
            stage="stage3",
            status="completed",
            outputs=outputs,
        )
        return result
    except Exception as exc:
        update_document_stage_record(
            document_id=document_id,
            stage="stage3",
            status="failed",
            error=str(exc),
        )
        raise
