"""문서 산출물 경로에서 일관된 document_id를 추론하는 공통 helper."""

from __future__ import annotations

from pathlib import Path


PIPELINE_OUTPUT_DIR_NAMES = {"stage2", "stage3", "stage4", "review"}


def derive_document_id_from_artifact_path(artifact_path: str | Path) -> str:
    """stage 산출물 파일 경로를 기준으로 문서 폴더 id를 계산한다."""
    resolved_path = Path(artifact_path).expanduser().resolve()
    parent_dir = resolved_path.parent
    if parent_dir.name in PIPELINE_OUTPUT_DIR_NAMES:
        return parent_dir.parent.name
    return parent_dir.name
