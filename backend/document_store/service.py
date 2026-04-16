"""문서 업로드/산출물 경로와 메타데이터를 관리한다."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .schemas import DocumentRecord, DocumentStageRecord


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
DEFAULT_DOCUMENTS_ROOT = BACKEND_DIR / "outputs"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sanitize_document_id(value: str) -> str:
    """파일명이나 사용자 입력을 문서 폴더 이름으로 안전하게 정규화한다."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "document"


def _default_stage_map() -> dict[str, DocumentStageRecord]:
    return {
        "upload": {"status": "uploaded", "updated_at": _now_iso(), "error": None},
        "stage1": {"status": "not_started", "updated_at": _now_iso(), "error": None},
        "stage2": {"status": "not_started", "updated_at": _now_iso(), "error": None},
        "review": {"status": "not_started", "updated_at": _now_iso(), "error": None},
        "stage3": {"status": "not_started", "updated_at": _now_iso(), "error": None},
        "stage4": {"status": "not_started", "updated_at": _now_iso(), "error": None},
    }


@dataclass(frozen=True)
class DocumentPaths:
    """문서 단위 표준 디렉터리/파일 경로 묶음."""

    root: Path
    source_dir: Path
    source_pdf: Path
    metadata_json: Path
    stage1_dir: Path
    stage1_raw_json: Path
    stage2_dir: Path
    stage2_cleaned_json: Path
    stage2_cleaned_md: Path
    stage2_preview_html: Path
    review_dir: Path
    review_decisions_json: Path
    reviewed_cleaned_json: Path
    reviewed_cleaned_md: Path
    reviewed_preview_html: Path
    stage3_dir: Path
    stage3_chunks_json: Path
    stage3_chunks_jsonl: Path
    stage3_chunks_md: Path
    stage3_parents_json: Path
    stage3_indexing_json: Path
    stage4_dir: Path
    stage4_retrieval_json: Path


def generate_document_id() -> str:
    """서비스에서 사용할 문서 ID를 만든다."""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"doc_{timestamp}_{uuid4().hex[:8]}"


def build_document_paths(
    document_id: str,
    *,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> DocumentPaths:
    """문서 ID를 기준으로 표준 저장 경로를 계산한다."""
    documents_root = Path(root).expanduser().resolve()
    document_root = documents_root / document_id
    source_dir = document_root / "source"
    stage1_dir = document_root / "stage1"
    stage2_dir = document_root / "stage2"
    review_dir = document_root / "review"
    stage3_dir = document_root / "stage3"
    stage4_dir = document_root / "stage4"

    return DocumentPaths(
        root=document_root,
        source_dir=source_dir,
        source_pdf=source_dir / "original.pdf",
        metadata_json=source_dir / "document.json",
        stage1_dir=stage1_dir,
        stage1_raw_json=stage1_dir / "raw.json",
        stage2_dir=stage2_dir,
        stage2_cleaned_json=stage2_dir / "cleaned.json",
        stage2_cleaned_md=stage2_dir / "cleaned.md",
        stage2_preview_html=stage2_dir / "preview.html",
        review_dir=review_dir,
        review_decisions_json=review_dir / "review_decisions.json",
        reviewed_cleaned_json=review_dir / "reviewed_cleaned.json",
        reviewed_cleaned_md=review_dir / "reviewed_cleaned.md",
        reviewed_preview_html=review_dir / "reviewed_preview.html",
        stage3_dir=stage3_dir,
        stage3_chunks_json=stage3_dir / "chunks.json",
        stage3_chunks_jsonl=stage3_dir / "chunks.jsonl",
        stage3_chunks_md=stage3_dir / "chunks.md",
        stage3_parents_json=stage3_dir / "parents.json",
        stage3_indexing_json=stage3_dir / "indexing.json",
        stage4_dir=stage4_dir,
        stage4_retrieval_json=stage4_dir / "retrieval.json",
    )


def _ensure_document_dirs(paths: DocumentPaths) -> None:
    for directory in (
        paths.root,
        paths.source_dir,
        paths.stage1_dir,
        paths.stage2_dir,
        paths.review_dir,
        paths.stage3_dir,
        paths.stage4_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def _write_metadata(path: Path, payload: DocumentRecord) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _infer_stage_outputs(paths: DocumentPaths) -> dict[str, dict[str, str]]:
    outputs: dict[str, dict[str, str]] = {}

    if paths.source_pdf.exists():
        outputs["upload"] = {"source_pdf_path": str(paths.source_pdf)}
    if paths.stage1_raw_json.exists():
        outputs["stage1"] = {"raw_json_path": str(paths.stage1_raw_json)}

    stage2_outputs: dict[str, str] = {}
    if paths.stage2_cleaned_json.exists():
        stage2_outputs["cleaned_json"] = str(paths.stage2_cleaned_json)
    if paths.stage2_cleaned_md.exists():
        stage2_outputs["cleaned_md"] = str(paths.stage2_cleaned_md)
    if paths.stage2_preview_html.exists():
        stage2_outputs["preview_html"] = str(paths.stage2_preview_html)
    if stage2_outputs:
        outputs["stage2"] = stage2_outputs

    review_outputs: dict[str, str] = {}
    if paths.review_decisions_json.exists():
        review_outputs["review_decisions_path"] = str(paths.review_decisions_json)
    if paths.reviewed_cleaned_json.exists():
        review_outputs["reviewed_cleaned_json"] = str(paths.reviewed_cleaned_json)
    if paths.reviewed_cleaned_md.exists():
        review_outputs["reviewed_cleaned_md"] = str(paths.reviewed_cleaned_md)
    if paths.reviewed_preview_html.exists():
        review_outputs["reviewed_preview_html"] = str(paths.reviewed_preview_html)
    if review_outputs:
        outputs["review"] = review_outputs

    stage3_outputs: dict[str, str] = {}
    if paths.stage3_chunks_json.exists():
        stage3_outputs["chunks_json"] = str(paths.stage3_chunks_json)
    if paths.stage3_chunks_jsonl.exists():
        stage3_outputs["chunks_jsonl"] = str(paths.stage3_chunks_jsonl)
    if paths.stage3_chunks_md.exists():
        stage3_outputs["chunks_md"] = str(paths.stage3_chunks_md)
    if paths.stage3_parents_json.exists():
        stage3_outputs["parents_json"] = str(paths.stage3_parents_json)
    if paths.stage3_indexing_json.exists():
        stage3_outputs["indexing_manifest"] = str(paths.stage3_indexing_json)
    if stage3_outputs:
        outputs["stage3"] = stage3_outputs

    if paths.stage4_retrieval_json.exists():
        outputs["stage4"] = {"retrieval_manifest": str(paths.stage4_retrieval_json)}

    return outputs


def sync_document_record(
    *,
    document_id: str,
    original_filename: str | None = None,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> DocumentRecord:
    """현재 디스크 산출물을 기준으로 document.json을 다시 맞춘다."""
    paths = build_document_paths(document_id, root=root)
    _ensure_document_dirs(paths)

    existing: DocumentRecord = {}
    if paths.metadata_json.exists():
        payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            existing = payload

    inferred_outputs = _infer_stage_outputs(paths)
    resolved_original_filename = (
        original_filename
        or str(existing.get("original_filename") or "").strip()
        or f"{document_id}.pdf"
    )
    record: DocumentRecord = {
        "document_id": document_id,
        "original_filename": resolved_original_filename,
        "uploaded_at": str(existing.get("uploaded_at") or _now_iso()),
        "stages": _default_stage_map(),
    }

    stages = dict(record["stages"])
    if paths.source_pdf.exists():
        stages["upload"] = {
            "status": "uploaded",
            "updated_at": _now_iso(),
            "error": None,
            "outputs": inferred_outputs.get("upload", {}),
        }
    else:
        stages["upload"] = {
            "status": "not_started",
            "updated_at": _now_iso(),
            "error": None,
        }

    for stage_name in ("stage1", "stage2", "stage3", "stage4"):
        outputs = inferred_outputs.get(stage_name, {})
        stages[stage_name] = {
            "status": "completed" if outputs else "not_started",
            "updated_at": _now_iso(),
            "error": None,
        }
        if outputs:
            stages[stage_name]["outputs"] = outputs

    review_outputs = inferred_outputs.get("review", {})
    stages["review"] = {
        "status": "completed" if paths.reviewed_cleaned_json.exists() else "running" if review_outputs else "not_started",
        "updated_at": _now_iso(),
        "error": None,
    }
    if review_outputs:
        stages["review"]["outputs"] = review_outputs

    record["stages"] = stages
    _write_metadata(paths.metadata_json, record)
    return record


def load_document_record(
    document_id: str,
    *,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> DocumentRecord:
    """문서 메타데이터를 읽는다."""
    paths = build_document_paths(document_id, root=root)
    if not paths.metadata_json.exists():
        raise FileNotFoundError(f"document metadata not found: {document_id}")
    payload = json.loads(paths.metadata_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid document metadata: {paths.metadata_json}")
    return payload


def create_document_record(
    *,
    original_filename: str,
    document_id: str | None = None,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> DocumentRecord:
    """새 업로드 문서의 메타데이터와 디렉터리를 만든다."""
    resolved_document_id = document_id or generate_document_id()
    if document_id is None:
        original_path = Path(original_filename)
        resolved_document_id = sanitize_document_id(original_path.stem)
    paths = build_document_paths(resolved_document_id, root=root)
    _ensure_document_dirs(paths)

    record: DocumentRecord = {
        "document_id": resolved_document_id,
        "original_filename": original_filename,
        "uploaded_at": _now_iso(),
        "stages": _default_stage_map(),
    }
    _write_metadata(paths.metadata_json, record)
    return record


def save_uploaded_pdf(
    *,
    document_id: str,
    content: bytes,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> Path:
    """업로드된 원본 PDF를 표준 위치에 저장한다."""
    paths = build_document_paths(document_id, root=root)
    _ensure_document_dirs(paths)
    paths.source_pdf.write_bytes(content)
    return paths.source_pdf


def update_document_stage_record(
    *,
    document_id: str,
    stage: str,
    status: str,
    outputs: dict[str, str] | None = None,
    error: str | None = None,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> DocumentRecord:
    """특정 stage의 상태와 산출물 메타를 갱신한다."""
    record = load_document_record(document_id, root=root)
    stages = dict(record.get("stages") or {})
    stage_record: DocumentStageRecord = {
        **(stages.get(stage) or {}),
        "status": status,
        "updated_at": _now_iso(),
        "error": error,
    }
    if outputs:
        stage_record["outputs"] = outputs
    stages[stage] = stage_record
    record["stages"] = stages

    paths = build_document_paths(document_id, root=root)
    _write_metadata(paths.metadata_json, record)
    return record


def list_document_records(
    *,
    root: str | Path = DEFAULT_DOCUMENTS_ROOT,
) -> list[DocumentRecord]:
    """문서 목록을 최신 업로드 순으로 반환한다."""
    documents_root = Path(root).expanduser().resolve()
    if not documents_root.exists():
        return []

    records: list[DocumentRecord] = []
    for metadata_path in documents_root.glob("*/source/document.json"):
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(payload, dict):
            records.append(payload)

    return sorted(
        records,
        key=lambda item: (
            str(item.get("uploaded_at") or ""),
            str(item.get("document_id") or ""),
        ),
        reverse=True,
    )


def get_effective_cleaned_json_path(paths: DocumentPaths) -> Path:
    """review overlay가 있으면 reviewed 결과를, 없으면 stage2 결과를 사용한다."""
    if paths.reviewed_cleaned_json.exists():
        return paths.reviewed_cleaned_json
    return paths.stage2_cleaned_json
