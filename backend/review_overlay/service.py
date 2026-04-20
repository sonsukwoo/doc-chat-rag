"""Stage2 cleaned 결과에 대한 review overlay를 저장하고 적용한다."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, TypedDict

from backend.app_db import sync_document_profile_snapshot
from backend.document_store import (
    DocumentPaths,
    build_document_paths,
    load_document_record,
    update_document_stage_record,
)
from backend.stage2_preprocess.utils import (
    clean_render_text,
    export_cleaned_elements,
    render_figure_html,
    render_figure_markdown,
    render_table_html,
    render_table_markdown,
    render_text_like_html,
    render_text_like_markdown,
    safe_write_json,
    safe_write_text,
)


ALLOWED_CATEGORY_OVERRIDES = {
    "paragraph",
    "heading",
    "list",
    "caption",
    "code",
}
TEXT_LIKE_CATEGORIES = {"paragraph", "heading", "list", "caption", "code", "footnote"}


class ElementDecision(TypedDict, total=False):
    dropped: bool
    category_override: str | None


class ReviewDecisionDocument(TypedDict, total=False):
    document_id: str
    updated_at: str
    element_decisions: dict[str, ElementDecision]
    exact_text_drop: list[str]


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _normalize_exact_text(text: str | None) -> str:
    normalized = clean_render_text(str(text or "")).lower()
    return normalized.strip()


def _load_cleaned_document(paths: DocumentPaths) -> dict[str, Any]:
    if not paths.stage2_cleaned_json.exists():
        raise FileNotFoundError("stage2 cleaned.json not found")

    payload = json.loads(paths.stage2_cleaned_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid cleaned.json payload")
    return payload


def load_review_decisions(document_id: str) -> ReviewDecisionDocument:
    """문서별 review decision 파일을 읽는다. 없으면 빈 구조를 반환한다."""
    paths = build_document_paths(document_id)
    if not paths.review_decisions_json.exists():
        return {
            "document_id": document_id,
            "updated_at": _now_iso(),
            "element_decisions": {},
            "exact_text_drop": [],
        }

    payload = json.loads(paths.review_decisions_json.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("invalid review_decisions.json payload")

    return {
        "document_id": document_id,
        "updated_at": str(payload.get("updated_at") or _now_iso()),
        "element_decisions": dict(payload.get("element_decisions") or {}),
        "exact_text_drop": list(payload.get("exact_text_drop") or []),
    }


def save_review_decisions(
    document_id: str,
    *,
    element_decisions: dict[str, dict[str, Any]] | None,
    exact_text_drop: list[str] | None,
) -> ReviewDecisionDocument:
    """프론트에서 전달한 review decisions를 정규화해 저장한다."""
    paths = build_document_paths(document_id)
    cleaned_document = _load_cleaned_document(paths)
    valid_element_ids = {
        int(element.get("id"))
        for element in (cleaned_document.get("elements") or [])
        if element.get("id") is not None
    }

    normalized_element_decisions: dict[str, ElementDecision] = {}
    for raw_element_id, raw_decision in (element_decisions or {}).items():
        try:
            element_id = int(raw_element_id)
        except (TypeError, ValueError):
            continue
        if element_id not in valid_element_ids:
            continue
        if not isinstance(raw_decision, dict):
            continue

        normalized: ElementDecision = {}
        if "dropped" in raw_decision and raw_decision["dropped"] is not None:
            normalized["dropped"] = bool(raw_decision["dropped"])

        category_override = raw_decision.get("category_override")
        if category_override in ALLOWED_CATEGORY_OVERRIDES:
            normalized["category_override"] = str(category_override)
        elif category_override in (None, ""):
            normalized["category_override"] = None

        if normalized:
            normalized_element_decisions[str(element_id)] = normalized

    normalized_exact_text_drop = sorted(
        {
            normalized
            for normalized in (
                _normalize_exact_text(item)
                for item in (exact_text_drop or [])
            )
            if normalized
        }
    )

    payload: ReviewDecisionDocument = {
        "document_id": document_id,
        "updated_at": _now_iso(),
        "element_decisions": normalized_element_decisions,
        "exact_text_drop": normalized_exact_text_drop,
    }
    safe_write_json(paths.review_decisions_json, payload)
    update_document_stage_record(
        document_id=document_id,
        stage="review",
        status="running",
        outputs={"review_decisions_path": str(paths.review_decisions_json)},
    )
    return payload


def _build_text_neighbors(elements: list[dict[str, Any]], index: int) -> tuple[str, str]:
    prev_context = ""
    next_context = ""

    for reverse_index in range(index - 1, -1, -1):
        candidate = elements[reverse_index]
        if candidate.get("category") not in TEXT_LIKE_CATEGORIES:
            continue
        text = clean_render_text(candidate.get("text", ""))
        if text:
            prev_context = text
            break

    for forward_index in range(index + 1, len(elements)):
        candidate = elements[forward_index]
        if candidate.get("category") not in TEXT_LIKE_CATEGORIES:
            continue
        text = clean_render_text(candidate.get("text", ""))
        if text:
            next_context = text
            break

    return prev_context, next_context


def build_review_source(document_id: str) -> dict[str, Any]:
    """프론트 review UI가 바로 쓸 수 있는 source payload를 만든다."""
    paths = build_document_paths(document_id)
    cleaned_document = _load_cleaned_document(paths)
    decisions = load_review_decisions(document_id)
    elements = list(cleaned_document.get("elements") or [])

    normalized_counts: dict[str, int] = {}
    for element in elements:
        normalized_text = _normalize_exact_text(element.get("text"))
        if not normalized_text:
            continue
        normalized_counts[normalized_text] = normalized_counts.get(normalized_text, 0) + 1

    exact_text_drop = set(decisions.get("exact_text_drop") or [])
    element_decisions = dict(decisions.get("element_decisions") or {})
    review_elements: list[dict[str, Any]] = []

    for index, element in enumerate(elements):
        element_id = int(element.get("id"))
        normalized_text = _normalize_exact_text(element.get("text"))
        same_text_count = normalized_counts.get(normalized_text, 0) if normalized_text else 0
        explicit_decision = element_decisions.get(str(element_id)) or {}
        explicit_dropped = explicit_decision.get("dropped")
        category_override = explicit_decision.get("category_override")
        dropped_by_exact = bool(normalized_text and normalized_text in exact_text_drop)
        effective_dropped = (
            bool(explicit_dropped)
            if explicit_dropped is not None
            else dropped_by_exact
        )
        effective_category = category_override or element.get("category")
        prev_context, next_context = _build_text_neighbors(elements, index)

        review_elements.append(
            {
                **element,
                "normalized_text": normalized_text,
                "same_text_count": same_text_count,
                "effective_category": effective_category,
                "dropped": effective_dropped,
                "drop_source": (
                    "element"
                    if explicit_dropped is not None
                    else "exact_text"
                    if dropped_by_exact
                    else "none"
                ),
                "review": {
                    "dropped": explicit_dropped,
                    "category_override": category_override,
                },
                "prev_context": prev_context,
                "next_context": next_context,
            }
        )

    dropped_count = sum(1 for element in review_elements if element["dropped"])
    return {
        "document_id": document_id,
        "source_pdf": cleaned_document.get("source_pdf"),
        "total_pages": cleaned_document.get("total_pages"),
        "document_profile": cleaned_document.get("document_profile"),
        "ordering_resolution": cleaned_document.get("ordering_resolution"),
        "allowed_category_overrides": sorted(ALLOWED_CATEGORY_OVERRIDES),
        "review_decisions": decisions,
        "counts": {
            "total_elements": len(review_elements),
            "dropped_elements": dropped_count,
        },
        "elements": review_elements,
    }


def _apply_decisions_to_elements(
    elements: list[dict[str, Any]],
    decisions: ReviewDecisionDocument,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    exact_text_drop = set(decisions.get("exact_text_drop") or [])
    element_decisions = dict(decisions.get("element_decisions") or {})
    reviewed_elements: list[dict[str, Any]] = []
    dropped_count = 0

    for element in elements:
        item = dict(element)
        element_id = int(item.get("id"))
        explicit_decision = element_decisions.get(str(element_id)) or {}
        normalized_text = _normalize_exact_text(item.get("text"))
        dropped_by_exact = bool(normalized_text and normalized_text in exact_text_drop)
        explicit_dropped = explicit_decision.get("dropped")
        effective_dropped = (
            bool(explicit_dropped)
            if explicit_dropped is not None
            else dropped_by_exact
        )

        if effective_dropped:
            dropped_count += 1
            continue

        category_override = explicit_decision.get("category_override")
        if category_override in ALLOWED_CATEGORY_OVERRIDES:
            item["category"] = category_override
            item["html"] = (
                render_text_like_html(item)
                if category_override in TEXT_LIKE_CATEGORIES
                else item.get("html", "")
            )

        reviewed_elements.append(item)

    stats = {
        "kept_elements": len(reviewed_elements),
        "dropped_elements": dropped_count,
    }
    return reviewed_elements, stats


def _render_reviewed_markdown(elements: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for element in elements:
        category = element.get("category")
        if category == "figure":
            block = render_figure_markdown(element)
        elif category == "table":
            block = render_table_markdown(element)
        else:
            block = render_text_like_markdown(element)
        block = block.strip()
        if block:
            blocks.append(block)
    return "\n\n".join(blocks).strip() + "\n"


def _render_reviewed_preview_html(elements: list[dict[str, Any]]) -> str:
    blocks: list[str] = []
    for element in elements:
        category = element.get("category")
        if category == "figure":
            block = render_figure_html(element)
        elif category == "table":
            block = render_table_html(element)
        else:
            block = render_text_like_html(element)
        block = block.strip()
        if block:
            blocks.append(block)

    return "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"ko\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "<title>Reviewed Document Preview</title>",
            "<style>",
            "body { max-width: 960px; margin: 40px auto; padding: 0 20px; font-family: -apple-system, BlinkMacSystemFont, sans-serif; line-height: 1.65; }",
            "img { max-width: 100%; height: auto; display: block; margin: 12px 0; border: 1px solid #ddd; }",
            "figure, .table-block { margin: 24px 0; }",
            "table { border-collapse: collapse; width: 100%; }",
            "th, td { border: 1px solid #ccc; padding: 6px 8px; }",
            ".figure-summary, .table-summary { color: #444; }",
            "</style>",
            "</head>",
            "<body>",
            *blocks,
            "</body>",
            "</html>",
        ]
    )


def apply_review_overlay(document_id: str) -> dict[str, Any]:
    """saved review decisions를 반영해 reviewed 산출물을 생성한다."""
    paths = build_document_paths(document_id)
    document_record = load_document_record(document_id)
    cleaned_document = _load_cleaned_document(paths)
    decisions = load_review_decisions(document_id)
    elements = list(cleaned_document.get("elements") or [])
    reviewed_elements, stats = _apply_decisions_to_elements(elements, decisions)

    reviewed_cleaned_payload = {
        "source_pdf": cleaned_document.get("source_pdf"),
        "total_pages": cleaned_document.get("total_pages"),
        "document_profile": cleaned_document.get("document_profile"),
        "ordering_resolution": cleaned_document.get("ordering_resolution"),
        "review_metadata": {
            "decision_path": str(paths.review_decisions_json),
            "updated_at": _now_iso(),
            **stats,
        },
        "elements": export_cleaned_elements(reviewed_elements),
    }
    reviewed_markdown = _render_reviewed_markdown(reviewed_elements)
    reviewed_preview_html = _render_reviewed_preview_html(reviewed_elements)

    safe_write_json(paths.reviewed_cleaned_json, reviewed_cleaned_payload)
    safe_write_text(paths.reviewed_cleaned_md, reviewed_markdown)
    safe_write_text(paths.reviewed_preview_html, reviewed_preview_html)
    output_paths = {
        "review_decisions_path": str(paths.review_decisions_json),
        "reviewed_cleaned_json": str(paths.reviewed_cleaned_json),
        "reviewed_cleaned_md": str(paths.reviewed_cleaned_md),
        "reviewed_preview_html": str(paths.reviewed_preview_html),
    }
    try:
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
            raw_profile=dict(reviewed_cleaned_payload.get("document_profile") or {}),
            elements=list(reviewed_cleaned_payload.get("elements") or []),
            source_stage="review",
        )
        update_document_stage_record(
            document_id=document_id,
            stage="review",
            status="completed",
            outputs=output_paths,
        )
    except Exception as exc:
        update_document_stage_record(
            document_id=document_id,
            stage="review",
            status="failed",
            error=str(exc),
            outputs=output_paths,
        )
        raise

    return {
        "document_id": document_id,
        "stats": stats,
        "output_paths": output_paths,
    }
