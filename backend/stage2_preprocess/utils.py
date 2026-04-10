"""Stage-2 preprocessing utility and render helpers."""

from __future__ import annotations

import base64
import json
import mimetypes
import re
from html import escape
from pathlib import Path
from typing import Any, Optional, Sequence

import fitz  # PyMuPDF
from markdownify import markdownify as html_to_markdown


JUNK_FIGURE_LABELS = {
    "logo",
    "icon",
    "qr_code",
    "bar_code",
    "page_thumbnail",
}
TABLE_LIKE_FIGURE_OVERLAP_THRESHOLD = 0.70
PAGE_COUNTER_PATTERN = re.compile(
    r"^\s*(?:page\s+\d+|\d+\s*(?:/|of)\s*\d+|-?\s*\d+\s*-?)\s*$",
    re.IGNORECASE,
)
SHORT_HEADING_PATTERN = re.compile(
    r"^\s*(?:\d+\.\s+|[가-하]\.\s+|[A-Za-z]\.\s+|[IVXLC]+\.\s+).+"
)


def normalize_whitespace(text: str) -> str:
    """여러 공백과 줄바꿈을 하나의 공백으로 정리한다."""
    return re.sub(r"\s+", " ", text).strip()


def clean_render_text(text: str) -> str:
    """Docling placeholder / HTML comment 같은 렌더링 노이즈를 제거한다."""
    if not text:
        return ""
    cleaned = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    cleaned = re.sub(
        r"Image not available\.[^.]*?(?:PdfPipelineOptions\([^)]*\))?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("🖼️❌", " ")
    return normalize_whitespace(cleaned)


def safe_mkdir(path: Path) -> Path:
    """대상 디렉토리를 안전하게 생성하고 그대로 반환한다."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_write_text(path: Path, content: str) -> Path:
    """UTF-8 텍스트 파일을 저장하고 경로를 반환한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def safe_write_json(path: Path, payload: dict[str, Any]) -> Path:
    """JSON payload를 UTF-8 pretty format으로 저장한다."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def bbox_to_rect(
    bbox: Sequence[float],
    page_height: float,
    coord_origin: Optional[str],
) -> fitz.Rect:
    """Docling bbox를 PyMuPDF crop용 Rect로 변환한다."""
    left_raw, top_raw, right_raw, bottom_raw = [float(v) for v in bbox]
    origin = (coord_origin or "").upper()

    if "TOP" in origin:
        top = min(top_raw, bottom_raw)
        bottom = max(top_raw, bottom_raw)
    else:
        docling_bottom = min(top_raw, bottom_raw)
        docling_top = max(top_raw, bottom_raw)
        top = page_height - docling_top
        bottom = page_height - docling_bottom

    left = min(left_raw, right_raw)
    right = max(left_raw, right_raw)
    return fitz.Rect(left, top, right, bottom)


def bbox_overlap_ratio(a: Sequence[float], b: Sequence[float]) -> float:
    """두 bbox의 겹침 정도를 작은 쪽 면적 기준 비율로 계산한다."""
    a_left, a_top, a_right, a_bottom = [float(v) for v in a]
    b_left, b_top, b_right, b_bottom = [float(v) for v in b]

    a_x1, a_x2 = min(a_left, a_right), max(a_left, a_right)
    a_y1, a_y2 = min(a_top, a_bottom), max(a_top, a_bottom)
    b_x1, b_x2 = min(b_left, b_right), max(b_left, b_right)
    b_y1, b_y2 = min(b_top, b_bottom), max(b_top, b_bottom)

    inter_x1 = max(a_x1, b_x1)
    inter_y1 = max(a_y1, b_y1)
    inter_x2 = min(a_x2, b_x2)
    inter_y2 = min(a_y2, b_y2)
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    a_area = max((a_x2 - a_x1) * (a_y2 - a_y1), 1.0)
    b_area = max((b_x2 - b_x1) * (b_y2 - b_y1), 1.0)
    return inter_area / min(a_area, b_area)


def guess_primary_picture_label(element: dict[str, Any]) -> tuple[Optional[str], float]:
    """picture_candidates의 top-1 label과 confidence를 반환한다."""
    candidates = element.get("picture_candidates") or []
    if not candidates:
        return None, 0.0
    first = candidates[0]
    return first.get("label"), float(first.get("confidence", 0.0) or 0.0)


def looks_like_page_counter(text: str) -> bool:
    """페이지 카운터처럼 보이는 짧은 텍스트인지 검사한다."""
    return bool(PAGE_COUNTER_PATTERN.match(text or ""))


def looks_like_short_heading(text: str) -> bool:
    """짧은 번호형 문구를 heading 승격 후보로 판단한다."""
    clean_text = clean_render_text(text)
    if len(clean_text) > 60:
        return False
    return bool(SHORT_HEADING_PATTERN.match(clean_text))


def is_obvious_junk_figure(element: dict[str, Any]) -> Optional[str]:
    """규칙만으로 확실히 제거할 수 있는 figure인지 판단한다."""
    label, confidence = guess_primary_picture_label(element)
    if label in JUNK_FIGURE_LABELS and confidence >= 0.55:
        return f"junk_label:{label}"
    return None


def image_to_data_url(image_path: Path) -> str:
    """로컬 crop 이미지를 멀티모달 모델 입력용 data URL로 변환한다."""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime_type = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def collect_page_context(elements: Sequence[dict[str, Any]], page: int) -> str:
    """같은 페이지의 텍스트 문맥 일부를 VLM 프롬프트용으로 모은다."""
    lines: list[str] = []
    for element in elements:
        if element.get("page") != page:
            continue
        if element.get("category") in {"figure", "table", "page_header", "page_footer"}:
            continue
        text = clean_render_text(element.get("text", ""))
        if not text:
            continue
        lines.append(text)
    return "\n".join(lines[:12])


def collect_document_profile_inputs(
    elements: Sequence[dict[str, Any]],
) -> tuple[list[str], list[str]]:
    """문서 주제 추론에 사용할 heading / 본문 샘플을 앞부분에서 모은다."""
    headings: list[str] = []
    body_snippets: list[str] = []

    for element in elements:
        category = element.get("category")
        if category in {"figure", "table", "caption", "page_header", "page_footer"}:
            continue

        text = clean_render_text(element.get("text", ""))
        if not text:
            continue

        if category == "heading":
            if text not in headings:
                headings.append(text)
        else:
            body_snippets.append(text)

        if len(headings) >= 12 and len(body_snippets) >= 18:
            break

    return headings[:12], body_snippets[:18]


def fallback_visual_summary(element: dict[str, Any]) -> Optional[str]:
    """VLM 실패 시 caption이나 top-1 label 기반 최소 summary를 만든다."""
    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    if caption:
        return caption
    label, _confidence = guess_primary_picture_label(element)
    if label:
        return f"{label.replace('_', ' ')} 관련 이미지"
    return None


def render_text_like_markdown(element: dict[str, Any]) -> str:
    """일반 텍스트 계열 element를 Markdown 조각으로 렌더링한다."""
    html = element.get("html") or ""
    text = clean_render_text(element.get("text", ""))
    if html:
        rendered = html_to_markdown(html, heading_style="ATX").strip()
        if rendered:
            return rendered
    return text


def render_figure_markdown(element: dict[str, Any]) -> str:
    """figure element를 최종 Markdown 블록으로 렌더링한다."""
    image_path = element.get("image_path")
    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    alt_text = caption or fallback_visual_summary(element) or "Figure"
    blocks: list[str] = []
    if image_path:
        blocks.append(f"![{alt_text}]({image_path})")
    if caption:
        blocks.append(caption)
    summary = clean_render_text(element.get("visual_summary", ""))
    if summary:
        blocks.append(f"요약: {summary}")
    return "\n\n".join(blocks).strip()


def render_table_markdown(element: dict[str, Any]) -> str:
    """table element를 Markdown 블록으로 렌더링한다."""
    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    table_markdown = (element.get("table") or {}).get("markdown") or ""
    summary = clean_render_text(element.get("table_summary", ""))

    blocks: list[str] = []
    if caption:
        blocks.append(f"**{caption}**")
    if table_markdown:
        blocks.append(table_markdown.strip())
    if summary:
        blocks.append(f"요약: {summary}")
    return "\n\n".join(blocks).strip()


def render_text_like_html(element: dict[str, Any]) -> str:
    """일반 텍스트 계열 element를 HTML fragment로 렌더링한다."""
    html = element.get("html") or ""
    text = clean_render_text(element.get("text", ""))
    if html:
        return html
    if not text:
        return ""
    if element.get("category") == "heading":
        return f"<h2>{escape(text)}</h2>"
    return f"<p>{escape(text)}</p>"


def render_figure_html(element: dict[str, Any]) -> str:
    """figure element를 preview용 HTML fragment로 렌더링한다."""
    image_path = element.get("image_path")
    if not image_path:
        return ""

    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    alt_text = caption or fallback_visual_summary(element) or "Figure"
    summary = clean_render_text(element.get("visual_summary", ""))

    parts = [
        "<figure>",
        f"<img src=\"{escape(image_path)}\" alt=\"{escape(alt_text)}\" />",
    ]
    if caption:
        parts.append(f"<figcaption>{escape(caption)}</figcaption>")
    if summary:
        parts.append(f"<p class=\"figure-summary\">요약: {escape(summary)}</p>")
    parts.append("</figure>")
    return "\n".join(parts)


def render_table_html(element: dict[str, Any]) -> str:
    """table element를 preview용 HTML fragment로 렌더링한다."""
    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    summary = clean_render_text(element.get("table_summary", ""))
    table_html = element.get("html") or ""
    image_path = element.get("image_path")

    parts = ["<section class=\"table-block\">"]
    if caption:
        parts.append(f"<p><strong>{escape(caption)}</strong></p>")
    if table_html:
        parts.append(table_html)
    if summary:
        parts.append(f"<p class=\"table-summary\">요약: {escape(summary)}</p>")
    if image_path:
        parts.append(
            f"<details><summary>Original crop</summary><img src=\"{escape(image_path)}\" alt=\"Original table crop\" /></details>"
        )
    parts.append("</section>")
    return "\n".join(parts)
