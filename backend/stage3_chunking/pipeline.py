"""Stage-3 chunking pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import (
    DEFAULT_CHUNKS_MD_NAME,
    DEFAULT_CHUNKS_JSON_NAME,
    DEFAULT_CHUNKS_JSONL_NAME,
    DEFAULT_CLEANED_JSON_PATH,
    DEFAULT_PARENTS_JSON_NAME,
    STAGE3_PARENT_MAX_TOKENS,
    STAGE3_TEXT_MAX_TOKENS,
    STAGE3_TEXT_MIN_TOKENS,
    STAGE3_TEXT_OVERLAP_TOKENS,
    STAGE3_TEXT_TARGET_TOKENS,
)
from .embeddings import SemanticEmbeddingClient
from .schemas import (
    ChunkPayload,
    ParentPayload,
    ChunkSourceElement,
    Stage3ChunkStats,
    Stage3Input,
    Stage3Output,
    Stage3OutputPaths,
)

TEXTUAL_CATEGORIES = {"paragraph", "footnote", "list", "code"}
PROSE_CATEGORIES = {"paragraph", "footnote"}
SEMANTIC_BOUNDARY_CATEGORIES = {"list", "code", "table", "figure"}
VISUAL_CATEGORIES = {"table", "figure"}
REFERENCE_SECTION_HINT_TERMS = {
    "references",
    "reference",
    "bibliography",
    "works cited",
    "참고문헌",
    "참고 자료",
    "참고자료",
}
EMAIL_PATTERN = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)
URL_PATTERN = re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)
YEAR_PATTERN = re.compile(r"\b(?:19|20)\d{2}\b")
CITATION_BRACKET_PATTERN = re.compile(r"\[[0-9,\-\s]{1,20}\]")
CITATION_PAREN_PATTERN = re.compile(r"\([^)]{0,80}\b(?:19|20)\d{2}\b[^)]{0,80}\)")
SENTENCE_END_PATTERN = re.compile(r"[.!?。！？]\s*$")


@dataclass(frozen=True)
class TextSegment:
    """하나의 text chunk 내부에서 분할 가능한 최소 단위."""

    text: str
    source_elements: list[ChunkSourceElement]
    element_ids: list[int]
    pages: list[int]


@dataclass(frozen=True)
class AtomicTextUnit:
    """cleaned element를 text chunk용 원자 단위로 변환한 내부 표현."""

    element_id: int
    page: int
    category: str
    text: str
    heading_path: tuple[str, ...]
    group_type: str
    order: int

    def to_segment(self) -> TextSegment:
        return TextSegment(
            text=self.text,
            source_elements=[
                {
                    "element_id": self.element_id,
                    "page": self.page,
                    "category": self.category,
                }
            ],
            element_ids=[self.element_id],
            pages=[self.page],
        )


@dataclass
class ChunkDraft:
    """최종 ChunkPayload로 직렬화되기 전 내부 중간 표현."""

    chunk_type: str
    heading_path: list[str]
    base_text: str
    pages: list[int]
    element_ids: list[int]
    source_elements: list[ChunkSourceElement]
    metadata: dict[str, Any]
    order_key: tuple[int, int]
    semantic_eligible: bool = False
    segments: list[TextSegment] = field(default_factory=list)


def build_stage3_output_paths(
    *,
    cleaned_json_path: str | Path,
    output_dir: str | Path | None = None,
) -> Stage3OutputPaths:
    """stage3가 기록할 chunk 산출물 경로를 계산한다."""
    cleaned_path = Path(cleaned_json_path).expanduser().resolve()
    default_output_dir = (
        cleaned_path.parent.parent / "stage3"
        if cleaned_path.parent.name in {"stage2", "review"}
        else cleaned_path.parent.resolve()
    )
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else default_output_dir
    )
    return {
        "chunks_json": str((resolved_output_dir / DEFAULT_CHUNKS_JSON_NAME).resolve()),
        "chunks_jsonl": str((resolved_output_dir / DEFAULT_CHUNKS_JSONL_NAME).resolve()),
        "chunks_md": str((resolved_output_dir / DEFAULT_CHUNKS_MD_NAME).resolve()),
        "parents_json": str((resolved_output_dir / DEFAULT_PARENTS_JSON_NAME).resolve()),
    }


def _normalize_text(text: str | None) -> str:
    """chunk 조립 전에 사용할 기본 텍스트 정리를 수행한다."""
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _estimate_tokens(text: str) -> int:
    """정확한 tokenizer 대신 보수적인 추정 토큰 수를 계산한다."""
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    word_units = len(re.findall(r"\S+", normalized))
    char_units = len(re.sub(r"\s+", "", normalized))
    return max(word_units, max(1, char_units // 4))


def _build_section_title_from_heading_path(heading_path: Iterable[str]) -> str | None:
    """heading_path를 사람이 읽기 쉬운 단일 섹션 문자열로 평탄화한다."""
    normalized = [str(item).strip() for item in heading_path if str(item).strip()]
    if not normalized:
        return None
    return " > ".join(normalized)


def _normalize_hint_text(text: str | None) -> str:
    """역할 힌트 비교용 텍스트를 느슨하게 정규화한다."""
    if not text:
        return ""
    normalized = re.sub(r"\s+", " ", str(text).strip().lower())
    return normalized


def _unique_ints(values: Iterable[int]) -> list[int]:
    seen: set[int] = set()
    ordered: list[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _unique_source_elements(
    source_elements: Iterable[ChunkSourceElement],
) -> list[ChunkSourceElement]:
    seen: set[tuple[int, int, str]] = set()
    ordered: list[ChunkSourceElement] = []
    for source in source_elements:
        key = (source["element_id"], source["page"], source["category"])
        if key in seen:
            continue
        seen.add(key)
        ordered.append(source)
    return ordered


def _count_nonempty_lines(text: str) -> int:
    return len([line for line in text.splitlines() if line.strip()])


def _estimate_sentence_like_ratio(text: str) -> float:
    """문장 끝맺음이 있는 라인 비율로 본문성 여부를 보수적으로 추정한다."""
    lines = [_normalize_text(line) for line in text.splitlines() if _normalize_text(line)]
    if not lines:
        return 0.0
    sentence_like_lines = sum(
        1
        for line in lines
        if SENTENCE_END_PATTERN.search(line)
    )
    return sentence_like_lines / len(lines)


def _build_sparse_role_hints(
    chunk: ChunkPayload,
    *,
    total_pages: int,
) -> dict[str, Any]:
    """BM25 전용 필터링에 쓸 약한 역할 힌트를 계산한다."""
    metadata = chunk.get("metadata") or {}
    text = _normalize_text(chunk.get("text"))
    section_title = _build_section_title_from_heading_path(
        chunk.get("heading_path") or []
    )
    normalized_section_title = _normalize_hint_text(section_title)
    normalized_text = _normalize_hint_text(text)
    last_heading = _normalize_hint_text((chunk.get("heading_path") or [""])[-1])

    estimated_tokens = int(metadata.get("estimated_tokens") or _estimate_tokens(text))
    line_count = _count_nonempty_lines(text)
    has_email = bool(EMAIL_PATTERN.search(text))
    has_url = bool(URL_PATTERN.search(text))
    year_like_count = len(YEAR_PATTERN.findall(text))
    citation_like_count = len(CITATION_BRACKET_PATTERN.findall(text)) + len(
        CITATION_PAREN_PATTERN.findall(text)
    )
    sentence_like_ratio = _estimate_sentence_like_ratio(text)

    page_start = None
    pages = [int(page) for page in chunk.get("pages") or [] if str(page).isdigit()]
    if pages:
        page_start = sorted(set(pages))[0]
    tail_page_hint = bool(page_start is not None and total_pages >= 4 and page_start >= total_pages - 1)
    early_page_hint = bool(page_start is not None and page_start <= 2)
    average_line_tokens = estimated_tokens / line_count if line_count else float(estimated_tokens)

    reference_heading_hint = any(
        term in normalized_section_title
        for term in REFERENCE_SECTION_HINT_TERMS
    )
    reference_like = bool(
        reference_heading_hint
        or (
            str(chunk.get("chunk_type") or "") == "text"
            and tail_page_hint
            and citation_like_count >= 2
            and year_like_count >= 2
            and sentence_like_ratio < 0.5
        )
    )
    front_matter_like = bool(
        str(chunk.get("chunk_type") or "") == "text"
        and early_page_hint
        and not reference_like
        and estimated_tokens <= 160
        and (
            has_email
            or (
                not section_title
                and line_count >= 3
                and average_line_tokens <= 8
                and sentence_like_ratio < 0.34
            )
        )
    )
    title_only = bool(
        str(chunk.get("chunk_type") or "") == "text"
        and estimated_tokens <= 18
        and line_count <= 2
        and sentence_like_ratio < 0.34
        and not has_email
        and not has_url
        and (
            (last_heading and (normalized_text == last_heading or normalized_text in last_heading or last_heading in normalized_text))
            or (
                normalized_section_title
                and (
                    normalized_text == normalized_section_title
                    or normalized_text in normalized_section_title
                    or normalized_section_title in normalized_text
                )
            )
        )
    )

    sparse_role_hints: list[str] = []
    if reference_like:
        sparse_role_hints.append("reference_like")
    if front_matter_like:
        sparse_role_hints.append("front_matter_like")
    if title_only:
        sparse_role_hints.append("title_only")

    return {
        "section_title": section_title,
        "line_count": line_count,
        "has_email": has_email,
        "has_url": has_url,
        "year_like_count": year_like_count,
        "citation_like_count": citation_like_count,
        "sentence_like_ratio": round(sentence_like_ratio, 4),
        "sparse_role_hints": sparse_role_hints,
    }


def _annotate_sparse_filter_metadata(
    chunks: list[ChunkPayload],
    *,
    total_pages: int,
) -> list[ChunkPayload]:
    """stage4가 BM25 필터링에 재사용할 중립 메타데이터를 chunk에 추가한다."""
    for chunk in chunks:
        metadata = dict(chunk.get("metadata") or {})
        sparse_metadata = _build_sparse_role_hints(
            chunk,
            total_pages=total_pages,
        )
        metadata.update(sparse_metadata)
        chunk["metadata"] = metadata
    return chunks


def _element_sort_key(element: dict[str, Any]) -> tuple[int, int]:
    page = int(element.get("page") or 0)
    order = int(element.get("resolved_order") or element.get("order") or 0)
    return (page, order)


def _infer_heading_level(element: dict[str, Any]) -> int:
    """heading level이 별도 필드에 없을 때 html/text로 보수적으로 추정한다."""
    explicit_level = element.get("heading_level")
    if isinstance(explicit_level, int) and 1 <= explicit_level <= 6:
        return explicit_level

    html = element.get("html") or ""
    match = re.search(r"<h([1-6])\b", html)
    if match:
        return int(match.group(1))

    text = _normalize_text(element.get("text"))
    numeric_match = re.match(r"^(\d+(?:\.\d+)*)", text)
    if numeric_match:
        return min(6, numeric_match.group(1).count(".") + 1)
    return 2


def _update_heading_path(
    heading_stack: list[str],
    *,
    level: int,
    heading_text: str,
) -> list[str]:
    """새 heading을 반영해 현재 heading path를 갱신한다."""
    normalized_level = min(6, max(1, level))
    while len(heading_stack) < normalized_level:
        heading_stack.append("")
    heading_stack[normalized_level - 1] = heading_text
    del heading_stack[normalized_level:]
    return [item for item in heading_stack if item]


def _build_chunk_source(element: dict[str, Any]) -> ChunkSourceElement:
    return {
        "element_id": int(element["id"]),
        "page": int(element.get("page") or 0),
        "category": str(element.get("category") or ""),
    }


def _extract_visual_caption(element: dict[str, Any]) -> str:
    category = str(element.get("category") or "")
    for key in ("resolved_caption", "internal_caption_text"):
        text = _normalize_text(element.get(key))
        if text:
            return text

    fallback_text = _normalize_text(element.get("text"))
    if not fallback_text:
        return ""

    if category == "table":
        if fallback_text.startswith("|"):
            return ""
        if len(fallback_text) > 240:
            return ""
        return fallback_text

    if len(fallback_text) > 320:
        return ""
    return fallback_text


def _strip_duplicate_leading_caption(body_text: str, caption: str) -> str:
    """table body 선두에 caption이 그대로 반복되면 한 번 제거한다."""
    normalized_body = _normalize_text(body_text)
    normalized_caption = _normalize_text(caption)
    if not normalized_body or not normalized_caption:
        return normalized_body

    body_lines = normalized_body.splitlines()
    while body_lines and not body_lines[0].strip():
        body_lines.pop(0)
    if not body_lines:
        return ""

    first_line = _normalize_text(body_lines[0])
    if first_line != normalized_caption:
        return normalized_body

    return "\n".join(body_lines[1:]).strip()


def _extract_table_body(element: dict[str, Any]) -> str:
    caption = _extract_visual_caption(element)
    table_payload = element.get("table") or {}
    markdown = _normalize_text(table_payload.get("markdown"))
    if markdown:
        return _strip_duplicate_leading_caption(markdown, caption)
    html_excerpt = _normalize_text(element.get("html"))
    if html_excerpt:
        return html_excerpt
    return _strip_duplicate_leading_caption(
        _normalize_text(element.get("text")),
        caption,
    )


def _extract_figure_summary(element: dict[str, Any]) -> str:
    return _normalize_text(
        element.get("visual_summary")
        or element.get("summary")
        or element.get("text")
    )


def _build_visual_chunk_text(
    *,
    caption: str,
    summary: str,
    body_text: str = "",
) -> str:
    parts: list[str] = []
    if caption:
        parts.append(caption)
    if summary:
        parts.append(summary)
    if body_text:
        parts.append(body_text)
    return "\n\n".join(part for part in parts if part)


def _take_tail_for_overlap(text: str, overlap_tokens: int) -> str:
    if overlap_tokens <= 0:
        return ""
    approximate_chars = overlap_tokens * 4
    normalized = _normalize_text(text)
    if len(normalized) <= approximate_chars:
        return normalized
    tail = normalized[-approximate_chars:]
    if "\n" in tail:
        tail = tail.split("\n", maxsplit=1)[-1]
    return tail.strip()


def _split_text_to_sentence_segments(
    text: str,
    source: ChunkSourceElement,
) -> list[TextSegment]:
    """단일 거대 paragraph를 문장 단위로 쪼개는 보조 fallback."""
    sentences = [
        _normalize_text(part)
        for part in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", text)
        if _normalize_text(part)
    ]
    if not sentences:
        return []
    return [
        TextSegment(
            text=sentence,
            source_elements=[source],
            element_ids=[source["element_id"]],
            pages=[source["page"]],
        )
        for sentence in sentences
    ]


def _join_segment_texts(segments: list[TextSegment], *, group_type: str) -> str:
    separator = "\n\n" if group_type == "prose" else "\n"
    return separator.join(segment.text for segment in segments if segment.text)


def _build_text_chunk_from_segments(
    *,
    parent: ChunkDraft,
    segments: list[TextSegment],
    hard_split_applied: bool,
) -> ChunkDraft:
    group_type = str(parent.metadata.get("group_type") or "prose")
    source_elements = _unique_source_elements(
        source
        for segment in segments
        for source in segment.source_elements
    )
    pages = _unique_ints(page for segment in segments for page in segment.pages)
    element_ids = _unique_ints(
        element_id for segment in segments for element_id in segment.element_ids
    )
    metadata = dict(parent.metadata)
    metadata["semantic_split_applied"] = False
    metadata["semantic_merge_applied"] = False
    metadata["hard_split_applied"] = hard_split_applied
    base_text = _join_segment_texts(segments, group_type=group_type)
    metadata["estimated_tokens"] = _estimate_tokens(base_text)
    return ChunkDraft(
        chunk_type="text",
        heading_path=list(parent.heading_path),
        base_text=base_text,
        pages=pages,
        element_ids=element_ids,
        source_elements=source_elements,
        metadata=metadata,
        order_key=parent.order_key,
        semantic_eligible=parent.semantic_eligible,
        segments=list(segments),
    )


def _hard_split_segments(
    segments: list[TextSegment],
    *,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
) -> list[list[TextSegment]]:
    """semantic embedding이 없을 때 사용할 크기 기반 fallback split."""
    if not segments:
        return []

    parts: list[list[TextSegment]] = []
    current: list[TextSegment] = []
    current_tokens = 0

    for segment in segments:
        segment_tokens = _estimate_tokens(segment.text)
        if (
            current
            and current_tokens >= min_tokens
            and (
                current_tokens + segment_tokens > max_tokens
                or current_tokens >= target_tokens
            )
        ):
            parts.append(current)
            current = []
            current_tokens = 0

        current.append(segment)
        current_tokens += segment_tokens

    if current:
        if parts and current_tokens < min_tokens:
            parts[-1].extend(current)
        else:
            parts.append(current)
    return parts


def _split_text_draft(
    draft: ChunkDraft,
) -> list[ChunkDraft]:
    """긴 prose chunk를 구조 기반 hard split으로 분할한다."""
    estimated_tokens = _estimate_tokens(draft.base_text)
    if estimated_tokens <= STAGE3_TEXT_MAX_TOKENS:
        draft.metadata["estimated_tokens"] = estimated_tokens
        draft.metadata["semantic_split_applied"] = False
        draft.metadata["semantic_merge_applied"] = False
        draft.metadata["hard_split_applied"] = False
        return [draft]

    segments = list(draft.segments)
    if not segments and draft.source_elements:
        segments = _split_text_to_sentence_segments(draft.base_text, draft.source_elements[0])
    elif len(segments) == 1:
        segments = _split_text_to_sentence_segments(
            draft.base_text,
            draft.source_elements[0],
        ) or segments

    if len(segments) <= 1:
        draft.metadata["estimated_tokens"] = estimated_tokens
        draft.metadata["semantic_split_applied"] = False
        draft.metadata["semantic_merge_applied"] = False
        draft.metadata["hard_split_applied"] = False
        return [draft]

    split_parts = _hard_split_segments(
        segments,
        target_tokens=STAGE3_TEXT_TARGET_TOKENS,
        max_tokens=STAGE3_TEXT_MAX_TOKENS,
        min_tokens=STAGE3_TEXT_MIN_TOKENS,
    )
    hard_split_applied = len(split_parts) > 1

    return [
        _build_text_chunk_from_segments(
            parent=draft,
            segments=part,
            hard_split_applied=hard_split_applied,
        )
        for part in split_parts
        if part
    ]


def _build_visual_chunk(
    element: dict[str, Any],
    *,
    heading_path: list[str],
) -> ChunkDraft:
    category = str(element.get("category"))
    caption = _extract_visual_caption(element)
    summary = _normalize_text(
        element.get("table_summary")
        if category == "table"
        else _extract_figure_summary(element)
    )
    body_text = _extract_table_body(element) if category == "table" else ""
    base_text = _build_visual_chunk_text(
        caption=caption,
        summary=summary,
        body_text=body_text,
    )
    return ChunkDraft(
        chunk_type=category,
        heading_path=list(heading_path),
        base_text=base_text,
        pages=[int(element.get("page") or 0)],
        element_ids=[int(element["id"])],
        source_elements=[_build_chunk_source(element)],
        metadata={
            "group_type": category,
            "caption": caption or None,
            "summary_text": summary or None,
            "summary_present": bool(summary),
            "image_path": element.get("image_path"),
            "estimated_tokens": _estimate_tokens(base_text),
        },
        order_key=_element_sort_key(element),
    )


def _classify_text_group(category: str) -> str | None:
    if category in PROSE_CATEGORIES:
        return "prose"
    if category == "list":
        return "list"
    if category == "code":
        return "code"
    return None


def _flush_pending_text_units(
    chunks: list[ChunkDraft],
    pending_units: list[AtomicTextUnit],
) -> None:
    if not pending_units:
        return

    group_type = pending_units[0].group_type
    segments = [unit.to_segment() for unit in pending_units]
    base_text = _join_segment_texts(segments, group_type=group_type)
    chunks.append(
        ChunkDraft(
            chunk_type="text",
            heading_path=list(pending_units[0].heading_path),
            base_text=base_text,
            pages=_unique_ints(unit.page for unit in pending_units),
            element_ids=_unique_ints(unit.element_id for unit in pending_units),
            source_elements=_unique_source_elements(
                _build_chunk_source(
                    {
                        "id": unit.element_id,
                        "page": unit.page,
                        "category": unit.category,
                    }
                )
                for unit in pending_units
            ),
            metadata={
                "group_type": group_type,
                "source_run_id": f"text-run-{pending_units[0].element_id}",
                "semantic_split_applied": False,
                "semantic_merge_applied": False,
                "hard_split_applied": False,
                "estimated_tokens": _estimate_tokens(base_text),
            },
            order_key=(pending_units[0].page, pending_units[0].order),
            semantic_eligible=group_type == "prose",
            segments=segments,
        )
    )
    pending_units.clear()


def _build_initial_chunk_drafts(elements: list[dict[str, Any]]) -> list[ChunkDraft]:
    """cleaned elements를 읽어 구조 기반 초안 chunk를 만든다."""
    drafts: list[ChunkDraft] = []
    heading_stack: list[str] = []
    current_heading_path: list[str] = []
    pending_units: list[AtomicTextUnit] = []

    for element in sorted(elements, key=_element_sort_key):
        category = str(element.get("category") or "")
        text = _normalize_text(element.get("text"))

        if category == "heading" and text:
            _flush_pending_text_units(drafts, pending_units)
            current_heading_path = _update_heading_path(
                heading_stack,
                level=_infer_heading_level(element),
                heading_text=text,
            )
            continue

        if category in VISUAL_CATEGORIES:
            _flush_pending_text_units(drafts, pending_units)
            drafts.append(
                _build_visual_chunk(
                    element,
                    heading_path=current_heading_path,
                )
            )
            continue

        if category == "caption":
            # caption은 독립 text chunk로 만들지 않고 대응 visual chunk가 흡수한다.
            continue

        if category not in TEXTUAL_CATEGORIES or not text:
            continue

        group_type = _classify_text_group(category)
        if group_type is None:
            continue

        current_order = int(element.get("resolved_order") or element.get("order") or 0)
        next_unit = AtomicTextUnit(
            element_id=int(element["id"]),
            page=int(element.get("page") or 0),
            category=category,
            text=text,
            heading_path=tuple(current_heading_path),
            group_type=group_type,
            order=current_order,
        )
        if pending_units:
            last_unit = pending_units[-1]
            if (
                last_unit.heading_path != next_unit.heading_path
                or last_unit.group_type != next_unit.group_type
                or category in SEMANTIC_BOUNDARY_CATEGORIES
                and last_unit.category != category
            ):
                _flush_pending_text_units(drafts, pending_units)
        pending_units.append(next_unit)

    _flush_pending_text_units(drafts, pending_units)
    return drafts


def _apply_hard_split(
    drafts: list[ChunkDraft],
) -> list[ChunkDraft]:
    split_drafts: list[ChunkDraft] = []
    for draft in drafts:
        if draft.chunk_type != "text":
            split_drafts.append(draft)
            continue
        split_drafts.extend(_split_text_draft(draft))
    return split_drafts


def _finalize_chunk_payloads(drafts: list[ChunkDraft]) -> list[ChunkPayload]:
    """내부 draft를 최종 ChunkPayload 목록으로 직렬화한다."""
    ordered = sorted(drafts, key=lambda draft: draft.order_key)
    payloads: list[ChunkPayload] = []
    type_counters = {"text": 0, "table": 0, "figure": 0}
    previous_text_by_heading: dict[tuple[str, ...], ChunkDraft] = {}

    for draft in ordered:
        chunk_type = draft.chunk_type
        if chunk_type not in type_counters:
            type_counters[chunk_type] = 0
        type_counters[chunk_type] += 1
        chunk_id = f"{chunk_type}-{type_counters[chunk_type]:04d}"

        rendered_text = draft.base_text
        metadata = dict(draft.metadata)

        if chunk_type == "text":
            heading_key = tuple(draft.heading_path)
            overlap_text = ""
            previous_chunk = previous_text_by_heading.get(heading_key)
            same_source_run = (
                previous_chunk is not None
                and previous_chunk.metadata.get("source_run_id")
                and previous_chunk.metadata.get("source_run_id")
                == metadata.get("source_run_id")
            )
            if same_source_run and previous_chunk is not None:
                overlap_text = _take_tail_for_overlap(
                    previous_chunk.base_text,
                    STAGE3_TEXT_OVERLAP_TOKENS,
                )
            rendered_text = draft.base_text
            metadata["overlap_applied"] = bool(overlap_text)
            metadata["overlap_text"] = overlap_text or None
            previous_text_by_heading[heading_key] = draft

        payloads.append(
            {
                "chunk_id": chunk_id,
                "chunk_type": chunk_type,  # type: ignore[typeddict-item]
                "text": rendered_text,
                "pages": draft.pages,
                "heading_path": draft.heading_path,
                "element_ids": draft.element_ids,
                "source_elements": draft.source_elements,
                "metadata": {
                    **metadata,
                    "estimated_tokens": _estimate_tokens(rendered_text),
                },
            }
        )
    return payloads


def _build_stats(chunks: list[ChunkPayload]) -> Stage3ChunkStats:
    text_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "text"]
    table_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "table"]
    figure_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "figure"]
    return {
        "total_chunks": len(chunks),
        "text_chunks": len(text_chunks),
        "table_chunks": len(table_chunks),
        "figure_chunks": len(figure_chunks),
        "semantic_split_chunks": sum(
            1
            for chunk in text_chunks
            if chunk["metadata"].get("semantic_split_applied")
        ),
        "semantic_merge_chunks": sum(
            1
            for chunk in text_chunks
            if chunk["metadata"].get("semantic_merge_applied")
        ),
    }


def _derive_document_id(*, cleaned_json_path: Path) -> str:
    """stage3 출력물 묶음을 식별할 문서 id를 폴더명 기준으로 만든다."""
    if cleaned_json_path.parent.name in {"stage2", "review"}:
        return cleaned_json_path.parent.parent.name
    return cleaned_json_path.parent.name


def _build_parent_payloads(
    chunks: list[ChunkPayload],
    *,
    document_id: str,
) -> tuple[list[ChunkPayload], list[ParentPayload]]:
    """정렬된 child chunk를 상위 parent 문맥 블록으로 묶는다."""
    if not chunks:
        return (chunks, [])

    parent_counter = 0
    parents: list[ParentPayload] = []
    current_group: list[ChunkPayload] = []
    current_heading_path: tuple[str, ...] | None = None
    current_tokens = 0

    def flush_current_group() -> None:
        nonlocal parent_counter, current_group, current_heading_path, current_tokens
        if not current_group:
            return

        parent_counter += 1
        parent_id = f"parent-{parent_counter:04d}"
        pages = _unique_ints(
            int(page)
            for chunk in current_group
            for page in chunk.get("pages") or []
            if isinstance(page, int) or str(page).isdigit()
        )
        heading_path = list(current_group[0].get("heading_path") or [])
        parent_chunk_types: list[str] = []
        seen_types: set[str] = set()
        for chunk in current_group:
            chunk_type = str(chunk.get("chunk_type") or "")
            if chunk_type and chunk_type not in seen_types:
                seen_types.add(chunk_type)
                parent_chunk_types.append(chunk_type)

        parent_text = "\n\n".join(
            str(chunk.get("text") or "").strip()
            for chunk in current_group
            if str(chunk.get("text") or "").strip()
        )
        child_chunk_ids = [
            str(chunk.get("chunk_id") or "")
            for chunk in current_group
            if str(chunk.get("chunk_id") or "")
        ]

        for chunk in current_group:
            chunk["parent_id"] = parent_id

        parents.append(
            {
                "parent_id": parent_id,
                "document_id": document_id,
                "heading_path": heading_path,
                "section_title": _build_section_title_from_heading_path(heading_path),
                "pages": pages,
                "page_start": pages[0] if pages else None,
                "page_end": pages[-1] if pages else None,
                "child_chunk_ids": child_chunk_ids,
                "chunk_types": parent_chunk_types,
                "text": parent_text,
                "metadata": {
                    "child_count": len(current_group),
                    "estimated_tokens": _estimate_tokens(parent_text),
                    "has_visual": any(
                        chunk_type in {"table", "figure"}
                        for chunk_type in parent_chunk_types
                    ),
                },
            }
        )

        current_group = []
        current_heading_path = None
        current_tokens = 0

    for chunk in chunks:
        heading_path = tuple(chunk.get("heading_path") or [])
        chunk_tokens = _estimate_tokens(str(chunk.get("text") or ""))
        should_flush = False

        if current_group and heading_path != current_heading_path:
            should_flush = True
        elif (
            current_group
            and current_tokens + chunk_tokens > STAGE3_PARENT_MAX_TOKENS
        ):
            should_flush = True

        if should_flush:
            flush_current_group()

        current_group.append(chunk)
        current_heading_path = heading_path
        current_tokens += chunk_tokens

    flush_current_group()
    return (chunks, parents)


def _render_chunk_preview_markdown(
    chunks: list[ChunkPayload],
    *,
    cleaned_json_path: Path,
) -> str:
    """사람이 청크 결과를 빠르게 검수할 수 있는 markdown preview를 만든다."""
    stats = _build_stats(chunks)
    lines: list[str] = [
        "# Chunk Preview",
        "",
        f"- source: `{cleaned_json_path}`",
        f"- total_chunks: `{stats['total_chunks']}`",
        f"- text_chunks: `{stats['text_chunks']}`",
        f"- table_chunks: `{stats['table_chunks']}`",
        f"- figure_chunks: `{stats['figure_chunks']}`",
        f"- semantic_split_chunks: `{stats['semantic_split_chunks']}`",
        f"- semantic_merge_chunks: `{stats['semantic_merge_chunks']}`",
        "",
    ]

    for index, chunk in enumerate(chunks, start=1):
        heading_label = " > ".join(chunk.get("heading_path") or []) or "(없음)"
        pages_label = ", ".join(str(page) for page in chunk.get("pages") or [])
        element_ids_label = ", ".join(str(element_id) for element_id in chunk.get("element_ids") or [])
        metadata = chunk.get("metadata") or {}
        overlap_text = metadata.get("overlap_text") or ""
        caption_text = metadata.get("caption") or ""
        summary_text = metadata.get("summary_text") or ""

        lines.extend(
            [
                "-------------",
                "",
                f"## {index}번 청크",
                "",
                f"- chunk_id: `{chunk.get('chunk_id')}`",
                f"- parent_id: `{chunk.get('parent_id')}`",
                f"- chunk_type: `{chunk.get('chunk_type')}`",
                f"- pages: `{pages_label}`",
                f"- heading_path: `{heading_label}`",
                f"- element_ids: `{element_ids_label}`",
                f"- estimated_tokens: `{metadata.get('estimated_tokens', 0)}`",
                f"- overlap_applied: `{bool(metadata.get('overlap_applied'))}`",
                "",
            ]
        )
        if caption_text:
            lines.extend(
                [
                    "### 캡션",
                    "",
                    "```text",
                    str(caption_text),
                    "```",
                    "",
                ]
            )
        if summary_text:
            lines.extend(
                [
                    "### 요약",
                    "",
                    "```text",
                    str(summary_text),
                    "```",
                    "",
                ]
            )
        if overlap_text:
            lines.extend(
                [
                    "### 이전 문맥",
                    "",
                    "```text",
                    str(overlap_text),
                    "```",
                    "",
                ]
            )
        lines.extend(
            [
                "### 본문",
                "",
                "```text",
                chunk.get("text", ""),
                "```",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def _write_chunks(
    chunks: list[ChunkPayload],
    parents: list[ParentPayload],
    *,
    document_id: str,
    cleaned_json_path: Path,
    output_paths: Stage3OutputPaths,
) -> None:
    json_path = Path(output_paths["chunks_json"])
    jsonl_path = Path(output_paths["chunks_jsonl"])
    markdown_path = Path(output_paths["chunks_md"])
    parents_path = Path(output_paths["parents_json"])
    json_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _build_stats(chunks)
    json_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "cleaned_json_path": str(cleaned_json_path),
                "stats": stats,
                "chunks": chunks,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False))
            handle.write("\n")
    markdown_path.write_text(
        _render_chunk_preview_markdown(
            chunks,
            cleaned_json_path=cleaned_json_path,
        )
    )
    parents_path.write_text(
        json.dumps(
            {
                "document_id": document_id,
                "cleaned_json_path": str(cleaned_json_path),
                "parent_count": len(parents),
                "parents": parents,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _load_cleaned_document(cleaned_json_path: Path) -> dict[str, Any]:
    payload = json.loads(cleaned_json_path.read_text())
    if not isinstance(payload, dict):
        raise ValueError("cleaned.json은 dict 형태여야 합니다.")
    return payload


def run_stage3_chunking(
    inputs: Stage3Input,
    *,
    embedding_client: SemanticEmbeddingClient | None = None,
) -> Stage3Output:
    """cleaned.json을 읽어 retrieval용 chunk 산출물을 생성한다."""
    cleaned_json_path = Path(
        inputs.get("cleaned_json_path") or DEFAULT_CLEANED_JSON_PATH
    ).expanduser().resolve()
    output_dir = (
        Path(inputs["output_dir"]).expanduser().resolve()
        if inputs.get("output_dir")
        else (
            cleaned_json_path.parent.parent / "stage3"
            if cleaned_json_path.parent.name in {"stage2", "review"}
            else cleaned_json_path.parent.resolve()
        )
    )
    output_paths = build_stage3_output_paths(
        cleaned_json_path=cleaned_json_path,
        output_dir=output_dir,
    )

    cleaned_document = _load_cleaned_document(cleaned_json_path)
    elements = list(cleaned_document.get("elements") or [])

    initial_drafts = _build_initial_chunk_drafts(elements)
    split_drafts = _apply_hard_split(initial_drafts)
    chunks = _finalize_chunk_payloads(split_drafts)
    total_pages = max(
        (int(element.get("page") or 0) for element in elements),
        default=0,
    )
    chunks = _annotate_sparse_filter_metadata(
        chunks,
        total_pages=total_pages,
    )
    document_id = _derive_document_id(cleaned_json_path=cleaned_json_path)
    chunks, parents = _build_parent_payloads(
        chunks,
        document_id=document_id,
    )
    _write_chunks(
        chunks,
        parents,
        document_id=document_id,
        cleaned_json_path=cleaned_json_path,
        output_paths=output_paths,
    )

    stats = _build_stats(chunks)
    return {
        "cleaned_json_path": str(cleaned_json_path),
        "output_dir": str(output_dir),
        "output_paths": output_paths,
        "planned_outputs": output_paths,
        "chunk_count": len(chunks),
        "parent_count": len(parents),
        "stats": stats,
        "semantic_enabled": False,
        "semantic_fallback_reason": None,
        "status": "completed",
    }


def prepare_stage3_chunking(
    inputs: Stage3Input,
    *,
    embedding_client: SemanticEmbeddingClient | None = None,
) -> Stage3Output:
    """기존 함수명을 유지하면서 실제 chunking 실행까지 담당한다."""
    return run_stage3_chunking(inputs, embedding_client=embedding_client)


def main() -> None:
    """기본 cleaned.json 경로를 기준으로 stage3를 실행한다."""
    response = run_stage3_chunking(
        {
            "cleaned_json_path": str(DEFAULT_CLEANED_JSON_PATH),
        }
    )
    print(json.dumps(response, ensure_ascii=False, indent=2))
