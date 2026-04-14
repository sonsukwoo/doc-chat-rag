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
    STAGE3_ENABLE_SEMANTIC_MERGE,
    STAGE3_ENABLE_SEMANTIC_SPLIT,
    STAGE3_SEMANTIC_MERGE_CANDIDATE_MAX_TOKENS,
    STAGE3_SEMANTIC_MERGE_SIM_THRESHOLD,
    STAGE3_SEMANTIC_SPLIT_SIM_THRESHOLD,
    STAGE3_TEXT_MAX_TOKENS,
    STAGE3_TEXT_MIN_TOKENS,
    STAGE3_TEXT_OVERLAP_TOKENS,
    STAGE3_TEXT_TARGET_TOKENS,
)
from .embeddings import SemanticEmbeddingClient, cosine_similarity
from .schemas import (
    ChunkPayload,
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
    resolved_output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir is not None
        else cleaned_path.parent.resolve()
    )
    return {
        "chunks_json": str((resolved_output_dir / DEFAULT_CHUNKS_JSON_NAME).resolve()),
        "chunks_jsonl": str((resolved_output_dir / DEFAULT_CHUNKS_JSONL_NAME).resolve()),
        "chunks_md": str((resolved_output_dir / DEFAULT_CHUNKS_MD_NAME).resolve()),
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


def _extract_table_body(element: dict[str, Any]) -> str:
    table_payload = element.get("table") or {}
    markdown = _normalize_text(table_payload.get("markdown"))
    if markdown:
        return markdown
    html_excerpt = _normalize_text(element.get("html"))
    if html_excerpt:
        return html_excerpt
    return _normalize_text(element.get("text"))


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
    semantic_split_applied: bool,
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
    metadata["semantic_split_applied"] = semantic_split_applied
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
    group_type: str,
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


def _semantic_split_segments(
    segments: list[TextSegment],
    *,
    embedding_client: SemanticEmbeddingClient,
    target_tokens: int,
    max_tokens: int,
    min_tokens: int,
    similarity_threshold: float,
) -> list[list[TextSegment]] | None:
    """인접 segment 유사도가 낮아지는 지점을 우선 경계로 사용하는 split."""
    if len(segments) < 2:
        return None

    embeddings = embedding_client.embed_texts(segment.text for segment in segments)
    if embeddings is None:
        return None

    similarities = [
        cosine_similarity(left, right)
        for left, right in zip(embeddings, embeddings[1:])
    ]

    parts: list[list[TextSegment]] = []
    current: list[TextSegment] = []
    current_tokens = 0

    for index, segment in enumerate(segments):
        segment_tokens = _estimate_tokens(segment.text)
        should_split = False

        if current:
            previous_similarity = similarities[index - 1]
            projected_tokens = current_tokens + segment_tokens
            if current_tokens >= min_tokens and projected_tokens > max_tokens:
                should_split = True
            elif (
                current_tokens >= target_tokens
                and previous_similarity < similarity_threshold
            ):
                should_split = True
            elif (
                current_tokens >= min_tokens
                and projected_tokens > target_tokens
                and previous_similarity < similarity_threshold
            ):
                should_split = True

        if should_split:
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
    *,
    embedding_client: SemanticEmbeddingClient,
) -> list[ChunkDraft]:
    """긴 prose chunk를 semantic split 또는 hard split으로 분할한다."""
    estimated_tokens = _estimate_tokens(draft.base_text)
    if estimated_tokens <= STAGE3_TEXT_MAX_TOKENS:
        draft.metadata["estimated_tokens"] = estimated_tokens
        draft.metadata["semantic_split_applied"] = False
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
        return [draft]

    split_parts: list[list[TextSegment]] | None = None
    semantic_applied = False
    if draft.semantic_eligible and STAGE3_ENABLE_SEMANTIC_SPLIT:
        split_parts = _semantic_split_segments(
            segments,
            embedding_client=embedding_client,
            target_tokens=STAGE3_TEXT_TARGET_TOKENS,
            max_tokens=STAGE3_TEXT_MAX_TOKENS,
            min_tokens=STAGE3_TEXT_MIN_TOKENS,
            similarity_threshold=STAGE3_SEMANTIC_SPLIT_SIM_THRESHOLD,
        )
        semantic_applied = split_parts is not None

    if split_parts is None:
        split_parts = _hard_split_segments(
            segments,
            group_type=str(draft.metadata.get("group_type") or "prose"),
            target_tokens=STAGE3_TEXT_TARGET_TOKENS,
            max_tokens=STAGE3_TEXT_MAX_TOKENS,
            min_tokens=STAGE3_TEXT_MIN_TOKENS,
        )

    return [
        _build_text_chunk_from_segments(
            parent=draft,
            segments=part,
            semantic_split_applied=semantic_applied,
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
                "semantic_split_applied": False,
                "semantic_merge_applied": False,
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


def _apply_semantic_split(
    drafts: list[ChunkDraft],
    *,
    embedding_client: SemanticEmbeddingClient,
) -> list[ChunkDraft]:
    split_drafts: list[ChunkDraft] = []
    for draft in drafts:
        if draft.chunk_type != "text":
            split_drafts.append(draft)
            continue
        split_drafts.extend(
            _split_text_draft(
                draft,
                embedding_client=embedding_client,
            )
        )
    return split_drafts


def _merge_text_chunks(
    left: ChunkDraft,
    right: ChunkDraft,
    *,
    semantic_merge_applied: bool,
) -> ChunkDraft:
    merged_segments = left.segments + right.segments
    parent = ChunkDraft(
        chunk_type="text",
        heading_path=list(left.heading_path),
        base_text="",
        pages=[],
        element_ids=[],
        source_elements=[],
        metadata={
            "group_type": left.metadata.get("group_type") or "prose",
            "semantic_split_applied": bool(
                left.metadata.get("semantic_split_applied")
                or right.metadata.get("semantic_split_applied")
            ),
            "semantic_merge_applied": semantic_merge_applied,
        },
        order_key=left.order_key,
        semantic_eligible=left.semantic_eligible and right.semantic_eligible,
        segments=merged_segments,
    )
    merged = _build_text_chunk_from_segments(
        parent=parent,
        segments=merged_segments,
        semantic_split_applied=bool(parent.metadata["semantic_split_applied"]),
    )
    merged.metadata["semantic_merge_applied"] = semantic_merge_applied
    return merged


def _apply_semantic_merge(
    drafts: list[ChunkDraft],
    *,
    embedding_client: SemanticEmbeddingClient,
) -> list[ChunkDraft]:
    if not STAGE3_ENABLE_SEMANTIC_MERGE:
        return drafts

    merged_drafts: list[ChunkDraft] = []
    index = 0
    while index < len(drafts):
        current = drafts[index]
        if (
            index + 1 >= len(drafts)
            or current.chunk_type != "text"
            or str(current.metadata.get("group_type")) != "prose"
        ):
            merged_drafts.append(current)
            index += 1
            continue

        next_chunk = drafts[index + 1]
        same_heading = current.heading_path == next_chunk.heading_path
        same_group = (
            next_chunk.chunk_type == "text"
            and str(next_chunk.metadata.get("group_type")) == "prose"
        )
        current_tokens = _estimate_tokens(current.base_text)
        next_tokens = _estimate_tokens(next_chunk.base_text)
        combined_tokens = current_tokens + next_tokens
        candidate_small = (
            current_tokens <= STAGE3_SEMANTIC_MERGE_CANDIDATE_MAX_TOKENS
            or next_tokens <= STAGE3_SEMANTIC_MERGE_CANDIDATE_MAX_TOKENS
        )

        if not (
            same_heading
            and same_group
            and candidate_small
            and combined_tokens <= STAGE3_TEXT_MAX_TOKENS
        ):
            merged_drafts.append(current)
            index += 1
            continue

        embeddings = embedding_client.embed_texts([current.base_text, next_chunk.base_text])
        if embeddings is None:
            merged_drafts.append(current)
            index += 1
            continue

        similarity = cosine_similarity(embeddings[0], embeddings[1])
        if similarity < STAGE3_SEMANTIC_MERGE_SIM_THRESHOLD:
            merged_drafts.append(current)
            index += 1
            continue

        current = _merge_text_chunks(
            current,
            next_chunk,
            semantic_merge_applied=True,
        )
        drafts[index + 1] = current
        index += 1

    return merged_drafts


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
            if previous_chunk is not None:
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
    *,
    cleaned_json_path: Path,
    output_paths: Stage3OutputPaths,
) -> None:
    json_path = Path(output_paths["chunks_json"])
    jsonl_path = Path(output_paths["chunks_jsonl"])
    markdown_path = Path(output_paths["chunks_md"])
    json_path.parent.mkdir(parents=True, exist_ok=True)

    stats = _build_stats(chunks)
    json_path.write_text(
        json.dumps(
            {
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
        else cleaned_json_path.parent.resolve()
    )
    output_paths = build_stage3_output_paths(
        cleaned_json_path=cleaned_json_path,
        output_dir=output_dir,
    )

    cleaned_document = _load_cleaned_document(cleaned_json_path)
    elements = list(cleaned_document.get("elements") or [])

    semantic_client = embedding_client or SemanticEmbeddingClient()

    initial_drafts = _build_initial_chunk_drafts(elements)
    split_drafts = _apply_semantic_split(
        initial_drafts,
        embedding_client=semantic_client,
    )
    final_drafts = _apply_semantic_merge(
        split_drafts,
        embedding_client=semantic_client,
    )
    chunks = _finalize_chunk_payloads(final_drafts)
    _write_chunks(
        chunks,
        cleaned_json_path=cleaned_json_path,
        output_paths=output_paths,
    )

    status = (
        "completed_with_semantic_fallback"
        if semantic_client.last_error
        else "completed"
    )
    stats = _build_stats(chunks)
    return {
        "cleaned_json_path": str(cleaned_json_path),
        "output_dir": str(output_dir),
        "output_paths": output_paths,
        "planned_outputs": output_paths,
        "chunk_count": len(chunks),
        "stats": stats,
        "semantic_enabled": semantic_client.enabled,
        "semantic_fallback_reason": semantic_client.last_error,
        "status": status,
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
