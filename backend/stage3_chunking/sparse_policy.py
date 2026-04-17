"""Stage-3 sparse BM25 branch에 태울 청크를 선별하는 정책 모듈."""

from __future__ import annotations

import re
from typing import Any, Iterable


SPARSE_MIN_TOKENS = 6
SPARSE_MIN_TOKENS_WITHOUT_SECTION = 18
SPARSE_MIN_SENTENCE_RATIO_WITHOUT_SECTION = 0.2
SPARSE_EARLY_PAGE_META_SENTENCE_RATIO = 0.34
SPARSE_EARLY_PAGE_META_MAX_LINE_TOKENS = 8
SPARSE_PREVIEW_MAX_SENTENCES = 2
SPARSE_PREVIEW_MAX_TOKENS = 96
VISUAL_CHUNK_TYPES = {"table", "figure"}


def _normalize_text(text: str | None) -> str:
    if not text:
        return ""
    normalized = str(text).replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _normalize_hint_text(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", str(text).strip().lower())


def _estimate_tokens(text: str) -> int:
    normalized = _normalize_text(text)
    if not normalized:
        return 0
    word_units = len(re.findall(r"\S+", normalized))
    char_units = len(re.sub(r"\s+", "", normalized))
    return max(word_units, max(1, char_units // 4))


def _dedupe_sparse_parts(parts: Iterable[str | None]) -> list[str]:
    """sparse_text를 만들 때 중복되는 lexical anchor를 한 번만 남긴다."""
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        normalized = _normalize_text(part)
        if not normalized:
            continue
        key = _normalize_hint_text(normalized)
        if key in seen:
            continue
        seen.add(key)
        ordered.append(normalized)
    return ordered


def _build_sparse_preview_text(
    text: str,
    *,
    max_sentences: int = SPARSE_PREVIEW_MAX_SENTENCES,
    max_tokens: int = SPARSE_PREVIEW_MAX_TOKENS,
) -> str:
    """긴 본문 전체 대신 sparse 검색에 필요한 앞부분 lexical anchor만 남긴다."""
    normalized = _normalize_text(text)
    if not normalized:
        return ""

    candidates = [
        _normalize_text(part)
        for part in re.split(r"(?<=[.!?。！？])\s+|\n{2,}", normalized)
        if _normalize_text(part)
    ]
    if not candidates:
        candidates = [
            _normalize_text(line)
            for line in normalized.splitlines()
            if _normalize_text(line)
        ]
    if not candidates:
        return normalized

    selected: list[str] = []
    token_count = 0
    for candidate in candidates:
        candidate_tokens = _estimate_tokens(candidate)
        if selected and (
            len(selected) >= max_sentences
            or token_count + candidate_tokens > max_tokens
        ):
            break
        selected.append(candidate)
        token_count += candidate_tokens

    preview = "\n".join(selected).strip()
    return preview or normalized


def _build_sparse_text(
    *,
    chunk_type: str,
    body_text: str,
    section_title: str | None,
    caption: str | None,
    summary_text: str | None,
) -> str:
    """dense 본문과 분리된 sparse 전용 lexical anchor 텍스트를 만든다."""
    if chunk_type in VISUAL_CHUNK_TYPES:
        return "\n\n".join(
            _dedupe_sparse_parts([section_title, caption, summary_text])
        )

    preview = _build_sparse_preview_text(body_text)
    return "\n\n".join(_dedupe_sparse_parts([section_title, preview]))


def determine_sparse_policy(
    *,
    chunk_type: str,
    body_text: str,
    section_title: str | None,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """실무형 보수 정책으로 BM25 branch에 태울 청크만 선별한다."""
    group_type = str(metadata.get("group_type") or "")
    estimated_tokens = int(
        metadata.get("estimated_tokens") or _estimate_tokens(body_text)
    )
    sentence_like_ratio = float(metadata.get("sentence_like_ratio") or 0.0)
    line_count = int(metadata.get("line_count") or 0)
    average_line_tokens = float(
        metadata.get("average_line_tokens") or float(estimated_tokens)
    )
    sparse_role_hints = [
        str(item).strip()
        for item in metadata.get("sparse_role_hints") or []
        if str(item).strip()
    ]
    sparse_text = _build_sparse_text(
        chunk_type=chunk_type,
        body_text=body_text,
        section_title=section_title,
        caption=str(metadata.get("caption") or ""),
        summary_text=str(metadata.get("summary_text") or ""),
    )

    if chunk_type in VISUAL_CHUNK_TYPES:
        keep = bool(
            sparse_text
            and (
                _normalize_text(metadata.get("caption"))
                or _normalize_text(metadata.get("summary_text"))
            )
        )
        return {
            "sparse_keep": keep,
            "sparse_text": sparse_text if keep else "",
            "sparse_exclude_reason": None if keep else "missing_visual_anchor",
        }

    if sparse_role_hints:
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "role_hint_filtered",
        }

    if bool(metadata.get("has_email")) or bool(metadata.get("has_url")):
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "contact_or_url_metadata",
        }

    if group_type != "prose":
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "non_prose_text",
        }

    if estimated_tokens < SPARSE_MIN_TOKENS:
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "too_short_for_sparse",
        }

    if estimated_tokens < SPARSE_MIN_TOKENS_WITHOUT_SECTION and not section_title:
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "short_text_without_section_anchor",
        }

    if (
        bool(metadata.get("early_page_hint"))
        and line_count >= 3
        and average_line_tokens <= SPARSE_EARLY_PAGE_META_MAX_LINE_TOKENS
        and sentence_like_ratio < SPARSE_EARLY_PAGE_META_SENTENCE_RATIO
    ):
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "early_page_meta_like",
        }

    if (
        sentence_like_ratio < SPARSE_MIN_SENTENCE_RATIO_WITHOUT_SECTION
        and not section_title
    ):
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "low_sentence_density",
        }

    if not sparse_text:
        return {
            "sparse_keep": False,
            "sparse_text": "",
            "sparse_exclude_reason": "empty_sparse_text",
        }

    return {
        "sparse_keep": True,
        "sparse_text": sparse_text,
        "sparse_exclude_reason": None,
    }
