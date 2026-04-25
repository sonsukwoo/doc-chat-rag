"""Shared document selection helpers for stage5."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


def normalize_match_text(value: str) -> str:
    normalized = str(value or "").strip().lower()
    normalized = re.sub(r"[^0-9a-z가-힣\s]", " ", normalized)
    return " ".join(normalized.split())


def tokenize_match_terms(value: str) -> set[str]:
    normalized = normalize_match_text(value)
    return {token for token in normalized.split() if len(token) >= 2}


def _ordered_match_terms(value: str) -> list[str]:
    normalized = normalize_match_text(value)
    return [token for token in normalized.split() if len(token) >= 2]


def iter_ordered_document_profiles(
    active_document_ids: list[str] | None,
    raw_profiles: list[dict[str, Any]] | None,
) -> list[dict[str, object]]:
    normalized_document_ids = [
        str(item).strip()
        for item in active_document_ids or []
        if str(item).strip()
    ]
    profile_by_id: dict[str, dict[str, object]] = {}
    for raw_profile in raw_profiles or []:
        if not isinstance(raw_profile, dict):
            continue
        document_id = str(raw_profile.get("document_id") or "").strip()
        if not document_id:
            continue
        profile_by_id[document_id] = dict(raw_profile)

    ordered_profiles: list[dict[str, object]] = []
    for index, document_id in enumerate(normalized_document_ids, start=1):
        profile = dict(profile_by_id.get(document_id) or {})
        profile["document_id"] = document_id
        profile["document_order"] = index
        ordered_profiles.append(profile)
    return ordered_profiles


def extract_numeric_filename_aliases(original_filename: str) -> list[str]:
    filename = Path(str(original_filename or "").strip()).name.strip()
    stem = Path(filename).stem.strip()
    if not stem.isdigit():
        return []
    aliases = [filename] if filename else []
    aliases.append(f"{stem} pdf")
    return list(dict.fromkeys(aliases))


def _has_meaningful_phrase_overlap(
    query_terms: list[str],
    value: str,
) -> bool:
    candidate_terms = _ordered_match_terms(value)
    if len(candidate_terms) < 2:
        return False
    matches: list[tuple[int, int, str]] = []
    for candidate_index, candidate_term in enumerate(candidate_terms):
        for query_index, query_term in enumerate(query_terms):
            if (
                candidate_term == query_term
                or candidate_term in query_term
                or query_term in candidate_term
            ):
                matches.append((candidate_index, query_index, candidate_term))

    for left_index, (left_candidate_index, left_query_index, left_term) in enumerate(
        matches
    ):
        for right_candidate_index, right_query_index, right_term in matches[
            left_index + 1 :
        ]:
            if right_candidate_index <= left_candidate_index:
                continue
            if right_query_index <= left_query_index:
                continue
            if right_query_index - left_query_index > 6:
                continue
            if len(left_term) + len(right_term) >= 4:
                return True
    return False


def extract_explicit_document_ids(
    query_text: str,
    ordered_profiles: list[dict[str, object]],
) -> list[str]:
    normalized_query = normalize_match_text(query_text)
    query_terms = _ordered_match_terms(query_text)
    selected_document_ids: list[str] = []
    for profile in ordered_profiles:
        document_id = str(profile.get("document_id") or "").strip()
        if not document_id:
            continue

        original_filename = str(profile.get("original_filename") or "").strip()
        filename_stem = Path(original_filename).stem.strip()
        title = str(profile.get("title") or "").strip()
        document_type = str(profile.get("document_type") or "").strip()
        candidate_aliases = [
            normalize_match_text(document_id),
            normalize_match_text(original_filename),
            normalize_match_text(title),
        ]
        if filename_stem and not filename_stem.isdigit():
            candidate_aliases.append(normalize_match_text(filename_stem))

        explicit_match = any(
            candidate_alias and candidate_alias in normalized_query
            for candidate_alias in candidate_aliases
        )
        if not explicit_match:
            for alias in extract_numeric_filename_aliases(original_filename):
                if re.search(re.escape(normalize_match_text(alias)), normalized_query):
                    explicit_match = True
                    break
        if not explicit_match:
            explicit_match = any(
                _has_meaningful_phrase_overlap(query_terms, value)
                for value in (title, document_type, filename_stem)
            )

        if explicit_match and document_id not in selected_document_ids:
            selected_document_ids.append(document_id)

    return selected_document_ids
