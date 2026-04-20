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
    stem = Path(str(original_filename or "").strip()).stem.strip()
    if not stem.isdigit():
        return []
    return [
        f"{stem}번 문서",
        f"{stem}번",
        f"{stem}.pdf",
        f"{stem} pdf",
    ]


def extract_explicit_document_ids(
    query_text: str,
    ordered_profiles: list[dict[str, object]],
) -> list[str]:
    normalized_query = normalize_match_text(query_text)
    selected_document_ids: list[str] = []
    for profile in ordered_profiles:
        document_id = str(profile.get("document_id") or "").strip()
        if not document_id:
            continue

        original_filename = str(profile.get("original_filename") or "").strip()
        filename_stem = Path(original_filename).stem.strip()
        title = str(profile.get("title") or "").strip()
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
        if not explicit_match and original_filename:
            for alias in extract_numeric_filename_aliases(original_filename):
                if re.search(re.escape(normalize_match_text(alias)), normalized_query):
                    explicit_match = True
                    break

        if explicit_match and document_id not in selected_document_ids:
            selected_document_ids.append(document_id)

    return selected_document_ids
