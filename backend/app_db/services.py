"""Application Postgres service layer."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TypedDict

from backend.thread_identity import resolve_thread_collection_name

from .repositories import ChatRepository, DocumentRepository, app_db_connection


class ThreadRuntimeContext(TypedDict, total=False):
    """thread 기반 검색/대화에 필요한 최소 런타임 컨텍스트."""

    thread_id: str
    thread_name: str
    collection_name: str
    default_retrieval_mode: str
    active_document_ids: list[str]
    document_profiles: list["DocumentRuntimeProfilePayload"]


class DocumentRuntimeProfilePayload(TypedDict, total=False):
    """stage5가 문서 관련성 판단에 쓰는 최소 문서 프로파일."""

    document_id: str
    original_filename: str
    title: str
    document_type: str
    main_topics: list[str]
    keywords: list[str]
    section_titles: list[str]
    short_summary: str


class ExpandedContextBlockPayload(TypedDict, total=False):
    """child chunk를 상위 parent 문맥으로 확장한 결과."""

    document_id: str
    parent_id: str
    section_title: str | None
    page_start: int | None
    page_end: int | None
    heading_path: list[str]
    matched_chunk_ids: list[str]
    window_chunk_ids: list[str]
    context_text: str
    expansion_mode: str


class VisualAssetPayload(TypedDict, total=False):
    """표/이미지 원본 asset 메타데이터."""

    asset_ref: str
    document_id: str
    chunk_id: str
    asset_kind: str
    relative_path: str
    asset_stage: str
    page: int | None
    caption: str | None
    summary_text: str | None
    heading_path: list[str]
    pages: list[int]


def _normalize_document_ids(active_document_ids: list[str] | None) -> list[str]:
    return [
        str(item).strip()
        for item in active_document_ids or []
        if str(item).strip()
    ]


def _normalize_string_list(values: list[Any] | None, *, limit: int) -> list[str]:
    return [
        str(item).strip()
        for item in values or []
        if str(item).strip()
    ][:limit]


def _extract_section_titles(elements: list[dict[str, Any]] | None) -> list[str]:
    section_titles: list[str] = []
    for element in elements or []:
        if not isinstance(element, dict):
            continue
        category = str(element.get("category") or "").strip().lower()
        if category != "heading":
            continue
        text = str(element.get("text") or "").strip()
        if text and text not in section_titles:
            section_titles.append(text)
        if len(section_titles) >= 8:
            break
    return section_titles


def _build_document_profile_summary(raw_profile: dict[str, Any]) -> str:
    title = str(raw_profile.get("title") or "").strip()
    document_type = str(raw_profile.get("document_type") or "").strip()
    main_topics = [
        str(item).strip()
        for item in raw_profile.get("main_topics") or []
        if str(item).strip()
    ]
    return " / ".join(
        part
        for part in (
            title,
            document_type,
            ", ".join(main_topics[:3]) if main_topics else "",
        )
        if part
    )


def _derive_document_profile_keywords(
    raw_profile: dict[str, Any],
    *,
    section_titles: list[str],
) -> list[str]:
    keywords = _normalize_string_list(
        list(raw_profile.get("keywords") or []),
        limit=8,
    )
    if keywords:
        return keywords

    derived_keywords: list[str] = []
    seen: set[str] = set()
    candidates = [
        raw_profile.get("title"),
        *(raw_profile.get("main_topics") or []),
        *section_titles,
    ]
    for candidate in candidates:
        keyword = str(candidate or "").strip()
        normalized_keyword = keyword.casefold()
        if not keyword or normalized_keyword in seen:
            continue
        seen.add(normalized_keyword)
        derived_keywords.append(keyword)
        if len(derived_keywords) >= 8:
            break
    return derived_keywords


def _prepare_document_profile_source(
    *,
    raw_profile: dict[str, Any] | None,
    elements: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    prepared_profile = dict(raw_profile or {})
    section_titles = _normalize_string_list(
        list(prepared_profile.get("section_titles") or []),
        limit=8,
    )
    if not section_titles:
        section_titles = _extract_section_titles(elements)
    if section_titles:
        prepared_profile["section_titles"] = section_titles
    keywords = _derive_document_profile_keywords(
        prepared_profile,
        section_titles=section_titles,
    )
    if keywords:
        prepared_profile["keywords"] = keywords
    return prepared_profile


def _normalize_document_profile(
    *,
    document_id: str,
    original_filename: str,
    raw_profile: dict[str, Any] | None,
    short_summary: str | None = None,
) -> DocumentRuntimeProfilePayload:
    normalized_profile = dict(raw_profile or {})
    title = (
        str(normalized_profile.get("title") or "").strip()
        or original_filename
        or document_id
    )
    document_type = str(normalized_profile.get("document_type") or "").strip() or "문서"
    main_topics = _normalize_string_list(
        list(normalized_profile.get("main_topics") or []),
        limit=6,
    )
    keywords = _normalize_string_list(
        list(normalized_profile.get("keywords") or []),
        limit=8,
    )
    section_titles = _normalize_string_list(
        list(normalized_profile.get("section_titles") or []),
        limit=8,
    )
    resolved_summary = str(short_summary or "").strip() or _build_document_profile_summary(
        normalized_profile
    ) or " / ".join(
        part
        for part in (
            title,
            document_type,
            ", ".join(main_topics[:3]) if main_topics else "",
        )
        if part
    )
    return {
        "document_id": document_id,
        "original_filename": original_filename,
        "title": title,
        "document_type": document_type,
        "main_topics": main_topics,
        "keywords": keywords,
        "section_titles": section_titles,
        "short_summary": resolved_summary,
    }


def _is_placeholder_document_profile(
    *,
    document_id: str,
    original_filename: str,
    raw_profile: dict[str, Any] | None,
    short_summary: str | None = None,
) -> bool:
    runtime_profile = _normalize_document_profile(
        document_id=document_id,
        original_filename=original_filename,
        raw_profile=raw_profile,
        short_summary=short_summary,
    )
    if (
        runtime_profile.get("main_topics")
        or runtime_profile.get("keywords")
        or runtime_profile.get("section_titles")
    ):
        return False

    title = str(runtime_profile.get("title") or "").strip()
    document_type = str(runtime_profile.get("document_type") or "").strip() or "문서"
    summary = str(runtime_profile.get("short_summary") or "").strip()
    trivial_titles = {
        str(document_id or "").strip(),
        str(original_filename or "").strip(),
    }
    trivial_summaries = {
        "",
        f"{document_id} / 문서",
        f"{original_filename} / 문서",
        f"{title} / 문서",
    }
    return title in trivial_titles and document_type == "문서" and summary in trivial_summaries


def _read_profile_payload(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _recover_document_profile_source(
    document_row: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]] | None, str | None, str | None]:
    metadata = dict(document_row.get("metadata") or {})
    metadata_profile = dict(metadata.get("document_profile") or {})
    metadata_summary = str(metadata.get("document_profile_summary") or "").strip() or None

    storage_root = str(document_row.get("storage_root") or "").strip()
    artifact_profile: dict[str, Any] = {}
    artifact_elements: list[dict[str, Any]] | None = None
    artifact_source_stage: str | None = None
    if storage_root:
        artifact_candidates = (
            ("review", Path(storage_root) / "review" / "reviewed_cleaned.json"),
            ("stage2", Path(storage_root) / "stage2" / "cleaned.json"),
        )
        for source_stage, path in artifact_candidates:
            payload = _read_profile_payload(path)
            if payload is None:
                continue
            artifact_profile = dict(payload.get("document_profile") or {})
            artifact_elements = list(payload.get("elements") or [])
            artifact_source_stage = source_stage
            if artifact_profile or artifact_elements:
                break

    recovered_profile = dict(artifact_profile)
    recovered_profile.update(
        {
            key: value
            for key, value in metadata_profile.items()
            if value not in (None, "", [], {})
        }
    )
    if recovered_profile or artifact_elements:
        return (
            recovered_profile,
            artifact_elements,
            metadata_summary,
            artifact_source_stage or "metadata_recovery",
        )
    return {}, None, metadata_summary, None


def _upsert_document_profile_snapshot(
    *,
    document_repository: DocumentRepository,
    document_id: str,
    original_filename: str,
    raw_profile: dict[str, Any] | None,
    elements: list[dict[str, Any]] | None = None,
    source_stage: str,
    short_summary: str | None = None,
) -> DocumentRuntimeProfilePayload:
    prepared_profile = _prepare_document_profile_source(
        raw_profile=raw_profile,
        elements=elements,
    )
    runtime_profile = _normalize_document_profile(
        document_id=document_id,
        original_filename=original_filename,
        raw_profile=prepared_profile,
        short_summary=short_summary,
    )
    document_repository.upsert_document_profile(
        document_id=document_id,
        title=runtime_profile["title"],
        document_type=runtime_profile["document_type"],
        main_topics=list(runtime_profile.get("main_topics") or []),
        keywords=list(runtime_profile.get("keywords") or []),
        section_titles=list(runtime_profile.get("section_titles") or []),
        short_summary=str(runtime_profile.get("short_summary") or ""),
        profile_json=prepared_profile,
        source_stage=source_stage,
    )
    return runtime_profile


def _split_qualified_ref(value: str) -> tuple[str | None, str]:
    normalized = str(value or "").strip()
    if not normalized:
        return (None, "")
    if ":" not in normalized:
        return (None, normalized)
    document_id, chunk_or_asset_id = normalized.split(":", 1)
    return (document_id.strip() or None, chunk_or_asset_id.strip())


def _match_chunk_ref(
    *,
    document_id: str,
    chunk_id: str,
    requested_refs: list[str],
) -> bool:
    for requested_ref in requested_refs:
        requested_document_id, requested_chunk_id = _split_qualified_ref(requested_ref)
        if requested_chunk_id != chunk_id:
            continue
        if requested_document_id is None or requested_document_id == document_id:
            return True
    return False


def _slice_window_chunk_ids(
    *,
    child_chunk_ids: list[str],
    matched_chunk_ids: list[str],
    window_size: int,
) -> list[str]:
    if not child_chunk_ids or not matched_chunk_ids:
        return []
    matched_positions = [
        index
        for index, child_chunk_id in enumerate(child_chunk_ids)
        if child_chunk_id in matched_chunk_ids
    ]
    if not matched_positions:
        return []
    start = max(0, min(matched_positions) - max(0, window_size))
    end = min(len(child_chunk_ids), max(matched_positions) + max(0, window_size) + 1)
    return child_chunk_ids[start:end]


def load_thread_runtime_context(thread_id: str) -> ThreadRuntimeContext | None:
    """thread 메타데이터와 현재 연결된 문서 ID 목록을 함께 읽는다."""
    normalized_thread_id = str(thread_id or "").strip()
    if not normalized_thread_id:
        return None

    with app_db_connection() as connection:
        chat_repository = ChatRepository(connection)
        document_repository = DocumentRepository(connection)
        thread = chat_repository.get_thread(normalized_thread_id)
        if thread is None:
            return None
        active_document_ids = document_repository.list_active_document_ids(
            normalized_thread_id
        )
        raw_document_rows = document_repository.list_documents(active_document_ids)
        profile_rows = document_repository.list_document_profiles(active_document_ids)
        profile_by_document_id = {
            str(row.get("document_id") or "").strip(): dict(row)
            for row in profile_rows
            if str(row.get("document_id") or "").strip()
        }
        thread_metadata = dict(thread.get("metadata") or {})
        document_profiles: list[DocumentRuntimeProfilePayload] = []
        for row in raw_document_rows:
            document_id = str(row.get("document_id") or "").strip()
            if not document_id:
                continue

            original_filename = str(row.get("original_filename") or "").strip()
            profile_row = profile_by_document_id.get(document_id)
            raw_profile = (
                dict(profile_row.get("profile_json") or {})
                if profile_row
                else {}
            )
            fallback_profile = (
                {
                    "title": profile_row.get("title"),
                    "document_type": profile_row.get("document_type"),
                    "main_topics": list(profile_row.get("main_topics") or []),
                    "keywords": list(profile_row.get("keywords") or []),
                    "section_titles": list(profile_row.get("section_titles") or []),
                }
                if profile_row
                else {}
            )
            if not raw_profile:
                raw_profile = fallback_profile
            short_summary = (
                str(profile_row.get("short_summary") or "").strip()
                if profile_row
                else None
            )

            if not raw_profile or _is_placeholder_document_profile(
                document_id=document_id,
                original_filename=original_filename,
                raw_profile=raw_profile,
                short_summary=short_summary,
            ):
                recovered_profile, recovered_elements, recovered_summary, recovered_stage = (
                    _recover_document_profile_source(row)
                )
                if recovered_profile or recovered_elements:
                    document_profiles.append(
                        _upsert_document_profile_snapshot(
                            document_repository=document_repository,
                            document_id=document_id,
                            original_filename=original_filename,
                            raw_profile=recovered_profile or raw_profile,
                            elements=recovered_elements,
                            source_stage=recovered_stage or "metadata_recovery",
                            short_summary=recovered_summary or short_summary,
                        )
                    )
                    continue

            document_profiles.append(
                _normalize_document_profile(
                    document_id=document_id,
                    original_filename=original_filename,
                    raw_profile=raw_profile,
                    short_summary=short_summary,
                )
            )
        return {
            "thread_id": normalized_thread_id,
            "thread_name": str(thread.get("thread_name") or normalized_thread_id),
            "collection_name": resolve_thread_collection_name(
                normalized_thread_id,
                metadata=thread_metadata,
            ),
            "default_retrieval_mode": str(
                thread.get("default_retrieval_mode") or "dense"
            ).strip()
            or "dense",
            "active_document_ids": active_document_ids,
            "document_profiles": document_profiles,
        }


def try_load_thread_runtime_context(thread_id: str) -> ThreadRuntimeContext | None:
    """DB 연결 실패까지 포함해 안전하게 thread 컨텍스트를 읽는다."""
    try:
        return load_thread_runtime_context(thread_id)
    except Exception:
        return None


def sync_document_runtime_metadata(
    *,
    thread_id: str,
    document_id: str,
    original_filename: str,
    normalized_filename: str,
    storage_root: str | Path,
    source_pdf_path: str | None,
    parents: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
    document_profile: dict[str, Any] | None = None,
    document_profile_elements: list[dict[str, Any]] | None = None,
    document_profile_source_stage: str = "stage3",
) -> None:
    """stage3 이후 document/thread/profile/parent/asset 메타데이터를 Postgres와 동기화한다."""
    normalized_storage_root = str(Path(storage_root).expanduser().resolve())
    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        existing_document = document_repository.get_document(document_id)
        merged_metadata = dict((existing_document or {}).get("metadata") or {})
        merged_metadata.update(dict(metadata or {}))
        document_repository.upsert_document(
            document_id=document_id,
            original_filename=original_filename,
            normalized_filename=normalized_filename,
            storage_root=normalized_storage_root,
            source_pdf_path=source_pdf_path,
            source_kind=str((existing_document or {}).get("source_kind") or "upload"),
            lifecycle_status=str(
                (existing_document or {}).get("lifecycle_status") or "active"
            ),
            file_hash=str((existing_document or {}).get("file_hash") or "").strip() or None,
            metadata=merged_metadata,
        )
        if document_profile is not None or document_profile_elements is not None:
            _upsert_document_profile_snapshot(
                document_repository=document_repository,
                document_id=document_id,
                original_filename=original_filename,
                raw_profile=document_profile,
                elements=document_profile_elements,
                source_stage=document_profile_source_stage,
            )
        document_repository.attach_document_to_thread(
            thread_id=thread_id,
            document_id=document_id,
            slot_key=normalized_filename,
        )
        document_repository.replace_document_chunks(
            document_id=document_id,
            chunks=chunks,
        )
        document_repository.replace_document_parents(
            document_id=document_id,
            parents=parents,
        )
        document_repository.replace_document_assets(
            document_id=document_id,
            chunks=chunks,
        )


def sync_document_profile_snapshot(
    *,
    document_id: str,
    original_filename: str,
    normalized_filename: str,
    storage_root: str | Path,
    source_pdf_path: str | None,
    raw_profile: dict[str, Any] | None,
    elements: list[dict[str, Any]] | None = None,
    source_stage: str = "stage2",
) -> DocumentRuntimeProfilePayload:
    """문서 프로파일 아티팩트를 Postgres source-of-truth 테이블에 동기화한다."""
    normalized_storage_root = str(Path(storage_root).expanduser().resolve())
    resolved_original_filename = str(original_filename or "").strip() or f"{document_id}.pdf"
    resolved_normalized_filename = (
        str(normalized_filename or "").strip() or resolved_original_filename
    )
    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        existing_document = document_repository.get_document(document_id)
        metadata = dict((existing_document or {}).get("metadata") or {})
        document_repository.upsert_document(
            document_id=document_id,
            original_filename=resolved_original_filename,
            normalized_filename=resolved_normalized_filename,
            storage_root=normalized_storage_root,
            source_pdf_path=source_pdf_path,
            source_kind=str((existing_document or {}).get("source_kind") or "upload"),
            lifecycle_status=str(
                (existing_document or {}).get("lifecycle_status") or "active"
            ),
            file_hash=str((existing_document or {}).get("file_hash") or "").strip() or None,
            metadata=metadata,
        )
        return _upsert_document_profile_snapshot(
            document_repository=document_repository,
            document_id=document_id,
            original_filename=resolved_original_filename,
            raw_profile=raw_profile,
            elements=elements,
            source_stage=source_stage,
        )


def load_expanded_context_blocks(
    *,
    thread_id: str | None,
    active_document_ids: list[str],
    chunk_ids: list[str],
    window_size: int = 1,
) -> list[ExpandedContextBlockPayload]:
    """현재 thread의 child chunk ids를 parent 문맥 블록으로 확장한다."""
    del thread_id
    normalized_document_ids = _normalize_document_ids(active_document_ids)
    normalized_chunk_refs = [str(item).strip() for item in chunk_ids if str(item).strip()]
    if not normalized_document_ids or not normalized_chunk_refs:
        return []

    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        parent_rows = document_repository.list_document_parents(normalized_document_ids)
        chunk_rows = document_repository.list_document_chunks(normalized_document_ids)

    chunk_text_lookup: dict[tuple[str, str], str] = {}
    chunk_ids_by_parent: dict[tuple[str, str], list[str]] = {}
    for row in chunk_rows:
        document_id = str(row.get("document_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        parent_id = str(row.get("parent_id") or "").strip()
        if not document_id or not chunk_id:
            continue
        chunk_text_lookup[(document_id, chunk_id)] = str(row.get("text") or "").strip()
        if parent_id:
            chunk_ids_by_parent.setdefault((document_id, parent_id), []).append(chunk_id)

    context_blocks: list[ExpandedContextBlockPayload] = []
    for row in parent_rows:
        document_id = str(row.get("document_id") or "").strip()
        parent_id = str(row.get("parent_id") or "").strip()
        child_chunk_ids = [
            str(item).strip()
            for item in row.get("chunk_ids") or []
            if str(item).strip()
        ]
        matched_chunk_ids = [
            child_chunk_id
            for child_chunk_id in child_chunk_ids
            if _match_chunk_ref(
                document_id=document_id,
                chunk_id=child_chunk_id,
                requested_refs=normalized_chunk_refs,
            )
        ]
        if not matched_chunk_ids:
            continue

        stored_chunk_ids = chunk_ids_by_parent.get((document_id, parent_id), [])
        ordered_chunk_ids = stored_chunk_ids or child_chunk_ids
        window_chunk_ids = _slice_window_chunk_ids(
            child_chunk_ids=ordered_chunk_ids,
            matched_chunk_ids=matched_chunk_ids,
            window_size=window_size,
        )
        selected_texts = [
            chunk_text_lookup.get((document_id, selected_chunk_id), "").strip()
            for selected_chunk_id in window_chunk_ids
        ]
        context_text = "\n\n".join(text for text in selected_texts if text).strip()
        if not context_text:
            context_text = str(row.get("body_text") or "").strip()
        expansion_mode = (
            "postgres_window"
            if len(window_chunk_ids) > 1
            else "postgres_child"
        )
        if not window_chunk_ids:
            expansion_mode = "postgres_parent_fallback"

        context_blocks.append(
            {
                "document_id": document_id,
                "parent_id": parent_id,
                "section_title": row.get("section_title"),
                "page_start": row.get("page_start"),
                "page_end": row.get("page_end"),
                "heading_path": [
                    str(item)
                    for item in row.get("heading_path") or []
                    if str(item)
                ],
                "matched_chunk_ids": matched_chunk_ids,
                "window_chunk_ids": window_chunk_ids,
                "context_text": context_text,
                "expansion_mode": expansion_mode,
            }
        )

    return context_blocks


def load_visual_assets(
    *,
    thread_id: str | None,
    active_document_ids: list[str],
    asset_refs: list[str] | None = None,
    chunk_ids: list[str] | None = None,
) -> list[VisualAssetPayload]:
    """현재 thread 범위에서 visual asset 메타데이터를 읽는다."""
    del thread_id
    normalized_document_ids = _normalize_document_ids(active_document_ids)
    normalized_asset_refs = [str(item).strip() for item in asset_refs or [] if str(item).strip()]
    normalized_chunk_refs = [str(item).strip() for item in chunk_ids or [] if str(item).strip()]
    if not normalized_document_ids:
        return []

    with app_db_connection() as connection:
        document_repository = DocumentRepository(connection)
        asset_rows = document_repository.list_document_assets(normalized_document_ids)

    visual_assets: list[VisualAssetPayload] = []
    for row in asset_rows:
        document_id = str(row.get("document_id") or "").strip()
        chunk_id = str(row.get("chunk_id") or "").strip()
        asset_ref = str(row.get("asset_ref") or "").strip()

        if normalized_asset_refs and asset_ref not in normalized_asset_refs:
            continue
        if normalized_chunk_refs and not _match_chunk_ref(
            document_id=document_id,
            chunk_id=chunk_id,
            requested_refs=normalized_chunk_refs,
        ):
            continue
        if not normalized_asset_refs and not normalized_chunk_refs:
            continue

        metadata = dict(row.get("metadata") or {})
        visual_assets.append(
            {
                "asset_ref": asset_ref,
                "document_id": document_id,
                "chunk_id": chunk_id,
                "asset_kind": str(row.get("asset_kind") or "").strip(),
                "relative_path": str(row.get("relative_path") or "").strip(),
                "asset_stage": "stage2",
                "page": row.get("page"),
                "caption": row.get("caption"),
                "summary_text": row.get("summary_text"),
                "heading_path": [
                    str(item)
                    for item in metadata.get("heading_path") or []
                    if str(item)
                ],
                "pages": [
                    int(item)
                    for item in metadata.get("pages") or []
                    if isinstance(item, int) or str(item).isdigit()
                ],
            }
        )

    return visual_assets
