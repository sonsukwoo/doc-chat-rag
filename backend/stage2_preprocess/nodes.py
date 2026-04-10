"""Stage-2 preprocessing LangGraph nodes.

이 파일은 그래프 노드를 실행 순서대로 배치해, 파일 하나만 읽어도 처리 흐름이 보이게 유지한다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage, SystemMessage

from .llm import get_base_model
from .state import DocumentProfileResult, FigureReviewResult, PreprocessState, TableSummaryResult
from .utils import (
    TABLE_LIKE_FIGURE_OVERLAP_THRESHOLD,
    bbox_overlap_ratio,
    bbox_to_rect,
    clean_render_text,
    collect_document_profile_inputs,
    collect_page_context,
    fallback_visual_summary,
    guess_primary_picture_label,
    image_to_data_url,
    is_obvious_junk_figure,
    looks_like_page_counter,
    looks_like_short_heading,
    render_figure_html,
    render_figure_markdown,
    render_table_html,
    render_table_markdown,
    render_text_like_html,
    render_text_like_markdown,
    safe_mkdir,
    safe_write_json,
    safe_write_text,
)

# ---------------------------------------------------------------------------
# load_raw_document 노드
# raw.json과 source.pdf를 읽어 기본 상태를 만든다.
# ---------------------------------------------------------------------------
def load_raw_document(state: PreprocessState) -> dict[str, Any]:
    """Node: raw.json과 source.pdf를 읽어 stage-2 작업의 기본 상태를 만든다."""
    raw_json_path = Path(state["raw_json_path"]).expanduser().resolve()
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))

    source_pdf_path = Path(
        state.get("source_pdf_path") or payload.get("source_pdf") or raw_json_path.with_suffix(".pdf")
    ).expanduser().resolve()
    output_dir = raw_json_path.parent.resolve()

    with fitz.open(str(source_pdf_path)) as pdf:
        page_metrics = {
            page_index + 1: {
                "width": float(page.rect.width),
                "height": float(page.rect.height),
            }
            for page_index, page in enumerate(pdf)
        }

    return {
        "source_pdf_path": str(source_pdf_path),
        "output_dir": str(output_dir),
        "total_pages": payload.get("total_pages") or len(page_metrics),
        "elements": payload.get("elements", []),
        "page_metrics": page_metrics,
        "logs": [f"loaded:{raw_json_path.name}:{len(payload.get('elements', []))}"],
    }

# ---------------------------------------------------------------------------
# resolve_captions 노드
# caption_refs와 caption element를 연결해 visual caption을 보강한다.
# ---------------------------------------------------------------------------
def resolve_captions(state: PreprocessState) -> dict[str, Any]:
    """Node: caption_refs와 docling_ref를 연결해 visual에 resolved_caption을 붙인다."""
    elements = state["elements"]
    caption_map: dict[str, str] = {}
    for element in elements:
        if element.get("category") != "caption":
            continue
        docling_ref = element.get("docling_ref")
        text = clean_render_text(element.get("text", ""))
        if docling_ref and text:
            caption_map[str(docling_ref)] = text

    resolved: list[dict[str, Any]] = []
    for element in elements:
        item = dict(element)
        resolved_caption = None
        for ref in item.get("caption_refs") or []:
            if ref in caption_map:
                resolved_caption = caption_map[ref]
                break
        if resolved_caption:
            item["resolved_caption"] = resolved_caption
        resolved.append(item)

    return {
        "elements": resolved,
        "logs": [f"captions_resolved:{len(caption_map)}"],
    }

# ---------------------------------------------------------------------------
# normalize_elements 노드
# 텍스트를 정리하고 heading 승격 및 picture top-1 보조 필드를 붙인다.
# ---------------------------------------------------------------------------
def normalize_elements(state: PreprocessState) -> dict[str, Any]:
    """Node: 텍스트 정리, heading 승격, picture top-1 보조 필드 추가를 수행한다."""
    normalized: list[dict[str, Any]] = []

    for element in state["elements"]:
        item = dict(element)
        item["text"] = clean_render_text(item.get("text", ""))
        if item.get("internal_caption_text"):
            item["internal_caption_text"] = clean_render_text(item["internal_caption_text"])
        if item.get("resolved_caption"):
            item["resolved_caption"] = clean_render_text(item["resolved_caption"])

        if item.get("category") == "paragraph" and looks_like_short_heading(item.get("text", "")):
            item["raw_category"] = item["category"]
            item["category"] = "heading"

        label, confidence = guess_primary_picture_label(item)
        if label:
            item["primary_picture_label"] = label
            item["primary_picture_confidence"] = confidence

        normalized.append(item)

    return {
        "elements": normalized,
        "logs": [f"normalized:{len(normalized)}"],
    }

# ---------------------------------------------------------------------------
# infer_document_profile 노드
# 문서 앞부분 문맥으로 전체 주제와 관련 visual 힌트를 추론한다.
# ---------------------------------------------------------------------------
def infer_document_profile(state: PreprocessState) -> dict[str, Any]:
    """Node: 문서 앞부분 문맥을 보고 전체 문서 주제 프로파일을 추론한다."""
    headings, body_snippets = collect_document_profile_inputs(state["elements"])
    heading_block = "\n".join(f"- {heading}" for heading in headings) or "- (없음)"
    body_block = "\n".join(f"- {snippet}" for snippet in body_snippets) or "- (없음)"
    profiler = get_base_model().with_structured_output(DocumentProfileResult)

    prompt = (
        "너는 문서 전처리기다. 아래 문서 앞부분 요소를 보고 이 문서의 주제를 요약하라. "
        "이 결과는 이후 이미지 relevance 판단 기준으로 사용된다.\n\n"
        "[heading 후보]\n"
        f"{heading_block}\n\n"
        "[본문 샘플]\n"
        f"{body_block}\n\n"
        "relevant_visual_types에는 이 문서에서 본문 이해에 도움이 될 시각자료 유형을 적고, "
        "irrelevant_visual_hints에는 문맥상 무관할 가능성이 높은 광고/배너/장식 이미지 유형을 적어라. "
        "모든 값은 한국어 중심으로 작성하되, 시각자료 라벨은 flow_chart, screenshot_from_computer, table 같은 짧은 라벨을 섞어 써도 된다."
    )

    fallback_title = headings[0] if headings else (body_snippets[0] if body_snippets else "문서")
    fallback_topics = headings[:5] or body_snippets[:5] or ["문서"]

    try:
        result = profiler.invoke(
            [
                SystemMessage(
                    content=(
                        "출력은 반드시 구조화 스키마를 따르라. "
                        "title, document_type, main_topics, relevant_visual_types, irrelevant_visual_hints를 모두 채워라."
                    )
                ),
                HumanMessage(content=prompt),
            ]
        )
        profile = result.model_dump()
    except Exception as exc:  # pragma: no cover - runtime model dependency
        profile = DocumentProfileResult(
            title=fallback_title,
            document_type="문서",
            main_topics=fallback_topics[:5],
            relevant_visual_types=[
                "flow_chart",
                "screenshot_from_computer",
                "table",
                "line_chart",
                "bar_chart",
            ],
            irrelevant_visual_hints=["광고 배너", "문맥과 무관한 홍보 이미지", f"fallback:{exc}"],
        ).model_dump()

    return {
        "document_profile": profile,
        "logs": [f"document_profile:{profile.get('document_type', 'unknown')}"],
    }

# ---------------------------------------------------------------------------
# rule_filter_elements 노드
# 규칙만으로 확실한 junk 요소를 먼저 제거한다.
# ---------------------------------------------------------------------------
def rule_filter_elements(state: PreprocessState) -> dict[str, Any]:
    """Node: 규칙만으로 확실한 junk 요소를 먼저 제거한다."""
    filtered: list[dict[str, Any]] = []
    dropped = 0

    for element in state["elements"]:
        item = dict(element)
        category = item.get("category")

        if category in {"page_header", "page_footer"}:
            dropped += 1
            continue

        if looks_like_page_counter(item.get("text", "")):
            dropped += 1
            continue

        if category == "figure":
            drop_reason = is_obvious_junk_figure(item)
            if drop_reason:
                item["drop_reason"] = drop_reason
                dropped += 1
                continue

        filtered.append(item)

    return {
        "elements": filtered,
        "logs": [f"rule_filtered:dropped={dropped}:kept={len(filtered)}"],
    }

# ---------------------------------------------------------------------------
# build_visual_tasks 노드
# crop과 모델 검토가 필요한 figure/table 작업 목록을 만든다.
# ---------------------------------------------------------------------------
def build_visual_tasks(state: PreprocessState) -> dict[str, Any]:
    """Node: crop/VLM 대상 visual만 별도의 작업 목록으로 만든다."""
    tasks: list[dict[str, Any]] = []
    figure_review_ids: list[int] = []
    table_summary_ids: list[int] = []

    for element in state["elements"]:
        category = element.get("category")
        bbox = element.get("bbox")
        if category not in {"figure", "table"} or not bbox:
            continue

        task = {
            "element_id": int(element["id"]),
            "kind": category,
            "page": int(element["page"]),
            "bbox": bbox,
            "coord_origin": element.get("coord_origin"),
            "label": element.get("primary_picture_label"),
        }
        tasks.append(task)

        if category == "figure":
            figure_review_ids.append(int(element["id"]))
        elif category == "table":
            table_summary_ids.append(int(element["id"]))

    return {
        "visual_tasks": tasks,
        "figure_review_ids": figure_review_ids,
        "table_summary_ids": table_summary_ids,
        "logs": [f"visual_tasks:{len(tasks)}"],
    }

# ---------------------------------------------------------------------------
# crop_visuals 노드
# bbox 기준으로 figure/table 원본 이미지를 실제 파일로 저장한다.
# ---------------------------------------------------------------------------
def crop_visuals(state: PreprocessState) -> dict[str, Any]:
    """Node: figure/table bbox 기준 crop 이미지를 실제 파일로 저장한다."""
    source_pdf_path = Path(state["source_pdf_path"])
    output_dir = Path(state["output_dir"])
    page_metrics = state["page_metrics"]
    figures_dir = safe_mkdir(output_dir / "figures")
    tables_dir = safe_mkdir(output_dir / "tables")

    cropped_assets: dict[int, dict[str, str]] = {}
    counters: dict[tuple[str, int], int] = defaultdict(int)

    with fitz.open(str(source_pdf_path)) as pdf:
        for task in state.get("visual_tasks", []):
            element_id = int(task["element_id"])
            kind = task["kind"]
            page = int(task["page"])
            bbox = task["bbox"]
            coord_origin = task.get("coord_origin")
            page_metric = page_metrics.get(page)
            if not page_metric or page < 1 or page > len(pdf):
                continue

            rect = bbox_to_rect(
                bbox=bbox,
                page_height=page_metric["height"],
                coord_origin=coord_origin,
            )
            if rect.width <= 1 or rect.height <= 1:
                continue

            counters[(kind, page)] += 1
            seq = counters[(kind, page)]
            page_obj = pdf[page - 1]
            crop = page_obj.get_pixmap(clip=rect, matrix=fitz.Matrix(2, 2), alpha=False)

            if kind == "figure":
                label = task.get("label") or "uncategorized"
                target_dir = safe_mkdir(figures_dir / str(label))
                file_path = target_dir / f"page_{page}_figure_{seq}.png"
            else:
                target_dir = tables_dir
                file_path = target_dir / f"page_{page}_table_{seq}.png"

            crop.save(str(file_path))
            rel_path = file_path.relative_to(output_dir).as_posix()
            cropped_assets[element_id] = {
                "relative_path": rel_path,
                "absolute_path": str(file_path),
            }

    return {
        "cropped_assets": cropped_assets,
        "logs": [f"cropped_assets:{len(cropped_assets)}"],
    }

# ---------------------------------------------------------------------------
# review_single_figure 노드
# figure 하나를 멀티모달 모델로 검토해 keep/drop과 summary를 만든다.
# ---------------------------------------------------------------------------
def review_single_figure(state: PreprocessState) -> dict[str, Any]:
    """Node: figure 하나를 멀티모달 모델로 검토해 keep/drop + summary를 생성한다."""
    request = state["figure_review_request"]
    element_id = int(request["element_id"])
    element = request["element"]
    image_path = Path(request["absolute_path"])
    document_profile = request.get("document_profile") or {}
    page_context = request["page_context"]
    reviewer = get_base_model().with_structured_output(FigureReviewResult)

    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    label, confidence = guess_primary_picture_label(element)
    profile_text = json.dumps(document_profile, ensure_ascii=False, indent=2)
    prompt = (
        "너는 문서 전처리기다. 주어진 이미지를 보고 문서와 관련 있으면 keep, "
        "문서와 무관한 장식/광고/로고/아이콘이면 drop으로 판단하라. "
        "특히 문서 전체 주제와 현재 페이지 문맥을 우선 기준으로 relevance를 판단하라. "
        "keep일 때만 검색에 도움이 되는 짧은 한국어 summary를 작성하라. "
        "이미지 안에 읽을 수 있는 텍스트가 있으면 제목, 축 라벨, 범례, 버튼명, 메뉴명, 주요 숫자, 단계명 같은 핵심 문구를 summary에 자연스럽게 반영하라. "
        "단, 보이지 않는 텍스트를 추측하지 말고 실제로 식별 가능한 내용만 포함하라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- picture top1: {label} ({confidence:.3f})\n"
        f"- caption: {caption or '없음'}\n"
        f"- same-page text context:\n{page_context or '(없음)'}"
    )

    try:
        result = reviewer.invoke(
            [
                SystemMessage(
                    content=(
                        "출력은 반드시 구조화 스키마를 따르라. "
                        "summary와 reason은 반드시 한국어로 작성하라. "
                        "drop이면 summary는 null로 둔다. "
                        "keep일 때는 이미지 안에서 식별 가능한 텍스트를 검색에 도움이 되도록 summary에 반영하라."
                    )
                ),
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(image_path)},
                        },
                    ]
                ),
            ]
        )
        payload = result.model_dump()
    except Exception as exc:  # pragma: no cover - runtime model dependency
        payload = FigureReviewResult(
            action="keep",
            document_relevant=True,
            summary=fallback_visual_summary(element),
            reason=f"vlm_error:{exc}",
        ).model_dump()

    return {
        "figure_reviews": {element_id: payload},
        "logs": [f"figure_reviewed:{element_id}"],
    }

# ---------------------------------------------------------------------------
# summarize_tables 노드
# table crop와 현재 markdown을 바탕으로 표 summary를 batch 생성한다.
# ---------------------------------------------------------------------------
def summarize_tables(state: PreprocessState) -> dict[str, Any]:
    """Node: table crop와 현재 markdown을 바탕으로 table summary를 batch로 생성한다."""
    elements_by_id = {int(element["id"]): element for element in state["elements"]}
    cropped_assets = state.get("cropped_assets", {})
    summarizer = get_base_model().with_structured_output(TableSummaryResult)

    requests: list[list[Any]] = []
    request_ids: list[int] = []

    for element_id in state.get("table_summary_ids", []):
        element = elements_by_id.get(element_id)
        if not element:
            continue

        asset = cropped_assets.get(element_id)
        caption = clean_render_text(
            element.get("resolved_caption") or element.get("internal_caption_text") or ""
        )
        page_context = collect_page_context(state["elements"], int(element.get("page", 1)))

        prompt = (
            "너는 문서 전처리기다. 표의 구조를 복원하지 말고, 원본 이미지와 주변 문맥을 보고 "
            "검색에 도움이 되도록 핵심만 짧게 한국어로 요약하라.\n\n"
            f"- caption: {caption or '없음'}\n"
            f"- same-page text context:\n{page_context or '(없음)'}"
        )

        messages: list[Any] = [
            SystemMessage(content="출력은 반드시 구조화 스키마를 따르며, summary와 reason은 한국어로 작성하라."),
        ]

        if asset:
            messages.append(
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(Path(asset["absolute_path"]))},
                        },
                    ]
                )
            )
        else:
            messages.append(HumanMessage(content=prompt))

        request_ids.append(element_id)
        requests.append(messages)

    table_summaries: dict[int, dict[str, Any]] = {}
    if not requests:
        return {
            "table_summaries": table_summaries,
            "logs": ["table_summaries:0"],
        }

    try:
        results = summarizer.batch(requests)
        for element_id, result in zip(request_ids, results, strict=False):
            table_summaries[element_id] = result.model_dump()
    except Exception as exc:  # pragma: no cover - runtime model dependency
        for element_id in request_ids:
            element = elements_by_id[element_id]
            caption = clean_render_text(
                element.get("resolved_caption") or element.get("internal_caption_text") or ""
            )
            fallback = caption or clean_render_text(element.get("text", ""))[:240]
            table_summaries[element_id] = TableSummaryResult(
                summary=fallback or "표 요약 생성 실패",
                reason=f"table_summary_error:{exc}",
            ).model_dump()

    return {
        "table_summaries": table_summaries,
        "logs": [f"table_summaries:{len(table_summaries)}"],
    }

# ---------------------------------------------------------------------------
# clean_elements 노드
# VLM 결과와 crop 정보를 반영해 최종 element 목록을 정리한다.
# ---------------------------------------------------------------------------
def clean_elements(state: PreprocessState) -> dict[str, Any]:
    """Node: VLM 결과 반영, caption dedupe, table-like figure 중복 제거를 수행한다."""
    figure_reviews = state.get("figure_reviews", {})
    table_summaries = state.get("table_summaries", {})
    cropped_assets = state.get("cropped_assets", {})

    used_caption_refs = set()
    for element in state["elements"]:
        if element.get("category") in {"figure", "table"}:
            used_caption_refs.update(element.get("caption_refs") or [])

    kept: list[dict[str, Any]] = []
    for element in state["elements"]:
        item = dict(element)
        element_id = int(item["id"])
        category = item.get("category")

        if category == "caption" and item.get("docling_ref") in used_caption_refs:
            continue

        if category == "figure":
            review = figure_reviews.get(element_id)
            if review and review.get("action") == "drop":
                continue
            asset = cropped_assets.get(element_id)
            if asset:
                item["image_path"] = asset["relative_path"]
                item["image_abs_path"] = asset["absolute_path"]
            if review and review.get("summary"):
                item["visual_summary"] = review["summary"]
                item["visual_reason"] = review.get("reason")

        if category == "table":
            asset = cropped_assets.get(element_id)
            if asset:
                item["image_path"] = asset["relative_path"]
                item["image_abs_path"] = asset["absolute_path"]
            if element_id in table_summaries:
                item["table_summary"] = table_summaries[element_id]["summary"]
                item["table_summary_reason"] = table_summaries[element_id].get("reason")

        kept.append(item)

    tables_by_page: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for element in kept:
        if element.get("category") == "table":
            tables_by_page[int(element.get("page", 1))].append(element)

    final_elements: list[dict[str, Any]] = []
    for element in kept:
        if element.get("category") != "figure":
            final_elements.append(element)
            continue

        primary_label = element.get("primary_picture_label")
        if primary_label != "table":
            final_elements.append(element)
            continue

        overlaps = [
            bbox_overlap_ratio(element["bbox"], table["bbox"])
            for table in tables_by_page.get(int(element.get("page", 1)), [])
            if table.get("bbox") and element.get("bbox")
        ]
        if overlaps and max(overlaps) >= TABLE_LIKE_FIGURE_OVERLAP_THRESHOLD:
            continue

        final_elements.append(element)

    final_elements.sort(key=lambda item: int(item.get("order", item.get("id", 0))))
    return {
        "cleaned_elements": final_elements,
        "logs": [f"cleaned_elements:{len(final_elements)}"],
    }

# ---------------------------------------------------------------------------
# render_markdown 노드
# cleaned element를 문서 순서대로 조합해 최종 Markdown을 만든다.
# ---------------------------------------------------------------------------
def render_markdown(state: PreprocessState) -> dict[str, Any]:
    """Node: cleaned element를 문서 순서대로 조합해 최종 Markdown을 생성한다."""
    blocks: list[str] = []

    for element in state["cleaned_elements"]:
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

    cleaned_markdown = "\n\n".join(blocks).strip() + "\n"
    return {
        "cleaned_markdown": cleaned_markdown,
        "logs": [f"rendered_markdown:{len(blocks)}"],
    }

# ---------------------------------------------------------------------------
# render_preview_html 노드
# 검수용 preview HTML을 생성한다.
# ---------------------------------------------------------------------------
def render_preview_html(state: PreprocessState) -> dict[str, Any]:
    """Node: 검수용 preview HTML을 생성한다."""
    blocks: list[str] = []

    for element in state["cleaned_elements"]:
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

    preview_html = "\n".join(
        [
            "<!doctype html>",
            "<html lang=\"ko\">",
            "<head>",
            "<meta charset=\"utf-8\" />",
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />",
            "<title>Document Preview</title>",
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

    return {
        "preview_html": preview_html,
        "logs": [f"rendered_preview_html:{len(blocks)}"],
    }

# ---------------------------------------------------------------------------
# write_outputs 노드
# cleaned.json, cleaned.md, preview.html을 디스크에 저장한다.
# ---------------------------------------------------------------------------
def write_outputs(state: PreprocessState) -> dict[str, Any]:
    """Node: cleaned JSON / Markdown / preview HTML 파일을 디스크에 저장한다."""
    output_dir = Path(state["output_dir"])
    cleaned_json_path = safe_write_json(
        output_dir / "cleaned.json",
        {
            "source_pdf": state["source_pdf_path"],
            "total_pages": state["total_pages"],
            "document_profile": state.get("document_profile"),
            "elements": state["cleaned_elements"],
        },
    )
    cleaned_md_path = safe_write_text(output_dir / "cleaned.md", state["cleaned_markdown"])
    preview_html_path = safe_write_text(output_dir / "preview.html", state["preview_html"])

    return {
        "output_paths": {
            "cleaned_json": str(cleaned_json_path),
            "cleaned_md": str(cleaned_md_path),
            "preview_html": str(preview_html_path),
        },
        "logs": ["outputs_written"],
    }
