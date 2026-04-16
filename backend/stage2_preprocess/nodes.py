"""Stage-2 preprocessing LangGraph nodes.

이 파일은 그래프 노드를 실행 순서대로 배치해, 파일 하나만 읽어도 처리 흐름이 보이게 유지한다.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
from langchain_core.messages import HumanMessage
from langgraph.runtime import Runtime
from langgraph.types import default_retry_on

from .llm import (
    MODEL_RETRY_MAX_ATTEMPTS,
    get_base_model,
    get_text_model,
)
from .state import (
    CroppedAsset,
    DocumentProfilePayload,
    DocumentProfileResult,
    ElementPayload,
    FigureReviewResult,
    FigureReviewPayload,
    FigureReviewRequest,
    OutputPaths,
    OrderingResolutionPayload,
    PageMetric,
    PreprocessState,
    TableSummaryInput,
    TableSummaryPayload,
    TableSummaryRoute,
    TableSummaryRouteResult,
    TableSummaryResult,
    VisualTask,
)
from .utils import (
    TABLE_LIKE_FIGURE_OVERLAP_THRESHOLD,
    bbox_overlap_ratio,
    bbox_to_rect,
    clean_render_text,
    compact_html_for_prompt,
    collect_document_profile_inputs,
    collect_neighbor_body_texts,
    export_cleaned_elements,
    fallback_visual_summary,
    format_document_profile_for_prompt,
    guess_primary_picture_label,
    image_to_data_url,
    is_generic_full_page_figure,
    is_obvious_junk_figure,
    reorder_visual_outliers_by_bbox,
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


def _should_raise_for_retry(
    exc: Exception,
    runtime: Runtime[Any],
    *,
    max_attempts: int = MODEL_RETRY_MAX_ATTEMPTS,
) -> bool:
    """일시 오류면 LangGraph retry_policy가 재시도할 수 있게 예외를 다시 올린다."""
    attempt = getattr(getattr(runtime, "execution_info", None), "node_attempt", 1)
    return attempt < max_attempts and default_retry_on(exc)


# ---------------------------------------------------------------------------
# load_raw_document 노드
# raw.json과 source.pdf를 읽어 기본 상태를 만든다.
# ---------------------------------------------------------------------------
def load_raw_document(state: PreprocessState) -> dict[str, Any]:
    """Node: raw.json과 source.pdf를 읽어 stage-2 작업의 기본 상태를 만든다."""
    raw_json_path = Path(state["raw_json_path"]).expanduser().resolve()
    payload = json.loads(raw_json_path.read_text(encoding="utf-8"))

    if raw_json_path.parent.name == "stage1":
        default_output_dir = raw_json_path.parent.parent / "stage2"
        default_source_pdf_path = raw_json_path.parent.parent / "source" / "original.pdf"
    else:
        default_output_dir = raw_json_path.parent
        default_source_pdf_path = raw_json_path.with_suffix(".pdf")

    source_pdf_path = Path(
        state.get("source_pdf_path")
        or payload.get("source_pdf")
        or default_source_pdf_path
    ).expanduser().resolve()
    output_dir = Path(
        state.get("output_dir") or default_output_dir
    ).expanduser().resolve()

    with fitz.open(str(source_pdf_path)) as pdf:
        page_metrics: dict[int, PageMetric] = {
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
# 텍스트를 정리하고 picture top-1 보조 필드를 붙인다.
# ---------------------------------------------------------------------------
def normalize_elements(state: PreprocessState) -> dict[str, Any]:
    """Node: 텍스트 정리와 picture top-1 보조 필드 추가를 수행한다."""
    normalized: list[dict[str, Any]] = []

    for element in state["elements"]:
        item = dict(element)
        item["text"] = clean_render_text(item.get("text", ""))
        if item.get("internal_caption_text"):
            item["internal_caption_text"] = clean_render_text(item["internal_caption_text"])
        if item.get("resolved_caption"):
            item["resolved_caption"] = clean_render_text(item["resolved_caption"])

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
    profiler = get_text_model().with_structured_output(DocumentProfileResult)

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
    used_fallback = False
    fallback_error_name = ""

    try:
        result = profiler.invoke(
            [
                HumanMessage(content=prompt),
            ]
        )
        profile: DocumentProfilePayload = result.model_dump()
    except Exception as exc:  # pragma: no cover - runtime model dependency
        used_fallback = True
        fallback_error_name = type(exc).__name__
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
            irrelevant_visual_hints=["광고 배너", "문맥과 무관한 홍보 이미지"],
        ).model_dump()

    log_message = f"document_profile:{profile.get('document_type', 'unknown')}"
    if used_fallback:
        log_message += f":fallback={fallback_error_name}"

    return {
        "document_profile": profile,
        "logs": [log_message],
    }

# ---------------------------------------------------------------------------
# rule_filter_elements 노드
# 규칙만으로 확실한 visual junk 요소만 먼저 제거한다.
# ---------------------------------------------------------------------------
def rule_filter_elements(state: PreprocessState) -> dict[str, Any]:
    """Node: 규칙만으로 확실한 visual junk 요소만 먼저 제거한다."""
    filtered: list[dict[str, Any]] = []
    dropped = 0
    page_metrics = state.get("page_metrics", {})

    for element in state["elements"]:
        item = dict(element)
        category = item.get("category")

        if category == "figure":
            if is_obvious_junk_figure(item, page_metrics):
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
    tasks: list[VisualTask] = []
    figure_review_ids: list[int] = []
    table_summary_ids: list[int] = []

    for element in state["elements"]:
        category = element.get("category")
        bbox = element.get("bbox")
        if category not in {"figure", "table"} or not bbox:
            continue

        task: VisualTask = {
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

    cropped_assets: dict[int, CroppedAsset] = {}
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
# build_figure_review_requests 노드
# figure review에 필요한 request payload를 먼저 조립해 상태에 적재한다.
# ---------------------------------------------------------------------------
def build_figure_review_requests(state: PreprocessState) -> dict[str, Any]:
    """Node: figure fan-out worker들이 사용할 request 목록을 만든다."""
    elements_by_id: dict[int, ElementPayload] = {
        int(element["id"]): element for element in state["elements"]
    }
    document_profile: DocumentProfilePayload = state["document_profile"]
    requests: list[FigureReviewRequest] = []

    for element_id in state.get("figure_review_ids", []):
        asset = state.get("cropped_assets", {}).get(element_id)
        element = elements_by_id.get(element_id)
        if not asset or not element:
            continue

        prev_body_text, next_body_text = collect_neighbor_body_texts(
            state["elements"],
            element_id,
        )
        requests.append(
            FigureReviewRequest(
                element_id=element_id,
                element=element,
                absolute_path=asset["absolute_path"],
                document_profile=document_profile,
                prev_body_text=prev_body_text,
                next_body_text=next_body_text,
            )
        )

    return {
        "figure_review_requests": requests,
        "logs": [f"figure_review_requests:{len(requests)}"],
    }

# ---------------------------------------------------------------------------
# review_single_figure 노드
# figure 하나를 멀티모달 모델로 검토해 keep/drop과 summary를 만든다.
# ---------------------------------------------------------------------------
def review_single_figure(
    state: PreprocessState,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    """Node: figure 하나를 멀티모달 모델로 검토해 keep/drop + summary를 생성한다."""
    request: FigureReviewRequest = state["figure_review_request"]
    element_id = int(request["element_id"])
    element = request["element"]
    image_path = Path(request["absolute_path"])
    document_profile: DocumentProfilePayload = request["document_profile"]
    prev_body_text = request.get("prev_body_text") or ""
    next_body_text = request.get("next_body_text") or ""
    reviewer = get_base_model().with_structured_output(FigureReviewResult)

    caption = clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )
    profile_text = format_document_profile_for_prompt(document_profile)
    local_context_lines: list[str] = []
    if prev_body_text:
        local_context_lines.append(f"- previous body text: {prev_body_text}")
    if next_body_text:
        local_context_lines.append(f"- next body text: {next_body_text}")
    local_context_block = "\n".join(local_context_lines) or "- 없음"
    prompt = (
        "이미지와 document_profile을 보고 이미지가 문서 본문과 관련이 있으면 keep, "
        "문서와 무관한 광고·장식·로고·아이콘이면 drop으로 판단하라. "
        "판단할 때는 아래 document profile에 담긴 문서 주제와 핵심 토픽을 우선 참고하라. "
        "아래 local body context는 이미지 주변의 본문 텍스트로, 보조 힌트로만 참고하라. "
        "keep이면 RAG 검색에 도움이 되는 한국어 요약을 작성하고, drop이면 summary는 null로 반환하라. "
        "이미지 안의 식별 가능한 텍스트, 도표, 그래프는 요약에 반영하고, 보이지 않는 내용은 추측하지 말라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- caption: {caption or '없음'}\n"
        f"- local body context:\n{local_context_block}"
    )

    try:
        result = reviewer.invoke(
            [
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
        payload: FigureReviewPayload = result.model_dump()
    except Exception as exc:  # pragma: no cover - runtime model dependency
        if _should_raise_for_retry(exc, runtime):
            raise
        if is_generic_full_page_figure(element, state.get("page_metrics")):
            payload = FigureReviewResult(action="drop", summary=None).model_dump()
        else:
            payload = FigureReviewResult(
                action="keep",
                summary=fallback_visual_summary(element),
            ).model_dump()

    return {
        "figure_reviews": {element_id: payload},
        "logs": [f"figure_reviewed:{element_id}"],
    }

# ---------------------------------------------------------------------------
# table summary 공통 helper
# 단계별 노드에서 공통으로 재사용하는 prompt / fallback / batch helper를 둔다.
# ---------------------------------------------------------------------------
def _extract_element_caption(element: dict[str, Any]) -> str:
    """figure/table 공통 caption 후보를 하나의 문자열로 정리한다."""
    return clean_render_text(
        element.get("resolved_caption") or element.get("internal_caption_text") or ""
    )


def _build_local_body_context_block(
    *,
    prev_body_text: str,
    next_body_text: str,
) -> str:
    """앞뒤 본문 힌트를 prompt용 문자열로 정리한다."""
    local_context_lines: list[str] = []
    if prev_body_text:
        local_context_lines.append(f"- previous body text: {prev_body_text}")
    if next_body_text:
        local_context_lines.append(f"- next body text: {next_body_text}")
    return "\n".join(local_context_lines) or "- 없음"


def _build_table_route_prompt(
    *,
    profile_text: str,
    caption: str,
    html_excerpt: str,
) -> str:
    """table html만으로 요약 가능한지 판단하는 라우팅 prompt를 만든다."""
    return (
        "아래 table HTML만 보고, 이미지를 보지 않아도 이 표를 한국어로 요약할 수 있는지 판단하라. "
        "열 이름, 행 이름, 값 구조가 HTML에 충분히 드러나면 text, "
        "구조가 깨져 있거나 의미 파악이 어려워 이미지 확인이 필요하면 vlm을 반환하라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- caption: {caption or '없음'}\n"
        f"- table html:\n{html_excerpt}"
    )


def _build_table_text_summary_prompt(
    *,
    profile_text: str,
    caption: str,
    source_block: str,
) -> str:
    """text 모델용 table summary prompt를 만든다."""
    return (
        "아래 table HTML을 보고 RAG 검색에 도움이 되도록 핵심만 짧게 한국어로 요약하라. "
        "표 구조를 복원하려 하지 말고, 제목·열 이름·행 이름·핵심 값 관계가 드러나면 이를 반영하라. "
        "보이지 않는 내용은 추측하지 말라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- caption: {caption or '없음'}\n"
        f"- table html:\n{source_block}"
    )


def _build_table_context_fallback_prompt(
    *,
    profile_text: str,
    caption: str,
    text_excerpt: str,
    local_context_block: str,
) -> str:
    """이미지 없이 caption/text/context만으로 요약할 때 쓸 fallback prompt를 만든다."""
    return (
        "이미지가 없으므로 아래 caption, table text, local body context만 참고해 "
        "RAG 검색에 도움이 되도록 핵심만 짧게 한국어로 요약하라. "
        "표 구조를 복원하려 하지 말고, 제목·핵심 주제·비교 대상이 드러나면 이를 반영하라. "
        "보이지 않는 내용은 추측하지 말라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- caption: {caption or '없음'}\n"
        f"- table text:\n{text_excerpt or '(없음)'}\n"
        f"- local body context:\n{local_context_block}"
    )


def _build_table_vlm_summary_prompt(
    *,
    profile_text: str,
    caption: str,
    local_context_block: str,
) -> str:
    """VLM용 table summary prompt를 만든다."""
    return (
        "표의 구조를 복원하지 말고, 이미지를 보고 RAG 검색에 도움이 되도록 핵심만 짧게 한국어로 요약하라. "
        "판단할 때는 아래 document profile에 담긴 문서 주제와 핵심 토픽을 우선 참고하라. "
        "아래 local body context는 표 주변의 본문 텍스트로, 보조 힌트로만 참고하라. "
        "표 안의 식별 가능한 제목, 열 이름, 비교 축, 주요 수치는 요약에 반영하고, 보이지 않는 내용은 추측하지 말라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- caption: {caption or '없음'}\n"
        f"- local body context:\n{local_context_block}"
    )


def _prepare_table_summary_inputs(
    *,
    table_ids: list[int],
    elements: list[ElementPayload],
    elements_by_id: dict[int, ElementPayload],
    cropped_assets: dict[int, CroppedAsset],
) -> tuple[dict[int, TableSummaryInput], dict[int, TableSummaryRoute]]:
    """table summary에 필요한 공통 입력과 기본 route를 미리 만든다."""
    prepared_inputs: dict[int, TableSummaryInput] = {}
    seeded_route_results: dict[int, TableSummaryRoute] = {}

    for element_id in table_ids:
        element = elements_by_id.get(element_id)
        if not element:
            continue

        asset = cropped_assets.get(element_id)
        caption = _extract_element_caption(element)
        html_excerpt = compact_html_for_prompt(element.get("html", ""))
        text_excerpt = clean_render_text(element.get("text", ""))[:800]
        prev_body_text, next_body_text = collect_neighbor_body_texts(
            elements,
            element_id,
        )
        local_context_block = _build_local_body_context_block(
            prev_body_text=prev_body_text,
            next_body_text=next_body_text,
        )

        prepared_inputs[element_id] = {
            "asset": asset,
            "caption": caption,
            "html_excerpt": html_excerpt,
            "text_excerpt": text_excerpt,
            "local_context_block": local_context_block,
        }

        # html이 없으면 image 유무만으로 기본 경로를 결정한다.
        if not html_excerpt:
            seeded_route_results[element_id] = "vlm" if asset else "text"
            continue
    return prepared_inputs, seeded_route_results


def _build_table_summary_fallback(element: ElementPayload) -> TableSummaryPayload:
    """table summary 생성 실패 시 사용할 최소 fallback payload를 만든다."""
    caption = _extract_element_caption(element)
    fallback = caption or clean_render_text(element.get("text", ""))[:240]
    return TableSummaryResult(
        summary=fallback or "표 요약 생성 실패",
    ).model_dump()


def _run_text_table_summary_batch(
    *,
    text_summarizer: Any,
    text_request_ids: list[int],
    text_requests: list[list[Any]],
    elements_by_id: dict[int, ElementPayload],
    runtime: Runtime[Any],
) -> dict[int, TableSummaryPayload]:
    """text 경로 table summary batch를 실행하고 실패 시 fallback으로 대체한다."""
    summaries: dict[int, TableSummaryPayload] = {}
    if not text_requests:
        return summaries

    try:
        text_results = text_summarizer.batch(text_requests)
        for element_id, result in zip(text_request_ids, text_results, strict=False):
            summaries[element_id] = result.model_dump()
    except Exception as exc:
        if _should_raise_for_retry(exc, runtime):
            raise
        for element_id in text_request_ids:
            element = elements_by_id[element_id]
            summaries[element_id] = _build_table_summary_fallback(element)

    return summaries


def _run_vlm_table_summary_batch(
    *,
    vlm_summarizer: Any,
    vlm_request_ids: list[int],
    vlm_requests: list[list[Any]],
    elements_by_id: dict[int, ElementPayload],
    runtime: Runtime[Any],
) -> dict[int, TableSummaryPayload]:
    """VLM 경로 table summary batch를 실행하고 실패 시 fallback으로 대체한다."""
    summaries: dict[int, TableSummaryPayload] = {}
    if not vlm_requests:
        return summaries

    try:
        vlm_results = vlm_summarizer.batch(vlm_requests)
        for element_id, result in zip(vlm_request_ids, vlm_results, strict=False):
            summaries[element_id] = result.model_dump()
    except Exception as exc:
        if _should_raise_for_retry(exc, runtime):
            raise
        for element_id in vlm_request_ids:
            element = elements_by_id[element_id]
            summaries[element_id] = _build_table_summary_fallback(element)

    return summaries


def _resolve_table_route_targets(
    table_ids: list[int],
    prepared_inputs: dict[int, TableSummaryInput],
    route_results: dict[int, TableSummaryRoute],
) -> tuple[list[int], list[int]]:
    """route 결과를 바탕으로 text/VLM batch 대상 id를 계산한다."""
    text_target_ids: list[int] = []
    vlm_target_ids: list[int] = []

    for element_id in table_ids:
        prepared = prepared_inputs.get(element_id)
        if not prepared:
            continue

        asset = prepared.get("asset")
        route = route_results.get(element_id, "vlm" if asset else "text")

        # image가 없으면 vlm 경로여도 text 모델로 fallback 처리한다.
        if route == "vlm" and asset:
            vlm_target_ids.append(element_id)
            continue

        text_target_ids.append(element_id)

    return text_target_ids, vlm_target_ids


# ---------------------------------------------------------------------------
# prepare_table_summary_inputs 노드
# table summary에 필요한 공통 입력과 기본 route를 상태에 적재한다.
# ---------------------------------------------------------------------------
def prepare_table_summary_inputs(state: PreprocessState) -> dict[str, Any]:
    """Node: table summary용 공통 입력 payload를 만든다."""
    elements_by_id: dict[int, ElementPayload] = {
        int(element["id"]): element for element in state["elements"]
    }
    cropped_assets = state.get("cropped_assets", {})
    table_ids = [int(element_id) for element_id in state.get("table_summary_ids", [])]
    prepared_inputs, seeded_route_results = (
        _prepare_table_summary_inputs(
            table_ids=table_ids,
            elements=state["elements"],
            elements_by_id=elements_by_id,
            cropped_assets=cropped_assets,
        )
    )

    return {
        "table_summary_inputs": prepared_inputs,
        "table_summary_routes": seeded_route_results,
        "logs": [f"table_summary_inputs:{len(prepared_inputs)}"],
    }


# ---------------------------------------------------------------------------
# route_table_summaries 노드
# table별로 text/vlm 어느 경로로 summary를 만들지 batch 라우팅한다.
# ---------------------------------------------------------------------------
def route_table_summaries(
    state: PreprocessState,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    """Node: table별 summary 경로를 text/vlm으로 batch 라우팅한다."""
    prepared_inputs: dict[int, TableSummaryInput] = state.get("table_summary_inputs", {})
    seeded_route_results: dict[int, TableSummaryRoute] = dict(
        state.get("table_summary_routes", {})
    )
    profile_text = format_document_profile_for_prompt(state.get("document_profile") or {})
    route_reviewer = get_text_model().with_structured_output(TableSummaryRouteResult)

    route_request_ids: list[int] = []
    route_requests: list[list[Any]] = []
    for element_id, prepared in prepared_inputs.items():
        if element_id in seeded_route_results:
            continue

        route_request_ids.append(element_id)
        route_requests.append(
            [
                HumanMessage(
                    content=_build_table_route_prompt(
                        profile_text=profile_text,
                        caption=prepared["caption"],
                        html_excerpt=prepared["html_excerpt"],
                    )
                )
            ]
        )

    route_results: dict[int, TableSummaryRoute] = dict(seeded_route_results)
    if route_requests:
        try:
            route_batch_results = route_reviewer.batch(route_requests)
            for element_id, result in zip(route_request_ids, route_batch_results, strict=False):
                route_results[element_id] = result.route
        except Exception as exc:
            if _should_raise_for_retry(exc, runtime):
                raise
            for element_id in route_request_ids:
                asset = prepared_inputs[element_id].get("asset")
                route_results[element_id] = "vlm" if asset else "text"

    text_target_ids, vlm_target_ids = _resolve_table_route_targets(
        table_ids=[int(element_id) for element_id in state.get("table_summary_ids", [])],
        prepared_inputs=prepared_inputs,
        route_results=route_results,
    )

    return {
        "table_summary_routes": route_results,
        "logs": [
            (
                "table_routes:"
                f"text={len(text_target_ids)}:"
                f"vlm={len(vlm_target_ids)}"
            )
        ],
    }


# ---------------------------------------------------------------------------
# summarize_tables_text 노드
# text 경로 table들과 asset이 없는 vlm fallback table들을 text batch로 요약한다.
# ---------------------------------------------------------------------------
def summarize_tables_text(
    state: PreprocessState,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    """Node: text 모델로 처리 가능한 table summary를 batch 생성한다."""
    prepared_inputs: dict[int, TableSummaryInput] = state.get("table_summary_inputs", {})
    route_results: dict[int, TableSummaryRoute] = state.get("table_summary_routes", {})
    profile_text = format_document_profile_for_prompt(state.get("document_profile") or {})
    elements_by_id: dict[int, ElementPayload] = {
        int(element["id"]): element for element in state["elements"]
    }
    text_summarizer = get_text_model().with_structured_output(TableSummaryResult)

    text_request_ids: list[int] = []
    text_requests: list[list[Any]] = []
    for element_id in [int(item) for item in state.get("table_summary_ids", [])]:
        prepared = prepared_inputs.get(element_id)
        if not prepared:
            continue

        caption = prepared["caption"]
        html_excerpt = prepared["html_excerpt"]
        text_excerpt = prepared["text_excerpt"]
        local_context_block = prepared["local_context_block"]
        asset = prepared["asset"]
        route = route_results.get(element_id, "vlm" if asset else "text")

        if route == "vlm" and asset:
            continue

        if route == "text":
            source_block = html_excerpt or text_excerpt or "(없음)"
            prompt = _build_table_text_summary_prompt(
                profile_text=profile_text,
                caption=caption,
                source_block=source_block,
            )
        else:
            prompt = _build_table_context_fallback_prompt(
                profile_text=profile_text,
                caption=caption,
                text_excerpt=text_excerpt,
                local_context_block=local_context_block,
            )

        text_request_ids.append(element_id)
        text_requests.append([HumanMessage(content=prompt)])

    table_summaries = _run_text_table_summary_batch(
        text_summarizer=text_summarizer,
        text_request_ids=text_request_ids,
        text_requests=text_requests,
        elements_by_id=elements_by_id,
        runtime=runtime,
    )

    return {
        "table_summaries": table_summaries,
        "logs": [f"table_summaries_text:{len(table_summaries)}"],
    }


# ---------------------------------------------------------------------------
# summarize_tables_vlm 노드
# image가 필요한 table들을 VLM batch로 요약한다.
# ---------------------------------------------------------------------------
def summarize_tables_vlm(
    state: PreprocessState,
    runtime: Runtime[Any],
) -> dict[str, Any]:
    """Node: VLM이 필요한 table summary를 batch 생성한다."""
    prepared_inputs: dict[int, TableSummaryInput] = state.get("table_summary_inputs", {})
    route_results: dict[int, TableSummaryRoute] = state.get("table_summary_routes", {})
    profile_text = format_document_profile_for_prompt(state.get("document_profile") or {})
    elements_by_id: dict[int, ElementPayload] = {
        int(element["id"]): element for element in state["elements"]
    }
    vlm_summarizer = get_base_model().with_structured_output(TableSummaryResult)

    vlm_request_ids: list[int] = []
    vlm_requests: list[list[Any]] = []
    for element_id in [int(item) for item in state.get("table_summary_ids", [])]:
        prepared = prepared_inputs.get(element_id)
        if not prepared:
            continue

        asset = prepared["asset"]
        if not asset:
            continue

        route = route_results.get(element_id, "vlm")
        if route != "vlm":
            continue

        prompt = _build_table_vlm_summary_prompt(
            profile_text=profile_text,
            caption=prepared["caption"],
            local_context_block=prepared["local_context_block"],
        )
        vlm_request_ids.append(element_id)
        vlm_requests.append(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": image_to_data_url(Path(asset["absolute_path"]))},
                        },
                    ]
                ),
            ]
        )

    table_summaries = _run_vlm_table_summary_batch(
        vlm_summarizer=vlm_summarizer,
        vlm_request_ids=vlm_request_ids,
        vlm_requests=vlm_requests,
        elements_by_id=elements_by_id,
        runtime=runtime,
    )

    return {
        "table_summaries": table_summaries,
        "logs": [f"table_summaries_vlm:{len(table_summaries)}"],
    }

# ---------------------------------------------------------------------------
# clean_elements 노드
# VLM 결과와 crop 정보를 반영해 최종 element 목록을 정리한다.
# ---------------------------------------------------------------------------
def clean_elements(state: PreprocessState) -> dict[str, Any]:
    """Node: VLM 결과 반영, caption dedupe, table-like figure 중복 제거를 수행한다."""
    figure_reviews: dict[int, FigureReviewPayload] = state.get("figure_reviews", {})
    table_summaries: dict[int, TableSummaryPayload] = state.get("table_summaries", {})
    cropped_assets: dict[int, CroppedAsset] = state.get("cropped_assets", {})

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
            if review and review.get("summary"):
                item["visual_summary"] = review["summary"]

        if category == "table":
            asset = cropped_assets.get(element_id)
            if asset:
                item["image_path"] = asset["relative_path"]
            if element_id in table_summaries:
                item["table_summary"] = table_summaries[element_id]["summary"]

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
# resolve_visual_order_outliers 노드
# 같은 페이지의 visual outlier만 bbox 기준으로 보수적으로 재배치한다.
# ---------------------------------------------------------------------------
def resolve_visual_order_outliers(state: PreprocessState) -> dict[str, Any]:
    """Node: figure/table/caption 중 순서가 크게 어긋난 visual만 bbox 기준으로 보정한다."""
    cleaned_elements = [
        dict(element) for element in state.get("cleaned_elements", [])
    ]
    if not cleaned_elements:
        return {
            "cleaned_elements": cleaned_elements,
            "ordering_resolution": OrderingResolutionPayload(
                applied=False,
                adjusted_ids=[],
                rank_gap_threshold=3,
            ),
            "logs": ["visual_order_resolved:0"],
        }

    resolved_elements, adjusted_ids = reorder_visual_outliers_by_bbox(
        cleaned_elements,
        state["page_metrics"],
    )

    for resolved_index, element in enumerate(resolved_elements, start=1):
        element["resolved_order"] = resolved_index

    return {
        "cleaned_elements": resolved_elements,
        "ordering_resolution": OrderingResolutionPayload(
            applied=bool(adjusted_ids),
            adjusted_ids=adjusted_ids,
            rank_gap_threshold=3,
        ),
        "logs": [f"visual_order_resolved:{len(adjusted_ids)}"],
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
            "ordering_resolution": state.get("ordering_resolution"),
            "elements": export_cleaned_elements(state["cleaned_elements"]),
        },
    )
    cleaned_md_path = safe_write_text(output_dir / "cleaned.md", state["cleaned_markdown"])
    preview_html_path = safe_write_text(output_dir / "preview.html", state["preview_html"])

    output_paths: OutputPaths = {
        "cleaned_json": str(cleaned_json_path),
        "cleaned_md": str(cleaned_md_path),
        "preview_html": str(preview_html_path),
    }

    return {
        "output_paths": output_paths,
        "logs": ["outputs_written"],
    }
