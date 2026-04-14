"""Stage-2 preprocessing graph wiring and CLI entrypoint."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Any, Literal

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .llm import DEFAULT_RAW_JSON_PATH
from .nodes import (
    build_figure_review_requests,
    build_visual_tasks,
    clean_elements,
    crop_visuals,
    infer_document_profile,
    load_raw_document,
    normalize_elements,
    prepare_table_summary_inputs,
    render_markdown,
    render_preview_html,
    resolve_visual_order_outliers,
    resolve_captions,
    review_single_figure,
    route_table_summaries,
    rule_filter_elements,
    summarize_tables_text,
    summarize_tables_vlm,
    write_outputs,
)
from .state import (
    PreprocessInputState,
    PreprocessOutputState,
    PreprocessState,
)


def route_figure_reviews(
    state: PreprocessState,
) -> list[Send] | Literal["prepare_table_summary_inputs"]:
    """미리 조립된 figure review request를 Send fan-out으로 뿌린다."""
    requests = state.get("figure_review_requests", [])
    if not requests:
        return "prepare_table_summary_inputs"

    return [
        Send(
            "review_single_figure",
            {
                "figure_review_request": request,
            },
        )
        for request in requests
    ]


def route_table_summary_batches(state: PreprocessState) -> Any:
    """table route 결과를 바탕으로 text/VLM batch 노드 실행 경로를 정한다."""
    prepared_inputs = state.get("table_summary_inputs", {})
    route_results = state.get("table_summary_routes", {})
    table_ids = [int(element_id) for element_id in state.get("table_summary_ids", [])]
    has_text = False
    has_vlm = False

    for element_id in table_ids:
        prepared = prepared_inputs.get(element_id)
        if not prepared:
            continue

        asset = prepared.get("asset")
        route = route_results.get(element_id, "vlm" if asset else "text")

        if route == "vlm" and asset:
            has_vlm = True
        else:
            has_text = True

    if has_text and has_vlm:
        return ["summarize_tables_text", "summarize_tables_vlm"]
    if has_text:
        return "summarize_tables_text"
    if has_vlm:
        return "summarize_tables_vlm"
    return "clean_elements"


def _register_nodes(graph: StateGraph) -> None:
    """그래프에 stage-2 노드를 일괄 등록한다."""
    graph.add_node("load_raw_document", load_raw_document)
    graph.add_node("resolve_captions", resolve_captions)
    graph.add_node("normalize_elements", normalize_elements)
    graph.add_node("infer_document_profile", infer_document_profile)
    graph.add_node("rule_filter_elements", rule_filter_elements)
    graph.add_node("build_visual_tasks", build_visual_tasks)
    graph.add_node("crop_visuals", crop_visuals)
    graph.add_node("build_figure_review_requests", build_figure_review_requests)
    graph.add_node("review_single_figure", review_single_figure)
    graph.add_node("prepare_table_summary_inputs", prepare_table_summary_inputs)
    graph.add_node("route_table_summaries", route_table_summaries)
    graph.add_node("summarize_tables_text", summarize_tables_text)
    graph.add_node("summarize_tables_vlm", summarize_tables_vlm)
    graph.add_node("clean_elements", clean_elements)
    graph.add_node("resolve_visual_order_outliers", resolve_visual_order_outliers)
    graph.add_node("render_markdown", render_markdown)
    graph.add_node("render_preview_html", render_preview_html)
    graph.add_node("write_outputs", write_outputs)


def _register_edges(graph: StateGraph) -> None:
    """그래프의 순차/조건부 엣지를 한 곳에서 정의한다."""
    graph.add_edge(START, "load_raw_document")
    graph.add_edge("load_raw_document", "resolve_captions")
    graph.add_edge("resolve_captions", "normalize_elements")
    graph.add_edge("normalize_elements", "infer_document_profile")
    graph.add_edge("infer_document_profile", "rule_filter_elements")
    graph.add_edge("rule_filter_elements", "build_visual_tasks")
    graph.add_edge("build_visual_tasks", "crop_visuals")
    graph.add_edge("crop_visuals", "build_figure_review_requests")
    graph.add_conditional_edges(
        "build_figure_review_requests",
        route_figure_reviews,
        ["review_single_figure", "prepare_table_summary_inputs"],
    )
    graph.add_edge("review_single_figure", "prepare_table_summary_inputs")
    graph.add_edge("prepare_table_summary_inputs", "route_table_summaries")
    graph.add_conditional_edges(
        "route_table_summaries",
        route_table_summary_batches,
        ["summarize_tables_text", "summarize_tables_vlm", "clean_elements"],
    )
    graph.add_edge("summarize_tables_text", "clean_elements")
    graph.add_edge("summarize_tables_vlm", "clean_elements")
    graph.add_edge("clean_elements", "resolve_visual_order_outliers")
    graph.add_edge("resolve_visual_order_outliers", "render_markdown")
    graph.add_edge("render_markdown", "render_preview_html")
    graph.add_edge("render_preview_html", "write_outputs")
    graph.add_edge("write_outputs", END)


def build_graph() -> Any:
    """stage-2 StateGraph를 조립하고 compiled graph를 반환한다."""
    preprocess_graph = StateGraph(
        PreprocessState,
        input_schema=PreprocessInputState,
        output_schema=PreprocessOutputState,
    )
    _register_nodes(preprocess_graph)
    _register_edges(preprocess_graph)
    return preprocess_graph.compile()


@lru_cache(maxsize=1)
def get_agent() -> Any:
    """compiled graph를 필요할 때 한 번만 생성해 재사용한다."""
    return build_graph()


class _LazyAgent:
    """기존 `agent` import 호환성을 유지하는 지연 프록시."""

    def __getattr__(self, name: str) -> Any:
        return getattr(get_agent(), name)


agent = _LazyAgent()


def main() -> None:
    """기본 raw.json 입력으로 stage-2 그래프를 한 번 실행한다."""
    graph_input: PreprocessInputState = {
        "raw_json_path": str(DEFAULT_RAW_JSON_PATH),
    }
    response = get_agent().invoke(graph_input)
    print(
        json.dumps(
            {
                "output_paths": response.get("output_paths"),
                "cleaned_element_count": len(response.get("cleaned_elements", [])),
                "logs": response.get("logs", []),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
