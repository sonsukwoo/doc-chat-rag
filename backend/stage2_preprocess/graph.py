"""Stage-2 preprocessing graph wiring and CLI entrypoint."""

from __future__ import annotations

import json
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .llm import DEFAULT_RAW_JSON_PATH
from .nodes import (
    build_visual_tasks,
    clean_elements,
    crop_visuals,
    infer_document_profile,
    load_raw_document,
    normalize_elements,
    render_markdown,
    render_preview_html,
    resolve_captions,
    review_single_figure,
    rule_filter_elements,
    summarize_tables,
    write_outputs,
)
from .state import PreprocessState
from .utils import collect_page_context


def route_figure_reviews(state: PreprocessState) -> Any:
    """crop된 figure 각각을 독립 worker로 보내 병렬 검토한다."""
    elements_by_id = {int(element["id"]): element for element in state["elements"]}
    sends: list[Send] = []

    for element_id in state.get("figure_review_ids", []):
        asset = state.get("cropped_assets", {}).get(element_id)
        element = elements_by_id.get(element_id)
        if not asset or not element:
            continue

        sends.append(
            Send(
                "review_single_figure",
                {
                    "figure_review_request": {
                        "element_id": element_id,
                        "element": element,
                        "absolute_path": asset["absolute_path"],
                        "document_profile": state.get("document_profile", {}),
                        "page_context": collect_page_context(
                            state["elements"],
                            int(element.get("page", 1)),
                        ),
                    }
                },
            )
        )

    return sends or "summarize_tables"


preprocess_graph = StateGraph(PreprocessState)

preprocess_graph.add_node("load_raw_document", load_raw_document)
preprocess_graph.add_node("resolve_captions", resolve_captions)
preprocess_graph.add_node("normalize_elements", normalize_elements)
preprocess_graph.add_node("infer_document_profile", infer_document_profile)
preprocess_graph.add_node("rule_filter_elements", rule_filter_elements)
preprocess_graph.add_node("build_visual_tasks", build_visual_tasks)
preprocess_graph.add_node("crop_visuals", crop_visuals)
preprocess_graph.add_node("review_single_figure", review_single_figure)
preprocess_graph.add_node("summarize_tables", summarize_tables)
preprocess_graph.add_node("clean_elements", clean_elements)
preprocess_graph.add_node("render_markdown", render_markdown)
preprocess_graph.add_node("render_preview_html", render_preview_html)
preprocess_graph.add_node("write_outputs", write_outputs)

preprocess_graph.add_edge(START, "load_raw_document")
preprocess_graph.add_edge("load_raw_document", "resolve_captions")
preprocess_graph.add_edge("resolve_captions", "normalize_elements")
preprocess_graph.add_edge("normalize_elements", "infer_document_profile")
preprocess_graph.add_edge("infer_document_profile", "rule_filter_elements")
preprocess_graph.add_edge("rule_filter_elements", "build_visual_tasks")
preprocess_graph.add_edge("build_visual_tasks", "crop_visuals")
preprocess_graph.add_conditional_edges(
    "crop_visuals",
    route_figure_reviews,
    ["review_single_figure", "summarize_tables"],
)
preprocess_graph.add_edge("review_single_figure", "summarize_tables")
preprocess_graph.add_edge("summarize_tables", "clean_elements")
preprocess_graph.add_edge("clean_elements", "render_markdown")
preprocess_graph.add_edge("render_markdown", "render_preview_html")
preprocess_graph.add_edge("render_preview_html", "write_outputs")
preprocess_graph.add_edge("write_outputs", END)

agent = preprocess_graph.compile()


def main() -> None:
    """기본 raw.json 입력으로 stage-2 그래프를 한 번 실행한다."""
    graph_input: PreprocessState = {
        "raw_json_path": str(DEFAULT_RAW_JSON_PATH),
        "logs": [],
    }
    response = agent.invoke(graph_input)
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
