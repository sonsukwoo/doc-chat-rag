"""Stage-2 visual input inspector.

raw.json 기준으로 figure/table에 실제로 들어가는 입력값을 재구성해
디버깅용 JSON으로 저장하고 stdout에도 요약해서 보여준다.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.stage2_preprocess.llm import DEFAULT_RAW_JSON_PATH
from backend.stage2_preprocess.nodes import (
    build_visual_tasks,
    crop_visuals,
    infer_document_profile,
    load_raw_document,
    normalize_elements,
    review_short_text_candidates,
    resolve_captions,
    rule_filter_elements,
)
from backend.stage2_preprocess.utils import (
    clean_render_text,
    collect_neighbor_body_texts,
    safe_write_json,
    safe_write_text,
)


def parse_args() -> argparse.Namespace:
    """CLI 인자를 파싱한다."""
    parser = argparse.ArgumentParser(
        description="figure/table 모델 입력값을 재구성해 확인한다.",
    )
    parser.add_argument(
        "--raw-json",
        default=str(DEFAULT_RAW_JSON_PATH),
        help="대상 raw.json 경로",
    )
    parser.add_argument(
        "--kind",
        choices=["all", "figure", "table"],
        default="all",
        help="출력할 visual 종류",
    )
    parser.add_argument(
        "--page",
        type=int,
        default=None,
        help="특정 페이지로 필터링",
    )
    parser.add_argument(
        "--id",
        dest="element_ids",
        type=int,
        nargs="*",
        default=None,
        help="특정 element id만 출력",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=20,
        help="최대 출력 개수",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="디버그 JSON 출력 경로. 기본값은 raw.json 폴더의 debug_visual_inputs.json",
    )
    return parser.parse_args()


def load_cached_document_profile(raw_json_path: Path) -> dict[str, Any] | None:
    """같은 폴더의 cleaned.json에서 기존 document_profile을 읽는다."""
    cleaned_json_path = raw_json_path.parent / "cleaned.json"
    if not cleaned_json_path.exists():
        return None

    try:
        payload = json.loads(cleaned_json_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    document_profile = payload.get("document_profile")
    return document_profile if isinstance(document_profile, dict) else None


def run_deterministic_stage2(raw_json_path: Path) -> dict[str, Any]:
    """LLM 요약 노드를 제외한 stage-2 전처리 흐름을 순서대로 재구성한다."""
    state: dict[str, Any] = {"raw_json_path": str(raw_json_path)}
    state.update(load_raw_document(state))
    state.update(resolve_captions(state))
    state.update(normalize_elements(state))

    document_profile = load_cached_document_profile(raw_json_path)
    if document_profile is not None:
        state["document_profile"] = document_profile
    else:
        state.update(infer_document_profile(state))

    state.update(rule_filter_elements(state))
    state.update(review_short_text_candidates(state))
    state.update(build_visual_tasks(state))
    state.update(crop_visuals(state))
    return state


def build_visual_debug_rows(
    state: dict[str, Any],
    kind_filter: str,
    page_filter: int | None,
    element_ids: set[int] | None,
    limit: int,
) -> list[dict[str, Any]]:
    """현재 stage-2 로직 기준 figure/table 입력값을 보기 쉽게 정리한다."""
    document_profile = state.get("document_profile") or {}
    elements = state.get("elements", [])
    elements_by_id = {int(element["id"]): element for element in elements}
    cropped_assets = state.get("cropped_assets", {})
    rows: list[dict[str, Any]] = []

    def _append_row(element_id: int, kind: str) -> None:
        element = elements_by_id.get(element_id)
        if not element:
            return

        page = int(element.get("page", 1) or 1)
        if page_filter is not None and page != page_filter:
            return
        if element_ids is not None and element_id not in element_ids:
            return

        asset = cropped_assets.get(element_id) or {}
        caption = clean_render_text(
            element.get("resolved_caption") or element.get("internal_caption_text") or ""
        )
        prev_body_text, next_body_text = collect_neighbor_body_texts(elements, element_id)

        rows.append(
            {
                "kind": kind,
                "element_id": element_id,
                "page": page,
                "resolved_order": element.get("resolved_order", element.get("order")),
                "image_relative_path": asset.get("relative_path"),
                "image_absolute_path": asset.get("absolute_path"),
                "caption": caption or None,
                "prev_body_text": prev_body_text or None,
                "next_body_text": next_body_text or None,
                "document_profile": document_profile,
                "element_text": element.get("text"),
            }
        )

    if kind_filter in {"all", "figure"}:
        for element_id in state.get("figure_review_ids", []):
            _append_row(int(element_id), "figure")

    if kind_filter in {"all", "table"}:
        for element_id in state.get("table_summary_ids", []):
            _append_row(int(element_id), "table")

    rows.sort(key=lambda item: (int(item["page"]), int(item["resolved_order"]), int(item["element_id"])))
    return rows[:limit]


def build_model_prompt(row: dict[str, Any]) -> str:
    """현재 figure/table 노드와 같은 의미의 prompt를 재구성한다."""
    profile_text = json.dumps(row.get("document_profile") or {}, ensure_ascii=False, indent=2)
    local_context_lines: list[str] = []
    if row.get("prev_body_text"):
        local_context_lines.append(f"- previous body text: {row['prev_body_text']}")
    if row.get("next_body_text"):
        local_context_lines.append(f"- next body text: {row['next_body_text']}")
    local_context_block = "\n".join(local_context_lines) or "- 없음"
    caption = row.get("caption") or "없음"

    if row["kind"] == "figure":
        return (
            "이미지를 보고 문서 본문 이해에 직접 도움이 되면 keep, "
            "문맥과 무관한 광고·장식·로고·아이콘이면 drop으로 판단하라. "
            "판단할 때는 아래 document profile에 담긴 문서 주제와 핵심 토픽을 우선 참고하라. "
            "아래 local body context는 이미지 주변의 본문 텍스트로, 보조 힌트로만 참고하라. "
            "keep이면 RAG 검색에 도움이 되는 한국어 요약을 작성하고, drop이면 summary는 null로 반환하라. "
            "이미지 안의 식별 가능한 텍스트, 도표, 그래프는 요약에 반영하고, 보이지 않는 내용은 추측하지 말라.\n\n"
            f"- document profile:\n{profile_text}\n\n"
            f"- caption: {caption}\n"
            f"- local body context:\n{local_context_block}"
        )

    return (
        "표의 구조를 복원하지 말고, 이미지를 보고 RAG 검색에 도움이 되도록 핵심만 짧게 한국어로 요약하라. "
        "판단할 때는 아래 document profile에 담긴 문서 주제와 핵심 토픽을 우선 참고하라. "
        "아래 local body context는 표 주변의 본문 텍스트로, 보조 힌트로만 참고하라. "
        "표 안의 식별 가능한 제목, 열 이름, 비교 축, 주요 수치는 요약에 반영하고, 보이지 않는 내용은 추측하지 말라.\n\n"
        f"- document profile:\n{profile_text}\n\n"
        f"- caption: {caption}\n"
        f"- local body context:\n{local_context_block}"
    )


def format_debug_text(payload: dict[str, Any]) -> str:
    """디버깅용 텍스트 파일 내용을 만든다."""
    lines: list[str] = []
    lines.append(f"raw_json_path: {payload.get('raw_json_path')}")
    lines.append(f"source_pdf_path: {payload.get('source_pdf_path')}")
    lines.append(f"count: {payload.get('count', 0)}")
    lines.append("")

    for row in payload.get("items", []):
        prompt = build_model_prompt(row)
        image_path = row.get("image_relative_path") or "(없음)"
        lines.append("=" * 100)
        lines.append(
            f"[{row['kind']}] id={row['element_id']} page={row['page']} "
            f"resolved_order={row['resolved_order']}"
        )
        lines.append("")
        lines.append("{")
        lines.append("  'messages': [")
        lines.append("    HumanMessage(content=[")
        lines.append("      {")
        lines.append("        'type': 'text',")
        lines.append("        'text':")
        lines.append("        '''")
        lines.extend(prompt.splitlines())
        lines.append("        '''")
        lines.append("      },")
        lines.append("      {")
        lines.append("        'type': 'image_path',")
        lines.append(f"        'path': '{image_path}'")
        lines.append("      }")
        lines.append("    ])")
        lines.append("  ]")
        lines.append("}")
        lines.append("")
        lines.append(f"caption: {row.get('caption') or '(없음)'}")
        lines.append(f"prev_body_text: {row.get('prev_body_text') or '(없음)'}")
        lines.append(f"next_body_text: {row.get('next_body_text') or '(없음)'}")
        lines.append(f"element_text: {row.get('element_text') or '(없음)'}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def print_human_summary(rows: list[dict[str, Any]]) -> None:
    """stdout에 사람이 바로 읽을 수 있는 요약을 출력한다."""
    if not rows:
        print("대상 visual 입력이 없습니다.")
        return

    for row in rows:
        print("=" * 80)
        print(
            f"[{row['kind']}] id={row['element_id']} page={row['page']} "
            f"resolved_order={row['resolved_order']}"
        )
        print(f"image: {row.get('image_relative_path') or '(없음)'}")
        print(f"caption: {row.get('caption') or '(없음)'}")
        print(f"prev_body_text: {row.get('prev_body_text') or '(없음)'}")
        print(f"next_body_text: {row.get('next_body_text') or '(없음)'}")
        print(f"element_text: {row.get('element_text') or '(없음)'}")
        print("document_profile:")
        print(json.dumps(row.get("document_profile") or {}, ensure_ascii=False, indent=2))


def main() -> None:
    """CLI 엔트리포인트."""
    args = parse_args()
    raw_json_path = Path(args.raw_json).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else raw_json_path.parent / "debug_visual_inputs.json"
    )
    text_output_path = output_path.with_suffix(".txt")
    element_ids = set(args.element_ids) if args.element_ids else None

    state = run_deterministic_stage2(raw_json_path)
    rows = build_visual_debug_rows(
        state=state,
        kind_filter=args.kind,
        page_filter=args.page,
        element_ids=element_ids,
        limit=args.limit,
    )

    payload = {
        "raw_json_path": str(raw_json_path),
        "source_pdf_path": state.get("source_pdf_path"),
        "document_profile": state.get("document_profile"),
        "count": len(rows),
        "items": rows,
    }
    safe_write_json(output_path, payload)
    safe_write_text(text_output_path, format_debug_text(payload))

    print(f"debug json: {output_path}")
    print(f"debug text: {text_output_path}")
    print_human_summary(rows)


if __name__ == "__main__":
    main()
