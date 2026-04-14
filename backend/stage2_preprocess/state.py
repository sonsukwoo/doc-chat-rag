"""Stage-2 preprocessing state and structured output schemas."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional

from pydantic import BaseModel, Field
from typing_extensions import TypedDict


class FigureReviewResult(BaseModel):
    """Figure VLM 판단 결과."""

    action: Literal["keep", "drop"] = Field(
        description="문서와 관련 있으면 keep, 무관하면 drop."
    )
    summary: Optional[str] = Field(description="검색에 도움이 되는 1~3문장 한국어 요약.")


class TableSummaryResult(BaseModel):
    """Table summary 결과."""

    summary: str = Field(description="검색에 도움이 되는 1~3문장 한국어 요약.")


class TableSummaryRouteResult(BaseModel):
    """Table summary 라우팅 결과."""

    route: Literal["text", "vlm"] = Field(
        description="HTML만으로 충분히 요약 가능하면 text, 이미지까지 봐야 하면 vlm."
    )


class DocumentProfileResult(BaseModel):
    """문서 전체 맥락을 요약한 프로파일."""

    title: str = Field(description="문서의 핵심 제목 또는 대표 제목.")
    document_type: str = Field(
        description="문서 유형을 짧은 한국어로 작성. 예: 기술 문서, 강의 자료, 논문, 웹 문서."
    )
    main_topics: list[str] = Field(
        description="문서의 핵심 주제 3~6개를 한국어 키워드로 작성."
    )
    relevant_visual_types: list[str] = Field(
        description=(
            "문서 이해에 직접 도움이 될 시각자료 유형. "
            "가능하면 flow_chart, screenshot_from_computer, table 같은 짧은 라벨을 사용."
        )
    )
    irrelevant_visual_hints: list[str] = Field(
        description="문맥상 무관할 가능성이 높은 이미지 힌트를 한국어로 작성. 예: 게임 광고 이미지, 웹 배너."
    )


def merge_result_maps(
    current: Optional[dict[int, dict[str, Any]]],
    update: Optional[dict[int, dict[str, Any]]],
) -> dict[int, dict[str, Any]]:
    """병렬 노드가 반환한 id 기반 dict 결과를 하나로 합친다."""
    return {**(current or {}), **(update or {})}


class PreprocessState(TypedDict, total=False):
    """2차 전처리 그래프의 공유 상태."""

    raw_json_path: str  # stage-1 raw JSON 파일 경로
    source_pdf_path: str  # 원본 PDF 파일 경로
    output_dir: str  # cleaned 결과물과 crop 자산을 저장할 문서 폴더
    total_pages: int  # 문서 전체 페이지 수
    elements: list[dict[str, Any]]  # 현재 처리 중인 element 목록
    document_profile: dict[str, Any]  # 문서 전체 주제/유형/관련 visual 힌트 요약
    page_metrics: dict[int, dict[str, float]]  # 페이지별 width / height 정보
    visual_tasks: list[dict[str, Any]]  # crop / VLM 검토 대상 figure·table 작업 목록
    figure_review_ids: list[int]  # VLM 검토가 필요한 figure element id 목록
    table_summary_ids: list[int]  # summary 생성이 필요한 table element id 목록
    cropped_assets: dict[int, dict[str, str]]  # element id별 crop 이미지 상대/절대 경로
    table_summary_inputs: dict[int, dict[str, Any]]  # table summary를 위한 중간 입력 payload
    table_summary_routes: dict[int, str]  # table id별 text/vlm 라우팅 결과
    figure_review_request: dict[str, Any]  # fan-out worker가 개별 figure 검토에 쓰는 입력 payload
    figure_reviews: Annotated[dict[int, dict[str, Any]], merge_result_maps]  # figure id별 keep/drop + summary 결과
    table_summaries: Annotated[dict[int, dict[str, Any]], merge_result_maps]  # table id별 summary 결과
    cleaned_elements: list[dict[str, Any]]  # 최종 keep/drop 반영 후 정리된 element 목록
    ordering_resolution: dict[str, Any]  # bbox 순서 보정 적용 여부와 조정된 element id 목록
    cleaned_markdown: str  # 최종 Markdown 문자열
    preview_html: str  # 검수용 preview HTML 문자열
    output_paths: dict[str, str]  # 저장된 cleaned 결과물 파일 경로 모음
    logs: Annotated[list[str], operator.add]  # 노드 진행 로그 누적
