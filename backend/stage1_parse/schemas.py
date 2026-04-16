"""Stage-1 input/output schemas."""

from __future__ import annotations

from typing import Literal

from typing_extensions import TypedDict


class Stage1Input(TypedDict, total=False):
    """stage1 진입 시 외부에서 전달하는 최소 입력."""

    pdf_path: str
    output_root: str
    output_dir: str
    json_name: str
    copy_source_pdf: bool


class Stage1ProcessResult(TypedDict):
    """PDF 1개를 raw element JSON으로 변환한 결과 메타데이터."""

    status: Literal["success"]
    source_pdf: str
    json_path: str
    asset_dir: str
    copied_pdf_path: str | None
    element_count: int
