"""Stage-1 Docling parsing runtime configuration."""

from __future__ import annotations

from pathlib import Path


PACKAGE_DIR = Path(__file__).resolve().parent
BACKEND_DIR = PACKAGE_DIR.parent
OUTPUT_ROOT = BACKEND_DIR / "outputs"

# stage1을 단독 실행할 때 사용할 기본 PDF 입력 경로다.
# 이후 상위 통합 파이프라인이 생기면 함수 인자로 넘기는 방식으로 교체할 수 있다.
INPUT_PDF_PATH = Path(
    "/Users/sonseog-u/Downloads/데이터/청킹 문서 예제/1.pdf"
)

# 결과물은 항상 backend 하위에만 저장한다.
COPY_SOURCE_PDF = True
VISUAL_CATEGORIES = {"figure", "table"}

# 현재 전략: OCR은 끄고 Docling의 구조 파싱 결과를 기준으로 후처리한다.
DO_OCR = False
DO_PICTURE_CLASSIFICATION = True
