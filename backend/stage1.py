"""Docling 기반 1차 PDF 파서.

이 파일의 역할은 PDF를 바로 최종 문서로 정제하는 것이 아니라,
Docling이 반환한 문서 구조를 서비스용 raw element list로 평탄화하고
그 결과를 JSON으로 저장하는 것이다.

현재 stage-1 파이프라인의 큰 흐름은 아래와 같다.

1. 입력 PDF와 페이지 메타데이터를 수집한다.
2. Docling으로 문서를 파싱해 문서 객체를 만든다.
3. 문서 객체를 순회하며 element 단위 JSON 구조로 정규화한다.
4. 문서별 폴더에 raw JSON과 원본 PDF를 저장한다.

HTML preview, Markdown 생성, visual crop, junk visual 제거는
stage-2 전처리 그래프로 이동한다.
"""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

try:
    import fitz  # PyMuPDF
except ImportError:  # pragma: no cover - runtime dependency
    fitz = None

try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions
    from docling.document_converter import DocumentConverter, PdfFormatOption
except ImportError:  # pragma: no cover - runtime dependency
    InputFormat = None
    PdfPipelineOptions = None
    DocumentConverter = None
    PdfFormatOption = None

# ---------------------------------------------------------------------------
# Runtime configuration
# ---------------------------------------------------------------------------
BACKEND_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = BACKEND_DIR / "outputs"
INPUT_PDF_PATH = Path("/Users/sonseog-u/Downloads/데이터/청킹 문서 예제/2.pdf")

# 결과물은 항상 backend 하위에만 저장하고, 테스트 PDF 경로는 여기서 바꿔 쓴다.
COPY_SOURCE_PDF = True
VISUAL_CATEGORIES = {"figure", "table"}
# 현재 전략: OCR은 끄고 Docling의 구조 파싱 결과를 기준으로 후처리한다.
DO_OCR = False
DO_PICTURE_CLASSIFICATION = True
PICTURE_CLASS_PATTERN = re.compile(
    r"class_name='(?P<label>[^']+)'\s+confidence=(?P<confidence>[-+eE0-9.]+)"
)


# ---------------------------------------------------------------------------
# Small utility helpers
# ---------------------------------------------------------------------------
def sanitize_stem(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-") or "document"


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def clean_render_text(text: str) -> str:
    # 최종 렌더링 시 보기 싫은 placeholder와 주석성 텍스트를 제거한다.
    if not text:
        return ""
    cleaned = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    cleaned = re.sub(
        r"Image not available\.[^.]*?(?:PdfPipelineOptions\([^)]*\))?",
        " ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = cleaned.replace("🖼️❌", " ")
    return normalize_whitespace(cleaned)


def build_picture_candidates(payload: Any) -> List[Dict[str, Any]]:
    """Docling picture classification raw payload를 top-k 후보 리스트로 정리한다."""
    if payload in (None, "", {}, []):
        return []

    predicted_classes = _find_nested_value(payload, ("predicted_classes",))
    candidates: List[Dict[str, Any]] = []

    if isinstance(predicted_classes, list):
        for item in predicted_classes:
            if not isinstance(item, str):
                continue
            matched = PICTURE_CLASS_PATTERN.search(item)
            if not matched:
                continue
            try:
                confidence = float(matched.group("confidence"))
            except ValueError:
                continue
            candidates.append(
                {
                    "label": matched.group("label"),
                    "confidence": confidence,
                }
            )

    if candidates:
        return candidates

    predictions = _find_nested_value(payload, ("predictions",))
    if not isinstance(predictions, list):
        return []

    for item in predictions:
        if not isinstance(item, dict):
            continue
        label = item.get("class_name") or item.get("label") or item.get("name")
        confidence = (
            item.get("confidence") or item.get("score") or item.get("probability")
        )
        if label in (None, "") or confidence in (None, ""):
            continue
        try:
            candidates.append(
                {
                    "label": str(label),
                    "confidence": float(confidence),
                }
            )
        except (TypeError, ValueError):
            continue
    return candidates


def to_jsonable(value: Any, depth: int = 0, max_depth: int = 3) -> Any:
    # Docling 내부 객체를 JSON에 저장할 수 있는 단순 dict/list/scalar 형태로 축약한다.
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if depth >= max_depth:
        return str(value)
    if isinstance(value, dict):
        result = {}
        for key, item in list(value.items())[:10]:
            result[str(key)] = to_jsonable(item, depth + 1, max_depth=max_depth)
        return result
    if isinstance(value, (list, tuple, set)):
        return [
            to_jsonable(item, depth + 1, max_depth=max_depth)
            for item in list(value)[:10]
        ]

    attrs = {}
    for key in (
        "label",
        "name",
        "class_name",
        "type",
        "confidence",
        "score",
        "probability",
    ):
        if hasattr(value, key):
            attrs[key] = to_jsonable(
                getattr(value, key),
                depth + 1,
                max_depth=max_depth,
            )
    if attrs:
        return attrs

    if hasattr(value, "__dict__"):
        result = {}
        for key, item in list(vars(value).items())[:10]:
            if key.startswith("_"):
                continue
            result[str(key)] = to_jsonable(item, depth + 1, max_depth=max_depth)
        if result:
            return result

    return str(value)


def _find_nested_value(payload: Any, keys: Sequence[str]) -> Optional[Any]:
    if payload is None:
        return None

    if isinstance(payload, dict):
        for key in keys:
            if key in payload and payload[key] not in (None, ""):
                return payload[key]
        for item in payload.values():
            found = _find_nested_value(item, keys)
            if found not in (None, ""):
                return found
        return None

    if isinstance(payload, list):
        for item in payload:
            found = _find_nested_value(item, keys)
            if found not in (None, ""):
                return found

    return None


def extract_picture_classification(element: Any) -> Optional[Any]:
    # picture classification이 켜져 있으면 관련 메타데이터를 최대한 안전하게 추출한다.
    candidate_values: List[Any] = []
    seen_names = set()

    meta = getattr(element, "meta", None)
    classification_meta = (
        getattr(meta, "classification", None) if meta is not None else None
    )
    if classification_meta is not None:
        candidate_values.append(classification_meta)
        seen_names.add("classification")

    for attr_name in (
        "picture_classification",
        "classification",
        "predictions",
        "prediction",
        "classifier_output",
    ):
        if not hasattr(element, attr_name):
            continue
        value = getattr(element, attr_name)
        if value is None:
            continue
        seen_names.add(attr_name)
        candidate_values.append(value)

    if hasattr(element, "__dict__"):
        for attr_name, value in vars(element).items():
            lowered = attr_name.lower()
            if attr_name in seen_names:
                continue
            if (
                "class" not in lowered
                and "annot" not in lowered
                and "predict" not in lowered
            ):
                continue
            if value is None:
                continue
            candidate_values.append(value)

    normalized_candidates = []
    for value in candidate_values:
        normalized = to_jsonable(value)
        if normalized in (None, "", {}, []):
            continue
        normalized_candidates.append(normalized)

    if not normalized_candidates:
        return None

    normalized_payload: Any
    if len(normalized_candidates) == 1:
        normalized_payload = normalized_candidates[0]
    else:
        normalized_payload = normalized_candidates
    return normalized_payload


def ensure_within_backend(path: Path) -> Path:
    # 산출물이 실수로 backend 밖으로 나가지 않도록 경로를 강제한다.
    resolved = path.resolve()
    backend_resolved = BACKEND_DIR.resolve()
    try:
        resolved.relative_to(backend_resolved)
    except ValueError as exc:
        raise ValueError(f"Path escapes backend directory: {resolved}") from exc
    return resolved


def safe_mkdir(path: Path) -> Path:
    safe_path = ensure_within_backend(path)
    safe_path.mkdir(parents=True, exist_ok=True)
    return safe_path


def safe_write_json(path: Path, payload: Dict[str, Any]) -> Path:
    safe_path = ensure_within_backend(path)
    safe_path.parent.mkdir(parents=True, exist_ok=True)
    safe_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return safe_path


def safe_call(obj: Any, method_name: str, *args: Any, **kwargs: Any) -> Optional[Any]:
    # Docling 버전마다 메서드 시그니처가 조금 달라서, 가능한 호출 형태를 순차 시도한다.
    if not hasattr(obj, method_name):
        return None

    method = getattr(obj, method_name)
    try:
        return method(*args, **kwargs)
    except TypeError:
        try:
            return method(*args)
        except TypeError:
            try:
                return method()
            except Exception:
                return None
        except Exception:
            return None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Docling metadata extraction helpers
# ---------------------------------------------------------------------------
def get_label_name(element: Any) -> str:
    label = getattr(element, "label", None)
    if label is None:
        return type(element).__name__
    return str(getattr(label, "name", None) or label)


def bbox_to_list(bbox: Any) -> Optional[List[float]]:
    if bbox is None:
        return None
    return [float(bbox.l), float(bbox.t), float(bbox.r), float(bbox.b)]


def coord_origin_name(bbox: Any) -> Optional[str]:
    if bbox is None:
        return None
    origin = getattr(bbox, "coord_origin", None)
    if origin is None:
        return None
    return str(getattr(origin, "name", None) or origin)


def extract_caption_refs(element: Any) -> List[str]:
    # figure/table이 연결하고 있는 caption ref를 미리 추출해 둔다.
    refs = []
    for ref in getattr(element, "captions", []) or []:
        cref = getattr(ref, "cref", None)
        if not cref:
            cref = safe_call(ref, "get_ref")
        if cref:
            refs.append(str(cref))
    return refs


def get_text_content(element: Any, doc: Any = None) -> str:
    # text가 없으면 markdown/text/html export를 순서대로 시도해서 사람이 읽을 텍스트를 확보한다.
    text = getattr(element, "text", None)
    if isinstance(text, str) and text.strip():
        return normalize_whitespace(text)

    for method_name in ("export_to_markdown", "export_to_text", "export_to_html"):
        result = safe_call(element, method_name, doc=doc)
        if isinstance(result, str) and result.strip():
            return normalize_whitespace(result)

        result = safe_call(element, method_name)
        if isinstance(result, str) and result.strip():
            return normalize_whitespace(result)

    return ""


def map_category(raw_type: str) -> str:
    # Docling 원본 label을 우리 쪽 공통 category로 단순화한다.
    key = raw_type.upper().replace(" ", "_")
    mapping = {
        "TITLE": "heading",
        "SECTION_HEADER": "heading",
        "TEXT": "paragraph",
        "BODY_TEXT": "paragraph",
        "PARAGRAPH": "paragraph",
        "LIST_ITEM": "list",
        "BULLET_LIST_ITEM": "list",
        "PICTURE": "figure",
        "FIGURE": "figure",
        "IMAGE": "figure",
        "TABLE": "table",
        "CAPTION": "caption",
        "CODE": "code",
        "FOOTNOTE": "footnote",
        "FORMULA": "formula",
        "PAGE_HEADER": "page_header",
        "PAGE_FOOTER": "page_footer",
    }
    return mapping.get(key, "paragraph")


# ---------------------------------------------------------------------------
# Main parser
# ---------------------------------------------------------------------------
class UpstageStyleDoclingParser:
    """Docling 결과를 raw JSON element로 정규화하는 1차 파서."""

    def __init__(self, output_root: Path = OUTPUT_ROOT) -> None:
        if DocumentConverter is None:
            raise RuntimeError(
                "docling is not installed. Install it before running this script."
            )
        if PdfPipelineOptions is None or InputFormat is None or PdfFormatOption is None:
            raise RuntimeError(
                "Docling PDF pipeline options are unavailable in this environment."
            )
        if fitz is None:
            raise RuntimeError(
                "PyMuPDF (fitz) is not installed. Install it before running this script."
            )

        self.output_root = safe_mkdir(output_root)
        # Docling PDF 파이프라인 설정. 지금은 OCR을 끄고 구조 파싱만 사용한다.
        pipeline_options = PdfPipelineOptions()
        pipeline_options.do_ocr = DO_OCR
        # 그림을 단순 PICTURE로만 두지 않고, 가능하면 chart/logo/diagram 분류 메타데이터도 요청한다.
        if hasattr(pipeline_options, "do_picture_classification"):
            pipeline_options.do_picture_classification = DO_PICTURE_CLASSIFICATION
        self.converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options,
                )
            }
        )

    def process_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """PDF 하나를 처리해 raw JSON과 원본 PDF 사본을 만든다."""
        pdf_path = Path(pdf_path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        stem = sanitize_stem(pdf_path.stem)
        assets_dir = safe_mkdir(self.output_root / stem)

        # 1) 최소 페이지 메타데이터를 확보한다.
        page_metrics = self._collect_page_metrics(pdf_path)

        # 2) Docling 문서 객체를 만든 뒤, 3) 이를 서비스용 raw element list로 정규화한다.
        document = self._convert_document(pdf_path)
        elements = self._collect_elements(document=document)
        ordered_elements = self._preserve_docling_order(elements=elements)

        # 4) stage 1에서는 raw JSON만 저장한다. crop / html preview / markdown은 stage 2로 넘긴다.
        payload = self._build_json_payload(
            pdf_path=pdf_path,
            assets_dir=assets_dir,
            page_metrics=page_metrics,
            elements=ordered_elements,
        )

        json_path = safe_write_json(assets_dir / f"{stem}.json", payload)
        copied_pdf_path = self._copy_source_pdf(pdf_path=pdf_path, stem=stem)

        return {
            "status": "success",
            "source_pdf": str(pdf_path),
            "json_path": str(json_path),
            "asset_dir": str(assets_dir),
            "copied_pdf_path": str(copied_pdf_path) if copied_pdf_path else None,
            "element_count": len(ordered_elements),
        }

    def _collect_page_metrics(self, pdf_path: Path) -> Dict[int, Dict[str, float]]:
        """페이지별 width / height를 수집한다."""
        with fitz.open(str(pdf_path)) as pdf:
            return {
                page_index + 1: {
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                }
                for page_index, page in enumerate(pdf)
            }

    def _convert_document(self, pdf_path: Path) -> Any:
        """입력 PDF를 Docling 문서 객체로 변환한다."""
        conversion = self.converter.convert(str(pdf_path))
        return conversion.document

    def _build_json_payload(
        self,
        pdf_path: Path,
        assets_dir: Path,
        page_metrics: Dict[int, Dict[str, float]],
        elements: Sequence[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """최종 JSON 파일에 저장할 payload를 조립한다."""
        json_elements = [self._element_for_json(element) for element in elements]
        return {
            "source_pdf": str(pdf_path),
            "asset_dir": str(assets_dir),
            "total_pages": len(page_metrics),
            "docling_options": {
                "do_ocr": DO_OCR,
                "do_picture_classification": DO_PICTURE_CLASSIFICATION,
            },
            "elements": json_elements,
        }

    def _copy_source_pdf(self, pdf_path: Path, stem: str) -> Optional[Path]:
        """원본 PDF를 결과 폴더에 함께 보관할지 결정한다."""
        if not COPY_SOURCE_PDF:
            return None

        copied_pdf_path = ensure_within_backend(self.output_root / stem / f"{stem}.pdf")
        copied_pdf_path.parent.mkdir(parents=True, exist_ok=True)
        if pdf_path.resolve() == copied_pdf_path.resolve():
            return copied_pdf_path
        shutil.copy2(pdf_path, copied_pdf_path)
        return copied_pdf_path

    def _element_for_json(self, element: Dict[str, Any]) -> Dict[str, Any]:
        """JSON 저장 직전에 빈/null 보조 필드를 걷어낸다."""
        serialized = dict(element)
        if serialized.get("text") in (None, ""):
            serialized.pop("text", None)
        if serialized.get("html") in (None, ""):
            serialized.pop("html", None)
        if not serialized.get("caption_refs"):
            serialized.pop("caption_refs", None)
        if serialized.get("internal_caption_text") in (None, ""):
            serialized.pop("internal_caption_text", None)
        return serialized

    def _collect_elements(
        self,
        document: Any,
    ) -> List[Dict[str, Any]]:
        """Docling iterate_items() 결과를 서비스용 element list로 평탄화한다."""
        elements: List[Dict[str, Any]] = []

        for element_id, (element, _level) in enumerate(
            document.iterate_items(), start=1
        ):
            raw_type = get_label_name(element)
            category = map_category(raw_type)
            prov = element.prov[0] if getattr(element, "prov", None) else None
            page_no = int(getattr(prov, "page_no", 1)) if prov else 1
            bbox_obj = getattr(prov, "bbox", None) if prov else None
            bbox = bbox_to_list(bbox_obj)
            text = get_text_content(element, doc=document)
            html = self._element_html(
                element=element,
                document=document,
                category=category,
            )
            picture_candidates: List[Dict[str, Any]] = []

            if not text and not html and category not in VISUAL_CATEGORIES:
                continue

            item: Dict[str, Any] = {
                "id": element_id,
                "docling_ref": str(getattr(element, "self_ref", "")) or None,
                "category": category,
                "raw_type": raw_type,
                "page": page_no,
                "bbox": bbox,
                "coord_origin": coord_origin_name(bbox_obj),
                "text": text,
                "html": html,
                "caption_refs": extract_caption_refs(element),
                "internal_caption_text": None,
            }

            if category == "table":
                item["table"] = self._table_payload(element=element, document=document)
            if category in VISUAL_CATEGORIES:
                # figure/table은 별도 caption 블록이 없어도 내부 caption을 가질 수 있으니 따로 보관한다.
                internal_caption = safe_call(element, "caption_text", doc=document)
                if isinstance(internal_caption, str) and internal_caption.strip():
                    item["internal_caption_text"] = clean_render_text(internal_caption)
            if category == "figure":
                picture_classification = extract_picture_classification(element)
                picture_candidates = build_picture_candidates(picture_classification)
                if picture_candidates:
                    item["picture_candidates"] = picture_candidates

            elements.append(item)

        return elements

    def _element_html(self, element: Any, document: Any, category: str) -> str:
        """각 요소를 HTML fragment로 바꾼다."""
        if category == "table":
            html = safe_call(element, "export_to_html", doc=document)
            if isinstance(html, str) and html.strip():
                return html

        html = safe_call(element, "export_to_html", doc=document)
        if isinstance(html, str) and html.strip():
            return html

        text = get_text_content(element, doc=document)
        if not text:
            return ""

        if category == "heading":
            return f"<h2>{text}</h2>"
        if category == "list":
            return f"<ul><li>{text}</li></ul>"
        if category == "code":
            return f"<pre><code>{text}</code></pre>"
        if category == "formula":
            return f"<pre>{text}</pre>"
        if category in {"page_header", "page_footer", "footnote"}:
            return f"<p data-category='{category}'>{text}</p>"
        if category == "caption":
            return f"<p><em>{text}</em></p>"
        return f"<p>{text}</p>"

    def _table_payload(self, element: Any, document: Any) -> Dict[str, Any]:
        # stage 1에서는 top-level html이 이미 있으므로, table 내부엔 markdown만 남긴다.
        return {
            "markdown": safe_call(element, "export_to_markdown", doc=document),
        }

    def _preserve_docling_order(
        self,
        elements: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Docling reading order를 유지한 채, 후속 처리를 위한 order만 부여한다."""
        ordered = list(elements)
        for order_index, element in enumerate(ordered, start=1):
            element["order"] = order_index
        return ordered


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main() -> None:
    parser = UpstageStyleDoclingParser()
    result = parser.process_pdf(INPUT_PDF_PATH)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
