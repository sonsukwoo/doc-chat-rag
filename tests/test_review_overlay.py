import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.document_store.service import build_document_paths
from backend.review_overlay.service import (
    apply_review_overlay,
    build_review_source,
    save_review_decisions,
)


class ReviewOverlayTests(unittest.TestCase):
    def test_save_build_apply_review_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            paths = build_document_paths("doc_test_review", root=root)
            for directory in (
                paths.root,
                paths.source_dir,
                paths.stage1_dir,
                paths.stage2_dir,
                paths.review_dir,
                paths.stage3_dir,
                paths.stage4_dir,
            ):
                directory.mkdir(parents=True, exist_ok=True)

            cleaned_payload = {
                "source_pdf": str(paths.source_pdf),
                "total_pages": 1,
                "document_profile": {"title": "테스트"},
                "ordering_resolution": {"applied": False, "adjusted_ids": [], "rank_gap_threshold": 3},
                "elements": [
                    {
                        "id": 1,
                        "page": 1,
                        "category": "paragraph",
                        "text": "광고가 출력될 자리입니다.",
                        "html": "<p>광고가 출력될 자리입니다.</p>",
                    },
                    {
                        "id": 2,
                        "page": 1,
                        "category": "paragraph",
                        "text": "핵심 본문입니다.",
                        "html": "<p>핵심 본문입니다.</p>",
                    },
                ],
            }
            paths.stage2_cleaned_json.write_text(
                json.dumps(cleaned_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            with patch(
                "backend.review_overlay.service.build_document_paths",
                side_effect=lambda document_id: build_document_paths(document_id, root=root),
            ), patch(
                "backend.review_overlay.service.update_document_stage_record",
                return_value=None,
            ):
                saved = save_review_decisions(
                    "doc_test_review",
                    element_decisions={
                        "2": {"category_override": "heading"},
                    },
                    exact_text_drop=["광고가 출력될 자리입니다."],
                )

                self.assertEqual(saved["exact_text_drop"], ["광고가 출력될 자리입니다."])
                self.assertEqual(
                    saved["element_decisions"]["2"]["category_override"],
                    "heading",
                )

                review_source = build_review_source("doc_test_review")
                self.assertEqual(review_source["counts"]["dropped_elements"], 1)
                element_two = next(
                    element for element in review_source["elements"] if element["id"] == 2
                )
                self.assertEqual(element_two["effective_category"], "heading")

                result = apply_review_overlay("doc_test_review")
                reviewed_payload = json.loads(paths.reviewed_cleaned_json.read_text(encoding="utf-8"))

                self.assertTrue(paths.reviewed_cleaned_json.exists())
                self.assertTrue(paths.reviewed_cleaned_md.exists())
                self.assertTrue(paths.reviewed_preview_html.exists())
                self.assertEqual(result["stats"]["dropped_elements"], 1)
                self.assertEqual(len(reviewed_payload["elements"]), 1)
                self.assertEqual(reviewed_payload["elements"][0]["category"], "heading")
