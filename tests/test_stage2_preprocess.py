import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langgraph.runtime import Runtime
from langgraph.types import Send

from backend.stage2_preprocess.graph import (
    build_graph,
    route_figure_reviews,
)
from backend.stage2_preprocess.llm import MODEL_RETRY_MAX_ATTEMPTS
from backend.stage2_preprocess.nodes import (
    build_figure_review_requests,
    clean_elements,
    route_table_summaries,
)


class _FakeStructuredModel:
    def __init__(self, results):
        self._results = results

    def batch(self, requests):
        return self._results[: len(requests)]


class _FakeModelFactory:
    def __init__(self, results):
        self._results = results

    def with_structured_output(self, _schema):
        return _FakeStructuredModel(self._results)


class Stage2PreprocessTests(unittest.TestCase):
    def test_graph_compile_smoke_and_retry_policy(self):
        graph = build_graph()

        self.assertIsNotNone(graph)
        for node_name in (
            "review_single_figure",
            "route_table_summaries",
            "summarize_tables_text",
            "summarize_tables_vlm",
        ):
            retry_policies = graph.nodes[node_name].retry_policy
            self.assertTrue(retry_policies)
            self.assertEqual(retry_policies[0].max_attempts, MODEL_RETRY_MAX_ATTEMPTS)

    def test_build_figure_review_requests_and_send_count(self):
        state = {
            "elements": [
                {
                    "id": 1,
                    "category": "paragraph",
                    "page": 1,
                    "order": 1,
                    "text": "문서 소개 문단입니다.",
                },
                {
                    "id": 2,
                    "category": "figure",
                    "page": 1,
                    "order": 2,
                    "text": "",
                },
                {
                    "id": 3,
                    "category": "paragraph",
                    "page": 1,
                    "order": 3,
                    "text": "첫 번째 그림 뒤 설명 문단입니다.",
                },
                {
                    "id": 4,
                    "category": "figure",
                    "page": 1,
                    "order": 4,
                    "text": "",
                },
                {
                    "id": 5,
                    "category": "paragraph",
                    "page": 1,
                    "order": 5,
                    "text": "두 번째 그림 뒤 설명 문단입니다.",
                },
            ],
            "figure_review_ids": [2, 4],
            "cropped_assets": {
                2: {
                    "relative_path": "figures/a.png",
                    "absolute_path": "/tmp/figures/a.png",
                },
                4: {
                    "relative_path": "figures/b.png",
                    "absolute_path": "/tmp/figures/b.png",
                },
            },
            "document_profile": {
                "title": "테스트 문서",
                "document_type": "기술 문서",
                "main_topics": ["파이프라인"],
                "relevant_visual_types": ["diagram"],
                "irrelevant_visual_hints": ["광고 배너"],
            },
        }

        result = build_figure_review_requests(state)
        requests = result["figure_review_requests"]

        self.assertEqual(len(requests), 2)
        self.assertEqual(requests[0]["element_id"], 2)
        self.assertEqual(requests[0]["prev_body_text"], "문서 소개 문단입니다.")
        self.assertEqual(requests[0]["next_body_text"], "첫 번째 그림 뒤 설명 문단입니다.")

        sends = route_figure_reviews({"figure_review_requests": requests})
        self.assertEqual(len(sends), 2)
        self.assertTrue(all(isinstance(send, Send) for send in sends))
        self.assertEqual([send.node for send in sends], ["review_single_figure", "review_single_figure"])

    @patch("backend.stage2_preprocess.nodes.get_text_model")
    def test_route_table_summaries(self, mock_get_text_model):
        mock_get_text_model.return_value = _FakeModelFactory(
            [SimpleNamespace(route="text")]
        )
        state = {
            "document_profile": {
                "title": "테스트 문서",
                "document_type": "기술 문서",
                "main_topics": ["표"],
                "relevant_visual_types": ["table"],
                "irrelevant_visual_hints": ["광고"],
            },
            "table_summary_ids": [10, 20],
            "table_summary_inputs": {
                10: {
                    "asset": None,
                    "caption": "표 1",
                    "html_excerpt": "<table><tr><td>A</td></tr></table>",
                    "text_excerpt": "A",
                    "local_context_block": "- 없음",
                },
                20: {
                    "asset": {
                        "relative_path": "tables/20.png",
                        "absolute_path": "/tmp/tables/20.png",
                    },
                    "caption": "표 2",
                    "html_excerpt": "",
                    "text_excerpt": "fallback",
                    "local_context_block": "- 없음",
                },
            },
            "table_summary_routes": {20: "vlm"},
        }

        result = route_table_summaries(state, Runtime())

        self.assertEqual(result["table_summary_routes"][10], "text")
        self.assertEqual(result["table_summary_routes"][20], "vlm")
        self.assertEqual(result["logs"], ["table_routes:text=1:vlm=1"])

    def test_clean_elements_smoke(self):
        state = {
            "elements": [
                {
                    "id": 1,
                    "category": "caption",
                    "docling_ref": "cap-table",
                    "page": 1,
                    "order": 1,
                    "text": "Table 1",
                },
                {
                    "id": 2,
                    "category": "table",
                    "page": 1,
                    "order": 2,
                    "bbox": [0, 0, 10, 10],
                    "caption_refs": ["cap-table"],
                    "text": "표 본문",
                },
                {
                    "id": 3,
                    "category": "figure",
                    "page": 1,
                    "order": 3,
                    "bbox": [20, 20, 30, 30],
                    "caption_refs": ["cap-fig"],
                    "text": "",
                },
                {
                    "id": 4,
                    "category": "caption",
                    "docling_ref": "cap-fig",
                    "page": 1,
                    "order": 4,
                    "text": "Figure 1",
                },
                {
                    "id": 5,
                    "category": "paragraph",
                    "page": 1,
                    "order": 5,
                    "text": "본문 단락",
                },
            ],
            "figure_reviews": {
                3: {
                    "action": "drop",
                    "summary": None,
                }
            },
            "table_summaries": {
                2: {
                    "summary": "표 요약",
                }
            },
            "cropped_assets": {
                2: {
                    "relative_path": "tables/page_1_table_1.png",
                    "absolute_path": "/tmp/tables/page_1_table_1.png",
                },
                3: {
                    "relative_path": "figures/page_1_figure_1.png",
                    "absolute_path": "/tmp/figures/page_1_figure_1.png",
                },
            },
        }

        result = clean_elements(state)
        cleaned_elements = result["cleaned_elements"]
        cleaned_ids = [element["id"] for element in cleaned_elements]

        self.assertEqual(cleaned_ids, [2, 5])
        self.assertEqual(cleaned_elements[0]["table_summary"], "표 요약")
        self.assertEqual(
            cleaned_elements[0]["image_path"],
            "tables/page_1_table_1.png",
        )


if __name__ == "__main__":
    unittest.main()
