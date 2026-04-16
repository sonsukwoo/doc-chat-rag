import json
import tempfile
import unittest
from pathlib import Path

from backend.stage4_retrieval.evaluation import run_stage4_retrieval_evaluation


def _fake_retrieval_runner(inputs, *, persist_manifest=True):
    query = str(inputs["query"])
    doc_id = str(inputs["document_id"])
    retrievals_by_query = {
        ("1", "핵심 기여"): [
            {
                "chunk_id": "text-0004",
                "parent_id": "parent-0003",
                "score": 0.91,
                "chunk_type": "text",
                "section_title": "Abstract",
            },
            {
                "chunk_id": "text-0001",
                "parent_id": "parent-0001",
                "score": 0.33,
                "chunk_type": "text",
                "section_title": None,
            },
        ],
        ("1", "한계"): [
            {
                "chunk_id": "text-0029",
                "parent_id": "parent-0015",
                "score": 0.84,
                "chunk_type": "text",
                "section_title": "7. Conclusions",
            },
            {
                "chunk_id": "text-0030",
                "parent_id": "parent-0016",
                "score": 0.79,
                "chunk_type": "text",
                "section_title": "8. Limitations",
            },
        ],
    }
    if doc_id == "2":
        return {
            "status": "skipped",
            "skip_reason": "missing_qdrant_config",
            "retrievals": [],
        }

    retrievals = retrievals_by_query[(doc_id, query)]
    return {
        "status": "completed",
        "skip_reason": None,
        "retrievals": retrievals,
    }


class Stage4EvaluationTests(unittest.TestCase):
    def test_run_stage4_retrieval_evaluation_smoke(self):
        eval_payload = {
            "documents": [
                {
                    "doc_id": "1",
                    "source_dir": "backend/outputs/1",
                },
                {
                    "doc_id": "2",
                    "source_dir": "backend/outputs/2",
                },
            ],
            "cases": [
                {
                    "case_id": "doc1_q001",
                    "doc_id": "1",
                    "query": "핵심 기여",
                    "query_type": "summary",
                    "difficulty": "easy",
                    "gold_chunk_ids": ["text-0004"],
                    "gold_parent_ids": ["parent-0003"],
                    "notes": "abstract",
                },
                {
                    "case_id": "doc1_q002",
                    "doc_id": "1",
                    "query": "한계",
                    "query_type": "limitation",
                    "difficulty": "easy",
                    "gold_chunk_ids": ["text-0030"],
                    "gold_parent_ids": ["parent-0016"],
                    "notes": "limitations",
                },
                {
                    "case_id": "doc2_q001",
                    "doc_id": "2",
                    "query": "사전 준비",
                    "query_type": "setup",
                    "difficulty": "easy",
                    "gold_chunk_ids": ["text-0020"],
                    "gold_parent_ids": [],
                    "notes": "setup",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            eval_set_path = temp_path / "retrieval_eval.json"
            report_path = temp_path / "retrieval_report.json"
            eval_set_path.write_text(
                json.dumps(eval_payload, ensure_ascii=False, indent=2)
            )

            result = run_stage4_retrieval_evaluation(
                eval_set_path=eval_set_path,
                top_k=5,
                output_path=report_path,
                retrieval_runner=_fake_retrieval_runner,
            )

            self.assertEqual(result["case_count"], 3)
            self.assertEqual(result["completed_case_count"], 2)
            self.assertEqual(result["skipped_case_count"], 1)
            self.assertEqual(result["error_case_count"], 0)
            self.assertAlmostEqual(result["metrics"]["chunk_hit_rate"], 1.0)
            self.assertAlmostEqual(result["metrics"]["chunk_mean_recall"], 1.0)
            self.assertAlmostEqual(result["metrics"]["chunk_mrr"], 0.75)
            self.assertAlmostEqual(result["metrics"]["parent_hit_rate"], 1.0)
            self.assertTrue(report_path.exists())

            first_case = result["cases"][0]
            self.assertEqual(first_case["matched_chunk_ids"], ["text-0004"])
            self.assertTrue(first_case["chunk_hit"])
            self.assertEqual(first_case["chunk_mrr"], 1.0)

            second_case = result["cases"][1]
            self.assertEqual(second_case["matched_chunk_ids"], ["text-0030"])
            self.assertEqual(second_case["chunk_mrr"], 0.5)

            third_case = result["cases"][2]
            self.assertEqual(third_case["status"], "skipped")
            self.assertEqual(third_case["chunk_hit"], None)


if __name__ == "__main__":
    unittest.main()
