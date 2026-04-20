from __future__ import annotations

import unittest
from unittest.mock import Mock, sentinel

from backend.stage4_retrieval.service import search_thread_knowledge


def _build_hit(document_id: str, index: int) -> dict[str, object]:
    return {
        "point_id": f"{document_id}-point-{index}",
        "document_id": document_id,
        "chunk_id": f"{document_id}-chunk-{index}",
        "parent_id": f"{document_id}-parent-{index // 2}",
        "score": 1.0 - (index * 0.01),
        "dense_score": 1.0 - (index * 0.01),
        "bm25_score": None,
        "chunk_type": "text",
        "text": f"{document_id} text {index}",
        "section_title": f"{document_id} section {index}",
        "primary_page": index + 1,
        "page_start": index + 1,
        "page_end": index + 1,
        "has_asset": False,
        "asset_kind": None,
        "asset_relative_path": None,
        "caption": None,
    }


class Stage4SearchServiceTests(unittest.TestCase):
    def test_multi_document_search_collects_per_document_candidates_then_global_reranks(
        self,
    ):
        qdrant_client = Mock()
        qdrant_client.close = Mock()

        original_run_scoped_retrieval = (
            __import__(
                "backend.stage4_retrieval.service",
                fromlist=["_run_scoped_retrieval"],
            )._run_scoped_retrieval
        )
        original_apply_global_postprocess_to_hits = (
            __import__(
                "backend.stage4_retrieval.service",
                fromlist=["_apply_global_postprocess_to_hits"],
            )._apply_global_postprocess_to_hits
        )

        captured_candidates: list[dict[str, object]] = []

        def fake_run_scoped_retrieval(**kwargs):
            document_id = kwargs["normalized_document_ids"][0]
            self.assertFalse(kwargs["resolved_enable_rerank"])
            self.assertFalse(kwargs["resolved_enable_mmr"])
            return {
                "query": kwargs["normalized_query"],
                "active_document_ids": [document_id],
                "retrievals": [_build_hit(document_id, index) for index in range(8)],
                "retrieved_count": 8,
                "score_threshold_applied": None,
                "score_fallback_applied": False,
                "rerank_applied": False,
                "rerank_error": None,
                "mmr_applied": False,
            }

        def fake_apply_global_postprocess_to_hits(**kwargs):
            captured_candidates.extend(list(kwargs["retrieval_hits"]))
            self.assertEqual(kwargs["resolved_top_k"], 10)
            self.assertTrue(kwargs["resolved_enable_rerank"])
            return {
                "retrievals": list(kwargs["retrieval_hits"])[:10],
                "rerank_applied": True,
                "rerank_error": None,
                "mmr_applied": False,
            }

        module = __import__(
            "backend.stage4_retrieval.service",
            fromlist=["_run_scoped_retrieval", "_apply_global_postprocess_to_hits"],
        )
        module._run_scoped_retrieval = fake_run_scoped_retrieval
        module._apply_global_postprocess_to_hits = fake_apply_global_postprocess_to_hits
        try:
            result = search_thread_knowledge(
                query="세 문서를 비교해줘",
                thread_id="thread-stage4-multi",
                active_document_ids=["doc-1", "doc-2", "doc-3"],
                collection_name="test-collection",
                retrieval_mode="dense",
                top_k=10,
                fetch_k=24,
                per_document_top_k=8,
                use_per_document_search=True,
                enable_rerank=True,
                embedding_client=sentinel.embedding_client,
                qdrant_client=qdrant_client,
            )
        finally:
            module._run_scoped_retrieval = original_run_scoped_retrieval
            module._apply_global_postprocess_to_hits = (
                original_apply_global_postprocess_to_hits
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["per_document_search_used"])
        self.assertTrue(result["rerank_applied"])
        self.assertEqual(result["retrieved_count"], 10)
        self.assertEqual(len(captured_candidates), 24)
        self.assertEqual(
            [hit["document_id"] for hit in captured_candidates[:3]],
            ["doc-1", "doc-1", "doc-1"],
        )


if __name__ == "__main__":
    unittest.main()
