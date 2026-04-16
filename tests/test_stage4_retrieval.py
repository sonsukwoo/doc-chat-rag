import json
import tempfile
import unittest
from pathlib import Path

from backend.stage4_retrieval.pipeline import run_stage4_retrieval


class _FakeEmbeddingClient:
    def __init__(self):
        self.enabled = True
        self.last_error = None

    def embed_texts(self, texts):
        normalized = list(texts)
        return [[float(index + 1), float(len(text)), 0.5] for index, text in enumerate(normalized)]


class _FakeQdrantClient:
    def __init__(self):
        self.calls = []

    def query_points(
        self,
        *,
        collection_name,
        query,
        limit=10,
        with_payload=True,
        with_vector=False,
        using=None,
        prefetch=None,
        query_filter=None,
        score_threshold=None,
    ):
        self.calls.append(
            {
                "collection_name": collection_name,
                "query": query,
                "limit": limit,
                "with_payload": with_payload,
                "with_vector": with_vector,
                "using": using,
                "prefetch": prefetch,
                "query_filter": query_filter,
                "score_threshold": score_threshold,
            }
        )
        return [
            {
                "id": "point-1",
                "score": 0.91,
                "payload": {
                    "document_id": "sample",
                    "chunk_id": "text-0001",
                    "parent_id": "parent-0001",
                    "chunk_type": "text",
                    "text": "첫 번째 청크입니다.",
                    "section_title": "1. 소개",
                    "primary_page": 1,
                    "page_start": 1,
                    "page_end": 1,
                    "has_asset": False,
                },
            },
            {
                "id": "point-2",
                "score": 0.72,
                "payload": {
                    "document_id": "sample",
                    "chunk_id": "table-0001",
                    "parent_id": "parent-0002",
                    "chunk_type": "table",
                    "text": "표 청크입니다.",
                    "section_title": "2. 결과",
                    "primary_page": 2,
                    "page_start": 2,
                    "page_end": 2,
                    "has_asset": True,
                    "asset_kind": "table",
                    "asset_relative_path": "tables/page_2_table_1.png",
                    "caption": "Table 1. 예시 표",
                },
            },
        ]


class Stage4RetrievalTests(unittest.TestCase):
    def test_run_stage4_retrieval_dense_smoke(self):
        chunks_payload = {
            "cleaned_json_path": "/tmp/sample/cleaned.json",
            "chunks": [
                {
                    "chunk_id": "text-0001",
                    "parent_id": "parent-0001",
                    "chunk_type": "text",
                    "text": "첫 번째 청크입니다.",
                },
                {
                    "chunk_id": "table-0001",
                    "parent_id": "parent-0002",
                    "chunk_type": "table",
                    "text": "표 청크입니다.",
                },
            ],
        }
        parents_payload = {
            "document_id": "sample",
            "parents": [
                {
                    "parent_id": "parent-0001",
                    "section_title": "1. 소개",
                    "page_start": 1,
                    "page_end": 1,
                    "child_chunk_ids": ["text-0001"],
                },
                {
                    "parent_id": "parent-0002",
                    "section_title": "2. 결과",
                    "page_start": 2,
                    "page_end": 2,
                    "child_chunk_ids": ["table-0001"],
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_json_path = temp_path / "chunks.json"
            parents_json_path = temp_path / "parents.json"
            chunks_json_path.write_text(
                json.dumps(chunks_payload, ensure_ascii=False, indent=2)
            )
            parents_json_path.write_text(
                json.dumps(parents_payload, ensure_ascii=False, indent=2)
            )

            fake_qdrant = _FakeQdrantClient()
            result = run_stage4_retrieval(
                {
                    "query": "표 관련 내용을 찾아줘",
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                    "top_k": 5,
                    "retrieval_mode": "dense",
                },
                embedding_client=_FakeEmbeddingClient(),
                qdrant_client=fake_qdrant,
            )

            manifest_path = Path(result["output_paths"]["retrieval_manifest"])

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["query"], "표 관련 내용을 찾아줘")
            self.assertEqual(result["document_id"], "sample")
            self.assertEqual(result["chunk_count"], 2)
            self.assertEqual(result["parent_count"], 2)
            self.assertEqual(result["top_k"], 5)
            self.assertEqual(result["fetch_k"], 20)
            self.assertEqual(result["retrieval_mode"], "dense")
            self.assertEqual(result["fetched_count"], 2)
            self.assertEqual(result["retrieved_count"], 2)
            self.assertEqual(len(result["retrievals"]), 2)
            self.assertTrue(result["document_filter_applied"])
            self.assertEqual(len(fake_qdrant.calls), 1)
            self.assertEqual(fake_qdrant.calls[0]["using"], "dense")
            self.assertFalse(manifest_path.exists())
            self.assertEqual(
                Path(result["parents_json_path"]).resolve(),
                parents_json_path.resolve(),
            )

            first_hit = result["retrievals"][0]
            self.assertEqual(first_hit["chunk_id"], "text-0001")
            self.assertEqual(first_hit["parent_id"], "parent-0001")
            self.assertEqual(first_hit["parent_section_title"], "1. 소개")
            self.assertEqual(first_hit["parent_page_start"], 1)
            self.assertEqual(first_hit["parent_page_end"], 1)

            second_hit = result["retrievals"][1]
            self.assertEqual(second_hit["chunk_type"], "table")
            self.assertTrue(second_hit["has_asset"])
            self.assertEqual(second_hit["asset_kind"], "table")
            self.assertEqual(
                second_hit["asset_relative_path"],
                "tables/page_2_table_1.png",
            )
            self.assertEqual(second_hit["caption"], "Table 1. 예시 표")
            self.assertEqual(first_hit["dense_score"], 0.91)
            self.assertIsNone(first_hit["bm25_score"])

    def test_run_stage4_retrieval_hybrid_smoke(self):
        chunks_payload = {
            "cleaned_json_path": "/tmp/sample/cleaned.json",
            "chunks": [
                {
                    "chunk_id": "text-0001",
                    "parent_id": "parent-0001",
                    "chunk_type": "text",
                    "text": "첫 번째 청크입니다.",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_json_path = temp_path / "chunks.json"
            chunks_json_path.write_text(
                json.dumps(chunks_payload, ensure_ascii=False, indent=2)
            )

            fake_qdrant = _FakeQdrantClient()
            result = run_stage4_retrieval(
                {
                    "query": "첫 번째 청크를 찾아줘",
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                    "retrieval_mode": "hybrid",
                    "top_k": 3,
                    "dense_fetch_k": 11,
                    "bm25_fetch_k": 13,
                },
                embedding_client=_FakeEmbeddingClient(),
                qdrant_client=fake_qdrant,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["retrieval_mode"], "hybrid")
            self.assertEqual(result["dense_fetch_k"], 11)
            self.assertEqual(result["bm25_fetch_k"], 13)
            self.assertEqual(result["fetch_k"], 20)
            self.assertIsNone(result["hybrid_rrf_weights"])
            self.assertEqual(
                result["bm25_excluded_role_hints"],
                ["reference_like", "front_matter_like", "title_only"],
            )
            self.assertEqual(len(fake_qdrant.calls), 1)
            first_call = fake_qdrant.calls[0]
            self.assertEqual(first_call["query"], {"fusion": "rrf"})
            self.assertEqual(len(first_call["prefetch"]), 2)
            self.assertEqual(first_call["prefetch"][0]["using"], "dense")
            self.assertEqual(first_call["prefetch"][1]["using"], "bm25")
            self.assertIn("filter", first_call["prefetch"][1])
            self.assertEqual(
                first_call["prefetch"][1]["filter"]["must_not"],
                [
                    {
                        "key": "sparse_role_hints",
                        "match": {"value": "reference_like"},
                    },
                    {
                        "key": "sparse_role_hints",
                        "match": {"value": "front_matter_like"},
                    },
                    {
                        "key": "sparse_role_hints",
                        "match": {"value": "title_only"},
                    },
                ],
            )
            self.assertEqual(
                first_call["prefetch"][1]["query"]["model"],
                "qdrant/bm25",
            )
            self.assertEqual(
                first_call["prefetch"][1]["query"]["options"]["tokenizer"],
                "multilingual",
            )
            self.assertIsNone(result["retrievals"][0]["dense_score"])
            self.assertIsNone(result["retrievals"][0]["bm25_score"])

    def test_run_stage4_retrieval_hybrid_weighted_rrf(self):
        chunks_payload = {
            "cleaned_json_path": "/tmp/sample/cleaned.json",
            "chunks": [
                {
                    "chunk_id": "text-0001",
                    "parent_id": "parent-0001",
                    "chunk_type": "text",
                    "text": "첫 번째 청크입니다.",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_json_path = temp_path / "chunks.json"
            chunks_json_path.write_text(
                json.dumps(chunks_payload, ensure_ascii=False, indent=2)
            )

            fake_qdrant = _FakeQdrantClient()
            result = run_stage4_retrieval(
                {
                    "query": "첫 번째 청크를 찾아줘",
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                    "retrieval_mode": "hybrid",
                    "hybrid_rrf_weights": [3.0, 1.0],
                },
                embedding_client=_FakeEmbeddingClient(),
                qdrant_client=fake_qdrant,
            )

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["hybrid_rrf_weights"], [3.0, 1.0])
            self.assertEqual(len(fake_qdrant.calls), 1)
            self.assertEqual(
                fake_qdrant.calls[0]["query"],
                {"rrf": {"weights": [3.0, 1.0]}},
            )

    def test_run_stage4_retrieval_without_query_is_skipped(self):
        chunks_payload = {
            "cleaned_json_path": "/tmp/sample/cleaned.json",
            "chunks": [],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_json_path = temp_path / "chunks.json"
            chunks_json_path.write_text(
                json.dumps(chunks_payload, ensure_ascii=False, indent=2)
            )

            result = run_stage4_retrieval(
                {
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                }
            )

            self.assertEqual(result["status"], "skipped")
            self.assertEqual(result["skip_reason"], "missing_query")
            self.assertEqual(result["retrievals"], [])

    def test_run_stage4_retrieval_can_persist_manifest_when_requested(self):
        chunks_payload = {
            "cleaned_json_path": "/tmp/sample/cleaned.json",
            "chunks": [
                {
                    "chunk_id": "text-0001",
                    "parent_id": "parent-0001",
                    "chunk_type": "text",
                    "text": "첫 번째 청크입니다.",
                }
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_json_path = temp_path / "chunks.json"
            chunks_json_path.write_text(
                json.dumps(chunks_payload, ensure_ascii=False, indent=2)
            )

            result = run_stage4_retrieval(
                {
                    "query": "소개를 찾아줘",
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                },
                embedding_client=_FakeEmbeddingClient(),
                qdrant_client=_FakeQdrantClient(),
                persist_manifest=True,
            )

            manifest_path = Path(result["output_paths"]["retrieval_manifest"])
            self.assertTrue(manifest_path.exists())

if __name__ == "__main__":
    unittest.main()
