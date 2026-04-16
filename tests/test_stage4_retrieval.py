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

            result = run_stage4_retrieval(
                {
                    "query": "표 관련 내용을 찾아줘",
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                    "top_k": 5,
                },
                embedding_client=_FakeEmbeddingClient(),
                qdrant_client=_FakeQdrantClient(),
            )

            manifest_path = Path(result["output_paths"]["retrieval_manifest"])

            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["query"], "표 관련 내용을 찾아줘")
            self.assertEqual(result["document_id"], "sample")
            self.assertEqual(result["chunk_count"], 2)
            self.assertEqual(result["parent_count"], 2)
            self.assertEqual(result["top_k"], 5)
            self.assertEqual(result["fetch_k"], 20)
            self.assertEqual(result["fetched_count"], 2)
            self.assertEqual(result["retrieved_count"], 2)
            self.assertEqual(len(result["retrievals"]), 2)
            self.assertTrue(result["document_filter_applied"])
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
