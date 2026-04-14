import json
import tempfile
import unittest
from pathlib import Path

from backend.stage3_indexing.pipeline import run_stage3_indexing


class _FakeEmbeddingClient:
    def __init__(self):
        self.enabled = True
        self.last_error = None

    def embed_texts(self, texts):
        normalized = list(texts)
        return [[float(index + 1), float(len(text)), 1.0] for index, text in enumerate(normalized)]


class _FakeQdrantClient:
    def __init__(self):
        self.collections = []
        self.upsert_batches = []

    def ensure_dense_collection(self, *, collection_name, vector_size, distance):
        self.collections.append(
            {
                "collection_name": collection_name,
                "vector_size": vector_size,
                "distance": distance,
            }
        )
        return {"created": True}

    def upsert_points(self, *, collection_name, points, wait=True):
        self.upsert_batches.append(
            {
                "collection_name": collection_name,
                "points": points,
                "wait": wait,
            }
        )
        return {"status": "ok"}

    def close(self):
        return None


class Stage3IndexingTests(unittest.TestCase):
    def test_run_stage3_indexing_smoke(self):
        sample_payload = {
            "cleaned_json_path": "/tmp/sample/cleaned.json",
            "chunks": [
                {
                    "chunk_id": "text-0001",
                    "chunk_type": "text",
                    "text": "첫 번째 청크 본문입니다.",
                    "pages": [1],
                    "heading_path": ["1. 소개"],
                    "element_ids": [1],
                    "source_elements": [
                        {"element_id": 1, "page": 1, "category": "paragraph"}
                    ],
                    "metadata": {"group_type": "prose"},
                },
                {
                    "chunk_id": "table-0001",
                    "chunk_type": "table",
                    "text": "Table 1. 예시 표\n\n요약 텍스트",
                    "pages": [2, 3],
                    "heading_path": ["2. 실험", "2.1 결과"],
                    "element_ids": [5],
                    "source_elements": [
                        {"element_id": 5, "page": 2, "category": "table"}
                    ],
                    "metadata": {
                        "caption": "Table 1. 예시 표",
                        "image_path": "tables/page_2_table_1.png",
                        "summary_text": "요약 텍스트",
                    },
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            chunks_json_path = temp_path / "chunks.json"
            chunks_json_path.write_text(
                json.dumps(sample_payload, ensure_ascii=False, indent=2)
            )

            fake_qdrant = _FakeQdrantClient()
            result = run_stage3_indexing(
                {
                    "chunks_json_path": str(chunks_json_path),
                    "output_dir": str(temp_path),
                    "document_id": "doc-001",
                    "collection_name": "rag_chat_test",
                },
                embedding_client=_FakeEmbeddingClient(),
                qdrant_client=fake_qdrant,
            )

            manifest_path = Path(result["output_paths"]["indexing_manifest"])

            self.assertEqual(result["status"], "completed")
            self.assertTrue(result["indexing_enabled"])
            self.assertEqual(result["point_count"], 2)
            self.assertEqual(result["vector_size"], 3)
            self.assertTrue(manifest_path.exists())
            self.assertEqual(len(fake_qdrant.collections), 1)
            self.assertEqual(fake_qdrant.collections[0]["distance"], "Cosine")
            self.assertEqual(len(fake_qdrant.upsert_batches), 1)
            self.assertEqual(
                fake_qdrant.upsert_batches[0]["collection_name"],
                "rag_chat_test",
            )
            first_point = fake_qdrant.upsert_batches[0]["points"][0]
            self.assertEqual(first_point["payload"]["document_id"], "doc-001")
            self.assertEqual(first_point["payload"]["chunk_id"], "text-0001")
            self.assertEqual(first_point["payload"]["section_title"], "1. 소개")
            self.assertEqual(first_point["payload"]["primary_page"], 1)
            self.assertFalse(first_point["payload"]["has_asset"])
            self.assertNotIn("source_elements", first_point["payload"])
            self.assertNotIn("metadata", first_point["payload"])

            second_point = fake_qdrant.upsert_batches[0]["points"][1]
            self.assertEqual(second_point["payload"]["chunk_type"], "table")
            self.assertEqual(second_point["payload"]["section_title"], "2. 실험 > 2.1 결과")
            self.assertEqual(second_point["payload"]["primary_page"], 2)
            self.assertEqual(second_point["payload"]["page_start"], 2)
            self.assertEqual(second_point["payload"]["page_end"], 3)
            self.assertTrue(second_point["payload"]["has_asset"])
            self.assertEqual(second_point["payload"]["asset_kind"], "table")
            self.assertEqual(
                second_point["payload"]["asset_relative_path"],
                "tables/page_2_table_1.png",
            )
            self.assertEqual(second_point["payload"]["caption"], "Table 1. 예시 표")


if __name__ == "__main__":
    unittest.main()
