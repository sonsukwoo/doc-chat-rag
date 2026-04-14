import json
import tempfile
import unittest
from pathlib import Path

from backend.stage3_chunking.pipeline import run_stage3_chunking


class _DisabledEmbeddingClient:
    def __init__(self):
        self.enabled = False
        self.last_error = None

    def embed_texts(self, _texts):
        return None


class Stage3ChunkingTests(unittest.TestCase):
    def test_run_stage3_chunking_smoke(self):
        sample_payload = {
            "source_pdf": "/tmp/sample.pdf",
            "elements": [
                {
                    "id": 1,
                    "category": "heading",
                    "page": 1,
                    "order": 1,
                    "text": "1. 소개",
                    "html": "<h1>1. 소개</h1>",
                },
                {
                    "id": 2,
                    "category": "paragraph",
                    "page": 1,
                    "order": 2,
                    "text": "첫 번째 본문 문단입니다.",
                },
                {
                    "id": 3,
                    "category": "caption",
                    "page": 1,
                    "order": 3,
                    "text": "Table 1. 예시 표",
                },
                {
                    "id": 4,
                    "category": "table",
                    "page": 1,
                    "order": 4,
                    "text": "| A | B |\n|---|---|\n| 1 | 2 |",
                    "resolved_caption": "Table 1. 예시 표",
                    "table_summary": "예시 표 요약",
                    "table": {
                        "markdown": "| A | B |\n|---|---|\n| 1 | 2 |",
                    },
                    "image_path": "tables/sample.png",
                },
                {
                    "id": 5,
                    "category": "paragraph",
                    "page": 1,
                    "order": 5,
                    "text": "표 다음에 이어지는 설명 문단입니다.",
                },
                {
                    "id": 6,
                    "category": "figure",
                    "page": 1,
                    "order": 6,
                    "resolved_caption": "Figure 1. 예시 흐름도",
                    "visual_summary": "서비스 흐름을 설명하는 도식이다.",
                    "image_path": "figures/sample.png",
                    "text": "Figure 1. 예시 흐름도",
                },
            ],
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cleaned_json_path = temp_path / "cleaned.json"
            cleaned_json_path.write_text(
                json.dumps(sample_payload, ensure_ascii=False, indent=2)
            )

            result = run_stage3_chunking(
                {
                    "cleaned_json_path": str(cleaned_json_path),
                    "output_dir": str(temp_path),
                },
                embedding_client=_DisabledEmbeddingClient(),
            )

            chunks_json_path = Path(result["output_paths"]["chunks_json"])
            chunks_jsonl_path = Path(result["output_paths"]["chunks_jsonl"])
            chunks_md_path = Path(result["output_paths"]["chunks_md"])

            self.assertTrue(chunks_json_path.exists())
            self.assertTrue(chunks_jsonl_path.exists())
            self.assertTrue(chunks_md_path.exists())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["stats"]["table_chunks"], 1)
            self.assertEqual(result["stats"]["figure_chunks"], 1)

            stored = json.loads(chunks_json_path.read_text())
            chunks = stored["chunks"]
            chunk_types = [chunk["chunk_type"] for chunk in chunks]
            self.assertEqual(chunk_types.count("text"), 2)
            self.assertNotIn("caption", chunk_types)

            text_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "text")
            self.assertNotIn("섹션:", text_chunk["text"])
            self.assertNotIn("이전 문맥:", text_chunk["text"])

            table_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
            self.assertIn("Table 1. 예시 표", table_chunk["text"])
            self.assertIn("예시 표 요약", table_chunk["text"])
            self.assertNotIn("캡션:", table_chunk["text"])

            figure_chunk = next(
                chunk for chunk in chunks if chunk["chunk_type"] == "figure"
            )
            self.assertIn("Figure 1. 예시 흐름도", figure_chunk["text"])
            self.assertIn("서비스 흐름을 설명하는 도식이다.", figure_chunk["text"])
            self.assertNotIn("그림 요약:", figure_chunk["text"])

            preview_text = chunks_md_path.read_text()
            self.assertIn("## 1번 청크", preview_text)
            self.assertIn("-------------", preview_text)
            self.assertIn("```text", preview_text)
            self.assertIn("### 본문", preview_text)

    def test_large_paragraph_is_split_without_semantic(self):
        long_text = " ".join(
            f"이 문장은 청킹 테스트를 위한 예시 문장 {index} 입니다."
            for index in range(1, 120)
        )
        sample_payload = {
            "elements": [
                {
                    "id": 1,
                    "category": "heading",
                    "page": 1,
                    "order": 1,
                    "text": "1. 긴 본문",
                    "html": "<h1>1. 긴 본문</h1>",
                },
                {
                    "id": 2,
                    "category": "paragraph",
                    "page": 1,
                    "order": 2,
                    "text": long_text,
                },
            ]
        }

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            cleaned_json_path = temp_path / "cleaned.json"
            cleaned_json_path.write_text(
                json.dumps(sample_payload, ensure_ascii=False, indent=2)
            )

            result = run_stage3_chunking(
                {
                    "cleaned_json_path": str(cleaned_json_path),
                    "output_dir": str(temp_path),
                },
                embedding_client=_DisabledEmbeddingClient(),
            )

            stored = json.loads(Path(result["output_paths"]["chunks_json"]).read_text())
            text_chunks = [
                chunk for chunk in stored["chunks"] if chunk["chunk_type"] == "text"
            ]
            self.assertGreaterEqual(len(text_chunks), 2)
            self.assertTrue(
                all(chunk["metadata"]["estimated_tokens"] > 0 for chunk in text_chunks)
            )


if __name__ == "__main__":
    unittest.main()
