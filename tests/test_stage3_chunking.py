import json
import tempfile
import unittest
from pathlib import Path

from backend.stage3_chunking.pipeline import run_stage3_chunking
from backend.stage2_preprocess.utils import render_table_markdown


class _DisabledEmbeddingClient:
    def __init__(self):
        self.enabled = False
        self.last_error = None

    def embed_texts(self, _texts):
        return None


class Stage3ChunkingTests(unittest.TestCase):
    def test_duplicate_table_caption_is_deduped(self):
        table_element = {
            "id": 1,
            "category": "table",
            "page": 7,
            "order": 1,
            "resolved_caption": "Table 4. One-shot example for generating disease descriptions",
            "table_summary": "질병 설명 생성을 위한 원샷 예시 제공.",
            "table": {
                "markdown": (
                    "Table 4. One-shot example for generating disease descriptions\n\n"
                    "| Korean Original Example | English-Translated Example |\n"
                    "|---|---|\n"
                    "| 예시 | Example |"
                )
            },
            "html": (
                "<table><caption><div class=\"caption\">"
                "Table 4. One-shot example for generating disease descriptions"
                "</div></caption><tbody><tr><td>예시</td></tr></tbody></table>"
            ),
            "text": (
                "Table 4. One-shot example for generating disease descriptions | "
                "Korean Original Example | English-Translated Example |"
            ),
        }

        markdown = render_table_markdown(table_element)
        self.assertEqual(
            markdown.count("Table 4. One-shot example for generating disease descriptions"),
            1,
        )

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
            parents_json_path = Path(result["output_paths"]["parents_json"])

            self.assertTrue(chunks_json_path.exists())
            self.assertTrue(chunks_jsonl_path.exists())
            self.assertTrue(chunks_md_path.exists())
            self.assertTrue(parents_json_path.exists())
            self.assertEqual(result["status"], "completed")
            self.assertEqual(result["stats"]["table_chunks"], 1)
            self.assertEqual(result["stats"]["figure_chunks"], 1)

            stored = json.loads(chunks_json_path.read_text())
            chunks = stored["chunks"]
            parents_document = json.loads(parents_json_path.read_text())
            parents = parents_document["parents"]
            chunk_types = [chunk["chunk_type"] for chunk in chunks]
            self.assertEqual(chunk_types.count("text"), 2)
            self.assertNotIn("caption", chunk_types)
            self.assertTrue(all(chunk.get("parent_id") for chunk in chunks))
            self.assertGreaterEqual(len(parents), 1)

            text_chunks = [chunk for chunk in chunks if chunk["chunk_type"] == "text"]
            first_text_chunk = text_chunks[0]
            second_text_chunk = text_chunks[1]
            self.assertNotIn("섹션:", first_text_chunk["text"])
            self.assertNotIn("이전 문맥:", first_text_chunk["text"])
            self.assertFalse(first_text_chunk["metadata"]["overlap_applied"])
            self.assertFalse(second_text_chunk["metadata"]["overlap_applied"])

            table_chunk = next(chunk for chunk in chunks if chunk["chunk_type"] == "table")
            self.assertIn("Table 1. 예시 표", table_chunk["text"])
            self.assertIn("예시 표 요약", table_chunk["text"])
            self.assertNotIn("캡션:", table_chunk["text"])
            self.assertEqual(table_chunk["text"].count("Table 1. 예시 표"), 1)

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
            self.assertIn("- parent_id: `parent-", preview_text)

            first_parent = parents[0]
            self.assertEqual(first_parent["document_id"], temp_path.name)
            self.assertTrue(first_parent["child_chunk_ids"])
            self.assertIn("text", first_parent["chunk_types"])

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
            self.assertTrue(all(chunk.get("parent_id") for chunk in text_chunks))
            self.assertTrue(
                any(chunk["metadata"]["hard_split_applied"] for chunk in text_chunks)
            )
            self.assertFalse(text_chunks[0]["metadata"]["overlap_applied"])
            self.assertTrue(
                all(
                    chunk["metadata"]["overlap_applied"]
                    for chunk in text_chunks[1:]
                )
            )

    def test_page_continuation_prose_run_is_split_with_overlap(self):
        page1_text = " ".join(
            f"첫 페이지 본문 문장 {index} 입니다."
            for index in range(1, 70)
        )
        page2_text = " ".join(
            f"둘째 페이지 이어지는 본문 문장 {index} 입니다."
            for index in range(70, 140)
        )
        sample_payload = {
            "elements": [
                {
                    "id": 1,
                    "category": "heading",
                    "page": 1,
                    "order": 1,
                    "text": "1. 연속 본문",
                    "html": "<h1>1. 연속 본문</h1>",
                },
                {
                    "id": 2,
                    "category": "paragraph",
                    "page": 1,
                    "order": 2,
                    "text": page1_text,
                },
                {
                    "id": 3,
                    "category": "paragraph",
                    "page": 2,
                    "order": 1,
                    "text": page2_text,
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
            self.assertEqual(text_chunks[0]["pages"], [1])
            self.assertTrue(any(2 in chunk["pages"] for chunk in text_chunks))
            self.assertFalse(text_chunks[0]["metadata"]["overlap_applied"])
            self.assertTrue(
                all(
                    chunk["metadata"]["overlap_applied"]
                    for chunk in text_chunks[1:]
                )
            )

    def test_sparse_role_hints_are_derived_conservatively(self):
        sample_payload = {
            "elements": [
                {
                    "id": 1,
                    "category": "paragraph",
                    "page": 1,
                    "order": 1,
                    "text": "Alice Kim\nSchool of AI, Example University\nalice@example.com",
                },
                {
                    "id": 2,
                    "category": "heading",
                    "page": 4,
                    "order": 2,
                    "text": "References",
                    "html": "<h1>References</h1>",
                },
                {
                    "id": 3,
                    "category": "paragraph",
                    "page": 4,
                    "order": 3,
                    "text": "[1] Smith, J. (2024). Example Paper. https://example.org/paper",
                },
                {
                    "id": 4,
                    "category": "heading",
                    "page": 2,
                    "order": 4,
                    "text": "1. 소개",
                    "html": "<h1>1. 소개</h1>",
                },
                {
                    "id": 5,
                    "category": "paragraph",
                    "page": 2,
                    "order": 5,
                    "text": "이 문서는 검색 품질 개선 방법을 설명한다.",
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
            chunks = stored["chunks"]
            front_matter_chunk = next(
                chunk for chunk in chunks if "alice@example.com" in chunk["text"]
            )
            reference_chunk = next(
                chunk for chunk in chunks if "[1] Smith, J. (2024)." in chunk["text"]
            )
            prose_chunk = next(
                chunk for chunk in chunks if "검색 품질 개선" in chunk["text"]
            )

            self.assertIn(
                "front_matter_like",
                front_matter_chunk["metadata"]["sparse_role_hints"],
            )
            self.assertTrue(front_matter_chunk["metadata"]["has_email"])
            self.assertIn(
                "reference_like",
                reference_chunk["metadata"]["sparse_role_hints"],
            )
            self.assertGreaterEqual(
                reference_chunk["metadata"]["citation_like_count"],
                1,
            )
            self.assertEqual(prose_chunk["metadata"]["sparse_role_hints"], [])


if __name__ == "__main__":
    unittest.main()
