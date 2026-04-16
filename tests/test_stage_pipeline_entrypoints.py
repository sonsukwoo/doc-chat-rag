import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import fitz

from backend.stage2_preprocess.nodes import load_raw_document
from backend.stage3 import run_stage3


class StagePipelineEntrypointTests(unittest.TestCase):
    def test_load_raw_document_honors_explicit_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            raw_json_path = temp_path / "raw.json"
            source_pdf_path = temp_path / "original.pdf"
            explicit_output_dir = temp_path / "stage2"

            document = fitz.open()
            document.new_page(width=300, height=400)
            document.save(source_pdf_path)
            document.close()

            raw_json_path.write_text(
                json.dumps(
                    {
                        "source_pdf": str(source_pdf_path),
                        "total_pages": 1,
                        "elements": [{"id": 1, "category": "paragraph", "text": "본문"}],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )

            result = load_raw_document(
                {
                    "raw_json_path": str(raw_json_path),
                    "source_pdf_path": str(source_pdf_path),
                    "output_dir": str(explicit_output_dir),
                }
            )

            self.assertEqual(Path(result["output_dir"]).resolve(), explicit_output_dir.resolve())
            self.assertEqual(result["total_pages"], 1)
            self.assertEqual(len(result["elements"]), 1)

    def test_run_stage3_passes_document_id_and_collection_name_to_indexing(self):
        with patch("backend.stage3.run_stage3_chunking") as mock_chunking, patch(
            "backend.stage3.run_stage3_indexing"
        ) as mock_indexing:
            mock_chunking.return_value = {
                "output_paths": {"chunks_json": "/tmp/chunks.json"},
                "output_dir": "/tmp/stage3",
            }
            mock_indexing.return_value = {"status": "completed"}

            run_stage3(
                {
                    "cleaned_json_path": "/tmp/cleaned.json",
                    "output_dir": "/tmp/stage3",
                    "document_id": "doc_test_006",
                    "collection_name": "rag_chat_hybrid",
                }
            )

            mock_indexing.assert_called_once()
            indexing_inputs = mock_indexing.call_args.args[0]
            self.assertEqual(indexing_inputs["document_id"], "doc_test_006")
            self.assertEqual(indexing_inputs["collection_name"], "rag_chat_hybrid")
