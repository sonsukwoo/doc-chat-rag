import tempfile
import unittest
from pathlib import Path

from backend.document_store import (
    build_document_paths,
    create_document_record,
    get_effective_cleaned_json_path,
    list_document_records,
    load_document_record,
    save_uploaded_pdf,
    update_document_stage_record,
)


class DocumentStoreTests(unittest.TestCase):
    def test_create_document_record_and_paths(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            record = create_document_record(
                original_filename="sample.pdf",
                document_id="doc_test_001",
                root=root,
            )
            paths = build_document_paths("doc_test_001", root=root)

            self.assertEqual(record["document_id"], "doc_test_001")
            self.assertTrue(paths.source_dir.exists())
            self.assertTrue(paths.stage1_dir.exists())
            self.assertTrue(paths.stage2_dir.exists())
            self.assertTrue(paths.review_dir.exists())
            self.assertTrue(paths.stage3_dir.exists())
            self.assertTrue(paths.metadata_json.exists())

            loaded = load_document_record("doc_test_001", root=root)
            self.assertEqual(loaded["original_filename"], "sample.pdf")
            self.assertEqual(loaded["stages"]["stage1"]["status"], "not_started")

    def test_save_uploaded_pdf_and_update_stage_record(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_document_record(
                original_filename="sample.pdf",
                document_id="doc_test_002",
                root=root,
            )
            saved_path = save_uploaded_pdf(
                document_id="doc_test_002",
                content=b"%PDF-1.4\n",
                root=root,
            )

            updated = update_document_stage_record(
                document_id="doc_test_002",
                stage="stage1",
                status="completed",
                outputs={"raw_json_path": "/tmp/raw.json"},
                root=root,
            )

            self.assertTrue(saved_path.exists())
            self.assertEqual(updated["stages"]["stage1"]["status"], "completed")
            self.assertEqual(
                updated["stages"]["stage1"]["outputs"]["raw_json_path"],
                "/tmp/raw.json",
            )

    def test_effective_cleaned_json_prefers_reviewed_overlay(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_document_record(
                original_filename="sample.pdf",
                document_id="doc_test_003",
                root=root,
            )
            paths = build_document_paths("doc_test_003", root=root)
            paths.stage2_cleaned_json.write_text("{}")

            self.assertEqual(
                get_effective_cleaned_json_path(paths),
                paths.stage2_cleaned_json,
            )

            paths.reviewed_cleaned_json.write_text("{}")
            self.assertEqual(
                get_effective_cleaned_json_path(paths),
                paths.reviewed_cleaned_json,
            )

    def test_list_document_records_returns_latest_first(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            create_document_record(
                original_filename="first.pdf",
                document_id="doc_test_004",
                root=root,
            )
            create_document_record(
                original_filename="second.pdf",
                document_id="doc_test_005",
                root=root,
            )

            records = list_document_records(root=root)

            self.assertEqual(len(records), 2)
            self.assertEqual(records[0]["document_id"], "doc_test_005")
