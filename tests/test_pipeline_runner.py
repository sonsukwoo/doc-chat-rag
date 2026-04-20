from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.services.pipeline_runner import run_stage2_for_document, run_stage3_for_document


class PipelineRunnerTests(unittest.TestCase):
    def test_run_stage2_persists_document_profile_snapshot(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cleaned_json_path = root / "stage2" / "cleaned.json"
            source_pdf_path = root / "source" / "original.pdf"
            stage1_raw_json_path = root / "stage1" / "raw.json"
            cleaned_json_path.parent.mkdir(parents=True, exist_ok=True)
            source_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            stage1_raw_json_path.parent.mkdir(parents=True, exist_ok=True)

            cleaned_json_path.write_text(
                json.dumps(
                    {
                        "document_profile": {
                            "title": "랭그래프 실전 가이드",
                            "document_type": "기술 문서",
                            "main_topics": ["랭그래프", "RAG", "체크포인터"],
                        },
                        "elements": [
                            {"category": "heading", "text": "개요"},
                            {"category": "heading", "text": "create_agent"},
                        ],
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            source_pdf_path.write_bytes(b"%PDF-1.7")
            stage1_raw_json_path.write_text("{}", encoding="utf-8")

            fake_paths = type(
                "Paths",
                (),
                {
                    "root": root,
                    "source_pdf": source_pdf_path,
                    "stage1_raw_json": stage1_raw_json_path,
                    "stage2_dir": root / "stage2",
                    "stage2_cleaned_json": cleaned_json_path,
                },
            )()

            class _FakeAgent:
                def invoke(self, payload):
                    return {
                        "output_paths": {
                            "cleaned_json": str(cleaned_json_path),
                            "cleaned_md": str(root / "stage2" / "cleaned.md"),
                        }
                    }

            with patch(
                "backend.services.pipeline_runner.build_document_paths",
                return_value=fake_paths,
            ), patch(
                "backend.services.pipeline_runner.load_document_record",
                return_value={
                    "original_filename": "Guide Original.pdf",
                    "normalized_filename": "guide-normalized.pdf",
                },
            ), patch(
                "backend.services.pipeline_runner.get_agent",
                return_value=_FakeAgent(),
            ), patch(
                "backend.services.pipeline_runner.sync_document_profile_snapshot"
            ) as mock_sync_profile, patch(
                "backend.services.pipeline_runner.update_document_stage_record"
            ) as mock_update_stage:
                run_stage2_for_document("thread-alpha__guide")

        self.assertEqual(
            mock_sync_profile.call_args.kwargs["original_filename"],
            "Guide Original.pdf",
        )
        self.assertEqual(
            mock_sync_profile.call_args.kwargs["normalized_filename"],
            "guide-normalized.pdf",
        )
        self.assertEqual(
            mock_sync_profile.call_args.kwargs["raw_profile"]["title"],
            "랭그래프 실전 가이드",
        )
        self.assertEqual(
            mock_sync_profile.call_args.kwargs["elements"][0]["text"],
            "개요",
        )
        self.assertEqual(mock_sync_profile.call_args.kwargs["source_stage"], "stage2")
        self.assertTrue(mock_update_stage.called)

    def test_run_stage3_syncs_original_and_normalized_filename_metadata(self):
        captured_sync_kwargs: dict[str, object] = {}

        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cleaned_json_path = root / "review" / "reviewed_cleaned.json"
            parents_json_path = root / "stage3" / "parents.json"
            chunks_json_path = root / "stage3" / "chunks.json"
            source_pdf_path = root / "source" / "original.pdf"

            cleaned_json_path.parent.mkdir(parents=True, exist_ok=True)
            parents_json_path.parent.mkdir(parents=True, exist_ok=True)
            source_pdf_path.parent.mkdir(parents=True, exist_ok=True)

            cleaned_json_path.write_text(
                json.dumps(
                    {
                        "document_profile": {
                            "title": "랭그래프 실전 가이드",
                            "document_type": "기술 문서",
                            "main_topics": ["랭그래프", "RAG", "체크포인터"],
                        }
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            parents_json_path.write_text(
                json.dumps({"parents": [{"parent_id": "parent-1", "text": "body"}]}),
                encoding="utf-8",
            )
            chunks_json_path.write_text(
                json.dumps({"chunks": [{"chunk_id": "chunk-1", "text": "chunk"}]}),
                encoding="utf-8",
            )
            source_pdf_path.write_bytes(b"%PDF-1.7")

            fake_paths = type(
                "Paths",
                (),
                {
                    "root": root,
                    "source_pdf": source_pdf_path,
                    "stage3_dir": root / "stage3",
                    "stage3_parents_json": parents_json_path,
                    "stage3_chunks_json": chunks_json_path,
                },
            )()

            with patch(
                "backend.services.pipeline_runner.build_document_paths",
                return_value=fake_paths,
            ), patch(
                "backend.services.pipeline_runner.get_effective_cleaned_json_path",
                return_value=cleaned_json_path,
            ), patch(
                "backend.services.pipeline_runner.load_document_record",
                return_value={
                    "original_filename": "Guide Original.pdf",
                    "normalized_filename": "guide-normalized.pdf",
                },
            ), patch(
                "backend.services.pipeline_runner.run_stage3",
                return_value={
                    "chunking": {"output_paths": {"chunks_json": str(chunks_json_path)}},
                    "indexing": {"output_paths": {"parents_json": str(parents_json_path)}},
                },
            ), patch(
                "backend.services.pipeline_runner.update_document_stage_record"
            ) as mock_update_stage, patch(
                "backend.services.pipeline_runner.sync_document_runtime_metadata"
            ) as mock_sync_metadata:
                mock_sync_metadata.side_effect = lambda **kwargs: captured_sync_kwargs.update(kwargs)

                run_stage3_for_document(
                    "thread-alpha__guide",
                    thread_id="thread-alpha",
                    collection_name="rag_chat_hybrid_thread-alpha",
                )

        self.assertEqual(captured_sync_kwargs["original_filename"], "Guide Original.pdf")
        self.assertEqual(
            captured_sync_kwargs["normalized_filename"],
            "guide-normalized.pdf",
        )
        self.assertEqual(
            captured_sync_kwargs["document_profile"]["title"],
            "랭그래프 실전 가이드",
        )
        self.assertEqual(
            captured_sync_kwargs["document_profile_source_stage"],
            "review",
        )
        self.assertEqual(
            captured_sync_kwargs["document_profile_elements"],
            [],
        )
        self.assertTrue(mock_update_stage.called)

    def test_run_stage3_rolls_back_qdrant_points_when_runtime_sync_fails(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            cleaned_json_path = root / "review" / "reviewed_cleaned.json"
            parents_json_path = root / "stage3" / "parents.json"
            chunks_json_path = root / "stage3" / "chunks.json"
            indexing_manifest_path = root / "stage3" / "indexing.json"
            source_pdf_path = root / "source" / "original.pdf"

            cleaned_json_path.parent.mkdir(parents=True, exist_ok=True)
            parents_json_path.parent.mkdir(parents=True, exist_ok=True)
            source_pdf_path.parent.mkdir(parents=True, exist_ok=True)

            cleaned_json_path.write_text(
                json.dumps({"document_profile": {"title": "테스트 문서"}}),
                encoding="utf-8",
            )
            parents_json_path.write_text(
                json.dumps({"parents": [{"parent_id": "parent-1"}]}),
                encoding="utf-8",
            )
            chunks_json_path.write_text(
                json.dumps({"chunks": [{"chunk_id": "chunk-1", "text": "chunk"}]}),
                encoding="utf-8",
            )
            indexing_manifest_path.write_text(
                json.dumps({"status": "completed"}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            source_pdf_path.write_bytes(b"%PDF-1.7")

            fake_paths = type(
                "Paths",
                (),
                {
                    "root": root,
                    "source_pdf": source_pdf_path,
                    "stage3_dir": root / "stage3",
                    "stage3_parents_json": parents_json_path,
                    "stage3_chunks_json": chunks_json_path,
                    "stage3_indexing_json": indexing_manifest_path,
                },
            )()

            class _FakeQdrantClient:
                def __init__(self, *args, **kwargs):
                    self.delete_calls: list[dict[str, object]] = []

                def delete_points_by_filter(self, **kwargs):
                    self.delete_calls.append(dict(kwargs))

                def close(self):
                    return None

            fake_qdrant_client = _FakeQdrantClient()

            with patch(
                "backend.services.pipeline_runner.build_document_paths",
                return_value=fake_paths,
            ), patch(
                "backend.services.pipeline_runner.get_effective_cleaned_json_path",
                return_value=cleaned_json_path,
            ), patch(
                "backend.services.pipeline_runner.load_document_record",
                return_value={
                    "original_filename": "guide.pdf",
                    "normalized_filename": "guide.pdf",
                },
            ), patch(
                "backend.services.pipeline_runner.run_stage3",
                return_value={
                    "chunking": {"output_paths": {"chunks_json": str(chunks_json_path)}},
                    "indexing": {
                        "output_paths": {"indexing_manifest": str(indexing_manifest_path)}
                    },
                },
            ), patch(
                "backend.services.pipeline_runner.sync_document_runtime_metadata",
                side_effect=RuntimeError("runtime sync failed"),
            ), patch(
                "backend.services.pipeline_runner.update_document_stage_record"
            ) as mock_update_stage, patch(
                "backend.services.pipeline_runner.STAGE3_QDRANT_URL",
                "http://localhost:6333",
            ), patch(
                "backend.services.pipeline_runner.QdrantRestClient",
                return_value=fake_qdrant_client,
            ):
                with self.assertRaisesRegex(RuntimeError, "runtime sync failed"):
                    run_stage3_for_document(
                        "thread-alpha__guide",
                        thread_id="thread-alpha",
                        collection_name="rag_chat_hybrid_thread-alpha",
                    )

            self.assertEqual(len(fake_qdrant_client.delete_calls), 1)
            self.assertEqual(
                fake_qdrant_client.delete_calls[0]["collection_name"],
                "rag_chat_hybrid_thread-alpha",
            )
            self.assertEqual(
                fake_qdrant_client.delete_calls[0]["query_filter"],
                {
                    "must": [
                        {"key": "thread_id", "match": {"value": "thread-alpha"}},
                        {
                            "key": "document_id",
                            "match": {"value": "thread-alpha__guide"},
                        },
                    ]
                },
            )
            self.assertEqual(
                json.loads(indexing_manifest_path.read_text(encoding="utf-8"))["status"],
                "failed",
            )
            self.assertIn(
                "runtime sync failed",
                mock_update_stage.call_args.kwargs["error"],
            )
