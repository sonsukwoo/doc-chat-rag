from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.services.thread_service import create_thread, get_thread_detail, upload_document_to_thread
from backend.thread_identity import THREAD_COLLECTION_NAME_METADATA_KEY


class _FakeConnection:
    def commit(self) -> None:
        return None


class ThreadServiceTests(unittest.TestCase):
    def test_create_thread_sets_draft_metadata_and_collection(self):
        thread_rows: dict[str, dict] = {}
        thread_documents: dict[str, list[str]] = {}

        class _FakeChatRepository:
            def __init__(self, connection):
                self.connection = connection

            def get_thread(self, thread_id: str):
                return thread_rows.get(thread_id)

            def list_threads(self, *, include_archived: bool = False):
                rows = list(thread_rows.values())
                if not include_archived:
                    rows = [row for row in rows if row.get("archived_at") is None]
                return rows

            def upsert_thread(
                self,
                *,
                thread_id,
                thread_name,
                description=None,
                default_retrieval_mode="dense",
                metadata=None,
                last_user_message_at=None,
            ):
                previous = thread_rows.get(thread_id, {})
                thread_rows[thread_id] = {
                    "thread_id": thread_id,
                    "thread_name": thread_name,
                    "description": description,
                    "default_retrieval_mode": default_retrieval_mode,
                    "metadata": dict(metadata or {}),
                    "created_at": previous.get("created_at")
                    or datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "archived_at": None,
                    "last_user_message_at": last_user_message_at,
                }

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def list_active_document_ids(self, thread_id: str):
                return list(thread_documents.get(thread_id, []))

        @contextmanager
        def _fake_connection():
            yield _FakeConnection()

        with patch("backend.services.thread_service.app_db_connection", _fake_connection), patch(
            "backend.services.thread_service.ChatRepository", _FakeChatRepository
        ), patch(
            "backend.services.thread_service.DocumentRepository", _FakeDocumentRepository
        ):
            created = create_thread(
                thread_name="테스트 스레드",
                description="thread 설명",
                default_retrieval_mode="dense",
            )
            loaded = get_thread_detail(created["thread_id"])

        self.assertIsNotNone(loaded)
        self.assertEqual(created["thread_name"], "테스트 스레드")
        self.assertEqual(created["document_count"], 0)
        self.assertEqual(created["metadata"]["lifecycle_status"], "draft")
        self.assertNotIn(THREAD_COLLECTION_NAME_METADATA_KEY, created["metadata"])
        self.assertEqual(
            thread_rows[created["thread_id"]]["metadata"][THREAD_COLLECTION_NAME_METADATA_KEY],
            created["collection_name"],
        )
        self.assertEqual(loaded["thread_id"], created["thread_id"])

    def test_get_thread_detail_prefers_persisted_collection_name(self):
        persisted_collection_name = "rag_chat_hybrid_thread-persisted-fixed"

        class _FakeChatRepository:
            def __init__(self, connection):
                self.connection = connection

            def get_thread(self, thread_id: str):
                return {
                    "thread_id": thread_id,
                    "thread_name": "Persisted",
                    "description": None,
                    "default_retrieval_mode": "dense",
                    "metadata": {
                        "lifecycle_status": "draft",
                        THREAD_COLLECTION_NAME_METADATA_KEY: persisted_collection_name,
                    },
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "archived_at": None,
                    "last_user_message_at": None,
                }

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def list_active_document_ids(self, thread_id: str):
                return []

        @contextmanager
        def _fake_connection():
            yield _FakeConnection()

        with patch("backend.services.thread_service.app_db_connection", _fake_connection), patch(
            "backend.services.thread_service.ChatRepository", _FakeChatRepository
        ), patch(
            "backend.services.thread_service.DocumentRepository", _FakeDocumentRepository
        ):
            thread = get_thread_detail("thread-persisted")

        self.assertIsNotNone(thread)
        self.assertEqual(thread["collection_name"], persisted_collection_name)
        self.assertNotIn(THREAD_COLLECTION_NAME_METADATA_KEY, thread["metadata"])

    def test_upload_document_to_thread_uses_thread_scoped_document_id(self):
        captured: dict[str, object] = {}

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def upsert_document(self, **kwargs):
                captured["upsert_document"] = kwargs

            def attach_document_to_thread(self, **kwargs):
                captured["attach_document_to_thread"] = kwargs

        @contextmanager
        def _fake_connection():
            yield _FakeConnection()

        with TemporaryDirectory() as temp_dir:
            document_root = Path(temp_dir) / "thread-alpha__guide"
            source_pdf_path = document_root / "source" / "original.pdf"

            with patch(
                "backend.services.thread_service.get_thread_detail",
                side_effect=[
                    {
                        "thread_id": "thread-alpha",
                        "thread_name": "Alpha",
                        "collection_name": "rag_chat_hybrid_thread-alpha",
                        "description": None,
                        "default_retrieval_mode": "dense",
                        "metadata": {"lifecycle_status": "draft"},
                        "active_document_ids": [],
                        "document_count": 0,
                        "created_at": None,
                        "updated_at": None,
                        "archived_at": None,
                    },
                    {
                        "thread_id": "thread-alpha",
                        "thread_name": "Alpha",
                        "collection_name": "rag_chat_hybrid_thread-alpha",
                        "description": None,
                        "default_retrieval_mode": "dense",
                        "metadata": {"lifecycle_status": "draft"},
                        "active_document_ids": ["thread-alpha__guide"],
                        "document_count": 1,
                        "created_at": None,
                        "updated_at": None,
                        "archived_at": None,
                    },
                ],
            ), patch(
                "backend.services.thread_service.build_document_paths"
            ) as mock_build_paths, patch(
                "backend.services.thread_service.create_document_record"
            ) as mock_create_record, patch(
                "backend.services.thread_service.save_uploaded_pdf"
            ) as mock_save_uploaded_pdf, patch(
                "backend.services.thread_service.update_document_stage_record"
            ) as mock_update_stage, patch(
                "backend.services.thread_service.DocumentRepository",
                _FakeDocumentRepository,
            ), patch(
                "backend.services.thread_service.app_db_connection",
                _fake_connection,
            ):
                mock_build_paths.return_value = type(
                    "Paths",
                    (),
                    {"root": document_root, "source_pdf": source_pdf_path},
                )()
                mock_create_record.return_value = {
                    "document_id": "thread-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {},
                }
                mock_save_uploaded_pdf.return_value = source_pdf_path
                mock_update_stage.return_value = {
                    "document_id": "thread-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {"upload": {"status": "uploaded"}},
                }

                result = upload_document_to_thread(
                    thread_id="thread-alpha",
                    original_filename="guide.pdf",
                    content=b"%PDF-1.7",
                )

        self.assertEqual(result["document"]["document_id"], "thread-alpha__guide")
        self.assertEqual(captured["attach_document_to_thread"]["thread_id"], "thread-alpha")
        self.assertEqual(captured["attach_document_to_thread"]["document_id"], "thread-alpha__guide")
        self.assertEqual(captured["upsert_document"]["original_filename"], "guide.pdf")
        self.assertEqual(captured["upsert_document"]["normalized_filename"], "guide.pdf")
        self.assertEqual(mock_create_record.call_args.kwargs["normalized_filename"], "guide.pdf")
