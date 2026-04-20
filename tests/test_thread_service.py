from __future__ import annotations

import unittest
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from backend.services.room_service import (
    archive_room,
    create_room,
    create_room_with_document,
    get_room_detail,
    list_rooms,
    upload_document_to_room,
    update_room,
)
from backend.services.room_pipeline_service import (
    bootstrap_room_for_review,
    finalize_room_document_review,
)


class _FakeConnection:
    def commit(self) -> None:
        return None


class RoomServiceTests(unittest.TestCase):
    def test_create_room_sets_draft_metadata_and_collection(self):
        room_rows: dict[str, dict] = {}
        room_documents: dict[str, list[str]] = {}

        class _FakeChatRepository:
            def __init__(self, connection):
                self.connection = connection

            def get_room(self, room_id: str):
                return room_rows.get(room_id)

            def list_rooms(self, *, include_archived: bool = False):
                rows = list(room_rows.values())
                if not include_archived:
                    rows = [row for row in rows if row.get("archived_at") is None]
                return rows

            def upsert_room(
                self,
                *,
                room_id,
                room_name,
                collection_name,
                description=None,
                default_retrieval_mode="dense",
                metadata=None,
            ):
                previous = room_rows.get(room_id, {})
                room_rows[room_id] = {
                    "room_id": room_id,
                    "room_name": room_name,
                    "collection_name": collection_name,
                    "description": description,
                    "default_retrieval_mode": default_retrieval_mode,
                    "metadata": dict(metadata or {}),
                    "created_at": previous.get("created_at")
                    or datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "archived_at": None,
                }

            def archive_room(self, room_id: str):
                row = room_rows.get(room_id)
                if row is None or row.get("archived_at") is not None:
                    return False
                row["archived_at"] = datetime.now(timezone.utc)
                row["updated_at"] = datetime.now(timezone.utc)
                return True

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def list_active_document_ids(self, room_id: str):
                return list(room_documents.get(room_id, []))

        @contextmanager
        def _fake_connection():
            yield _FakeConnection()

        with patch("backend.services.room_service.app_db_connection", _fake_connection), patch(
            "backend.services.room_service.ChatRepository", _FakeChatRepository
        ), patch(
            "backend.services.room_service.DocumentRepository", _FakeDocumentRepository
        ):
            created = create_room(
                room_name="테스트 방",
                description="room 설명",
                default_retrieval_mode="dense",
            )
            loaded = get_room_detail(created["room_id"])

        self.assertIsNotNone(loaded)
        self.assertEqual(created["room_name"], "테스트 방")
        self.assertEqual(created["document_count"], 0)
        self.assertEqual(created["metadata"]["lifecycle_status"], "draft")
        self.assertTrue(created["collection_name"].startswith("rag_chat_hybrid_"))
        self.assertEqual(loaded["room_id"], created["room_id"])

    def test_update_and_archive_room(self):
        initial_created_at = datetime.now(timezone.utc)
        room_rows = {
            "room-alpha": {
                "room_id": "room-alpha",
                "room_name": "기존 방",
                "collection_name": "rag_chat_hybrid_room-alpha",
                "description": None,
                "default_retrieval_mode": "dense",
                "metadata": {"lifecycle_status": "draft"},
                "created_at": initial_created_at,
                "updated_at": initial_created_at,
                "archived_at": None,
            }
        }
        room_documents = {"room-alpha": ["doc-1", "doc-2"]}

        class _FakeChatRepository:
            def __init__(self, connection):
                self.connection = connection

            def get_room(self, room_id: str):
                return room_rows.get(room_id)

            def list_rooms(self, *, include_archived: bool = False):
                rows = list(room_rows.values())
                if not include_archived:
                    rows = [row for row in rows if row.get("archived_at") is None]
                return rows

            def upsert_room(
                self,
                *,
                room_id,
                room_name,
                collection_name,
                description=None,
                default_retrieval_mode="dense",
                metadata=None,
            ):
                room_rows[room_id] = {
                    **room_rows[room_id],
                    "room_name": room_name,
                    "collection_name": collection_name,
                    "description": description,
                    "default_retrieval_mode": default_retrieval_mode,
                    "metadata": dict(metadata or {}),
                    "updated_at": datetime.now(timezone.utc),
                    "archived_at": None,
                }

            def archive_room(self, room_id: str):
                row = room_rows.get(room_id)
                if row is None:
                    return False
                row["archived_at"] = datetime.now(timezone.utc)
                row["updated_at"] = datetime.now(timezone.utc)
                return True

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def list_active_document_ids(self, room_id: str):
                return list(room_documents.get(room_id, []))

        @contextmanager
        def _fake_connection():
            yield _FakeConnection()

        with patch("backend.services.room_service.app_db_connection", _fake_connection), patch(
            "backend.services.room_service.ChatRepository", _FakeChatRepository
        ), patch(
            "backend.services.room_service.DocumentRepository", _FakeDocumentRepository
        ):
            updated = update_room(
                "room-alpha",
                room_name="수정된 방",
                default_retrieval_mode="hybrid",
                metadata={"owner": "tester"},
            )
            visible_rooms = list_rooms()
            archived = archive_room("room-alpha")
            hidden_after_archive = get_room_detail("room-alpha")
            archived_detail = get_room_detail("room-alpha", include_archived=True)

        self.assertEqual(updated["room_name"], "수정된 방")
        self.assertEqual(updated["default_retrieval_mode"], "hybrid")
        self.assertEqual(updated["document_count"], 2)
        self.assertEqual(updated["metadata"]["owner"], "tester")
        self.assertEqual(len(visible_rooms), 1)
        self.assertIsNotNone(archived)
        self.assertIsNone(hidden_after_archive)
        self.assertIsNotNone(archived_detail)
        self.assertIsNotNone(archived_detail["archived_at"])

    def test_upload_document_to_room_uses_room_scoped_document_id(self):
        captured: dict[str, object] = {}

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def upsert_document(self, **kwargs):
                captured["upsert_document"] = kwargs

            def attach_document_to_room(self, **kwargs):
                captured["attach_document_to_room"] = kwargs

        @contextmanager
        def _fake_connection():
            yield _FakeConnection()

        with TemporaryDirectory() as temp_dir:
            document_root = Path(temp_dir) / "room-alpha__guide"
            source_pdf_path = document_root / "source" / "original.pdf"

            with patch(
                "backend.services.room_service.get_room_detail",
                side_effect=[
                    {
                        "room_id": "room-alpha",
                        "room_name": "Alpha",
                        "collection_name": "rag_chat_hybrid_room-alpha",
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
                        "room_id": "room-alpha",
                        "room_name": "Alpha",
                        "collection_name": "rag_chat_hybrid_room-alpha",
                        "description": None,
                        "default_retrieval_mode": "dense",
                        "metadata": {"lifecycle_status": "draft"},
                        "active_document_ids": ["room-alpha__guide"],
                        "document_count": 1,
                        "created_at": None,
                        "updated_at": None,
                        "archived_at": None,
                    },
                ],
            ), patch(
                "backend.services.room_service.build_document_paths"
            ) as mock_build_paths, patch(
                "backend.services.room_service.create_document_record"
            ) as mock_create_record, patch(
                "backend.services.room_service.save_uploaded_pdf"
            ) as mock_save_uploaded_pdf, patch(
                "backend.services.room_service.update_document_stage_record"
            ) as mock_update_stage, patch(
                "backend.services.room_service.DocumentRepository",
                _FakeDocumentRepository,
            ), patch(
                "backend.services.room_service.app_db_connection",
                _fake_connection,
            ):
                mock_build_paths.return_value = type(
                    "Paths",
                    (),
                    {"root": document_root, "source_pdf": source_pdf_path},
                )()
                mock_create_record.return_value = {
                    "document_id": "room-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {},
                }
                mock_save_uploaded_pdf.return_value = source_pdf_path
                mock_update_stage.return_value = {
                    "document_id": "room-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {"upload": {"status": "uploaded"}},
                }

                result = upload_document_to_room(
                    room_id="room-alpha",
                    original_filename="guide.pdf",
                    content=b"%PDF-1.7",
                )

        self.assertEqual(
            mock_create_record.call_args.kwargs["document_id"],
            "room-alpha__guide",
        )
        self.assertEqual(
            captured["upsert_document"]["document_id"],
            "room-alpha__guide",
        )
        self.assertEqual(
            captured["attach_document_to_room"]["slot_key"],
            "guide.pdf",
        )
        self.assertEqual(result["room"]["document_count"], 1)

    def test_create_room_with_document_archives_room_when_upload_fails(self):
        with patch(
            "backend.services.room_service.create_room",
            return_value={"room_id": "room-alpha"},
        ), patch(
            "backend.services.room_service.upload_document_to_room",
            side_effect=ValueError("only PDF upload is supported"),
        ), patch("backend.services.room_service.archive_room") as mock_archive_room:
            with self.assertRaises(ValueError):
                create_room_with_document(
                    room_name="Alpha",
                    original_filename="broken.pdf",
                    content=b"",
                )

        mock_archive_room.assert_called_once_with("room-alpha")

    def test_bootstrap_room_for_review_runs_stage1_stage2(self):
        with patch(
            "backend.services.room_pipeline_service.create_room_with_document",
            return_value={
                "room": {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "active_document_ids": ["room-alpha__guide"],
                },
                "document": {"document_id": "room-alpha__guide"},
            },
        ), patch(
            "backend.services.room_pipeline_service.run_stage1_for_document"
        ) as mock_stage1, patch(
            "backend.services.room_pipeline_service.run_stage2_for_document"
        ) as mock_stage2, patch(
            "backend.services.room_pipeline_service.update_room"
        ) as mock_update_room, patch(
            "backend.services.room_pipeline_service.get_room_detail",
            side_effect=[
                {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "active_document_ids": ["room-alpha__guide"],
                    "metadata": {"lifecycle_status": "draft"},
                },
                {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "active_document_ids": ["room-alpha__guide"],
                    "metadata": {"lifecycle_status": "draft"},
                },
                {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "active_document_ids": ["room-alpha__guide"],
                    "metadata": {"lifecycle_status": "review_pending"},
                },
            ],
        ), patch(
            "backend.services.room_pipeline_service.load_document_record",
            return_value={
                "document_id": "room-alpha__guide",
                "original_filename": "guide.pdf",
                "stages": {},
            },
        ):
            result = bootstrap_room_for_review(
                room_name="Alpha",
                original_filename="guide.pdf",
                content=b"%PDF-1.7",
            )

        mock_stage1.assert_called_once_with("room-alpha__guide")
        mock_stage2.assert_called_once_with("room-alpha__guide")
        mock_update_room.assert_called_once_with(
            "room-alpha",
            metadata={"lifecycle_status": "review_pending"},
        )
        self.assertEqual(result["next_step"], "review")
        self.assertEqual(result["document"]["document_id"], "room-alpha__guide")
        self.assertEqual(
            result["review"]["source_url"],
            "/documents/room-alpha__guide/review/source",
        )

    def test_finalize_room_document_review_runs_stage3_with_room_collection(self):
        with patch(
            "backend.services.room_pipeline_service.get_room_detail",
            side_effect=[
                {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "active_document_ids": ["room-alpha__guide"],
                    "metadata": {"lifecycle_status": "review_pending"},
                },
                {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "active_document_ids": ["room-alpha__guide"],
                    "metadata": {"lifecycle_status": "ready"},
                },
            ],
        ), patch(
            "backend.services.room_pipeline_service.apply_review_overlay",
            return_value={"stats": {"dropped_elements": 1}},
        ), patch(
            "backend.services.room_pipeline_service.run_stage3_for_document",
            return_value={"status": "completed"},
        ) as mock_stage3, patch(
            "backend.services.room_pipeline_service.update_room"
        ) as mock_update_room, patch(
            "backend.services.room_pipeline_service.load_document_record",
            return_value={
                "document_id": "room-alpha__guide",
                "original_filename": "guide.pdf",
                "stages": {},
            },
        ):
            result = finalize_room_document_review(
                room_id="room-alpha",
                document_id="room-alpha__guide",
            )

        mock_stage3.assert_called_once_with(
            "room-alpha__guide",
            room_id="room-alpha",
            collection_name="rag_chat_hybrid_room-alpha",
        )
        mock_update_room.assert_called_once_with(
            "room-alpha",
            metadata={"lifecycle_status": "ready"},
        )
        self.assertEqual(result["next_step"], "chat_ready")
        self.assertEqual(result["indexing"]["status"], "completed")
