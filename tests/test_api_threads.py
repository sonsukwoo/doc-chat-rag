from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import create_app


class RoomApiTests(unittest.TestCase):
    def test_post_room_returns_created_room(self):
        app = create_app()
        client = TestClient(app)

        with patch("backend.api.routes.rooms.create_room") as mock_create_room:
            mock_create_room.return_value = {
                "room_id": "room-test-001",
                "room_name": "테스트 방",
                "collection_name": "rag_chat_hybrid_room-test-001",
                "description": None,
                "default_retrieval_mode": "dense",
                "metadata": {"lifecycle_status": "draft"},
                "active_document_ids": [],
                "document_count": 0,
                "created_at": None,
                "updated_at": None,
                "archived_at": None,
            }
            response = client.post(
                "/rooms",
                json={"room_name": "테스트 방"},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["room"]["room_id"], "room-test-001")
        self.assertEqual(payload["room"]["metadata"]["lifecycle_status"], "draft")

    def test_get_rooms_returns_room_list(self):
        app = create_app()
        client = TestClient(app)

        with patch("backend.api.routes.rooms.list_rooms") as mock_list_rooms:
            mock_list_rooms.return_value = [
                {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "description": None,
                    "default_retrieval_mode": "dense",
                    "metadata": {"lifecycle_status": "ready"},
                    "active_document_ids": ["doc-1"],
                    "document_count": 1,
                    "created_at": None,
                    "updated_at": None,
                    "archived_at": None,
                }
            ]
            response = client.get("/rooms")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["rooms"]), 1)
        self.assertEqual(payload["rooms"][0]["room_id"], "room-alpha")

    def test_post_room_with_document_returns_bootstrap_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch(
            "backend.api.routes.rooms.create_room_with_document"
        ) as mock_create_room_with_document:
            mock_create_room_with_document.return_value = {
                "room": {
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
                "document": {
                    "document_id": "room-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {"upload": {"status": "uploaded"}},
                },
                "paths": {"source_pdf_path": "/tmp/guide.pdf"},
            }
            response = client.post(
                "/rooms/with-document",
                data={"room_name": "Alpha"},
                files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["room"]["room_id"], "room-alpha")
        self.assertEqual(payload["document"]["document_id"], "room-alpha__guide")

    def test_get_room_documents_returns_document_list(self):
        app = create_app()
        client = TestClient(app)

        with patch("backend.api.routes.rooms.get_room_detail") as mock_get_room_detail, patch(
            "backend.api.routes.rooms.list_room_document_records"
        ) as mock_list_room_document_records:
            mock_get_room_detail.return_value = {
                "room_id": "room-alpha",
                "room_name": "Alpha",
            }
            mock_list_room_document_records.return_value = [
                {
                    "document_id": "room-alpha__guide",
                    "original_filename": "guide.pdf",
                    "uploaded_at": None,
                    "stages": {"upload": {"status": "uploaded"}},
                    "source_pdf_path": "/tmp/guide.pdf",
                }
            ]
            response = client.get("/rooms/room-alpha/documents")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["documents"]), 1)
        self.assertEqual(payload["documents"][0]["document_id"], "room-alpha__guide")

    def test_post_room_bootstrap_returns_review_ready_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch(
            "backend.api.routes.rooms.bootstrap_room_for_review"
        ) as mock_bootstrap_room_for_review:
            mock_bootstrap_room_for_review.return_value = {
                "room": {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "description": None,
                    "default_retrieval_mode": "dense",
                    "metadata": {"lifecycle_status": "review_pending"},
                    "active_document_ids": ["room-alpha__guide"],
                    "document_count": 1,
                    "created_at": None,
                    "updated_at": None,
                    "archived_at": None,
                },
                "document": {
                    "document_id": "room-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {"stage2": {"status": "completed"}},
                },
                "stage_status": {
                    "upload": "completed",
                    "stage1": "completed",
                    "stage2": "completed",
                    "review": "pending",
                    "stage3": "not_started",
                },
                "review": {
                    "source_url": "/documents/room-alpha__guide/review/source",
                },
                "next_step": "review",
            }
            response = client.post(
                "/rooms/bootstrap",
                data={"room_name": "Alpha"},
                files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["next_step"], "review")
        self.assertEqual(payload["document"]["document_id"], "room-alpha__guide")

    def test_post_room_document_upload_returns_document_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch("backend.api.routes.rooms.upload_document_to_room") as mock_upload:
            mock_upload.return_value = {
                "room": {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_room-alpha",
                    "description": None,
                    "default_retrieval_mode": "dense",
                    "metadata": {"lifecycle_status": "ready"},
                    "active_document_ids": ["room-alpha__guide"],
                    "document_count": 1,
                    "created_at": None,
                    "updated_at": None,
                    "archived_at": None,
                },
                "document": {
                    "document_id": "room-alpha__guide",
                    "original_filename": "guide.pdf",
                    "stages": {"upload": {"status": "uploaded"}},
                },
                "paths": {"source_pdf_path": "/tmp/guide.pdf"},
            }
            response = client.post(
                "/rooms/room-alpha/documents/upload",
                files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["document"]["document_id"], "room-alpha__guide")

    def test_post_room_document_process_upload_returns_review_ready_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch(
            "backend.api.routes.rooms.upload_room_document_for_review"
        ) as mock_upload_room_document_for_review:
            mock_upload_room_document_for_review.return_value = {
                "room": {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                },
                "document": {
                    "document_id": "room-alpha__guide",
                },
                "stage_status": {
                    "stage1": "completed",
                    "stage2": "completed",
                    "review": "pending",
                },
                "review": {
                    "source_url": "/documents/room-alpha__guide/review/source",
                },
                "next_step": "review",
            }
            response = client.post(
                "/rooms/room-alpha/documents/process-upload",
                files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")},
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["next_step"], "review")

    def test_post_room_document_finalize_review_returns_chat_ready_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch(
            "backend.api.routes.rooms.finalize_room_document_review"
        ) as mock_finalize_room_document_review:
            mock_finalize_room_document_review.return_value = {
                "room": {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                },
                "document": {
                    "document_id": "room-alpha__guide",
                },
                "stage_status": {
                    "review": "completed",
                    "stage3": "completed",
                },
                "indexing": {"status": "completed"},
                "next_step": "chat_ready",
            }
            response = client.post(
                "/rooms/room-alpha/documents/room-alpha__guide/finalize-review",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["next_step"], "chat_ready")

    def test_post_room_document_prepare_review_returns_review_ready_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch(
            "backend.api.routes.rooms.prepare_uploaded_room_document_for_review"
        ) as mock_prepare_uploaded_room_document_for_review:
            mock_prepare_uploaded_room_document_for_review.return_value = {
                "room": {
                    "room_id": "room-alpha",
                    "room_name": "Alpha",
                },
                "document": {
                    "document_id": "room-alpha__guide",
                },
                "stage_status": {
                    "stage1": "completed",
                    "stage2": "completed",
                    "review": "pending",
                },
                "next_step": "review",
            }
            response = client.post(
                "/rooms/room-alpha/documents/room-alpha__guide/prepare-review",
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["next_step"], "review")
