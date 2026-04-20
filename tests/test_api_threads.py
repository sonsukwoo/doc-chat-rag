from __future__ import annotations

import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.api.main import create_app


class ThreadApiTests(unittest.TestCase):
    def test_post_thread_returns_created_thread(self):
        app = create_app()
        client = TestClient(app)

        with patch("backend.api.routes.threads.create_thread") as mock_create_thread:
            mock_create_thread.return_value = {
                "thread_id": "thread-test-001",
                "thread_name": "테스트 스레드",
                "collection_name": "rag_chat_hybrid_thread-test-001",
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
                "/threads",
                json={"thread_name": "테스트 스레드"},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["thread"]["thread_id"], "thread-test-001")
        self.assertEqual(payload["thread"]["metadata"]["lifecycle_status"], "draft")

    def test_get_threads_returns_thread_list(self):
        app = create_app()
        client = TestClient(app)

        with patch("backend.api.routes.threads.list_threads") as mock_list_threads:
            mock_list_threads.return_value = [
                {
                    "thread_id": "thread-alpha",
                    "thread_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_thread-alpha",
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
            response = client.get("/threads")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(len(payload["threads"]), 1)
        self.assertEqual(payload["threads"][0]["thread_id"], "thread-alpha")

    def test_post_thread_bootstrap_returns_review_ready_payload(self):
        app = create_app()
        client = TestClient(app)

        with patch(
            "backend.api.routes.threads.bootstrap_thread_for_review"
        ) as mock_bootstrap_thread_for_review:
            mock_bootstrap_thread_for_review.return_value = {
                "thread": {
                    "thread_id": "thread-alpha",
                    "thread_name": "Alpha",
                    "collection_name": "rag_chat_hybrid_thread-alpha",
                    "description": None,
                    "default_retrieval_mode": "dense",
                    "metadata": {"lifecycle_status": "review_pending"},
                    "active_document_ids": ["thread-alpha__guide"],
                    "document_count": 1,
                    "created_at": None,
                    "updated_at": None,
                    "archived_at": None,
                },
                "document": {
                    "document_id": "thread-alpha__guide",
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
                    "source_url": "/documents/thread-alpha__guide/review/source",
                },
                "next_step": "review",
            }
            response = client.post(
                "/threads/bootstrap",
                data={"thread_name": "Alpha"},
                files={"file": ("guide.pdf", b"%PDF-1.7", "application/pdf")},
            )

        self.assertEqual(response.status_code, 201)
        payload = response.json()
        self.assertEqual(payload["next_step"], "review")
        self.assertEqual(payload["document"]["document_id"], "thread-alpha__guide")
