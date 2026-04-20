import unittest
from contextlib import contextmanager
from unittest.mock import patch

from backend.app_db.config import (
    APP_CHECKPOINT_SCHEMA,
    APP_DATABASE_NAME,
    build_checkpoint_uri,
)
from backend.app_db.ddl import build_schema_ddl
from backend.app_db.services import load_expanded_context_blocks, load_visual_assets


class AppDbTests(unittest.TestCase):
    def test_checkpoint_uri_contains_search_path(self):
        uri = build_checkpoint_uri()
        self.assertIn(APP_DATABASE_NAME, uri)
        self.assertIn("search_path", uri)
        self.assertIn(APP_CHECKPOINT_SCHEMA, uri)

    def test_schema_ddl_contains_core_tables(self):
        ddl_text = "\n".join(build_schema_ddl())
        self.assertIn("rooms", ddl_text)
        self.assertIn("threads", ddl_text)
        self.assertIn("documents", ddl_text)
        self.assertIn("document_parents", ddl_text)
        self.assertIn("document_chunks", ddl_text)
        self.assertIn("document_stage_status", ddl_text)

    def test_load_expanded_context_blocks_uses_parent_rows(self):
        parent_rows = [
            {
                "parent_id": "parent-0001",
                "document_id": "doc-1",
                "section_title": "1. 소개",
                "page_start": 1,
                "page_end": 2,
                "heading_path": ["1. 소개"],
                "chunk_ids": ["text-0001", "text-0002", "text-0003"],
                "body_text": "문맥 전체",
                "metadata": {},
            }
        ]
        chunk_rows = [
            {
                "document_id": "doc-1",
                "chunk_id": "text-0001",
                "parent_id": "parent-0001",
                "chunk_index": 1,
                "text": "이전 문맥입니다.",
            },
            {
                "document_id": "doc-1",
                "chunk_id": "text-0002",
                "parent_id": "parent-0001",
                "chunk_index": 2,
                "text": "핵심 본문입니다.",
            },
            {
                "document_id": "doc-1",
                "chunk_id": "text-0003",
                "parent_id": "parent-0001",
                "chunk_index": 3,
                "text": "다음 문맥입니다.",
            },
        ]

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def list_document_parents(self, document_ids):
                return parent_rows

            def list_document_chunks(self, document_ids):
                return chunk_rows

        @contextmanager
        def _fake_connection():
            yield object()

        with patch("backend.app_db.services.app_db_connection", _fake_connection), patch(
            "backend.app_db.services.DocumentRepository",
            _FakeDocumentRepository,
        ):
            blocks = load_expanded_context_blocks(
                room_id="room-1",
                active_document_ids=["doc-1"],
                chunk_ids=["text-0002"],
                window_size=1,
            )

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0]["parent_id"], "parent-0001")
        self.assertEqual(blocks[0]["matched_chunk_ids"], ["text-0002"])
        self.assertEqual(blocks[0]["window_chunk_ids"], ["text-0001", "text-0002", "text-0003"])
        self.assertEqual(
            blocks[0]["context_text"],
            "이전 문맥입니다.\n\n핵심 본문입니다.\n\n다음 문맥입니다.",
        )
        self.assertEqual(blocks[0]["expansion_mode"], "postgres_window")

    def test_load_visual_assets_filters_by_asset_ref(self):
        asset_rows = [
            {
                "asset_ref": "doc-1:figure-0001",
                "document_id": "doc-1",
                "chunk_id": "figure-0001",
                "asset_kind": "figure",
                "relative_path": "figures/page_1_figure_1.png",
                "page": 1,
                "caption": "테스트 그림",
                "summary_text": "요약",
                "metadata": {"heading_path": ["1. 소개"], "pages": [1]},
            },
            {
                "asset_ref": "doc-1:table-0001",
                "document_id": "doc-1",
                "chunk_id": "table-0001",
                "asset_kind": "table",
                "relative_path": "tables/page_2_table_1.png",
                "page": 2,
                "caption": "테스트 표",
                "summary_text": "표 요약",
                "metadata": {"heading_path": ["2. 결과"], "pages": [2]},
            },
        ]

        class _FakeDocumentRepository:
            def __init__(self, connection):
                self.connection = connection

            def list_document_assets(self, document_ids):
                return asset_rows

        @contextmanager
        def _fake_connection():
            yield object()

        with patch("backend.app_db.services.app_db_connection", _fake_connection), patch(
            "backend.app_db.services.DocumentRepository",
            _FakeDocumentRepository,
        ):
            assets = load_visual_assets(
                room_id="room-1",
                active_document_ids=["doc-1"],
                asset_refs=["doc-1:figure-0001"],
            )

        self.assertEqual(len(assets), 1)
        self.assertEqual(assets[0]["asset_ref"], "doc-1:figure-0001")
        self.assertEqual(assets[0]["asset_stage"], "stage2")
        self.assertEqual(assets[0]["heading_path"], ["1. 소개"])
