import unittest

from backend.app_db.config import (
    APP_CHECKPOINT_SCHEMA,
    APP_DATABASE_NAME,
    build_checkpoint_uri,
)
from backend.app_db.ddl import build_schema_ddl


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
        self.assertIn("document_stage_status", ddl_text)
