import unittest

from backend.common import derive_document_id_from_artifact_path


class DocumentIdTests(unittest.TestCase):
    def test_derives_document_id_from_stage_output_paths(self):
        self.assertEqual(
            derive_document_id_from_artifact_path("/tmp/sample-doc/stage2/cleaned.json"),
            "sample-doc",
        )
        self.assertEqual(
            derive_document_id_from_artifact_path("/tmp/sample-doc/review/reviewed_cleaned.json"),
            "sample-doc",
        )
        self.assertEqual(
            derive_document_id_from_artifact_path("/tmp/sample-doc/stage3/chunks.json"),
            "sample-doc",
        )
        self.assertEqual(
            derive_document_id_from_artifact_path("/tmp/sample-doc/stage4/retrieval.json"),
            "sample-doc",
        )

    def test_derives_document_id_from_flat_artifact_path(self):
        self.assertEqual(
            derive_document_id_from_artifact_path("/tmp/sample-doc/cleaned.json"),
            "sample-doc",
        )


if __name__ == "__main__":
    unittest.main()
