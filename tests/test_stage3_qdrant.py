import unittest

from backend.stage3_indexing.qdrant import QdrantRestClient


class _SchemaAwareQdrantClient(QdrantRestClient):
    def __init__(self, existing_collection):
        self._existing_collection = existing_collection

    def get_collection(self, collection_name: str):
        return self._existing_collection


class Stage3QdrantTests(unittest.TestCase):
    def test_ensure_dense_collection_accepts_matching_schema(self):
        client = _SchemaAwareQdrantClient(
            {
                "result": {
                    "config": {
                        "params": {
                            "vectors": {
                                "size": 3,
                                "distance": "Cosine",
                            }
                        }
                    }
                }
            }
        )

        result = client.ensure_dense_collection(
            collection_name="rag_chat",
            vector_size=3,
            distance="Cosine",
        )

        self.assertFalse(result["created"])

    def test_ensure_dense_collection_rejects_mismatched_schema(self):
        client = _SchemaAwareQdrantClient(
            {
                "result": {
                    "config": {
                        "params": {
                            "vectors": {
                                "size": 1024,
                                "distance": "Dot",
                            }
                        }
                    }
                }
            }
        )

        with self.assertRaises(ValueError) as context:
            client.ensure_dense_collection(
                collection_name="rag_chat",
                vector_size=2560,
                distance="Cosine",
            )

        self.assertIn("schema mismatch", str(context.exception))


if __name__ == "__main__":
    unittest.main()
