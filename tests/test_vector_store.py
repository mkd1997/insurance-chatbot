import unittest
from unittest.mock import Mock, patch

from backend.app.schemas import ChunkRecord, DocumentRecord
from backend.app.vector_store import QdrantChunkStore


class VectorStoreTests(unittest.TestCase):
    def _sample_chunk(self) -> ChunkRecord:
        return ChunkRecord(
            chunk_id="chunk-1",
            doc_id="doc-1",
            text="sample chunk text",
            source_url="https://www.uhcprovider.com/en/policies-protocols/sample.html",
            heading_path="Sample Policy > Coverage",
            token_count=12,
            chunk_index=0,
        )

    def _sample_doc(self) -> DocumentRecord:
        return DocumentRecord(
            doc_id="doc-1",
            source_url="https://www.uhcprovider.com/en/policies-protocols/sample.html",
            title="Sample Policy",
            doc_type="html",
            section="Coverage",
            content_hash="abc123",
        )

    def test_requires_qdrant_url(self) -> None:
        with self.assertRaises(ValueError):
            QdrantChunkStore(
                url="",
                api_key="",
                collection_name="test",
                vector_size=1536,
            )

    @patch("backend.app.vector_store.QdrantClient")
    def test_ensure_collection_creates_and_indexes(self, qdrant_client_mock: Mock) -> None:
        client = Mock()
        client.get_collections.return_value.collections = []
        qdrant_client_mock.return_value = client

        store = QdrantChunkStore(
            url="https://qdrant.example.com",
            api_key="secret",
            collection_name="policy_chunks",
            vector_size=1536,
        )
        store.ensure_collection()

        client.create_collection.assert_called_once()
        self.assertEqual(client.create_payload_index.call_count, 3)

    @patch("backend.app.vector_store.QdrantClient")
    def test_upsert_chunks(self, qdrant_client_mock: Mock) -> None:
        client = Mock()
        client.get_collections.return_value.collections = []
        qdrant_client_mock.return_value = client
        store = QdrantChunkStore(
            url="https://qdrant.example.com",
            api_key="secret",
            collection_name="policy_chunks",
            vector_size=2,
        )

        count = store.upsert_chunks(
            [self._sample_chunk()],
            [[0.11, 0.22]],
            {"doc-1": self._sample_doc()},
        )
        self.assertEqual(count, 1)
        client.upsert.assert_called_once()

    @patch("backend.app.vector_store.QdrantClient")
    def test_upsert_mismatch_raises(self, qdrant_client_mock: Mock) -> None:
        qdrant_client_mock.return_value = Mock()
        store = QdrantChunkStore(
            url="https://qdrant.example.com",
            api_key="secret",
            collection_name="policy_chunks",
            vector_size=2,
        )
        with self.assertRaises(ValueError):
            store.upsert_chunks([self._sample_chunk()], [], {"doc-1": self._sample_doc()})

    @patch("backend.app.vector_store.QdrantClient")
    def test_search_chunks_returns_scored_hits(self, qdrant_client_mock: Mock) -> None:
        client = Mock()
        point = Mock()
        point.id = "chunk-1"
        point.score = 0.88
        point.payload = {
            "chunk_id": "chunk-1",
            "doc_id": "doc-1",
            "text": "sample text",
            "source_url": "https://www.uhcprovider.com/en/policies-protocols/sample.html",
            "title": "Sample Policy",
            "section": "Coverage",
            "heading_path": "Sample Policy > Coverage",
        }
        client.query_points.return_value = Mock(points=[point])
        qdrant_client_mock.return_value = client

        store = QdrantChunkStore(
            url="https://qdrant.example.com",
            api_key="secret",
            collection_name="policy_chunks",
            vector_size=2,
        )
        hits = store.search_chunks([0.1, 0.2], limit=3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].score, 0.88)
        self.assertEqual(hits[0].doc_id, "doc-1")


if __name__ == "__main__":
    unittest.main()
