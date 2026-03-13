import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.app.embeddings import OpenAIEmbeddingClient


class EmbeddingClientTests(unittest.TestCase):
    def test_requires_api_key(self) -> None:
        with self.assertRaises(ValueError):
            OpenAIEmbeddingClient(api_key="", model="text-embedding-3-small")

    @patch("backend.app.embeddings.OpenAI")
    def test_embed_texts(self, openai_mock: Mock) -> None:
        mock_client = Mock()
        mock_client.embeddings.create.return_value = SimpleNamespace(
            data=[
                SimpleNamespace(embedding=[0.1, 0.2]),
                SimpleNamespace(embedding=[0.3, 0.4]),
            ]
        )
        openai_mock.return_value = mock_client

        client = OpenAIEmbeddingClient(api_key="test-key", model="text-embedding-3-small")
        vectors = client.embed_texts(["a", "b"])

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        mock_client.embeddings.create.assert_called_once()

    @patch("backend.app.embeddings.OpenAI")
    def test_embed_texts_empty(self, openai_mock: Mock) -> None:
        client = OpenAIEmbeddingClient(api_key="test-key", model="text-embedding-3-small")
        vectors = client.embed_texts([])
        self.assertEqual(vectors, [])
        openai_mock.return_value.embeddings.create.assert_not_called()


if __name__ == "__main__":
    unittest.main()

