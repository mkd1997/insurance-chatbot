import unittest
from dataclasses import dataclass

from backend.app.qa import QAService, REFUSAL_TEXT
from backend.app.schemas import RetrievedChunk


@dataclass
class _FakeEmbeddingClient:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2] for _ in texts]


@dataclass
class _FakeStore:
    hits: list[RetrievedChunk]

    def search_chunks(self, query_vector: list[float], limit: int) -> list[RetrievedChunk]:
        _ = query_vector
        return self.hits[:limit]


@dataclass
class _FakeLLM:
    answer_text: str
    calls: int = 0

    def answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        _ = question
        _ = retrieved_chunks
        self.calls += 1
        return self.answer_text


class QAServiceTests(unittest.TestCase):
    def _hit(self, score: float) -> RetrievedChunk:
        return RetrievedChunk(
            chunk_id="chunk-1",
            doc_id="doc-1",
            score=score,
            text="Coverage applies to eligible treatments with authorization.",
            source_url="https://www.uhcprovider.com/en/policies-protocols/sample.html",
            title="Sample Policy",
            section="Coverage",
            heading_path="Sample Policy > Coverage",
        )

    def test_refuses_when_retrieval_below_threshold(self) -> None:
        llm = _FakeLLM(answer_text="Should not be called")
        service = QAService(
            embedding_client=_FakeEmbeddingClient(),
            store=_FakeStore(hits=[self._hit(0.51)]),
            llm_client=llm,
            score_threshold=0.75,
        )

        response = service.answer_question("What is covered?", top_k=5)

        self.assertTrue(response.is_refusal)
        self.assertEqual(response.answer, REFUSAL_TEXT)
        self.assertEqual(llm.calls, 0)
        self.assertEqual(response.retrieval_debug.top_k_scores, [0.51])

    def test_answers_when_retrieval_above_threshold(self) -> None:
        llm = _FakeLLM(answer_text="Coverage is allowed for medically necessary care.")
        service = QAService(
            embedding_client=_FakeEmbeddingClient(),
            store=_FakeStore(hits=[self._hit(0.89)]),
            llm_client=llm,
            score_threshold=0.75,
        )

        response = service.answer_question("What is covered?", top_k=5)

        self.assertFalse(response.is_refusal)
        self.assertEqual(llm.calls, 1)
        self.assertIn("Coverage is allowed", response.answer)
        self.assertEqual(len(response.citations), 1)
        self.assertEqual(response.citations[0].source_url, self._hit(0.89).source_url)

    def test_forces_refusal_when_no_valid_citations(self) -> None:
        hit = self._hit(0.91)
        hit.source_url = ""
        hit.text = ""
        llm = _FakeLLM(answer_text="Answer text without usable citation payload.")
        service = QAService(
            embedding_client=_FakeEmbeddingClient(),
            store=_FakeStore(hits=[hit]),
            llm_client=llm,
            score_threshold=0.75,
        )

        response = service.answer_question("What is covered?", top_k=5)

        self.assertTrue(response.is_refusal)
        self.assertEqual(response.answer, REFUSAL_TEXT)
        self.assertEqual(response.citations, [])

    def test_deduplicates_citations(self) -> None:
        llm = _FakeLLM(answer_text="Coverage applies with prior authorization.")
        duplicate_1 = self._hit(0.93)
        duplicate_2 = self._hit(0.88)
        service = QAService(
            embedding_client=_FakeEmbeddingClient(),
            store=_FakeStore(hits=[duplicate_1, duplicate_2]),
            llm_client=llm,
            score_threshold=0.75,
        )

        response = service.answer_question("What is covered?", top_k=5)

        self.assertFalse(response.is_refusal)
        self.assertEqual(len(response.citations), 1)


if __name__ == "__main__":
    unittest.main()
