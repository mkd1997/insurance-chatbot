from __future__ import annotations

from .schemas import ChatResponse, Citation, RetrievalDebug

REFUSAL_TEXT = "I don't know based on the provided policy documents."


class QAService:
    def __init__(
        self,
        *,
        embedding_client,
        store,
        llm_client,
        score_threshold: float,
    ) -> None:
        self._embedding_client = embedding_client
        self._store = store
        self._llm_client = llm_client
        self._score_threshold = score_threshold

    def _build_citations(self, hits: list, limit: int = 3) -> list[Citation]:
        citations: list[Citation] = []
        seen_keys: set[tuple[str, str | None]] = set()
        for hit in hits:
            source_url = (hit.source_url or "").strip()
            excerpt = (hit.text or "").strip()
            if not source_url or not excerpt:
                continue
            key = (source_url, hit.section or hit.heading_path)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            citations.append(
                Citation(
                    source_url=source_url,
                    title=hit.title,
                    section=hit.section or hit.heading_path,
                    excerpt=excerpt[:300],
                )
            )
            if len(citations) >= limit:
                break
        return citations

    def answer_question(self, question: str, top_k: int) -> ChatResponse:
        vectors = self._embedding_client.embed_texts([question])
        if not vectors:
            return ChatResponse(
                answer=REFUSAL_TEXT,
                is_refusal=True,
                citations=[],
                retrieval_debug=RetrievalDebug(top_k_scores=[]),
            )

        query_vector = vectors[0]
        hits = self._store.search_chunks(query_vector=query_vector, limit=top_k)
        scores = [hit.score for hit in hits]

        if not hits or max(scores) < self._score_threshold:
            return ChatResponse(
                answer=REFUSAL_TEXT,
                is_refusal=True,
                citations=[],
                retrieval_debug=RetrievalDebug(top_k_scores=scores),
            )

        answer = self._llm_client.answer(question, hits) or REFUSAL_TEXT
        is_refusal = answer.strip() == REFUSAL_TEXT
        citations = self._build_citations(hits) if not is_refusal else []
        if not is_refusal and not citations:
            return ChatResponse(
                answer=REFUSAL_TEXT,
                is_refusal=True,
                citations=[],
                retrieval_debug=RetrievalDebug(top_k_scores=scores),
            )

        return ChatResponse(
            answer=answer,
            is_refusal=is_refusal,
            citations=citations,
            retrieval_debug=RetrievalDebug(top_k_scores=scores),
        )
