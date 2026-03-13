from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

class OpenAIEmbeddingClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for embedding generation.")
        self._model = model
        self._client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        logger.info("Generating embeddings for %d texts", len(texts))
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [row.embedding for row in response.data]
