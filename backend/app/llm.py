from __future__ import annotations

import logging

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from .schemas import RetrievedChunk

logger = logging.getLogger(__name__)


class OpenAIChatClient:
    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required for answer generation.")
        self._model = model
        self._client = OpenAI(api_key=api_key)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def answer(self, question: str, retrieved_chunks: list[RetrievedChunk]) -> str:
        context_lines: list[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            context_lines.append(
                f"[{idx}] source={chunk.source_url} title={chunk.title or ''} "
                f"section={chunk.section or ''} text={chunk.text}"
            )

        system_prompt = (
            "You answer only from provided policy context. "
            "If context is insufficient, respond exactly: "
            "'I don't know based on the provided policy documents.'"
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Policy context:\n" + "\n".join(context_lines)
        )

        completion = self._client.chat.completions.create(
            model=self._model,
            temperature=0,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        logger.info("Generated answer using %d retrieved chunks", len(retrieved_chunks))
        content = completion.choices[0].message.content if completion.choices else ""
        return (content or "").strip()
