from __future__ import annotations

import hashlib
import re

import tiktoken

from .extractor import ExtractedDocument
from .schemas import ChunkRecord

ENCODING_NAME = "cl100k_base"
_ENCODER = None
_ENCODER_UNAVAILABLE = False


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _estimate_tokens_without_encoder(text: str) -> int:
    # Offline-safe fallback token approximation.
    if not text.strip():
        return 0
    return len(re.findall(r"\S+", text))


def _encoding():
    global _ENCODER, _ENCODER_UNAVAILABLE
    if _ENCODER_UNAVAILABLE:
        return None
    if _ENCODER is not None:
        return _ENCODER
    try:
        _ENCODER = tiktoken.get_encoding(ENCODING_NAME)
        return _ENCODER
    except Exception:
        _ENCODER_UNAVAILABLE = True
        return None


def token_count(text: str) -> int:
    encoder = _encoding()
    if encoder is None:
        return _estimate_tokens_without_encoder(text)
    return len(encoder.encode(text))


def build_doc_id(doc: ExtractedDocument) -> str:
    raw = f"{doc.source_url}|{doc.doc_type}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


def build_chunk_id(doc_id: str, chunk_text: str, chunk_index: int) -> str:
    raw = f"{doc_id}|{chunk_index}|{chunk_text}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _split_sentences(text: str) -> list[str]:
    normalized = _normalize_whitespace(text)
    if not normalized:
        return []
    parts = re.split(r"(?<=[.!?])\s+", normalized)
    return [part.strip() for part in parts if part.strip()]


def chunk_document(
    doc: ExtractedDocument,
    *,
    max_tokens: int = 350,
    overlap_tokens: int = 40,
) -> list[ChunkRecord]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if overlap_tokens < 0:
        raise ValueError("overlap_tokens must be >= 0")
    if overlap_tokens >= max_tokens:
        raise ValueError("overlap_tokens must be less than max_tokens")

    sentences = _split_sentences(doc.text)
    if not sentences:
        return []

    doc_id = build_doc_id(doc)
    heading_path = f"{doc.title} > {doc.section}" if doc.section else doc.title

    chunks: list[ChunkRecord] = []
    current_sentences: list[str] = []

    for sentence in sentences:
        if token_count(sentence) > max_tokens:
            words = sentence.split()
            sentence = " ".join(words[: max(1, len(words) // 2)])

        candidate = " ".join(current_sentences + [sentence]).strip()
        if not current_sentences or token_count(candidate) <= max_tokens:
            current_sentences.append(sentence)
            continue

        chunk_text = _normalize_whitespace(" ".join(current_sentences))
        chunks.append(
            ChunkRecord(
                chunk_id=build_chunk_id(doc_id, chunk_text, len(chunks)),
                doc_id=doc_id,
                text=chunk_text,
                source_url=doc.source_url,
                heading_path=heading_path,
                token_count=token_count(chunk_text),
                chunk_index=len(chunks),
            )
        )

        if overlap_tokens > 0:
            overlap: list[str] = []
            for prev in reversed(current_sentences):
                prospective = " ".join(reversed([prev, *overlap]))
                if token_count(prospective) > overlap_tokens and overlap:
                    break
                overlap.insert(0, prev)
            current_sentences = overlap + [sentence]
        else:
            current_sentences = [sentence]

    if current_sentences:
        chunk_text = _normalize_whitespace(" ".join(current_sentences))
        chunks.append(
            ChunkRecord(
                chunk_id=build_chunk_id(doc_id, chunk_text, len(chunks)),
                doc_id=doc_id,
                text=chunk_text,
                source_url=doc.source_url,
                heading_path=heading_path,
                token_count=token_count(chunk_text),
                chunk_index=len(chunks),
            )
        )

    return chunks
