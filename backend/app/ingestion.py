from __future__ import annotations

import asyncio
import hashlib
import logging
from datetime import datetime, timezone
from typing import Callable

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from .chunker import build_doc_id, chunk_document
from .crawler import CrawlResult, crawl_policy_links
from .embeddings import OpenAIEmbeddingClient
from .extractor import ExtractedDocument, extract_html_document, extract_pdf_document
from .schemas import DocumentRecord, IngestCounters
from .vector_store import QdrantChunkStore

logger = logging.getLogger(__name__)


def compute_content_hash(text: str) -> str:
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


class IngestionService:
    def __init__(
        self,
        *,
        store: QdrantChunkStore,
        embedding_client: OpenAIEmbeddingClient,
        crawler_func: Callable[[str], CrawlResult] | None = None,
        html_fetcher: Callable[[str], str] | None = None,
        pdf_fetcher: Callable[[str], bytes] | None = None,
    ) -> None:
        self._store = store
        self._embedding_client = embedding_client
        self._crawler_func = crawler_func or self._crawl
        self._html_fetcher = html_fetcher or self._fetch_html
        self._pdf_fetcher = pdf_fetcher or self._fetch_pdf

    def _crawl(self, seed_url: str) -> CrawlResult:
        logger.info("Starting crawl for seed URL: %s", seed_url)
        return asyncio.run(crawl_policy_links(seed_url))

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _fetch_html(self, url: str) -> str:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.text

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
    def _fetch_pdf(self, url: str) -> bytes:
        with httpx.Client(timeout=30.0, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            return response.content

    def _to_document_record(self, extracted: ExtractedDocument) -> DocumentRecord:
        now = datetime.now(timezone.utc)
        doc_id = build_doc_id(extracted)
        return DocumentRecord(
            doc_id=doc_id,
            source_url=extracted.source_url,
            title=extracted.title,
            doc_type=extracted.doc_type,  # type: ignore[arg-type]
            section=extracted.section,
            content_hash=compute_content_hash(extracted.text),
            last_seen_at=now,
        )

    def run_incremental(self, seed_url: str) -> IngestCounters:
        logger.info("Running incremental ingestion for: %s", seed_url)
        self._store.ensure_collection()
        crawl_result = self._crawler_func(seed_url)

        extracted_docs: list[ExtractedDocument] = []
        for html_url in crawl_result.html_urls:
            html = self._html_fetcher(html_url)
            extracted_docs.append(extract_html_document(html_url, html))

        for pdf_url in crawl_result.pdf_urls:
            pdf_bytes = self._pdf_fetcher(pdf_url)
            extracted_docs.append(extract_pdf_document(pdf_url, pdf_bytes))

        current_docs = [self._to_document_record(doc) for doc in extracted_docs]
        docs_by_id = {doc.doc_id: doc for doc in current_docs}
        existing_hashes = self._store.fetch_doc_hashes()

        changed_doc_ids = [
            doc.doc_id
            for doc in current_docs
            if existing_hashes.get(doc.doc_id) != doc.content_hash
        ]
        stale_doc_ids = [doc_id for doc_id in existing_hashes if doc_id not in docs_by_id]

        chunks = []
        chunk_docs_by_id: dict[str, DocumentRecord] = {}
        for extracted in extracted_docs:
            doc_id = build_doc_id(extracted)
            if doc_id not in changed_doc_ids:
                continue
            produced = chunk_document(extracted)
            if not produced:
                continue
            chunks.extend(produced)
            chunk_docs_by_id[doc_id] = docs_by_id[doc_id]

        deleted_chunks = self._store.delete_chunks_by_doc_ids(stale_doc_ids + changed_doc_ids)
        upserted_chunks = 0
        if chunks:
            vectors = self._embedding_client.embed_texts([chunk.text for chunk in chunks])
            upserted_chunks = self._store.upsert_chunks(chunks, vectors, chunk_docs_by_id)

        logger.info(
            "Ingestion complete: discovered=%d processed=%d upserted=%d deleted=%d",
            len(crawl_result.html_urls) + len(crawl_result.pdf_urls),
            len(current_docs),
            upserted_chunks,
            deleted_chunks,
        )

        return IngestCounters(
            discovered_urls=len(crawl_result.html_urls) + len(crawl_result.pdf_urls),
            processed_docs=len(current_docs),
            upserted_chunks=upserted_chunks,
            deleted_chunks=deleted_chunks,
        )
