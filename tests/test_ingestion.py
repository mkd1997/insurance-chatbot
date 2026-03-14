import unittest
from dataclasses import dataclass

import httpx

from backend.app.chunker import build_doc_id
from backend.app.crawler import CrawlResult
from backend.app.extractor import ExtractedDocument, extract_html_document
from backend.app.ingestion import IngestionService, _should_retry_fetch, compute_content_hash


@dataclass
class _FakeStore:
    existing_hashes: dict[str, str]
    ensured: bool = False
    deleted_doc_ids: list[str] | None = None
    upsert_count: int = 0

    def ensure_collection(self) -> None:
        self.ensured = True

    def fetch_doc_hashes(self) -> dict[str, str]:
        return dict(self.existing_hashes)

    def delete_chunks_by_doc_ids(self, doc_ids: list[str]) -> int:
        self.deleted_doc_ids = list(doc_ids)
        return len(doc_ids)

    def upsert_chunks(self, chunks, vectors, documents_by_id) -> int:
        self.upsert_count = len(chunks)
        return len(chunks)


@dataclass
class _FakeEmbeddingClient:
    calls: int = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return [[0.1, 0.2] for _ in texts]


class IngestionServiceTests(unittest.TestCase):
    def _doc(self, url: str, text: str, doc_type: str = "html") -> ExtractedDocument:
        return ExtractedDocument(
            source_url=url,
            title="Policy",
            doc_type=doc_type,
            text=text,
            section="Coverage",
        )

    def test_incremental_updates_changed_and_stale_docs(self) -> None:
        seed = (
            "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/"
            "commercial-medical-drug-policies.html"
        )
        changed_url = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/a.html"
        new_url = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/b.html"
        stale_doc = self._doc(
            "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/stale.html",
            "Old content",
        )
        stale_doc_id = build_doc_id(stale_doc)

        changed_doc = self._doc(changed_url, "Updated coverage criteria for this month.")
        unchanged_html = (
            "<html><body><h1>Policy</h1><h2>Coverage</h2>"
            "<p>Same text each run.</p></body></html>"
        )
        unchanged_extracted = extract_html_document(new_url, unchanged_html)
        unchanged_doc = self._doc(new_url, "Same text each run.")
        unchanged_doc_id = build_doc_id(unchanged_doc)
        unchanged_hash = compute_content_hash(unchanged_extracted.text)

        store = _FakeStore(
            existing_hashes={
                stale_doc_id: "some-old-hash",
                build_doc_id(changed_doc): "old-hash",
                unchanged_doc_id: unchanged_hash,
            }
        )
        embeddings = _FakeEmbeddingClient()

        def crawl_func(_seed: str) -> CrawlResult:
            self.assertEqual(_seed, seed)
            return CrawlResult(html_urls=[changed_url, new_url], pdf_urls=[])

        html_map = {
            changed_url: (
                "<html><body><h1>Policy</h1><h2>Coverage</h2>"
                "<p>Updated coverage criteria for this month.</p></body></html>"
            ),
            new_url: unchanged_html,
        }

        service = IngestionService(
            store=store,  # type: ignore[arg-type]
            embedding_client=embeddings,  # type: ignore[arg-type]
            crawler_func=crawl_func,
            html_fetcher=lambda url: html_map[url],
            pdf_fetcher=lambda _url: b"",
        )

        counters = service.run_incremental(seed_url=seed)

        self.assertTrue(store.ensured)
        self.assertEqual(counters.discovered_urls, 2)
        self.assertEqual(counters.processed_docs, 2)
        self.assertGreaterEqual(counters.upserted_chunks, 1)
        self.assertIsNotNone(store.deleted_doc_ids)
        self.assertIn(stale_doc_id, store.deleted_doc_ids or [])
        self.assertNotIn(unchanged_doc_id, store.deleted_doc_ids or [])
        self.assertEqual(embeddings.calls, 1)

    def test_incremental_skips_403_and_404_fetch_failures(self) -> None:
        seed = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/root.html"
        ok_html_url = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/ok.html"
        forbidden_html_url = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/forbidden.html"
        missing_pdf_url = "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/missing.pdf"

        store = _FakeStore(existing_hashes={})
        embeddings = _FakeEmbeddingClient()

        def crawl_func(_seed: str) -> CrawlResult:
            return CrawlResult(
                html_urls=[ok_html_url, forbidden_html_url],
                pdf_urls=[missing_pdf_url],
            )

        def html_fetcher(url: str) -> str:
            if url == forbidden_html_url:
                request = httpx.Request("GET", url)
                response = httpx.Response(403, request=request)
                raise httpx.HTTPStatusError("Forbidden", request=request, response=response)
            return "<html><body><h1>Policy</h1><p>Reachable content.</p></body></html>"

        def pdf_fetcher(url: str) -> bytes:
            request = httpx.Request("GET", url)
            response = httpx.Response(404, request=request)
            raise httpx.HTTPStatusError("Not found", request=request, response=response)

        service = IngestionService(
            store=store,  # type: ignore[arg-type]
            embedding_client=embeddings,  # type: ignore[arg-type]
            crawler_func=crawl_func,
            html_fetcher=html_fetcher,
            pdf_fetcher=pdf_fetcher,
        )

        counters = service.run_incremental(seed_url=seed)

        self.assertEqual(counters.discovered_urls, 3)
        self.assertEqual(counters.processed_docs, 1)
        self.assertGreaterEqual(counters.upserted_chunks, 1)
        self.assertEqual(embeddings.calls, 1)

    def test_retry_filter_does_not_retry_403_or_404(self) -> None:
        for status_code in (403, 404):
            request = httpx.Request("GET", f"https://example.com/{status_code}")
            response = httpx.Response(status_code, request=request)
            exc = httpx.HTTPStatusError("status error", request=request, response=response)
            self.assertFalse(_should_retry_fetch(exc))

        self.assertTrue(_should_retry_fetch(httpx.ConnectError("boom")))


if __name__ == "__main__":
    unittest.main()
