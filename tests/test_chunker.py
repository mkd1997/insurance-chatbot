import unittest

from backend.app.chunker import build_doc_id, chunk_document
from backend.app.extractor import ExtractedDocument


class ChunkerTests(unittest.TestCase):
    def _sample_doc(self) -> ExtractedDocument:
        text = (
            "Coverage begins on the policy effective date. "
            "Members must meet medical necessity criteria. "
            "Prior authorization may be required for some treatments. "
            "Experimental services are not covered unless explicitly listed. "
            "Appeals can be submitted through the provider portal."
        )
        return ExtractedDocument(
            source_url="https://www.uhcprovider.com/en/policies-protocols/sample.html",
            title="Commercial Medical Policy",
            doc_type="html",
            text=text,
            section="Coverage Criteria",
        )

    def test_doc_id_is_deterministic(self) -> None:
        doc = self._sample_doc()
        self.assertEqual(build_doc_id(doc), build_doc_id(doc))

    def test_chunk_document_respects_max_tokens(self) -> None:
        doc = self._sample_doc()
        chunks = chunk_document(doc, max_tokens=25, overlap_tokens=5)
        self.assertGreaterEqual(len(chunks), 2)
        for chunk in chunks:
            self.assertLessEqual(chunk.token_count, 25)
            self.assertEqual(chunk.source_url, doc.source_url)
            self.assertEqual(chunk.heading_path, "Commercial Medical Policy > Coverage Criteria")

    def test_chunk_ids_and_indexes_are_stable(self) -> None:
        doc = self._sample_doc()
        first = chunk_document(doc, max_tokens=30, overlap_tokens=6)
        second = chunk_document(doc, max_tokens=30, overlap_tokens=6)
        self.assertEqual([c.chunk_id for c in first], [c.chunk_id for c in second])
        self.assertEqual([c.chunk_index for c in first], list(range(len(first))))
        self.assertEqual([c.chunk_index for c in second], list(range(len(second))))

    def test_empty_text_returns_no_chunks(self) -> None:
        doc = ExtractedDocument(
            source_url="https://www.uhcprovider.com/en/policies-protocols/empty.html",
            title="Empty Policy",
            doc_type="html",
            text="   ",
            section=None,
        )
        self.assertEqual(chunk_document(doc), [])


if __name__ == "__main__":
    unittest.main()

