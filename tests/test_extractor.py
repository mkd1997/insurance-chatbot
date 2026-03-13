import unittest
from unittest.mock import Mock, patch

from backend.app.extractor import extract_html_document, extract_pdf_document


class ExtractorTests(unittest.TestCase):
    def test_extract_html_document_with_section_and_title(self) -> None:
        html = """
        <html>
            <head><title>Ignored Page Title</title></head>
            <body>
                <h1> Commercial Medical Policy </h1>
                <h2>Coverage Criteria</h2>
                <p>First paragraph about eligibility.</p>
                <ul><li>Item A</li><li>Item B</li></ul>
            </body>
        </html>
        """
        doc = extract_html_document(
            "https://www.uhcprovider.com/en/policies-protocols/sample-policy.html",
            html,
        )
        self.assertEqual(doc.doc_type, "html")
        self.assertEqual(doc.title, "Commercial Medical Policy")
        self.assertEqual(doc.section, "Coverage Criteria")
        self.assertIn("First paragraph about eligibility.", doc.text)
        self.assertIn("Item A", doc.text)

    def test_extract_html_document_uses_title_tag_fallback(self) -> None:
        html = """
        <html>
            <head><title> Policy Page Fallback </title></head>
            <body><p>Policy content.</p></body>
        </html>
        """
        doc = extract_html_document(
            "https://www.uhcprovider.com/en/policies-protocols/fallback.html",
            html,
        )
        self.assertEqual(doc.title, "Policy Page Fallback")
        self.assertEqual(doc.section, None)
        self.assertIn("Policy content.", doc.text)

    @patch("backend.app.extractor.PdfReader")
    def test_extract_pdf_document_normalizes_text(self, pdf_reader_mock: Mock) -> None:
        page1 = Mock()
        page1.extract_text.return_value = "Page 1 text\nwith spacing."
        page2 = Mock()
        page2.extract_text.return_value = "Page 2 text."
        pdf_reader_mock.return_value.pages = [page1, page2]

        doc = extract_pdf_document(
            "https://www.uhcprovider.com/en/policies-protocols/docs/policy-file.pdf",
            b"%PDF-1.4 mock",
        )
        self.assertEqual(doc.doc_type, "pdf")
        self.assertEqual(doc.title, "policy file.pdf")
        self.assertIn("Page 1 text with spacing.", doc.text)
        self.assertIn("Page 2 text.", doc.text)


if __name__ == "__main__":
    unittest.main()

