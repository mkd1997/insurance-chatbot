from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from pypdf import PdfReader


@dataclass(frozen=True)
class ExtractedDocument:
    source_url: str
    title: str
    doc_type: str
    text: str
    section: str | None = None


def _normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def _title_from_url(url: str) -> str:
    path = urlparse(url).path.strip("/")
    if not path:
        return "Untitled Policy Document"
    leaf = path.split("/")[-1]
    return leaf.replace("-", " ").replace("_", " ").strip() or "Untitled Policy Document"


def extract_html_document(source_url: str, html: str) -> ExtractedDocument:
    soup = BeautifulSoup(html, "lxml")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    heading = soup.find("h1")
    page_title = soup.title.string if soup.title and soup.title.string else None
    title = _normalize_whitespace(heading.get_text()) if heading else None
    if not title and page_title:
        title = _normalize_whitespace(page_title)
    if not title:
        title = _title_from_url(source_url)

    section_tag = soup.find("h2")
    section = _normalize_whitespace(section_tag.get_text()) if section_tag else None

    text_blocks: list[str] = []
    for element in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "td"]):
        text = _normalize_whitespace(element.get_text(" ", strip=True))
        if text:
            text_blocks.append(text)

    text = _normalize_whitespace(" ".join(text_blocks))
    return ExtractedDocument(
        source_url=source_url,
        title=title,
        doc_type="html",
        text=text,
        section=section,
    )


def extract_pdf_document(source_url: str, pdf_bytes: bytes) -> ExtractedDocument:
    reader = PdfReader(BytesIO(pdf_bytes))
    text_blocks: list[str] = []

    for page in reader.pages:
        page_text = page.extract_text() or ""
        cleaned = _normalize_whitespace(page_text)
        if cleaned:
            text_blocks.append(cleaned)

    title = _title_from_url(source_url)
    text = _normalize_whitespace(" ".join(text_blocks))
    return ExtractedDocument(
        source_url=source_url,
        title=title,
        doc_type="pdf",
        text=text,
        section=None,
    )

