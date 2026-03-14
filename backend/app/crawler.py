from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urldefrag, urljoin, urlparse

from playwright.async_api import async_playwright

POLICY_PATH_KEYWORDS = (
    "polic",
    "policy",
    "protocol",
    "medical-drug-policies",
    "coverage",
    "benefit",
    "clinical",
)

PDF_EXTENSIONS = (".pdf", ".PDF")


@dataclass(frozen=True)
class CrawlResult:
    html_urls: list[str]
    pdf_urls: list[str]


def normalize_url(url: str) -> str:
    cleaned, _fragment = urldefrag(url.strip())
    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path or "/"
    normalized = parsed._replace(scheme=scheme, netloc=netloc, path=path)
    return normalized.geturl()


def is_same_domain(candidate_url: str, seed_url: str) -> bool:
    return urlparse(candidate_url).netloc == urlparse(seed_url).netloc


def is_pdf_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in PDF_EXTENSIONS)


def is_policy_scoped_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(keyword in path for keyword in POLICY_PATH_KEYWORDS)


def classify_links(
    hrefs: Iterable[str],
    base_url: str,
    seed_url: str,
) -> tuple[set[str], set[str]]:
    html_urls: set[str] = set()
    pdf_urls: set[str] = set()

    for href in hrefs:
        if not href:
            continue
        raw_href = href.strip()
        if not raw_href or raw_href.startswith("#"):
            continue
        if raw_href.startswith(("mailto:", "javascript:", "tel:")):
            continue
        absolute = normalize_url(urljoin(base_url, raw_href))
        if not absolute.startswith(("http://", "https://")):
            continue
        if not is_same_domain(absolute, seed_url):
            continue
        if not is_policy_scoped_url(absolute):
            continue
        if is_pdf_url(absolute):
            pdf_urls.add(absolute)
        else:
            html_urls.add(absolute)
    return html_urls, pdf_urls


async def crawl_policy_links(
    seed_url: str,
    *,
    max_pages: int = 40,
    timeout_ms: int = 20000,
) -> CrawlResult:
    seed = normalize_url(seed_url)
    queue: deque[str] = deque([seed])
    visited: set[str] = set()
    all_html: set[str] = {seed}
    all_pdfs: set[str] = set()

    async with async_playwright() as playwright:
        browser = await playwright.chromium.launch(headless=True)
        context = await browser.new_context()
        page = await context.new_page()

        try:
            while queue and len(visited) < max_pages:
                current = queue.popleft()
                if current in visited:
                    continue
                visited.add(current)

                try:
                    await page.goto(current, wait_until="domcontentloaded", timeout=timeout_ms)
                except Exception:
                    # Crawl continues even if a page fails to load.
                    continue

                hrefs = await page.eval_on_selector_all(
                    "a[href]",
                    "anchors => anchors.map(a => a.getAttribute('href'))",
                )
                html_urls, pdf_urls = classify_links(hrefs, current, seed)
                all_pdfs.update(pdf_urls)

                next_urls = sorted(url for url in html_urls if url not in visited)
                for url in next_urls:
                    all_html.add(url)
                    queue.append(url)
        finally:
            await context.close()
            await browser.close()

    return CrawlResult(
        html_urls=sorted(all_html),
        pdf_urls=sorted(all_pdfs),
    )
