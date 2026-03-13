import unittest

from backend.app.crawler import classify_links, is_policy_scoped_url, normalize_url


class CrawlerLinkClassificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.seed_url = (
            "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/"
            "commercial-medical-drug-policies.html"
        )
        self.base_url = self.seed_url

    def test_normalize_url_drops_fragment(self) -> None:
        raw = (
            "HTTPS://www.UHCProvider.com/en/policies-protocols/commercial-policies/"
            "a.html#section1"
        )
        normalized = normalize_url(raw)
        self.assertEqual(
            normalized,
            "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/a.html",
        )

    def test_classify_links_policy_scope_same_domain(self) -> None:
        hrefs = [
            "/en/policies-protocols/commercial-policies/policy-a.html",
            "/en/policies-protocols/commercial-policies/policy-b.pdf",
            "https://www.uhcprovider.com/en/other/press-release.html",
            "https://external.example.com/policy-x.pdf",
            "#ignored",
            "mailto:test@example.com",
        ]

        html_urls, pdf_urls = classify_links(hrefs, self.base_url, self.seed_url)

        self.assertEqual(
            html_urls,
            {
                "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/"
                "policy-a.html"
            },
        )
        self.assertEqual(
            pdf_urls,
            {
                "https://www.uhcprovider.com/en/policies-protocols/commercial-policies/"
                "policy-b.pdf"
            },
        )

    def test_policy_scope_keyword_filter(self) -> None:
        self.assertTrue(is_policy_scoped_url("https://x.com/en/policies/abc"))
        self.assertTrue(is_policy_scoped_url("https://x.com/en/protocols/abc"))
        self.assertFalse(is_policy_scoped_url("https://x.com/en/news/abc"))


if __name__ == "__main__":
    unittest.main()

