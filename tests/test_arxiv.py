import unittest
from datetime import datetime, timezone
from urllib.error import HTTPError

from paperclaw.arxiv import ArxivClient, build_query, canonical_id, parse_feed, parse_rss


FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <id>http://arxiv.org/abs/2609.01234v2</id>
    <updated>2026-09-14T01:00:00Z</updated>
    <published>2026-09-13T01:00:00Z</published>
    <title>  A Quadruped Robot  </title>
    <summary> A useful abstract. </summary>
    <author><name>Alice</name></author>
    <category term="cs.RO" />
    <link rel="alternate" href="https://arxiv.org/abs/2609.01234" />
    <link title="pdf" href="https://arxiv.org/pdf/2609.01234.pdf" type="application/pdf" />
  </entry>
</feed>'''

RSS_FEED = b'''<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:dc="http://purl.org/dc/elements/1.1/">
  <channel>
    <item>
      <title>A Quadruped Robot</title>
      <link>https://arxiv.org/abs/2609.01234</link>
      <description>arXiv:2609.01234v1 Announce Type: new
Abstract: A quadruped robot abstract.</description>
      <category>cs.RO</category>
      <pubDate>Sun, 13 Sep 2026 01:00:00 +0000</pubDate>
      <dc:creator>Alice, Bob</dc:creator>
    </item>
  </channel>
</rss>'''


class ArxivTests(unittest.TestCase):
    def test_canonical_id(self):
        self.assertEqual(canonical_id("https://arxiv.org/abs/2401.00001v3"), "2401.00001")
        self.assertEqual(canonical_id("hep-th/9901001v2"), "hep-th/9901001")

    def test_parse_feed(self):
        papers = parse_feed(FEED)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2609.01234")
        self.assertEqual(papers[0].authors, ["Alice"])
        self.assertEqual(papers[0].published.tzinfo, timezone.utc)
        self.assertEqual(papers[0].pdf_url, "https://arxiv.org/pdf/2609.01234.pdf")

    def test_parse_rss(self):
        papers = parse_rss(RSS_FEED)
        self.assertEqual(len(papers), 1)
        self.assertEqual(papers[0].arxiv_id, "2609.01234")
        self.assertEqual(papers[0].authors, ["Alice", "Bob"])
        self.assertIn("quadruped robot abstract", papers[0].abstract)

    def test_query_contains_fields_and_date(self):
        query = build_query(["quadruped robot", "grasping"], ["cs.RO"], 3)
        self.assertIn('ti:"quadruped robot"', query)
        self.assertIn("abs:grasping", query)
        self.assertIn("cat:cs.RO", query)
        self.assertIn("submittedDate", query)

    def test_retry_and_pagination(self):
        calls = []
        responses = [RuntimeError("temporary"), FEED, FEED]

        def http_get(url, timeout):
            calls.append(url)
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        client = ArxivClient(
            "https://example.test/api",
            delay_seconds=0,
            http_get=http_get,
            sleep=lambda _: None,
        )
        papers = client.search(["quadruped"], ["cs.RO"], 1, 2)
        self.assertEqual(len(papers), 1)
        self.assertGreaterEqual(len(calls), 2)

    def test_rate_limit_honors_retry_after(self):
        calls = []
        sleeps = []
        responses = [
            HTTPError("https://example.test/api", 429, "Too Many Requests", {"Retry-After": "7"}, None),
            FEED,
        ]

        def http_get(url, timeout):
            calls.append(url)
            value = responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return value

        client = ArxivClient(
            "https://example.test/api",
            max_retries=2,
            rate_limit_backoff_seconds=60,
            http_get=http_get,
            sleep=sleeps.append,
        )
        papers = client.search(["quadruped"], ["cs.RO"], 1, 2)
        self.assertEqual(len(papers), 1)
        self.assertEqual(sleeps[0], 7.0)
        self.assertEqual(len(calls), 2)

    def test_not_acceptable_falls_back_to_post(self):
        calls = []

        def http_get(url, timeout):
            calls.append("GET")
            raise HTTPError(url, 406, "Not Acceptable", {}, None)

        def http_post(url, timeout):
            calls.append("POST")
            return FEED

        client = ArxivClient(
            "https://example.test/api",
            delay_seconds=0,
            max_retries=1,
            http_get=http_get,
            http_post=http_post,
            sleep=lambda _: None,
        )
        papers = client.search(["quadruped"], ["cs.RO"], 1, 1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(calls, ["GET", "POST"])

    def test_post_request_method_does_not_issue_get(self):
        calls = []

        def http_get(url, timeout):
            calls.append("GET")
            raise AssertionError("GET should not be used for post request mode")

        def http_post(url, timeout):
            calls.append("POST")
            return FEED

        client = ArxivClient(
            "https://example.test/api",
            delay_seconds=0,
            request_method="post",
            http_get=http_get,
            http_post=http_post,
            sleep=lambda _: None,
        )
        papers = client.search(["quadruped"], ["cs.RO"], 1, 1)
        self.assertEqual(len(papers), 1)
        self.assertEqual(calls, ["POST"])

    def test_406_falls_back_to_rss(self):
        calls = []

        def http_get(url, timeout):
            calls.append(url)
            return RSS_FEED

        def http_post(url, timeout):
            raise HTTPError(url, 406, "Not Acceptable", {}, None)

        client = ArxivClient(
            "https://example.test/api",
            delay_seconds=0,
            max_retries=1,
            request_method="post",
            rss_url="https://rss.example.test/rss",
            http_get=http_get,
            http_post=http_post,
            sleep=lambda _: None,
        )
        papers = client.search(
            ["quadruped"],
            ["cs.RO"],
            30,
            5,
            now=datetime(2026, 9, 18, tzinfo=timezone.utc),
        )
        self.assertEqual(len(papers), 1)
        self.assertEqual(calls, ["https://rss.example.test/rss/cs.RO"])


if __name__ == "__main__":
    unittest.main()
