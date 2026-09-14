import unittest
from datetime import timezone

from paperclaw.arxiv import ArxivClient, build_query, canonical_id, parse_feed


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


if __name__ == "__main__":
    unittest.main()
