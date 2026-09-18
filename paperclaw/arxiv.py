from __future__ import annotations

import os
import re
import time
import html
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Callable

from .models import Paper

ATOM_NS = "http://www.w3.org/2005/Atom"
ARXIV_NS = "http://arxiv.org/schemas/atom"
NS = {"atom": ATOM_NS, "arxiv": ARXIV_NS}
RSS_NS = {"dc": "http://purl.org/dc/elements/1.1/"}


class ArxivNotAcceptable(RuntimeError):
    """The configured arXiv API endpoint rejected the request with HTTP 406."""


def canonical_id(raw_id: str) -> str:
    value = raw_id.strip().rstrip("/")
    match = re.search(r"arxiv\.org/(?:abs|pdf)/(.+?)(?:\.pdf)?$", value, re.I)
    value = match.group(1) if match else value
    return re.sub(r"v\d+$", "", value)


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _text(node: ET.Element | None) -> str:
    return " ".join((node.text or "").split()) if node is not None else ""


def parse_feed(xml_bytes: bytes) -> list[Paper]:
    root = ET.fromstring(xml_bytes)
    papers: list[Paper] = []
    for entry in root.findall("atom:entry", NS):
        raw_id = _text(entry.find("atom:id", NS))
        arxiv_id = canonical_id(raw_id)
        if not arxiv_id:
            continue
        links = entry.findall("atom:link", NS)
        abs_url = raw_id or f"https://arxiv.org/abs/{arxiv_id}"
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        for link in links:
            href = link.attrib.get("href", "")
            if link.attrib.get("type") == "application/pdf" or link.attrib.get("title") == "pdf":
                pdf_url = href
            elif link.attrib.get("rel") == "alternate":
                abs_url = href
        published = _parse_datetime(_text(entry.find("atom:published", NS)))
        if published is None:
            continue
        categories = [
            node.attrib.get("term", "")
            for node in entry.findall("atom:category", NS)
            if node.attrib.get("term")
        ]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=_text(entry.find("atom:title", NS)),
                authors=[_text(node) for node in entry.findall("atom:author/atom:name", NS)],
                abstract=_text(entry.find("atom:summary", NS)),
                published=published,
                updated=_parse_datetime(_text(entry.find("atom:updated", NS))),
                categories=categories,
                abs_url=abs_url,
                pdf_url=pdf_url,
            )
        )
    return papers


def parse_rss(xml_bytes: bytes) -> list[Paper]:
    root = ET.fromstring(xml_bytes)
    papers: list[Paper] = []
    for item in root.findall("./channel/item"):
        abs_url = _text(item.find("link"))
        arxiv_id = canonical_id(abs_url)
        if not arxiv_id:
            continue
        title = html.unescape(_text(item.find("title")))
        description = html.unescape(_text(item.find("description")))
        abstract_match = re.search(r"Abstract:\s*(.*)", description, flags=re.S)
        abstract = " ".join((abstract_match.group(1) if abstract_match else description).split())
        published_text = _text(item.find("pubDate"))
        try:
            published = parsedate_to_datetime(published_text)
        except (TypeError, ValueError, OverflowError):
            continue
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        authors_text = _text(item.find("dc:creator", RSS_NS))
        authors = [author.strip() for author in authors_text.split(",") if author.strip()]
        categories = [
            _text(node)
            for node in item.findall("category")
            if _text(node)
        ]
        papers.append(
            Paper(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                abstract=abstract,
                published=published,
                updated=published,
                categories=categories,
                abs_url=abs_url,
                pdf_url=f"https://arxiv.org/pdf/{arxiv_id}.pdf",
            )
        )
    return papers


def _date_range(days: int, now: datetime | None = None) -> str:
    current = now or datetime.now(timezone.utc)
    start = current - timedelta(days=days)
    start_text = start.strftime("%Y%m%d%H%M")
    end_text = current.strftime("%Y%m%d%H%M")
    return f"submittedDate:[{start_text} TO {end_text}]"


def _field_query(keywords: list[str]) -> str:
    terms = []
    for keyword in keywords:
        term = f'"{keyword}"' if " " in keyword or "-" in keyword else keyword
        terms.extend([f"ti:{term}", f"abs:{term}"])
    return "(" + " OR ".join(terms) + ")"


def build_query(keywords: list[str], subject_categories: list[str], days: int) -> str:
    category_query = " OR ".join(f"cat:{category}" for category in subject_categories)
    return f"{_field_query(keywords)} AND ({category_query}) AND {_date_range(days)}"


class ArxivClient:
    def __init__(
        self,
        api_url: str,
        timeout: int = 45,
        delay_seconds: float = 3,
        max_retries: int = 3,
        rate_limit_backoff_seconds: float = 60,
        rate_limit_max_backoff_seconds: float = 300,
        request_method: str = "get",
        rss_url: str = "https://rss.arxiv.org/rss",
        http_get: Callable[[str, int], bytes] | None = None,
        http_post: Callable[[str, int], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self.rate_limit_max_backoff_seconds = rate_limit_max_backoff_seconds
        self.request_method = request_method.lower()
        if self.request_method not in {"get", "post"}:
            raise ValueError("request_method must be 'get' or 'post'")
        self.rss_url = rss_url.rstrip("/")
        self.http_get = http_get or self._default_http_get
        self.http_post = http_post or self._default_http_post
        self.sleep = sleep
        self._rss_cache: dict[str, list[Paper]] = {}

    @staticmethod
    def _default_http_get(url: str, timeout: int) -> bytes:
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": "paper-claw/0.1 (+https://github.com/zejun04/paper_claw)",
                "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
            },
        )
        with ArxivClient._open_url(request, timeout) as response:
            return response.read()

    @staticmethod
    def _default_http_post(url: str, timeout: int) -> bytes:
        parsed = urllib.parse.urlsplit(url)
        endpoint = urllib.parse.urlunsplit(
            (parsed.scheme, parsed.netloc, parsed.path, "", "")
        )
        request = urllib.request.Request(
            endpoint,
            data=parsed.query.encode("ascii"),
            headers={
                "User-Agent": "paper-claw/0.1 (+https://github.com/zejun04/paper_claw)",
                "Accept": "application/atom+xml, application/xml;q=0.9, */*;q=0.8",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with ArxivClient._open_url(request, timeout) as response:
            return response.read()

    @staticmethod
    def _open_url(request: urllib.request.Request, timeout: int):
        if os.environ.get("GITHUB_ACTIONS", "").lower() == "true":
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            return opener.open(request, timeout=timeout)
        return urllib.request.urlopen(request, timeout=timeout)

    def _request(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                request = self.http_post if self.request_method == "post" else self.http_get
                return request(url, self.timeout)
            except urllib.error.HTTPError as exc:
                last_error = exc
                if exc.code == 406 and self.request_method == "get":
                    try:
                        return self.http_post(url, self.timeout)
                    except urllib.error.HTTPError as post_exc:
                        if post_exc.code == 406:
                            raise ArxivNotAcceptable(
                                "arXiv API 返回 HTTP 406，GET 和 POST 均不可用"
                            ) from post_exc
                        last_error = post_exc
                    except Exception as post_exc:
                        last_error = post_exc
                elif exc.code == 406:
                    raise ArxivNotAcceptable(
                        f"arXiv API 返回 HTTP 406（{self.request_method.upper()} 请求不可用）"
                    ) from exc
                if exc.code == 429:
                    wait_seconds = self._rate_limit_wait(exc, attempt)
                elif 500 <= exc.code < 600:
                    wait_seconds = self.delay_seconds * (attempt + 1)
                else:
                    raise RuntimeError(f"arXiv API 请求失败: HTTP {exc.code}: {exc.reason}") from exc
                if attempt + 1 < self.max_retries:
                    self.sleep(wait_seconds)
            except Exception as exc:
                last_error = exc
                if attempt + 1 < self.max_retries:
                    self.sleep(self.delay_seconds * (attempt + 1))
        if isinstance(last_error, urllib.error.HTTPError) and last_error.code == 429:
            raise RuntimeError(
                f"arXiv API 请求失败: HTTP 429，已重试 {self.max_retries - 1} 次；"
                "请稍后再运行，或减少查询频率"
            ) from last_error
        raise RuntimeError(f"arXiv API 请求失败: {last_error}") from last_error

    def _search_rss(
        self,
        keywords: list[str],
        subject_categories: list[str],
        days: int,
        max_results: int,
        now: datetime | None,
    ) -> list[Paper]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        cutoff = current - timedelta(days=days)
        keyword_values = [keyword.casefold() for keyword in keywords]
        matched: dict[str, Paper] = {}
        for index, category in enumerate(subject_categories):
            if category not in self._rss_cache:
                if self._rss_cache:
                    self.sleep(self.delay_seconds)
                rss_bytes = self.http_get(f"{self.rss_url}/{category}", self.timeout)
                self._rss_cache[category] = parse_rss(rss_bytes)
            for paper in self._rss_cache[category]:
                haystack = f"{paper.title} {paper.abstract}".casefold()
                if paper.published < cutoff or paper.published > current:
                    continue
                if keyword_values and not any(keyword in haystack for keyword in keyword_values):
                    continue
                matched.setdefault(paper.arxiv_id, paper)
        return sorted(
            matched.values(),
            key=lambda paper: (paper.published, paper.arxiv_id),
            reverse=True,
        )[:max_results]

    def _rate_limit_wait(self, error: urllib.error.HTTPError, attempt: int) -> float:
        retry_after = error.headers.get("Retry-After") if error.headers else None
        if retry_after:
            try:
                return max(1.0, float(retry_after))
            except ValueError:
                try:
                    retry_at = parsedate_to_datetime(retry_after)
                    if retry_at.tzinfo is None:
                        retry_at = retry_at.replace(tzinfo=timezone.utc)
                    return max(1.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError, OverflowError):
                    pass
        return min(
            self.rate_limit_max_backoff_seconds,
            self.rate_limit_backoff_seconds * (2**attempt),
        )

    def search(
        self,
        keywords: list[str],
        subject_categories: list[str],
        days: int,
        max_results: int,
        now: datetime | None = None,
    ) -> list[Paper]:
        papers: list[Paper] = []
        page_size = min(100, max_results)
        start = 0
        query = build_query(keywords, subject_categories, days)
        while len(papers) < max_results:
            params = urllib.parse.urlencode(
                {
                    "search_query": query,
                    "start": start,
                    "max_results": min(page_size, max_results - len(papers)),
                    "sortBy": "submittedDate",
                    "sortOrder": "descending",
                }
            )
            try:
                xml_bytes = self._request(f"{self.api_url}?{params}")
            except ArxivNotAcceptable:
                return self._search_rss(keywords, subject_categories, days, max_results, now)
            page = parse_feed(xml_bytes)
            papers.extend(page)
            if len(page) < page_size:
                self.sleep(self.delay_seconds)
                break
            start += len(page)
            self.sleep(self.delay_seconds)
        else:
            self.sleep(self.delay_seconds)
        unique: dict[str, Paper] = {}
        for paper in papers:
            unique.setdefault(paper.arxiv_id, paper)
        return list(unique.values())[:max_results]
