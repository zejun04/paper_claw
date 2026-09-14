from __future__ import annotations

import re
import time
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
        http_get: Callable[[str, int], bytes] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.api_url = api_url
        self.timeout = timeout
        self.delay_seconds = delay_seconds
        self.max_retries = max_retries
        self.rate_limit_backoff_seconds = rate_limit_backoff_seconds
        self.rate_limit_max_backoff_seconds = rate_limit_max_backoff_seconds
        self.http_get = http_get or self._default_http_get
        self.sleep = sleep

    @staticmethod
    def _default_http_get(url: str, timeout: int) -> bytes:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "paper-claw/0.1 (+https://github.com/zejun04/paper_claw)"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.read()

    def _request(self, url: str) -> bytes:
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                return self.http_get(url, self.timeout)
            except urllib.error.HTTPError as exc:
                last_error = exc
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
            xml_bytes = self._request(f"{self.api_url}?{params}")
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
