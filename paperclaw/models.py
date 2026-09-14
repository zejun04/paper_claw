from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Paper:
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: datetime
    updated: datetime | None
    categories: list[str]
    abs_url: str
    pdf_url: str
    matched_categories: set[str] = field(default_factory=set)


@dataclass
class RelevanceDecision:
    include: bool
    relevance: float
    primary_category: str | None
    matched_categories: list[str]
    evidence: list[str]
    tags: list[str]


@dataclass
class Analysis:
    tldr: str
    institutions: list[str]
    contributions: list[str]
    limitations: list[str]
    questions: list[dict[str, Any]]
    code_url: str | None = None
    data_url: str | None = None


@dataclass
class ExistingRecord:
    path: str
    status: str
