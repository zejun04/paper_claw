from __future__ import annotations

import os
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

from .arxiv import ArxivClient
from .config import load_config
from .llm import analyze_paper, classify_candidates
from .models import Analysis, Paper, RelevanceDecision
from .pdf import download_pdf, extract_text
from .render import output_path, render_markdown
from .state import discover_existing, load_state, mark_success, save_state


def _heuristic_score(paper: Paper, keywords: list[str]) -> int:
    title = paper.title.lower()
    haystack = f"{paper.title} {paper.abstract}".lower()
    return sum(2 if keyword.lower() in title else 1 for keyword in keywords if keyword.lower() in haystack)


def _preselect(papers: list[Paper], categories: dict[str, dict[str, Any]], limit: int) -> list[Paper]:
    selected: dict[str, Paper] = {}
    for category, category_config in categories.items():
        category_papers = [paper for paper in papers if category in paper.matched_categories]
        category_papers.sort(
            key=lambda item: (
                _heuristic_score(item, category_config.get("keywords", [])),
                item.published,
            ),
            reverse=True,
        )
        for paper in category_papers[:limit]:
            selected[paper.arxiv_id] = paper
    return sorted(
        selected.values(),
        key=lambda item: (item.published, item.arxiv_id),
        reverse=True,
    )


def _select(
    papers: list[Paper],
    decisions: dict[str, RelevanceDecision],
    categories: dict[str, dict[str, Any]],
    limit: int,
    threshold: float,
) -> list[tuple[Paper, RelevanceDecision]]:
    grouped: dict[str, list[tuple[Paper, RelevanceDecision]]] = defaultdict(list)
    for paper in papers:
        decision = decisions.get(paper.arxiv_id)
        if (
            decision
            and decision.include
            and decision.relevance >= threshold
            and decision.primary_category in categories
        ):
            grouped[decision.primary_category].append((paper, decision))
    selected: list[tuple[Paper, RelevanceDecision]] = []
    for category in categories:
        values = sorted(
            grouped[category],
            key=lambda item: (item[1].relevance, item[0].published),
            reverse=True,
        )
        selected.extend(values[:limit])
    return selected


def _cap_selected(
    selected: list[tuple[Paper, RelevanceDecision]],
    categories: dict[str, dict[str, Any]],
    limit: int,
) -> list[tuple[Paper, RelevanceDecision]]:
    if limit <= 0 or len(selected) <= limit:
        return selected
    buckets: dict[str, list[tuple[Paper, RelevanceDecision]]] = {
        category: [] for category in categories
    }
    for item in selected:
        primary = item[1].primary_category
        if primary in buckets:
            buckets[primary].append(item)
    capped: list[tuple[Paper, RelevanceDecision]] = []
    while len(capped) < limit:
        added = False
        for category in categories:
            if buckets[category] and len(capped) < limit:
                capped.append(buckets[category].pop(0))
                added = True
        if not added:
            break
    return capped


def run(
    root: Path,
    days: int | None = None,
    dry_run: bool = False,
    now=None,
    arxiv_client: ArxivClient | None = None,
    classifier: Callable[..., dict[str, RelevanceDecision]] = classify_candidates,
    analyzer: Callable[..., Analysis] = analyze_paper,
    pdf_downloader: Callable[..., Path] = download_pdf,
    text_extractor: Callable[..., str] = extract_text,
) -> dict[str, Any]:
    config = load_config(root / ".paperclaw" / "config.yaml")
    fetch_config = config["fetch"]
    analysis_config = config["analysis"]
    categories = config["categories"]
    directories = list(categories)
    state_path = root / ".paperclaw" / "state.json"
    state = load_state(state_path)
    existing = discover_existing(root, directories)
    is_initial = not state.get("last_success_at") and not existing
    lookback_days = days if days is not None else (
        fetch_config["initial_lookback_days"] if is_initial else fetch_config["daily_lookback_days"]
    )
    client = arxiv_client or ArxivClient(
        api_url=config["arxiv"]["api_url"],
        timeout=fetch_config["request_timeout_seconds"],
        delay_seconds=fetch_config["request_delay_seconds"],
        max_retries=fetch_config["max_retries"],
        rate_limit_backoff_seconds=fetch_config["rate_limit_backoff_seconds"],
        rate_limit_max_backoff_seconds=fetch_config["rate_limit_max_backoff_seconds"],
        request_method=config["arxiv"].get("request_method", "get"),
        rss_url=config["arxiv"].get("rss_url", "https://rss.arxiv.org/rss"),
    )

    candidates: dict[str, Paper] = {}
    query_counts: dict[str, int] = {}
    for category, category_config in categories.items():
        papers = client.search(
            keywords=category_config.get("keywords", []),
            subject_categories=config["arxiv"]["subject_categories"],
            days=lookback_days,
            max_results=fetch_config["candidate_limit"],
            now=now,
        )
        query_counts[category] = len(papers)
        for paper in papers:
            paper.matched_categories.add(category)
            if paper.arxiv_id in candidates:
                candidates[paper.arxiv_id].matched_categories.add(category)
            else:
                candidates[paper.arxiv_id] = paper

    pending_or_new = [
        paper
        for paper in candidates.values()
        if paper.arxiv_id not in existing or existing[paper.arxiv_id].status not in {"complete", "degraded"}
    ]
    selected_for_classification = _preselect(
        pending_or_new,
        categories,
        fetch_config["preselect_limit"],
    )
    if dry_run:
        return {
            "lookback_days": lookback_days,
            "query_counts": query_counts,
            "candidate_count": len(candidates),
            "preselected": [paper.arxiv_id for paper in selected_for_classification],
            "selected": [],
            "written": [],
        }

    if not selected_for_classification:
        return {
            "lookback_days": lookback_days,
            "query_counts": query_counts,
            "candidate_count": len(candidates),
            "preselected": [],
            "selected": [],
            "written": [],
            "errors": [],
        }

    model = os.environ.get("OPENAI_MODEL")
    if not model:
        raise RuntimeError("未设置 OPENAI_MODEL，请在 GitHub Actions Variables 中配置")
    decisions: dict[str, RelevanceDecision] = {}
    batch_size = fetch_config["preselect_limit"]
    for start in range(0, len(selected_for_classification), batch_size):
        batch = selected_for_classification[start : start + batch_size]
        decisions.update(
            classifier(
                batch,
                categories,
                model,
                analysis_config["relevance_threshold"],
                analysis_config.get("classification_max_output_tokens", 3000),
            )
        )
    selected = _select(
        selected_for_classification,
        decisions,
        categories,
        fetch_config["max_per_category"],
        analysis_config["relevance_threshold"],
    )
    selected = _cap_selected(
        selected,
        categories,
        int(analysis_config.get("max_papers_per_run", 5)),
    )

    processed: dict[str, dict[str, str]] = {}
    written: list[str] = []
    errors: list[str] = []
    with tempfile.TemporaryDirectory(prefix="paperclaw-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, (paper, decision) in enumerate(selected):
            if index:
                time.sleep(float(analysis_config.get("request_delay_seconds", 0)))
            path = (
                Path(existing[paper.arxiv_id].path)
                if paper.arxiv_id in existing
                else output_path(root, decision.primary_category, paper)
            )
            analysis: Analysis | None = None
            status = "complete"
            analysis_error: str | None = None
            try:
                pdf_path = pdf_downloader(
                    paper.pdf_url,
                    temp_root / f"{paper.arxiv_id.replace('/', '-')}.pdf",
                )
                source_text = text_extractor(pdf_path, analysis_config["max_pdf_characters"])
                source_text = source_text[: analysis_config["max_analysis_characters"]]
            except Exception as exc:
                source_text = paper.abstract
                analysis_error = f"PDF 获取或解析失败，已退化为摘要：{exc}"
            try:
                analysis = analyzer(
                    paper,
                    decision,
                    source_text,
                    model,
                    analysis_config.get("analysis_max_output_tokens", 3500),
                )
                if analysis_error:
                    status = "degraded"
            except Exception as exc:
                status = "pending"
                analysis_error = f"模型分析失败，将在后续任务重试：{exc}"
                errors.append(f"{paper.arxiv_id}: {exc}")
            content = render_markdown(paper, decision, analysis, status, analysis_error)
            if path.exists() and existing.get(paper.arxiv_id) and existing[paper.arxiv_id].status == "complete":
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
            written.append(str(path.relative_to(root)))
            processed[paper.arxiv_id] = {"path": str(path.relative_to(root)), "status": status}

    if processed:
        mark_success(state, processed)
        save_state(state_path, state)
    return {
        "lookback_days": lookback_days,
        "query_counts": query_counts,
        "candidate_count": len(candidates),
        "preselected": [paper.arxiv_id for paper in selected_for_classification],
        "selected": [paper.arxiv_id for paper, _ in selected],
        "written": written,
        "errors": errors,
    }
