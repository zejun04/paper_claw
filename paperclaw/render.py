from __future__ import annotations

import json
import re
import unicodedata
from datetime import timezone
from pathlib import Path

from .llm import QUESTIONS
from .models import Analysis, Paper, RelevanceDecision


def _yaml_string(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def slugify(value: str, max_length: int = 72) -> str:
    normalized = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized.lower()).strip("-")
    return slug[:max_length].rstrip("-") or "paper"


def output_path(root: Path, category: str, paper: Paper) -> Path:
    date = paper.published.astimezone(timezone.utc).date()
    safe_id = paper.arxiv_id.replace("/", "-")
    return root / category / f"{date.year:04d}" / f"{date.month:02d}" / (
        f"{date:%Y%m%d}-{safe_id}-{slugify(paper.title)}.md"
    )


def _bullets(values: list[str], fallback: str = "论文未说明") -> str:
    if not values:
        return f"- {fallback}"
    return "\n".join(f"- {value}" for value in values)


def render_markdown(
    paper: Paper,
    decision: RelevanceDecision,
    analysis: Analysis | None,
    analysis_status: str = "complete",
    analysis_error: str | None = None,
) -> str:
    matched = list(dict.fromkeys(decision.matched_categories + sorted(paper.matched_categories)))
    lines = [
        "---",
        f"arxiv_id: {_yaml_string(paper.arxiv_id)}",
        f"title: {_yaml_string(paper.title)}",
        f"published: {_yaml_string(paper.published.isoformat())}",
        f"primary_category: {_yaml_string(decision.primary_category or 'unknown')}",
        f"matched_categories: {_yaml_string(', '.join(matched))}",
        f"relevance: {decision.relevance:.3f}",
        f"analysis_status: {_yaml_string(analysis_status)}",
        f"arxiv_abs: {_yaml_string(paper.abs_url)}",
        f"arxiv_pdf: {_yaml_string(paper.pdf_url)}",
        "---",
        "",
        f"# [{paper.published:%Y%m%d}] {paper.title}",
        "",
        "## 基础信息",
        "",
        "| 项目 | 内容 |",
        "| --- | --- |",
        f"| 作者 | {', '.join(paper.authors) or '论文未说明'} |",
        f"| 日期 | {paper.published:%Y-%m-%d} |",
        f"| arXiv | [abs]({paper.abs_url}) / [pdf]({paper.pdf_url}) |",
        f"| arXiv 分类 | {', '.join(paper.categories) or '论文未说明'} |",
        f"| 相关性 | {decision.relevance:.3f} |",
        f"| 命中标签 | {', '.join(decision.tags) or '论文未说明'} |",
        f"| 筛选依据 | {'；'.join(decision.evidence) or '论文未说明'} |",
        "",
        "## 摘要",
        "",
        paper.abstract or "论文未说明",
        "",
    ]
    if analysis_error:
        lines.extend(["> 深度分析暂未完成：" + analysis_error, ""])
    if analysis is None:
        lines.extend(["## TL;DR", "", "论文未说明（等待下一次任务重试）。", ""])
        return "\n".join(lines) + "\n"

    lines.extend(
        [
            "## TL;DR",
            "",
            analysis.tldr,
            "",
            "## 单位",
            "",
            _bullets(analysis.institutions),
            "",
            "## 主要贡献",
            "",
            _bullets(analysis.contributions),
            "",
            "## 局限与待改进方向",
            "",
            _bullets(analysis.limitations),
            "",
        ]
    )
    if analysis.code_url or analysis.data_url:
        lines.extend(["## 开源链接", ""])
        if analysis.code_url:
            lines.append(f"- 代码：[链接]({analysis.code_url})")
        if analysis.data_url:
            lines.append(f"- 数据：[链接]({analysis.data_url})")
        lines.append("")
    lines.extend(["## 10 个问题深度分析", ""])
    for index, item in enumerate(analysis.questions[:10]):
        question = QUESTIONS[index]
        conclusion = item.get("conclusion") or "论文未说明"
        lines.extend([f"### Q{index + 1}: {question}", "", conclusion, ""])
        points = item.get("points") or []
        if points:
            lines.extend([_bullets([str(point) for point in points]), ""])
    return "\n".join(lines) + "\n"
