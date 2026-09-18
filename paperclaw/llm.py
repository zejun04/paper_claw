from __future__ import annotations

import json
import os
from typing import Any

from .models import Analysis, Paper, RelevanceDecision

QUESTIONS = [
    "本文主要解决什么问题？",
    "前人技术路线是什么？",
    "前人方案有哪些局限性？",
    "本文的核心思路是什么？",
    "方法亮点有哪些？",
    "主要贡献是什么？",
    "实验设置、数据集和评价指标是什么？",
    "代码或数据是否开源？",
    "对本文的客观评价是什么？",
    "有哪些批判性问题和可改进方向？",
]


def _client():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError(
            "缺少 OpenAI SDK，请先运行: python -m pip install -r requirements.txt"
        ) from exc
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 OPENAI_API_KEY")
    return OpenAI(api_key=api_key)


def _output_text(response: Any) -> str:
    value = getattr(response, "output_text", None)
    if value:
        return value
    raise RuntimeError("OpenAI API 未返回 output_text")


def _create_json_response(
    model: str,
    system: str,
    user: str,
    schema: dict[str, Any],
    max_output_tokens: int,
) -> Any:
    client = _client()
    return client.responses.create(
        model=model,
        store=False,
        max_output_tokens=max_output_tokens,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system}]},
            {"role": "user", "content": [{"type": "input_text", "text": user}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": schema["name"],
                "strict": True,
                "schema": schema["schema"],
            }
        },
    )


def classify_candidates(
    papers: list[Paper],
    categories: dict[str, dict[str, Any]],
    model: str,
    threshold: float,
    max_output_tokens: int = 3000,
) -> dict[str, RelevanceDecision]:
    if not papers:
        return {}
    category_text = "\n".join(
        f"- {name}: {value.get('description', '')}; 关键词: {', '.join(value.get('keywords', []))}"
        for name, value in categories.items()
    )
    paper_text = "\n\n".join(
        f"ID: {paper.arxiv_id}\n标题: {paper.title}\n摘要: {paper.abstract}\n候选目录: {', '.join(sorted(paper.matched_categories))}"
        for paper in papers
    )
    schema = {
        "name": "paper_relevance_batch",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "decisions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "arxiv_id": {"type": "string"},
                            "include": {"type": "boolean"},
                            "relevance": {"type": "number"},
                            "primary_category": {"type": ["string", "null"]},
                            "matched_categories": {"type": "array", "items": {"type": "string"}},
                            "evidence": {"type": "array", "items": {"type": "string"}},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": [
                            "arxiv_id",
                            "include",
                            "relevance",
                            "primary_category",
                            "matched_categories",
                            "evidence",
                            "tags",
                        ],
                    },
                }
            },
            "required": ["decisions"],
        },
    }
    system = (
        "你是机器人论文筛选器。只依据给出的标题、摘要和候选目录判断相关性。"
        "不得使用外部知识补充论文事实。relevance 必须是 0 到 1。"
        f"只有 relevance >= {threshold} 且确实属于目标方向时 include 才为 true。"
        "primary_category 必须是给定目录之一；无关论文为 null。"
    )
    user = f"目标目录:\n{category_text}\n\n候选论文:\n{paper_text}"
    payload = json.loads(
        _output_text(_create_json_response(model, system, user, schema, max_output_tokens))
    )
    result: dict[str, RelevanceDecision] = {}
    valid_categories = set(categories)
    for item in payload.get("decisions", []):
        arxiv_id = str(item.get("arxiv_id", ""))
        primary = item.get("primary_category")
        relevance = max(0.0, min(1.0, float(item.get("relevance", 0))))
        if primary not in valid_categories:
            primary = None
        include = bool(item.get("include")) and primary is not None and relevance >= threshold
        matched = [name for name in item.get("matched_categories", []) if name in valid_categories]
        if primary and primary not in matched:
            matched.insert(0, primary)
        result[arxiv_id] = RelevanceDecision(
            include=include,
            relevance=relevance,
            primary_category=primary,
            matched_categories=matched,
            evidence=[str(value) for value in item.get("evidence", [])],
            tags=[str(value) for value in item.get("tags", [])],
        )
    return result


def analyze_paper(
    paper: Paper,
    decision: RelevanceDecision,
    source_text: str,
    model: str,
    max_output_tokens: int = 3500,
) -> Analysis:
    schema = {
        "name": "robotics_paper_analysis",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "tldr": {"type": "string"},
                "institutions": {"type": "array", "items": {"type": "string"}},
                "contributions": {"type": "array", "items": {"type": "string"}},
                "limitations": {"type": "array", "items": {"type": "string"}},
                "questions": {
                    "type": "array",
                    "minItems": 10,
                    "maxItems": 10,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "question": {"type": "string"},
                            "conclusion": {"type": "string"},
                            "points": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["question", "conclusion", "points"],
                    },
                },
                "code_url": {"type": ["string", "null"]},
                "data_url": {"type": ["string", "null"]},
            },
            "required": [
                "tldr",
                "institutions",
                "contributions",
                "limitations",
                "questions",
                "code_url",
                "data_url",
            ],
        },
    }
    system = (
        "你是中文机器人论文分析助手。只能依据给出的论文元数据、摘要和正文。"
        "不得虚构机构、代码、数据集、实验数字、对比结果或引用。"
        "如果正文没有说明，必须明确写‘论文未说明’。"
        "输出应简洁、技术准确，questions 必须严格按给定的 10 个问题返回。"
    )
    user = (
        f"论文 ID: {paper.arxiv_id}\n标题: {paper.title}\n作者: {', '.join(paper.authors)}\n"
        f"主分类: {decision.primary_category}\n摘要:\n{paper.abstract}\n\n论文正文（可能被截断）:\n{source_text}\n\n"
        "固定问题:\n" + "\n".join(f"{index + 1}. {question}" for index, question in enumerate(QUESTIONS))
    )
    payload = json.loads(
        _output_text(_create_json_response(model, system, user, schema, max_output_tokens))
    )
    questions = payload.get("questions", [])[:10]
    while len(questions) < 10:
        questions.append(
            {
                "question": QUESTIONS[len(questions)],
                "conclusion": "论文未说明",
                "points": [],
            }
        )
    return Analysis(
        tldr=str(payload.get("tldr", "论文未说明")),
        institutions=[str(value) for value in payload.get("institutions", [])],
        contributions=[str(value) for value in payload.get("contributions", [])],
        limitations=[str(value) for value in payload.get("limitations", [])],
        questions=questions,
        code_url=payload.get("code_url"),
        data_url=payload.get("data_url"),
    )
