import unittest
from datetime import datetime, timezone
from pathlib import Path

from paperclaw.models import Analysis, Paper, RelevanceDecision
from paperclaw.render import output_path, render_markdown, slugify


class RenderTests(unittest.TestCase):
    def setUp(self):
        self.paper = Paper(
            arxiv_id="2609.01234",
            title="A Quadruped Robot for Testing",
            authors=["Alice", "Bob"],
            abstract="This is an abstract.",
            published=datetime(2026, 9, 13, tzinfo=timezone.utc),
            updated=None,
            categories=["cs.RO"],
            abs_url="https://arxiv.org/abs/2609.01234",
            pdf_url="https://arxiv.org/pdf/2609.01234.pdf",
        )
        self.decision = RelevanceDecision(
            True,
            0.91,
            "Quard-robot",
            ["Quard-robot"],
            ["quadruped"],
            ["locomotion"],
        )
        self.analysis = Analysis(
            tldr="一句话总结。",
            institutions=["某大学"],
            contributions=["贡献一"],
            limitations=["局限一"],
            questions=[
                {"question": f"问题 {i}", "conclusion": "结论", "points": ["要点"]}
                for i in range(10)
            ],
        )

    def test_slug_and_path(self):
        self.assertEqual(slugify("A Quadruped Robot!"), "a-quadruped-robot")
        path = output_path(Path("."), "Quard-robot", self.paper)
        self.assertEqual(
            str(path).replace("\\", "/"),
            "Quard-robot/2026/09/20260913-2609.01234-a-quadruped-robot-for-testing.md",
        )

    def test_markdown_has_required_sections(self):
        text = render_markdown(self.paper, self.decision, self.analysis)
        for value in [
            "arxiv_id:",
            "## 基础信息",
            "## 摘要",
            "## TL;DR",
            "## 主要贡献",
            "## 局限与待改进方向",
            "## 10 个问题深度分析",
        ]:
            self.assertIn(value, text)
        self.assertEqual(text.count("### Q"), 10)

    def test_pending_analysis_is_explicit(self):
        text = render_markdown(self.paper, self.decision, None, "pending", "模型不可用")
        self.assertIn('analysis_status: "pending"', text)
        self.assertIn("模型不可用", text)
        self.assertIn("等待下一次任务重试", text)


if __name__ == "__main__":
    unittest.main()
