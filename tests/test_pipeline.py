import json
import os
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from paperclaw.models import Analysis, Paper, RelevanceDecision
from paperclaw.pipeline import run


class FakeArxiv:
    def search(self, keywords, subject_categories, days, max_results, now=None):
        return [
            Paper(
                arxiv_id="2609.00001",
                title="Quadruped policy",
                authors=["A"],
                abstract="quadruped robot abstract",
                published=datetime(2026, 9, 13, tzinfo=timezone.utc),
                updated=None,
                categories=["cs.RO"],
                abs_url="https://arxiv.org/abs/2609.00001",
                pdf_url="https://arxiv.org/pdf/2609.00001.pdf",
            )
        ]


def fake_classifier(papers, categories, model, threshold, max_output_tokens=3000):
    return {
        paper.arxiv_id: RelevanceDecision(
            True,
            0.9,
            "Quard-robot",
            ["Quard-robot"],
            ["quadruped"],
            ["quadruped"],
        )
        for paper in papers
    }


def fake_analyzer(paper, decision, source_text, model, max_output_tokens=3500):
    return Analysis(
        tldr="TLDR",
        institutions=[],
        contributions=["Contribution"],
        limitations=[],
        questions=[
            {"question": str(i), "conclusion": "Conclusion", "points": []}
            for i in range(10)
        ],
    )


class PipelineTests(unittest.TestCase):
    def test_dry_run_does_not_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = run(root, dry_run=True, arxiv_client=FakeArxiv())
            self.assertEqual(result["candidate_count"], 1)
            self.assertFalse((root / ".paperclaw" / "state.json").exists())
            self.assertFalse(list(root.rglob("*.md")))

    def test_writes_one_primary_category_and_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old_model = os.environ.get("OPENAI_MODEL")
            os.environ["OPENAI_MODEL"] = "test-model"
            try:
                result = run(
                    root,
                    arxiv_client=FakeArxiv(),
                    classifier=fake_classifier,
                    analyzer=fake_analyzer,
                    pdf_downloader=lambda url, destination: destination,
                    text_extractor=lambda path, max_chars: "paper text",
                )
            finally:
                if old_model is None:
                    os.environ.pop("OPENAI_MODEL", None)
                else:
                    os.environ["OPENAI_MODEL"] = old_model
            self.assertEqual(len(result["written"]), 1)
            self.assertEqual(len(list((root / "Quard-robot").rglob("*.md"))), 1)
            self.assertFalse((root / "humanoid").exists())
            state = json.loads(
                (root / ".paperclaw" / "state.json").read_text(encoding="utf-8")
            )
            self.assertEqual(state["processed"]["2609.00001"]["status"], "complete")

            result_again = run(
                root,
                arxiv_client=FakeArxiv(),
                classifier=fake_classifier,
                analyzer=fake_analyzer,
                pdf_downloader=lambda url, destination: destination,
                text_extractor=lambda path, max_chars: "paper text",
            )
            self.assertEqual(result_again["written"], [])


if __name__ == "__main__":
    unittest.main()
