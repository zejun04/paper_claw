import json
import unittest
from datetime import datetime, timezone
from unittest.mock import patch

from paperclaw.llm import QUESTIONS, analyze_paper, classify_candidates
from paperclaw.models import Paper


class FakeResponses:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return type("Response", (), {"output_text": json.dumps(self.output, ensure_ascii=False)})()


class FakeClient:
    def __init__(self, output):
        self.responses = FakeResponses(output)


def paper():
    return Paper(
        arxiv_id="2609.00001",
        title="Quadruped policy",
        authors=["A"],
        abstract="quadruped abstract",
        published=datetime(2026, 9, 13, tzinfo=timezone.utc),
        updated=None,
        categories=["cs.RO"],
        abs_url="https://arxiv.org/abs/2609.00001",
        pdf_url="https://arxiv.org/pdf/2609.00001.pdf",
        matched_categories={"Quard-robot"},
    )


class LlmTests(unittest.TestCase):
    def test_classification_uses_structured_output_and_clamps_values(self):
        client = FakeClient(
            {
                "decisions": [
                    {
                        "arxiv_id": "2609.00001",
                        "include": True,
                        "relevance": 2,
                        "primary_category": "Quard-robot",
                        "matched_categories": ["Quard-robot"],
                        "evidence": ["标题明确提到 quadruped"],
                        "tags": ["quadruped"],
                    }
                ]
            }
        )
        with patch("paperclaw.llm._client", return_value=client):
            decisions = classify_candidates(
                [paper()],
                {"Quard-robot": {"description": "四足", "keywords": ["quadruped"]}},
                "test-model",
                0.65,
            )
        self.assertEqual(decisions["2609.00001"].relevance, 1.0)
        self.assertTrue(client.responses.calls[0]["text"]["format"]["strict"])

    def test_analysis_fills_missing_questions_without_inventing(self):
        client = FakeClient(
            {
                "tldr": "TLDR",
                "institutions": [],
                "contributions": [],
                "limitations": [],
                "questions": [],
                "code_url": None,
                "data_url": None,
            }
        )
        decision = type(
            "Decision",
            (),
            {"primary_category": "Quard-robot"},
        )()
        with patch("paperclaw.llm._client", return_value=client):
            result = analyze_paper(paper(), decision, "paper text", "test-model")
        self.assertEqual(len(result.questions), 10)
        self.assertEqual(result.questions[0]["question"], QUESTIONS[0])
        self.assertEqual(result.questions[0]["conclusion"], "论文未说明")


if __name__ == "__main__":
    unittest.main()
