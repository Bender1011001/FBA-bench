from __future__ import annotations

import pytest

from agents.skill_modules.summarize import SummarizeSkill
from agents.skill_modules.base import SkillExecutionError


def test_summarize_truncates_under_token_budget():
    text = (
        "S1 short. "
        "S2 short. "
        "S3 very very long sentence with many tokens that might exceed small budgets. "
        "S4 short! "
        "S5 short? "
        "S6 short. "
        "S7 short. "
        "S8 short."
    )
    skill = SummarizeSkill()
    out = skill.run({"text": text, "max_tokens": 6})
    # Expect first sentences accumulated until 6 tokens total;
    # S1 short. -> 2 tokens, S2 short. -> 2 more (4), next sentence S3 very very... starts with many tokens, exceeds budget -> stop.
    assert out["summary"] == "S1 short. S2 short."


@pytest.mark.parametrize(
    "text,budget,expected",
    [
        ("Just one sentence here.", 100, "Just one sentence here."),
        ("A. B. C.", 2, "A."),  # budget too small to include multiple
    ],
)
def test_summarize_edge_cases(text, budget, expected):
    skill = SummarizeSkill()
    out = skill.run({"text": text, "max_tokens": budget})
    assert out["summary"] == expected


def test_summarize_invalid_input():
    skill = SummarizeSkill()
    with pytest.raises(SkillExecutionError):
        skill.run({"text": "", "max_tokens": 10})