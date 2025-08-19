from __future__ import annotations

import pytest

from agents.skill_modules.transform_text import TransformTextSkill
from agents.skill_modules.base import SkillExecutionError


def test_transform_text_operations_apply_in_order():
    skill = TransformTextSkill()
    params = {
        "text": "  Hello   WORLD ",
        "operations": ["strip", "lower", "dedupe_whitespace", "title"],
    }
    out = skill.run(params)
    assert out["text"] == "Hello World"


def test_transform_text_unknown_operation_raises():
    skill = TransformTextSkill()
    with pytest.raises(SkillExecutionError):
        skill.run({"text": "abc", "operations": ["unknown_op"]})