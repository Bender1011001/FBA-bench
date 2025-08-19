import pytest

from agents.skill_modules.registry import create, list_skills
from agents.skill_modules.calculator import CalculatorSkill
from agents.skill_modules.summarize import SummarizeSkill
from agents.skill_modules.extract_fields import ExtractFieldsSkill
from agents.skill_modules.lookup import LookupSkill
from agents.skill_modules.transform_text import TransformTextSkill


def test_list_skills_includes_expected():
    names = {entry["name"] for entry in list_skills()}
    assert {"calculator", "summarize", "extract_fields", "lookup", "transform_text"}.issubset(names)


def test_create_known_and_unknown_skill():
    inst = create("calculator")
    assert isinstance(inst, CalculatorSkill)

    with pytest.raises(ValueError) as ei:
        create("nonexistent_skill_123")
    msg = str(ei.value)
    # Should include supported list
    assert "Supported:" in msg
    # And include at least one known skill name
    assert "calculator" in msg


def test_list_skills_metadata_has_schemas():
    items = list_skills()
    # Find calculator
    calc = next((x for x in items if x["name"] == "calculator"), None)
    assert calc is not None
    assert isinstance(calc.get("input_schema"), dict)
    assert isinstance(calc.get("output_schema"), dict)