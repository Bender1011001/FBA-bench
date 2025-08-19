from __future__ import annotations

from agents.skill_modules.extract_fields import ExtractFieldsSkill


def test_extract_fields_basic_kv_lines():
    text = "Name: Jane Doe\nDate: 2025-07-04\nNote: Example"
    skill = ExtractFieldsSkill()
    out = skill.run({"text": text, "fields": ["name", "date"]})
    assert out["extracted"] == {"name": "Jane Doe", "date": "2025-07-04"}


def test_extract_fields_json_like_block_and_missing_fields():
    text = """
    Info:
      Something here
    {"Name": "Alice", "Age": 30}
    """
    skill = ExtractFieldsSkill()
    out = skill.run({"text": text, "fields": ["name", "date", "age"]})
    # Should find name and age, omit date because it's missing
    assert out["extracted"]["name"] == "Alice"
    assert out["extracted"]["age"] == "30"
    assert "date" not in out["extracted"]