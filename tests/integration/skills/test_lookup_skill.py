from __future__ import annotations

from agents.skill_modules.lookup import LookupSkill


def test_lookup_hit_and_miss_case_sensitive():
    table = {"SKU-42": "Widget XL", "sku-42": "different"}
    skill = LookupSkill()

    # Exact hit (case-sensitive)
    out_hit = skill.run({"key": "SKU-42", "table": table})
    assert out_hit["found"] is True
    assert out_hit["value"] == "Widget XL"

    # Miss due to case sensitivity
    out_miss = skill.run({"key": "SKU-42 ", "table": table})  # note trailing space -> not in dict
    assert out_miss["found"] is False
    assert out_miss["value"] is None

    # Different-cased key exists independently
    out_case = skill.run({"key": "sku-42", "table": table})
    assert out_case["found"] is True
    assert out_case["value"] == "different"