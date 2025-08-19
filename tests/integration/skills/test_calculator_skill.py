import pytest

from agents.skill_modules.calculator import CalculatorSkill
from agents.skill_modules.base import SkillExecutionError


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("2+2*3", 8.0),
        ("(2+2)*3", 12.0),
        ("2**3", 8.0),
        ("12*(3+4)", 84.0),
        ("10/4", 2.5),
        ("10//4", 2.0),
        ("10%4", 2.0),
        ("-5 + 2", -3.0),
        ("--5", 5.0),
    ],
)
def test_calculator_valid_expressions(expr, expected):
    skill = CalculatorSkill()
    out = skill.run({"expression": expr})
    assert pytest.approx(out["result"], rel=1e-9) == expected
    assert isinstance(out["steps"], list) and len(out["steps"]) > 0


@pytest.mark.parametrize(
    "expr",
    [
        "import os",
        "__import__('os')",
        "os.system('echo hi')",
        "lambda x: x",
        "a + 1",
        "2**(1000)",  # too large exponent should be blocked
        "1/0",  # division by zero
    ],
)
def test_calculator_invalid_expressions(expr):
    skill = CalculatorSkill()
    with pytest.raises(SkillExecutionError):
        skill.run({"expression": expr})