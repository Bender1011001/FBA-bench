from __future__ import annotations

import abc
import asyncio
import logging
from typing import Any, Dict, Optional, Type, Union, Protocol

import ast
import operator as op

try:
    # Prefer centralized logging if available
    from fba_bench.core.logging import setup_logging  # type: ignore
except Exception:  # pragma: no cover - optional during import time
    setup_logging = None  # type: ignore

from pydantic import BaseModel, Field, ValidationError, ConfigDict


# Initialize logger with project defaults if available
if setup_logging:
    try:
        setup_logging()
    except Exception:
        # Do not fail on logging setup errors
        pass

logger = logging.getLogger(__name__)


class SkillExecutionError(Exception):
    """Raised when a skill fails during execution in a controlled manner."""


class Skill(abc.ABC):
    """
    Protocol/ABC for deterministic, dependency-light skills.

    Contract:
    - name: unique key for registry
    - description: human-friendly summary
    - input_model / output_model: Pydantic v2 models for IO
    - run()/arun(): deterministic, safe execution paths
    """

    # Metadata (override in subclasses)
    name: str = "skill"
    description: str = "Base skill"

    # Pydantic v2 models (override in subclasses)
    input_model: Type[BaseModel] = BaseModel
    output_model: Type[BaseModel] = BaseModel

    def __init__(self, config: Optional[dict] = None) -> None:
        self.config = config or {}

    @abc.abstractmethod
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the skill synchronously.

        Implementations MUST:
        - Validate parameters against input_model
        - Return a dict matching output_model schema
        - Raise SkillExecutionError with deterministic messages on failure
        """
        raise NotImplementedError

    async def arun(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Default async adapter wraps sync run in a threadpool.

        Skills may override for true async behavior if desired.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, params)


# Pydantic Base classes with common config for deterministic behavior
class SkillInputModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_default=True,
        frozen=False,
        arbitrary_types_allowed=False,
    )


class SkillOutputModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        validate_default=True,
        frozen=False,
        arbitrary_types_allowed=False,
    )


# Safe arithmetic evaluation helpers for calculator-like skills

# Allowed operators mapping for AST evaluation
_ALLOWED_BINOPS: Dict[type, Any] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
}

_ALLOWED_UNARYOPS: Dict[type, Any] = {
    ast.UAdd: op.pos,
    ast.USub: op.neg,
}

_MAX_POWER_EXPONENT = 9  # guard against huge exponentiation
_MAX_ABS_VALUE = 1e12  # guard against overflow magnitude
_MAX_NODES = 256  # guard complexity
_MAX_DEPTH = 32  # guard recursion


class _SafeArithmeticEvaluator(ast.NodeVisitor):
    """
    Deterministic, side-effect free evaluator for arithmetic expressions.

    Supports: +, -, *, /, //, %, **, parentheses and unary +/-.
    Disallows names, attributes, calls, and other node types.
    """

    def __init__(self) -> None:
        self._node_count = 0
        self._depth = 0

    def generic_visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        raise SkillExecutionError(f"Disallowed expression component: {type(node).__name__}")

    def _check_limits(self, value: Union[int, float]) -> None:
        if not isinstance(value, (int, float)):
            raise SkillExecutionError("Non-numeric value encountered during evaluation")
        if abs(value) > _MAX_ABS_VALUE:
            raise SkillExecutionError("Result exceeds allowed magnitude")

    def visit(self, node: ast.AST) -> Any:  # type: ignore[override]
        self._node_count += 1
        if self._node_count > _MAX_NODES:
            raise SkillExecutionError("Expression too complex")
        self._depth += 1
        try:
            if self._depth > _MAX_DEPTH:
                raise SkillExecutionError("Expression nesting too deep")
            return super().visit(node)
        finally:
            self._depth -= 1

    def visit_Expr(self, node: ast.Expr) -> Any:  # type: ignore[override]
        return self.visit(node.value)

    def visit_Constant(self, node: ast.Constant) -> Any:  # type: ignore[override]
        if isinstance(node.value, (int, float)):
            self._check_limits(float(node.value))
            return float(node.value)
        raise SkillExecutionError("Only numeric constants are allowed")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:  # type: ignore[override]
        op_type = type(node.op)
        if op_type not in _ALLOWED_UNARYOPS:
            raise SkillExecutionError(f"Unary operator not allowed: {op_type.__name__}")
        operand = self.visit(node.operand)
        result = _ALLOWED_UNARYOPS[op_type](operand)
        self._check_limits(result)
        return float(result)

    def visit_BinOp(self, node: ast.BinOp) -> Any:  # type: ignore[override]
        left = self.visit(node.left)
        right = self.visit(node.right)
        op_type = type(node.op)
        if op_type not in _ALLOWED_BINOPS:
            raise SkillExecutionError(f"Operator not allowed: {op_type.__name__}")

        # Specific guard for exponentiation
        if op_type is ast.Pow:
            if abs(right) > _MAX_POWER_EXPONENT:
                raise SkillExecutionError("Exponent too large")

        try:
            result = _ALLOWED_BINOPS[op_type](left, right)
        except ZeroDivisionError:
            raise SkillExecutionError("Division by zero is not allowed")
        except OverflowError:
            raise SkillExecutionError("Numeric overflow during calculation")
        except Exception as e:
            raise SkillExecutionError(f"Error during calculation: {e}")

        self._check_limits(result)
        return float(result)

    # Disallow everything else
    def visit_Name(self, node: ast.Name) -> Any:  # type: ignore[override]
        raise SkillExecutionError("Names are not allowed in expressions")

    def visit_Call(self, node: ast.Call) -> Any:  # type: ignore[override]
        raise SkillExecutionError("Function calls are not allowed in expressions")

    def visit_Attribute(self, node: ast.Attribute) -> Any:  # type: ignore[override]
        raise SkillExecutionError("Attributes are not allowed in expressions")

    def visit_Subscript(self, node: ast.Subscript) -> Any:  # type: ignore[override]
        raise SkillExecutionError("Subscripts are not allowed in expressions")

    def visit_List(self, node: ast.List) -> Any:  # type: ignore[override]
        raise SkillExecutionError("Lists are not allowed in expressions")

    def visit_Tuple(self, node: ast.Tuple) -> Any:  # type: ignore[override]
        raise SkillExecutionError("Tuples are not allowed in expressions")


def safe_arithmetic_eval(expression: str) -> float:
    """
    Evaluate a simple arithmetic expression safely using AST with a strict whitelist.

    Args:
        expression: string containing arithmetic

    Returns:
        float result

    Raises:
        SkillExecutionError on invalid syntax, disallowed nodes, or bounds violations.
    """
    try:
        # Parse in eval mode; only expression allowed
        tree = ast.parse(expression, mode="eval")
    except SyntaxError:
        raise SkillExecutionError("Invalid expression syntax")

    evaluator = _SafeArithmeticEvaluator()
    result = evaluator.visit(tree.body)
    # Final magnitude check
    if abs(result) > _MAX_ABS_VALUE:
        raise SkillExecutionError("Result exceeds allowed magnitude")
    return float(result)