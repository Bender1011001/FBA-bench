from __future__ import annotations

import ast
import logging
from typing import Any, Dict, Optional

try:
    from pydantic import BaseModel, Field, ValidationError
except Exception:  # pragma: no cover
    class BaseModel:  # type: ignore
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)
        def model_dump(self, *_, **__):
            return self.__dict__
    def Field(*_, **__):  # type: ignore
        return None

from .registry import register_metric

logger = logging.getLogger(__name__)


class CustomScriptableContext(BaseModel):
    expression: str = Field(description="Boolean expression to evaluate safely")
    # Optional allowlist of variable names (if omitted, a safe default set is derived)
    allowed_names: Optional[list[str]] = Field(
        default=None,
        description="Optional allowlist of names; defaults to safe run fields and metric keys",
    )

    class Config:
        frozen = False


class CustomScriptableOutput(BaseModel):
    result: bool
    expression: str

    def as_dict(self) -> Dict[str, Any]:
        return {"result": bool(self.result), "expression": str(self.expression)}


# -----------------------------
# Safe expression evaluator
# -----------------------------
_ALLOWED_BINOPS = {
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.Mod, ast.Pow,
    ast.BitAnd, ast.BitOr, ast.BitXor, ast.FloorDiv,
}
_ALLOWED_CMP = {ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE}
_ALLOWED_BOOLOPS = {ast.And, ast.Or}
_ALLOWED_UNARY = {ast.USub, ast.UAdd, ast.Not}
_ALLOWED_CONST_TYPES = (int, float, bool, str)


class _SafeEval(ast.NodeVisitor):
    def __init__(self, env: Dict[str, Any], allowed_names: set[str]):
        self.env = env
        self.allowed_names = allowed_names

    def visit_Module(self, node: ast.Module):  # type: ignore[override]
        if len(node.body) != 1 or not isinstance(node.body[0], ast.Expr):
            raise ValueError("Only a single expression is allowed")
        return self.visit(node.body[0].value)  # type: ignore

    def visit_Expr(self, node: ast.Expr):
        return self.visit(node.value)

    def visit_Name(self, node: ast.Name):
        if node.id not in self.allowed_names:
            raise NameError(f"name '{node.id}' is not allowed")
        return self.env.get(node.id, None)

    def visit_Constant(self, node: ast.Constant):
        if not isinstance(node.value, _ALLOWED_CONST_TYPES):
            raise ValueError("Unsupported constant type")
        return node.value

    def visit_UnaryOp(self, node: ast.UnaryOp):
        if type(node.op) not in _ALLOWED_UNARY:
            raise ValueError("Unary operator not allowed")
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.Not):
            return not bool(operand)
        if isinstance(node.op, ast.USub):
            return -float(operand)
        if isinstance(node.op, ast.UAdd):
            return +float(operand)
        raise ValueError("Invalid unary op")

    def visit_BoolOp(self, node: ast.BoolOp):
        if type(node.op) not in _ALLOWED_BOOLOPS:
            raise ValueError("Boolean operator not allowed")
        if isinstance(node.op, ast.And):
            res = True
            for v in node.values:
                res = bool(res) and bool(self.visit(v))
                if not res:
                    break
            return res
        if isinstance(node.op, ast.Or):
            res = False
            for v in node.values:
                res = bool(res) or bool(self.visit(v))
                if res:
                    break
            return res
        raise ValueError("Invalid boolean op")

    def visit_BinOp(self, node: ast.BinOp):
        if type(node.op) not in _ALLOWED_BINOPS:
            raise ValueError("Binary operator not allowed")
        left = self.visit(node.left)
        right = self.visit(node.right)
        # numeric operations only
        lf = float(left)
        rf = float(right)
        if isinstance(node.op, ast.Add):
            return lf + rf
        if isinstance(node.op, ast.Sub):
            return lf - rf
        if isinstance(node.op, ast.Mult):
            return lf * rf
        if isinstance(node.op, ast.Div):
            return lf / rf
        if isinstance(node.op, ast.FloorDiv):
            return lf // rf
        if isinstance(node.op, ast.Mod):
            return lf % rf
        if isinstance(node.op, ast.Pow):
            return lf ** rf
        if isinstance(node.op, ast.BitAnd):
            return int(lf) & int(rf)
        if isinstance(node.op, ast.BitOr):
            return int(lf) | int(rf)
        if isinstance(node.op, ast.BitXor):
            return int(lf) ^ int(rf)
        raise ValueError("Unsupported binop")

    def visit_Compare(self, node: ast.Compare):
        left = self.visit(node.left)
        result = True
        current = left
        for op, comparator in zip(node.ops, node.comparators):
            right = self.visit(comparator)
            if type(op) not in _ALLOWED_CMP:
                raise ValueError("Comparison operator not allowed")
            ok: bool
            if isinstance(op, ast.Eq):
                ok = current == right
            elif isinstance(op, ast.NotEq):
                ok = current != right
            elif isinstance(op, ast.Lt):
                ok = current < right
            elif isinstance(op, ast.LtE):
                ok = current <= right
            elif isinstance(op, ast.Gt):
                ok = current > right
            elif isinstance(op, ast.GtE):
                ok = current >= right
            else:
                ok = False
            if not ok:
                result = False
                break
            current = right
        return result

    # Disallowed nodes
    def generic_visit(self, node: ast.AST):
        disallowed = (
            ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.IfExp,
            ast.Dict, ast.List, ast.Set, ast.Tuple, ast.ListComp, ast.DictComp,
            ast.SetComp, ast.GeneratorExp, ast.Await, ast.Yield, ast.YieldFrom,
            ast.Import, ast.ImportFrom, ast.With, ast.For, ast.While, ast.If,
            ast.Assign, ast.Delete, ast.Try, ast.Raise, ast.ClassDef, ast.FunctionDef,
            ast.NamedExpr,
        )
        if isinstance(node, disallowed):
            raise ValueError("Disallowed syntax in expression")
        return super().generic_visit(node)


def _build_env(run: Dict[str, Any]) -> Dict[str, Any]:
    env: Dict[str, Any] = {}
    # Top-level run fields
    for k in ("duration_ms", "seed"):
        if k in run:
            env[k] = run[k]
    # Status as convenience booleans
    status = str(run.get("status", "success"))
    env["status_success"] = status == "success"
    env["status_timeout"] = status == "timeout"
    env["status_error"] = status == "error"
    # Flatten metrics fields one level: allow referencing metric keys directly if they are numeric/bool
    metrics = run.get("metrics") or {}
    for mk, mv in metrics.items():
        if isinstance(mv, (int, float, bool)):
            env[mk] = mv
        elif isinstance(mv, dict):
            # Promote common scalar fields
            for subk in ("latency_ms", "fast_enough", "accuracy", "coverage", "policy_violations", "compliant", "robustness_score", "efficiency", "completeness"):
                if subk in mv and isinstance(mv[subk], (int, float, bool)):
                    env[f"{mk}__{subk}"] = mv[subk]
    return env


def evaluate(run: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Custom scriptable metric.

    Evaluate a minimal safe boolean expression against run fields and metric values.

    Example:
      expression: "duration_ms < 1500 and (accuracy >= 0.8 or technical_performance__fast_enough)"
      Output: {"result": bool, "expression": str}
    """
    try:
        if not isinstance(run, dict):
            return {"result": False, "expression": "", "error": "invalid_run_type"}
        try:
            ctx = CustomScriptableContext(**(context or {}))  # type: ignore
        except ValidationError as ve:
            logger.error(f"custom_scriptable: invalid context {ve}")
            return {"result": False, "expression": "", "error": "validation_error"}

        expr = (ctx.expression or "").strip()
        if not expr:
            return {"result": False, "expression": "", "error": "empty_expression"}

        env = _build_env(run)

        allowed_names = set(ctx.allowed_names or list(env.keys()))
        # Always block dangerous identifiers just in case
        for blocked in ("__import__", "open", "eval", "exec", "os", "sys", "subprocess"):
            if blocked in allowed_names:
                allowed_names.remove(blocked)

        # Parse and evaluate safely
        tree = ast.parse(expr, mode="exec")
        evaluator = _SafeEval(env, allowed_names)
        value = evaluator.visit(tree)
        return CustomScriptableOutput(result=bool(value), expression=expr).as_dict()
    except Exception as e:
        logger.exception("custom_scriptable metric failed")
        return {"result": False, "expression": context.get("expression") if isinstance(context, dict) else "", "error": "exception", "reason": str(e)}


register_metric("custom_scriptable", evaluate)