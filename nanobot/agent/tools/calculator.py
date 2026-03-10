"""Safe math calculator using Python's ast module."""

import ast
import math
import operator
from typing import Any

from nanobot.agent.tools.base import Tool

# Allowed operators
_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Allowed math functions
_FUNCTIONS = {
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
    "sum": sum,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "asin": math.asin,
    "acos": math.acos,
    "atan": math.atan,
    "log": math.log,
    "log10": math.log10,
    "log2": math.log2,
    "exp": math.exp,
    "ceil": math.ceil,
    "floor": math.floor,
    "factorial": math.factorial,
    "gcd": math.gcd,
    "radians": math.radians,
    "degrees": math.degrees,
}

_CONSTANTS = {
    "pi": math.pi,
    "e": math.e,
    "tau": math.tau,
    "inf": math.inf,
}


def _safe_eval(node: ast.AST) -> Any:
    """Recursively evaluate an AST node, only allowing safe math operations."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, complex)):
            return node.value
        raise ValueError(f"Unsupported constant: {node.value!r}")

    if isinstance(node, ast.Name):
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        raise ValueError(f"Unknown variable: {node.id}")

    if isinstance(node, ast.UnaryOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op_func(_safe_eval(node.operand))

    if isinstance(node, ast.BinOp):
        op_func = _OPERATORS.get(type(node.op))
        if op_func is None:
            raise ValueError(f"Unsupported operator: {type(node.op).__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return op_func(left, right)

    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise ValueError("Only simple function calls allowed")
        func_name = node.func.id
        if func_name not in _FUNCTIONS:
            raise ValueError(f"Unknown function: {func_name}")
        args = [_safe_eval(arg) for arg in node.args]
        return _FUNCTIONS[func_name](*args)

    if isinstance(node, ast.List):
        return [_safe_eval(el) for el in node.elts]

    if isinstance(node, ast.Tuple):
        return tuple(_safe_eval(el) for el in node.elts)

    raise ValueError(f"Unsupported expression: {type(node).__name__}")


class CalculatorTool(Tool):
    """Evaluate mathematical expressions safely."""

    @property
    def name(self) -> str:
        return "calculator"

    @property
    def description(self) -> str:
        return (
            "Evaluate a mathematical expression. Supports arithmetic (+, -, *, /, //, %, **), "
            "functions (sqrt, sin, cos, tan, log, exp, abs, round, min, max, ceil, floor, factorial), "
            "and constants (pi, e, tau)."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate, e.g. 'sqrt(144) + 3 * pi'",
                },
            },
            "required": ["expression"],
        }

    async def execute(self, **kwargs: Any) -> str:
        expression = kwargs.get("expression", "").strip()
        if not expression:
            return "Error: No expression provided."

        try:
            tree = ast.parse(expression, mode="eval")
            result = _safe_eval(tree)
            # Format nicely
            if isinstance(result, float) and result == int(result) and not math.isinf(result):
                result = int(result)
            return f"{expression} = {result}"
        except (ValueError, TypeError, ZeroDivisionError, OverflowError) as e:
            return f"Error: {e}"
        except SyntaxError:
            return f"Error: Invalid expression syntax: {expression}"
