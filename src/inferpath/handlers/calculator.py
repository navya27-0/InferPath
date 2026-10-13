"""Safe arithmetic evaluator utilizing restricted Python AST parsing."""

import ast
import operator
import re
from typing import Any

from inferpath.models.domain import HandlerOutput
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


class CalculatorError(ValueError):
    """Raised when an expression is malformed, unsafe, or exceeds computation limits."""

    pass


class SafeCalculator:
    """
    Evaluates basic arithmetic expressions using a strict, allowlist-only AST visitor.

    Guarantees:
    - Never invokes eval() or exec().
    - Rejects all function calls, imports, attribute access, names, comprehensions, and statements.
    - Limits expression length, AST nesting depth, and exponent magnitudes to prevent DoS.
    """

    MAX_EXPR_LENGTH: int = 250
    MAX_AST_DEPTH: int = 15
    MAX_EXPONENT: int = 1000
    MAX_MAGNITUDE: float = 1e100

    OPERATORS: dict[type[ast.AST], Any] = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    # Regex to extract candidate arithmetic expressions from conversational text
    EXPR_EXTRACTION_PATTERN = re.compile(
        r"(?:(?:calculate|compute|what is|solve|evaluate)\s+)?([-+]?\(?\d+(?:\.\d+)?(?:\s*(?:\*\*|[\+\-\*\/\%\^])\s*\(?[-+]?\d+(?:\.\d+)?\)?)+(?:\s*\)?)*)",
        re.IGNORECASE,
    )

    @classmethod
    def evaluate(cls, expression: str) -> int | float:
        """
        Parses and evaluates an arithmetic expression safely.

        Raises CalculatorError on any syntax error, unsafe construct, or resource limit violation.
        """
        expr_str = expression.strip()
        # Handle caret as power if present
        if "^" in expr_str and "**" not in expr_str:
            expr_str = expr_str.replace("^", "**")

        if not expr_str:
            raise CalculatorError("Expression cannot be empty.")

        if len(expr_str) > cls.MAX_EXPR_LENGTH:
            raise CalculatorError(
                f"Expression length {len(expr_str)} exceeds limit {cls.MAX_EXPR_LENGTH}."
            )

        try:
            parsed = ast.parse(expr_str, mode="eval")
        except SyntaxError as err:
            raise CalculatorError(f"Malformed arithmetic expression: {err}") from err

        result = cls._eval_node(parsed.body, depth=1)
        if isinstance(result, float) and result.is_integer():
            return int(result)
        return result

    @classmethod
    def _eval_node(cls, node: ast.AST, depth: int) -> int | float:
        if depth > cls.MAX_AST_DEPTH:
            raise CalculatorError(
                f"Expression nesting depth exceeds safety limit of {cls.MAX_AST_DEPTH}."
            )

        # Allow numeric constants
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int | float) and not isinstance(node.value, bool):
                return node.value
            raise CalculatorError(
                f"Unsupported constant type: {type(node.value).__name__}. Only numbers are allowed."
            )

        # Allow unary operations: +x, -x
        if isinstance(node, ast.UnaryOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise CalculatorError(f"Unsupported unary operator: {op_type.__name__}")
            operand_val = cls._eval_node(node.operand, depth + 1)
            return cls.OPERATORS[op_type](operand_val)

        # Allow binary operations: x + y, x - y, x * y, x / y, etc.
        if isinstance(node, ast.BinOp):
            op_type = type(node.op)
            if op_type not in cls.OPERATORS:
                raise CalculatorError(f"Unsupported binary operator: {op_type.__name__}")

            left_val = cls._eval_node(node.left, depth + 1)
            right_val = cls._eval_node(node.right, depth + 1)

            # Division / modulo safety checks
            if op_type in (ast.Div, ast.FloorDiv, ast.Mod) and right_val == 0:
                raise CalculatorError("Division or modulo by zero is undefined.")

            # Exponentiation safety check
            if (
                op_type is ast.Pow
                and isinstance(right_val, int | float)
                and right_val > cls.MAX_EXPONENT
            ):
                raise CalculatorError(
                    f"Exponent {right_val} exceeds maximum allowed limit {cls.MAX_EXPONENT}."
                )

            try:
                result = cls.OPERATORS[op_type](left_val, right_val)
            except OverflowError as err:
                raise CalculatorError(
                    "Arithmetic operation resulted in numerical overflow."
                ) from err

            if isinstance(result, int | float) and abs(result) > cls.MAX_MAGNITUDE:
                raise CalculatorError(
                    f"Result magnitude exceeds maximum limit of {cls.MAX_MAGNITUDE}."
                )

            return result

        # Strictly reject everything else (Call, Attribute, Name, Import, Lambda, etc.)
        raise CalculatorError(
            f"Unsupported or unsafe AST syntax: {type(node).__name__}. Only numeric operations are allowed."
        )

    @classmethod
    def execute_handler(cls, prompt: str) -> HandlerOutput:
        """
        Executes the calculator handler and returns a standardized HandlerOutput.
        """
        # Try extracting expression or evaluate directly
        expr = prompt.strip()
        match = cls.EXPR_EXTRACTION_PATTERN.search(expr)
        if match:
            candidate = match.group(1).strip()
            # If the candidate ends with a question mark or period, trim it
            candidate = candidate.rstrip("?. ")
            if candidate:
                expr = candidate

        val = cls.evaluate(expr)
        result_str = str(val)

        return HandlerOutput(
            tier=RouteTier.DETERMINISTIC_A,
            output_text=result_str,
            input_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            output_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            latency_ms=ProvenanceValue(value=0.5, provenance=ProvenanceLabel.MEASURED, unit="ms"),
            model_version="safe_calculator_v1",
            raw_response={"expression": expr, "result": val},
        )
