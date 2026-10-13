"""Deterministic evaluation checkers for InferPath benchmark evaluation."""

import json
import re
from typing import Any

from pydantic import BaseModel, Field

from inferpath.evaluation.dataset import DatasetItem


class CheckerResult(BaseModel):
    """Structured result produced by evaluating an output with a checker."""

    passed: bool = Field(description="Whether the output met the required criteria")
    score: float = Field(ge=0.0, le=1.0, description="Normalized quality score between 0.0 and 1.0")
    detail: str = Field(description="Explanation of the check outcome")
    checker_type: str = Field(
        description="Checker type used (exact, regex, numeric_tolerance, rubric)"
    )


class ExactMatchChecker:
    """Exact string match checker, ignoring surrounding whitespace."""

    @staticmethod
    def evaluate(output_text: str, config: dict[str, Any]) -> CheckerResult:
        reference = str(config.get("reference", "")).strip()
        actual = output_text.strip()
        case_sensitive = config.get("case_sensitive", False)

        matches = actual == reference if case_sensitive else actual.lower() == reference.lower()

        # Also check if reference is contained if exact strict match fails
        norm_ref = reference.replace("°", "").replace(" ", "").lower()
        norm_act = actual.replace("°", "").replace(" ", "").lower()

        if not matches and reference.lower() in actual.lower():
            score = 0.90
            passed = True
            detail = f"Substring match for reference: '{reference}'"
        elif not matches and norm_ref in norm_act:
            score = 0.90
            passed = True
            detail = f"Normalized match for reference: '{reference}'"
        elif matches:
            score = 1.0
            passed = True
            detail = f"Exact match with reference: '{reference}'"
        else:
            score = 0.0
            passed = False
            detail = f"Expected '{reference}', got '{actual[:100]}'"

        return CheckerResult(
            passed=passed,
            score=score,
            detail=detail,
            checker_type="exact",
        )


class RegexChecker:
    """Regex pattern checker validating required substrings or structural formats."""

    @staticmethod
    def evaluate(output_text: str, config: dict[str, Any]) -> CheckerResult:
        pattern_str = config.get("pattern", "")
        if not pattern_str:
            return CheckerResult(
                passed=True, score=1.0, detail="Empty regex pattern", checker_type="regex"
            )

        flags = re.DOTALL
        if not config.get("case_sensitive", False):
            flags |= re.IGNORECASE

        match = re.search(pattern_str, output_text, flags)
        if match:
            return CheckerResult(
                passed=True,
                score=1.0,
                detail=f"Matched regex pattern '{pattern_str}'",
                checker_type="regex",
            )
        return CheckerResult(
            passed=False,
            score=0.0,
            detail=f"Pattern '{pattern_str}' not found in output",
            checker_type="regex",
        )


class NumericToleranceChecker:
    """Extracts numeric values from text and checks against expected value with tolerance."""

    @staticmethod
    def evaluate(output_text: str, config: dict[str, Any]) -> CheckerResult:
        exp_val = config.get("expected") if "expected" in config else config.get("reference", 0.0)
        expected = float(exp_val)
        tolerance = float(config.get("tolerance", 0.01))

        # Extract all floats/ints from output_text
        found_numbers = [
            float(m.replace(",", ""))
            for m in re.findall(
                r"[-+]?\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|[-+]?\b\d+(?:\.\d+)?\b", output_text
            )
        ]

        if not found_numbers:
            return CheckerResult(
                passed=False,
                score=0.0,
                detail="No numeric values extracted from output",
                checker_type="numeric_tolerance",
            )

        # Check if any extracted number matches within tolerance
        best_diff = min(abs(n - expected) for n in found_numbers)
        if best_diff <= tolerance:
            return CheckerResult(
                passed=True,
                score=1.0,
                detail=f"Found number within tolerance: error={best_diff:.4f} <= {tolerance}",
                checker_type="numeric_tolerance",
            )

        # Partial credit if within 5x tolerance
        if best_diff <= tolerance * 5:
            score = max(0.0, 1.0 - (best_diff / (tolerance * 5)))
            return CheckerResult(
                passed=False,
                score=round(score, 2),
                detail=f"Closest number differed by {best_diff:.4f} (tolerance {tolerance})",
                checker_type="numeric_tolerance",
            )

        return CheckerResult(
            passed=False,
            score=0.0,
            detail=f"Expected {expected} ± {tolerance}, closest was {min(found_numbers, key=lambda n: abs(n - expected))}",
            checker_type="numeric_tolerance",
        )


class RubricChecker:
    """Deterministic rubric checker evaluating keyword coverage, structure, and format."""

    @staticmethod
    def evaluate(output_text: str, config: dict[str, Any]) -> CheckerResult:
        clean_text = output_text.strip().lower()
        required_keywords = config.get("required_keywords", [])
        prohibited_keywords = config.get(
            "prohibited_keywords", config.get("forbidden_keywords", [])
        )

        # Check prohibited keywords
        for p in prohibited_keywords:
            if p.lower() in clean_text:
                return CheckerResult(
                    passed=False,
                    score=0.0,
                    detail=f"Contained prohibited keyword '{p}'",
                    checker_type="rubric",
                )

        if not required_keywords:
            # Non-empty baseline check
            passed = len(clean_text) > 20
            return CheckerResult(
                passed=passed,
                score=1.0 if passed else 0.2,
                detail="Non-empty rubric baseline satisfied" if passed else "Output too short",
                checker_type="rubric",
            )

        matched_count = sum(1 for kw in required_keywords if kw.lower() in clean_text)
        coverage_ratio = matched_count / len(required_keywords)

        # Also check JSON validity if required
        if config.get("require_json"):
            try:
                json.loads(output_text)
            except Exception:
                coverage_ratio *= 0.5

        passed = coverage_ratio >= 0.60
        return CheckerResult(
            passed=passed,
            score=round(coverage_ratio, 2),
            detail=f"Matched {matched_count}/{len(required_keywords)} required rubric keywords",
            checker_type="rubric",
        )


def evaluate_output(
    output_text: str, item_or_config: DatasetItem | dict[str, Any]
) -> CheckerResult:
    """Dispatches evaluation of an output text against a DatasetItem or configuration dictionary."""
    if isinstance(item_or_config, DatasetItem):
        ctype = item_or_config.checker_type
        cfg = item_or_config.checker_config
    else:
        ctype = item_or_config.get("checker_type", "exact")
        cfg = item_or_config

    if ctype == "exact":
        return ExactMatchChecker.evaluate(output_text, cfg)
    if ctype == "regex":
        return RegexChecker.evaluate(output_text, cfg)
    if ctype == "numeric_tolerance":
        return NumericToleranceChecker.evaluate(output_text, cfg)
    if ctype == "rubric":
        return RubricChecker.evaluate(output_text, cfg)

    # Fallback to non-empty
    passed = bool(output_text and output_text.strip())
    return CheckerResult(
        passed=passed,
        score=1.0 if passed else 0.0,
        detail="Generic non-empty checker fallback",
        checker_type="fallback",
    )
