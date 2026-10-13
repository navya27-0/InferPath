"""Deterministic format constraint validator (JSON and required headings)."""

import json
import re
from typing import Any

from inferpath.models.domain import UserRequest, ValidationResult


class FormatValidator:
    """
    Validates format constraints:
    1. If expected_format is 'json', verifies that the output is syntactically valid JSON.
    2. If required_headings is specified in request metadata, verifies their presence.
    """

    name: str = "format_validator"

    def validate(self, output_text: str, request: UserRequest, **kwargs: Any) -> ValidationResult:
        expected_format = (
            request.task_parameters.expected_format.lower()
            if request.task_parameters.expected_format
            else "text"
        )
        if request.metadata.get("expected_format"):
            expected_format = str(request.metadata["expected_format"]).lower()

        # 1. JSON validation
        if expected_format == "json":
            clean = output_text.strip()
            if "```json" in clean:
                parts = clean.split("```json")
                if len(parts) > 1:
                    clean = parts[1].split("```")[0].strip()
            elif "```" in clean:
                parts = clean.split("```")
                if len(parts) > 1:
                    clean = parts[1].strip()

            try:
                json.loads(clean)
            except json.JSONDecodeError as err:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Output is not valid JSON as requested: {err}",
                    validator_name=self.name,
                    failure_code="FORMAT_INVALID_JSON",
                    fallback_recommended=True,
                )

        # 2. Required headings validation
        required_headings = request.metadata.get("required_headings") or []
        if isinstance(required_headings, list) and required_headings:
            missing_headings: list[str] = []
            for heading in required_headings:
                pattern = re.compile(rf"(?:^|\n)\s*#*\s*{re.escape(heading)}\b", re.IGNORECASE)
                if not pattern.search(output_text):
                    missing_headings.append(heading)

            if missing_headings:
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Output is missing required headings: {', '.join(missing_headings)}",
                    validator_name=self.name,
                    failure_code="FORMAT_MISSING_HEADINGS",
                    fallback_recommended=True,
                )

        return ValidationResult(
            is_valid=True,
            validator_name=self.name,
        )
