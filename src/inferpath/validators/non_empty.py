"""Deterministic non-empty response validator."""

from typing import Any

from inferpath.models.domain import UserRequest, ValidationResult


class NonEmptyValidator:
    """Rejects empty, whitespace-only, or null outputs."""

    name: str = "non_empty"

    def validate(self, output_text: str, request: UserRequest, **kwargs: Any) -> ValidationResult:
        if not output_text or not output_text.strip():
            return ValidationResult(
                is_valid=False,
                rejection_reason="Output is empty or contains only whitespace.",
                validator_name=self.name,
                failure_code="EMPTY_OUTPUT",
                fallback_recommended=True,
            )
        return ValidationResult(
            is_valid=True,
            validator_name=self.name,
        )
