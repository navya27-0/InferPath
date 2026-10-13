"""Deterministic output length constraint validator."""

from typing import Any

from inferpath.models.domain import UserRequest, ValidationResult


class LengthValidator:
    """
    Enforces explicit minimum and maximum character bounds without treating token estimates as measured.
    Bounds can be supplied in task_parameters or request metadata.
    """

    name: str = "length_validator"

    def __init__(self, default_min_chars: int = 1, default_max_chars: int = 50_000):
        self.default_min_chars = default_min_chars
        self.default_max_chars = default_max_chars

    def validate(self, output_text: str, request: UserRequest, **kwargs: Any) -> ValidationResult:
        min_chars = request.metadata.get("min_length", self.default_min_chars)
        max_chars = request.metadata.get("max_length", self.default_max_chars)

        text_len = len(output_text)
        if text_len < min_chars:
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Output length {text_len} characters is less than required minimum {min_chars}.",
                validator_name=self.name,
                failure_code="LENGTH_TOO_SHORT",
                fallback_recommended=True,
            )

        if text_len > max_chars:
            return ValidationResult(
                is_valid=False,
                rejection_reason=f"Output length {text_len} characters exceeds maximum bound of {max_chars}.",
                validator_name=self.name,
                failure_code="LENGTH_TOO_LONG",
                fallback_recommended=True,
            )

        return ValidationResult(
            is_valid=True,
            validator_name=self.name,
        )
