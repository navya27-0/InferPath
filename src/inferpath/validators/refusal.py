"""Deterministic refusal detector for LLM completions."""

import re
from typing import Any

from inferpath.models.domain import UserRequest, ValidationResult

REFUSAL_PATTERNS = [
    re.compile(
        r"\b(i cannot fulfill this request|as an ai(?: language model)?,? i cannot|i am unable to fulfill|i cannot assist with|i must decline|i apologize, but i cannot|i'm sorry, but i cannot|i am not able to help with that)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(i cannot provide instructions for|i cannot generate (?:harmful|dangerous|illegal)|i cannot comply with this request)\b",
        re.IGNORECASE,
    ),
]


class RefusalDetector:
    """
    Identifies explicit model refusal phrases and distinguishes them from ordinary valid content.
    """

    name: str = "refusal_detector"

    def validate(self, output_text: str, request: UserRequest, **kwargs: Any) -> ValidationResult:
        clean = output_text.strip()

        # Check if the prompt itself asked about refusals (e.g. safety analysis)
        prompt_lower = request.prompt.lower()
        if "refusal" in prompt_lower or "declining requests" in prompt_lower:
            return ValidationResult(is_valid=True, validator_name=self.name)

        for pattern in REFUSAL_PATTERNS:
            match = pattern.search(clean)
            if match:
                refusal_phrase = match.group(0)
                return ValidationResult(
                    is_valid=False,
                    rejection_reason=f"Model emitted explicit refusal: '{refusal_phrase}'.",
                    validator_name=self.name,
                    failure_code="MODEL_REFUSAL",
                    fallback_recommended=True,
                )

        return ValidationResult(
            is_valid=True,
            validator_name=self.name,
        )
