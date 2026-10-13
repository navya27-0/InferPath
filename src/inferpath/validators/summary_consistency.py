"""Optional summary consistency validator comparing key facts in source with summary."""

import re
from typing import Any

from inferpath.models.domain import UserRequest, ValidationResult


class SummaryConsistencyValidator:
    """
    Optional summary consistency check that verifies key entities/terms from the source context
    are preserved in the generated summary.

    Notice:
    This check is optional, bounded by the run budget, and explicitly noted:
    A model self-check or heuristic check is NOT independent proof of factual correctness.
    """

    name: str = "summary_consistency"

    def validate(self, output_text: str, request: UserRequest, **kwargs: Any) -> ValidationResult:
        enabled = request.metadata.get("enable_summary_consistency", False)
        if not enabled:
            return ValidationResult(is_valid=True, validator_name=self.name)

        source_text = request.context or request.prompt
        if not source_text or len(source_text.split()) < 10:
            return ValidationResult(is_valid=True, validator_name=self.name)

        # Extract capitalized entities / key numbers from source
        source_entities = set(re.findall(r"\b[A-Z][a-z]{2,}\b|\b\d+(?:\.\d+)?\b", source_text))
        # Filter common stopwords
        common_words = {"The", "This", "That", "There", "When", "What", "Where", "Which", "With"}
        key_entities = {e for e in source_entities if e not in common_words}

        if not key_entities:
            return ValidationResult(is_valid=True, validator_name=self.name)

        summary_lower = output_text.lower()
        matched = [e for e in key_entities if e.lower() in summary_lower]
        overlap_ratio = len(matched) / len(key_entities)

        # If overlap is extremely low (< 10%) on a long text, flag inconsistency
        if len(key_entities) >= 5 and overlap_ratio < 0.10:
            return ValidationResult(
                is_valid=False,
                rejection_reason=(
                    f"Summary consistency check failed: only {len(matched)} of {len(key_entities)} "
                    f"key facts/entities preserved ({overlap_ratio:.1%})."
                ),
                validator_name=self.name,
                failure_code="SUMMARY_INCONSISTENCY",
                fallback_recommended=True,
            )

        return ValidationResult(
            is_valid=True,
            validator_name=self.name,
        )
