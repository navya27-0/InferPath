"""Composable response validator interface and composite validator pipeline."""

from typing import Any, Protocol, runtime_checkable

from inferpath.models.domain import UserRequest, ValidationResult
from inferpath.models.enums import RouteTier


@runtime_checkable
class ResponseValidator(Protocol):
    """Protocol for deterministic and model response validators."""

    name: str

    def validate(self, output_text: str, request: UserRequest, **kwargs: Any) -> ValidationResult:
        """Validates output against explicit rules and expectations."""
        ...


class CompositeValidator:
    """Chains multiple validators in sequence, stopping on first failure."""

    def __init__(self, validators: list[ResponseValidator] | None = None):
        self.validators = validators or []

    def add_validator(self, validator: ResponseValidator) -> None:
        self.validators.append(validator)

    def validate(
        self,
        output_text: str,
        request: UserRequest,
        current_tier: RouteTier = RouteTier.SMALL_MODEL_C,
    ) -> ValidationResult:
        """
        Runs all configured validators in order.
        Returns first failure with fallback recommended for Tier C, or successful result.
        """
        for v in self.validators:
            result = v.validate(output_text, request)
            if not result.is_valid:
                # If current tier is small model, recommend escalation to large model
                if current_tier == RouteTier.SMALL_MODEL_C:
                    result.fallback_recommended = True
                    result.next_recommended_tier = RouteTier.LARGE_MODEL_D
                else:
                    result.fallback_recommended = False
                    result.next_recommended_tier = None
                return result

        return ValidationResult(
            is_valid=True,
            rejection_reason=None,
            validator_name="composite",
            failure_code=None,
            fallback_recommended=False,
            next_recommended_tier=None,
        )
