"""Response validators package."""

from inferpath.validators.base import CompositeValidator, ResponseValidator
from inferpath.validators.format import FormatValidator
from inferpath.validators.length import LengthValidator
from inferpath.validators.non_empty import NonEmptyValidator
from inferpath.validators.refusal import RefusalDetector
from inferpath.validators.summary_consistency import SummaryConsistencyValidator


def create_default_validator() -> CompositeValidator:
    """Builds standard composite validator pipeline."""
    return CompositeValidator(
        validators=[
            NonEmptyValidator(),
            LengthValidator(),
            FormatValidator(),
            RefusalDetector(),
            SummaryConsistencyValidator(),
        ]
    )


__all__ = [
    "ResponseValidator",
    "CompositeValidator",
    "NonEmptyValidator",
    "LengthValidator",
    "FormatValidator",
    "RefusalDetector",
    "SummaryConsistencyValidator",
    "create_default_validator",
]
