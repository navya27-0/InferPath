"""Telemetry and provenance-wrapped metric structures."""

from typing import Generic, TypeVar

from pydantic import BaseModel, model_validator

from inferpath.models.enums import ProvenanceLabel

T = TypeVar("T")


class RangeBounds(BaseModel, Generic[T]):
    """Empirical or statistical uncertainty bounds for estimated values."""

    low: T
    central: T
    high: T

    @model_validator(mode="after")
    def validate_bounds_order(self) -> "RangeBounds[T]":
        if (
            isinstance(self.low, (int, float))
            and isinstance(self.central, (int, float))
            and isinstance(self.high, (int, float))
            and (self.low > self.central or self.central > self.high)
        ):
            raise ValueError(
                f"Range bounds must satisfy low <= central <= high. Got low={self.low}, central={self.central}, high={self.high}"
            )
        return self


class BoundaryDefinition(BaseModel):
    """Explicit declaration of what is included and excluded in an environmental estimate."""

    included: str
    excluded: str
    notes: str | None = None


class ProvenanceValue(BaseModel, Generic[T]):
    """
    Strongly-typed wrapper ensuring no numeric value can exist without an explicit
    provenance tag, unit, and optional formula/system boundary documentation.
    """

    value: T
    provenance: ProvenanceLabel
    unit: str
    formula_or_source: str | None = None
    system_boundary: BoundaryDefinition | None = None
    range_bounds: RangeBounds[T] | None = None

    @model_validator(mode="after")
    def validate_provenance_rules(self) -> "ProvenanceValue[T]":
        # Rule: Estimated values (energy, water) MUST provide a system boundary
        if self.provenance == ProvenanceLabel.ESTIMATED and self.system_boundary is None:
            raise ValueError(
                "Estimated values must state an explicit system boundary (included and excluded factors)."
            )
        # Rule: Never allow simulated to masquerade as measured
        if self.provenance == ProvenanceLabel.MEASURED and self.range_bounds is not None:
            raise ValueError(
                "Measured values represent direct observations and cannot carry estimated range bounds."
            )
        return self


# Backward-compatible alias
MetricValue = ProvenanceValue
