"""Unit tests verifying non-negotiable metric provenance rules."""

import pytest
from pydantic import ValidationError

from inferpath.models.enums import ProvenanceLabel
from inferpath.models.telemetry import (
    BoundaryDefinition,
    ProvenanceValue,
    RangeBounds,
)


def test_provenance_labels_exist():
    """Verifies that all 4 required provenance classes are defined."""
    assert ProvenanceLabel.MEASURED == "measured"
    assert ProvenanceLabel.CALCULATED == "calculated"
    assert ProvenanceLabel.ESTIMATED == "estimated"
    assert ProvenanceLabel.SIMULATED == "simulated"


def test_valid_measured_metric():
    """Measured metric succeeds without boundary or range bounds."""
    metric = ProvenanceValue[int](
        value=150,
        provenance=ProvenanceLabel.MEASURED,
        unit="tokens",
    )
    assert metric.value == 150
    assert metric.provenance == ProvenanceLabel.MEASURED
    assert metric.unit == "tokens"


def test_measured_metric_cannot_have_range_bounds():
    """A measured metric represents observed fact and cannot carry estimated range bounds."""
    with pytest.raises(ValidationError, match="cannot carry estimated range bounds"):
        ProvenanceValue[float](
            value=12.5,
            provenance=ProvenanceLabel.MEASURED,
            unit="ms",
            range_bounds=RangeBounds(low=10.0, central=12.5, high=15.0),
        )


def test_estimated_metric_requires_system_boundary():
    """Estimated metrics must state an explicit system boundary."""
    with pytest.raises(ValidationError, match="must state an explicit system boundary"):
        ProvenanceValue[float](
            value=0.15,
            provenance=ProvenanceLabel.ESTIMATED,
            unit="Wh",
            # system_boundary omitted!
        )


def test_valid_estimated_metric_with_bounds_and_boundary():
    """Valid estimated energy metric with boundary and bounds succeeds."""
    boundary = BoundaryDefinition(
        included="Scope 2 Datacenter operational compute",
        excluded="Client device, network transit, Scope 3 embodied carbon",
    )
    metric = ProvenanceValue[float](
        value=0.15,
        provenance=ProvenanceLabel.ESTIMATED,
        unit="Wh",
        system_boundary=boundary,
        range_bounds=RangeBounds(low=0.05, central=0.15, high=0.35),
    )
    assert metric.value == 0.15
    assert metric.provenance == ProvenanceLabel.ESTIMATED
    assert metric.range_bounds.low == 0.05
    assert metric.range_bounds.high == 0.35
    assert "Scope 2" in metric.system_boundary.included


def test_range_bounds_ordering_validation():
    """RangeBounds rejects invalid low > central or central > high."""
    with pytest.raises(ValidationError, match="low <= central <= high"):
        RangeBounds(low=10.0, central=5.0, high=15.0)

    with pytest.raises(ValidationError, match="low <= central <= high"):
        RangeBounds(low=1.0, central=5.0, high=4.0)
