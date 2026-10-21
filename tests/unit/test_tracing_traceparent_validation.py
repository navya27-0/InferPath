"""Comprehensive negative and positive validation tests for W3C traceparent parsing."""

import pytest

from inferpath.telemetry.tracing import Tracer


@pytest.fixture
def tracer():
    return Tracer(service_name="test-tracer")


def test_valid_traceparent(tracer):
    header = "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01"
    parsed = tracer.parse_w3c_traceparent(header)
    assert parsed == ("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7")


def test_valid_traceparent_uppercase_normalized(tracer):
    header = "00-4BF92F3577B34DA6A3CE929D0E0E4736-00F067AA0BA902B7-01"
    parsed = tracer.parse_w3c_traceparent(header)
    assert parsed == ("4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7")


@pytest.mark.parametrize(
    "invalid_header",
    [
        None,
        "",
        "   ",
        # Non-hex characters
        "00-4bf92f3577b34da6a3ce929d0e0e473z-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902bz-01",
        "0g-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-0g",
        # All zeros
        "00-00000000000000000000000000000000-00f067aa0ba902b7-01",
        "00-4bf92f3577b34da6a3ce929d0e0e4736-0000000000000000-01",
        # Forbidden version ff
        "ff-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
        # Extra fields on version 00
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01-extra",
        # Field length violations
        "00-4bf92f3577b34da6a3ce929d0e0e473-00f067aa0ba902b7-01",  # 31 chars
        "00-4bf92f3577b34da6a3ce929d0e0e4736a-00f067aa0ba902b7-01",  # 33 chars
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b-01",  # 15 chars
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7a-01",  # 17 chars
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-1",  # 1 char flag
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-001",  # 3 char flag
        # Missing fields
        "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7",
        "00-4bf92f3577b34da6a3ce929d0e0e4736",
        # Delimiter mismatch
        "00_4bf92f3577b34da6a3ce929d0e0e4736_00f067aa0ba902b7_01",
    ],
)
def test_invalid_traceparents_rejected(tracer, invalid_header):
    assert tracer.parse_w3c_traceparent(invalid_header) is None
