"""OpenTelemetry-compatible distributed tracing with W3C Trace Context support."""

from __future__ import annotations

import contextlib
import os
import secrets
import string
import time
from collections.abc import Generator
from typing import Any

from inferpath.telemetry.structured_logging import current_span_id, current_trace_id

_HEX_CHARS = frozenset(string.hexdigits.lower() + string.hexdigits.upper())


def _is_valid_hex_field(val: str, expected_len: int) -> bool:
    """Verifies that a field has exact expected length and consists solely of hexadecimal characters."""
    if len(val) != expected_len:
        return False
    return all(c in _HEX_CHARS for c in val)


def generate_trace_id() -> str:
    """Generates a standard 128-bit (32 hex character) OpenTelemetry trace ID."""
    return secrets.token_hex(16)


def generate_span_id() -> str:
    """Generates a standard 64-bit (16 hex character) OpenTelemetry span ID."""
    return secrets.token_hex(8)


class Span:
    """Represents a discrete timed execution span in an OpenTelemetry-compatible trace."""

    def __init__(
        self,
        name: str,
        trace_id: str,
        span_id: str,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ):
        self.name = name
        self.trace_id = trace_id
        self.span_id = span_id
        self.parent_span_id = parent_span_id
        self.start_time_ns = time.time_ns()
        self.end_time_ns: int | None = None
        self.status: str = "UNSET"
        self.attributes: dict[str, Any] = attributes.copy() if attributes else {}
        self.events: list[dict[str, Any]] = []

    def set_attribute(self, key: str, value: Any) -> Span:
        """Sets an attribute on this span."""
        self.attributes[key] = value
        return self

    def add_event(self, name: str, attributes: dict[str, Any] | None = None) -> None:
        """Records an event with timestamp within the span lifetime."""
        self.events.append(
            {
                "name": name,
                "timestamp_ns": time.time_ns(),
                "attributes": attributes or {},
            }
        )

    def finish(self, status: str = "OK") -> None:
        """Closes the span and records the end timestamp."""
        if self.end_time_ns is None:
            self.end_time_ns = time.time_ns()
            self.status = status

    @property
    def duration_ms(self) -> float:
        """Duration of span in milliseconds."""
        end = self.end_time_ns or time.time_ns()
        return round((end - self.start_time_ns) / 1_000_000.0, 3)

    def to_dict(self) -> dict[str, Any]:
        """Serializes span data to dictionary representation."""
        return {
            "name": self.name,
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
            "start_time_ns": self.start_time_ns,
            "end_time_ns": self.end_time_ns,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
        }

    def export_w3c_traceparent(self) -> str:
        """Exports W3C Trace Context string: 00-{trace_id}-{span_id}-01."""
        return f"00-{self.trace_id}-{self.span_id}-01"


class Tracer:
    """Lightweight in-process OpenTelemetry tracer."""

    def __init__(self, service_name: str = "inferpath"):
        self.service_name = service_name
        self._completed_spans: list[Span] = []
        self._max_history = 10_000

    def parse_w3c_traceparent(self, header_value: str | None) -> tuple[str, str] | None:
        """Parses and validates W3C traceparent header: version-trace_id-parent_id-trace_flags.

        Strict W3C TraceContext specification compliance:
        - Must be delimited by hyphens.
        - Version '00' must have exactly 4 fields (no extra fields allowed).
        - Version 'ff' is forbidden.
        - trace_id must be exactly 32 hex characters and MUST NOT be all zeros.
        - parent_id (span_id) must be exactly 16 hex characters and MUST NOT be all zeros.
        - trace_flags must be exactly 2 hex characters.

        Returns (trace_id, parent_span_id) if fully valid, or None if malformed/unsupported.
        """
        if not header_value:
            return None
        parts = header_value.strip().split("-")
        if len(parts) < 4:
            return None

        version = parts[0]
        if not _is_valid_hex_field(version, 2):
            return None
        if version.lower() == "ff":
            return None
        if version == "00" and len(parts) != 4:
            return None

        trace_id = parts[1]
        parent_id = parts[2]
        trace_flags = parts[3]

        if not _is_valid_hex_field(trace_id, 32):
            return None
        if not _is_valid_hex_field(parent_id, 16):
            return None
        if not _is_valid_hex_field(trace_flags, 2):
            return None

        if set(trace_id) == {"0"}:
            return None
        if set(parent_id) == {"0"}:
            return None

        return trace_id.lower(), parent_id.lower()

    def start_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Creates a new span under the active or given trace context."""
        active_trace = trace_id or current_trace_id.get() or generate_trace_id()
        active_parent = parent_span_id or current_span_id.get()
        active_span_id = generate_span_id()

        attrs = {"service.name": self.service_name}
        if attributes:
            attrs.update(attributes)

        return Span(
            name=name,
            trace_id=active_trace,
            span_id=active_span_id,
            parent_span_id=active_parent,
            attributes=attrs,
        )

    @contextlib.contextmanager
    def start_as_current_span(
        self,
        name: str,
        trace_id: str | None = None,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Generator[Span, None, None]:
        """Context manager setting active span in contextvars."""
        span = self.start_span(
            name=name,
            trace_id=trace_id,
            parent_span_id=parent_span_id,
            attributes=attributes,
        )
        token_t = current_trace_id.set(span.trace_id)
        token_s = current_span_id.set(span.span_id)
        try:
            yield span
            span.finish(status="OK")
        except Exception as exc:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(exc))
            span.finish(status="ERROR")
            raise
        finally:
            current_trace_id.reset(token_t)
            current_span_id.reset(token_s)
            self._record_completed_span(span)

    def _record_completed_span(self, span: Span) -> None:
        self._completed_spans.append(span)
        if len(self._completed_spans) > self._max_history:
            self._completed_spans.pop(0)

    def get_completed_spans(self) -> list[Span]:
        """Returns list of completed recorded spans."""
        return list(self._completed_spans)

    def clear(self) -> None:
        """Clears completed spans history."""
        self._completed_spans.clear()


# Default process-level tracer
GLOBAL_TRACER = Tracer(service_name=os.getenv("INFERPATH_SERVICE_NAME", "inferpath"))
