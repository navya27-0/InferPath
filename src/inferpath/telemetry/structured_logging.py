"""Structured JSON logging with PII/API-key redaction and OpenTelemetry context propagation."""

from __future__ import annotations

import contextvars
import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from inferpath.security.hashing import scrub_api_keys

# Context variables for trace and request propagation across async/thread boundaries
current_trace_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_trace_id", default=None
)
current_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_span_id", default=None
)
current_request_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_request_id", default=None
)


def _sanitize_extra_value(val: Any) -> Any:
    """Recursively scrub API keys and sensitive credentials from log extra fields."""
    if isinstance(val, str):
        return scrub_api_keys(val)
    if isinstance(val, dict):
        return {
            (scrub_api_keys(k) if isinstance(k, str) else k): _sanitize_extra_value(v)
            for k, v in val.items()
        }
    if isinstance(val, (list, tuple, set)):
        return [_sanitize_extra_value(v) for v in val]
    return val


class StructuredJsonFormatter(logging.Formatter):
    """
    Formats standard Python logging records as machine-parseable JSON lines.

    Automatically injects:
    - ISO-8601 UTC timestamp
    - Log level and logger name
    - Scrubbed message body (stripping raw OpenRouter and internal API keys)
    - Active OpenTelemetry trace_id and span_id from contextvars
    - Request ID when operating in request scope
    """

    def __init__(self, service_name: str = "inferpath", environment: str = "production"):
        super().__init__()
        self.service_name = service_name
        self.environment = environment

    def format(self, record: logging.LogRecord) -> str:
        # Format message and apply sensitive pattern / API-key scrubber
        raw_message = record.getMessage()
        safe_message = scrub_api_keys(raw_message)

        log_entry: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": safe_message,
            "service": self.service_name,
            "environment": self.environment,
        }

        # Inject context variables if active
        trace_id = current_trace_id.get()
        if trace_id:
            log_entry["trace_id"] = trace_id

        span_id = current_span_id.get()
        if span_id:
            log_entry["span_id"] = span_id

        request_id = current_request_id.get()
        if request_id:
            log_entry["request_id"] = request_id

        # Attach custom extra fields passed via extra={...}
        reserved = {
            "name",
            "msg",
            "args",
            "levelname",
            "levelno",
            "pathname",
            "filename",
            "module",
            "exc_info",
            "exc_text",
            "stack_info",
            "lineno",
            "funcName",
            "created",
            "msecs",
            "relativeCreated",
            "thread",
            "threadName",
            "processName",
            "process",
            "message",
        }
        extras = {
            (scrub_api_keys(k) if isinstance(k, str) else k): _sanitize_extra_value(v)
            for k, v in record.__dict__.items()
            if k not in reserved
        }
        if extras:
            log_entry["extra"] = extras

        if record.exc_info:
            log_entry["exception"] = scrub_api_keys(self.formatException(record.exc_info))

        return json.dumps(
            log_entry,
            ensure_ascii=False,
            default=lambda o: scrub_api_keys(str(o)),
        )


def configure_structured_logging(
    level: str = "INFO",
    json_format: bool = True,
    service_name: str = "inferpath",
    environment: str = "production",
    log_level: str | None = None,
) -> logging.Logger:
    """Configures root or inferpath logging with StructuredJsonFormatter."""
    effective_level = log_level if log_level is not None else level
    root_logger = logging.getLogger("inferpath")
    numeric_level = getattr(logging, effective_level.upper(), logging.INFO)
    root_logger.setLevel(numeric_level)

    # Avoid duplicate handlers on re-configuration
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    handler = logging.StreamHandler(sys.stdout)
    if json_format:
        handler.setFormatter(
            StructuredJsonFormatter(service_name=service_name, environment=environment)
        )
    else:
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s [%(name)s] %(message)s")
        )

    root_logger.addHandler(handler)
    return root_logger
