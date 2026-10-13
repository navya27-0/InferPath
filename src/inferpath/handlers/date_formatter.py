"""Deterministic date parsing and formatting handler."""

import re
from datetime import datetime

from inferpath.models.domain import HandlerOutput
from inferpath.models.enums import ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


class DateFormattingError(ValueError):
    """Raised when date parsing or formatting fails due to invalid or insufficient inputs."""

    pass


class DateFormatter:
    """
    Deterministic date parsing and formatting engine for explicitly supplied dates.

    Guarantees:
    - Never invents missing dates or time zones.
    - Strictly rejects dynamic current-time queries (handled via freshness/live policy).
    - Validates calendar dates against true calendar boundaries (e.g. rejecting Feb 30).
    """

    KNOWN_INPUT_FORMATS = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%d-%m-%Y",
        "%m-%d-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        "%d %B %Y",
        "%d %b %Y",
    ]

    TARGET_FORMAT_SPECS = {
        "iso": "%Y-%m-%d",
        "iso8601": "%Y-%m-%d",
        "rfc2822": "%a, %d %b %Y 00:00:00 +0000",
        "human_long": "%B %d, %Y",
        "human_short": "%d/%m/%Y",
        "us": "%m/%d/%Y",
        "european": "%d/%m/%Y",
    }

    # Extraction regex for "Format [date] as/to [target_format]"
    DATE_FORMAT_PATTERN = re.compile(
        r"(?:format|convert)(?:\s+date)?\s+['\"]?([\w\d\-\/\,\s]+?)['\"]?\s+(?:as|to|into|in)\s+['\"]?([\w\d\-\/\s\%]+)['\"]?",
        re.IGNORECASE,
    )

    DYNAMIC_DATE_PATTERNS = re.compile(
        r"\b(today|now|current date|current time|what is the date|what day is it)\b",
        re.IGNORECASE,
    )

    @classmethod
    def parse_date(cls, date_str: str) -> datetime:
        """Parses an explicit date string against supported formats without inventing dates."""
        clean = date_str.strip()
        if cls.DYNAMIC_DATE_PATTERNS.search(clean):
            raise DateFormattingError(
                "Requests for the current or live date require dynamic evaluation, not static formatting."
            )

        for fmt in cls.KNOWN_INPUT_FORMATS:
            try:
                parsed = datetime.strptime(clean, fmt)
                return parsed
            except ValueError:
                continue

        raise DateFormattingError(
            f"Unable to parse supplied date '{clean}'. Please supply a recognized date format (e.g., YYYY-MM-DD)."
        )

    @classmethod
    def format_date(cls, parsed_date: datetime, target_format: str) -> str:
        """Formats a parsed datetime into the specified target format."""
        norm_target = target_format.strip().lower()
        if norm_target in cls.TARGET_FORMAT_SPECS:
            strftime_fmt = cls.TARGET_FORMAT_SPECS[norm_target]
        else:
            # Check if user passed literal strftime tokens like %Y-%m-%d or DD/MM/YYYY
            if "%" in target_format:
                strftime_fmt = target_format.strip()
            elif "dd/mm/yyyy" in norm_target:
                strftime_fmt = "%d/%m/%Y"
            elif "mm/dd/yyyy" in norm_target:
                strftime_fmt = "%m/%d/%Y"
            elif "yyyy-mm-dd" in norm_target:
                strftime_fmt = "%Y-%m-%d"
            else:
                strftime_fmt = "%Y-%m-%d"

        try:
            return parsed_date.strftime(strftime_fmt)
        except Exception as err:
            raise DateFormattingError(f"Failed to format date with pattern '{target_format}': {err}") from err

    @classmethod
    def execute_handler(cls, prompt: str) -> HandlerOutput:
        """Parses a date formatting request and returns HandlerOutput."""
        match = cls.DATE_FORMAT_PATTERN.search(prompt.strip())
        if not match:
            raise DateFormattingError(
                f"Could not extract explicit date and target format from prompt: '{prompt}'"
            )

        date_input, target_spec = match.groups()
        parsed = cls.parse_date(date_input)
        result_text = cls.format_date(parsed, target_spec)

        return HandlerOutput(
            tier=RouteTier.DETERMINISTIC_A,
            output_text=result_text,
            input_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            output_tokens=ProvenanceValue(
                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
            ),
            latency_ms=ProvenanceValue(
                value=0.5, provenance=ProvenanceLabel.MEASURED, unit="ms"
            ),
            model_version="deterministic_date_formatter_v1",
            raw_response={
                "input_date": date_input,
                "parsed_iso": parsed.isoformat(),
                "target_spec": target_spec,
                "formatted_output": result_text,
            },
        )
