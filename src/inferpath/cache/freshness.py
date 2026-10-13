"""Rule-based freshness classifier detecting time-sensitive queries."""

import re

# Temporal keywords and phrases that invalidate static caching
TEMPORAL_PATTERNS = [
    re.compile(r"\b(today|tonight|this morning|this afternoon|this evening)\b", re.IGNORECASE),
    re.compile(r"\b(now|right now|currently|current)\b", re.IGNORECASE),
    re.compile(r"\b(latest|recent|recently|yesterday|tomorrow)\b", re.IGNORECASE),
    re.compile(
        r"\b(breaking news|headline news|live score|live sports|game score)\b", re.IGNORECASE
    ),
    re.compile(r"\b(weather|forecast|temperature outside)\b", re.IGNORECASE),
    re.compile(
        r"\b(stock price|share price|ticker price|crypto price|bitcoin price|exchange rate)\b",
        re.IGNORECASE,
    ),
]


class FreshnessClassifier:
    """
    Rule-based analyzer detecting time-sensitive intent in prompts.
    Time-sensitive requests bypass the exact cache completely to prevent serving stale data.
    """

    @classmethod
    def is_time_sensitive(cls, prompt: str) -> tuple[bool, str | None]:
        """
        Evaluates whether a prompt is time-sensitive.
        Returns: (is_time_sensitive, reason_string_if_matched).
        """
        if not prompt or not prompt.strip():
            return False, None

        for pattern in TEMPORAL_PATTERNS:
            match = pattern.search(prompt)
            if match:
                matched_token = match.group(0)
                return True, f"Time-sensitive temporal keyword detected: '{matched_token}'"

        return False, None
