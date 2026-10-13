"""Rule-based freshness classifier detecting time-sensitive queries."""

import re

# Temporal keywords and phrases that indicate dynamic or live data requirements
TEMPORAL_PATTERNS = [
    re.compile(
        r"\b(what time is it|current time|current date|what is today's date|what date is it|what day is it|what day is today)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\b(today|tonight|this morning|this afternoon|this evening)\b", re.IGNORECASE),
    re.compile(r"\b(now|right now|currently|current)\b", re.IGNORECASE),
    re.compile(r"\b(latest|recent|recently|yesterday|tomorrow)\b", re.IGNORECASE),
    re.compile(
        r"\b(breaking news|headline news|live score|live sports|game score)\b", re.IGNORECASE
    ),
    re.compile(r"\b(weather|forecast|temperature outside)\b", re.IGNORECASE),
    re.compile(
        r"\b(stock price|share price|ticker price|crypto price|bitcoin price|exchange rate|interest rate|currency exchange)\b"
        r"|\b(?:convert|exchange)\s+\d+(?:\.\d+)?\s*(?:usd|eur|gbp|inr|jpy|cad|aud)\s+(?:to|in|into)\s*(?:usd|eur|gbp|inr|jpy|cad|aud)\b",
        re.IGNORECASE,
    ),
]

# Rhetorical idioms where 'today' or 'now' does not ask for changing/live data
# e.g., "Today we know that dinosaurs went extinct...", "Scientists now understand that..."
RHETORICAL_TIMELESS_PATTERNS = [
    re.compile(
        r"\b(today (?:we|scientists|humanity|historians|researchers|scholars) (?:know|understand|believe|agree|recognize))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b((?:we|scientists|humanity|researchers) now (?:know|understand|understands|believe|believes|recognize|recognizes))\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(now (?:understood|believed|known) that)\b",
        re.IGNORECASE,
    ),
]


class FreshnessClassifier:
    """
    Rule-based analyzer detecting time-sensitive intent in prompts.

    Policy:
    - Time-sensitive requests bypass the exact cache completely to prevent serving stale data.
    - Explicit current-time requests and live-data queries (weather, stocks, rates) are flagged.
    - Rhetorical idioms (e.g., 'today we know that') are filtered to prevent false positive cache misses.
    - When freshness cannot safely be ruled out, a conservative policy applies: mark as time-sensitive.

    Known limitations:
    - False positives: Metaphorical phrases containing temporal tokens may trigger cache bypass.
    - False negatives: Subtly time-dependent queries without overt temporal markers (e.g. "Who is the mayor of Chicago?")
      require external entity linking and cannot be reliably resolved via static regex alone.
    """

    @classmethod
    def is_time_sensitive(cls, prompt: str) -> tuple[bool, str | None]:
        """
        Evaluates whether a prompt is time-sensitive.
        Returns: (is_time_sensitive, reason_string_if_matched).
        """
        if not prompt or not prompt.strip():
            return False, None

        # Check rhetorical idioms first if they contain a temporal word but are timeless
        for rhetorical in RHETORICAL_TIMELESS_PATTERNS:
            if rhetorical.search(prompt):
                # If there are no other overt live-data patterns (like weather, stock price, breaking news, what time is it)
                has_overt_live = any(
                    p.search(prompt) for p in TEMPORAL_PATTERNS[0:1] + TEMPORAL_PATTERNS[4:7]
                )
                if not has_overt_live:
                    return False, None

        for pattern in TEMPORAL_PATTERNS:
            match = pattern.search(prompt)
            if match:
                matched_token = match.group(0)
                return True, f"Time-sensitive temporal keyword detected: '{matched_token}'"

        return False, None
