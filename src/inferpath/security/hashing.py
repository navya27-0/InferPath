"""Cryptographic hashing and text scrubbing utilities for security and privacy."""

import hashlib
import re

API_KEY_PATTERNS = [
    re.compile(r"sk-or-v1-[a-zA-Z0-9]{32,}", re.IGNORECASE),
    re.compile(r"sk-[a-zA-Z0-9]{32,}", re.IGNORECASE),
    re.compile(r"(?:api_key|bearer|token)[\s:=]+([a-zA-Z0-9_\-]{16,})", re.IGNORECASE),
]


def hash_text(text: str) -> str:
    """Computes standard hexadecimal SHA-256 digest of input text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_prompt(prompt: str) -> str:
    """Produces SHA-256 digest of normalized prompt string."""
    normalized = " ".join(prompt.strip().split())
    return hash_text(normalized)


def hash_context(context: str | None) -> str:
    """Produces SHA-256 digest of context string, or 'none' if empty."""
    if not context or not context.strip():
        return "none"
    return hash_text(context.strip())


def scrub_api_keys(text: str) -> str:
    """Redacts known API key patterns from log strings and error messages."""
    sanitized = text
    for pattern in API_KEY_PATTERNS:
        sanitized = pattern.sub("[REDACTED_API_KEY]", sanitized)
    return sanitized
