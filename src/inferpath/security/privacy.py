"""Privacy policy enforcement and privacy-preserving audit sanitization."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import yaml

from inferpath.models.domain import PrivacyDecision, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.security.hashing import hash_prompt, hash_text, scrub_api_keys


class PrivacyViolationError(PermissionError):
    """Raised when an operation violates active privacy policies."""

    def __init__(
        self,
        message: str,
        privacy_class: str | None = None,
        target_tier: str | None = None,
        code: str = "PRIVACY_BLOCKED",
    ):
        clean_msg = scrub_api_keys(message)
        super().__init__(clean_msg)
        self.message = clean_msg
        self.privacy_class = privacy_class
        self.target_tier = target_tier
        self.code = code

    def __str__(self) -> str:
        return self.message


def parse_classification_confidence(raw_val: Any) -> float | None:
    """Parses and validates caller-supplied classification confidence.

    Returns a finite float between 0.0 and 1.0, or None if invalid/unparseable/non-finite.
    """
    if raw_val is None:
        return None
    try:
        val = float(raw_val)
        if math.isnan(val) or math.isinf(val):
            return None
        if val < 0.0 or val > 1.0:
            return None
        return val
    except (TypeError, ValueError):
        return None


class PrivacyPolicyEnforcer:
    """Enforces non-negotiable privacy boundaries:

    1. Requests classified INTERNAL or CONFIDENTIAL_PRIVATE are blocked from external transmission
       by default. Redacted private data is NOT sent externally unless an approved exception is active.
    2. Private requests are forbidden from being cached.
    3. Private request prompt text is never logged beyond a cryptographic SHA-256 hash.
    4. API keys are scrubbed and never persisted.
    """

    def __init__(self, policy_config_path: str | None = None):
        self.policy_config_path = policy_config_path
        self.policies: dict[str, Any] = {}
        self.approved_exceptions: dict[str, Any] = {}
        self._load_policy()

    def _load_policy(self) -> None:
        if self.policy_config_path and Path(self.policy_config_path).exists():
            with open(self.policy_config_path, encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
                self.policies = data.get("privacy_classes", {})
                self.approved_exceptions = data.get("approved_exceptions", {})
        else:
            # Safe in-memory defaults
            self.policies = {
                "public": {"allow_external_transmission": True, "allow_caching": True},
                "internal": {"allow_external_transmission": False, "allow_caching": False},
                "confidential_private": {
                    "allow_external_transmission": False,
                    "allow_caching": False,
                },
            }
            self.approved_exceptions = {
                "permitted_external_namespaces": [],
                "allow_redacted_transmission_override": False,
            }

    def redact_text(self, text: str | None) -> str:
        """Redacts sensitive credentials, PII patterns, and confidential markers from text."""
        if not text:
            return ""

        redacted = text
        # 1. API Keys, Tokens & Hashes
        redacted = re.sub(r"sk-[a-zA-Z0-9_\-]{16,}", "[REDACTED_API_KEY]", redacted)
        redacted = re.sub(
            r"(?i)\b(bearer\s+)[a-zA-Z0-9_\-\.]{16,}", r"\1[REDACTED_TOKEN]", redacted
        )
        redacted = re.sub(
            r"(?i)\b(api[_\-]?key[=:\s]+)[a-zA-Z0-9_\-]{8,}", r"\1[REDACTED_KEY]", redacted
        )
        redacted = re.sub(
            r"\b(ghp_[a-zA-Z0-9]{20,}|xox[baprs]-[a-zA-Z0-9_\-]+)\b",
            "[REDACTED_TOKEN]",
            redacted,
        )

        # 2. SSN & National IDs
        redacted = re.sub(r"\b\d{3}-\d{2}-\d{4}\b", "[REDACTED_SSN]", redacted)
        redacted = re.sub(
            r"(?i)\b(ssn|social\s+security\s+number)[=:\s]+\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
            r"\1=[REDACTED_SSN]",
            redacted,
        )

        # 3. Credit Card / CVV
        redacted = re.sub(r"\b(?:\d{4}[-\s]?){3}\d{4}\b", "[REDACTED_CREDIT_CARD]", redacted)
        redacted = re.sub(r"(?i)\b(cvv|cvc)[=:\s]+\d{3,4}\b", r"\1=[REDACTED_CVV]", redacted)

        # 4. Email addresses
        redacted = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "[REDACTED_EMAIL]",
            redacted,
        )

        # 5. Passwords / Secrets / Credentials
        redacted = re.sub(
            r"(?i)\b(pass(?:word|wd)|pwd)[=:\s]+[^\s,;]+", r"\1=[REDACTED_PASSWORD]", redacted
        )
        redacted = re.sub(
            r"(?i)\b(trade[_\s]?secret|confidential[_\s]?secret|client[_\s]?secret|secret[_\s]?key|private[_\s]?key)[=:\s]+[^\s,;]+",
            r"\1=[REDACTED_SECRET]",
            redacted,
        )

        # 6. Obfuscated sensitive patterns (de-spaced letters, e.g. 's e c r e t', 'p 4 s s w 0 r d')
        redacted = re.sub(
            r"(?i)\b(?:s\s*e\s*c\s*r\s*e\s*t|p\s*4\s*s\s*s\s*w\s*0\s*r\s*d)\b",
            "[REDACTED_SECRET]",
            redacted,
        )
        redacted = re.sub(r"(?i)\b(?:trade[_\s]?secret)\b", "[REDACTED_CONFIDENTIAL]", redacted)

        # 7. Apply analyzer SENSITIVE_CONTENT_PATTERNS defensively
        from inferpath.router.analyzer import SENSITIVE_CONTENT_PATTERNS

        for pattern in SENSITIVE_CONTENT_PATTERNS:
            redacted = pattern.sub("[REDACTED_SENSITIVE]", redacted)

        return redacted

    def redact_request_for_transmission(self, request: UserRequest) -> UserRequest:
        """Constructs a verified redacted copy of UserRequest for external transmission."""
        redacted_prompt = self.redact_text(request.prompt)
        redacted_context = self.redact_text(request.context) if request.context else None
        redacted_metadata: dict[str, Any] = {}
        for k, v in request.metadata.items():
            redacted_metadata[k] = self.redact_text(str(v))

        return request.model_copy(
            update={
                "prompt": redacted_prompt,
                "context": redacted_context,
                "metadata": redacted_metadata,
            }
        )

    def verify_redacted(self, text: str) -> bool:
        """Verifies that redacted text contains no residual sensitive patterns."""
        from inferpath.router.analyzer import contains_sensitive_patterns

        return not contains_sensitive_patterns(text)

    def evaluate_privacy(
        self, request: UserRequest, allow_preview_in_profile: bool = True
    ) -> PrivacyDecision:
        """Computes a unified, unambiguous PrivacyDecision across routing, caching, logging, and fallback."""
        from inferpath.router.analyzer import contains_sensitive_patterns

        text_items = [request.prompt]
        if request.context:
            text_items.append(request.context)
        if request.metadata:
            text_items.extend(f"{k}: {v}" for k, v in request.metadata.items())
        combined = " \n ".join(text_items)
        has_sensitive_pattern = contains_sensitive_patterns(combined)

        # Explicit policy boundary: treat uncertain classification as defensive local-only
        raw_confidence = request.metadata.get("classification_confidence")
        if raw_confidence is not None:
            parsed_conf = parse_classification_confidence(raw_confidence)
            # If invalid (non-numeric, NaN, out of bounds) or below 0.7 threshold, treat as uncertain
            is_confidence_uncertain = (parsed_conf is None) or (parsed_conf < 0.7)
        else:
            is_confidence_uncertain = False

        is_uncertain = bool(
            request.metadata.get("uncertain_classification", False)
            or is_confidence_uncertain
            or request.metadata.get("privacy_uncertain", False)
        )
        if is_uncertain:
            return PrivacyDecision(
                allow_local_processing=True,
                allow_external_transmission=False,
                allow_cache=False,
                allow_prompt_preview=False,
                allow_telemetry=True,
                reason="Privacy classification is uncertain; enforced local-only execution and caching prohibited as a defensive boundary.",
            )

        is_private = request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        )
        is_local_only = bool(
            request.metadata.get("local_only", False)
            or (request.allow_external_transmission is False)
        )

        if is_private:
            override_allowed = self.approved_exceptions.get(
                "allow_redacted_transmission_override", False
            )
            ext_tx = bool(request.allow_external_transmission and override_allowed)
            return PrivacyDecision(
                allow_local_processing=True,
                allow_external_transmission=ext_tx,
                allow_cache=False,
                allow_prompt_preview=False,
                allow_telemetry=True,
                reason=(
                    f"Privacy classification '{request.privacy_class.value}' requires redacted transmission override for external transmission; caching prohibited."
                    if ext_tx
                    else f"Privacy classification '{request.privacy_class.value}' strictly prohibits external transmission without an approved override exception; caching prohibited."
                ),
            )

        if has_sensitive_pattern:
            return PrivacyDecision(
                allow_local_processing=True,
                allow_external_transmission=False,
                allow_cache=False,
                allow_prompt_preview=False,
                allow_telemetry=True,
                reason="Sensitive content detected (PII or credential pattern); external transmission and caching prohibited.",
            )

        if is_local_only:
            return PrivacyDecision(
                allow_local_processing=True,
                allow_external_transmission=False,
                allow_cache=True,
                allow_prompt_preview=allow_preview_in_profile,
                allow_telemetry=True,
                reason="Caller requested local-only execution; external transmission prohibited.",
            )

        # Public, no sensitive patterns, external allowed
        return PrivacyDecision(
            allow_local_processing=True,
            allow_external_transmission=True,
            allow_cache=True,
            allow_prompt_preview=allow_preview_in_profile,
            allow_telemetry=True,
            reason="Public request; local execution, external transmission, and caching permitted.",
        )

    def can_transmit_externally(
        self, request: UserRequest, target_tier: RouteTier
    ) -> tuple[bool, str | None]:
        """Determines whether a request may be sent to an external model API (Tiers C or D)."""
        decision = self.evaluate_privacy(request)
        if target_tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B):
            return decision.allow_local_processing, None
        if not decision.allow_external_transmission:
            return False, decision.reason
        return True, None

    def is_cache_permitted(self, request: UserRequest) -> bool:
        """Enforces privacy rule: Private or sensitive requests are NEVER cached."""
        decision = self.evaluate_privacy(request)
        return decision.allow_cache

    def sanitize_for_audit(
        self, request: UserRequest, allow_preview_in_profile: bool = True
    ) -> dict[str, Any]:
        """Sanitizes request metadata for SQLite and JSONL audit logging."""
        prompt_digest = hash_prompt(request.prompt)
        decision = self.evaluate_privacy(request, allow_preview_in_profile=allow_preview_in_profile)
        prompt_sample = (
            scrub_api_keys(request.prompt[:120]) if decision.allow_prompt_preview else None
        )

        # Scrub metadata of credentials and private prompt content
        clean_metadata: dict[str, Any] = {}
        for k, v in request.metadata.items():
            str_v = str(v)
            str_v = scrub_api_keys(str_v)

            # In non-public requests, ensure private prompt text is not leaked via metadata
            if request.privacy_class != PrivacyClass.PUBLIC:
                if request.prompt and request.prompt in str_v:
                    str_v = str_v.replace(request.prompt, "[REDACTED_PRIVATE_PROMPT]")
                lower_k = k.lower()
                if any(
                    sensitive in lower_k
                    for sensitive in ("prompt", "raw_input", "secret", "private", "confidential")
                ):
                    str_v = f"hash:{hash_text(str_v)}"

            clean_metadata[k] = str_v

        return {
            "request_id": request.request_id,
            "prompt_sha256": prompt_digest,
            "prompt_sample": prompt_sample,
            "privacy_class": request.privacy_class.value,
            "metadata": clean_metadata,
        }
