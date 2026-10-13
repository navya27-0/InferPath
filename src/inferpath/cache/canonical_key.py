"""Canonical cache key generation with conservative normalization and deterministic hashing."""

import hashlib
import json
import re
import unicodedata
from typing import Any

from inferpath.models.domain import TaskParameters
from inferpath.models.enums import PrivacyClass, RouteTier


def normalize_prompt(prompt: str) -> str:
    """
    Conservatively normalizes prompt text:
    1. Unicode NFC normalization (canonical composition).
    2. Normalize linebreaks (\\r\\n and \\r to \\n).
    3. Collapse multiple consecutive horizontal whitespace (spaces/tabs) into a single space.
    4. Strip leading and trailing whitespace.

    CRITICAL INVARIANTS:
    - Never changes casing (preserves quotes, acronyms, and proper nouns).
    - Never strips punctuation or symbols.
    - Never alters numbers or units.
    - Meaning-changing differences produce distinct strings.
    """
    if not prompt:
        return ""

    # 1. Unicode NFC normalization
    normalized = unicodedata.normalize("NFC", prompt)

    # 2. Standardize newlines
    normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")

    # 3. Collapse multiple horizontal spaces per line and trim
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in normalized.split("\n")]

    # 4. Filter empty leading/trailing lines and join
    return "\n".join(lines).strip()


class CanonicalKeyBuilder:
    """
    Computes strict SHA-256 cache keys over canonical JSON payloads.
    Ensures that any variation in prompt, parameters, scope, route, or version produces a distinct key.
    """

    @classmethod
    def build_cache_key(
        cls,
        prompt: str,
        task_parameters: TaskParameters | dict[str, Any] | None,
        context_hash: str | None = None,
        user_scope: str = "default",
        route: RouteTier | str = "",
        handler_version: str = "v1",
        template_version: str | None = None,
        privacy_class: PrivacyClass | str = PrivacyClass.PUBLIC,
    ) -> str:
        """Computes the deterministic SHA-256 cache key hex digest."""
        key, _, _ = cls.build_canonical_descriptor(
            prompt=prompt,
            task_parameters=task_parameters,
            context_hash=context_hash,
            user_scope=user_scope,
            route=route,
            handler_version=handler_version,
            template_version=template_version,
            privacy_class=privacy_class,
        )
        return key

    @classmethod
    def build_canonical_descriptor(
        cls,
        prompt: str,
        task_parameters: TaskParameters | dict[str, Any] | None,
        context_hash: str | None = None,
        user_scope: str = "default",
        route: RouteTier | str = "",
        handler_version: str = "v1",
        template_version: str | None = None,
        privacy_class: PrivacyClass | str = PrivacyClass.PUBLIC,
    ) -> tuple[str, str, str]:
        """
        Builds the canonical JSON string and SHA-256 key.
        Returns: (cache_key, canonical_json_string, normalized_prompt).
        """
        norm_prompt = normalize_prompt(prompt)

        # Normalize task parameters into a stable sorted dictionary
        if isinstance(task_parameters, TaskParameters):
            params_dict = {
                "audience": task_parameters.audience,
                "expected_format": task_parameters.expected_format,
                "language": task_parameters.language,
                "max_latency_ms": task_parameters.max_latency_ms,
                "max_tokens": task_parameters.max_tokens,
                "quality_preference": task_parameters.quality_preference,
            }
        elif isinstance(task_parameters, dict):
            params_dict = {k: v for k, v in sorted(task_parameters.items())}
        else:
            params_dict = {}

        route_str = route.value if isinstance(route, RouteTier) else str(route)
        privacy_str = (
            privacy_class.value if isinstance(privacy_class, PrivacyClass) else str(privacy_class)
        )

        # Construct canonical dictionary with stable keys
        canonical_payload = {
            "context_hash": context_hash.strip()
            if context_hash and context_hash.strip()
            else "none",
            "handler_version": handler_version.strip() if handler_version else "v1",
            "normalized_prompt": norm_prompt,
            "privacy_class": privacy_str,
            "route": route_str,
            "task_parameters": params_dict,
            "template_version": template_version.strip() if template_version else "none",
            "user_scope": user_scope.strip() if user_scope else "default",
        }

        # Deterministic JSON serialization: sorted keys, compact separators, UTF-8
        canonical_json = json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

        cache_key = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
        return cache_key, canonical_json, norm_prompt
