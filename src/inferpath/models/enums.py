"""Enumerations defining route tiers, provenance labels, and privacy classifications."""

from enum import StrEnum


class RouteTier(StrEnum):
    """The four operational routing paths in InferPath."""

    DETERMINISTIC_A = "deterministic_a"  # Tier A: Deterministic rules, mathematical eval, APIs
    TEMPLATE_B = "template_b"  # Tier B: Template engine with strict slot filling
    SMALL_MODEL_C = "small_model_c"  # Tier C: Low-resource LLM (e.g. Gemini Flash / Llama 8B)
    LARGE_MODEL_D = "large_model_d"  # Tier D: High-capability LLM (e.g. Claude Sonnet / GPT-4o)


class ProvenanceLabel(StrEnum):
    """Provenance tagging for reported metric values."""

    MEASURED = "measured"  # Directly observed in real execution (tokens, latency, status)
    CALCULATED = "calculated"  # Derived deterministically from measured values + dated price table
    ESTIMATED = (
        "estimated"  # Derived from model assumptions with low/central/high ranges (energy, water)
    )
    SIMULATED = "simulated"  # Produced by mock adapters or synthetic runs
    PROVIDER_REPORTED = "provider_reported"  # Reported directly by upstream commercial provider API
    UNAVAILABLE = "unavailable"  # Usage or metric data missing, unparseable, or not reported


class PrivacyClass(StrEnum):
    """Data classification levels enforcing caching and transmission boundaries."""

    PUBLIC = "public"  # External API permitted, cache eligible, text logged
    INTERNAL = "internal"  # No external transmission, no caching, hash-only logging
    CONFIDENTIAL_PRIVATE = (
        "confidential_private"  # Strict isolation, no external transmission, no cache, hash-only
    )


class TaskType(StrEnum):
    """Classified nature of the incoming query."""

    ARITHMETIC_MATH = "arithmetic_math"
    UNIT_CONVERSION = "unit_conversion"
    STRING_MANIPULATION = "string_manipulation"
    TEMPLATE_SLOT = "template_slot"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    COMPLEX_REASONING = "complex_reasoning"
    OPEN_ENDED_QA = "open_ended_qa"
    UNKNOWN = "unknown"
