"""Dynamic model registry and provider version tracking for InferPath.

Tracks:
- Provider name, model ID, model version, release date, and pricing date.
- Input and output pricing per 1M tokens.
- Context window capacity.
- Capability metadata (reasoning, code, math, multilingual).
- Calibrated latency profile (p50, p95).
- Quality prior distributions across task families.
- Deprecation status and replacement model pointer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from inferpath.models.enums import RouteTier


@dataclass
class ModelMetadata:
    provider: str
    model_id: str
    version_tag: str
    pricing_date: str
    tier_mapping: RouteTier
    input_cost_per_1m: float
    output_cost_per_1m: float
    context_window_tokens: int
    p50_latency_ms: float
    p95_latency_ms: float
    quality_prior: float  # [0.0, 1.0]
    capabilities: list[str] = field(default_factory=list)
    is_deprecated: bool = False
    replacement_model_id: str | None = None
    last_evaluated: str = "2026-10-19"


class ModelRegistry:
    """Manages registered provider models, tracks drift, and provides lookup for routing."""

    def __init__(self) -> None:
        self._models: dict[str, ModelMetadata] = {}
        self._initialize_canonical_models()

    def _initialize_canonical_models(self) -> None:
        # Tier C: Small / Flash models
        self.register_model(
            ModelMetadata(
                provider="google",
                model_id="google/gemini-2.5-flash",
                version_tag="2026-06",
                pricing_date="2026-10",
                tier_mapping=RouteTier.SMALL_MODEL_C,
                input_cost_per_1m=0.10,
                output_cost_per_1m=0.40,
                context_window_tokens=1048576,
                p50_latency_ms=28.0,
                p95_latency_ms=75.0,
                quality_prior=0.74,
                capabilities=["fast_inference", "large_context", "multilingual", "json_mode"],
            )
        )
        # Tier D: Frontier models
        self.register_model(
            ModelMetadata(
                provider="anthropic",
                model_id="anthropic/claude-3.7-sonnet",
                version_tag="2026-03",
                pricing_date="2026-10",
                tier_mapping=RouteTier.LARGE_MODEL_D,
                input_cost_per_1m=3.00,
                output_cost_per_1m=15.00,
                context_window_tokens=200000,
                p50_latency_ms=85.0,
                p95_latency_ms=210.0,
                quality_prior=0.94,
                capabilities=[
                    "deep_reasoning",
                    "advanced_coding",
                    "system_design",
                    "nuanced_writing",
                ],
            )
        )

    def register_model(self, meta: ModelMetadata) -> None:
        self._models[meta.model_id] = meta

    def get_model(self, model_id: str) -> ModelMetadata | None:
        return self._models.get(model_id)

    def list_models_for_tier(self, tier: RouteTier) -> list[ModelMetadata]:
        return [m for m in self._models.values() if m.tier_mapping == tier and not m.is_deprecated]

    def simulate_price_drift(self, model_id: str, price_multiplier: float) -> None:
        """Simulates price change in commercial market (e.g. 50% price cut)."""
        if model_id in self._models:
            m = self._models[model_id]
            m.input_cost_per_1m *= price_multiplier
            m.output_cost_per_1m *= price_multiplier

    def simulate_capability_leap(self, model_id: str, new_quality: float) -> None:
        """Simulates small model upgrade surpassing older large models."""
        if model_id in self._models:
            self._models[model_id].quality_prior = new_quality

    def deprecate_model(self, model_id: str, replacement_id: str) -> None:
        """Marks model as deprecated and redirects to successor."""
        if model_id in self._models:
            self._models[model_id].is_deprecated = True
            self._models[model_id].replacement_model_id = replacement_id
