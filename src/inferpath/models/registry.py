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

from inferpath.models.enums import ProvenanceLabel, RouteTier


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
    price_provenance: ProvenanceLabel = ProvenanceLabel.CALCULATED
    latency_provenance: ProvenanceLabel = ProvenanceLabel.ESTIMATED
    quality_provenance: ProvenanceLabel = ProvenanceLabel.ESTIMATED
    capability_provenance: ProvenanceLabel = ProvenanceLabel.ESTIMATED


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
                input_cost_per_1m=0.075,
                output_cost_per_1m=0.30,
                context_window_tokens=1048576,
                p50_latency_ms=28.0,
                p95_latency_ms=75.0,
                quality_prior=0.74,
                capabilities=["fast_inference", "large_context", "multilingual", "json_mode"],
                price_provenance=ProvenanceLabel.CALCULATED,
            )
        )
        self.register_model(
            ModelMetadata(
                provider="meta",
                model_id="meta-llama/llama-3.1-8b-instruct",
                version_tag="2026-07",
                pricing_date="2026-10",
                tier_mapping=RouteTier.SMALL_MODEL_C,
                input_cost_per_1m=0.055,
                output_cost_per_1m=0.055,
                context_window_tokens=128000,
                p50_latency_ms=35.0,
                p95_latency_ms=90.0,
                quality_prior=0.71,
                capabilities=["fast_inference", "open_weights", "instruction_following"],
                price_provenance=ProvenanceLabel.CALCULATED,
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
                price_provenance=ProvenanceLabel.CALCULATED,
            )
        )
        self.register_model(
            ModelMetadata(
                provider="openai",
                model_id="openai/gpt-4o",
                version_tag="2026-05",
                pricing_date="2026-10",
                tier_mapping=RouteTier.LARGE_MODEL_D,
                input_cost_per_1m=2.50,
                output_cost_per_1m=10.00,
                context_window_tokens=128000,
                p50_latency_ms=75.0,
                p95_latency_ms=190.0,
                quality_prior=0.93,
                capabilities=["multimodal", "fast_frontier", "structured_json", "reasoning"],
                price_provenance=ProvenanceLabel.CALCULATED,
            )
        )
        # Test fixture / mock models
        self.register_model(
            ModelMetadata(
                provider="mock",
                model_id="mock-small",
                version_tag="2026-10",
                pricing_date="2026-10",
                tier_mapping=RouteTier.SMALL_MODEL_C,
                input_cost_per_1m=0.05,
                output_cost_per_1m=0.20,
                context_window_tokens=8192,
                p50_latency_ms=10.0,
                p95_latency_ms=25.0,
                quality_prior=0.70,
                capabilities=["simulated_deterministic"],
                price_provenance=ProvenanceLabel.SIMULATED,
            )
        )
        self.register_model(
            ModelMetadata(
                provider="mock",
                model_id="mock-large",
                version_tag="2026-10",
                pricing_date="2026-10",
                tier_mapping=RouteTier.LARGE_MODEL_D,
                input_cost_per_1m=2.50,
                output_cost_per_1m=10.00,
                context_window_tokens=16384,
                p50_latency_ms=20.0,
                p95_latency_ms=50.0,
                quality_prior=0.90,
                capabilities=["simulated_deterministic"],
                price_provenance=ProvenanceLabel.SIMULATED,
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
