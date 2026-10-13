"""Domain objects for requests, routing decisions, handler outputs, and pipeline results."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, model_validator

from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue


class TaskParameters(BaseModel):
    """Task parameters and user preferences required for routing and caching."""

    max_tokens: int | None = None
    expected_format: str = "text"  # e.g., "text", "json", "markdown", "code"
    language: str = "en"
    audience: str = "general"  # e.g., "general", "technical", "child", "expert"
    quality_preference: str = "balanced"  # e.g., "fast", "balanced", "high"
    max_latency_ms: float | None = None


class UserRequest(BaseModel):
    """Normalized client request submitted to the InferPath router."""

    request_id: str
    prompt: str
    context: str | None = None
    task_parameters: TaskParameters = Field(default_factory=TaskParameters)
    context_hash: str | None = None
    privacy_class: PrivacyClass = PrivacyClass.PUBLIC
    allow_external_transmission: bool = False
    is_time_sensitive: bool | None = None  # None indicates automatic detection needed
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def quality(self) -> str:
        return self.task_parameters.quality_preference

    @property
    def format(self) -> str:
        return self.task_parameters.expected_format

    @property
    def language(self) -> str:
        return self.task_parameters.language

    @property
    def latency(self) -> float | None:
        return self.task_parameters.max_latency_ms

    @property
    def preferences(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "format": self.format,
            "language": self.language,
            "latency": self.latency,
            "privacy_class": self.privacy_class.value,
        }


class RoutingDecision(BaseModel):
    """The decision made during Stage 3 (Analyze and Select Eligible Route)."""

    selected_tier: RouteTier
    route: RouteTier | None = None
    eligible_tiers: list[RouteTier] = Field(default_factory=list)
    reason: str
    rule_id: str | None = None  # Explicit rule identifier (e.g., "R01", "R02") from router table
    confidence: float
    time_sensitive: bool = False
    cache_eligible: bool = True
    template_id: str | None = None
    target_model_name: str | None = None

    @model_validator(mode="after")
    def sync_route_and_tier(self) -> "RoutingDecision":
        if self.route is None:
            self.route = self.selected_tier
        elif self.selected_tier is None:
            self.selected_tier = self.route
        return self


class ModelResult(BaseModel):
    """Standardized output produced by ModelAdapter implementations (Prompt 1)."""

    text: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd_calculated: float
    provider_reported_cost_usd: float | None = None
    price_table_date: str
    source: ProvenanceLabel = ProvenanceLabel.SIMULATED
    model_id: str


class HandlerOutput(BaseModel):
    """Standardized output produced by any route handler (A, B, C, or D)."""

    tier: RouteTier
    output_text: str
    text: str | None = None
    route: RouteTier | None = None
    cache_hit: bool = False
    validated: bool = True
    fallback_used: bool = False
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    input_tokens: ProvenanceValue[int]
    output_tokens: ProvenanceValue[int]
    latency_ms: ProvenanceValue[float]
    time_to_first_token_ms: ProvenanceValue[float] | None = None
    model_version: str
    raw_response: dict[str, Any] | None = None

    @model_validator(mode="after")
    def sync_output_aliases(self) -> "HandlerOutput":
        if self.text is None:
            self.text = self.output_text
        if self.route is None:
            self.route = self.tier
        return self


class ValidationResult(BaseModel):
    """Outcome of Stage 6: Output Validation."""

    is_valid: bool
    rejection_reason: str | None = None
    fallback_recommended: bool = False
    next_recommended_tier: RouteTier | None = None


class FallbackRecord(BaseModel):
    """Audit entry documenting an escalation step during Stage 7."""

    from_tier: RouteTier
    to_tier: RouteTier
    rejection_reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineResult(BaseModel):
    """Final artifact emitted by the 9-stage pipeline."""

    request_id: str
    prompt_hash: str
    final_output: str
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    initial_tier: RouteTier
    final_tier: RouteTier
    was_cached: bool = False
    fallback_occurred: bool = False
    fallback_history: list[FallbackRecord] = Field(default_factory=list)
    total_latency_ms: ProvenanceValue[float]
    cost_usd: ProvenanceValue[float]
    energy_wh: ProvenanceValue[float]
    water_ml: ProvenanceValue[float]
    privacy_class: PrivacyClass
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def text(self) -> str:
        return self.final_output

    @property
    def route(self) -> RouteTier:
        return self.final_tier

    @property
    def cache_hit(self) -> bool:
        return self.was_cached

    @property
    def fallback_used(self) -> bool:
        return self.fallback_occurred


# Canonical Prompt 1 specification aliases
Request = UserRequest
RouteDecision = RoutingDecision
Response = HandlerOutput
RunRecord = PipelineResult
