"""Domain objects for requests, routing decisions, handler outputs, and pipeline results."""

import math
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

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
    minimum_quality: float = Field(default=0.70, ge=0.0, le=1.0)
    maximum_cost_usd: float | None = None
    maximum_latency_ms: float | None = None

    @field_validator("minimum_quality", "maximum_cost_usd", "maximum_latency_ms", "max_latency_ms")
    @classmethod
    def validate_finite_constraints(cls, v: float | None, info) -> float | None:
        if v is None:
            return v
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be a finite number, got {v}")
        if info.field_name == "minimum_quality" and not (0.0 <= v <= 1.0):
            raise ValueError(f"minimum_quality must be between 0.0 and 1.0, got {v}")
        if info.field_name == "maximum_cost_usd" and (v < 0.0 or v > 1000.0):
            raise ValueError(f"maximum_cost_usd must be between 0.0 and 1000.0, got {v}")
        if info.field_name in ("maximum_latency_ms", "max_latency_ms") and (
            v < 0.0 or v > 300_000.0
        ):
            raise ValueError(f"{info.field_name} must be between 0.0 and 300000.0, got {v}")
        return v


class UserRequest(BaseModel):
    """Normalized client request submitted to the InferPath router."""

    request_id: str
    prompt: str
    context: str | None = None
    task_parameters: TaskParameters = Field(default_factory=TaskParameters)
    context_hash: str | None = None
    privacy_class: PrivacyClass = PrivacyClass.PUBLIC
    allow_external_transmission: bool | None = None
    is_time_sensitive: bool | None = None  # None indicates automatic detection needed
    user_scope: str = "default"
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def set_default_external_transmission(self) -> "UserRequest":
        if self.allow_external_transmission is None:
            self.allow_external_transmission = self.privacy_class == PrivacyClass.PUBLIC
        return self

    @property
    def quality(self) -> str:
        return self.task_parameters.quality_preference

    @property
    def minimum_quality(self) -> float:
        return self.task_parameters.minimum_quality

    @property
    def maximum_cost_usd(self) -> float | None:
        return self.task_parameters.maximum_cost_usd

    @property
    def maximum_latency_ms(self) -> float | None:
        if self.task_parameters.maximum_latency_ms is not None:
            return self.task_parameters.maximum_latency_ms
        return self.task_parameters.max_latency_ms

    @property
    def format(self) -> str:
        return self.task_parameters.expected_format

    @property
    def language(self) -> str:
        return self.task_parameters.language

    @property
    def latency(self) -> float | None:
        return self.maximum_latency_ms

    @property
    def preferences(self) -> dict[str, Any]:
        return {
            "quality": self.quality,
            "minimum_quality": self.minimum_quality,
            "maximum_cost_usd": self.maximum_cost_usd,
            "maximum_latency_ms": self.maximum_latency_ms,
            "format": self.format,
            "language": self.language,
            "latency": self.latency,
            "privacy_class": self.privacy_class.value,
        }


class PrivacyDecision(BaseModel):
    """Explicit privacy decision governing execution, external transmission, caching, and auditing."""

    allow_local_processing: bool = True
    allow_external_transmission: bool = False
    allow_cache: bool = False
    allow_prompt_preview: bool = False
    allow_telemetry: bool = True
    reason: str | None = None


class RoutingDecision(BaseModel):
    """The decision made during Step 3 (Analyze and Select Eligible Route)."""

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
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    constraints_satisfied: bool = True
    unsatisfied_constraints: list[str] = Field(default_factory=list)

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
    input_tokens: int | None = None
    output_tokens: int | None = None
    latency_ms: float
    cost_usd_calculated: float | None = None
    provider_reported_cost_usd: float | None = None
    price_table_date: str
    source: ProvenanceLabel = ProvenanceLabel.SIMULATED
    model_id: str
    usage_available: bool = True


class ClarificationRequest(BaseModel):
    """Structured request emitted when required template or task inputs are missing."""

    template_id: str | None = None
    task_intent: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    provided_fields: dict[str, Any] = Field(default_factory=dict)
    clarification_prompt: str
    needs_clarification: bool = True


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
    clarification_request: ClarificationRequest | None = None
    input_tokens: ProvenanceValue[int | None]
    output_tokens: ProvenanceValue[int | None]
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
    validator_name: str | None = None
    failure_code: str | None = None
    fallback_recommended: bool = False
    next_recommended_tier: RouteTier | None = None


class FallbackRecord(BaseModel):
    """Audit entry documenting an escalation step during Stage 7."""

    from_tier: RouteTier
    to_tier: RouteTier
    rejection_reason: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))


class PipelineResult(BaseModel):
    """Final artifact emitted by the 11-step request lifecycle pipeline."""

    request_id: str
    prompt_hash: str
    final_output: str
    needs_clarification: bool = False
    clarification_prompt: str | None = None
    initial_tier: RouteTier
    final_tier: RouteTier
    rule_id: str | None = None
    was_cached: bool = False
    validation_passed: bool = True
    fallback_occurred: bool = False
    fallback_history: list[FallbackRecord] = Field(default_factory=list)
    route_trace: list[dict[str, Any]] = Field(default_factory=list)
    total_latency_ms: ProvenanceValue[float]
    cost_usd: ProvenanceValue[float | None]
    energy_wh: ProvenanceValue[float | None]
    water_ml: ProvenanceValue[float | None]
    input_tokens: ProvenanceValue[int | None] | None = None
    output_tokens: ProvenanceValue[int | None] | None = None
    provider_reported_cost_usd: ProvenanceValue[float | None] | None = None
    estimated_call_cost_usd: ProvenanceValue[float | None] | None = None
    actual_calculated_cost_usd: ProvenanceValue[float | None] | None = None
    estimated_quality: float | None = None
    validated_correctness: bool = True
    empirical_quality: float | None = None
    constraints_satisfied: bool = True
    unsatisfied_constraints: list[str] = Field(default_factory=list)
    requested_minimum_quality: float | None = None
    requested_maximum_cost_usd: float | None = None
    requested_maximum_latency_ms: float | None = None
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
