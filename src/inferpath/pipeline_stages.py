"""Modular pipeline stages: cost accumulation, fallback coordination, and result finalization."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from inferpath.models.domain import (
    FallbackRecord,
    HandlerOutput,
    PipelineResult,
    RoutingDecision,
    UserRequest,
)
from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue
from inferpath.router.formulation import DEFAULT_TIER_PRIORS
from inferpath.security.hashing import hash_prompt

if TYPE_CHECKING:
    from inferpath.evaluation.environmental import EnvironmentalEstimator
    from inferpath.security.budget import RunBudgetTracker
    from inferpath.security.privacy import PrivacyPolicyEnforcer


@dataclass
class CostAccumulator:
    """
    Accumulates cost, token usage, and latency across multiple execution attempts (e.g. fallbacks).
    Prevents understating total request cost when a small-model request fails validation and escalates.
    """

    total_cost_usd: float | None = 0.0
    cost_available: bool = True
    total_input_tokens: int | None = 0
    total_output_tokens: int | None = 0
    tokens_available: bool = True
    provider_reported_cost_usd: float | None = None
    call_records: list[dict[str, Any]] = field(default_factory=list)

    def record_call(self, tier: RouteTier, output: HandlerOutput) -> None:
        """Records resource consumption from an individual tier execution attempt."""
        cost: float | None = None
        has_cost_entry = False

        if output.raw_response and "cost_usd_calculated" in output.raw_response:
            raw_c = output.raw_response["cost_usd_calculated"]
            has_cost_entry = True
            if raw_c is not None:
                try:
                    cost = float(raw_c)
                except (ValueError, TypeError):
                    cost = None
            else:
                cost = None
        elif tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B):
            cost = 0.0
            has_cost_entry = True

        # Accumulate total cost if cost is available
        if cost is not None and self.cost_available:
            self.total_cost_usd = (self.total_cost_usd or 0.0) + cost
        elif tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D) and (
            cost is None or not has_cost_entry
        ):
            self.cost_available = False
            self.total_cost_usd = None

        # Token accumulation
        in_tok = output.input_tokens.value if output.input_tokens else None
        out_tok = output.output_tokens.value if output.output_tokens else None
        if in_tok is not None and self.tokens_available:
            self.total_input_tokens = (self.total_input_tokens or 0) + in_tok
        elif tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D) and in_tok is None:
            self.tokens_available = False
            self.total_input_tokens = None

        if out_tok is not None and self.tokens_available:
            self.total_output_tokens = (self.total_output_tokens or 0) + out_tok
        elif tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D) and out_tok is None:
            self.tokens_available = False
            self.total_output_tokens = None

        if (
            output.raw_response
            and output.raw_response.get("provider_reported_cost_usd") is not None
        ):
            try:
                p_cost = float(output.raw_response["provider_reported_cost_usd"])
                self.provider_reported_cost_usd = (self.provider_reported_cost_usd or 0.0) + p_cost
            except (ValueError, TypeError):
                pass

        self.call_records.append(
            {
                "tier": tier,
                "cost_usd": cost,
                "cost_available": cost is not None,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "latency_ms": output.latency_ms.value if output.latency_ms else 0.0,
                "model_version": output.model_version,
            }
        )


class FallbackCoordinator:
    """Evaluates whether one-level fallback escalation (Tier C -> Tier D) is permitted."""

    def __init__(
        self,
        routing_settings: Any,
        privacy_enforcer: PrivacyPolicyEnforcer,
        large_model_id: str,
        budget_tracker: RunBudgetTracker | None = None,
        model_registry: Any | None = None,
    ):
        self.routing_settings = routing_settings
        self.privacy_enforcer = privacy_enforcer
        self.large_model_id = large_model_id
        self.budget_tracker = budget_tracker
        self.model_registry = model_registry

    def can_attempt_fallback(
        self,
        request: UserRequest,
        decision: RoutingDecision,
        current_tier: RouteTier,
        val_result: Any,
        route_trace: list[dict[str, Any]],
    ) -> bool:
        """Step 9: Checks whether fallback escalation from Tier C to Tier D is authorized."""
        if (
            val_result.is_valid
            or not self.routing_settings.enable_fallback
            or current_tier != RouteTier.SMALL_MODEL_C
        ):
            return False

        # Privacy check for fallback
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_tx, _ = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.LARGE_MODEL_D
            )
            if not can_tx:
                route_trace.append({"step": "fallback_blocked", "reason": "privacy_policy"})
                return False

        # Budget check for fallback
        if self.budget_tracker:
            est_d_cost = self.budget_tracker.estimate_call_cost(
                self.large_model_id,
                request.prompt,
                request.task_parameters.max_tokens,
            )
            if est_d_cost > self.budget_tracker.remaining_budget:
                route_trace.append({"step": "fallback_blocked", "reason": "budget_exhaustion"})
                return False

        # Caller constraints and eligibility check for fallback
        if decision.eligible_tiers and RouteTier.LARGE_MODEL_D not in decision.eligible_tiers:
            route_trace.append({"step": "fallback_blocked", "reason": "tier_d_ineligible"})
            return False

        # Dynamically lookup capabilities from model registry or configuration
        meta = self.model_registry.get_model(self.large_model_id) if self.model_registry else None
        max_quality_limit = getattr(
            getattr(self.routing_settings, "fallback", None),
            "max_quality_without_fallback",
            None,
        ) or (meta.quality_prior if meta else 0.96)
        expected_latency_limit = getattr(
            getattr(self.routing_settings, "fallback", None),
            "large_model_expected_latency_ms",
            None,
        ) or (meta.p95_latency_ms if meta else 210.0)

        if (
            request.task_parameters.minimum_quality is not None
            and request.task_parameters.minimum_quality > max_quality_limit
        ):
            route_trace.append({"step": "fallback_blocked", "reason": "user_quality_constraint"})
            return False

        if request.task_parameters.maximum_cost_usd is not None:
            est_cost = (
                self.budget_tracker.estimate_call_cost(
                    self.large_model_id,
                    request.prompt,
                    request.task_parameters.max_tokens,
                )
                if self.budget_tracker
                else 0.005
            )
            if est_cost > request.task_parameters.maximum_cost_usd:
                route_trace.append({"step": "fallback_blocked", "reason": "user_budget_constraint"})
                return False

        if (
            request.maximum_latency_ms is not None
            and request.maximum_latency_ms < expected_latency_limit
        ):
            route_trace.append({"step": "fallback_blocked", "reason": "user_latency_constraint"})
            return False

        return True


class ResultFinalizer:
    """Finalizes PipelineResult schemas, aggregating costs and environmental estimates across executions."""

    def __init__(self, estimator: EnvironmentalEstimator):
        self.estimator = estimator

    def finalize(
        self,
        request: UserRequest,
        decision: RoutingDecision,
        output: HandlerOutput,
        was_cached: bool,
        fallback_occurred: bool,
        fallback_history: list[FallbackRecord],
        validation_passed: bool,
        elapsed_ms: float,
        route_trace: list[dict[str, Any]],
        cost_accumulator: CostAccumulator | None = None,
    ) -> PipelineResult:
        """Constructs sanitized PipelineResult with accumulated cost and combined environmental footprints."""
        if cost_accumulator is not None and cost_accumulator.call_records and not was_cached:
            total_cost_usd = cost_accumulator.total_cost_usd
            provider_reported_cost = cost_accumulator.provider_reported_cost_usd
            total_input_tokens = cost_accumulator.total_input_tokens
            total_output_tokens = cost_accumulator.total_output_tokens

            # Check if any model tier had missing tokens
            model_tokens_missing = any(
                rec["tier"] in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D)
                and (rec["input_tokens"] is None or rec["output_tokens"] is None)
                for rec in cost_accumulator.call_records
            )
            if model_tokens_missing:
                final_energy = ProvenanceValue(
                    value=None,
                    provenance=ProvenanceLabel.UNAVAILABLE,
                    unit="Wh",
                    system_boundary=self.estimator.energy_boundary,
                )
                final_water = ProvenanceValue(
                    value=None,
                    provenance=ProvenanceLabel.UNAVAILABLE,
                    unit="mL",
                    system_boundary=self.estimator.total_water_boundary,
                )
            elif len(cost_accumulator.call_records) == 1:
                rec = cost_accumulator.call_records[0]
                call_tokens = (rec["input_tokens"] or 0) + (rec["output_tokens"] or 0)
                est = self.estimator.estimate(
                    tier=rec["tier"],
                    model_id=rec["model_version"],
                    total_tokens=call_tokens,
                    was_cached=False,
                )
                final_energy = est.energy_wh
                final_water = est.total_water_ml
            else:
                agg_energy_wh = 0.0
                agg_water_ml = 0.0
                for record in cost_accumulator.call_records:
                    call_tokens = (record["input_tokens"] or 0) + (record["output_tokens"] or 0)
                    est = self.estimator.estimate(
                        tier=record["tier"],
                        model_id=record["model_version"],
                        total_tokens=call_tokens,
                        was_cached=False,
                    )
                    agg_energy_wh += est.energy_wh.value
                    agg_water_ml += est.total_water_ml.value

                final_energy = ProvenanceValue(
                    value=round(agg_energy_wh, 6),
                    provenance=ProvenanceLabel.ESTIMATED,
                    unit="Wh",
                    system_boundary=self.estimator.energy_boundary,
                )
                final_water = ProvenanceValue(
                    value=round(agg_water_ml, 4),
                    provenance=ProvenanceLabel.ESTIMATED,
                    unit="mL",
                    system_boundary=self.estimator.total_water_boundary,
                )
        else:
            total_cost_usd = 0.0
            provider_reported_cost = None
            if (
                not was_cached
                and output.raw_response
                and "cost_usd_calculated" in output.raw_response
            ):
                raw_c = output.raw_response["cost_usd_calculated"]
                if raw_c is not None:
                    try:
                        total_cost_usd = float(raw_c)
                    except (ValueError, TypeError):
                        total_cost_usd = None
                else:
                    total_cost_usd = None
            elif not was_cached and output.tier in (
                RouteTier.SMALL_MODEL_C,
                RouteTier.LARGE_MODEL_D,
            ):
                total_cost_usd = None

            if (
                not was_cached
                and output.raw_response
                and output.raw_response.get("provider_reported_cost_usd") is not None
            ):
                try:
                    provider_reported_cost = float(
                        output.raw_response["provider_reported_cost_usd"]
                    )
                except (ValueError, TypeError):
                    provider_reported_cost = None

            total_input_tokens = (
                output.input_tokens.value
                if (output.input_tokens and output.input_tokens.value is not None)
                else None
            )
            total_output_tokens = (
                output.output_tokens.value
                if (output.output_tokens and output.output_tokens.value is not None)
                else None
            )
            total_tokens = (total_input_tokens or 0) + (total_output_tokens or 0)
            model_id = (
                output.raw_response.get("model_id") if output.raw_response else output.model_version
            )
            if (
                output.tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D)
                and not was_cached
                and (total_input_tokens is None or total_output_tokens is None)
            ):
                final_energy = ProvenanceValue(
                    value=None,
                    provenance=ProvenanceLabel.UNAVAILABLE,
                    unit="Wh",
                    system_boundary=self.estimator.energy_boundary,
                )
                final_water = ProvenanceValue(
                    value=None,
                    provenance=ProvenanceLabel.UNAVAILABLE,
                    unit="mL",
                    system_boundary=self.estimator.total_water_boundary,
                )
            else:
                env_est = self.estimator.estimate(
                    tier=output.tier,
                    model_id=model_id,
                    total_tokens=total_tokens,
                    was_cached=was_cached,
                )
                final_energy = env_est.energy_wh
                final_water = env_est.total_water_ml

        final_tier = output.tier
        prompt_digest = hash_prompt(request.prompt)

        # Evaluate actual post-execution constraints
        actual_constraints_satisfied = decision.constraints_satisfied
        unsatisfied_actual = list(decision.unsatisfied_constraints)
        if (
            request.maximum_cost_usd is not None
            and total_cost_usd is not None
            and total_cost_usd > request.maximum_cost_usd
        ):
            actual_constraints_satisfied = False
            if "budget" not in unsatisfied_actual:
                unsatisfied_actual.append("budget")
        if request.maximum_latency_ms is not None and elapsed_ms > request.maximum_latency_ms:
            actual_constraints_satisfied = False
            if "latency" not in unsatisfied_actual:
                unsatisfied_actual.append("latency")
        if request.minimum_quality is not None and not validation_passed:
            actual_constraints_satisfied = False
            if "quality" not in unsatisfied_actual:
                unsatisfied_actual.append("quality")

        prior = DEFAULT_TIER_PRIORS.get(final_tier)
        est_qual = prior.quality_prior if prior else None

        cost_val = round(total_cost_usd, 6) if total_cost_usd is not None else None
        cost_prov = (
            ProvenanceLabel.CALCULATED
            if total_cost_usd is not None
            else ProvenanceLabel.UNAVAILABLE
        )

        return PipelineResult(
            request_id=request.request_id,
            prompt_hash=prompt_digest,
            final_output=output.output_text,
            needs_clarification=output.needs_clarification,
            clarification_prompt=output.clarification_prompt,
            initial_tier=decision.selected_tier,
            final_tier=final_tier,
            rule_id=decision.rule_id,
            was_cached=was_cached,
            validation_passed=validation_passed,
            fallback_occurred=fallback_occurred,
            fallback_history=fallback_history,
            route_trace=route_trace,
            total_latency_ms=ProvenanceValue(
                value=round(elapsed_ms, 2),
                provenance=ProvenanceLabel.MEASURED,
                unit="ms",
            ),
            cost_usd=ProvenanceValue(
                value=cost_val,
                provenance=cost_prov,
                unit="USD",
            ),
            energy_wh=final_energy,
            water_ml=final_water,
            privacy_class=request.privacy_class,
            input_tokens=ProvenanceValue(
                value=total_input_tokens,
                provenance=output.input_tokens.provenance
                if (output.input_tokens and total_input_tokens is not None)
                else ProvenanceLabel.UNAVAILABLE,
                unit="tokens",
            )
            if output.input_tokens
            else None,
            output_tokens=ProvenanceValue(
                value=total_output_tokens,
                provenance=output.output_tokens.provenance
                if (output.output_tokens and total_output_tokens is not None)
                else ProvenanceLabel.UNAVAILABLE,
                unit="tokens",
            )
            if output.output_tokens
            else None,
            provider_reported_cost_usd=ProvenanceValue(
                value=round(provider_reported_cost, 6),
                provenance=ProvenanceLabel.PROVIDER_REPORTED,
                unit="USD",
            )
            if provider_reported_cost is not None
            else None,
            estimated_call_cost_usd=ProvenanceValue(
                value=round(float(output.raw_response["estimated_call_cost_usd"]), 6)
                if (
                    output.raw_response
                    and output.raw_response.get("estimated_call_cost_usd") is not None
                )
                else cost_val,
                provenance=ProvenanceLabel.CALCULATED
                if (
                    output.raw_response
                    and output.raw_response.get("estimated_call_cost_usd") is not None
                )
                or cost_val is not None
                else ProvenanceLabel.UNAVAILABLE,
                unit="USD",
            )
            if (
                output.raw_response
                and output.raw_response.get("estimated_call_cost_usd") is not None
            )
            or cost_val is not None
            else None,
            actual_calculated_cost_usd=ProvenanceValue(
                value=cost_val,
                provenance=cost_prov,
                unit="USD",
            ),
            constraints_satisfied=actual_constraints_satisfied,
            unsatisfied_constraints=unsatisfied_actual,
            requested_minimum_quality=request.minimum_quality,
            requested_maximum_cost_usd=request.maximum_cost_usd,
            requested_maximum_latency_ms=request.maximum_latency_ms,
            estimated_quality=est_qual,
            validated_correctness=validation_passed,
            empirical_quality=None,
        )
