"""Single orchestration pipeline for InferPath executing the 11-step routing sequence."""

import time
from typing import Any

from inferpath.cache.canonical_key import CanonicalKeyBuilder
from inferpath.cache.eligibility import CacheEligibilityPolicy
from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings, load_pricing_table, load_settings
from inferpath.evaluation.environmental import EnvironmentalEstimator
from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel, ModelAdapter
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter
from inferpath.models.domain import (
    FallbackRecord,
    HandlerOutput,
    PipelineResult,
    RoutingDecision,
    UserRequest,
)
from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue
from inferpath.router import BaseRouter, RuleBasedRouter
from inferpath.router.analyzer import contains_sensitive_patterns
from inferpath.router.strategies import ConstraintsUnsatisfiableError
from inferpath.security.budget import RunBudgetTracker
from inferpath.security.hashing import hash_prompt
from inferpath.security.privacy import PrivacyPolicyEnforcer, PrivacyViolationError
from inferpath.telemetry.experiment_logger import ExperimentLogger
from inferpath.validators import CompositeValidator, create_default_validator


class PipelineExecutionError(Exception):
    """Raised when pipeline execution encounters an unrecoverable failure."""

    pass


class RequestPipeline:
    """
    Unified 11-step request pipeline:
    1. Validate request schema & required parameters
    2. Enforce privacy policy
    3. Apply freshness policy
    4. Analyze request and select route (RuleBasedRouter)
    5. Build canonical cache key and check eligibility
    6. Exact-cache lookup on eligible cacheable requests (skips Tier A)
    7. Execute selected handler or model adapter on cache miss
    8. Deterministic output validation (CompositeValidator)
    9. Bounded fallback escalation (depth = 1: C -> D once)
    10. Cache validated, eligible results
    11. Structured telemetry and audit logging (ExperimentLogger)
    """

    def __init__(
        self,
        settings: Settings | None = None,
        pricing_table: PricingTable | None = None,
        router: BaseRouter | RuleBasedRouter | None = None,
        cache: ExactCache | None = None,
        validator: CompositeValidator | None = None,
        small_adapter: ModelAdapter | None = None,
        large_adapter: ModelAdapter | None = None,
        logger: ExperimentLogger | None = None,
        budget_tracker: RunBudgetTracker | None = None,
        estimator: EnvironmentalEstimator | None = None,
    ):
        self.settings = settings or load_settings()
        self.pricing_table = pricing_table or load_pricing_table()
        self.privacy_enforcer = PrivacyPolicyEnforcer()
        self.logger = logger or ExperimentLogger()
        self.router = router or RuleBasedRouter(
            routing_settings=self.settings.routing,
            privacy_enforcer=self.privacy_enforcer,
            logger=self.logger,
        )
        self.cache = cache or ExactCache(db_path=self.settings.database_path)
        self.eligibility_policy = CacheEligibilityPolicy()
        self.validator = validator or create_default_validator()
        self.budget_tracker = budget_tracker or RunBudgetTracker(
            max_usd=self.settings.routing.max_usd_per_run,
            pricing_table=self.pricing_table,
        )
        self.estimator = estimator or EnvironmentalEstimator()

        # Adapters: use injected or build based on configuration / mock mode
        self.small_adapter = small_adapter or self._create_small_adapter()
        self.large_adapter = large_adapter or self._create_large_adapter()

    def _create_small_adapter(self) -> ModelAdapter:
        # If no API key is set, use MockSmallModel to guarantee hermetic offline execution
        if not self.settings.openrouter.api_key:
            rates = self.pricing_table.rates_per_1m.get(
                self.settings.openrouter.small_model, {"prompt": 0.075, "completion": 0.30}
            )
            return MockSmallModel(
                model_id=self.settings.openrouter.small_model,
                prompt_price_per_1m=rates["prompt"],
                completion_price_per_1m=rates["completion"],
                simulated_base_latency_ms=120.0,
                price_table_date=self.pricing_table.effective_date,
            )
        return OpenRouterAdapter(
            model_id=self.settings.openrouter.small_model,
            pricing_table=self.pricing_table,
            api_key=self.settings.openrouter.api_key,
            base_url=self.settings.openrouter.base_url,
            timeout_seconds=self.settings.openrouter.timeout_seconds,
            max_retries=self.settings.openrouter.max_retries,
            budget_tracker=self.budget_tracker,
        )

    def _create_large_adapter(self) -> ModelAdapter:
        if not self.settings.openrouter.api_key:
            rates = self.pricing_table.rates_per_1m.get(
                self.settings.openrouter.large_model, {"prompt": 3.0, "completion": 15.0}
            )
            return MockLargeModel(
                model_id=self.settings.openrouter.large_model,
                prompt_price_per_1m=rates["prompt"],
                completion_price_per_1m=rates["completion"],
                simulated_base_latency_ms=450.0,
                price_table_date=self.pricing_table.effective_date,
            )
        return OpenRouterAdapter(
            model_id=self.settings.openrouter.large_model,
            pricing_table=self.pricing_table,
            api_key=self.settings.openrouter.api_key,
            base_url=self.settings.openrouter.base_url,
            timeout_seconds=self.settings.openrouter.timeout_seconds,
            max_retries=self.settings.openrouter.max_retries,
            budget_tracker=self.budget_tracker,
        )

    def execute(self, request: UserRequest) -> PipelineResult:
        """Executes the request through the complete 11-step pipeline."""
        start_time = time.perf_counter()
        route_trace: list[dict[str, Any]] = []

        # Step 1: Validate request schema
        if not request.prompt or not request.prompt.strip():
            raise PipelineExecutionError("Request prompt cannot be empty.")
        prompt_digest = hash_prompt(request.prompt)
        route_trace.append(
            {"step": "validation_schema", "status": "passed", "prompt_hash": prompt_digest}
        )

        # Step 2: Enforce privacy policy
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            # Will be verified again during routing if model tier is selected
            route_trace.append(
                {
                    "step": "privacy_check",
                    "privacy_class": request.privacy_class.value,
                    "allow_external": can_transmit,
                }
            )

        # Step 3 & 4: Analyze and Route
        decision: RoutingDecision = self.router.route(request)
        route_trace.append(
            {
                "step": "route_selection",
                "tier": decision.selected_tier.value,
                "rule_id": decision.rule_id,
                "reason": decision.reason,
                "confidence": decision.confidence,
            }
        )

        # Enforce privacy check on selected route
        can_tx, tx_reason = self.privacy_enforcer.can_transmit_externally(
            request, decision.selected_tier
        )
        if not can_tx or decision.rule_id == "R01":
            raise PrivacyViolationError(
                tx_reason or decision.reason,
                privacy_class=request.privacy_class.value,
                target_tier=decision.selected_tier.value,
            )

        # Enforce constraint satisfaction check before dispatch
        if decision.rule_id == "CONSTRAINTS_UNSATISFIABLE" or not decision.constraints_satisfied:
            route_trace.append(
                {
                    "step": "constraint_check",
                    "status": "unsatisfiable",
                    "eligible_tiers": [t.value for t in decision.eligible_tiers],
                    "unsatisfied_constraints": decision.unsatisfied_constraints,
                    "reason": decision.reason,
                }
            )
            raise ConstraintsUnsatisfiableError(
                message=decision.reason,
                unsatisfied_constraints=decision.unsatisfied_constraints,
                eligible_tiers=decision.eligible_tiers,
            )

        # In addition, check caller constraints if model tier is selected
        q_min = request.task_parameters.minimum_quality if request.task_parameters else None
        b_max = request.task_parameters.maximum_cost_usd if request.task_parameters else None
        l_max = request.task_parameters.maximum_latency_ms if request.task_parameters else None
        if decision.selected_tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
            unsatisfied = []
            if q_min is not None and q_min > 0.96:
                unsatisfied.append("quality")
            if b_max is not None and b_max < 0.00001:
                unsatisfied.append("budget")
            if l_max is not None and l_max < 20.0:
                unsatisfied.append("latency")
            if unsatisfied:
                b_str = f"${b_max:.6f}" if b_max is not None else "None"
                l_str = f"{l_max:.1f}ms" if l_max is not None else "None"
                reason = (
                    f"Constraints unsatisfiable: no candidate tier satisfies all caller constraints "
                    f"(unsatisfied constraints: {', '.join(unsatisfied)}; "
                    f"Q_min={q_min}, B_max={b_str}, L_max={l_str})."
                )
                route_trace.append(
                    {
                        "step": "constraint_check",
                        "status": "unsatisfiable",
                        "eligible_tiers": [],
                        "unsatisfied_constraints": unsatisfied,
                        "reason": reason,
                    }
                )
                raise ConstraintsUnsatisfiableError(
                    message=reason,
                    unsatisfied_constraints=unsatisfied,
                    eligible_tiers=[],
                )

        # Step 5: Canonical cache key and eligibility
        handler_version = self._get_handler_version(decision)
        template_version = "1.0.0" if decision.selected_tier == RouteTier.TEMPLATE_B else None
        cache_key = CanonicalKeyBuilder.build_key(
            normalized_prompt=request.prompt,
            task_parameters=request.task_parameters,
            context_hash=request.context_hash,
            user_scope=request.user_scope,
            selected_route=decision.selected_tier.value,
            handler_version=handler_version,
            template_version=template_version,
            privacy_class=request.privacy_class.value,
        )
        is_cache_eligible, elig_reason = self.eligibility_policy.is_eligible(request, decision)
        route_trace.append(
            {
                "step": "cache_eligibility",
                "is_eligible": is_cache_eligible,
                "reason": elig_reason,
            }
        )

        # Step 6: Exact-cache lookup
        if is_cache_eligible and self.privacy_enforcer.is_cache_permitted(request):
            cached_output = self.cache.get(cache_key)
            if cached_output:
                route_trace.append({"step": "cache_lookup", "status": "hit"})
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return self._finalize_result(
                    request=request,
                    decision=decision,
                    output=cached_output,
                    was_cached=True,
                    fallback_occurred=False,
                    fallback_history=[],
                    validation_passed=True,
                    elapsed_ms=elapsed_ms,
                    route_trace=route_trace,
                )
            route_trace.append({"step": "cache_lookup", "status": "miss"})
        else:
            route_trace.append({"step": "cache_lookup", "status": "bypassed"})

        # Step 7: Execute on cache miss
        current_tier = decision.selected_tier
        output = self._execute_tier(request, decision, current_tier)
        route_trace.append(
            {
                "step": "handler_execution",
                "tier": current_tier.value,
                "latency_ms": output.latency_ms.value,
            }
        )

        # If template required clarification, return immediately without caching
        if output.needs_clarification:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            route_trace.append(
                {"step": "clarification_required", "missing": decision.missing_fields}
            )
            return self._finalize_result(
                request=request,
                decision=decision,
                output=output,
                was_cached=False,
                fallback_occurred=False,
                fallback_history=[],
                validation_passed=False,
                elapsed_ms=elapsed_ms,
                route_trace=route_trace,
            )

        # Step 8: Deterministic output validation
        val_result = self.validator.validate(output.output_text, request, current_tier=current_tier)
        route_trace.append(
            {
                "step": "output_validation",
                "is_valid": val_result.is_valid,
                "failure_code": val_result.failure_code,
                "rejection_reason": val_result.rejection_reason,
            }
        )

        # Step 9: Bounded Fallback (Depth max 1)
        fallback_occurred = False
        fallback_history: list[FallbackRecord] = []

        if (
            not val_result.is_valid
            and self.settings.routing.enable_fallback
            and current_tier == RouteTier.SMALL_MODEL_C
        ):
            can_fallback = True

            # Privacy check for fallback
            if request.privacy_class in (
                PrivacyClass.INTERNAL,
                PrivacyClass.CONFIDENTIAL_PRIVATE,
            ):
                can_tx, _ = self.privacy_enforcer.can_transmit_externally(
                    request, RouteTier.LARGE_MODEL_D
                )
                if not can_tx:
                    can_fallback = False
                    route_trace.append({"step": "fallback_blocked", "reason": "privacy_policy"})

            # Budget check for fallback
            if can_fallback and self.budget_tracker:
                est_d_cost = self.budget_tracker.estimate_call_cost(
                    self.large_adapter.model_id,
                    request.prompt,
                    request.task_parameters.max_tokens,
                )
                if est_d_cost > self.budget_tracker.remaining_budget:
                    can_fallback = False
                    route_trace.append({"step": "fallback_blocked", "reason": "budget_exhaustion"})

            # Caller constraints and eligibility check for fallback
            if can_fallback:
                if (
                    decision.eligible_tiers
                    and RouteTier.LARGE_MODEL_D not in decision.eligible_tiers
                ):
                    can_fallback = False
                    route_trace.append({"step": "fallback_blocked", "reason": "tier_d_ineligible"})
                elif request.task_parameters.maximum_cost_usd is not None:
                    est_cost = (
                        self.budget_tracker.estimate_call_cost(
                            self.large_adapter.model_id,
                            request.prompt,
                            request.task_parameters.max_tokens,
                        )
                        if self.budget_tracker
                        else 0.005
                    )
                    if est_cost > request.task_parameters.maximum_cost_usd:
                        can_fallback = False
                        route_trace.append(
                            {"step": "fallback_blocked", "reason": "user_budget_constraint"}
                        )
                elif (
                    request.task_parameters.maximum_latency_ms is not None
                    and request.task_parameters.maximum_latency_ms < 250.0
                ):
                    can_fallback = False
                    route_trace.append(
                        {"step": "fallback_blocked", "reason": "user_latency_constraint"}
                    )

            if can_fallback:
                fb_rec = FallbackRecord(
                    from_tier=RouteTier.SMALL_MODEL_C,
                    to_tier=RouteTier.LARGE_MODEL_D,
                    rejection_reason=val_result.rejection_reason
                    or "Validation failed on small model.",
                )
                fallback_history.append(fb_rec)
                fallback_occurred = True
                current_tier = RouteTier.LARGE_MODEL_D

                # Execute large model
                output = self._execute_tier(request, decision, RouteTier.LARGE_MODEL_D)
                route_trace.append(
                    {
                        "step": "fallback_execution",
                        "to_tier": RouteTier.LARGE_MODEL_D.value,
                        "latency_ms": output.latency_ms.value,
                    }
                )

                # Re-validate large model output
                val_result = self.validator.validate(
                    output.output_text, request, current_tier=RouteTier.LARGE_MODEL_D
                )
                route_trace.append(
                    {
                        "step": "fallback_validation",
                        "is_valid": val_result.is_valid,
                        "failure_code": val_result.failure_code,
                    }
                )

        # Step 10: Cache validated eligible result
        store_elig = self.eligibility_policy.check_storage_eligibility(
            request=request, route=current_tier, output=output, validation_result=val_result
        )
        if (
            val_result.is_valid
            and store_elig.is_eligible
            and self.privacy_enforcer.is_cache_permitted(request)
            and not output.needs_clarification
        ):
            ttl = self.settings.routing.cache_ttls.get(
                current_tier.value, self.settings.routing.default_cache_ttl_seconds
            )
            self.cache.store(
                cache_key=cache_key,
                request=request,
                output=output,
                route=current_tier,
                ttl_seconds=ttl,
            )
            route_trace.append({"step": "cache_write", "status": "stored", "ttl": ttl})
        else:
            route_trace.append(
                {"step": "cache_write", "status": "bypassed", "reason": store_elig.bypass_reason}
            )

        # Step 11: Finalize outcome and log telemetry
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._finalize_result(
            request=request,
            decision=decision,
            output=output,
            was_cached=False,
            fallback_occurred=fallback_occurred,
            fallback_history=fallback_history,
            validation_passed=val_result.is_valid,
            elapsed_ms=elapsed_ms,
            route_trace=route_trace,
        )

    async def execute_async(self, request: UserRequest) -> PipelineResult:
        """Executes the request through the complete 11-step pipeline asynchronously."""
        start_time = time.perf_counter()
        route_trace: list[dict[str, Any]] = []

        # Step 1: Validate request schema
        if not request.prompt or not request.prompt.strip():
            raise PipelineExecutionError("Request prompt cannot be empty.")
        prompt_digest = hash_prompt(request.prompt)
        route_trace.append(
            {"step": "validation_schema", "status": "passed", "prompt_hash": prompt_digest}
        )

        # Step 2: Enforce privacy policy
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            # Will be verified again during routing if model tier is selected
            route_trace.append(
                {
                    "step": "privacy_check",
                    "privacy_class": request.privacy_class.value,
                    "allow_external": can_transmit,
                }
            )

        # Step 3 & 4: Analyze and Route
        decision: RoutingDecision = self.router.route(request)
        route_trace.append(
            {
                "step": "route_selection",
                "tier": decision.selected_tier.value,
                "rule_id": decision.rule_id,
                "reason": decision.reason,
                "confidence": decision.confidence,
            }
        )

        # Enforce privacy check on selected route
        can_tx, tx_reason = self.privacy_enforcer.can_transmit_externally(
            request, decision.selected_tier
        )
        if not can_tx or decision.rule_id == "R01":
            raise PrivacyViolationError(
                tx_reason or decision.reason,
                privacy_class=request.privacy_class.value,
                target_tier=decision.selected_tier.value,
            )

        # Enforce constraint satisfaction check before dispatch
        if decision.rule_id == "CONSTRAINTS_UNSATISFIABLE" or not decision.constraints_satisfied:
            route_trace.append(
                {
                    "step": "constraint_check",
                    "status": "unsatisfiable",
                    "eligible_tiers": [t.value for t in decision.eligible_tiers],
                    "unsatisfied_constraints": decision.unsatisfied_constraints,
                    "reason": decision.reason,
                }
            )
            raise ConstraintsUnsatisfiableError(
                message=decision.reason,
                unsatisfied_constraints=decision.unsatisfied_constraints,
                eligible_tiers=decision.eligible_tiers,
            )

        # In addition, check caller constraints if model tier is selected
        q_min = request.task_parameters.minimum_quality if request.task_parameters else None
        b_max = request.task_parameters.maximum_cost_usd if request.task_parameters else None
        l_max = request.task_parameters.maximum_latency_ms if request.task_parameters else None
        if decision.selected_tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
            unsatisfied = []
            if q_min is not None and q_min > 0.96:
                unsatisfied.append("quality")
            if b_max is not None and b_max < 0.00001:
                unsatisfied.append("budget")
            if l_max is not None and l_max < 20.0:
                unsatisfied.append("latency")
            if unsatisfied:
                b_str = f"${b_max:.6f}" if b_max is not None else "None"
                l_str = f"{l_max:.1f}ms" if l_max is not None else "None"
                reason = (
                    f"Constraints unsatisfiable: no candidate tier satisfies all caller constraints "
                    f"(unsatisfied constraints: {', '.join(unsatisfied)}; "
                    f"Q_min={q_min}, B_max={b_str}, L_max={l_str})."
                )
                route_trace.append(
                    {
                        "step": "constraint_check",
                        "status": "unsatisfiable",
                        "eligible_tiers": [],
                        "unsatisfied_constraints": unsatisfied,
                        "reason": reason,
                    }
                )
                raise ConstraintsUnsatisfiableError(
                    message=reason,
                    unsatisfied_constraints=unsatisfied,
                    eligible_tiers=[],
                )

        # Step 5: Canonical cache key and eligibility
        handler_version = self._get_handler_version(decision)
        template_version = "1.0.0" if decision.selected_tier == RouteTier.TEMPLATE_B else None
        cache_key = CanonicalKeyBuilder.build_key(
            normalized_prompt=request.prompt,
            task_parameters=request.task_parameters,
            context_hash=request.context_hash,
            user_scope=request.user_scope,
            selected_route=decision.selected_tier.value,
            handler_version=handler_version,
            template_version=template_version,
            privacy_class=request.privacy_class.value,
        )
        is_cache_eligible, elig_reason = self.eligibility_policy.is_eligible(request, decision)
        route_trace.append(
            {
                "step": "cache_eligibility",
                "is_eligible": is_cache_eligible,
                "reason": elig_reason,
            }
        )

        # Step 6: Exact-cache lookup
        if is_cache_eligible and self.privacy_enforcer.is_cache_permitted(request):
            if hasattr(self.cache, "lookup_async"):
                cached_output = await self.cache.lookup_async(cache_key)
            elif hasattr(self.cache, "get_async"):
                cached_output = await self.cache.get_async(cache_key)
            else:
                cached_output = self.cache.get(cache_key)
            if cached_output:
                route_trace.append({"step": "cache_lookup", "status": "hit"})
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                return self._finalize_result(
                    request=request,
                    decision=decision,
                    output=cached_output,
                    was_cached=True,
                    fallback_occurred=False,
                    fallback_history=[],
                    validation_passed=True,
                    elapsed_ms=elapsed_ms,
                    route_trace=route_trace,
                )
            route_trace.append({"step": "cache_lookup", "status": "miss"})
        else:
            route_trace.append({"step": "cache_lookup", "status": "bypassed"})

        # Step 7: Execute on cache miss
        current_tier = decision.selected_tier
        output = await self._execute_tier_async(request, decision, current_tier)
        route_trace.append(
            {
                "step": "handler_execution",
                "tier": current_tier.value,
                "latency_ms": output.latency_ms.value,
            }
        )

        # If template required clarification, return immediately without caching
        if output.needs_clarification:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            route_trace.append(
                {"step": "clarification_required", "missing": decision.missing_fields}
            )
            return self._finalize_result(
                request=request,
                decision=decision,
                output=output,
                was_cached=False,
                fallback_occurred=False,
                fallback_history=[],
                validation_passed=False,
                elapsed_ms=elapsed_ms,
                route_trace=route_trace,
            )

        # Step 8: Deterministic output validation
        val_result = self.validator.validate(output.output_text, request, current_tier=current_tier)
        route_trace.append(
            {
                "step": "output_validation",
                "is_valid": val_result.is_valid,
                "failure_code": val_result.failure_code,
                "rejection_reason": val_result.rejection_reason,
            }
        )

        # Step 9: Bounded Fallback (Depth max 1)
        fallback_occurred = False
        fallback_history: list[FallbackRecord] = []

        if (
            not val_result.is_valid
            and self.settings.routing.enable_fallback
            and current_tier == RouteTier.SMALL_MODEL_C
        ):
            can_fallback = True

            # Privacy check for fallback
            if request.privacy_class in (
                PrivacyClass.INTERNAL,
                PrivacyClass.CONFIDENTIAL_PRIVATE,
            ):
                can_tx, _ = self.privacy_enforcer.can_transmit_externally(
                    request, RouteTier.LARGE_MODEL_D
                )
                if not can_tx:
                    can_fallback = False
                    route_trace.append({"step": "fallback_blocked", "reason": "privacy_policy"})

            # Budget check for fallback
            if can_fallback and self.budget_tracker:
                est_d_cost = self.budget_tracker.estimate_call_cost(
                    self.large_adapter.model_id,
                    request.prompt,
                    request.task_parameters.max_tokens,
                )
                if est_d_cost > self.budget_tracker.remaining_budget:
                    can_fallback = False
                    route_trace.append({"step": "fallback_blocked", "reason": "budget_exhaustion"})

            # Caller constraints and eligibility check for fallback
            if can_fallback:
                if (
                    decision.eligible_tiers
                    and RouteTier.LARGE_MODEL_D not in decision.eligible_tiers
                ):
                    can_fallback = False
                    route_trace.append({"step": "fallback_blocked", "reason": "tier_d_ineligible"})
                elif request.task_parameters.maximum_cost_usd is not None:
                    est_cost = (
                        self.budget_tracker.estimate_call_cost(
                            self.large_adapter.model_id,
                            request.prompt,
                            request.task_parameters.max_tokens,
                        )
                        if self.budget_tracker
                        else 0.005
                    )
                    if est_cost > request.task_parameters.maximum_cost_usd:
                        can_fallback = False
                        route_trace.append(
                            {"step": "fallback_blocked", "reason": "user_budget_constraint"}
                        )
                elif (
                    request.task_parameters.maximum_latency_ms is not None
                    and request.task_parameters.maximum_latency_ms < 250.0
                ):
                    can_fallback = False
                    route_trace.append(
                        {"step": "fallback_blocked", "reason": "user_latency_constraint"}
                    )

            if can_fallback:
                fb_rec = FallbackRecord(
                    from_tier=RouteTier.SMALL_MODEL_C,
                    to_tier=RouteTier.LARGE_MODEL_D,
                    rejection_reason=val_result.rejection_reason
                    or "Validation failed on small model.",
                )
                fallback_history.append(fb_rec)
                fallback_occurred = True
                current_tier = RouteTier.LARGE_MODEL_D

                # Execute large model asynchronously
                output = await self._execute_tier_async(request, decision, RouteTier.LARGE_MODEL_D)
                route_trace.append(
                    {
                        "step": "fallback_execution",
                        "to_tier": RouteTier.LARGE_MODEL_D.value,
                        "latency_ms": output.latency_ms.value,
                    }
                )

                # Re-validate large model output
                val_result = self.validator.validate(
                    output.output_text, request, current_tier=RouteTier.LARGE_MODEL_D
                )
                route_trace.append(
                    {
                        "step": "fallback_validation",
                        "is_valid": val_result.is_valid,
                        "failure_code": val_result.failure_code,
                    }
                )

        # Step 10: Cache validated eligible result
        store_elig = self.eligibility_policy.check_storage_eligibility(
            request=request, route=current_tier, output=output, validation_result=val_result
        )
        if (
            val_result.is_valid
            and store_elig.is_eligible
            and self.privacy_enforcer.is_cache_permitted(request)
            and not output.needs_clarification
        ):
            ttl = self.settings.routing.cache_ttls.get(
                current_tier.value, self.settings.routing.default_cache_ttl_seconds
            )
            if hasattr(self.cache, "store_async"):
                await self.cache.store_async(
                    cache_key=cache_key,
                    request=request,
                    output=output,
                    route=current_tier,
                    ttl_seconds=ttl,
                )
            elif hasattr(self.cache, "set_async"):
                await self.cache.set_async(
                    cache_key=cache_key,
                    output=output,
                    ttl_seconds=ttl,
                    request=request,
                    route=current_tier,
                )
            else:
                self.cache.store(
                    cache_key=cache_key,
                    request=request,
                    output=output,
                    route=current_tier,
                    ttl_seconds=ttl,
                )
            route_trace.append({"step": "cache_write", "status": "stored", "ttl": ttl})
        else:
            route_trace.append(
                {"step": "cache_write", "status": "bypassed", "reason": store_elig.bypass_reason}
            )

        # Step 11: Finalize outcome and log telemetry
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        return self._finalize_result(
            request=request,
            decision=decision,
            output=output,
            was_cached=False,
            fallback_occurred=fallback_occurred,
            fallback_history=fallback_history,
            validation_passed=val_result.is_valid,
            elapsed_ms=elapsed_ms,
            route_trace=route_trace,
        )

    def _execute_tier(
        self, request: UserRequest, decision: RoutingDecision, tier: RouteTier
    ) -> HandlerOutput:
        """Dispatches execution to the corresponding handler or adapter."""
        if tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
            can_tx, tx_reason = self.privacy_enforcer.can_transmit_externally(request, tier)
            if not can_tx:
                raise PrivacyViolationError(
                    tx_reason
                    or f"Blocked external transmission of {request.privacy_class.value} data.",
                    privacy_class=request.privacy_class.value,
                    target_tier=tier.value,
                )

        if tier == RouteTier.DETERMINISTIC_A:
            return self.router.execute_local_handler(request, decision)

        if tier == RouteTier.TEMPLATE_B:
            return self.router.execute_local_handler(request, decision)

        if tier == RouteTier.SMALL_MODEL_C:
            if not isinstance(self.small_adapter, OpenRouterAdapter) and self.budget_tracker:
                est_cost = self.budget_tracker.estimate_call_cost(
                    self.small_adapter.model_id, request.prompt, request.task_parameters.max_tokens
                )
                res_tok = self.budget_tracker.reserve(est_cost)
                try:
                    model_res = self.small_adapter.generate(
                        request.prompt, max_tokens=request.task_parameters.max_tokens
                    )
                except Exception:
                    self.budget_tracker.release(res_tok)
                    raise
                self.budget_tracker.reconcile(res_tok, model_res.cost_usd_calculated)
            else:
                model_res = self.small_adapter.generate(
                    request.prompt, max_tokens=request.task_parameters.max_tokens
                )
            return self._model_result_to_handler_output(model_res, RouteTier.SMALL_MODEL_C)

        if tier == RouteTier.LARGE_MODEL_D:
            if not isinstance(self.large_adapter, OpenRouterAdapter) and self.budget_tracker:
                est_cost = self.budget_tracker.estimate_call_cost(
                    self.large_adapter.model_id, request.prompt, request.task_parameters.max_tokens
                )
                res_tok = self.budget_tracker.reserve(est_cost)
                try:
                    model_res = self.large_adapter.generate(
                        request.prompt, max_tokens=request.task_parameters.max_tokens
                    )
                except Exception:
                    self.budget_tracker.release(res_tok)
                    raise
                self.budget_tracker.reconcile(res_tok, model_res.cost_usd_calculated)
            else:
                model_res = self.large_adapter.generate(
                    request.prompt, max_tokens=request.task_parameters.max_tokens
                )
            return self._model_result_to_handler_output(model_res, RouteTier.LARGE_MODEL_D)

        raise PipelineExecutionError(f"Unsupported execution tier: {tier}")

    async def _execute_tier_async(
        self, request: UserRequest, decision: RoutingDecision, tier: RouteTier
    ) -> HandlerOutput:
        """Dispatches execution asynchronously to the corresponding handler or adapter."""
        if tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
            can_tx, tx_reason = self.privacy_enforcer.can_transmit_externally(request, tier)
            if not can_tx:
                raise PrivacyViolationError(
                    tx_reason
                    or f"Blocked external transmission of {request.privacy_class.value} data.",
                    privacy_class=request.privacy_class.value,
                    target_tier=tier.value,
                )

        if tier == RouteTier.DETERMINISTIC_A:
            return self.router.execute_local_handler(request, decision)

        if tier == RouteTier.TEMPLATE_B:
            return self.router.execute_local_handler(request, decision)

        if tier == RouteTier.SMALL_MODEL_C:
            if not isinstance(self.small_adapter, OpenRouterAdapter) and self.budget_tracker:
                est_cost = self.budget_tracker.estimate_call_cost(
                    self.small_adapter.model_id, request.prompt, request.task_parameters.max_tokens
                )
                res_tok = self.budget_tracker.reserve(est_cost)
                try:
                    model_res = await self.small_adapter.generate_async(
                        request.prompt, max_tokens=request.task_parameters.max_tokens
                    )
                except Exception:
                    self.budget_tracker.release(res_tok)
                    raise
                self.budget_tracker.reconcile(res_tok, model_res.cost_usd_calculated)
            else:
                model_res = await self.small_adapter.generate_async(
                    request.prompt, max_tokens=request.task_parameters.max_tokens
                )
            return self._model_result_to_handler_output(model_res, RouteTier.SMALL_MODEL_C)

        if tier == RouteTier.LARGE_MODEL_D:
            if not isinstance(self.large_adapter, OpenRouterAdapter) and self.budget_tracker:
                est_cost = self.budget_tracker.estimate_call_cost(
                    self.large_adapter.model_id, request.prompt, request.task_parameters.max_tokens
                )
                res_tok = self.budget_tracker.reserve(est_cost)
                try:
                    model_res = await self.large_adapter.generate_async(
                        request.prompt, max_tokens=request.task_parameters.max_tokens
                    )
                except Exception:
                    self.budget_tracker.release(res_tok)
                    raise
                self.budget_tracker.reconcile(res_tok, model_res.cost_usd_calculated)
            else:
                model_res = await self.large_adapter.generate_async(
                    request.prompt, max_tokens=request.task_parameters.max_tokens
                )
            return self._model_result_to_handler_output(model_res, RouteTier.LARGE_MODEL_D)

        raise PipelineExecutionError(f"Unsupported execution tier: {tier}")

    def _model_result_to_handler_output(self, model_res: Any, tier: RouteTier) -> HandlerOutput:
        return HandlerOutput(
            tier=tier,
            output_text=model_res.text,
            validated=True,
            input_tokens=ProvenanceValue(
                value=model_res.input_tokens,
                provenance=model_res.source,
                unit="tokens",
            ),
            output_tokens=ProvenanceValue(
                value=model_res.output_tokens,
                provenance=model_res.source,
                unit="tokens",
            ),
            latency_ms=ProvenanceValue(
                value=model_res.latency_ms,
                provenance=ProvenanceLabel.MEASURED,
                unit="ms",
            ),
            model_version=model_res.model_id,
            raw_response={
                "cost_usd_calculated": model_res.cost_usd_calculated,
                "provider_reported_cost_usd": model_res.provider_reported_cost_usd,
                "price_table_date": model_res.price_table_date,
            },
        )

    def _get_handler_version(self, decision: RoutingDecision) -> str:
        if decision.selected_tier == RouteTier.DETERMINISTIC_A:
            return "deterministic_v1"
        if decision.selected_tier == RouteTier.TEMPLATE_B:
            return f"jinja2_{decision.template_id or 'v1'}"
        if decision.selected_tier == RouteTier.SMALL_MODEL_C:
            return self.small_adapter.model_id
        return self.large_adapter.model_id

    def _finalize_result(
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
    ) -> PipelineResult:
        # Determine cost
        cost_usd = 0.0
        if not was_cached and output.raw_response and "cost_usd_calculated" in output.raw_response:
            cost_usd = float(output.raw_response["cost_usd_calculated"])

        # Determine environmental estimates via EnvironmentalEstimator
        total_tokens = output.input_tokens.value + output.output_tokens.value
        model_id = (
            output.raw_response.get("model_id") if output.raw_response else output.model_version
        )
        env_est = self.estimator.estimate(
            tier=output.tier,
            model_id=model_id,
            total_tokens=total_tokens,
            was_cached=was_cached,
        )

        final_tier = output.tier
        prompt_digest = hash_prompt(request.prompt)

        result = PipelineResult(
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
                value=round(cost_usd, 6),
                provenance=ProvenanceLabel.CALCULATED,
                unit="USD",
            ),
            energy_wh=env_est.energy_wh,
            water_ml=env_est.total_water_ml,
            privacy_class=request.privacy_class,
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            provider_reported_cost_usd=(
                ProvenanceValue(
                    value=float(output.raw_response["provider_reported_cost_usd"]),
                    provenance=ProvenanceLabel.MEASURED,
                    unit="USD",
                )
                if (
                    output.raw_response
                    and output.raw_response.get("provider_reported_cost_usd") is not None
                )
                else None
            ),
        )

        # Telemetry logging to JSONL
        self._log_experiment_record(request, result, output)

        return result

    def _log_experiment_record(
        self, request: UserRequest, result: PipelineResult, output: HandlerOutput
    ) -> None:
        cache_status = "miss"
        if result.was_cached:
            cache_status = "hit"
        elif request.privacy_class != PrivacyClass.PUBLIC:
            cache_status = "bypassed_private"
        elif result.initial_tier == RouteTier.DETERMINISTIC_A:
            cache_status = "bypassed_deterministic"

        meta: dict[str, Any] = {
            "validation_passed": result.validation_passed,
            "fallback_used": result.fallback_occurred,
            "rule_id": result.rule_id,
            "privacy_class": result.privacy_class.value,
        }
        if request.privacy_class == PrivacyClass.PUBLIC and not contains_sensitive_patterns(
            request.prompt
        ):
            meta["prompt_sample"] = request.prompt[:120]

        record = {
            "run_id": f"run_{request.request_id}",
            "request_id": request.request_id,
            "prompt_hash": result.prompt_hash,
            "route": result.final_tier.value,
            "rule_id": result.rule_id,
            "cache_status": cache_status,
            "latency_ms": result.total_latency_ms.value,
            "latency_provenance": result.total_latency_ms.provenance,
            "input_tokens": output.input_tokens.value,
            "input_tokens_provenance": output.input_tokens.provenance,
            "output_tokens": output.output_tokens.value,
            "output_tokens_provenance": output.output_tokens.provenance,
            "cost_usd": result.cost_usd.value,
            "cost_provenance": result.cost_usd.provenance,
            "fallback_events": [fb.model_dump() for fb in result.fallback_history],
            "metadata": meta,
        }
        self.logger.log(record)
