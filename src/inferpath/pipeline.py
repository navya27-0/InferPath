"""Single orchestration pipeline for InferPath executing the 11-step routing sequence."""

import time
from typing import Any

from inferpath.cache.canonical_key import CanonicalKeyBuilder
from inferpath.cache.eligibility import CacheEligibilityPolicy
from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings, load_pricing_table, load_settings
from inferpath.evaluation.environmental import EnvironmentalEstimator
from inferpath.handlers.factory import AdapterFactory
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
from inferpath.pipeline_stages import CostAccumulator, FallbackCoordinator, ResultFinalizer
from inferpath.router import BaseRouter, RuleBasedRouter
from inferpath.router.analyzer import contains_sensitive_patterns
from inferpath.router.strategies import ConstraintsUnsatisfiableError
from inferpath.security.budget import RunBudgetTracker
from inferpath.security.hashing import hash_prompt
from inferpath.security.privacy import PrivacyPolicyEnforcer, PrivacyViolationError
from inferpath.security.secret_manager import GLOBAL_SECRET_MANAGER
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
        adapter_mode: str | None = None,
        allow_mock_fallback: bool | None = None,
        adapter_factory: type[AdapterFactory] | AdapterFactory | None = None,
    ):
        self.settings = settings or load_settings()
        self.pricing_table = pricing_table or load_pricing_table()
        self.adapter_mode = adapter_mode
        self.allow_mock_fallback = allow_mock_fallback if allow_mock_fallback is not None else True
        self.adapter_factory = adapter_factory or AdapterFactory

        # Resolve credentials from SecretManager if not present in settings
        if not self.settings.openrouter.api_key:
            resolved_key = GLOBAL_SECRET_MANAGER.get("OPENROUTER_API_KEY", "")
            if resolved_key:
                self.settings.openrouter.api_key = resolved_key

        if (
            self.adapter_mode == "real"
            and not self.allow_mock_fallback
            and not self.settings.openrouter.api_key
        ):
            raise PipelineExecutionError(
                "Pipeline configured with adapter_mode='real' and allow_mock_fallback=False "
                "requires OPENROUTER_API_KEY, but no credentials were found."
            )
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

        # Modular pipeline stages: fallback coordination and result finalization
        try:
            large_model_id = str(self.large_adapter.model_id)
        except AttributeError:
            large_model_id = "anthropic/claude-3.7-sonnet"

        self.fallback_coordinator = FallbackCoordinator(
            routing_settings=self.settings.routing,
            privacy_enforcer=self.privacy_enforcer,
            large_model_id=large_model_id,
            budget_tracker=self.budget_tracker,
        )
        self.finalizer = ResultFinalizer(self.estimator)

    def _build_mock_small(self) -> MockSmallModel:
        """Constructs deterministic MockSmallModel via AdapterFactory."""
        return self.adapter_factory.build_mock_small(self.settings, self.pricing_table)

    def _build_openrouter_small(self) -> OpenRouterAdapter:
        """Constructs live OpenRouterAdapter via AdapterFactory."""
        return self.adapter_factory.build_openrouter_small(
            self.settings, self.pricing_table, self.budget_tracker
        )

    def _build_mock_large(self) -> MockLargeModel:
        """Constructs deterministic MockLargeModel via AdapterFactory."""
        return self.adapter_factory.build_mock_large(self.settings, self.pricing_table)

    def _build_openrouter_large(self) -> OpenRouterAdapter:
        """Constructs live OpenRouterAdapter via AdapterFactory."""
        return self.adapter_factory.build_openrouter_large(
            self.settings, self.pricing_table, self.budget_tracker
        )

    def _create_small_adapter(self) -> ModelAdapter:
        """Creates small model adapter explicitly based on mode, credentials, and fallback policy."""
        try:
            return self.adapter_factory.create_small_adapter(
                settings=self.settings,
                pricing_table=self.pricing_table,
                budget_tracker=self.budget_tracker,
                adapter_mode=self.adapter_mode,
                allow_mock_fallback=self.allow_mock_fallback,
            )
        except RuntimeError as e:
            raise PipelineExecutionError(str(e)) from e

    def _create_large_adapter(self) -> ModelAdapter:
        """Creates large model adapter explicitly based on mode, credentials, and fallback policy."""
        try:
            return self.adapter_factory.create_large_adapter(
                settings=self.settings,
                pricing_table=self.pricing_table,
                budget_tracker=self.budget_tracker,
                adapter_mode=self.adapter_mode,
                allow_mock_fallback=self.allow_mock_fallback,
            )
        except RuntimeError as e:
            raise PipelineExecutionError(str(e)) from e

    def _validate_request(self, request: UserRequest, route_trace: list[dict[str, Any]]) -> str:
        """Step 1 & 2: Validates schema and privacy transmission feasibility."""
        if not request.prompt or not request.prompt.strip():
            raise PipelineExecutionError("Request prompt cannot be empty.")
        prompt_digest = hash_prompt(request.prompt)
        route_trace.append(
            {"step": "validation_schema", "status": "passed", "prompt_hash": prompt_digest}
        )

        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, _ = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            route_trace.append(
                {
                    "step": "privacy_check",
                    "privacy_class": request.privacy_class.value,
                    "allow_external": can_transmit,
                }
            )
        return prompt_digest

    def _route_and_check_constraints(
        self, request: UserRequest, route_trace: list[dict[str, Any]]
    ) -> RoutingDecision:
        """Step 3 & 4: Selects optimal route and verifies caller/system constraint satisfiability."""
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

        return decision

    def _prepare_cache_key_and_eligibility(
        self, request: UserRequest, decision: RoutingDecision, route_trace: list[dict[str, Any]]
    ) -> tuple[str, bool]:
        """Step 5: Builds canonical cache key and evaluates eligibility policy."""
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
        return cache_key, is_cache_eligible

    def _can_attempt_fallback(
        self,
        request: UserRequest,
        decision: RoutingDecision,
        current_tier: RouteTier,
        val_result: Any,
        route_trace: list[dict[str, Any]],
    ) -> bool:
        """Step 9: Checks whether fallback escalation from Tier C to Tier D is authorized."""
        return self.fallback_coordinator.can_attempt_fallback(
            request=request,
            decision=decision,
            current_tier=current_tier,
            val_result=val_result,
            route_trace=route_trace,
        )

    def _store_cache_if_eligible(
        self,
        cache_key: str,
        request: UserRequest,
        current_tier: RouteTier,
        output: HandlerOutput,
        val_result: Any,
        route_trace: list[dict[str, Any]],
    ) -> None:
        """Step 10: Writes validated eligible outputs to the exact cache synchronously."""
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

    async def _store_cache_if_eligible_async(
        self,
        cache_key: str,
        request: UserRequest,
        current_tier: RouteTier,
        output: HandlerOutput,
        val_result: Any,
        route_trace: list[dict[str, Any]],
    ) -> None:
        """Step 10: Writes validated eligible outputs to the exact cache asynchronously."""
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

    def execute(self, request: UserRequest) -> PipelineResult:
        """Executes the request through the complete 11-step pipeline."""
        start_time = time.perf_counter()
        route_trace: list[dict[str, Any]] = []

        # Step 1 & 2: Schema validation and privacy policy
        self._validate_request(request, route_trace)

        # Step 3 & 4: Route selection and constraint satisfaction
        decision = self._route_and_check_constraints(request, route_trace)

        # Step 5: Canonical cache key and eligibility
        cache_key, is_cache_eligible = self._prepare_cache_key_and_eligibility(
            request, decision, route_trace
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

        # Step 7: Execute on cache miss (synchronous handler invocation)
        current_tier = decision.selected_tier
        cost_accumulator = CostAccumulator()
        output = self._execute_tier(request, decision, current_tier)
        cost_accumulator.record_call(current_tier, output)
        route_trace.append(
            {
                "step": "handler_execution",
                "tier": current_tier.value,
                "latency_ms": output.latency_ms.value,
            }
        )

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
                cost_accumulator=cost_accumulator,
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
        if self._can_attempt_fallback(request, decision, current_tier, val_result, route_trace):
            fb_rec = FallbackRecord(
                from_tier=RouteTier.SMALL_MODEL_C,
                to_tier=RouteTier.LARGE_MODEL_D,
                rejection_reason=val_result.rejection_reason or "Validation failed on small model.",
            )
            fallback_history.append(fb_rec)
            fallback_occurred = True
            current_tier = RouteTier.LARGE_MODEL_D

            output = self._execute_tier(request, decision, RouteTier.LARGE_MODEL_D)
            cost_accumulator.record_call(RouteTier.LARGE_MODEL_D, output)
            route_trace.append(
                {
                    "step": "fallback_execution",
                    "to_tier": RouteTier.LARGE_MODEL_D.value,
                    "latency_ms": output.latency_ms.value,
                }
            )

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
        self._store_cache_if_eligible(
            cache_key=cache_key,
            request=request,
            current_tier=current_tier,
            output=output,
            val_result=val_result,
            route_trace=route_trace,
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
            cost_accumulator=cost_accumulator,
        )

    async def execute_async(self, request: UserRequest) -> PipelineResult:
        """Executes the request through the complete 11-step pipeline asynchronously."""
        start_time = time.perf_counter()
        route_trace: list[dict[str, Any]] = []

        # Step 1 & 2: Schema validation and privacy policy (shared helper)
        self._validate_request(request, route_trace)

        # Step 3 & 4: Route selection and constraint satisfaction (shared helper)
        decision = self._route_and_check_constraints(request, route_trace)

        # Step 5: Canonical cache key and eligibility (shared helper)
        cache_key, is_cache_eligible = self._prepare_cache_key_and_eligibility(
            request, decision, route_trace
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

        # Step 7: Execute on cache miss (asynchronous handler invocation)
        current_tier = decision.selected_tier
        cost_accumulator = CostAccumulator()
        output = await self._execute_tier_async(request, decision, current_tier)
        cost_accumulator.record_call(current_tier, output)
        route_trace.append(
            {
                "step": "handler_execution",
                "tier": current_tier.value,
                "latency_ms": output.latency_ms.value,
            }
        )

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
                cost_accumulator=cost_accumulator,
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
        if self._can_attempt_fallback(request, decision, current_tier, val_result, route_trace):
            fb_rec = FallbackRecord(
                from_tier=RouteTier.SMALL_MODEL_C,
                to_tier=RouteTier.LARGE_MODEL_D,
                rejection_reason=val_result.rejection_reason or "Validation failed on small model.",
            )
            fallback_history.append(fb_rec)
            fallback_occurred = True
            current_tier = RouteTier.LARGE_MODEL_D

            output = await self._execute_tier_async(request, decision, RouteTier.LARGE_MODEL_D)
            cost_accumulator.record_call(RouteTier.LARGE_MODEL_D, output)
            route_trace.append(
                {
                    "step": "fallback_execution",
                    "to_tier": RouteTier.LARGE_MODEL_D.value,
                    "latency_ms": output.latency_ms.value,
                }
            )

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
        await self._store_cache_if_eligible_async(
            cache_key=cache_key,
            request=request,
            current_tier=current_tier,
            output=output,
            val_result=val_result,
            route_trace=route_trace,
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
            cost_accumulator=cost_accumulator,
        )

    def _check_tier_transmission(self, request: UserRequest, tier: RouteTier) -> None:
        """Enforces privacy checks before external network transmission on model tiers."""
        if tier in (RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D):
            can_tx, tx_reason = self.privacy_enforcer.can_transmit_externally(request, tier)
            if not can_tx:
                raise PrivacyViolationError(
                    tx_reason
                    or f"Blocked external transmission of {request.privacy_class.value} data.",
                    privacy_class=request.privacy_class.value,
                    target_tier=tier.value,
                )

    def _execute_tier(
        self, request: UserRequest, decision: RoutingDecision, tier: RouteTier
    ) -> HandlerOutput:
        """Dispatches execution synchronously to the corresponding handler or adapter."""
        self._check_tier_transmission(request, tier)

        if tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B):
            return self.router.execute_local_handler(request, decision)

        if tier == RouteTier.SMALL_MODEL_C:
            adapter = self.small_adapter
        elif tier == RouteTier.LARGE_MODEL_D:
            adapter = self.large_adapter
        else:
            raise PipelineExecutionError(f"Unsupported execution tier: {tier}")

        if not isinstance(adapter, OpenRouterAdapter) and self.budget_tracker:
            est_cost = self.budget_tracker.estimate_call_cost(
                adapter.model_id, request.prompt, request.task_parameters.max_tokens
            )
            res_tok = self.budget_tracker.reserve(est_cost)
            try:
                model_res = adapter.generate(
                    request.prompt, max_tokens=request.task_parameters.max_tokens
                )
            except Exception:
                self.budget_tracker.release(res_tok)
                raise
            self.budget_tracker.reconcile(res_tok, model_res.cost_usd_calculated)
        else:
            model_res = adapter.generate(
                request.prompt, max_tokens=request.task_parameters.max_tokens
            )
        return self._model_result_to_handler_output(model_res, tier)

    async def _execute_tier_async(
        self, request: UserRequest, decision: RoutingDecision, tier: RouteTier
    ) -> HandlerOutput:
        """Dispatches execution asynchronously to the corresponding handler or adapter."""
        self._check_tier_transmission(request, tier)

        if tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B):
            return self.router.execute_local_handler(request, decision)

        if tier == RouteTier.SMALL_MODEL_C:
            adapter = self.small_adapter
        elif tier == RouteTier.LARGE_MODEL_D:
            adapter = self.large_adapter
        else:
            raise PipelineExecutionError(f"Unsupported execution tier: {tier}")

        if not isinstance(adapter, OpenRouterAdapter) and self.budget_tracker:
            est_cost = self.budget_tracker.estimate_call_cost(
                adapter.model_id, request.prompt, request.task_parameters.max_tokens
            )
            res_tok = self.budget_tracker.reserve(est_cost)
            try:
                model_res = await adapter.generate_async(
                    request.prompt, max_tokens=request.task_parameters.max_tokens
                )
            except Exception:
                self.budget_tracker.release(res_tok)
                raise
            self.budget_tracker.reconcile(res_tok, model_res.cost_usd_calculated)
        else:
            model_res = await adapter.generate_async(
                request.prompt, max_tokens=request.task_parameters.max_tokens
            )
        return self._model_result_to_handler_output(model_res, tier)

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
            try:
                return str(self.small_adapter.model_id)
            except AttributeError:
                return "google/gemini-2.5-flash"
        try:
            return str(self.large_adapter.model_id)
        except AttributeError:
            return "anthropic/claude-3.7-sonnet"

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
        cost_accumulator: CostAccumulator | None = None,
    ) -> PipelineResult:
        result = self.finalizer.finalize(
            request=request,
            decision=decision,
            output=output,
            was_cached=was_cached,
            fallback_occurred=fallback_occurred,
            fallback_history=fallback_history,
            validation_passed=validation_passed,
            elapsed_ms=elapsed_ms,
            route_trace=route_trace,
            cost_accumulator=cost_accumulator,
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

        in_tok = (
            result.input_tokens.value
            if result.input_tokens
            else (output.input_tokens.value if output.input_tokens else 0)
        )
        in_prov = (
            result.input_tokens.provenance
            if result.input_tokens
            else (
                output.input_tokens.provenance
                if output.input_tokens
                else ProvenanceLabel.CALCULATED
            )
        )
        out_tok = (
            result.output_tokens.value
            if result.output_tokens
            else (output.output_tokens.value if output.output_tokens else 0)
        )
        out_prov = (
            result.output_tokens.provenance
            if result.output_tokens
            else (
                output.output_tokens.provenance
                if output.output_tokens
                else ProvenanceLabel.CALCULATED
            )
        )

        record = {
            "run_id": f"run_{request.request_id}",
            "request_id": request.request_id,
            "prompt_hash": result.prompt_hash,
            "route": result.final_tier.value,
            "rule_id": result.rule_id,
            "cache_status": cache_status,
            "latency_ms": result.total_latency_ms.value,
            "latency_provenance": result.total_latency_ms.provenance,
            "input_tokens": in_tok,
            "input_tokens_provenance": in_prov,
            "output_tokens": out_tok,
            "output_tokens_provenance": out_prov,
            "cost_usd": result.cost_usd.value,
            "cost_provenance": result.cost_usd.provenance,
            "fallback_events": [fb.model_dump() for fb in result.fallback_history],
            "metadata": meta,
        }
        self.logger.log(record)
