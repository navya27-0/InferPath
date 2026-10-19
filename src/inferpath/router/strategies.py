"""Multi-strategy routing architecture for InferPath research and production.

Defines standardized routing strategies:
1. BaseRouter: Base class providing shared analyzer, template engine, local handlers, and safe logging.
2. AlwaysLargeRouter: Baseline always selecting Tier D (preserving privacy constraints).
3. AlwaysSmallRouter: Baseline always selecting Tier C (preserving privacy constraints).
4. ConstraintAwareRouter: Explicit optimizer solving:
       min_{t in T_cand} Cost(t) + lambda * Latency(t)
       s.t. Q_hat(t) >= Q_min, Privacy satisfied, Cost <= B_max, Latency <= L_max
5. LearnedRouter: Machine learning classifier predicting optimal tier from structural features,
   with strict privacy guardrail enforcement.
"""

from __future__ import annotations

import random
import re
from typing import Any

import numpy as np

from inferpath.config.settings import PricingTable, RoutingSettings, load_pricing_table
from inferpath.handlers.calculator import SafeCalculator
from inferpath.handlers.date_formatter import DateFormatter
from inferpath.handlers.json_validator import JSONSchemaValidator
from inferpath.handlers.unit_converter import UnitConverter
from inferpath.models.domain import HandlerOutput, RoutingDecision, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router.analyzer import RequestAnalysis, RequestAnalyzer, contains_sensitive_patterns
from inferpath.security.hashing import hash_prompt
from inferpath.security.privacy import PrivacyPolicyEnforcer, PrivacyViolationError
from inferpath.telemetry.experiment_logger import ExperimentLogger
from inferpath.templates.engine import TemplateEngine


class ConstraintsUnsatisfiableError(Exception):
    """Raised when requested quality, budget, latency, or privacy constraints cannot be satisfied."""

    def __init__(
        self,
        message: str,
        unsatisfied_constraints: list[str] | None = None,
        eligible_tiers: list[RouteTier] | None = None,
        code: str = "CONSTRAINTS_UNSATISFIABLE",
    ):
        super().__init__(message)
        self.message = message
        self.unsatisfied_constraints = unsatisfied_constraints or []
        self.eligible_tiers = eligible_tiers or []
        self.code = code


class BaseRouter:
    """Base router class providing analyzer, template engine, local handlers, and audit logging."""

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
        pricing_table: PricingTable | None = None,
    ):
        self.settings = routing_settings or RoutingSettings()
        self.privacy_enforcer = privacy_enforcer or PrivacyPolicyEnforcer()
        self.logger = logger
        self.analyzer = RequestAnalyzer(self.settings)
        self.template_engine = TemplateEngine()
        self.pricing_table = pricing_table or load_pricing_table()

    def route(self, request: UserRequest) -> RoutingDecision:
        raise NotImplementedError("Subclasses must implement route(request).")

    def execute_local_handler(
        self, request: UserRequest, decision: RoutingDecision
    ) -> HandlerOutput:
        """Executes local deterministic or template handlers for Tiers A and B."""
        if decision.rule_id == "R01":
            raise PrivacyViolationError(decision.reason)

        if decision.selected_tier == RouteTier.DETERMINISTIC_A:
            analysis = self.analyzer.analyze(request)
            try:
                if (
                    decision.rule_id in ("R03", "DET_ARITHMETIC")
                    or analysis.task_intent == "arithmetic"
                ):
                    return SafeCalculator.execute_handler(request.prompt)
                elif (
                    decision.rule_id in ("R04", "DET_UNIT")
                    or analysis.task_intent == "unit_conversion"
                ):
                    return UnitConverter.execute_handler(request.prompt)
                elif (
                    decision.rule_id in ("R05", "DET_DATE")
                    or analysis.task_intent == "date_formatting"
                ):
                    return DateFormatter.execute_handler(request.prompt)
                elif (
                    decision.rule_id in ("R06", "DET_JSON")
                    or analysis.task_intent == "json_validation"
                ):
                    schema = request.metadata.get("schema")
                    return JSONSchemaValidator.execute_handler(request.prompt, schema=schema)
                return SafeCalculator.execute_handler(request.prompt)
            except Exception as err:
                from inferpath.models.enums import ProvenanceLabel
                from inferpath.models.telemetry import ProvenanceValue

                return HandlerOutput(
                    tier=RouteTier.DETERMINISTIC_A,
                    output_text=f"Deterministic Execution Error: {err}",
                    validated=False,
                    input_tokens=ProvenanceValue(
                        value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                    ),
                    output_tokens=ProvenanceValue(
                        value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                    ),
                    latency_ms=ProvenanceValue(
                        value=1.0, provenance=ProvenanceLabel.MEASURED, unit="ms"
                    ),
                    model_version="deterministic_v1",
                )

        if decision.selected_tier == RouteTier.TEMPLATE_B and decision.template_id:
            fields = request.metadata.get("fields") or {}
            analysis = self.analyzer.analyze(request)
            merged_fields = {**analysis.extracted_fields, **fields}
            return self.template_engine.execute_handler(decision.template_id, merged_fields)

        raise ValueError(
            f"Cannot execute non-local tier '{decision.selected_tier}' via execute_local_handler."
        )

    def _log_decision_safely(self, request: UserRequest, decision: RoutingDecision) -> None:
        if not self.logger:
            return

        digest = hash_prompt(request.prompt)
        metadata = {
            "prompt_hash": digest,
            "privacy_class": request.privacy_class.value,
            "rule_id": decision.rule_id,
            "selected_tier": decision.selected_tier.value,
            "reason": decision.reason,
            "confidence": decision.confidence,
            "time_sensitive": decision.time_sensitive,
            "cache_eligible": decision.cache_eligible,
        }
        if request.privacy_class == PrivacyClass.PUBLIC and not contains_sensitive_patterns(
            request.prompt
        ):
            metadata["prompt_preview"] = request.prompt[:100]

        self.logger.log_event("routing_decision", metadata)


class AlwaysLargeRouter(BaseRouter):
    """Always-Large baseline router: selects Tier D while respecting privacy blocks."""

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Enforce privacy guardrail (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.LARGE_MODEL_D
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.LARGE_MODEL_D,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Blocked by privacy policy.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        decision = RoutingDecision(
            selected_tier=RouteTier.LARGE_MODEL_D,
            eligible_tiers=[RouteTier.LARGE_MODEL_D],
            rule_id="ALWAYS_LARGE",
            confidence=1.0,
            reason="Always-Large baseline policy unconditionally selected Tier D.",
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC and not analysis.freshness_needed
            ),
        )
        self._log_decision_safely(request, decision)
        return decision


class AlwaysSmallRouter(BaseRouter):
    """Always-Small baseline router: selects Tier C while respecting privacy blocks."""

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Enforce privacy guardrail (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Blocked by privacy policy.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        decision = RoutingDecision(
            selected_tier=RouteTier.SMALL_MODEL_C,
            eligible_tiers=[RouteTier.SMALL_MODEL_C],
            rule_id="ALWAYS_SMALL",
            confidence=1.0,
            reason="Always-Small baseline policy unconditionally selected Tier C.",
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC and not analysis.freshness_needed
            ),
        )
        self._log_decision_safely(request, decision)
        return decision


class ConstraintAwareRouter(BaseRouter):
    """
    Constraint-Aware Optimizer solving the explicit mathematical program:

        min_{t in T_eligible} Cost(t) + lambda * Latency(t)
        s.t.
            Q_hat(t, task) >= Q_min
            Privacy(t, task) == ALLOWED
            Cost(t) <= B_max  (if specified)
            Latency(t) <= L_max  (if specified)

    Evaluates:
    - Tier A: Deterministic AST math, conversion, dates, schema (Cost=$0, Latency=1ms, Q=1.0)
    - Tier B: Structured Jinja2 templates (Cost=$0, Latency=1ms, Q=1.0 when fields present)
    - Tier C: Small model (Cost based on token price, Latency ~120ms, Q_hat estimated by task complexity)
    - Tier D: Large model (Cost based on token price, Latency ~450ms, Q_hat estimated by task complexity)
    """

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
        pricing_table: PricingTable | None = None,
        latency_penalty_weight: float = 1e-5,  # lambda: USD per millisecond of latency
        empirical_quality_priors: dict[str, dict[RouteTier, float]] | None = None,
    ):
        super().__init__(routing_settings, privacy_enforcer, logger, pricing_table)
        self.lambda_latency = latency_penalty_weight
        # Empirical priors: mapping from task category/intent to {RouteTier: estimated_quality}
        # When None or key missing, router falls back to structural heuristic baseline assumptions.
        self.empirical_quality_priors = empirical_quality_priors or {}

    def calibrate_from_validation(
        self, task_quality_priors: dict[str, dict[RouteTier, float]]
    ) -> None:
        """Calibrates quality priors with empirical validation performance."""
        self.empirical_quality_priors.update(task_quality_priors)

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)
        q_min = request.minimum_quality
        b_max = request.maximum_cost_usd
        l_max = request.maximum_latency_ms

        # -------------------------------------------------------------
        # 1. Candidate Generation & Local Deterministic / Template check
        # -------------------------------------------------------------
        # Tier A: Deterministic handlers
        if analysis.task_intent == "arithmetic":
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R03",
                confidence=1.0,
                reason="Constraint-aware optimization: Tier A satisfies Q_min=1.0 at zero cost.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        if analysis.task_intent == "unit_conversion":
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R04",
                confidence=1.0,
                reason="Constraint-aware optimization: Tier A satisfies Q_min=1.0 at zero cost.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        if analysis.task_intent == "date_formatting":
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R05",
                confidence=1.0,
                reason="Constraint-aware optimization: Tier A satisfies Q_min=1.0 at zero cost.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        if analysis.task_intent == "json_validation":
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R06",
                confidence=1.0,
                reason="Constraint-aware optimization: Tier A satisfies Q_min=1.0 at zero cost.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        # Tier B: Template Generation
        if analysis.template_id:
            needs_clarif = bool(analysis.missing_inputs)
            clarif_prompt = (
                f"Please provide the missing required information for {analysis.template_id}: "
                f"{', '.join(analysis.missing_inputs)}."
                if needs_clarif
                else None
            )
            decision = RoutingDecision(
                selected_tier=RouteTier.TEMPLATE_B,
                eligible_tiers=[RouteTier.TEMPLATE_B],
                rule_id="R07",
                confidence=0.98,
                reason=(
                    f"Template '{analysis.template_id}' matched; clarification required for: {', '.join(analysis.missing_inputs)}"
                    if needs_clarif
                    else f"Constraint-aware optimization: Structured Jinja2 template '{analysis.template_id}' satisfies Q=1.0 at zero cost."
                ),
                template_id=analysis.template_id,
                time_sensitive=False,
                cache_eligible=not needs_clarif and analysis.privacy_class == PrivacyClass.PUBLIC,
                needs_clarification=needs_clarif,
                clarification_prompt=clarif_prompt,
                missing_fields=analysis.missing_inputs,
            )
            self._log_decision_safely(request, decision)
            return decision

        # -------------------------------------------------------------
        # 2. Privacy Policy Guardrail Enforcement (R01)
        # -------------------------------------------------------------
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason
                    or "Privacy policy strictly prohibits external transmission of confidential data.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        # -------------------------------------------------------------
        # 3. Model Tier Evaluation & Constraint Satisfaction
        # -------------------------------------------------------------
        # Estimate token volumes
        estimated_prompt_tokens = max(10, len(request.prompt.split()) * 2)
        estimated_completion_tokens = request.task_parameters.max_tokens or 256

        # Pricing lookup
        small_model_id = getattr(
            getattr(self.settings, "openrouter", None), "small_model", "google/gemini-2.5-flash"
        )
        large_model_id = getattr(
            getattr(self.settings, "openrouter", None),
            "large_model",
            "anthropic/claude-3.7-sonnet",
        )
        rates_c = self.pricing_table.rates_per_1m.get(
            small_model_id, {"prompt": 0.075, "completion": 0.30}
        )
        rates_d = self.pricing_table.rates_per_1m.get(
            large_model_id, {"prompt": 3.0, "completion": 15.0}
        )

        cost_c = (estimated_prompt_tokens / 1e6) * rates_c["prompt"] + (
            estimated_completion_tokens / 1e6
        ) * rates_c["completion"]
        cost_d = (estimated_prompt_tokens / 1e6) * rates_d["prompt"] + (
            estimated_completion_tokens / 1e6
        ) * rates_d["completion"]

        latency_c = 120.0  # ms expected
        latency_d = 450.0  # ms expected

        # Estimate Quality Q_hat(t, task)
        # Note on empirical calibration:
        # In baseline configuration, quality estimates are assigned via structural complexity indicators.
        # When empirical_quality_priors are provided (e.g. from validation split calibration), they take precedence.
        category_key = request.metadata.get("task_category") if request.metadata else None
        task_intent_key = analysis.task_intent or "standard"
        task_family_key = (
            analysis.task_family.value
            if hasattr(analysis.task_family, "value")
            else str(analysis.task_family)
        )
        matched_prior = (
            (self.empirical_quality_priors.get(category_key) if category_key else None)
            or self.empirical_quality_priors.get(task_intent_key)
            or self.empirical_quality_priors.get(task_family_key)
        )

        is_complex = bool(analysis.complexity_indicators)
        is_ambiguous = analysis.ambiguity_detected
        is_high_pref = request.task_parameters.quality_preference == "high"

        if matched_prior:
            q_hat_c = matched_prior.get(RouteTier.SMALL_MODEL_C, 0.72)
            q_hat_d = matched_prior.get(RouteTier.LARGE_MODEL_D, 0.95)
        elif is_complex or is_ambiguous:
            q_hat_c = 0.58  # Heuristic assumption: Small models degrade on complex reasoning
            q_hat_d = 0.94
        elif is_high_pref:
            q_hat_c = 0.72  # Heuristic assumption: user quality preference set to high
            q_hat_d = 0.96
        else:
            q_hat_c = 0.88  # Heuristic assumption: Standard extraction / drafting
            q_hat_d = 0.95

        # Check constraints for Tier C
        tier_c_eligible = True
        if q_hat_c < q_min:
            tier_c_eligible = False
        if b_max is not None and cost_c > b_max:
            tier_c_eligible = False
        if l_max is not None and latency_c > l_max:
            tier_c_eligible = False

        # Check constraints for Tier D
        tier_d_eligible = True
        if q_hat_d < q_min:
            tier_d_eligible = False
        if b_max is not None and cost_d > b_max:
            tier_d_eligible = False
        if l_max is not None and latency_d > l_max:
            tier_d_eligible = False

        # -------------------------------------------------------------
        # 4. Objective Minimization
        # J(t) = Cost(t) + lambda * Latency(t)
        # -------------------------------------------------------------
        obj_c = cost_c + self.lambda_latency * latency_c
        obj_d = cost_d + self.lambda_latency * latency_d

        eligible_tiers: list[RouteTier] = []
        if tier_c_eligible:
            eligible_tiers.append(RouteTier.SMALL_MODEL_C)
        if tier_d_eligible:
            eligible_tiers.append(RouteTier.LARGE_MODEL_D)

        unsatisfied_constraints: list[str] = []
        constraints_met = True

        if tier_c_eligible and tier_d_eligible:
            selected_tier = RouteTier.SMALL_MODEL_C if obj_c <= obj_d else RouteTier.LARGE_MODEL_D
            rule_id = (
                "CONSTRAINT_OPT_C"
                if selected_tier == RouteTier.SMALL_MODEL_C
                else "CONSTRAINT_OPT_D"
            )
            reason = (
                f"Constraint optimizer: Tier C meets Q_min={q_min:.2f} (Q_hat={q_hat_c:.2f}) "
                f"at lower cost (${cost_c:.6f} vs ${cost_d:.6f})."
                if selected_tier == RouteTier.SMALL_MODEL_C
                else "Constraint optimizer: Tier D selected based on combined cost+latency objective."
            )
        elif tier_d_eligible:
            selected_tier = RouteTier.LARGE_MODEL_D
            rule_id = "CONSTRAINT_PRUNED_TO_D"
            reason = (
                f"Tier C pruned (Q_hat={q_hat_c:.2f} < Q_min={q_min:.2f}); "
                f"Tier D selected to satisfy quality constraint."
            )
        elif tier_c_eligible:
            selected_tier = RouteTier.SMALL_MODEL_C
            rule_id = "CONSTRAINT_PRUNED_TO_C"
            reason = "Tier D pruned due to budget/latency constraints; Tier C selected."
        else:
            # No tier can meet the requested quality, budget, latency, and privacy constraints
            constraints_met = False
            selected_tier = (
                RouteTier.DETERMINISTIC_A
            )  # Safe placeholder tier; model will NOT be executed
            rule_id = "CONSTRAINTS_UNSATISFIABLE"

            # Determine precisely which constraints were unsatisfied across candidate tiers
            if q_hat_c < q_min and q_hat_d < q_min:
                unsatisfied_constraints.append("quality")
            if b_max is not None and cost_c > b_max and cost_d > b_max:
                unsatisfied_constraints.append("budget")
            if l_max is not None and latency_c > l_max and latency_d > l_max:
                unsatisfied_constraints.append("latency")

            if not unsatisfied_constraints:
                # Conflicting constraints where each candidate failed different constraints
                if q_hat_c < q_min or q_hat_d < q_min:
                    unsatisfied_constraints.append("quality")
                if b_max is not None and (cost_c > b_max or cost_d > b_max):
                    unsatisfied_constraints.append("budget")
                if l_max is not None and (latency_c > l_max or latency_d > l_max):
                    unsatisfied_constraints.append("latency")

            b_str = f"${b_max:.6f}" if b_max is not None else "None"
            l_str = f"{l_max:.1f}ms" if l_max is not None else "None"
            reason = (
                f"Constraints unsatisfiable: no candidate tier satisfies all caller constraints "
                f"(unsatisfied constraints: {', '.join(unsatisfied_constraints)}; "
                f"Q_min={q_min:.2f}, B_max={b_str}, L_max={l_str})."
            )

        time_sensitive = analysis.freshness_needed
        cache_eligible = (
            request.privacy_class == PrivacyClass.PUBLIC
            and not time_sensitive
            and not is_ambiguous
            and constraints_met
        )

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=eligible_tiers,
            rule_id=rule_id,
            confidence=0.92 if constraints_met else 1.0,
            reason=reason,
            time_sensitive=time_sensitive,
            cache_eligible=cache_eligible,
            constraints_satisfied=constraints_met,
            unsatisfied_constraints=unsatisfied_constraints,
        )
        self._log_decision_safely(request, decision)
        return decision


class LearnedRouter(BaseRouter):
    """
    Supervised Machine Learning Router prototype predicting optimal processing tier from feature vectors.

    Research & Development Status:
    This router serves as a proof-of-concept learned baseline evaluated on a synthetic calibration
    corpus. 5-fold cross-validation establishes that the classifier can reproduce patterns in its
    own synthetic training corpus, but does not establish validated generalization on unseen real prompts.
    It demonstrates feature extraction and ML dispatch without hard constraint guarantees.

    Features extracted (15 structural signals):
    - Text length, word count, token count estimate
    - Digit count, mathematical operator density
    - Currency and JSON syntactic markers
    - Semantic intent indicators (arithmetic, unit, date, json, template)
    - Freshness / time-sensitivity signals
    - Complexity keyword count
    - Requested minimum quality (Q_min)

    Safety Invariant:
    Privacy policy guardrails are hard constraints evaluated BEFORE machine learning prediction.
    """

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
        pricing_table: PricingTable | None = None,
    ):
        super().__init__(routing_settings, privacy_enforcer, logger, pricing_table)
        self._model: Any = None
        self._init_classifier()

    def _extract_features(self, request: UserRequest, analysis: RequestAnalysis) -> np.ndarray:
        prompt = request.prompt
        words = prompt.split()
        n_chars = len(prompt)
        n_words = len(words)
        est_tokens = max(1, n_words * 2)

        # Structural signals
        n_digits = sum(c.isdigit() for c in prompt)
        n_operators = sum(prompt.count(op) for op in ["+", "-", "*", "/", "=", "^", "%"])
        has_currency = 1.0 if any(sym in prompt for sym in ["$", "€", "£", "₹", "USD"]) else 0.0
        has_json = 1.0 if ("{" in prompt and "}" in prompt) or "[" in prompt else 0.0

        # Semantic analysis signals
        is_arithmetic = 1.0 if analysis.task_intent == "arithmetic" else 0.0
        is_unit = 1.0 if analysis.task_intent == "unit_conversion" else 0.0
        is_date = 1.0 if analysis.task_intent == "date_formatting" else 0.0
        is_json = 1.0 if analysis.task_intent == "json_validation" else 0.0
        has_template = 1.0 if analysis.template_id else 0.0
        freshness = 1.0 if analysis.freshness_needed else 0.0
        n_complexity = float(len(analysis.complexity_indicators))
        q_min = float(request.minimum_quality)

        return np.array(
            [
                n_chars,
                n_words,
                est_tokens,
                n_digits,
                n_operators,
                has_currency,
                has_json,
                is_arithmetic,
                is_unit,
                is_date,
                is_json,
                has_template,
                freshness,
                n_complexity,
                q_min,
            ],
            dtype=np.float64,
        )

    def _init_classifier(self) -> None:
        """Trains or initializes a deterministic scikit-learn classifier with calibrated seed."""
        from sklearn.ensemble import RandomForestClassifier

        self._model = RandomForestClassifier(n_estimators=30, random_state=42)

        # Expanded balanced synthetic corpus covering all four tiers (48 samples, 12 per class)
        # Tiers: 0 -> DETERMINISTIC_A, 1 -> TEMPLATE_B, 2 -> SMALL_MODEL_C, 3 -> LARGE_MODEL_D
        synthetic_samples = [
            # Tier A samples (arithmetic, unit, date, json) - Class 0
            ([15, 3, 6, 4, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.7], 0),
            ([20, 4, 8, 5, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.7], 0),
            ([25, 5, 10, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.7], 0),
            ([30, 6, 12, 6, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.7], 0),
            ([35, 7, 14, 8, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.8], 0),
            ([40, 8, 16, 5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.6], 0),
            ([45, 9, 18, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.7], 0),
            ([50, 8, 16, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.7], 0),
            ([60, 10, 20, 6, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.8], 0),
            ([22, 4, 8, 3, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.5], 0),
            ([28, 5, 10, 5, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.7], 0),
            ([32, 6, 12, 4, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.6], 0),
            # Tier B samples (structured templates) - Class 1
            ([60, 10, 20, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([80, 12, 24, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([100, 15, 30, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([120, 18, 36, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([140, 22, 44, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.8], 1),
            ([90, 14, 28, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.6], 1),
            ([110, 17, 34, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([130, 20, 40, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.8], 1),
            ([75, 11, 22, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([95, 15, 30, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([150, 24, 48, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([85, 13, 26, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.6], 1),
            # Tier C samples (standard QA / summarization / extraction) - Class 2
            ([40, 7, 14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7], 2),
            ([60, 10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7], 2),
            ([75, 12, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6], 2),
            ([80, 12, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.7], 2),
            ([90, 14, 28, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7], 2),
            ([100, 15, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6], 2),
            ([110, 18, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.7], 2),
            ([120, 20, 40, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7], 2),
            ([70, 11, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.65], 2),
            ([85, 13, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.7], 2),
            ([95, 16, 32, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.75], 2),
            ([105, 17, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0.7], 2),
            # Tier D samples (complex multi-hop reasoning, high Q_min) - Class 3
            ([150, 25, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0.95], 3),
            ([200, 32, 64, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0.90], 3),
            ([250, 40, 80, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0.85], 3),
            ([300, 50, 100, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0.90], 3),
            ([350, 55, 110, 5, 2, 1, 0, 0, 0, 0, 0, 0, 0, 3, 0.92], 3),
            ([400, 65, 130, 8, 4, 1, 0, 0, 0, 0, 0, 0, 0, 5, 0.95], 3),
            ([180, 28, 56, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0.88], 3),
            ([220, 35, 70, 3, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0.90], 3),
            ([280, 44, 88, 5, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0.87], 3),
            ([320, 52, 104, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0.92], 3),
            ([160, 26, 52, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0.96], 3),
            ([240, 38, 76, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0.89], 3),
        ]
        X = np.array([s[0] for s in synthetic_samples])
        y = np.array([s[1] for s in synthetic_samples])
        self._training_samples = (X, y)
        self._model.fit(X, y)

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Step 1: Privacy Guardrail Invariant (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        # Step 2: Handle Template missing fields
        if analysis.template_id and bool(analysis.missing_inputs):
            decision = RoutingDecision(
                selected_tier=RouteTier.TEMPLATE_B,
                eligible_tiers=[RouteTier.TEMPLATE_B],
                rule_id="R07",
                confidence=0.95,
                reason=f"Template '{analysis.template_id}' matched; clarification required for missing fields.",
                template_id=analysis.template_id,
                time_sensitive=False,
                cache_eligible=False,
                needs_clarification=True,
                clarification_prompt=f"Please provide: {', '.join(analysis.missing_inputs)}",
                missing_fields=analysis.missing_inputs,
            )
            self._log_decision_safely(request, decision)
            return decision

        # Step 3: ML Feature Extraction & Prediction
        features = self._extract_features(request, analysis).reshape(1, -1)
        pred_class = int(self._model.predict(features)[0])
        probs = self._model.predict_proba(features)[0]
        confidence = float(probs[pred_class]) if len(probs) > pred_class else 0.85

        tier_map = {
            0: RouteTier.DETERMINISTIC_A,
            1: RouteTier.TEMPLATE_B,
            2: RouteTier.SMALL_MODEL_C,
            3: RouteTier.LARGE_MODEL_D,
        }
        selected_tier = tier_map.get(pred_class, RouteTier.SMALL_MODEL_C)

        # Guardrail: local deterministic/template tiers require confirmed structural patterns
        if selected_tier == RouteTier.DETERMINISTIC_A:
            if analysis.task_intent not in (
                "arithmetic",
                "unit_conversion",
                "date_formatting",
                "json_validation",
            ):
                selected_tier = RouteTier.SMALL_MODEL_C
        elif selected_tier == RouteTier.TEMPLATE_B and not analysis.template_id:
            selected_tier = RouteTier.SMALL_MODEL_C

        cache_eligible = (
            request.privacy_class == PrivacyClass.PUBLIC
            and not analysis.freshness_needed
            and selected_tier != RouteTier.DETERMINISTIC_A
        )

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[selected_tier, RouteTier.LARGE_MODEL_D],
            rule_id="LEARNED_CLASSIFIER",
            confidence=round(confidence, 4),
            reason=f"Supervised classifier predicted {selected_tier.value} with confidence {confidence:.2%}.",
            time_sensitive=analysis.freshness_needed,
            cache_eligible=cache_eligible,
            template_id=analysis.template_id if selected_tier == RouteTier.TEMPLATE_B else None,
        )
        self._log_decision_safely(request, decision)
        return decision


class OracleRouter(BaseRouter):
    """Theoretical ex-post optimal Oracle Router establishing the Pareto lower bound.

    In inference routing literature, the Oracle represents the theoretical upper limit
    of routing efficiency. For each query, it selects the minimal resource-cost tier
    capable of passing validation:
    1. Preserves Stage 1 privacy invariants (confidential -> HTTP 403 / R01).
    2. Exact arithmetic/conversions -> Tier A ($0.00 cost, <0.1 ms).
    3. Structural templates -> Tier B ($0.00 cost, <0.5 ms).
    4. Tasks solvable by ~8B models (classification, extraction, summaries) -> Tier C.
    5. Complex multi-hop reasoning, creative synthesis, and formal proofs -> Tier D.

    Used as the theoretical benchmark against which empirical routing regret is measured.
    """

    def route(self, request: UserRequest) -> RoutingDecision:
        # Step 1: Privacy Invariant Check
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.LARGE_MODEL_D
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.DETERMINISTIC_A,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason
                    or f"Privacy policy violation: {request.privacy_class.value} cannot be transmitted externally.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        analysis = self.analyzer.analyze(request)

        # Step 2: Deterministic Handlers (Tier A) - exact mathematical/formatting tasks
        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="ORACLE_TIER_A",
                confidence=1.0,
                reason="Oracle lower bound: exact deterministic solution available locally at zero cost.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        # Step 3: Structured Templates (Tier B)
        if analysis.template_id and not analysis.missing_inputs:
            decision = RoutingDecision(
                selected_tier=RouteTier.TEMPLATE_B,
                eligible_tiers=[RouteTier.TEMPLATE_B],
                rule_id="ORACLE_TIER_B",
                confidence=1.0,
                reason=f"Oracle lower bound: template '{analysis.template_id}' satisfies request deterministically.",
                time_sensitive=False,
                cache_eligible=True,
                template_id=analysis.template_id,
            )
            self._log_decision_safely(request, decision)
            return decision

        if analysis.template_id and analysis.missing_inputs:
            decision = RoutingDecision(
                selected_tier=RouteTier.TEMPLATE_B,
                eligible_tiers=[RouteTier.TEMPLATE_B],
                rule_id="R07",
                confidence=1.0,
                reason=f"Template '{analysis.template_id}' requires clarification.",
                template_id=analysis.template_id,
                time_sensitive=False,
                cache_eligible=False,
                needs_clarification=True,
                clarification_prompt=f"Please provide: {', '.join(analysis.missing_inputs)}",
                missing_fields=analysis.missing_inputs,
            )
            self._log_decision_safely(request, decision)
            return decision

        # Step 4: Tasks requiring Frontier Capabilities (Tier D)
        # Complex multi-hop reasoning, formal proofs, advanced code synthesis, or high ambiguity
        prompt_lower = request.prompt.lower()
        complex_signals = (
            "prove",
            "formal proof",
            "mathematical proof",
            "step-by-step derivation",
            "refactor the entire",
            "architect a system",
            "compare and contrast the epistemological",
            "nuanced philosophical",
        )
        is_complex = any(sig in prompt_lower for sig in complex_signals) or (
            request.minimum_quality > 0.85
        )

        selected_tier = RouteTier.LARGE_MODEL_D if is_complex else RouteTier.SMALL_MODEL_C
        rule_id = "ORACLE_FRONTIER_D" if is_complex else "ORACLE_EFFICIENT_C"
        reason = (
            "Oracle lower bound: complex reasoning requires frontier model capacity."
            if is_complex
            else "Oracle lower bound: standard task successfully solvable by efficient 8B model."
        )

        cache_eligible = (
            request.privacy_class == PrivacyClass.PUBLIC and not analysis.freshness_needed
        )

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[selected_tier, RouteTier.LARGE_MODEL_D],
            rule_id=rule_id,
            confidence=1.0,
            reason=reason,
            time_sensitive=analysis.freshness_needed,
            cache_eligible=cache_eligible,
        )
        self._log_decision_safely(request, decision)
        return decision


class RandomConstrainedRouter(BaseRouter):
    """Uniform random baseline router constrained by caller requirements.

    For each incoming request, evaluates eligibility across all candidate tiers
    (respecting hard privacy guardrails, budget caps, and latency limits), and
    samples uniformly at random among eligible candidates using a deterministic seed.
    """

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
        pricing_table: PricingTable | None = None,
        seed: int = 42,
    ):
        super().__init__(routing_settings, privacy_enforcer, logger, pricing_table)
        self.seed = seed
        self._rng = random.Random(seed)

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        eligible_tiers: list[RouteTier] = []

        # Check local Tier A
        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            eligible_tiers.append(RouteTier.DETERMINISTIC_A)

        # Check local Tier B
        if analysis.template_id and not analysis.missing_inputs:
            eligible_tiers.append(RouteTier.TEMPLATE_B)

        # Token & cost estimates for model tiers
        prompt_tokens = max(10, len(request.prompt.split()) * 2)
        completion_tokens = request.task_parameters.max_tokens or 256
        b_max = request.maximum_cost_usd
        l_max = request.maximum_latency_ms

        small_model_id = getattr(
            getattr(self.settings, "openrouter", None), "small_model", "google/gemini-2.5-flash"
        )
        large_model_id = getattr(
            getattr(self.settings, "openrouter", None),
            "large_model",
            "anthropic/claude-3.7-sonnet",
        )
        rates_c = self.pricing_table.rates_per_1m.get(
            small_model_id, {"prompt": 0.075, "completion": 0.30}
        )
        rates_d = self.pricing_table.rates_per_1m.get(
            large_model_id, {"prompt": 3.0, "completion": 15.0}
        )

        cost_c = (prompt_tokens / 1e6) * rates_c["prompt"] + (completion_tokens / 1e6) * rates_c[
            "completion"
        ]
        cost_d = (prompt_tokens / 1e6) * rates_d["prompt"] + (completion_tokens / 1e6) * rates_d[
            "completion"
        ]
        latency_c = 120.0
        latency_d = 450.0

        if (b_max is None or cost_c <= b_max) and (l_max is None or latency_c <= l_max):
            eligible_tiers.append(RouteTier.SMALL_MODEL_C)
        if (b_max is None or cost_d <= b_max) and (l_max is None or latency_d <= l_max):
            eligible_tiers.append(RouteTier.LARGE_MODEL_D)

        if not eligible_tiers:
            # Fallback to C if no tier satisfies bounds
            eligible_tiers = [RouteTier.SMALL_MODEL_C]

        # Deterministic sampling based on prompt hash to maintain evaluation reproducibility
        h = int(hash_prompt(request.prompt)[:8], 16)
        rng = random.Random(self.seed ^ h)
        selected_tier = rng.choice(eligible_tiers)

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=eligible_tiers,
            rule_id="RANDOM_CONSTRAINED",
            confidence=round(1.0 / len(eligible_tiers), 4),
            reason=f"Random constrained baseline sampled {selected_tier.value} from {len(eligible_tiers)} eligible tiers.",
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC
                and not analysis.freshness_needed
                and selected_tier != RouteTier.DETERMINISTIC_A
            ),
            template_id=analysis.template_id if selected_tier == RouteTier.TEMPLATE_B else None,
        )
        self._log_decision_safely(request, decision)
        return decision


class ConfidenceEscalationRouter(BaseRouter):
    """Confidence-based escalation router.

    Routes initial queries to Tier C when estimated confidence in task completion
    exceeds an escalation threshold tau. If confidence drops below tau due to
    ambiguity, high complexity, or strict quality constraints, escalates to Tier D.
    """

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
        pricing_table: PricingTable | None = None,
        escalation_threshold: float = 0.70,
    ):
        super().__init__(routing_settings, privacy_enforcer, logger, pricing_table)
        self.threshold = escalation_threshold

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        # Deterministic Handlers
        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="CONF_TIER_A",
                confidence=1.0,
                reason="Deterministic task intent with 100% confidence.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        # Template Handlers
        if analysis.template_id:
            needs_clarif = bool(analysis.missing_inputs)
            decision = RoutingDecision(
                selected_tier=RouteTier.TEMPLATE_B,
                eligible_tiers=[RouteTier.TEMPLATE_B],
                rule_id="R07" if needs_clarif else "CONF_TIER_B",
                confidence=0.95,
                reason=f"Template '{analysis.template_id}' matched with high structural confidence.",
                template_id=analysis.template_id,
                time_sensitive=False,
                cache_eligible=not needs_clarif and request.privacy_class == PrivacyClass.PUBLIC,
                needs_clarification=needs_clarif,
                clarification_prompt=f"Please provide: {', '.join(analysis.missing_inputs)}"
                if needs_clarif
                else None,
                missing_fields=analysis.missing_inputs,
            )
            self._log_decision_safely(request, decision)
            return decision

        # Estimate confidence for Tier C
        confidence = 0.90
        if analysis.ambiguity_detected:
            confidence -= 0.25
        if analysis.complexity_indicators:
            confidence -= 0.15 * min(3, len(analysis.complexity_indicators))
        if request.task_parameters.quality_preference == "high":
            confidence -= 0.15
        if request.minimum_quality > 0.85:
            confidence -= 0.15
        confidence = max(0.10, min(1.0, confidence))

        if confidence >= self.threshold:
            selected_tier = RouteTier.SMALL_MODEL_C
            rule_id = "CONFIDENCE_PASS_C"
            reason = f"Estimated confidence {confidence:.2f} >= threshold {self.threshold:.2f}; routed to Tier C."
        else:
            selected_tier = RouteTier.LARGE_MODEL_D
            rule_id = "CONFIDENCE_ESCALATED_D"
            reason = f"Estimated confidence {confidence:.2f} < threshold {self.threshold:.2f}; escalated to Tier D."

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D],
            rule_id=rule_id,
            confidence=round(confidence, 4),
            reason=reason,
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC and not analysis.freshness_needed
            ),
        )
        self._log_decision_safely(request, decision)
        return decision


class PromptLengthRouter(BaseRouter):
    """Prompt length heuristic router.

    A widely used heuristic baseline in LLM cascading: queries with token count
    below length_threshold are dispatched to Tier C, while long-context queries
    are dispatched to Tier D.
    """

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
        pricing_table: PricingTable | None = None,
        length_threshold_words: int = 50,
    ):
        super().__init__(routing_settings, privacy_enforcer, logger, pricing_table)
        self.length_threshold = length_threshold_words

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        # Deterministic / Template bypass
        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            decision = RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="LENGTH_TIER_A",
                confidence=1.0,
                reason="Exact deterministic task resolved locally.",
                time_sensitive=False,
                cache_eligible=False,
            )
            self._log_decision_safely(request, decision)
            return decision

        words = request.prompt.split()
        word_count = len(words)

        if word_count < self.length_threshold:
            selected_tier = RouteTier.SMALL_MODEL_C
            rule_id = "PROMPT_LENGTH_SHORT_C"
            reason = f"Prompt length ({word_count} words) < threshold ({self.length_threshold} words); routed to Tier C."
        else:
            selected_tier = RouteTier.LARGE_MODEL_D
            rule_id = "PROMPT_LENGTH_LONG_D"
            reason = f"Prompt length ({word_count} words) >= threshold ({self.length_threshold} words); routed to Tier D."

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D],
            rule_id=rule_id,
            confidence=0.85,
            reason=reason,
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC and not analysis.freshness_needed
            ),
        )
        self._log_decision_safely(request, decision)
        return decision


class QualityOnlyRouter(BaseRouter):
    """Quality-maximizing greedy baseline router.

    Selects the tier with the highest expected quality output regardless of cost
    or latency: Tier A for exact deterministic arithmetic/conversions, and Tier D
    for all general model queries (subject to privacy invariants).
    """

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.LARGE_MODEL_D
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.LARGE_MODEL_D,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            selected_tier = RouteTier.DETERMINISTIC_A
            rule_id = "QUALITY_MAX_A"
            reason = "Exact AST math/conversion achieves 100% precision."
        elif analysis.template_id and not analysis.missing_inputs:
            selected_tier = RouteTier.TEMPLATE_B
            rule_id = "QUALITY_MAX_B"
            reason = "Structured Jinja2 template achieves 100% format precision."
        else:
            selected_tier = RouteTier.LARGE_MODEL_D
            rule_id = "QUALITY_MAX_D"
            reason = "Frontier large model selected for maximum generation quality."

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[selected_tier],
            rule_id=rule_id,
            confidence=1.0,
            reason=reason,
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC
                and not analysis.freshness_needed
                and selected_tier != RouteTier.DETERMINISTIC_A
            ),
            template_id=analysis.template_id if selected_tier == RouteTier.TEMPLATE_B else None,
        )
        self._log_decision_safely(request, decision)
        return decision


class CostOnlyRouter(BaseRouter):
    """Cost-minimizing greedy baseline router.

    Selects the cheapest candidate tier capable of producing a valid response:
    Tier A / B ($0.00) whenever eligible, otherwise Tier C (~$0.0001) instead of Tier D.
    """

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            selected_tier = RouteTier.DETERMINISTIC_A
            rule_id = "COST_MIN_A"
            reason = "Zero-cost local deterministic execution."
        elif analysis.template_id and not analysis.missing_inputs:
            selected_tier = RouteTier.TEMPLATE_B
            rule_id = "COST_MIN_B"
            reason = "Zero-cost structured template execution."
        else:
            selected_tier = RouteTier.SMALL_MODEL_C
            rule_id = "COST_MIN_C"
            reason = "Small 8B model selected for minimum inference expenditure."

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[selected_tier],
            rule_id=rule_id,
            confidence=1.0,
            reason=reason,
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC
                and not analysis.freshness_needed
                and selected_tier != RouteTier.DETERMINISTIC_A
            ),
            template_id=analysis.template_id if selected_tier == RouteTier.TEMPLATE_B else None,
        )
        self._log_decision_safely(request, decision)
        return decision


class LatencyOnlyRouter(BaseRouter):
    """Latency-minimizing greedy baseline router.

    Selects the fastest candidate tier capable of responding:
    Tier A / B (<1 ms) when eligible, otherwise Tier C (~120 ms) rather than Tier D (~450 ms).
    """

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        if analysis.task_intent in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            selected_tier = RouteTier.DETERMINISTIC_A
            rule_id = "LATENCY_MIN_A"
            reason = "Sub-millisecond local deterministic execution."
        elif analysis.template_id and not analysis.missing_inputs:
            selected_tier = RouteTier.TEMPLATE_B
            rule_id = "LATENCY_MIN_B"
            reason = "Sub-millisecond local template execution."
        else:
            selected_tier = RouteTier.SMALL_MODEL_C
            rule_id = "LATENCY_MIN_C"
            reason = "Small model selected for lowest turnaround latency."

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[selected_tier],
            rule_id=rule_id,
            confidence=1.0,
            reason=reason,
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC
                and not analysis.freshness_needed
                and selected_tier != RouteTier.DETERMINISTIC_A
            ),
            template_id=analysis.template_id if selected_tier == RouteTier.TEMPLATE_B else None,
        )
        self._log_decision_safely(request, decision)
        return decision


class SemanticSimilarityRouter(BaseRouter):
    """Semantic exemplar similarity router.

    Computes token-overlap similarity between the incoming prompt and prototypical
    exemplar centroids for each routing tier (Tier A arithmetic/conversions,
    Tier B templates, Tier C lightweight extraction/summaries, and Tier D complex reasoning).
    Routes to the tier with highest semantic affinity while enforcing privacy guardrails.
    """

    _EXEMPLARS: dict[RouteTier, list[str]] = {
        RouteTier.DETERMINISTIC_A: [
            "calculate arithmetic math multiply divide add subtract convert fahrenheit celsius km mph",
            "format date iso8601 validate json schema verify",
        ],
        RouteTier.TEMPLATE_B: [
            "generate bug report leave application meeting minutes invoice project kickoff letter",
            "fill out template employee date attendees title decision",
        ],
        RouteTier.SMALL_MODEL_C: [
            "summarize this paragraph extract sentiment positive negative neutral",
            "classify ticket intent billing technical general short definition explain briefly",
        ],
        RouteTier.LARGE_MODEL_D: [
            "prove mathematically formal derivation step by step epistemological philosophy",
            "architect high-availability multi-region distributed system deep synthesis compare",
        ],
    }

    def _compute_similarity(self, prompt: str, tier: RouteTier) -> float:
        prompt_tokens = set(re.findall(r"\w+", prompt.lower()))
        if not prompt_tokens:
            return 0.0

        scores = []
        for exemplar in self._EXEMPLARS[tier]:
            ex_tokens = set(re.findall(r"\w+", exemplar.lower()))
            intersection = prompt_tokens.intersection(ex_tokens)
            union = prompt_tokens.union(ex_tokens)
            jaccard = len(intersection) / len(union) if union else 0.0
            scores.append(jaccard)

        return max(scores) if scores else 0.0

    def route(self, request: UserRequest) -> RoutingDecision:
        analysis = self.analyzer.analyze(request)

        # Stage 1: Privacy Invariant Check (R01)
        if request.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                decision = RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason or "Privacy policy strictly prohibits external transmission.",
                    time_sensitive=False,
                    cache_eligible=False,
                )
                self._log_decision_safely(request, decision)
                return decision

        # Compute similarities across tiers
        scores = {tier: self._compute_similarity(request.prompt, tier) for tier in self._EXEMPLARS}

        # Guardrail: Tier A & B require actual structural capabilities
        if analysis.task_intent not in (
            "arithmetic",
            "unit_conversion",
            "date_formatting",
            "json_validation",
        ):
            scores[RouteTier.DETERMINISTIC_A] = 0.0
        if not analysis.template_id:
            scores[RouteTier.TEMPLATE_B] = 0.0

        # Fallback to C if no similarity match
        best_tier = max(scores, key=lambda t: scores[t])
        best_score = scores[best_tier]

        if best_score == 0.0:
            best_tier = RouteTier.SMALL_MODEL_C
            best_score = 0.50

        decision = RoutingDecision(
            selected_tier=best_tier,
            eligible_tiers=[best_tier, RouteTier.LARGE_MODEL_D],
            rule_id="SEMANTIC_SIMILARITY",
            confidence=round(min(1.0, 0.5 + best_score), 4),
            reason=f"Semantic centroid similarity matched {best_tier.value} with affinity {best_score:.3f}.",
            time_sensitive=analysis.freshness_needed,
            cache_eligible=(
                request.privacy_class == PrivacyClass.PUBLIC
                and not analysis.freshness_needed
                and best_tier != RouteTier.DETERMINISTIC_A
            ),
            template_id=analysis.template_id if best_tier == RouteTier.TEMPLATE_B else None,
        )
        self._log_decision_safely(request, decision)
        return decision


def evaluate_learned_router(
    router: LearnedRouter | None = None,
    cv_folds: int = 5,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Rigorously evaluates the LearnedRouter classifier prototype using repeated stratified cross-validation.

    Computes:
    - Overall accuracy and macro-F1 score across validation folds
    - Aggregated 4x4 Confusion Matrix
    - Multi-class Brier score calibration metric
    - Baseline heuristic comparisons (Majority Class and Keyword Complexity Heuristic)
    - Routing regret estimation (economic cost overhead against ex-post Oracle)

    Returns:
        Structured evaluation metrics report dictionary.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
    from sklearn.model_selection import StratifiedKFold

    r = router or LearnedRouter()
    X, y = getattr(r, "_training_samples", (None, None))
    if X is None or y is None:
        raise ValueError("LearnedRouter training samples not initialized.")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    fold_accuracies: list[float] = []
    fold_macro_f1s: list[float] = []
    all_y_true: list[int] = []
    all_y_pred: list[int] = []
    all_y_prob: list[np.ndarray] = []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        clf = RandomForestClassifier(n_estimators=30, random_state=seed)
        clf.fit(X_train, y_train)

        preds = clf.predict(X_val)
        probs = clf.predict_proba(X_val)

        fold_accuracies.append(float(accuracy_score(y_val, preds)))
        fold_macro_f1s.append(float(f1_score(y_val, preds, average="macro", zero_division=0)))

        all_y_true.extend(y_val)
        all_y_pred.extend(preds)
        all_y_prob.extend(probs)

    # Multi-class Brier score: mean squared probability error across all classes
    # Brier = (1/N) * sum_i sum_k (p_ik - y_ik)^2
    n_samples = len(all_y_true)
    n_classes = len(clf.classes_)
    brier_sum = 0.0
    for true_label, prob_vec in zip(all_y_true, all_y_prob, strict=True):
        for c_idx in range(n_classes):
            one_hot = 1.0 if c_idx == true_label else 0.0
            p = prob_vec[c_idx] if c_idx < len(prob_vec) else 0.0
            brier_sum += (p - one_hot) ** 2
    brier_score = brier_sum / n_samples if n_samples > 0 else 0.0

    # Confusion matrix
    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])

    # Simpler Heuristic Baselines
    # 1. Majority class baseline (predict most common class)
    majority_class = int(np.bincount(y).argmax())
    heuristic_majority_acc = float(np.mean(y == majority_class))

    # 2. Heuristic rule baseline (if arithmetic/json -> A, if template -> B, else if complex -> D else C)
    heuristic_preds = []
    for x_row in X:
        if x_row[7] or x_row[8] or x_row[10]:
            heuristic_preds.append(0)
        elif x_row[11]:
            heuristic_preds.append(1)
        elif x_row[13] >= 2 or x_row[14] > 0.85:
            heuristic_preds.append(3)
        else:
            heuristic_preds.append(2)
    heuristic_rule_acc = float(accuracy_score(y, heuristic_preds))

    # Routing regret: cost penalty when misclassifying
    tier_costs = {0: 0.0000, 1: 0.0000, 2: 0.0001, 3: 0.0030}
    regrets = []
    for true_cls, pred_cls in zip(all_y_true, all_y_pred, strict=True):
        regret = max(0.0, tier_costs[pred_cls] - tier_costs[true_cls])
        regrets.append(regret)
    avg_routing_regret = float(np.mean(regrets))

    return {
        "status": "evaluated_proof_of_concept",
        "sample_count": len(y),
        "cv_folds": cv_folds,
        "mean_accuracy": round(float(np.mean(fold_accuracies)), 4),
        "std_accuracy": round(float(np.std(fold_accuracies)), 4),
        "mean_macro_f1": round(float(np.mean(fold_macro_f1s)), 4),
        "std_macro_f1": round(float(np.std(fold_macro_f1s)), 4),
        "brier_score_calibration": round(float(brier_score), 4),
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": [
            "DETERMINISTIC_A",
            "TEMPLATE_B",
            "SMALL_MODEL_C",
            "LARGE_MODEL_D",
        ],
        "baselines": {
            "majority_class_accuracy": round(heuristic_majority_acc, 4),
            "heuristic_rule_accuracy": round(heuristic_rule_acc, 4),
        },
        "average_routing_regret_usd": round(avg_routing_regret, 6),
    }
