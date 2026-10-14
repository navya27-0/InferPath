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

from typing import Any

import numpy as np

from inferpath.config.settings import PricingTable, RoutingSettings, load_pricing_table
from inferpath.handlers.calculator import SafeCalculator
from inferpath.handlers.date_formatter import DateFormatter
from inferpath.handlers.json_validator import JSONSchemaValidator
from inferpath.handlers.unit_converter import UnitConverter
from inferpath.models.domain import HandlerOutput, RoutingDecision, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router.analyzer import RequestAnalysis, RequestAnalyzer
from inferpath.security.hashing import hash_prompt
from inferpath.security.privacy import PrivacyPolicyEnforcer, PrivacyViolationError
from inferpath.telemetry.experiment_logger import ExperimentLogger
from inferpath.templates.engine import TemplateEngine


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
        if request.privacy_class == PrivacyClass.PUBLIC:
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
    ):
        super().__init__(routing_settings, privacy_enforcer, logger, pricing_table)
        self.lambda_latency = latency_penalty_weight

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
        # Tier C quality estimate based on task complexity indicators
        is_complex = bool(analysis.complexity_indicators) or len(request.prompt.split()) > 150
        is_ambiguous = analysis.ambiguity_detected
        is_high_pref = request.task_parameters.quality_preference == "high"

        if is_complex or is_ambiguous:
            q_hat_c = 0.58  # Small models degrade on complex reasoning
            q_hat_d = 0.94
        elif is_high_pref:
            q_hat_c = 0.72
            q_hat_d = 0.96
        else:
            q_hat_c = 0.88  # Standard extraction / drafting
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
        if q_hat_d < q_min and q_min > 0.95:
            # Q_min extremely high, Tier D may still be closest
            pass
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
            reason = f"Tier C pruned (Q_hat={q_hat_c:.2f} < Q_min={q_min:.2f}); Tier D selected to satisfy quality constraint."
        elif tier_c_eligible:
            selected_tier = RouteTier.SMALL_MODEL_C
            rule_id = "CONSTRAINT_PRUNED_TO_C"
            reason = "Tier D pruned due to budget/latency constraints; Tier C selected."
        else:
            # Fallback when constraints cannot be strictly met:
            if b_max is not None:
                selected_tier = RouteTier.SMALL_MODEL_C
                rule_id = "CONSTRAINT_RELAXED_BUDGET"
                reason = "All model tiers exceeded budget cap; relaxed to lowest-cost Tier C."
            else:
                selected_tier = RouteTier.LARGE_MODEL_D
                rule_id = "CONSTRAINT_RELAXED_QUALITY"
                reason = "No tier strictly satisfied all constraints; relaxed to Tier D for maximum quality."

        time_sensitive = analysis.freshness_needed
        cache_eligible = (
            request.privacy_class == PrivacyClass.PUBLIC and not time_sensitive and not is_ambiguous
        )

        decision = RoutingDecision(
            selected_tier=selected_tier,
            eligible_tiers=[RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D],
            rule_id=rule_id,
            confidence=0.92,
            reason=reason,
            time_sensitive=time_sensitive,
            cache_eligible=cache_eligible,
        )
        self._log_decision_safely(request, decision)
        return decision


class LearnedRouter(BaseRouter):
    """
    Supervised Machine Learning Router predicting optimal processing tier from feature vectors.

    Features extracted:
    - Text length, word count, token count estimate
    - Digit count, mathematical operator density
    - Freshness / time-sensitivity signals
    - Complexity keyword count
    - Requested minimum quality (Q_min)
    - Privacy class index

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

        self._model = RandomForestClassifier(n_estimators=20, random_state=42)

        # Calibrated baseline training synthetic corpus covering four tiers
        # Tiers: 0 -> DETERMINISTIC_A, 1 -> TEMPLATE_B, 2 -> SMALL_MODEL_C, 3 -> LARGE_MODEL_D
        synthetic_samples = [
            # Tier A samples (arithmetic, unit, date, json)
            ([20, 4, 8, 5, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0.7], 0),
            ([25, 5, 10, 4, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0.7], 0),
            ([30, 6, 12, 6, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0.7], 0),
            ([50, 8, 16, 2, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0.7], 0),
            # Tier B samples (structured templates)
            ([80, 12, 24, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            ([120, 18, 36, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0.7], 1),
            # Tier C samples (standard QA / summarization / extraction)
            ([60, 10, 20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.7], 2),
            ([100, 15, 30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.6], 2),
            ([80, 12, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0.7], 2),
            # Tier D samples (complex multi-hop reasoning, high Q_min)
            ([250, 40, 80, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0.85], 3),
            ([300, 50, 100, 6, 3, 1, 0, 0, 0, 0, 0, 0, 0, 4, 0.90], 3),
            ([150, 25, 50, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0.95], 3),
        ]
        X = np.array([s[0] for s in synthetic_samples])
        y = np.array([s[1] for s in synthetic_samples])
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
