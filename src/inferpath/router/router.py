"""Explicit, ordered rule-based router for InferPath.

Enforces strict priority order:
1. Privacy policy (R01)
2. Freshness detection and cache bypass (R02)
3. Deterministic handlers (R03: arithmetic, R04: unit conversion, R05: date formatting, R06: JSON validation)
4. Structured templates (R07) with strict clarification on missing fields
5. Complex reasoning heuristic (R08)
6. Small model tier (R09)
7. Ambiguous or high-quality escalation (R10)

The router only selects the route. Cache lookup occurs downstream in the pipeline.
"""

from inferpath.config.settings import RoutingSettings
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


class RuleBasedRouter:
    """
    Deterministic rule-based inference router.

    Invariants:
    - Rules are explicit, numbered (R01..R10), and evaluated in strict priority order.
    - Router NEVER makes LLM calls to make routing decisions.
    - Router NEVER accesses the cache directly (cache lookup is deferred to downstream pipeline).
    - Missing template fields trigger clarification requests, never hallucinated defaults or model calls.
    - Private prompt text is never logged beyond a SHA-256 digest.
    """

    def __init__(
        self,
        routing_settings: RoutingSettings | None = None,
        privacy_enforcer: PrivacyPolicyEnforcer | None = None,
        logger: ExperimentLogger | None = None,
    ):
        self.settings = routing_settings or RoutingSettings()
        self.privacy_enforcer = privacy_enforcer or PrivacyPolicyEnforcer()
        self.logger = logger
        self.analyzer = RequestAnalyzer(self.settings)
        self.template_engine = TemplateEngine()

    def route(self, request: UserRequest) -> RoutingDecision:
        """
        Analyzes the request and selects the optimal processing tier.
        Does not execute external models or query the cache.
        """
        analysis = self.analyzer.analyze(request)
        decision = self._evaluate_rules(request, analysis)

        # Log decision safely without exposing raw private prompts
        self._log_decision_safely(request, decision)

        return decision

    def _evaluate_rules(self, request: UserRequest, analysis: RequestAnalysis) -> RoutingDecision:
        # ---------------------------------------------------------
        # Priority 1: Deterministic & Template Local Capabilities
        # Note: Local deterministic and template tiers are safe for private requests
        # ---------------------------------------------------------

        # Check Deterministic R03: Arithmetic
        if analysis.task_intent == "arithmetic":
            return RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R03",
                confidence=analysis.confidence,
                reason="Arithmetic expression routed to safe deterministic AST calculator.",
                time_sensitive=False,
                cache_eligible=False,  # Skipped for deterministic by policy
            )

        # Check Deterministic R04: Unit Conversion
        if analysis.task_intent == "unit_conversion":
            return RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R04",
                confidence=analysis.confidence,
                reason="Unit conversion routed to deterministic dimensional converter.",
                time_sensitive=False,
                cache_eligible=False,  # Skipped for deterministic by policy
            )

        # Check Deterministic R05: Date Formatting
        if analysis.task_intent == "date_formatting":
            return RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R05",
                confidence=analysis.confidence,
                reason="Explicit date string routed to deterministic date formatter.",
                time_sensitive=False,
                cache_eligible=False,  # Skipped for deterministic by policy
            )

        # Check Deterministic R06: JSON & Schema Validation
        if analysis.task_intent == "json_validation":
            return RoutingDecision(
                selected_tier=RouteTier.DETERMINISTIC_A,
                eligible_tiers=[RouteTier.DETERMINISTIC_A],
                rule_id="R06",
                confidence=analysis.confidence,
                reason="JSON payload routed to deterministic parser and schema validator.",
                time_sensitive=False,
                cache_eligible=False,  # Skipped for deterministic by policy
            )

        # Check Template R07: Structured Template Document Generation
        if analysis.template_id:
            needs_clarif = bool(analysis.missing_inputs)
            clarif_prompt = (
                f"Please provide the missing required information for {analysis.template_id}: "
                f"{', '.join(analysis.missing_inputs)}."
                if needs_clarif
                else None
            )
            return RoutingDecision(
                selected_tier=RouteTier.TEMPLATE_B,
                eligible_tiers=[RouteTier.TEMPLATE_B],
                rule_id="R07",
                confidence=analysis.confidence,
                reason=(
                    f"Template '{analysis.template_id}' matched; clarification required for: {', '.join(analysis.missing_inputs)}"
                    if needs_clarif
                    else f"Document generation routed to structured Jinja2 template '{analysis.template_id}'."
                ),
                template_id=analysis.template_id,
                time_sensitive=False,
                cache_eligible=not needs_clarif and analysis.privacy_class == PrivacyClass.PUBLIC,
                needs_clarification=needs_clarif,
                clarification_prompt=clarif_prompt,
                missing_fields=analysis.missing_inputs,
            )

        # ---------------------------------------------------------
        # Priority 2: Privacy Policy Enforcement (R01)
        # Any task that is NOT local deterministic/template requires model execution.
        # If the request is private/confidential and external transmission is disallowed,
        # it MUST be rejected or blocked.
        # ---------------------------------------------------------
        if analysis.privacy_class in (
            PrivacyClass.INTERNAL,
            PrivacyClass.CONFIDENTIAL_PRIVATE,
        ):
            can_transmit, reason = self.privacy_enforcer.can_transmit_externally(
                request, RouteTier.SMALL_MODEL_C
            )
            if not can_transmit:
                return RoutingDecision(
                    selected_tier=RouteTier.SMALL_MODEL_C,
                    eligible_tiers=[],
                    rule_id="R01",
                    confidence=1.0,
                    reason=reason
                    or "Privacy policy strictly prohibits external transmission of confidential/internal data.",
                    time_sensitive=False,
                    cache_eligible=False,
                )

        # ---------------------------------------------------------
        # Priority 3: Freshness / Dynamic Information (R02)
        # ---------------------------------------------------------
        if analysis.freshness_needed:
            # Time-sensitive query routed to model with cache_eligible=False
            tier = (
                RouteTier.LARGE_MODEL_D
                if analysis.complexity_indicators
                or request.task_parameters.quality_preference == "high"
                else RouteTier.SMALL_MODEL_C
            )
            return RoutingDecision(
                selected_tier=tier,
                eligible_tiers=[tier, RouteTier.LARGE_MODEL_D],
                rule_id="R02",
                confidence=0.90,
                reason=f"Time-sensitive request detected ({analysis.freshness_reason}); exact caching bypassed.",
                time_sensitive=True,
                cache_eligible=False,
            )

        # ---------------------------------------------------------
        # Priority 4: Complex Task Heuristic (R08)
        # (Strictly config-driven keyword list, NOT prompt length)
        # ---------------------------------------------------------
        if analysis.complexity_indicators:
            return RoutingDecision(
                selected_tier=RouteTier.LARGE_MODEL_D,
                eligible_tiers=[RouteTier.LARGE_MODEL_D],
                rule_id="R08",
                confidence=0.85,
                reason=f"Complex reasoning heuristic triggered by keywords: {', '.join(analysis.complexity_indicators)}.",
                time_sensitive=False,
                cache_eligible=analysis.privacy_class == PrivacyClass.PUBLIC,
            )

        # ---------------------------------------------------------
        # Priority 5: Ambiguous / High Quality Preference (R10)
        # ---------------------------------------------------------
        if analysis.ambiguity_detected or request.task_parameters.quality_preference == "high":
            return RoutingDecision(
                selected_tier=RouteTier.LARGE_MODEL_D,
                eligible_tiers=[RouteTier.LARGE_MODEL_D],
                rule_id="R10",
                confidence=0.75,
                reason=(
                    "Underspecified or ambiguous query routed to large model tier for robust handling."
                    if analysis.ambiguity_detected
                    else "Explicit high quality preference routed to large model tier."
                ),
                time_sensitive=False,
                cache_eligible=analysis.privacy_class == PrivacyClass.PUBLIC
                and not analysis.ambiguity_detected,
            )

        # ---------------------------------------------------------
        # Priority 6: Standard Task Small Model Tier (R09)
        # ---------------------------------------------------------
        return RoutingDecision(
            selected_tier=RouteTier.SMALL_MODEL_C,
            eligible_tiers=[RouteTier.SMALL_MODEL_C, RouteTier.LARGE_MODEL_D],
            rule_id="R09",
            confidence=0.80,
            reason="Standard drafting/QA query routed to resource-efficient small model tier.",
            time_sensitive=False,
            cache_eligible=analysis.privacy_class == PrivacyClass.PUBLIC,
        )

    def execute_local_handler(
        self, request: UserRequest, decision: RoutingDecision
    ) -> HandlerOutput:
        """
        Executes local deterministic or template handlers for Tiers A and B.
        Raises PrivacyViolationError if R01 blocked the request.
        """
        if decision.rule_id == "R01":
            raise PrivacyViolationError(decision.reason)

        if decision.selected_tier == RouteTier.DETERMINISTIC_A:
            if decision.rule_id == "R03":
                return SafeCalculator.execute_handler(request.prompt)
            elif decision.rule_id == "R04":
                return UnitConverter.execute_handler(request.prompt)
            elif decision.rule_id == "R05":
                return DateFormatter.execute_handler(request.prompt)
            elif decision.rule_id == "R06":
                schema = request.metadata.get("schema")
                return JSONSchemaValidator.execute_handler(request.prompt, schema=schema)
            raise ValueError(f"Unknown deterministic rule ID: {decision.rule_id}")

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
        # Only log raw prompt sample if public and contains no sensitive patterns
        if request.privacy_class == PrivacyClass.PUBLIC and not contains_sensitive_patterns(
            request.prompt
        ):
            metadata["prompt_preview"] = request.prompt[:100]

        # Use structured logging without exposing secrets
        self.logger.log_event("routing_decision", metadata)
