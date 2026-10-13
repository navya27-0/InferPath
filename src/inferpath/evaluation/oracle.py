"""Route-level ground truth, Oracle selection, and regret calculations."""

from pydantic import BaseModel

from inferpath.evaluation.checkers import evaluate_output
from inferpath.evaluation.dataset import DatasetItem
from inferpath.evaluation.judge import LLMJudge, MockJudge
from inferpath.models.domain import UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.pipeline import RequestPipeline


class RouteExecutionResult(BaseModel):
    """Result of evaluating a specific route on a single prompt."""

    tier: RouteTier
    output_text: str
    is_eligible: bool
    ineligibility_reason: str | None = None
    validation_passed: bool
    quality_score: float
    meets_q_min: bool
    input_tokens: int
    output_tokens: int
    latency_ms: float
    cost_usd_calculated: float
    provider_reported_cost_usd: float | None = None


class ItemOracleRecord(BaseModel):
    """Oracle outcome and regret evaluation for a dataset item."""

    item_id: str
    prompt: str
    tier_stratum: str
    task_category: str
    q_min: float
    route_results: dict[str, RouteExecutionResult]
    oracle_tier: RouteTier | None = None
    oracle_cost_usd: float | None = None
    no_qualifying_route: bool = False
    routed_tier: RouteTier | None = None
    routed_cost_usd: float | None = None
    routed_meets_q_min: bool = False
    cost_regret: float | None = None
    normalized_regret: float | None = None
    is_over_escalation: bool = False
    is_under_escalation: bool = False


class OracleEvaluator:
    """Evaluates candidate routes and computes regret against the empirical oracle."""

    def __init__(
        self,
        pipeline: RequestPipeline,
        q_min: float = 0.70,
        judge: LLMJudge | None = None,
    ):
        self.pipeline = pipeline
        self.q_min = q_min
        self.judge = judge or MockJudge()

    def evaluate_item_routes(self, item: DatasetItem) -> ItemOracleRecord:
        """Executes applicable routes for an item and determines the oracle route."""
        req = UserRequest(
            request_id=f"oracle_{item.item_id}",
            prompt=item.prompt,
            privacy_class=item.privacy_class,
            is_time_sensitive=item.is_time_sensitive,
            task_parameters=item.task_parameters,
        )

        # Route tiers to test
        tested_tiers = [
            RouteTier.DETERMINISTIC_A,
            RouteTier.TEMPLATE_B,
            RouteTier.SMALL_MODEL_C,
            RouteTier.LARGE_MODEL_D,
        ]

        route_results: dict[str, RouteExecutionResult] = {}

        for tier in tested_tiers:
            # Check route eligibility:
            # Privacy restriction prevents external models
            if item.privacy_class != PrivacyClass.PUBLIC and tier in (
                RouteTier.SMALL_MODEL_C,
                RouteTier.LARGE_MODEL_D,
            ):
                route_results[tier.value] = RouteExecutionResult(
                    tier=tier,
                    output_text="[BLOCKED BY PRIVACY POLICY]",
                    is_eligible=False,
                    ineligibility_reason=f"Privacy class '{item.privacy_class.value}' prohibits external model routing.",
                    validation_passed=False,
                    quality_score=0.0,
                    meets_q_min=False,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    cost_usd_calculated=0.0,
                )
                continue

            # Execute the specific tier directly
            try:
                from inferpath.models.domain import RoutingDecision

                if tier in (RouteTier.DETERMINISTIC_A, RouteTier.TEMPLATE_B):
                    routed_decision = self.pipeline.router.route(req)
                    if routed_decision.selected_tier != tier:
                        route_results[tier.value] = RouteExecutionResult(
                            tier=tier,
                            output_text=f"[NOT SUPPORTED BY {tier.value}]",
                            is_eligible=False,
                            ineligibility_reason=f"Request does not match any {tier.value} rules or templates.",
                            validation_passed=False,
                            quality_score=0.0,
                            meets_q_min=False,
                            input_tokens=0,
                            output_tokens=0,
                            latency_ms=0.0,
                            cost_usd_calculated=0.0,
                        )
                        continue
                    decision = routed_decision
                else:
                    decision = RoutingDecision(
                        selected_tier=tier,
                        rule_id=f"ORACLE_TEST_{tier.name}",
                        reason=f"Forced execution of {tier.value} for ground truth.",
                        confidence=1.0,
                    )

                handler_out = self.pipeline._execute_tier(req, decision, tier)
                val_res = self.pipeline.validator.validate(
                    handler_out.output_text, req, current_tier=tier
                )

                # Quality evaluation
                checker_res = evaluate_output(handler_out.output_text, item)
                judge_res = self.judge.evaluate(item, handler_out.output_text, q_min=self.q_min)
                final_score = (checker_res.score * 0.6) + (judge_res.score * 0.4)

                cost_usd = 0.0
                if handler_out.raw_response and "cost_usd_calculated" in handler_out.raw_response:
                    cost_usd = float(handler_out.raw_response["cost_usd_calculated"])

                route_results[tier.value] = RouteExecutionResult(
                    tier=tier,
                    output_text=handler_out.output_text,
                    is_eligible=True,
                    validation_passed=val_res.is_valid,
                    quality_score=round(final_score, 3),
                    meets_q_min=final_score >= self.q_min and val_res.is_valid,
                    input_tokens=handler_out.input_tokens.value,
                    output_tokens=handler_out.output_tokens.value,
                    latency_ms=handler_out.latency_ms.value,
                    cost_usd_calculated=cost_usd,
                    provider_reported_cost_usd=None,
                )
            except Exception as e:
                route_results[tier.value] = RouteExecutionResult(
                    tier=tier,
                    output_text=f"[EXECUTION ERROR: {str(e)[:100]}]",
                    is_eligible=True,
                    validation_passed=False,
                    quality_score=0.0,
                    meets_q_min=False,
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=0.0,
                    cost_usd_calculated=0.0,
                )

        # Determine Oracle: cheapest eligible route that meets Q_min
        qualifying = [
            res
            for res in route_results.values()
            if res.is_eligible and res.meets_q_min and res.validation_passed
        ]

        if qualifying:
            # Sort by cost ascending, then latency ascending
            oracle = min(qualifying, key=lambda x: (x.cost_usd_calculated, x.latency_ms))
            oracle_tier = oracle.tier
            oracle_cost = oracle.cost_usd_calculated
            no_qualifying = False
        else:
            oracle_tier = None
            oracle_cost = None
            no_qualifying = True

        return ItemOracleRecord(
            item_id=item.item_id,
            prompt=item.prompt,
            tier_stratum=item.tier,
            task_category=item.task_category,
            q_min=self.q_min,
            route_results=route_results,
            oracle_tier=oracle_tier,
            oracle_cost_usd=oracle_cost,
            no_qualifying_route=no_qualifying,
        )

    def compute_regret_and_escalation(
        self,
        oracle_record: ItemOracleRecord,
        routed_tier: RouteTier,
        routed_cost_usd: float,
        routed_quality_score: float,
        routed_validation_passed: bool,
    ) -> ItemOracleRecord:
        """Enriches an oracle record with actual routed outcome and computes regret."""
        rec = oracle_record.model_copy()
        rec.routed_tier = routed_tier
        rec.routed_cost_usd = routed_cost_usd
        rec.routed_meets_q_min = (routed_quality_score >= self.q_min) and routed_validation_passed

        # Tiers order by typical cost
        tier_hierarchy = [
            RouteTier.DETERMINISTIC_A,
            RouteTier.TEMPLATE_B,
            RouteTier.SMALL_MODEL_C,
            RouteTier.LARGE_MODEL_D,
        ]

        if rec.oracle_cost_usd is not None:
            regret = max(0.0, routed_cost_usd - rec.oracle_cost_usd)
            rec.cost_regret = round(regret, 6)
            if rec.oracle_cost_usd > 0.0:
                rec.normalized_regret = round(regret / rec.oracle_cost_usd, 4)
            else:
                rec.normalized_regret = 0.0

            # Over-escalation: routed to higher-tier than oracle when oracle met Q_min
            if rec.oracle_tier:
                o_idx = tier_hierarchy.index(rec.oracle_tier)
                r_idx = tier_hierarchy.index(routed_tier)
                rec.is_over_escalation = r_idx > o_idx
                # Under-escalation: routed to lower-tier that failed Q_min while oracle succeeded
                rec.is_under_escalation = r_idx < o_idx and not rec.routed_meets_q_min

        return rec
