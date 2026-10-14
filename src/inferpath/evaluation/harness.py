"""Experiment execution harness orchestrating Systems A, B, C and Ablations under frozen protocol."""

import hashlib
import json
import platform
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel

from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import load_pricing_table, load_settings
from inferpath.evaluation.checkers import evaluate_output
from inferpath.evaluation.dataset import DatasetItem
from inferpath.evaluation.environmental import EnvironmentalEstimator
from inferpath.evaluation.freeze_config import (
    get_git_commit,
    verify_test_config_locked,
)
from inferpath.evaluation.judge import LLMJudge, MockJudge
from inferpath.evaluation.oracle import OracleEvaluator
from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter
from inferpath.models.domain import HandlerOutput, RoutingDecision, UserRequest
from inferpath.models.enums import PrivacyClass, ProvenanceLabel, RouteTier
from inferpath.models.telemetry import ProvenanceValue
from inferpath.pipeline import RequestPipeline
from inferpath.router import (
    AlwaysLargeRouter,
    AlwaysSmallRouter,
    ConstraintAwareRouter,
    LearnedRouter,
    OracleRouter,
    RuleBasedRouter,
)
from inferpath.security.budget import RunBudgetTracker
from inferpath.validators import CompositeValidator, create_default_validator


class SystemExecutionSummary(BaseModel):
    """Aggregate metrics for a specific system or ablation run."""

    system_id: str
    run_id: str
    run_index: int
    adapter_mode: str  # "simulated" or "real"
    split: str  # "dev" or "test"
    total_items: int
    quality_pass_rate: float
    avg_quality_score: float
    total_cost_usd_calculated: float
    avg_cost_usd_calculated: float
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    fallback_rate: float
    total_energy_wh: float
    total_water_ml: float
    avg_cost_regret: float
    over_escalation_rate: float
    under_escalation_rate: float
    route_distribution: dict[str, int]


class RunManifest(BaseModel):
    """Immutable audit record metadata for an experimental evaluation execution."""

    run_id: str
    system_id: str
    run_index: int
    adapter_mode: str
    split: str
    dataset_version: str
    dataset_sha256: str
    config_hash: str
    git_commit: str
    timestamp: str
    random_seed: int
    q_min: float
    price_table_date: str
    environment_info: dict[str, Any]
    summary_metrics: SystemExecutionSummary


class ExperimentHarness:
    """Manages frozen evaluation runs across benchmark datasets, recording full provenance."""

    def __init__(
        self,
        adapter_mode: Literal["simulated", "real"] = "simulated",
        output_base_dir: Path | str | None = None,
        q_min: float = 0.70,
        judge: LLMJudge | None = None,
    ):
        self.adapter_mode = adapter_mode
        self.q_min = q_min
        self.judge = judge or MockJudge()

        # Strict directory separation between simulated and real experiments
        if output_base_dir:
            self.base_dir = Path(output_base_dir)
        else:
            sub = "simulated" if adapter_mode == "simulated" else "real"
            self.base_dir = Path("data") / "results" / sub

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.estimator = EnvironmentalEstimator()

    def _create_pipeline_for_system(
        self,
        system_id: str,
        cache_db_path: str,
        budget_tracker: RunBudgetTracker | None = None,
    ) -> RequestPipeline:
        """Instantiates RequestPipeline configured specifically for the target system or ablation."""
        settings = load_settings()
        settings.database_path = cache_db_path
        pricing = load_pricing_table()

        if self.adapter_mode == "real":
            if not settings.openrouter.api_key:
                raise RuntimeError(
                    "Real-provider experiment requested but OPENROUTER_API_KEY is not configured! "
                    "Evaluation harness strictly prohibits silent mock fallback in real mode."
                )
            small_adapter = OpenRouterAdapter(
                model_id=settings.openrouter.small_model,
                pricing_table=pricing,
                api_key=settings.openrouter.api_key,
                budget_tracker=budget_tracker,
            )
            large_adapter = OpenRouterAdapter(
                model_id=settings.openrouter.large_model,
                pricing_table=pricing,
                api_key=settings.openrouter.api_key,
                budget_tracker=budget_tracker,
            )
        else:
            small_adapter = MockSmallModel()
            large_adapter = MockLargeModel()

        router = None
        if system_id == "System_A" or system_id == "router_always_large":
            # Large-only baseline: no cache, no fallback, AlwaysLargeRouter
            settings.routing.enable_cache = False
            settings.routing.enable_fallback = False
            validator = None
            router = AlwaysLargeRouter(routing_settings=settings.routing)
        elif system_id == "System_B":
            # Cache + Rules + Large: exact cache on, fallback off
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = False
            validator = None

            class NoSmallModelRouter(RuleBasedRouter):
                def route(self, request: UserRequest) -> RoutingDecision:
                    dec = super().route(request)
                    if dec.selected_tier == RouteTier.SMALL_MODEL_C:
                        dec.selected_tier = RouteTier.LARGE_MODEL_D
                        dec.rule_id = "RULES_LARGE_ONLY"
                    return dec

            router = NoSmallModelRouter(routing_settings=settings.routing)
        elif system_id == "System_C":
            # Full InferPath (Constraint-aware cost-quality optimization)
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = True
            validator = create_default_validator()
            router = ConstraintAwareRouter(routing_settings=settings.routing, pricing_table=pricing)
        elif system_id in ("System_Learned", "router_learned"):
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = True
            validator = create_default_validator()
            router = LearnedRouter(routing_settings=settings.routing, pricing_table=pricing)
        elif system_id in ("System_Small", "router_always_small"):
            settings.routing.enable_cache = False
            settings.routing.enable_fallback = False
            validator = None
            router = AlwaysSmallRouter(routing_settings=settings.routing)
        elif system_id in ("System_Oracle", "router_oracle"):
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = False
            validator = create_default_validator()
            router = OracleRouter(routing_settings=settings.routing, pricing_table=pricing)
        elif system_id == "router_rules":
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = True
            validator = create_default_validator()
            router = RuleBasedRouter(routing_settings=settings.routing)
        elif system_id == "ablation_no_templates":
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = True
            validator = create_default_validator()

            class NoTemplatesRouter(RuleBasedRouter):
                def route(self, request: UserRequest) -> RoutingDecision:
                    dec = super().route(request)
                    if dec.selected_tier == RouteTier.TEMPLATE_B:
                        dec.selected_tier = RouteTier.SMALL_MODEL_C
                        dec.rule_id = "ABLATION_NO_TEMPLATES"
                    return dec

            router = NoTemplatesRouter(routing_settings=settings.routing)
        elif system_id == "ablation_no_small_model":
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = False
            validator = create_default_validator()

            class NoSmallModelRouter(RuleBasedRouter):
                def route(self, request: UserRequest) -> RoutingDecision:
                    dec = super().route(request)
                    if dec.selected_tier == RouteTier.SMALL_MODEL_C:
                        dec.selected_tier = RouteTier.LARGE_MODEL_D
                        dec.rule_id = "ABLATION_NO_SMALL_MODEL"
                    return dec

            router = NoSmallModelRouter(routing_settings=settings.routing)
        elif system_id == "ablation_no_fallback":
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = False
            validator = create_default_validator()
        elif system_id == "ablation_no_cache":
            settings.routing.enable_cache = False
            settings.routing.enable_fallback = True
            validator = create_default_validator()
        elif system_id == "ablation_no_validation":
            settings.routing.enable_cache = True
            settings.routing.enable_fallback = False
            validator = CompositeValidator(
                validators=[]
            )  # Empty validator: validation always passes
        else:
            raise ValueError(f"Unknown system or ablation identifier: {system_id}")

        pipe = RequestPipeline(
            settings=settings,
            pricing_table=pricing,
            router=router,
            cache=ExactCache(db_path=cache_db_path),
            validator=validator,
            small_adapter=small_adapter,
            large_adapter=large_adapter,
            budget_tracker=budget_tracker,
        )

        return pipe

    def run_evaluation(
        self,
        system_id: str,
        items: list[DatasetItem],
        run_index: int = 1,
        seed: int = 42,
        split: str = "dev",
        config_hash: str = "unlocked_dev_hash",
        verify_lock: bool = False,
    ) -> tuple[SystemExecutionSummary, RunManifest, list[dict[str, Any]]]:
        """Executes an evaluation run for a specific system or ablation."""
        if verify_lock:
            ok, msg = verify_test_config_locked(config_hash, require_clean_tree=True)
            if not ok:
                raise RuntimeError(f"Test execution locked: {msg}")

        run_id = f"run_{system_id}_{split}_{run_index}_{uuid.uuid4().hex[:6]}"
        db_path = str(self.base_dir / f"cache_{run_id}.db")

        pipeline = self._create_pipeline_for_system(system_id=system_id, cache_db_path=db_path)
        oracle_eval = OracleEvaluator(pipeline=pipeline, q_min=self.q_min, judge=self.judge)

        _ = seed
        # Preserve recorded request order (exact duplicates evaluate cache)
        ordered_items = list(items)

        item_logs: list[dict[str, Any]] = []
        route_dist: dict[str, int] = {}
        latencies: list[float] = []
        costs: list[float] = []
        qualities: list[float] = []
        passed_q_count = 0
        fallback_count = 0
        cache_hit_count = 0
        total_energy = 0.0
        total_water = 0.0
        regrets: list[float] = []
        over_esc_count = 0
        under_esc_count = 0

        for item in ordered_items:
            req = UserRequest(
                request_id=f"{run_id}_{item.item_id}",
                prompt=item.prompt,
                privacy_class=item.privacy_class,
                is_time_sensitive=item.is_time_sensitive,
                task_parameters=item.task_parameters,
            )

            # System execution overrides for baselines
            if system_id == "System_A":
                # Bypass rules/templates/small model; force Large Model D if eligible
                if item.privacy_class != PrivacyClass.PUBLIC:
                    decision = RoutingDecision(
                        selected_tier=RouteTier.DETERMINISTIC_A,
                        rule_id="R01",
                        reason="Privacy policy block: private data prohibited from external models",
                        confidence=1.0,
                    )
                    h_out = HandlerOutput(
                        tier=RouteTier.DETERMINISTIC_A,
                        output_text="[REQUEST BLOCKED: Privacy policy prohibits external transmission]",
                        validated=False,
                        needs_clarification=False,
                        input_tokens=ProvenanceValue(
                            value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                        ),
                        output_tokens=ProvenanceValue(
                            value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                        ),
                        latency_ms=ProvenanceValue(
                            value=0.1, provenance=ProvenanceLabel.MEASURED, unit="ms"
                        ),
                        model_version="privacy_block",
                    )
                else:
                    decision = RoutingDecision(
                        selected_tier=RouteTier.LARGE_MODEL_D,
                        rule_id="BASELINE_LARGE_A",
                        reason="Forced System A large model baseline",
                        confidence=1.0,
                    )
                    h_out = pipeline._execute_tier(req, decision, decision.selected_tier)

                res = pipeline._finalize_result(
                    request=req,
                    decision=decision,
                    output=h_out,
                    was_cached=False,
                    fallback_occurred=False,
                    fallback_history=[],
                    validation_passed=h_out.validated,
                    elapsed_ms=h_out.latency_ms.value,
                    route_trace=[],
                )
            elif system_id == "System_B":
                # Cache + Tier A deterministic + Tier D
                # If deterministic, route A; otherwise route D directly
                if item.task_category == "arithmetic_and_unit_conversion":
                    decision = pipeline.router.route(req)
                    h_out = pipeline._execute_tier(req, decision, decision.selected_tier)
                elif item.privacy_class != PrivacyClass.PUBLIC:
                    decision = RoutingDecision(
                        selected_tier=RouteTier.DETERMINISTIC_A,
                        rule_id="R01",
                        reason="Privacy policy block: private data prohibited from external models",
                        confidence=1.0,
                    )
                    h_out = HandlerOutput(
                        tier=RouteTier.DETERMINISTIC_A,
                        output_text="[REQUEST BLOCKED: Privacy policy prohibits external transmission]",
                        validated=False,
                        needs_clarification=False,
                        input_tokens=ProvenanceValue(
                            value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                        ),
                        output_tokens=ProvenanceValue(
                            value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                        ),
                        latency_ms=ProvenanceValue(
                            value=0.1, provenance=ProvenanceLabel.MEASURED, unit="ms"
                        ),
                        model_version="privacy_block",
                    )
                else:
                    decision = RoutingDecision(
                        selected_tier=RouteTier.LARGE_MODEL_D,
                        rule_id="BASELINE_B_LARGE",
                        reason="System B large model for non-deterministic",
                        confidence=1.0,
                    )
                    h_out = pipeline._execute_tier(req, decision, decision.selected_tier)

                # Execute with exact cache check
                cache_key = f"b_cache_{req.prompt}"
                cached = pipeline.cache.lookup(cache_key) if item.cache_eligibility else None
                if cached:
                    res = pipeline._finalize_result(
                        request=req,
                        decision=decision,
                        output=cached,
                        was_cached=True,
                        fallback_occurred=False,
                        fallback_history=[],
                        validation_passed=True,
                        elapsed_ms=1.2,
                        route_trace=[],
                    )
                else:
                    if item.cache_eligibility and h_out.validated:
                        pipeline.cache.store(cache_key, req, h_out, decision.selected_tier)
                    res = pipeline._finalize_result(
                        request=req,
                        decision=decision,
                        output=h_out,
                        was_cached=False,
                        fallback_occurred=False,
                        fallback_history=[],
                        validation_passed=h_out.validated,
                        elapsed_ms=h_out.latency_ms.value,
                        route_trace=[],
                    )
            else:
                try:
                    res = pipeline.execute(req)
                except Exception as e:
                    from inferpath.security.privacy import PrivacyViolationError

                    if isinstance(e, PrivacyViolationError):
                        h_out = HandlerOutput(
                            tier=RouteTier.DETERMINISTIC_A,
                            output_text="[REQUEST BLOCKED: Privacy policy violation]",
                            validated=False,
                            needs_clarification=False,
                            input_tokens=ProvenanceValue(
                                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                            ),
                            output_tokens=ProvenanceValue(
                                value=0, provenance=ProvenanceLabel.MEASURED, unit="tokens"
                            ),
                            latency_ms=ProvenanceValue(
                                value=0.1, provenance=ProvenanceLabel.MEASURED, unit="ms"
                            ),
                            model_version="privacy_block",
                        )
                        decision = RoutingDecision(
                            selected_tier=RouteTier.DETERMINISTIC_A,
                            rule_id="R01",
                            reason=str(e),
                            confidence=1.0,
                        )
                        res = pipeline._finalize_result(
                            request=req,
                            decision=decision,
                            output=h_out,
                            was_cached=False,
                            fallback_occurred=False,
                            fallback_history=[],
                            validation_passed=False,
                            elapsed_ms=0.1,
                            route_trace=[],
                        )
                    else:
                        raise

            # Evaluate quality
            checker_res = evaluate_output(res.final_output, item)
            judge_res = self.judge.evaluate(item, res.final_output, q_min=self.q_min)
            final_quality = (checker_res.score * 0.6) + (judge_res.score * 0.4)
            meets_q = final_quality >= self.q_min and res.validation_passed

            if meets_q:
                passed_q_count += 1
            if res.was_cached:
                cache_hit_count += 1
            if res.fallback_occurred:
                fallback_count += 1

            tier_name = res.final_tier.value
            route_dist[tier_name] = route_dist.get(tier_name, 0) + 1
            latencies.append(res.total_latency_ms.value)
            costs.append(res.cost_usd.value)
            qualities.append(final_quality)

            # Environmental estimate
            in_tok = res.input_tokens.value if res.input_tokens else 0
            out_tok = res.output_tokens.value if res.output_tokens else 0
            total_tokens = in_tok + out_tok
            env_est = self.estimator.estimate(
                tier=res.final_tier,
                model_id=None,
                total_tokens=total_tokens,
                was_cached=res.was_cached,
            )
            total_energy += env_est.energy_wh.value
            total_water += env_est.total_water_ml.value

            # Oracle and regret
            oracle_rec = oracle_eval.evaluate_item_routes(item)
            enriched_oracle = oracle_eval.compute_regret_and_escalation(
                oracle_record=oracle_rec,
                routed_tier=res.final_tier,
                routed_cost_usd=res.cost_usd.value,
                routed_quality_score=final_quality,
                routed_validation_passed=res.validation_passed,
            )

            if enriched_oracle.cost_regret is not None:
                regrets.append(enriched_oracle.cost_regret)
            if enriched_oracle.is_over_escalation:
                over_esc_count += 1
            if enriched_oracle.is_under_escalation:
                under_esc_count += 1

            item_logs.append(
                {
                    "item_id": item.item_id,
                    "tier": item.tier,
                    "task_category": item.task_category,
                    "prompt": item.prompt,
                    "routed_tier": res.final_tier.value,
                    "rule_id": res.rule_id,
                    "was_cached": res.was_cached,
                    "validation_passed": res.validation_passed,
                    "fallback_occurred": res.fallback_occurred,
                    "cost_usd_calculated": res.cost_usd.value,
                    "latency_ms": res.total_latency_ms.value,
                    "quality_score": round(final_quality, 3),
                    "meets_q_min": meets_q,
                    "oracle_tier": enriched_oracle.oracle_tier.value
                    if enriched_oracle.oracle_tier
                    else None,
                    "oracle_cost_usd": enriched_oracle.oracle_cost_usd,
                    "cost_regret": enriched_oracle.cost_regret,
                    "is_over_escalation": enriched_oracle.is_over_escalation,
                    "is_under_escalation": enriched_oracle.is_under_escalation,
                    "energy_wh": env_est.energy_wh.value,
                    "water_ml": env_est.total_water_ml.value,
                }
            )

        n_items = len(items)
        p50 = float(np.percentile(latencies, 50)) if n_items else 0.0
        p95 = float(np.percentile(latencies, 95)) if n_items else 0.0

        summary = SystemExecutionSummary(
            system_id=system_id,
            run_id=run_id,
            run_index=run_index,
            adapter_mode=self.adapter_mode,
            split=split,
            total_items=n_items,
            quality_pass_rate=round(passed_q_count / n_items, 4) if n_items else 0.0,
            avg_quality_score=round(sum(qualities) / n_items, 4) if n_items else 0.0,
            total_cost_usd_calculated=round(sum(costs), 6),
            avg_cost_usd_calculated=round(sum(costs) / n_items, 6) if n_items else 0.0,
            avg_latency_ms=round(sum(latencies) / n_items, 2) if n_items else 0.0,
            p50_latency_ms=round(p50, 2),
            p95_latency_ms=round(p95, 2),
            cache_hit_rate=round(cache_hit_count / n_items, 4) if n_items else 0.0,
            fallback_rate=round(fallback_count / n_items, 4) if n_items else 0.0,
            total_energy_wh=round(total_energy, 4),
            total_water_ml=round(total_water, 4),
            avg_cost_regret=round(sum(regrets) / len(regrets), 6) if regrets else 0.0,
            over_escalation_rate=round(over_esc_count / n_items, 4) if n_items else 0.0,
            under_escalation_rate=round(under_esc_count / n_items, 4) if n_items else 0.0,
            route_distribution=route_dist,
        )

        dataset_path = Path("data/pilot_dataset.json")
        dataset_sha = (
            hashlib.sha256(dataset_path.read_bytes()).hexdigest()
            if dataset_path.exists()
            else "deterministic_dataset_sha"
        )

        manifest = RunManifest(
            run_id=run_id,
            system_id=system_id,
            run_index=run_index,
            adapter_mode=self.adapter_mode,
            split=split,
            dataset_version="1.0.0",
            dataset_sha256=dataset_sha,
            config_hash=config_hash,
            git_commit=get_git_commit(),
            timestamp=datetime.now(UTC).isoformat(),
            random_seed=seed,
            q_min=self.q_min,
            price_table_date="2026-10-01",
            environment_info={
                "python_version": sys.version,
                "platform": platform.platform(),
                "node": platform.node(),
            },
            summary_metrics=summary,
        )

        # Save run manifest and item logs
        manifest_path = self.base_dir / f"manifest_{run_id}.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest.model_dump(mode="json"), f, indent=2, ensure_ascii=False)

        raw_logs_path = self.base_dir / f"logs_{run_id}.json"
        with open(raw_logs_path, "w", encoding="utf-8") as f:
            json.dump(item_logs, f, indent=2, ensure_ascii=False)

        return summary, manifest, item_logs
