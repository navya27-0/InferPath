from typing import Protocol, runtime_checkable

from inferpath.models.domain import RoutingDecision, UserRequest
from inferpath.router.analyzer import RequestAnalysis, RequestAnalyzer
from inferpath.router.formulation import ConstrainedOptimizationPolicy
from inferpath.router.router import RuleBasedRouter
from inferpath.router.strategies import (
    AlwaysLargeRouter,
    AlwaysSmallRouter,
    BaseRouter,
    ConfidenceEscalationRouter,
    ConstraintAwareRouter,
    ConstraintsUnsatisfiableError,
    CostOnlyRouter,
    LatencyOnlyRouter,
    LearnedRouter,
    OracleRouter,
    PromptLengthRouter,
    QualityOnlyRouter,
    RandomConstrainedRouter,
    SemanticSimilarityRouter,
    evaluate_learned_router,
)


@runtime_checkable
class Router(Protocol):
    """Standardized router protocol implemented by all InferPath routing strategies."""

    def route(self, request: UserRequest) -> RoutingDecision: ...


__all__ = [
    "Router",
    "AlwaysLargeRouter",
    "AlwaysSmallRouter",
    "BaseRouter",
    "ConfidenceEscalationRouter",
    "ConstrainedOptimizationPolicy",
    "ConstraintAwareRouter",
    "ConstraintsUnsatisfiableError",
    "CostOnlyRouter",
    "LatencyOnlyRouter",
    "LearnedRouter",
    "OracleRouter",
    "PromptLengthRouter",
    "QualityOnlyRouter",
    "RandomConstrainedRouter",
    "RequestAnalysis",
    "RequestAnalyzer",
    "RuleBasedRouter",
    "SemanticSimilarityRouter",
    "evaluate_learned_router",
]
