"""Router evaluation utilities: confusion matrix analysis, paraphrasing, adversarial, multilingual, and OOD evaluations."""

from __future__ import annotations

from typing import Any

import numpy as np

from inferpath.evaluation.dataset import DatasetItem
from inferpath.models.domain import RoutingDecision, UserRequest
from inferpath.models.enums import PrivacyClass, RouteTier
from inferpath.router import BaseRouter

TIER_LABELS = [
    RouteTier.DETERMINISTIC_A.value,
    RouteTier.TEMPLATE_B.value,
    RouteTier.SMALL_MODEL_C.value,
    RouteTier.LARGE_MODEL_D.value,
]

TIER_COSTS = {
    RouteTier.DETERMINISTIC_A.value: 0.000000,
    RouteTier.TEMPLATE_B.value: 0.000000,
    RouteTier.SMALL_MODEL_C.value: 0.000025,
    RouteTier.LARGE_MODEL_D.value: 0.003000,
}


def derive_ground_truth_tier(item: DatasetItem) -> RouteTier:
    """Derives expected canonical ground truth tier from task category and complexity stratum."""
    cat = item.task_category
    tier_stratum = item.tier

    if cat == "arithmetic_and_unit_conversion":
        return RouteTier.DETERMINISTIC_A
    if cat == "templated_documents":
        return RouteTier.TEMPLATE_B
    if cat in ("multi_step_reasoning", "synthesis"):
        return RouteTier.LARGE_MODEL_D
    if tier_stratum == "difficult":
        return RouteTier.LARGE_MODEL_D
    return RouteTier.SMALL_MODEL_C


def compute_routing_confusion_matrix(
    ground_truth_tiers: list[RouteTier | str] | None = None,
    predicted_tiers: list[RouteTier | str] | None = None,
    *,
    y_true: list[RouteTier | str] | None = None,
    y_pred: list[RouteTier | str] | None = None,
) -> dict[str, Any]:
    """
    Computes a 4x4 routing confusion matrix and associated classification metrics.

    Rows: Ground Truth Tiers
    Columns: Predicted Tiers
    """
    actual_gt = ground_truth_tiers if ground_truth_tiers is not None else y_true
    actual_pred = predicted_tiers if predicted_tiers is not None else y_pred
    if actual_gt is None or actual_pred is None:
        raise ValueError(
            "Must provide either (ground_truth_tiers, predicted_tiers) or (y_true, y_pred)."
        )

    label_to_idx = {lbl: i for i, lbl in enumerate(TIER_LABELS)}
    n_classes = len(TIER_LABELS)
    matrix = np.zeros((n_classes, n_classes), dtype=int)

    for gt, pred in zip(actual_gt, actual_pred, strict=True):
        gt_str = gt.value if isinstance(gt, RouteTier) else str(gt)
        pred_str = pred.value if isinstance(pred, RouteTier) else str(pred)

        if gt_str in label_to_idx and pred_str in label_to_idx:
            row = label_to_idx[gt_str]
            col = label_to_idx[pred_str]
            matrix[row, col] += 1

    total_samples = int(np.sum(matrix))
    correct = int(np.trace(matrix))
    accuracy = round(float(correct / total_samples), 4) if total_samples > 0 else 0.0

    per_class: dict[str, dict[str, float]] = {}
    f1_list: list[float] = []

    for i, lbl in enumerate(TIER_LABELS):
        tp = int(matrix[i, i])
        fp = int(np.sum(matrix[:, i]) - tp)
        fn = int(np.sum(matrix[i, :]) - tp)

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
        f1 = (
            round(2 * precision * recall / (precision + recall), 4)
            if (precision + recall) > 0
            else 0.0
        )

        per_class[lbl] = {
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "support": int(np.sum(matrix[i, :])),
        }
        f1_list.append(f1)

    macro_f1 = round(float(np.mean(f1_list)), 4) if f1_list else 0.0

    # Routing regret calculation
    total_regret = 0.0
    for i, gt_lbl in enumerate(TIER_LABELS):
        for j, pred_lbl in enumerate(TIER_LABELS):
            count = matrix[i, j]
            if count > 0:
                cost_diff = max(0.0, TIER_COSTS[pred_lbl] - TIER_COSTS[gt_lbl])
                total_regret += count * cost_diff

    avg_regret = round(float(total_regret / total_samples), 6) if total_samples > 0 else 0.0

    return {
        "confusion_matrix": matrix.tolist(),
        "labels": TIER_LABELS,
        "total_items": total_samples,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "average_routing_regret_usd": avg_regret,
        "per_class": per_class,
    }


def evaluate_router_robustness(
    router: BaseRouter,
    test_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    """
    Evaluates a router across diverse test cases, reporting accuracy,
    decisions, and the resulting confusion matrix.
    """
    gt_list: list[RouteTier] = []
    pred_list: list[RouteTier] = []
    results: list[dict[str, Any]] = []

    for tc in test_cases:
        prompt = tc["prompt"]
        expected_tier = tc["expected_tier"]
        category = tc.get("category", "general")
        privacy_class = tc.get("privacy_class")

        req = UserRequest(
            request_id=f"test_{len(results)}",
            prompt=prompt,
            privacy_class=privacy_class or PrivacyClass.PUBLIC,
        )
        decision: RoutingDecision = router.route(req)

        matched = decision.selected_tier == expected_tier
        gt_list.append(expected_tier)
        pred_list.append(decision.selected_tier)

        results.append(
            {
                "prompt": prompt,
                "category": category,
                "expected_tier": expected_tier.value,
                "selected_tier": decision.selected_tier.value,
                "rule_id": decision.rule_id,
                "matched": matched,
            }
        )

    cm_report = compute_routing_confusion_matrix(gt_list, pred_list)
    cm_report["test_results"] = results
    return cm_report
