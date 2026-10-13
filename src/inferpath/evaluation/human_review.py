"""Blind human review dataset sampling, CSV generation, and inter-rater agreement."""

import csv
import random
from io import StringIO
from pathlib import Path

from inferpath.evaluation.dataset import DatasetItem


def compute_weighted_cohen_kappa(
    rater_a: list[int | float],
    rater_b: list[int | float],
    categories: list[int | float] | None = None,
) -> float:
    """
    Computes linearly- or quadratically-weighted Cohen's kappa for ordinal ratings.
    Quadratic weights: w_ij = 1 - (i - j)^2 / (k - 1)^2.
    """
    if len(rater_a) != len(rater_b) or len(rater_a) == 0:
        return 0.0

    if categories is None:
        cats = sorted(list(set(rater_a).union(set(rater_b))))
    else:
        cats = sorted(categories)

    k = len(cats)
    if k <= 1:
        return 1.0 if rater_a == rater_b else 0.0

    cat_to_idx = {c: i for i, c in enumerate(cats)}
    n = len(rater_a)

    # Confusion matrix obs_matrix
    obs_matrix = [[0.0] * k for _ in range(k)]
    for a, b in zip(rater_a, rater_b, strict=True):
        if a in cat_to_idx and b in cat_to_idx:
            obs_matrix[cat_to_idx[a]][cat_to_idx[b]] += 1.0

    # Marginals
    row_sums = [sum(obs_matrix[i][j] for j in range(k)) for i in range(k)]
    col_sums = [sum(obs_matrix[i][j] for i in range(k)) for j in range(k)]

    # Expected matrix E
    E = [[(row_sums[i] * col_sums[j]) / n for j in range(k)] for i in range(k)]

    # Quadratic weight matrix W
    W = [[1.0 - ((i - j) ** 2) / ((k - 1) ** 2) for j in range(k)] for i in range(k)]

    po = sum(W[i][j] * obs_matrix[i][j] for i in range(k) for j in range(k)) / n
    pe = sum(W[i][j] * E[i][j] for i in range(k) for j in range(k)) / n

    if 1.0 - pe == 0:
        return 1.0

    kappa = (po - pe) / (1.0 - pe)
    return round(float(kappa), 4)


def generate_blind_human_review_csv(
    test_items: list[DatasetItem],
    system_outputs: dict[str, dict[str, str]],  # {item_id: {system_id: output_text}}
    output_path: Path | str | None = None,
    sample_ratio: float = 0.20,
    seed: int = 42,
) -> str:
    """
    Generates a blinded human review CSV with randomized candidate order and no model identities.
    Ensures stratified representation across every (tier, task_category) stratum.
    """
    rng = random.Random(seed)

    # Stratified sampling
    strata: dict[tuple[str, str], list[DatasetItem]] = {}
    for item in test_items:
        key = (item.tier, item.task_category)
        strata.setdefault(key, []).append(item)

    sampled_items: list[DatasetItem] = []
    for _key, items in sorted(strata.items()):
        # At least 1 item per stratum, or sample_ratio proportion
        sample_count = max(1, int(round(len(items) * sample_ratio)))
        sampled_items.extend(rng.sample(items, min(sample_count, len(items))))

    # At least 30 items for dual review
    if len(sampled_items) < 30 and len(test_items) >= 30:
        remaining = [item for item in test_items if item not in sampled_items]
        sampled_items.extend(rng.sample(remaining, 30 - len(sampled_items)))

    # Output CSV buffer
    buffer = StringIO()
    writer = csv.writer(buffer)

    # Headers: strict zero model identities or judge scores
    headers = [
        "review_id",
        "item_id",
        "tier",
        "task_category",
        "prompt",
        "required_criteria",
        "candidate_a_text",
        "candidate_b_text",
        "reviewer_id",
        "score_a_1_to_5",
        "score_b_1_to_5",
        "rationale",
        "disagreement_notes",
    ]
    writer.writerow(headers)

    review_count = 1
    # Systems to compare blindly: e.g. Baseline A (large only) vs Full InferPath C
    for item in sampled_items:
        outputs = system_outputs.get(item.item_id, {})
        text_a = outputs.get(
            "System_A", outputs.get("large_only", "Mock Large Baseline completion")
        )
        text_b = outputs.get("System_C", outputs.get("full_inferpath", "Mock InferPath completion"))

        # Randomize assignment to Candidate A vs Candidate B
        if rng.random() > 0.5:
            cand_a, cand_b = text_a, text_b
        else:
            cand_a, cand_b = text_b, text_a

        # Primary review
        writer.writerow(
            [
                f"REV_{review_count:04d}_R1",
                item.item_id,
                item.tier,
                item.task_category,
                item.prompt,
                "; ".join(item.quality_criteria),
                cand_a,
                cand_b,
                "Reviewer_1",
                "",
                "",
                "",
                "",
            ]
        )

        # Dual independent review for the first 35 items
        if review_count <= 35:
            writer.writerow(
                [
                    f"REV_{review_count:04d}_R2",
                    item.item_id,
                    item.tier,
                    item.task_category,
                    item.prompt,
                    "; ".join(item.quality_criteria),
                    cand_a,
                    cand_b,
                    "Reviewer_2",
                    "",
                    "",
                    "",
                    "",
                ]
            )

        review_count += 1

    csv_content = buffer.getvalue()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)

    return csv_content
