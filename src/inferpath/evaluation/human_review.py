"""Blind human review dataset sampling, CSV generation, and inter-rater agreement."""

import csv
import random
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inferpath.evaluation.dataset import DatasetItem


class HumanReviewRecord(BaseModel):
    """Documented ingestion format for empirical human evaluation review records."""

    review_id: str = Field(description="Unique review identifier (e.g. REV_0001_R1)")
    item_id: str = Field(description="Stable item identifier referencing DatasetItem")
    reviewer_id: str = Field(description="Anonymized reviewer identifier (e.g. reviewer_01)")
    candidate_id: str = Field(description="Candidate identifier (e.g. candidate_a, candidate_b)")
    score: int = Field(ge=1, le=5, description="Ordinal quality score from 1 to 5")
    rubric_version: str = Field(default="1.0.0", description="Rubric version applied")
    provenance: str = Field(
        default="measured", description="Rating provenance (measured / synthetic)"
    )
    rationale: str | None = Field(default=None, description="Reviewer justification or notes")
    disagreement_notes: str | None = Field(
        default=None, description="Arbitration or consensus notes"
    )


def validate_ratings(ratings: list[int | float], min_val: int = 1, max_val: int = 5) -> bool:
    """Validates that all ratings are within expected ordinal boundaries."""
    if not ratings:
        return False
    return all(isinstance(r, (int, float)) and min_val <= r <= max_val for r in ratings)


def load_human_review_csv(csv_path: Path | str) -> list[HumanReviewRecord]:
    """Loads and validates human review records from a populated review CSV.

    Ignores uncompleted template rows where scores are empty.
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    records: list[HumanReviewRecord] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Check candidate A score
            score_a_str = row.get("score_a_1_to_5", "").strip()
            if score_a_str:
                try:
                    s_a = int(float(score_a_str))
                    records.append(
                        HumanReviewRecord(
                            review_id=f"{row.get('review_id', 'REV')}_A",
                            item_id=row.get("item_id", ""),
                            reviewer_id=row.get("reviewer_id", "anonymous"),
                            candidate_id="candidate_a",
                            score=s_a,
                            rubric_version="1.0.0",
                            provenance="measured",
                            rationale=row.get("rationale"),
                            disagreement_notes=row.get("disagreement_notes"),
                        )
                    )
                except (ValueError, TypeError):
                    pass

            # Check candidate B score
            score_b_str = row.get("score_b_1_to_5", "").strip()
            if score_b_str:
                try:
                    s_b = int(float(score_b_str))
                    records.append(
                        HumanReviewRecord(
                            review_id=f"{row.get('review_id', 'REV')}_B",
                            item_id=row.get("item_id", ""),
                            reviewer_id=row.get("reviewer_id", "anonymous"),
                            candidate_id="candidate_b",
                            score=s_b,
                            rubric_version="1.0.0",
                            provenance="measured",
                            rationale=row.get("rationale"),
                            disagreement_notes=row.get("disagreement_notes"),
                        )
                    )
                except (ValueError, TypeError):
                    pass

    return records


def compute_inter_rater_agreement(
    records: list[HumanReviewRecord],
) -> dict[str, Any]:
    """Computes inter-rater agreement from empirical review records.

    If genuine dual ratings are unavailable, explicitly reports that agreement
    has not yet been established rather than fabricating numbers.
    """
    if not records:
        return {
            "status": "not_established",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": 0,
            "message": "No empirical human review records available.",
        }

    # Group ratings by (item_id, candidate_id) -> {reviewer_id: score}
    paired: dict[tuple[str, str], dict[str, int]] = {}
    for r in records:
        paired.setdefault((r.item_id, r.candidate_id), {})[r.reviewer_id] = r.score

    # Find common reviewer pairs
    all_reviewers = sorted(list({r.reviewer_id for r in records}))
    if len(all_reviewers) < 2:
        return {
            "status": "not_established",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": len(records),
            "message": f"Requires at least 2 distinct reviewers, found {len(all_reviewers)}.",
        }

    r1_id, r2_id = all_reviewers[0], all_reviewers[1]
    scores_r1: list[int] = []
    scores_r2: list[int] = []

    for pair_scores in paired.values():
        if r1_id in pair_scores and r2_id in pair_scores:
            scores_r1.append(pair_scores[r1_id])
            scores_r2.append(pair_scores[r2_id])

    if len(scores_r1) < 5:
        return {
            "status": "not_established",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": len(scores_r1),
            "message": "Fewer than 5 overlapping items between dual reviewers.",
        }

    if not validate_ratings(scores_r1) or not validate_ratings(scores_r2):
        return {
            "status": "invalid_ratings",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": len(scores_r1),
            "message": "Submitted ratings failed range/type validation.",
        }

    kappa = compute_weighted_cohen_kappa(scores_r1, scores_r2, categories=[1, 2, 3, 4, 5])
    return {
        "status": "measured",
        "human_human_kappa": kappa,
        "human_judge_kappa": None,  # Separate when judge predictions are paired
        "sample_size": len(scores_r1),
        "reviewers": [r1_id, r2_id],
    }


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
