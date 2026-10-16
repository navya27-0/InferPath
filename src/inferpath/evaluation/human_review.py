"""Blind human review dataset sampling, CSV generation, and inter-rater agreement."""

import csv
import json
import logging
import random
from io import StringIO
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inferpath.evaluation.dataset import DatasetItem
from inferpath.models.enums import PrivacyClass
from inferpath.router.analyzer import contains_sensitive_patterns


class HumanReviewRecord(BaseModel):
    """Documented ingestion format for empirical human evaluation review records."""

    review_id: str = Field(description="Unique review identifier (e.g. REV_0001_R1)")
    item_id: str = Field(description="Stable item identifier referencing DatasetItem")
    reviewer_id: str = Field(description="Anonymized reviewer identifier (e.g. reviewer_01)")
    candidate_id: str = Field(description="Candidate identifier (e.g. candidate_a, candidate_b)")
    score: int = Field(ge=1, le=5, description="Ordinal quality score from 1 to 5")
    rubric_version: str = Field(default="1.0.0", description="Rubric version applied")
    provenance: str = Field(
        default="unknown", description="Rating provenance (measured / synthetic / unknown)"
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


EVALUATION_RUBRIC_5_POINT: dict[int, str] = {
    5: "Exceptional: Fully correct, comprehensive, well-structured, satisfies all constraints and criteria.",
    4: "Acceptable: Factually accurate and usable, minor stylistic or formatting imperfections.",
    3: "Marginal: Partially correct; key reasoning present but omitted non-critical details or minor format slip.",
    2: "Deficient: Significant inaccuracies, hallucinated elements, or failed structural validator.",
    1: "Failed: Completely incorrect, empty, refusal, offensive, or opposite of requested intent.",
}


def load_human_review_csv(csv_path: Path | str) -> list[HumanReviewRecord]:
    """Loads and validates human review records from a populated review CSV.

    Ignores uncompleted template rows where scores are empty.
    Validates rating values (1 to 5) and rating provenance (e.g. measured).
    """
    path = Path(csv_path)
    if not path.exists():
        return []

    def _parse_strict_rating(score_str: str) -> int | None:
        score_str = score_str.strip()
        if not score_str:
            return None
        try:
            val_float = float(score_str)
            if not val_float.is_integer():
                return None
            val_int = int(val_float)
            if 1 <= val_int <= 5:
                return val_int
            return None
        except (ValueError, TypeError):
            return None

    records: list[HumanReviewRecord] = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            raw_prov = row.get("provenance")
            prov = raw_prov.strip().lower() if raw_prov and raw_prov.strip() else "unknown"

            # Check candidate A score
            score_a_str = row.get("score_a_1_to_5", "").strip()
            if score_a_str:
                s_a = _parse_strict_rating(score_a_str)
                if s_a is not None:
                    records.append(
                        HumanReviewRecord(
                            review_id=f"{row.get('review_id', 'REV')}_A",
                            item_id=row.get("item_id", ""),
                            reviewer_id=row.get("reviewer_id", "anonymous"),
                            candidate_id="candidate_a",
                            score=s_a,
                            rubric_version="1.0.0",
                            provenance=prov,
                            rationale=row.get("rationale"),
                            disagreement_notes=row.get("disagreement_notes"),
                        )
                    )

            # Check candidate B score
            score_b_str = row.get("score_b_1_to_5", "").strip()
            if score_b_str:
                s_b = _parse_strict_rating(score_b_str)
                if s_b is not None:
                    records.append(
                        HumanReviewRecord(
                            review_id=f"{row.get('review_id', 'REV')}_B",
                            item_id=row.get("item_id", ""),
                            reviewer_id=row.get("reviewer_id", "anonymous"),
                            candidate_id="candidate_b",
                            score=s_b,
                            rubric_version="1.0.0",
                            provenance=prov,
                            rationale=row.get("rationale"),
                            disagreement_notes=row.get("disagreement_notes"),
                        )
                    )

    return records


def compute_inter_rater_agreement(
    records: list[HumanReviewRecord],
    judge_scores: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Computes inter-rater agreement from empirical review records.

    If genuine dual ratings are unavailable, explicitly reports that agreement
    has not yet been established rather than fabricating numbers.
    Strictly verifies that rating input values are valid and provenance is measured.
    """
    if not records:
        return {
            "status": "not_established",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": 0,
            "message": "No empirical human review records available.",
        }

    # Verify rating provenance: empirical inter-rater agreement strictly requires measured provenance
    non_measured = [r for r in records if r.provenance != "measured"]
    if non_measured:
        return {
            "status": "unverified_provenance",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": len(records),
            "message": f"Found {len(non_measured)} ratings with non-measured provenance ('{non_measured[0].provenance}'). Empirical agreement requires measured ratings.",
        }

    # Verify rating score bounds
    invalid_scores = [r for r in records if not (isinstance(r.score, int) and 1 <= r.score <= 5)]
    if invalid_scores:
        return {
            "status": "invalid_ratings",
            "human_human_kappa": None,
            "human_judge_kappa": None,
            "sample_size": len(records),
            "message": f"Found {len(invalid_scores)} ratings with out-of-bounds scores.",
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

    # Compute human-judge agreement if judge scores are provided
    hj_kappa = None
    if judge_scores:
        human_consensus: list[int] = []
        judge_ratings: list[int] = []
        for (item_id, _cand_id), scores in paired.items():
            if item_id in judge_scores:
                avg_score = round(sum(scores.values()) / len(scores.values()))
                raw_judge = judge_scores[item_id]
                judge_mapped = max(1, min(5, round(1 + raw_judge * 4)))
                human_consensus.append(avg_score)
                judge_ratings.append(judge_mapped)
        if len(human_consensus) >= 5:
            hj_kappa = compute_weighted_cohen_kappa(
                human_consensus, judge_ratings, categories=[1, 2, 3, 4, 5]
            )

    return {
        "status": "measured",
        "human_human_kappa": kappa,
        "human_judge_kappa": hj_kappa,
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
    key_path: Path | str | None = None,
    sample_ratio: float = 0.20,
    seed: int = 42,
) -> str:
    """
    Generates a blinded human review CSV with randomized candidate order and no model identities.
    Ensures stratified representation across every (tier, task_category) stratum.
    Enforces privacy policy: non-public items are strictly excluded from human review export.
    Never silently substitutes mock/placeholder responses.
    """
    rng = random.Random(seed)

    # 1. Enforce privacy policy: only export public items
    eligible_items: list[DatasetItem] = []
    for item in test_items:
        p_class = getattr(item, "privacy_class", PrivacyClass.PUBLIC)
        if isinstance(p_class, str):
            try:
                p_class = PrivacyClass(p_class)
            except ValueError:
                p_class = PrivacyClass.CONFIDENTIAL_PRIVATE
        if p_class == PrivacyClass.PUBLIC and not contains_sensitive_patterns(item.prompt):
            eligible_items.append(item)

    # 2. Stratified sampling across eligible public items
    strata: dict[tuple[str, str], list[DatasetItem]] = {}
    for item in eligible_items:
        key = (item.tier, item.task_category)
        strata.setdefault(key, []).append(item)

    sampled_items: list[DatasetItem] = []
    for _key, items in sorted(strata.items()):
        # At least 1 item per stratum, or sample_ratio proportion
        sample_count = max(1, int(round(len(items) * sample_ratio)))
        sampled_items.extend(rng.sample(items, min(sample_count, len(items))))

    # At least 30 items for dual review (if available)
    if len(sampled_items) < 30 and len(eligible_items) >= 30:
        remaining = [item for item in eligible_items if item not in sampled_items]
        needed = min(30 - len(sampled_items), len(remaining))
        sampled_items.extend(rng.sample(remaining, needed))

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
    unblinding_keys: dict[str, Any] = {}
    missing_items: list[str] = []

    for item in sampled_items:
        outputs = system_outputs.get(item.item_id, {})
        text_a = outputs.get("System_A") or outputs.get("large_only")
        text_b = outputs.get("System_C") or outputs.get("full_inferpath")

        # Never silently substitute placeholder responses; report missing candidate outputs
        if (
            not text_a
            or not text_b
            or text_a == "[REDACTED_CONFIDENTIAL]"
            or text_b == "[REDACTED_CONFIDENTIAL]"
            or contains_sensitive_patterns(item.prompt)
        ):
            missing_items.append(item.item_id)
            continue

        # Randomize assignment to Candidate A vs Candidate B
        if rng.random() > 0.5:
            cand_a, cand_b = text_a, text_b
            sys_a, sys_b = "System_A", "System_C"
        else:
            cand_a, cand_b = text_b, text_a
            sys_a, sys_b = "System_C", "System_A"

        rev_id_1 = f"REV_{review_count:04d}_R1"
        unblinding_keys[rev_id_1] = {
            "item_id": item.item_id,
            "candidate_a_system": sys_a,
            "candidate_b_system": sys_b,
        }

        # Primary review
        writer.writerow(
            [
                rev_id_1,
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
            rev_id_2 = f"REV_{review_count:04d}_R2"
            unblinding_keys[rev_id_2] = {
                "item_id": item.item_id,
                "candidate_a_system": sys_a,
                "candidate_b_system": sys_b,
            }
            writer.writerow(
                [
                    rev_id_2,
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

    if missing_items:
        logging.getLogger(__name__).warning(
            "Omitted %d items from blind human review sheet due to missing candidate outputs: %s",
            len(missing_items),
            missing_items[:5],
        )

    csv_content = buffer.getvalue()
    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="") as f:
            f.write(csv_content)

    if key_path:
        k_out = Path(key_path)
        k_out.parent.mkdir(parents=True, exist_ok=True)
        with open(k_out, "w", encoding="utf-8") as f:
            json.dump(unblinding_keys, f, indent=2)

    return csv_content
