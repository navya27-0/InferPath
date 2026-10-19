"""Pilot dataset schemas, loader, and split validation for InferPath evaluation."""

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from inferpath.models.enums import PrivacyClass

CategoryType = Literal[
    # Core 8 Pilot Categories
    "arithmetic_and_unit_conversion",
    "templated_documents",
    "rewriting",
    "short_explanations",
    "summarization",
    "classification",
    "multi_step_reasoning",
    "synthesis",
    # 10 Realistic Workload Classes
    "long_context",
    "code_generation",
    "debugging",
    "multi_turn",
    "tool_calling",
    "multilingual",
    "domain_specific_private",
    "prompt_injection",
    "freshness_sensitive",
    "distribution_shifted",
]

TierType = Literal["easy", "moderate", "difficult"]
SplitType = Literal["dev", "test"]
CheckerType = Literal["exact", "regex", "numeric_tolerance", "rubric"]


class DatasetItem(BaseModel):
    """Schema for a single pilot evaluation dataset item."""

    item_id: str = Field(description="Stable unique identifier for the item (e.g. item_001)")
    prompt: str = Field(description="User prompt text")
    tier: TierType = Field(
        description="Stratification tier (easy, moderate, difficult). Not a ground-truth routing label."
    )
    task_category: CategoryType = Field(
        description="One of the 8 required research task categories"
    )
    checker_type: CheckerType = Field(
        description="Evaluation checker mechanism (exact, regex, numeric_tolerance, rubric)"
    )
    checker_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Configuration for checker, e.g. reference answer, pattern, or tolerance",
    )
    split: SplitType = Field(description="Dataset split (dev or test)")
    task_family_id: str = Field(
        description="Semantic group / task family ID. Used for leakage-free split partitioning."
    )
    task_parameters: dict[str, Any] = Field(
        default_factory=dict,
        description="Task parameters such as expected_format, max_tokens, language, etc.",
    )
    privacy_class: PrivacyClass = Field(
        default=PrivacyClass.PUBLIC,
        description="Privacy classification enforcing transmission and caching rules.",
    )
    is_time_sensitive: bool = Field(
        default=False,
        description="Flag indicating whether query requires real-time information (bypasses cache).",
    )
    cache_eligibility: bool = Field(
        default=True,
        description="Whether this query is theoretically eligible for caching.",
    )
    quality_criteria: list[str] = Field(
        default_factory=list,
        description="Explicit criteria used by human reviewers and LLM judges.",
    )


class PilotDataset(BaseModel):
    """Collection model for the versioned InferPath pilot dataset."""

    version: str = "1.0.0"
    description: str = "InferPath Pilot Evaluation Benchmark Dataset"
    created_date: str = "2026-10-13"
    items: list[DatasetItem] = Field(default_factory=list)

    @property
    def total_count(self) -> int:
        return len(self.items)

    def get_split(self, split: SplitType) -> list[DatasetItem]:
        """Returns all items belonging to the requested split."""
        return [item for item in self.items if item.split == split]

    def compute_sha256(self) -> str:
        """Computes deterministic SHA-256 digest over sorted items."""
        serialized = json.dumps(
            [item.model_dump(mode="json") for item in sorted(self.items, key=lambda x: x.item_id)],
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def validate_split_integrity(self) -> tuple[bool, list[str]]:
        """
        Verifies that no task_family_id crosses the split boundary between dev and test.
        Returns (is_valid, error_messages).
        """
        dev_families = {item.task_family_id for item in self.items if item.split == "dev"}
        test_families = {item.task_family_id for item in self.items if item.split == "test"}

        overlap = dev_families.intersection(test_families)
        errors: list[str] = []
        if overlap:
            errors.append(
                f"Prohibited semantic-group leakage detected! Overlapping families: {sorted(overlap)}"
            )

        if len(self.items) < 300:
            errors.append(f"Pilot dataset must have at least 300 items, found {len(self.items)}.")

        # Verify all 8 categories exist in both splits
        for s in ("dev", "test"):
            categories_in_split = {item.task_category for item in self.items if item.split == s}
            required_cats = {
                "arithmetic_and_unit_conversion",
                "templated_documents",
                "rewriting",
                "short_explanations",
                "summarization",
                "classification",
                "multi_step_reasoning",
                "synthesis",
            }
            missing = required_cats - categories_in_split
            if missing:
                errors.append(f"Split '{s}' is missing required task categories: {sorted(missing)}")

        return (len(errors) == 0, errors)

    def get_strata_breakdown(self) -> dict[str, Any]:
        """Produces a breakdown count by (tier, category, split)."""
        breakdown: Counter[tuple[str, str, str]] = Counter()
        for item in self.items:
            breakdown[(item.split, item.tier, item.task_category)] += 1

        formatted: dict[str, Any] = {}
        for (split, tier, cat), count in sorted(breakdown.items()):
            formatted.setdefault(split, {}).setdefault(tier, {})[cat] = count
        return formatted


def load_pilot_dataset(dataset_path: Path | str | None = None) -> PilotDataset:
    """Loads and validates pilot dataset from disk."""
    if dataset_path is None:
        cand = Path.cwd() / "data" / "pilot_dataset.json"
        if not cand.exists():
            cand = (
                Path(__file__).resolve().parent.parent.parent.parent / "data" / "pilot_dataset.json"
            )
        dataset_path = cand

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Pilot dataset not found at {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return PilotDataset(**data)


def load_comprehensive_dataset(dataset_path: Path | str | None = None) -> PilotDataset:
    """Loads extended comprehensive workload dataset containing all 18 categories."""
    if dataset_path is None:
        cand = Path.cwd() / "data" / "comprehensive_workload_dataset.json"
        if not cand.exists():
            cand = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "data"
                / "comprehensive_workload_dataset.json"
            )
        dataset_path = cand

    path = Path(dataset_path)
    if not path.exists():
        raise FileNotFoundError(f"Comprehensive workload dataset not found at {path}")

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    return PilotDataset(**data)
