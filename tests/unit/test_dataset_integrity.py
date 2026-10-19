"""Unit tests validating split integrity, stratification balance, and leak-free partitioning of the pilot dataset."""

from inferpath.evaluation.dataset import load_pilot_dataset
from inferpath.models.enums import PrivacyClass


def test_pilot_dataset_count_and_categories():
    dataset = load_pilot_dataset()
    assert dataset.total_count >= 300
    assert dataset.total_count == 320

    dev_items = dataset.get_split("dev")
    test_items = dataset.get_split("test")

    assert len(dev_items) == 160
    assert len(test_items) == 160

    required_categories = {
        "arithmetic_and_unit_conversion",
        "templated_documents",
        "rewriting",
        "short_explanations",
        "summarization",
        "classification",
        "multi_step_reasoning",
        "synthesis",
    }

    dev_cats = {item.task_category for item in dev_items}
    test_cats = {item.task_category for item in test_items}

    assert dev_cats == required_categories
    assert test_cats == required_categories


def test_split_integrity_zero_semantic_group_leakage():
    dataset = load_pilot_dataset()
    is_valid, errors = dataset.validate_split_integrity()

    assert is_valid is True, f"Validation errors: {errors}"
    assert len(errors) == 0

    dev_families = {item.task_family_id for item in dataset.get_split("dev")}
    test_families = {item.task_family_id for item in dataset.get_split("test")}

    # Zero leakage requirement: absolutely no overlapping family IDs
    overlap = dev_families.intersection(test_families)
    assert len(overlap) == 0, f"Leaked families between dev and test: {overlap}"
    assert len(dev_families) == 40
    assert len(test_families) == 40


def test_dataset_contains_required_characteristics():
    dataset = load_pilot_dataset()

    # 1. Exact repeats (items with identical prompt in the same split)
    prompt_counts = {}
    for item in dataset.items:
        key = (item.split, item.prompt)
        prompt_counts[key] = prompt_counts.get(key, 0) + 1

    repeats = [k for k, v in prompt_counts.items() if v > 1]
    assert len(repeats) >= 10, f"Expected at least 10 repeat prompts, found {len(repeats)}"

    # 2. Time-sensitive items (bypassing cache)
    time_sens = [item for item in dataset.items if item.is_time_sensitive]
    assert len(time_sens) >= 8
    assert all(item.cache_eligibility is False for item in time_sens)

    # 3. Private items (never sent to external models)
    private_items = [item for item in dataset.items if item.privacy_class != PrivacyClass.PUBLIC]
    assert len(private_items) >= 10
    assert all(item.cache_eligibility is False for item in private_items)

    # 4. Checkers coverage
    checker_types = {item.checker_type for item in dataset.items}
    assert checker_types == {"exact", "regex", "numeric_tolerance", "rubric"}

    # 5. Tiers coverage
    tiers = {item.tier for item in dataset.items}
    assert tiers == {"easy", "moderate", "difficult"}


def test_deterministic_sha256_reproducibility():
    d1 = load_pilot_dataset()
    d2 = load_pilot_dataset()

    digest1 = d1.compute_sha256()
    digest2 = d2.compute_sha256()

    assert digest1 == digest2
    assert len(digest1) == 64


def test_comprehensive_workload_dataset():
    """Validates the comprehensive workload dataset containing all 18 classes."""
    from inferpath.evaluation.dataset import load_comprehensive_dataset

    comp_dataset = load_comprehensive_dataset()
    assert comp_dataset.total_count == 340
    is_valid, errors = comp_dataset.validate_split_integrity()
    assert is_valid is True, f"Validation errors: {errors}"

    categories = {item.task_category for item in comp_dataset.items}
    assert "long_context" in categories
    assert "code_generation" in categories
    assert "debugging" in categories
    assert "multi_turn" in categories
    assert "tool_calling" in categories
    assert "multilingual" in categories
    assert "domain_specific_private" in categories
    assert "prompt_injection" in categories
    assert "freshness_sensitive" in categories
    assert "distribution_shifted" in categories
