"""Generates sanitized public reproducibility manifests for published benchmarks."""

from __future__ import annotations

import json
from pathlib import Path

from inferpath.evaluation.reproducibility import (
    compute_file_sha256,
    sanitize_run_manifest,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURE_MANIFEST = (
    REPO_ROOT / "data" / "results" / "fixtures" / "manifest_run_fixture_evaluation.json"
)
OUTPUT_FILE = REPO_ROOT / "data" / "results" / "fixtures" / "sanitized_public_manifest.json"
DATASET_PATH = REPO_ROOT / "data" / "comprehensive_workload_dataset.json"
PRICING_PATH = REPO_ROOT / "configs" / "pricing_table_2026_10.yaml"


def main() -> int:
    raw_manifest = {}
    if FIXTURE_MANIFEST.exists():
        with open(FIXTURE_MANIFEST, encoding="utf-8") as f:
            raw_manifest = json.load(f)

    # Compute live file hashes
    raw_manifest["dataset_sha256"] = compute_file_sha256(DATASET_PATH)
    raw_manifest["pricing_table_sha256"] = compute_file_sha256(PRICING_PATH)
    raw_manifest["code_commit_sha"] = "57caf142107ba28b39167a8f09d7f1c562747fd8"

    sanitized = sanitize_run_manifest(raw_manifest)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(sanitized, f, indent=2)

    print(f"Sanitized public manifest saved to {OUTPUT_FILE}")
    print(f"  Dataset SHA-256: {sanitized['hashes']['dataset_sha256'][:16]}...")
    print(f"  Pricing Table SHA-256: {sanitized['hashes']['pricing_table_sha256'][:16]}...")
    return 0


if __name__ == "__main__":
    main()
