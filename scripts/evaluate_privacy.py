"""Script to evaluate privacy classification and adversarial DLP performance."""

from __future__ import annotations

import json
from pathlib import Path

from inferpath.evaluation.privacy_eval import evaluate_privacy_classifier

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_FILE = REPO_ROOT / "data" / "results" / "privacy_evaluation.json"


def main() -> int:
    results = evaluate_privacy_classifier()

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    cm = results["confusion_matrix"]
    m = results["metrics"]
    print(f"Privacy classification evaluation saved to {OUTPUT_FILE}")
    print(
        f"  Probes: {results['total_probes']} (TP={cm['true_positives']}, FP={cm['false_positives']}, TN={cm['true_negatives']}, FN={cm['false_negatives']})"
    )
    print(
        f"  Precision: {m['precision']:.3f}, Recall: {m['recall_sensitivity']:.3f}, F1: {m['f1_score']:.3f}"
    )
    print(f"  FPR: {m['false_positive_rate']:.3f}, FNR: {m['false_negative_rate']:.3f}")
    return 0


if __name__ == "__main__":
    main()
