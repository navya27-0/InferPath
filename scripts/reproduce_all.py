"""One-command master reproduction runner for InferPath research evaluation.

Executes and verifies:
1. Static repository consistency checks (versions, artifacts, core modules, links).
2. Hermetic unit and integration test suite.
3. Simulator calibration against empirical baseline.
4. Distribution shift evaluation across 10 shift dimensions.
5. Concurrency scalability and tail latency analysis.
6. Privacy classification & adversarial DLP evaluation.
7. Model registry and market drift evaluation.
8. Total Cost of Ownership (TCO) evaluation.
9. Workload trace caching evaluation.
10. Expanded multi-rater human evaluation study.
11. Public reproducibility manifest generation.
12. README benchmark results synchronization.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def run_step(step_name: str, cmd: list[str]) -> bool:
    print("\n==================================================")
    print(f"  Step: {step_name}")
    print(f"  Command: {' '.join(cmd)}")
    print("==================================================")
    start = time.time()
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    duration = time.time() - start

    if res.returncode == 0:
        print(f"  [PASS] {step_name} completed in {duration:.2f}s")
        if res.stdout.strip():
            # Print brief preview of stdout
            lines = res.stdout.strip().splitlines()
            for line in lines[:5]:
                print(f"    {line}")
            if len(lines) > 5:
                print(f"    ... ({len(lines) - 5} more lines)")
        return True
    else:
        print(f"  [FAIL] {step_name} exited with code {res.returncode} in {duration:.2f}s")
        if res.stderr.strip():
            print(f"  Error output:\n{res.stderr.strip()}")
        if res.stdout.strip():
            print(f"  Standard output:\n{res.stdout.strip()}")
        return False


def main() -> int:
    py = sys.executable
    steps = [
        ("Static Repository Consistency Check", [py, "scripts/verify_repo_consistency.py"]),
        ("Hermetic Offline Unit Test Suite", [py, "-m", "pytest", "-q", "-m", "not live"]),
        ("Simulator Calibration Evaluation", [py, "scripts/evaluate_simulator_calibration.py"]),
        ("Distribution Shift Evaluation", [py, "scripts/evaluate_distribution_shift.py"]),
        ("Concurrency Scalability Profiling", [py, "scripts/benchmark_scalability.py"]),
        ("Privacy & Adversarial DLP Evaluation", [py, "scripts/evaluate_privacy.py"]),
        ("Model Registry & Market Drift Evaluation", [py, "scripts/evaluate_model_drift.py"]),
        ("Total Cost of Ownership (TCO) Evaluation", [py, "scripts/evaluate_tco.py"]),
        ("Workload Trace Caching Evaluation", [py, "scripts/evaluate_caching_workloads.py"]),
        ("Expanded Multi-Rater Human Study", [py, "scripts/run_expanded_human_study.py"]),
        ("Public Reproducibility Manifest Generator", [py, "scripts/generate_public_manifest.py"]),
        ("README Results Table Synchronization", [py, "scripts/update_readme_results.py"]),
    ]

    total_start = time.time()
    all_passed = True
    passed_count = 0

    for name, cmd in steps:
        success = run_step(name, cmd)
        if success:
            passed_count += 1
        else:
            all_passed = False
            break

    total_dur = time.time() - total_start
    print("\n==================================================")
    print("  InferPath Master Reproduction Summary")
    print(f"  Steps Passed: {passed_count}/{len(steps)}")
    print(f"  Total Duration: {total_dur:.2f}s")
    print(f"  Status: {'ALL PASS' if all_passed else 'FAILED'}")
    print("==================================================")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
