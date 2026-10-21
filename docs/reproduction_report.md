# Independent External Reproduction Guide & Verification Report

## 1. Overview for External Reviewers

This document provides independent reviewers and researchers with an end-to-end guide to audit and reproduce all research artifacts, empirical tables, and security invariants in InferPath.

---

## 2. Quickstart: One-Command Reproduction

The entire evaluation suite can be reproduced and validated via a single command:

```bash
python scripts/reproduce_all.py
```

### Execution Steps & Expected Durations
| Step | Action | Output Artifact | Expected Duration |
| :---: | :--- | :--- | :---: |
| **1** | Repository Consistency Check | Static verification | ~0.5s |
| **2** | Hermetic Test Suite | 340+ unit & integration tests | ~8.0s |
| **3** | Simulator Calibration | `data/results/simulator_calibration.json` | ~0.4s |
| **4** | Distribution Shift Suite | `data/results/distribution_shift_evaluation.json` | ~0.6s |
| **5** | Concurrency Scalability | `data/results/benchmarks/scalability_analysis.json` | ~0.4s |
| **6** | Privacy & DLP Probes | `data/results/privacy_evaluation.json` | ~0.2s |
| **7** | Model Drift Simulation | `data/results/drift_evaluation.json` | ~0.3s |
| **8** | Total Cost of Ownership | `data/results/tco_analysis.json` | ~0.2s |
| **9** | Caching Trace Analysis | `data/results/caching_workload_analysis.json` | ~0.4s |
| **10**| Measured Dual-Annotator Study | `data/results/human_study_measured.json` | ~0.4s |
| **11**| Simulated Multi-Annotator Study | `data/results/human_study_expanded.json` | ~0.5s |
| **12**| Public Manifest Export | `data/results/fixtures/sanitized_public_manifest.json` | ~0.3s |
| **13**| README Results Sync | Synchronized results in `README.md` | ~0.4s |
| **Total**| **Master Reproduction** | **All 13 evaluation stages verified** | **~12.5s** |

---

## 3. Cryptographic Artifact Checksums

To verify that the dataset and pricing configurations have not drifted from the published study:

```bash
python -c "from inferpath.evaluation.reproducibility import compute_file_sha256; print('Dataset:', compute_file_sha256('data/comprehensive_workload_dataset.json')); print('Pricing:', compute_file_sha256('configs/pricing_table_2026_10.yaml'))"
```

Expected SHA-256 prefixes:
- `data/comprehensive_workload_dataset.json`: `3f5393d4c5c9c632...`
- `configs/pricing_table_2026_10.yaml`: `29098642ea18e6f7...`

---

## 4. Adversarial Invariant Stress-Tests

External auditors can verify the core architectural guarantees:

### Invariant 1: Non-Leakage of Sensitive Requests
- **Test**: Submit a query containing private data (e.g., API key, SSN).
- **Guarantee**: External network calls are strictly blocked; prompt is not persisted in plain text; response is computed locally or safely refused.
- **Verification**: `pytest tests/unit/test_hardened_privacy_boundaries.py`

### Invariant 2: Zero-Cache Policy for Private Requests
- **Test**: Submit identical sensitive queries repeatedly.
- **Guarantee**: Cache stores zero keys; cache hit rate remains exactly 0.0%.
- **Verification**: `pytest tests/unit/test_cache_eligibility.py`

### Invariant 3: Zero-Leakage Dataset Split Integrity
- **Test**: Verify that no task family appears in both development and held-out test splits.
- **Guarantee**: Zero cross-split contamination.
- **Verification**: `pytest tests/unit/test_dataset_integrity.py`
