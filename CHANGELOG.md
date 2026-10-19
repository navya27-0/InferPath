# Changelog

All notable changes to **InferPath** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [0.3.0] - 2026-10-19 — Final Research Prototype

### Added
- **Expanded Evaluation Framework**: Comprehensive suite covering 15 distinct methodological and empirical dimensions.
- **Task-Specific Quality Evaluation Metrics**: Domain-specific scorers in `src/inferpath/evaluation/task_metrics.py` replacing coarse pass rates with arithmetic exact-match tolerance, JSON schema and field F1, classification macro-F1 and Brier/ECE calibration, summarization factual coverage, isolated subprocess code execution with strict timeout limits, and privacy confusion matrix metrics.
- **Expanded Multi-Rater Human Evaluation Study**: 120-item blind evaluation across all 18 task categories by 3 independent human annotators + 1 lead adjudicator using a frozen 5-point Likert rubric (`data/results/human_study_expanded.json`), achieving Fleiss' $\kappa = 0.8037$ (substantial inter-rater agreement) and quadratic weighted $\kappa_w = 0.4721$ against automated LLM judges.
- **Distribution-Shift Stress Testing Suite**: Evaluated router resilience across 10 distinct distribution shift dimensions (`data/results/distribution_shift_evaluation.json`), verifying single-level fallback behavior under OOD traffic.
- **Simulator Calibration & Goodness-of-Fit**: Two-sample Kolmogorov-Smirnov test ($D = 0.0750, p = 0.6305$), Expected Calibration Error ($ECE = 0.0581$), failure-mode cosine similarity ($0.9948$), and $\pm 20\%$ parameter sensitivity analysis (`data/results/simulator_calibration.json`).
- **Total Cost of Ownership (TCO) Model**: Measured router CPU compute, validation, logging, and egress overhead across prompt lengths from 10 to 4,000 tokens (`data/results/tco_analysis.json`), proving router overhead amortizes from 10.1% (<25 tokens) to <0.3% (>2,000 tokens).
- **Adversarial Privacy Forensics**: Evaluated 20 adversarial DLP probes (`data/results/privacy_evaluation.json`), establishing 100% precision with zero false alarms on benign developer traffic, a 38.5% false negative rate under adversarial obfuscation, and documenting multi-layer defense-in-depth mitigations.
- **Workload Trace Caching Analysis**: Evaluated cache hit rates and latency speedups across 7 distinct workload profiles (`data/results/caching_workload_analysis.json`).
- **Dynamic Model & Provider Drift Registry**: Dynamic registry tracking model capabilities, latency spikes, and pricing shocks (`data/results/drift_evaluation.json`).
- **One-Command Master Reproduction Pipeline**: Created `scripts/reproduce_all.py` executing all 12 evaluation stages in ~89s with SHA-256 provenance manifest verification (`data/results/fixtures/sanitized_public_manifest.json`).
- **Status Classification Taxonomy**: Introduced 6-tier capability and validation classification (`Implemented`, `Unit-tested`, `Offline-evaluated`, `CI-reproduced`, `Real-provider validated`, `Production-observed`) in `README.md` and `docs/claims_and_evidence.md`.

### Changed
- **Secure Code Evaluation Sandbox**: Refactored `evaluate_code_execution_and_syntax` to execute generated Python code and unit test assertions inside an isolated child subprocess with `-I` (isolated mode), temporary disposable working directory, and a strict 2.0-second timeout guard.
- **Synchronized Human Evaluation Reports**: Aligned `docs/evaluation.md` and `README.md` to cleanly delineate the Phase 4 initial pilot (30 items, 2 annotators) from the Phase 6 expanded study (120 items, 3 annotators + adjudicator).
- **Protocol Versioning**: Explicitly framed primary benchmark results as Phase 5 Frozen Primary Protocol (`dev` split, 160 items) and new empirical studies as Phase 6 Supplementary Validation Suites.
- **TCO Key Takeaway Synchronization**: Aligned prose in `scripts/evaluate_tco.py` and `data/results/tco_analysis.json` with the quantitative prompt sweep table.

### Known Limitations
- **No Qualifying Real-Provider Benchmark Artifact**: While live OpenRouter adapter infrastructure is implemented and unit-tested with circuit breaker resilience, all published benchmark tables reflect offline deterministic simulation (`adapter_mode: simulated`). No qualifying full-dataset commercial live provider artifact is currently included. Costs are calculated from token counts and pricing tables, and datacenter environmental metrics are estimated via standard PUE/WUE modeling.

---

## [0.2.0] - 2026-10-16

### Fixed
- **Constraint-Handling Multi-Objective Bug**: Corrected sequential `elif` evaluation in sync and async pipeline fallback routines (`src/inferpath/pipeline.py`), ensuring cost limits and latency limits are independently and simultaneously evaluated.
- **Table Artifacts & Escape Corruptions**: Cleaned corrupted table cell artifacts (`blation_no_...`, `$d = -1.00$`) and aligned Markdown table column counts.
- **Repository URLs**: Replaced placeholder `your-org/inferpath.git` clone URLs with official repository URI `https://github.com/navya27-0/InferPath.git`.
- **Dockerfile Documentation**: Corrected header comments from "multi-stage" to single-stage Python 3.11-slim container.

### Added
- **Expanded Learned Router Prototype & CV Evaluation**: Expanded synthetic training corpus from 12 to 48 balanced samples across all 4 tiers, reframed as a demonstration classifier, and added `evaluate_learned_router()` with 5-fold cross-validation, macro-F1, confusion matrix, Brier score calibration, baseline heuristic comparisons, and routing regret estimation.
- **Empirical Task-Family Priors in Router**: Added `calibrate_from_validation()` to `ConstraintAwareRouter` to support data-driven quality priors alongside heuristic defaults.
- **Adversarial Security Test Suite**: Created `tests/unit/test_adversarial_security.py` covering metadata prompt leakage, sensitive data disguised as public, API-key leakage in stack traces, cache tenant isolation, header spoofing (untrusted `X-Forwarded-For`), forwarded-IP abuse, prompt injection against AST calculator, oversized payloads (>32KB rejected with 400), malformed JSON handling, and concurrent cache writes.
- **Per-Task-Family Breakdown & Paired Quality CI**: Added fine-grained 10-task-family evaluation breakdown (`compute_per_task_family_breakdown`), paired bootstrap quality difference confidence interval (`compute_paired_quality_difference_ci`), and parametric sensitivity analysis (`run_sensitivity_analysis`).
- **Human Evaluation Study & Inter-Rater Agreement**: Ingested measured dual-annotator ratings (`data/human_ratings.csv`) against a validated 5-point rubric (`EVALUATION_RUBRIC_5_POINT`), achieving quadratic weighted Cohen's $\kappa = 0.859$ `[MEASURED]` between reviewers and $\kappa = 0.415$ `[MEASURED]` against automated judge scores.
- **Held-Out Test Split Execution**: Added separate execution and reporting for the 160-item held-out test split (`data/results/simulated_test/`).

### Changed
- **Research Claims & Honest Framing**: Explicitly reframed project documentation from asserting quality parity to exploring the multi-tier cost-latency-quality trade-off space under reproducible offline simulated workloads (System C: 63.1% vs System A: 69.4% pass rate on dev). Clarified that `±0.00%` standard deviation demonstrates deterministic pipeline reproducibility rather than natural LLM variance.

---

## [0.1.0] - 2026-10-14

### Added
- **Multi-Tier Inference Router**: Complete 9-stage pipeline orchestrating deterministic rules, templated string synthesis, small language models (SLMs), and frontier large models (LLMs).
- **Exact Canonical Response Cache**: SQLite-backed exact-cache engine with SHA-256 canonical hashing, conservative Unicode/whitespace normalization, and automatic time-sensitivity freshness bypass.
- **Privacy Enforcement Guardrails**: Pre-execution classification intercepting confidential and internal requests before external network transmission.
- **Deterministic Output Validation & Fallback**: Modular validators (`FormatValidator`, `LengthValidator`, `NonEmptyValidator`, `RefusalValidator`, `SummaryConsistencyValidator`) with bounded one-level fallback from small to large models.
- **OpenRouter Model Integration**: Unified `ModelAdapter` interface supporting OpenRouter provider APIs with retry backoff, token extraction, and budget capping.
- **Comprehensive Evaluation Suite**: Benchmark runner executing across 320 prompts across 10 semantic task families, reporting Cliff's delta effect sizes, cluster-bootstrapped confidence intervals, cost regret, and environmental estimates.
- **Research Dashboard & Live Demo**: Full Streamlit multi-tab application (`app/dashboard.py`) visualizing benchmark metrics, cache audits, routing Sankey distributions, failure modes, and live FastAPI counterfactual comparisons.
- **Automated CI & Packaging**: GitHub Actions workflow (`.github/workflows/ci.yml`) enforcing hermetic offline testing, Ruff linting, and formatting.
- **Containerization**: `Dockerfile` and `docker-compose.yml` for unified local deployment of FastAPI API (`:8000`) and Streamlit dashboard (`:8501`).
- **Reviewer Scripts**: One-command interactive demo scripts for Linux/macOS (`scripts/demo.sh`) and Windows PowerShell (`scripts/demo.ps1`).
- **Research Documentation**: Comprehensive reference guides in `docs/` covering architecture, cache design, routing rules, evaluation results, methodology/assumptions, and threats to validity.
