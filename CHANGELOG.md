# Changelog

All notable changes to **InferPath** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
