# Changelog

All notable changes to **InferPath** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
