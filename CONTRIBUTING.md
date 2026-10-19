# Contributing to InferPath

Thank you for your interest in contributing to **InferPath**, an empirical research prototype for resource-aware AI inference routing.

---

## 1. Non-Negotiable Research Invariants

Before writing code or running benchmarks, please note our core research constraints:

1. **Zero Metric Fabrication**: Never hardcode, interpolate, or fabricate experimental observations. Every metric displayed or documented must originate from an actual logged run artifact in `data/results/`.
2. **Strict Provenance Labeling**: Every quantitative metric must be labeled with its exact provenance class:
   - `MEASURED`: Directly observed during runtime (tokens, latency ms, cache hits).
   - `CALCULATED`: Deterministically computed from measured values + dated configuration (USD cost).
   - `ESTIMATED`: Derived from heuristic external models (Scope 2 energy Wh, cooling water mL).
   - `SIMULATED`: Synthesized in hermetic offline test environments.
3. **No Environmental Greenwashing**: Datacenter energy and cooling water figures are modeled estimates (`[ESTIMATED]`). They must **never** be presented as physical measurements, headline claims, or marketing badges.
4. **Hermetic Test Suite**: The default test suite (`pytest -v`) must run entirely offline without network access or live API credentials (`OPENROUTER_API_KEY`). Live network tests must be marked `@pytest.mark.live`.

---

## 2. Development Setup

### Prerequisites
- Python 3.11+
- Git

### Initializing the Environment

#### Option A: Using `uv` (Recommended)
```bash
# Clone the repository
git clone https://github.com/navya27-0/InferPath.git
cd InferPath

# Install all dependencies and development extras
uv sync --all-extras

# Run tests
uv run pytest -q -m "not live"
```

#### Option B: Using standard `venv` and `pip`
```bash
# Clone the repository
git clone https://github.com/navya27-0/InferPath.git
cd InferPath

# Create virtual environment
python -m venv .venv

# Activate on Linux/macOS:
source .venv/bin/activate
# Activate on Windows:
.venv\Scripts\Activate.ps1

# Install package in editable mode with development dependencies
pip install -e ".[dev]"
```

### Deployment Profiles and Runtime Services
InferPath supports tiered deployment profiles configured via `INFERPATH_DEPLOYMENT_PROFILE`:
- `development`: In-memory rate limiting (120 RPM), local bind (`127.0.0.1`), relaxed auth.
- `staging`: Strict authentication, distributed rate limiting via Redis if configured.
- `production`: Mandatory API key authentication, strict IP extraction, optional distributed rate limiting (`INFERPATH_REDIS_URL`).
- `evaluation`: Zero network I/O, mock model adapters with deterministic outputs.

To run with distributed rate limiting across horizontal workers, specify:
```bash
export INFERPATH_REDIS_URL="redis://localhost:6379/0"
```

---

## 3. Code Standards and Tooling

We enforce strict formatting and linting rules using **Ruff**:

```bash
# Check linting rules
ruff check .

# Automatically format code
ruff format .

# Run hermetic test suite
pytest -q -m "not live"
```

- **Line Length**: 100 characters.
- **Python Version**: Target 3.11 (`py311`).
- **Typing**: Use static type annotations (`pydantic` models for structured data).

---

## 4. Submitting Changes

1. Create a descriptive feature branch:
   ```bash
   git checkout -b feature/my-enhancement
   ```
2. Verify all tests and linters pass:
   ```bash
   ruff check .
   ruff format --check .
   pytest -v
   ```
3. If benchmark results or README tables change, run:
   ```bash
   python scripts/update_readme_results.py
   ```
4. Open a Pull Request against `main`. All PRs must pass the GitHub Actions CI workflow (`.github/workflows/ci.yml`).
