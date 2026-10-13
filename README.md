# InferPath: Resource-Aware AI Inference Router

InferPath is an empirical research prototype designed for rigorous evaluation of a multi-tier inference router. For every incoming query, it selects the least resource-intensive processing tier that meets correctness, quality, privacy, and latency needs.

## Architecture

1. **Exact Cache**: Normalized composite cache key lookup (bypassed for deterministic, private, and time-sensitive requests).
2. **Routing Tiers**:
   - **Tier A**: Deterministic rules / safe APIs (exact math, unit conversion, string transformations).
   - **Tier B**: Templates with slot filling. Missing fields trigger structured clarification, never hallucination.
   - **Tier C**: Small model (via OpenRouter OpenAI-compatible API).
   - **Tier D**: Large model (via OpenRouter OpenAI-compatible API).
3. **Validation & Escalation**: Independent output validation with automatic fallback (A $\to$ C, B $\to$ C, C $\to$ D).
4. **Telemetry & Provenance**: Every metric value is strictly tagged as `measured`, `calculated`, `estimated`, or `simulated`.

## Setup

```bash
# Setup virtual environment with Python 3.11
uv venv --python 3.11
.venv\Scripts\activate

# Install dependencies
uv pip install -e ".[dev]"

# Configure environment
copy .env.example .env
# Edit .env and supply OPENROUTER_API_KEY
```

## Running Tests

```bash
pytest -v
```
