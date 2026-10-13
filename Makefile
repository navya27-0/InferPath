# InferPath Development Makefile

PYTHON := .venv/Scripts/python.exe
PYTEST := .venv/Scripts/pytest.exe
RUFF := .venv/Scripts/ruff.exe

.PHONY: help test lint format run eval

help:
	@echo "Available commands:"
	@echo "  make test   - Run default test suite (offline with network isolation)"
	@echo "  make lint   - Run Ruff linter"
	@echo "  make format - Format codebase with Ruff"
	@echo "  make run    - Execute mock model generation demo"
	@echo "  make eval   - Verify mock models ready for offline evaluation harness"

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

run:
	$(PYTHON) -c "from inferpath.handlers.mock_adapter import MockSmallModel; res = MockSmallModel().generate('Convert 25 Celsius to Fahrenheit'); print(f'InferPath Demo Run:\nPrompt: Convert 25 Celsius to Fahrenheit\nResult: {res.text}\nTokens: in={res.input_tokens}, out={res.output_tokens}\nLatency: {res.latency_ms}ms\nCost: $${res.cost_usd_calculated:.6f}\nSource: {res.source}')"

eval:
	$(PYTHON) -c "from inferpath.handlers.mock_adapter import MockSmallModel, MockLargeModel; s = MockSmallModel(); l = MockLargeModel(); print(f'InferPath Evaluation Pre-flight:\n- Mock Small Model: {s.model_id} (Price table: {s.price_table_date})\n- Mock Large Model: {l.model_id} (Price table: {l.price_table_date})\nReady for Phase 5 benchmark suite.')"
