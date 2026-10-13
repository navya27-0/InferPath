# InferPath Development Makefile
ifeq ($(OS),Windows_NT)
    PYTHON := .venv/Scripts/python.exe
    PYTEST := .venv/Scripts/pytest.exe
    RUFF := .venv/Scripts/ruff.exe
    STREAMLIT := .venv/Scripts/streamlit.exe
else
    PYTHON := .venv/bin/python
    PYTEST := .venv/bin/pytest
    RUFF := .venv/bin/ruff
    STREAMLIT := .venv/bin/streamlit
endif

.PHONY: help test lint format run eval run-dashboard

help:
	@echo "Available commands:"
	@echo "  make test          - Run default test suite (offline with network isolation)"
	@echo "  make lint          - Run Ruff linter"
	@echo "  make format        - Format codebase with Ruff"
	@echo "  make run           - Execute mock model generation demo"
	@echo "  make eval          - Execute offline research evaluation suite (simulated dev split, 3 repeats)"
	@echo "  make run-dashboard - Launch Streamlit research dashboard"

test:
	$(PYTEST) -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .

run:
	$(PYTHON) -c "from inferpath.handlers.mock_adapter import MockSmallModel; res = MockSmallModel().generate('Convert 25 Celsius to Fahrenheit'); print(f'InferPath Demo Run:\nPrompt: Convert 25 Celsius to Fahrenheit\nResult: {res.text}\nTokens: in={res.input_tokens}, out={res.output_tokens}\nLatency: {res.latency_ms}ms\nCost: $${res.cost_usd_calculated:.6f}\nSource: {res.source}')"

eval:
	$(PYTHON) -m inferpath.evaluation.runner --mode simulated --split dev --repeats 3

run-dashboard:
	$(STREAMLIT) run app/dashboard.py
