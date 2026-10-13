#!/usr/bin/env bash
# InferPath — 1-Command Reviewer Demo (Linux / macOS)
# Launches FastAPI backend in background, executes a test request, and opens the Streamlit dashboard.

set -euo pipefail

echo "======================================================================"
echo "  🧭 InferPath — Research Prototype 1-Command Demo"
echo "======================================================================"

# Determine Python binary
if [ -d ".venv" ] && [ -f ".venv/bin/python" ]; then
    PYTHON=".venv/bin/python"
    STREAMLIT=".venv/bin/streamlit"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON="python3"
    STREAMLIT="streamlit"
else
    echo "Error: Python 3.11+ is required but not found."
    exit 1
fi

echo "Using Python: $($PYTHON --version)"

# Check if port 8000 is already running
API_PID=""
if curl -s http://127.0.0.1:8000/v1/health >/dev/null 2>&1; then
    echo "✔ FastAPI backend is already running on http://127.0.0.1:8000"
else
    echo "Starting FastAPI backend in background (simulated/offline mode)..."
    ADAPTER_MODE=simulated $PYTHON -m uvicorn inferpath.api.main:app --host 127.0.0.1 --port 8000 >/dev/null 2>&1 &
    API_PID=$!

    # Wait for API to become ready
    echo "Waiting for API readiness at http://127.0.0.1:8000/v1/health..."
    for i in {1..30}; do
        if curl -s http://127.0.0.1:8000/v1/health >/dev/null 2>&1; then
            echo "✔ FastAPI backend ready."
            break
        fi
        sleep 0.5
    done
fi

cleanup() {
    if [ -n "$API_PID" ]; then
        echo "Shutting down background FastAPI backend (PID: $API_PID)..."
        kill "$API_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT INT TERM

echo ""
echo "Sending sample verification request to POST http://127.0.0.1:8000/v1/respond..."
RESPONSE=$(curl -s -X POST http://127.0.0.1:8000/v1/respond \
    -H "Content-Type: application/json" \
    -d '{"prompt": "Calculate: (15 * 8) + (240 / 6)", "privacy_class": "public"}')

echo "Response received:"
echo "$RESPONSE" | $PYTHON -m json.tool || echo "$RESPONSE"
echo ""

echo "Launching Streamlit Research Dashboard..."
echo "Press Ctrl+C to terminate the demo."
INFERPATH_API_URL="http://127.0.0.1:8000" $STREAMLIT run app/dashboard.py
