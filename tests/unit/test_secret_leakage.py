"""Security test: confirms secrets and API keys are strictly excluded from logs, traces, and DB."""

import httpx
import pytest

from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import PricingTable, Settings
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter, OpenRouterProviderError
from inferpath.models.domain import UserRequest
from inferpath.pipeline import RequestPipeline
from inferpath.telemetry.database import DatabaseManager
from inferpath.telemetry.experiment_logger import ExperimentLogger

FAKE_CANARY_SECRET = "sk-or-v1-canary-secret-token-777888999"


def test_openrouter_adapter_does_not_leak_secret_in_errors(tmp_path):
    """Confirm Authorization headers and API keys never leak in exception messages or logs."""
    pricing = PricingTable(
        effective_date="2026-10-01",
        source="Test",
        rates_per_1m={"google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30}},
    )

    # Provider returns 500 error mentioning headers
    def error_handler(request: httpx.Request) -> httpx.Response:
        # Provider response body simulating an echo error
        return httpx.Response(500, text=f"Error processing request with token {FAKE_CANARY_SECRET}")

    transport = httpx.MockTransport(error_handler)
    client = httpx.Client(transport=transport)

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=pricing,
        api_key=FAKE_CANARY_SECRET,
        client=client,
        max_retries=0,
    )

    with pytest.raises(OpenRouterProviderError) as exc_info:
        adapter.generate("Test prompt")

    # Exception string must be sanitized
    error_msg = str(exc_info.value)
    assert FAKE_CANARY_SECRET not in error_msg
    assert "[REDACTED_API_KEY]" in error_msg


def test_pipeline_logs_and_db_zero_secret_leakage(tmp_path):
    """End-to-end pipeline run verifies canary secret never appears in JSONL logs or SQLite records."""
    db_path = str(tmp_path / "secret_check.db")
    log_path = str(tmp_path / "experiments.jsonl")

    settings = Settings()
    settings.database_path = db_path

    pricing = PricingTable(
        effective_date="2026-10-01",
        source="Test",
        rates_per_1m={"google/gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30}},
    )

    logger = ExperimentLogger(log_path=log_path)
    db = DatabaseManager(db_path=db_path)
    db.init_sync()

    # Create adapter with canary secret
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"role": "assistant", "content": "Valid response"}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5},
            },
        )

    adapter = OpenRouterAdapter(
        model_id="google/gemini-2.5-flash",
        pricing_table=pricing,
        api_key=FAKE_CANARY_SECRET,
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )

    pipeline = RequestPipeline(
        settings=settings,
        pricing_table=pricing,
        cache=ExactCache(db_path=db_path),
        small_adapter=adapter,
        logger=logger,
    )

    req = UserRequest(
        request_id="sec_test_1",
        prompt="Sample request with secret sk-ant-secret-999 in prompt",
    )
    result = pipeline.execute(req)

    # 1. Inspect route trace
    trace_str = str(result.route_trace)
    assert FAKE_CANARY_SECRET not in trace_str

    # 2. Inspect generated JSONL log file
    with open(log_path, encoding="utf-8") as f:
        log_content = f.read()
    assert FAKE_CANARY_SECRET not in log_content

    # 3. Inspect SQLite database contents
    with db.get_connection() as conn:
        cursor = conn.cursor()
        for table in ["request_logs", "exact_cache"]:
            cursor.execute(f"SELECT * FROM {table}")
            rows = cursor.fetchall()
            for row in rows:
                row_str = str(row)
                assert FAKE_CANARY_SECRET not in row_str
