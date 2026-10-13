"""Unit tests for SQLite database manager and audit log persistence."""

import pytest

from inferpath.telemetry.database import DatabaseManager


@pytest.mark.asyncio
async def test_database_init_and_log_request(db_manager: DatabaseManager):
    """Verifies SQLite schema creation and request log persistence with privacy preservation."""
    sample_record = {
        "request_id": "test-req-001",
        "prompt_sha256": "abcdef1234567890abcdef1234567890",
        "prompt_sample": None,  # Private request: sample is strictly None
        "privacy_class": "confidential_private",
        "initial_tier": "deterministic_a",
        "final_tier": "deterministic_a",
        "was_cached": False,
        "fallback_occurred": False,
        "fallback_details": [],
        "input_tokens_value": 15,
        "input_tokens_provenance": "measured",
        "output_tokens_value": 25,
        "output_tokens_provenance": "measured",
        "latency_ms_value": 4.2,
        "latency_ms_provenance": "measured",
        "cost_usd_value": 0.0,
        "cost_usd_provenance": "calculated",
        "energy_wh_central": 0.000005,
        "energy_wh_low": 0.000001,
        "energy_wh_high": 0.00001,
        "energy_wh_provenance": "estimated",
        "water_ml_central": 0.000012,
        "water_ml_provenance": "estimated",
        "is_time_sensitive": False,
    }

    await db_manager.log_request(sample_record)

    logs = await db_manager.fetch_request_logs(limit=10)
    assert len(logs) == 1
    logged = logs[0]
    assert logged["request_id"] == "test-req-001"
    assert logged["privacy_class"] == "confidential_private"
    assert logged["prompt_sample"] is None
    assert logged["cost_usd_provenance"] == "calculated"
    assert logged["energy_wh_provenance"] == "estimated"
    assert logged["latency_ms_provenance"] == "measured"
