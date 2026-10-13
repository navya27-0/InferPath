"""Unit tests for ExperimentLogger and JSONL audit logging."""

from pathlib import Path

from inferpath.models.enums import ProvenanceLabel
from inferpath.security.hashing import hash_prompt
from inferpath.telemetry.experiment_logger import ExperimentLogger, ExperimentRecord


def test_experiment_logger_writes_valid_jsonl(tmp_path: Path):
    """Verifies that ExperimentLogger creates structured valid JSONL records."""
    log_file = tmp_path / "experiments.jsonl"
    logger = ExperimentLogger(log_file)

    record = ExperimentRecord(
        run_id="run-exp-001",
        request_id="req-123",
        prompt_hash=hash_prompt("Calculate 12 * 12"),
        route="deterministic_a",
        rule_id="R01",
        cache_status="bypassed_deterministic",
        latency_ms=2.5,
        latency_provenance=ProvenanceLabel.MEASURED,
        input_tokens=10,
        input_tokens_provenance=ProvenanceLabel.MEASURED,
        output_tokens=5,
        output_tokens_provenance=ProvenanceLabel.MEASURED,
        cost_usd=0.0,
        cost_provenance=ProvenanceLabel.CALCULATED,
        fallback_events=[],
        metadata={"client": "eval_suite"},
    )

    logged = logger.log(record)
    assert logged["run_id"] == "run-exp-001"
    assert log_file.exists()

    records = logger.read_records()
    assert len(records) == 1
    assert records[0]["request_id"] == "req-123"
    assert records[0]["cost_provenance"] == "calculated"
    assert records[0]["latency_provenance"] == "measured"


def test_experiment_logger_drops_raw_prompt_for_hash(tmp_path: Path):
    """Verifies that dict inputs containing raw prompt have the raw text dropped and hashed."""
    log_file = tmp_path / "experiments.jsonl"
    logger = ExperimentLogger(log_file)

    secret_prompt = "Confidential business negotiation terms"
    raw_dict = {
        "run_id": "run-exp-002",
        "request_id": "req-456",
        "prompt": secret_prompt,  # Should be converted to hash and dropped
        "route": "mock-small",
        "rule_id": "R03",
        "latency_ms": 42.0,
        "latency_provenance": ProvenanceLabel.SIMULATED,
        "input_tokens": 12,
        "input_tokens_provenance": ProvenanceLabel.SIMULATED,
        "output_tokens": 8,
        "output_tokens_provenance": ProvenanceLabel.SIMULATED,
        "cost_usd": 0.000005,
        "cost_provenance": ProvenanceLabel.SIMULATED,
    }

    logger.log(raw_dict)

    with open(log_file, encoding="utf-8") as f:
        file_content = f.read()

    assert secret_prompt not in file_content
    records = logger.read_records()
    assert records[0]["prompt_hash"] == hash_prompt(secret_prompt)
    assert "prompt" not in records[0]


def test_experiment_logger_scrubs_secrets_in_errors_and_metadata(tmp_path: Path):
    """Verifies that credentials in error messages and metadata are scrubbed."""
    log_file = tmp_path / "experiments.jsonl"
    logger = ExperimentLogger(log_file)

    raw_dict = {
        "run_id": "run-exp-003",
        "request_id": "req-789",
        "prompt_hash": "hash123",
        "route": "mock-large",
        "latency_ms": 110.0,
        "latency_provenance": ProvenanceLabel.SIMULATED,
        "input_tokens": 20,
        "input_tokens_provenance": ProvenanceLabel.SIMULATED,
        "output_tokens": 30,
        "output_tokens_provenance": ProvenanceLabel.SIMULATED,
        "cost_usd": 0.0003,
        "cost_provenance": ProvenanceLabel.SIMULATED,
        "error_message": "Failed with token sk-or-v1-0123456789abcdef0123456789abcdef",
        "metadata": {"auth_header": "Bearer sk-1234567890abcdef1234567890abcdef"},
    }

    logger.log(raw_dict)
    records = logger.read_records()

    assert "sk-or-v1" not in records[0]["error_message"]
    assert "[REDACTED_API_KEY]" in records[0]["error_message"]
    assert "sk-12345" not in records[0]["metadata"]["auth_header"]
    assert "[REDACTED_API_KEY]" in records[0]["metadata"]["auth_header"]
