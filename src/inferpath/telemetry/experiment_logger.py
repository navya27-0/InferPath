"""Structured JSONL experiment logger for empirical research runs."""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from inferpath.models.enums import ProvenanceLabel
from inferpath.security.hashing import hash_prompt, scrub_api_keys


class ExperimentRecord(BaseModel):
    """Structured record schema for each entry in data/logs/experiments.jsonl."""

    run_id: str
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())
    prompt_hash: str
    route: str
    rule_id: str | None = None
    cache_status: str = "miss"  # "hit", "miss", "bypassed_private", "bypassed_time_sensitive", "bypassed_deterministic"
    latency_ms: float
    latency_provenance: ProvenanceLabel
    input_tokens: int
    input_tokens_provenance: ProvenanceLabel
    output_tokens: int
    output_tokens_provenance: ProvenanceLabel
    cost_usd: float
    cost_provenance: ProvenanceLabel
    energy_wh_estimated: dict[str, float] | None = None
    energy_provenance: ProvenanceLabel | None = None
    water_ml_estimated: dict[str, float] | None = None
    water_provenance: ProvenanceLabel | None = None
    fallback_events: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None


class ExperimentLogger:
    """
    Manages writing and reading structured JSONL experiment run records.
    Strictly prevents API keys and raw prompt text from leaking into files.
    """

    def __init__(self, log_path: str | Path = "data/logs/experiments.jsonl"):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, record: ExperimentRecord | dict[str, Any]) -> dict[str, Any]:
        """Appends a sanitized experiment record as a JSON line."""
        if isinstance(record, dict):
            # If raw prompt is provided, convert to prompt_hash and drop raw prompt
            raw_prompt = record.pop("prompt", None) or record.pop("prompt_text", None)
            if raw_prompt and "prompt_hash" not in record:
                record["prompt_hash"] = hash_prompt(str(raw_prompt))
            elif "prompt_hash" not in record:
                record["prompt_hash"] = "unknown"

            if "timestamp" not in record:
                record["timestamp"] = datetime.now(UTC).isoformat()

            parsed = ExperimentRecord(**record)
        else:
            parsed = record

        record_dict = parsed.model_dump()

        # Sanitize any errors or metadata values of API keys
        if record_dict.get("error_message"):
            record_dict["error_message"] = scrub_api_keys(record_dict["error_message"])

        cleaned_meta: dict[str, Any] = {}
        for k, v in record_dict.get("metadata", {}).items():
            cleaned_meta[k] = scrub_api_keys(str(v))
        record_dict["metadata"] = cleaned_meta

        # Write line to JSONL file
        line = json.dumps(record_dict, ensure_ascii=False)
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line + "\n")

        return record_dict

    def log_event(self, event_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Appends an arbitrary structured telemetry event to JSONL, sanitizing secrets."""
        event_dict = {
            "event": event_name,
            "timestamp": datetime.now(UTC).isoformat(),
            **payload,
        }
        cleaned: dict[str, Any] = {}
        for k, v in event_dict.items():
            if isinstance(v, str):
                cleaned[k] = scrub_api_keys(v)
            else:
                cleaned[k] = v
        line = json.dumps(cleaned, ensure_ascii=False)
        with open(self.log_path, mode="a", encoding="utf-8") as f:
            f.write(line + "\n")
        return cleaned

    def read_records(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Reads and parses records from the JSONL log file."""
        if not self.log_path.exists():
            return []

        records: list[dict[str, Any]] = []
        with open(self.log_path, encoding="utf-8") as f:
            for line in f:
                stripped = line.strip()
                if stripped:
                    records.append(json.loads(stripped))
                if limit and len(records) >= limit:
                    break
        return records
