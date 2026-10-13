"""SQLite schema initialization and persistence manager."""

import json
import sqlite3
from pathlib import Path
from typing import Any

import aiosqlite

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS exact_cache (
    cache_key TEXT PRIMARY KEY,
    normalized_prompt TEXT NOT NULL,
    output_text TEXT NOT NULL,
    tier TEXT NOT NULL,
    response_payload TEXT NOT NULL,
    requirements_snapshot TEXT NOT NULL,
    version TEXT NOT NULL,
    validation_status TEXT NOT NULL,
    hit_count INTEGER DEFAULT 0,
    input_tokens INTEGER NOT NULL,
    output_tokens INTEGER NOT NULL,
    latency_ms REAL NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_exact_cache_expires ON exact_cache(expires_at);

CREATE TABLE IF NOT EXISTS request_logs (
    request_id TEXT PRIMARY KEY,
    prompt_sha256 TEXT NOT NULL,
    prompt_sample TEXT, -- Stored ONLY for 'public' privacy_class; strictly NULL for internal/private
    privacy_class TEXT NOT NULL,
    initial_tier TEXT NOT NULL,
    final_tier TEXT NOT NULL,
    was_cached BOOLEAN NOT NULL,
    fallback_occurred BOOLEAN NOT NULL,
    fallback_details TEXT, -- JSON array of FallbackRecord
    input_tokens_value INTEGER NOT NULL,
    input_tokens_provenance TEXT NOT NULL,
    output_tokens_value INTEGER NOT NULL,
    output_tokens_provenance TEXT NOT NULL,
    latency_ms_value REAL NOT NULL,
    latency_ms_provenance TEXT NOT NULL,
    cost_usd_value REAL NOT NULL,
    cost_usd_provenance TEXT NOT NULL,
    energy_wh_central REAL NOT NULL,
    energy_wh_low REAL NOT NULL,
    energy_wh_high REAL NOT NULL,
    energy_wh_provenance TEXT NOT NULL,
    water_ml_central REAL NOT NULL,
    water_ml_provenance TEXT NOT NULL,
    is_time_sensitive BOOLEAN NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_runs (
    run_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    dataset_name TEXT NOT NULL,
    dataset_hash TEXT NOT NULL,
    is_held_out BOOLEAN NOT NULL,
    is_simulated BOOLEAN NOT NULL, -- True if produced by mock adapters; NEVER merged with real benchmarks
    config_snapshot TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    result_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL REFERENCES evaluation_runs(run_id),
    prompt_id TEXT NOT NULL,
    chosen_tier TEXT NOT NULL,
    task_success BOOLEAN NOT NULL,
    quality_score REAL NOT NULL, -- Independent judge score, NEVER the router self-grading
    latency_ms REAL NOT NULL,
    cost_usd REAL NOT NULL,
    energy_wh REAL NOT NULL,
    provenance_record TEXT NOT NULL
);
"""


class DatabaseManager:
    """Manages SQLite connection lifecycle, schema migrations, and queries."""

    def __init__(self, db_path: str):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    def init_schema_sync(self) -> None:
        """Synchronously initializes database tables and indexes."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(SCHEMA_SQL)
            conn.commit()

    async def init_schema_async(self) -> None:
        """Asynchronously initializes database tables and indexes."""
        async with aiosqlite.connect(self.db_path) as db:
            await db.executescript(SCHEMA_SQL)
            await db.commit()

    async def log_request(self, record: dict[str, Any]) -> None:
        """Persists a sanitized request execution log entry."""
        sql = """
        INSERT INTO request_logs (
            request_id, prompt_sha256, prompt_sample, privacy_class,
            initial_tier, final_tier, was_cached, fallback_occurred, fallback_details,
            input_tokens_value, input_tokens_provenance,
            output_tokens_value, output_tokens_provenance,
            latency_ms_value, latency_ms_provenance,
            cost_usd_value, cost_usd_provenance,
            energy_wh_central, energy_wh_low, energy_wh_high, energy_wh_provenance,
            water_ml_central, water_ml_provenance,
            is_time_sensitive
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?,
            ?, ?, ?, ?,
            ?, ?,
            ?
        )
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                sql,
                (
                    record["request_id"],
                    record["prompt_sha256"],
                    record.get("prompt_sample"),
                    record["privacy_class"],
                    record["initial_tier"],
                    record["final_tier"],
                    int(record["was_cached"]),
                    int(record["fallback_occurred"]),
                    json.dumps(record.get("fallback_details", [])),
                    record["input_tokens_value"],
                    record["input_tokens_provenance"],
                    record["output_tokens_value"],
                    record["output_tokens_provenance"],
                    record["latency_ms_value"],
                    record["latency_ms_provenance"],
                    record["cost_usd_value"],
                    record["cost_usd_provenance"],
                    record["energy_wh_central"],
                    record["energy_wh_low"],
                    record["energy_wh_high"],
                    record["energy_wh_provenance"],
                    record["water_ml_central"],
                    record["water_ml_provenance"],
                    int(record["is_time_sensitive"]),
                ),
            )
            await db.commit()

    async def fetch_request_logs(self, limit: int = 50) -> list[dict[str, Any]]:
        """Retrieves audit execution logs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM request_logs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]
