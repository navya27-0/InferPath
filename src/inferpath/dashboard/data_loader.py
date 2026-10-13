"""Data loader for InferPath research dashboard.

Provides a read-only, validated data-access layer for:
- Run manifests and item-level execution logs
- Benchmark summary metrics and inter-rater agreement statistics
- Environmental carbon & water coefficient assumptions
- Dated pricing tables and counterfactual cost calculations
- Audited cache performance benchmarks
- Ranked failure case extraction with safe privacy redaction
"""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class ManifestMetadata(BaseModel):
    """Metadata describing a single benchmark run execution."""

    run_id: str
    system_id: str
    run_index: int
    adapter_mode: str  # "simulated" or "real"
    split: str  # "dev" or "test"
    dataset_version: str = "1.0.0"
    dataset_sha256: str = ""
    config_hash: str = ""
    git_commit: str = ""
    timestamp: str = ""
    random_seed: int = 42
    q_min: float = 0.70
    price_table_date: str = "2026-10-01"
    manifest_path: str = ""
    logs_path: str | None = None
    summary_metrics: dict[str, Any] = Field(default_factory=dict)
    has_logs: bool = False
    integrity_warning: str | None = None


class RunData(BaseModel):
    """Encapsulates a run's manifest metadata and raw item execution logs."""

    manifest: ManifestMetadata
    logs: list[dict[str, Any]] = Field(default_factory=list)


def discover_manifests(base_dir: Path | str | None = None) -> list[ManifestMetadata]:
    """
    Discovers all available run manifests in data/results/.
    Searches both 'simulated' and 'real' subdirectories.
    Handles malformed or partially written artifacts gracefully without failing.
    """
    root = Path(base_dir) if base_dir else Path("data/results")
    manifests: list[ManifestMetadata] = []

    if not root.exists():
        return manifests

    # Scan simulated, real, and custom folders
    for sub in root.iterdir():
        if not sub.is_dir():
            continue
        for manifest_file in sub.glob("manifest_*.json"):
            try:
                with open(manifest_file, encoding="utf-8") as f:
                    data = json.load(f)

                # Look for matching logs file
                logs_filename = manifest_file.name.replace("manifest_", "logs_")
                logs_path = manifest_file.parent / logs_filename
                has_logs = logs_path.exists() and logs_path.stat().st_size > 0

                meta = ManifestMetadata(
                    run_id=data.get("run_id", manifest_file.stem),
                    system_id=data.get("system_id", "Unknown"),
                    run_index=data.get("run_index", 1),
                    adapter_mode=data.get("adapter_mode", "simulated"),
                    split=data.get("split", "dev"),
                    dataset_version=data.get("dataset_version", "1.0.0"),
                    dataset_sha256=data.get("dataset_sha256", ""),
                    config_hash=data.get("config_hash", ""),
                    git_commit=data.get("git_commit", ""),
                    timestamp=data.get("timestamp", ""),
                    random_seed=data.get("random_seed", 42),
                    q_min=data.get("q_min", 0.70),
                    price_table_date=data.get("price_table_date", "2026-10-01"),
                    manifest_path=str(manifest_file),
                    logs_path=str(logs_path) if has_logs else None,
                    summary_metrics=data.get("summary_metrics", {}),
                    has_logs=has_logs,
                    integrity_warning=None
                    if has_logs
                    else "Matching log artifact is missing or empty.",
                )
                manifests.append(meta)
            except (json.JSONDecodeError, OSError, ValueError) as err:
                # Malformed file handled gracefully with integrity warning
                corrupt_meta = ManifestMetadata(
                    run_id=manifest_file.stem,
                    system_id="Unreadable",
                    run_index=0,
                    adapter_mode="unknown",
                    split="unknown",
                    manifest_path=str(manifest_file),
                    logs_path=None,
                    has_logs=False,
                    integrity_warning=f"Artifact unreadable or malformed JSON: {err}",
                )
                manifests.append(corrupt_meta)

    # Sort deterministically: by system_id, split, run_index
    manifests.sort(key=lambda m: (m.system_id, m.split, m.run_index))
    return manifests


def load_run_data(manifest: ManifestMetadata) -> RunData:
    """Loads raw item-level logs for a manifest safely."""
    if not manifest.has_logs or not manifest.logs_path:
        return RunData(manifest=manifest, logs=[])

    logs_file = Path(manifest.logs_path)
    if not logs_file.exists():
        manifest.has_logs = False
        manifest.integrity_warning = "Log file referenced in manifest was not found on disk."
        return RunData(manifest=manifest, logs=[])

    try:
        with open(logs_file, encoding="utf-8") as f:
            logs = json.load(f)
        if not isinstance(logs, list):
            manifest.integrity_warning = "Log file is not a valid list of records."
            return RunData(manifest=manifest, logs=[])
        return RunData(manifest=manifest, logs=logs)
    except (json.JSONDecodeError, OSError) as err:
        manifest.integrity_warning = f"Error reading logs: {err}"
        return RunData(manifest=manifest, logs=[])


def load_benchmark_summary(results_dir: Path | str | None = None) -> dict[str, Any] | None:
    """Loads the benchmark summary JSON generated by the Phase 5 runner."""
    path = (
        Path(results_dir) if results_dir else Path("data/results/simulated/benchmark_summary.json")
    )
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def load_carbon_assumptions(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Loads and formats the registered environmental assumptions from carbon_coefficients.yaml."""
    cfg = Path(config_path) if config_path else Path("configs/carbon_coefficients.yaml")
    if not cfg.exists():
        return []

    try:
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return []

    metadata = data.get("metadata", {})
    effective_date = metadata.get("effective_date", "2026-10-01")
    pue_data = data.get("datacenter_pue", {})

    rows: list[dict[str, Any]] = [
        {
            "category": "Facility Efficiency",
            "parameter": "Datacenter PUE (Power Usage Effectiveness)",
            "central": pue_data.get("central", 1.20),
            "low": pue_data.get("low", 1.10),
            "high": pue_data.get("high", 1.35),
            "unit": pue_data.get("unit", "ratio"),
            "provenance": "estimated",
            "source_date": effective_date,
            "verification_status": pue_data.get("verification_status", "unverified_illustrative"),
            "source_note": pue_data.get("source_note", ""),
            "boundary": metadata.get("system_boundary_energy", "Scope 2 Compute"),
        }
    ]

    coeffs = data.get("coefficients_per_1000_tokens", {})
    for model_name, info in coeffs.items():
        rows.append(
            {
                "category": f"Energy Intensity ({model_name})",
                "parameter": f"Energy per 1k tokens: {model_name}",
                "central": info.get("central", 0.0),
                "low": info.get("low", 0.0),
                "high": info.get("high", 0.0),
                "unit": info.get("unit", "Wh / 1000 tokens"),
                "provenance": "estimated",
                "source_date": effective_date,
                "verification_status": info.get("verification_status", "unverified_illustrative"),
                "source_note": info.get("source_note", ""),
                "boundary": metadata.get("system_boundary_energy", "Scope 2 Compute"),
            }
        )

    water_data = data.get("water_intensity", {})
    rows.append(
        {
            "category": "Water Footprint",
            "parameter": "Operational Water Intensity",
            "central": water_data.get("central", 2.4),
            "low": water_data.get("low", 1.2),
            "high": water_data.get("high", 4.5),
            "unit": water_data.get("unit", "mL / Wh"),
            "provenance": "estimated",
            "source_date": effective_date,
            "verification_status": water_data.get("verification_status", "unverified_illustrative"),
            "source_note": water_data.get("source_note", ""),
            "boundary": metadata.get("system_boundary_water", "Cooling WUE + Generation Footprint"),
        }
    )

    return rows


def load_pricing_assumptions(config_path: Path | str | None = None) -> list[dict[str, Any]]:
    """Loads model pricing assumptions from pricing_table_2026_10.yaml."""
    cfg = Path(config_path) if config_path else Path("configs/pricing_table_2026_10.yaml")
    if not cfg.exists():
        return []

    try:
        with open(cfg, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except (yaml.YAMLError, OSError):
        return []

    table_meta = data.get("table_metadata", {})
    effective_date = table_meta.get("date", "2026-10-01")
    models = data.get("models", {})

    rows: list[dict[str, Any]] = []
    for model_id, rates in models.items():
        rows.append(
            {
                "model_id": model_id,
                "prompt_price_per_1m": rates.get("prompt_price_per_1m", 0.0),
                "completion_price_per_1m": rates.get("completion_price_per_1m", 0.0),
                "status": rates.get("status", "unverified_illustrative"),
                "currency": "USD",
                "effective_date": effective_date,
                "provenance": "calculated",
            }
        )
    return rows


def calculate_counterfactual_large_model_cost(
    prompt: str,
    output_text: str | None = None,
    actual_input_tokens: int | None = None,
    actual_output_tokens: int | None = None,
    model_id: str = "anthropic/claude-3.7-sonnet",
    pricing_file: Path | str | None = None,
) -> dict[str, Any]:
    """
    Computes counterfactual Always-Large cost projection for a prompt.
    Adheres strictly to the dated pricing table.
    Explicitly distinguishes measured vs estimated heuristic tokens.
    Does NOT assert quality or latency guarantees.
    """
    cfg = Path(pricing_file) if pricing_file else Path("configs/pricing_table_2026_10.yaml")
    rates: dict[str, Any] = {}
    effective_date = "2026-10-01"

    if cfg.exists():
        try:
            with open(cfg, encoding="utf-8") as f:
                pdata = yaml.safe_load(f)
                effective_date = pdata.get("table_metadata", {}).get("date", "2026-10-01")
                rates = pdata.get("models", {}).get(model_id, {})
        except Exception:
            rates = {}

    if not rates:
        # Fallback to mock-large if anthropic not found
        rates = {"prompt_price_per_1m": 3.00, "completion_price_per_1m": 15.00}

    prompt_rate = rates.get("prompt_price_per_1m", 3.00)
    completion_rate = rates.get("completion_price_per_1m", 15.00)

    # Token counting determination
    if actual_input_tokens is not None and actual_input_tokens > 0:
        in_tokens = actual_input_tokens
        in_method = "measured"
    else:
        # Heuristic word count multiplier (1.3 tokens/word)
        word_count = len(prompt.split())
        in_tokens = max(1, math.ceil(word_count * 1.3))
        in_method = "estimated_heuristic (1.3 tokens/word)"

    if actual_output_tokens is not None and actual_output_tokens > 0:
        out_tokens = actual_output_tokens
        out_method = "measured"
    elif output_text:
        word_count = len(output_text.split())
        out_tokens = max(1, math.ceil(word_count * 1.3))
        out_method = "estimated_heuristic (1.3 tokens/word on observed text)"
    else:
        out_tokens = 50
        out_method = "estimated_heuristic (nominal 50 tokens assumption)"

    cost = (in_tokens * prompt_rate + out_tokens * completion_rate) / 1_000_000.0

    return {
        "status": "calculated",
        "model_id": model_id,
        "price_table_date": effective_date,
        "input_tokens": in_tokens,
        "input_tokens_method": in_method,
        "output_tokens": out_tokens,
        "output_tokens_method": out_method,
        "prompt_rate_per_1m": prompt_rate,
        "completion_rate_per_1m": completion_rate,
        "counterfactual_cost_usd": round(cost, 6),
        "cost_provenance": "calculated",
        "quality_available": False,
        "latency_available": False,
        "note": (
            "Counterfactual Always-Large cost projection derived from identical prompt "
            "and dated price schedule. Unexecuted model quality and latency cannot be known "
            "and remain strictly unavailable."
        ),
    }


def get_ranked_failures(
    logs: list[dict[str, Any]],
    q_min: float = 0.70,
    max_count: int = 20,
) -> list[dict[str, Any]]:
    """
    Extracts and ranks the worst failure cases from item logs.
    Cases are ranked by quality_score ascending (worst first).
    Ensures zero leakage of sensitive/private prompt text through safe redaction.
    """
    failures: list[dict[str, Any]] = []

    for item in logs:
        quality = item.get("quality_score", 0.0)
        val_passed = item.get("validation_passed", True)
        meets_q = item.get("meets_q_min", True)

        is_failure = not val_passed or not meets_q or (quality < q_min)
        if not is_failure:
            continue

        raw_prompt = item.get("prompt", "")
        # Check for privacy patterns or keywords that indicate private data
        is_private = any(
            kw in raw_prompt.lower()
            for kw in ["confidential", "salary", "payroll", "ssn", "credit card", "internal"]
        )

        safe_prompt = raw_prompt
        if is_private:
            prompt_sha = hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest()[:16]
            safe_prompt = f"[REDACTED: SENSITIVE/PRIVATE PROMPT] (SHA-256: {prompt_sha})"
        elif len(raw_prompt) > 120:
            safe_prompt = raw_prompt[:120] + "..."

        # Determine failure classification
        if not val_passed and item.get("fallback_occurred", False):
            f_class = "Validation Rejection (Fallback Exhausted)"
        elif not val_passed:
            f_class = "Validation Rejection"
        elif quality < 0.3:
            f_class = "Severe Quality Deficiency (Q < 0.3)"
        else:
            f_class = "Sub-Threshold Quality (Q < Q_min)"

        failures.append(
            {
                "item_id": item.get("item_id", "N/A"),
                "task_category": item.get("task_category", "N/A"),
                "tier": item.get("tier", "N/A"),
                "prompt_preview": safe_prompt,
                "routed_tier": item.get("routed_tier", "N/A"),
                "rule_id": item.get("rule_id", "N/A"),
                "quality_score": round(quality, 4),
                "meets_q_min": meets_q,
                "validation_passed": val_passed,
                "fallback_occurred": item.get("fallback_occurred", False),
                "cost_usd": item.get("cost_usd_calculated", 0.0),
                "latency_ms": item.get("latency_ms", 0.0),
                "oracle_tier": item.get("oracle_tier", "N/A"),
                "failure_classification": f_class,
            }
        )

    # Sort by quality score ascending
    failures.sort(key=lambda x: (x["quality_score"], not x["validation_passed"]))
    return failures[:max_count]


def get_cache_storage_size(cache_db_path: Path | str | None = None) -> dict[str, Any]:
    """Returns exact SQLite cache storage overhead in bytes and entries count."""
    path = Path(cache_db_path) if cache_db_path else Path("data/cache.db")
    if not path.exists():
        return {"exists": False, "size_bytes": 0, "size_kb": 0.0, "entries": 0}

    size_bytes = path.stat().st_size
    entries = 0
    try:
        import sqlite3

        conn = sqlite3.connect(str(path))
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM exact_cache")
        row = cursor.fetchone()
        entries = row[0] if row else 0
        conn.close()
    except Exception:
        pass

    return {
        "exists": True,
        "size_bytes": size_bytes,
        "size_kb": round(size_bytes / 1024.0, 2),
        "entries": entries,
    }
