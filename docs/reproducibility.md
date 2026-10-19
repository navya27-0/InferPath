# Reproducibility Protocols & Sanitized Run Manifests

## 1. Principles of Empirical Reproducibility

InferPath requires full experimental reproducibility without compromising proprietary customer prompt data or API credential secrecy.

Every benchmark trial produces:
1. **Cryptographic Provenance Record**: SHA-256 digests over datasets, router source code commit, and dated pricing tables.
2. **Deterministic Token & Latency Breakdowns**: Exact provider-reported prompt/completion tokens, calculated pricing, and millisecond-level latency distributions.
3. **Transport & Network Telemetry**: Full recording of HTTP status distributions, rate-limit backoffs, and retry attempts.
4. **Sanitized Public Manifest**: A scrubbed, public-facing manifest matching [`sanitized_public_manifest.json`](../data/results/fixtures/sanitized_public_manifest.json).

---

## 2. Public Manifest Schema & Verification

Artifact: [`data/results/fixtures/sanitized_public_manifest.json`](../data/results/fixtures/sanitized_public_manifest.json)

```json
{
  "manifest_version": "2.0.0",
  "run_id": "run_fixture_20261019_124500",
  "timestamp": "2026-10-19T12:45:00Z",
  "adapter_mode": "fixture",
  "provider": "openrouter",
  "model_id": "anthropic/claude-3.7-sonnet",
  "code_commit_sha": "57caf142107ba28b39167a8f09d7f1c562747fd8",
  "hashes": {
    "dataset_sha256": "3f5393d4c5c9c6328639...",
    "pricing_table_sha256": "29098642ea18e6f7779b...",
    "router_config_sha256": "38a798516da89fe20409..."
  },
  "summary_metrics": {
    "total_items": 160,
    "quality_pass_rate": 0.85,
    "avg_latency_ms": 78.4,
    "total_cost_usd_calculated": 0.0435
  },
  "network_and_transport": {
    "http_status_codes": { "200": 160 },
    "retry_count": 0,
    "rate_limit_events": 0
  }
}
```

### Verifying a Benchmark Run
Researchers can verify the integrity of the published results by executing:
```bash
python -c "from inferpath.evaluation.reproducibility import compute_file_sha256; print(compute_file_sha256('data/comprehensive_workload_dataset.json'))"
```
The resulting checksum must exactly match `dataset_sha256` in the manifest.
