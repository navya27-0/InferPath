"""Real system-level concurrency and load benchmark for InferPath API.

Evaluates throughput (RPS), p50/p90/p95/p99 latency percentiles,
status code distributions, and cache dynamics under concurrent ASGI client loads.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from inferpath.api.main import app

RESULTS_PATH = Path("data/results/benchmarks/concurrency_benchmark.json")

# Representative heterogeneous workload pool
SAMPLE_PROMPTS = [
    # Tier A: Math & Unit conversions
    {"prompt": "Calculate: 125 * 8 + 350", "privacy_class": "public"},
    {"prompt": "Convert 100 meters to feet", "privacy_class": "public"},
    {"prompt": "What is 250 miles in kilometers?", "privacy_class": "public"},
    # Tier B: Template slot-filling
    {
        "prompt": "Generate a formal meeting invitation for project review on Friday at 2pm.",
        "privacy_class": "public",
    },
    # Tier C: Summarization & standard Q&A
    {
        "prompt": "Summarize the key differences between synchronous and asynchronous I/O in Python.",
        "privacy_class": "public",
    },
    {
        "prompt": "Classify the sentiment of this review: 'The product arrived quickly and exceeded expectations.'",
        "privacy_class": "public",
    },
    # Tier D: Complex synthesis
    {
        "prompt": "Explain the architectural trade-offs between Paxos and Raft consensus algorithms.",
        "privacy_class": "public",
    },
    # Exact repeats to test concurrent SQLite cache lookup contention
    {"prompt": "Calculate: 125 * 8 + 350", "privacy_class": "public"},
    {"prompt": "Convert 100 meters to feet", "privacy_class": "public"},
    # Security guardrail: Confidential request (must intercept with HTTP 403)
    {
        "prompt": "Confidential internal earnings draft before public filing.",
        "privacy_class": "confidential_private",
        "allow_external_transmission": False,
    },
]


async def run_worker(
    worker_id: int,
    concurrency_level: int,
    client: httpx.AsyncClient,
    queue: asyncio.Queue[tuple[int, dict[str, Any]]],
    results: list[dict[str, Any]],
) -> None:
    while not queue.empty():
        req_idx, payload = await queue.get()
        headers = {"X-Forwarded-For": f"10.{concurrency_level}.{worker_id}.{req_idx % 200}"}
        start_time = time.perf_counter()
        status_code = 0
        tier = "unknown"
        cached = False
        try:
            resp = await client.post("/v1/respond", json=payload, headers=headers, timeout=30.0)
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = resp.status_code
            if resp.status_code == 200:
                data = resp.json()
                tier = data.get("routed_tier", "unknown")
                cached = data.get("cache_status") == "HIT"
        except Exception:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            status_code = 500

        results.append(
            {
                "status_code": status_code,
                "latency_ms": latency_ms,
                "tier": tier,
                "cached": cached,
            }
        )
        queue.task_done()


async def benchmark_concurrency_level(concurrency: int, total_requests: int) -> dict[str, Any]:
    # Use ASGI transport directly for zero-socket, highly deterministic latency profiling
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver") as client:
        queue: asyncio.Queue[tuple[int, dict[str, Any]]] = asyncio.Queue()
        for i in range(total_requests):
            payload = SAMPLE_PROMPTS[i % len(SAMPLE_PROMPTS)]
            queue.put_nowait((i, payload))

        results: list[dict[str, Any]] = []
        start_wall = time.perf_counter()

        workers = [
            asyncio.create_task(run_worker(i, concurrency, client, queue, results))
            for i in range(concurrency)
        ]
        await queue.join()
        for w in workers:
            w.cancel()

        total_wall_sec = time.perf_counter() - start_wall

    latencies = [r["latency_ms"] for r in results]
    success_count = sum(1 for r in results if r["status_code"] in (200, 403))
    cache_hits = sum(1 for r in results if r["cached"])
    privacy_blocks = sum(1 for r in results if r["status_code"] == 403)
    rate_limits = sum(1 for r in results if r["status_code"] == 429)

    return {
        "concurrency": concurrency,
        "total_requests": total_requests,
        "wall_time_sec": round(total_wall_sec, 3),
        "throughput_rps": round(total_requests / total_wall_sec, 1),
        "success_rate_pct": round(success_count / total_requests * 100.0, 1),
        "latency_min_ms": round(float(np.min(latencies)), 2),
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 2),
        "latency_p90_ms": round(float(np.percentile(latencies, 90)), 2),
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 2),
        "latency_p99_ms": round(float(np.percentile(latencies, 99)), 2),
        "latency_max_ms": round(float(np.max(latencies)), 2),
        "latency_mean_ms": round(float(np.mean(latencies)), 2),
        "cache_hits": cache_hits,
        "privacy_blocks": privacy_blocks,
        "rate_limits": rate_limits,
    }


async def main():
    print("=== InferPath Concurrency & Load Benchmark ===")
    concurrency_levels = [1, 5, 10, 20]
    requests_per_test = 100

    benchmarks = []
    for c in concurrency_levels:
        print(f"\n--> Profiling concurrency level: {c} workers ({requests_per_test} requests)...")
        res = await benchmark_concurrency_level(c, requests_per_test)
        benchmarks.append(res)
        print(
            f"    RPS: {res['throughput_rps']} | "
            f"p50: {res['latency_p50_ms']}ms | "
            f"p95: {res['latency_p95_ms']}ms | "
            f"p99: {res['latency_p99_ms']}ms | "
            f"Success: {res['success_rate_pct']}%"
        )

    # Save to disk
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "requests_per_level": requests_per_test,
                "benchmarks": benchmarks,
            },
            f,
            indent=2,
        )

    print(f"\nSaved structured load benchmark to {RESULTS_PATH}")
    print(
        "\n| Concurrency | Throughput (RPS) | Latency p50 | Latency p90 | Latency p95 | Latency p99 | Success Rate |"
    )
    print("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |")
    for b in benchmarks:
        print(
            f"| **{b['concurrency']} workers** | {b['throughput_rps']} RPS | "
            f"{b['latency_p50_ms']} ms | {b['latency_p90_ms']} ms | "
            f"{b['latency_p95_ms']} ms | {b['latency_p99_ms']} ms | {b['success_rate_pct']}% |"
        )


if __name__ == "__main__":
    asyncio.run(main())
