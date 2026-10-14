"""Prometheus-compatible metrics exposition collector for InferPath."""

from __future__ import annotations

import threading


class MetricsCollector:
    """Thread-safe Prometheus metrics registry and exporter for InferPath."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.requests_total: dict[tuple[str, str], int] = {}  # (tier, status) -> count
        self.latency_buckets: list[float] = [
            5.0,
            25.0,
            50.0,
            100.0,
            250.0,
            500.0,
            1000.0,
            2500.0,
            5000.0,
        ]
        self.latency_histogram: dict[float, int] = {b: 0 for b in self.latency_buckets}
        self.latency_count: int = 0
        self.latency_sum_ms: float = 0.0
        self.cost_usd_total: float = 0.0
        self.cache_hits_total: int = 0
        self.cache_misses_total: int = 0
        self.privacy_blocked_total: int = 0
        self.fallback_total: int = 0
        self.budget_exceeded_total: int = 0

    def record_request(
        self,
        tier: str,
        status: str,
        latency_ms: float,
        cost_usd: float = 0.0,
        was_cached: bool = False,
        fallback_used: bool = False,
    ) -> None:
        with self._lock:
            key = (tier, status)
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

            self.latency_count += 1
            self.latency_sum_ms += latency_ms
            for bucket in self.latency_buckets:
                if latency_ms <= bucket:
                    self.latency_histogram[bucket] += 1

            self.cost_usd_total += cost_usd

            if was_cached:
                self.cache_hits_total += 1
            else:
                self.cache_misses_total += 1

            if fallback_used:
                self.fallback_total += 1

    def record_privacy_block(self) -> None:
        with self._lock:
            self.privacy_blocked_total += 1
            key = ("BLOCKED", "privacy_violation")
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

    def record_budget_exceeded(self) -> None:
        with self._lock:
            self.budget_exceeded_total += 1
            key = ("BLOCKED", "budget_exceeded")
            self.requests_total[key] = self.requests_total.get(key, 0) + 1

    def export_prometheus(self) -> str:
        """Renders metrics in Prometheus standard text exposition format (version 0.0.4)."""
        lines: list[str] = []

        with self._lock:
            # Request counter
            lines.append(
                "# HELP inferpath_requests_total Total requests processed by tier and status."
            )
            lines.append("# TYPE inferpath_requests_total counter")
            if not self.requests_total:
                lines.append('inferpath_requests_total{tier="none",status="none"} 0')
            for (tier, status), count in sorted(self.requests_total.items()):
                lines.append(f'inferpath_requests_total{{tier="{tier}",status="{status}"}} {count}')

            # Latency histogram
            lines.append(
                "# HELP inferpath_request_duration_ms Request execution duration in milliseconds."
            )
            lines.append("# TYPE inferpath_request_duration_ms histogram")
            cum_count = 0
            for bucket in self.latency_buckets:
                cum_count = self.latency_histogram[bucket]
                lines.append(f'inferpath_request_duration_ms_bucket{{le="{bucket}"}} {cum_count}')
            lines.append(f'inferpath_request_duration_ms_bucket{{le="+Inf"}} {self.latency_count}')
            lines.append(f"inferpath_request_duration_ms_sum {self.latency_sum_ms:.2f}")
            lines.append(f"inferpath_request_duration_ms_count {self.latency_count}")

            # Cost total
            lines.append("# HELP inferpath_cost_usd_total Total inference cost in USD.")
            lines.append("# TYPE inferpath_cost_usd_total counter")
            lines.append(f"inferpath_cost_usd_total {self.cost_usd_total:.6f}")

            # Cache metrics
            lines.append("# HELP inferpath_cache_hits_total Total exact-cache hits.")
            lines.append("# TYPE inferpath_cache_hits_total counter")
            lines.append(f"inferpath_cache_hits_total {self.cache_hits_total}")

            lines.append("# HELP inferpath_cache_misses_total Total cache lookups that missed.")
            lines.append("# TYPE inferpath_cache_misses_total counter")
            lines.append(f"inferpath_cache_misses_total {self.cache_misses_total}")

            # Security and policy counters
            lines.append(
                "# HELP inferpath_privacy_blocked_total Total requests blocked by privacy guardrails."
            )
            lines.append("# TYPE inferpath_privacy_blocked_total counter")
            lines.append(f"inferpath_privacy_blocked_total {self.privacy_blocked_total}")

            lines.append(
                "# HELP inferpath_fallback_total Total requests escalated to higher tier via fallback."
            )
            lines.append("# TYPE inferpath_fallback_total counter")
            lines.append(f"inferpath_fallback_total {self.fallback_total}")

        return "\n".join(lines) + "\n"


GLOBAL_METRICS = MetricsCollector()
