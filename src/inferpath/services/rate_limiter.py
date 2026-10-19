"""Shared rate limiter service for InferPath API and routing pipelines.

Architectural Scope & Horizontal Scalability:
- Single-Process / Single-Worker Boundary:
  `RateLimiter` provides in-memory sliding window rate limiting designed for single-process,
  single-worker deployments, staging nodes, and local development. State is tracked in local
  memory via thread-safe coarse locks.
- Horizontal Multi-Worker Scaling:
  In multi-worker production deployments (e.g. Uvicorn with multiple workers or Kubernetes
  horizontal pod autoscaling), local process memory is partitioned. To coordinate global
  rate limits across independent processes, implement or configure `DistributedRateLimiter`
  backed by an external atomic store (such as Redis or Memcached).
"""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod
from typing import Any


class BaseRateLimiter(ABC):
    """Abstract interface for rate limiting services."""

    @abstractmethod
    def is_allowed(self, client_id: str) -> bool:
        """Determines if a request from client_id is permitted under current quota."""
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets all tracking histories."""
        raise NotImplementedError


class RateLimiter(BaseRateLimiter):
    """
    In-memory rate limiter using a sliding 60-second window with bounded storage.

    Scope: Single-process and single-worker deployments.
    For horizontally scaled multi-worker deployments, see `DistributedRateLimiter`.

    Features:
    - Thread-safe tracking with coarse locks.
    - Automatic eviction of expired client request timestamps.
    - Capacity-bounded client table to prevent unbounded memory growth under adversarial DDOS.
    """

    def __init__(self, requests_per_minute: int = 120, max_clients: int = 10_000):
        self.requests_per_minute = requests_per_minute
        self.max_clients = max_clients
        self._history: dict[str, list[float]] = {}
        self._lock = threading.Lock()

    def is_allowed(self, client_id: str) -> bool:
        """Checks if client_id is within rate quota using a sliding window."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            history = self._history.get(client_id, [])
            valid_history = [t for t in history if t > cutoff]

            if len(valid_history) >= self.requests_per_minute:
                self._history[client_id] = valid_history
                return False

            # Bounded capacity check
            if client_id not in self._history and len(self._history) >= self.max_clients:
                # Evict expired client histories
                expired_keys = [k for k, v in self._history.items() if not v or max(v) <= cutoff]
                for k in expired_keys:
                    del self._history[k]
                # If still at cap, evict oldest entry
                if len(self._history) >= self.max_clients:
                    oldest_k = next(iter(self._history))
                    del self._history[oldest_k]

            valid_history.append(now)
            self._history[client_id] = valid_history
            return True

    def reset(self) -> None:
        """Clears all client histories."""
        with self._lock:
            self._history.clear()

    def get_stats(self) -> dict[str, int]:
        """Returns diagnostic metadata about active rate-limited clients."""
        with self._lock:
            return {
                "active_clients": len(self._history),
                "max_clients": self.max_clients,
                "requests_per_minute": self.requests_per_minute,
            }


class DistributedRateLimiter(BaseRateLimiter):
    """
    Interface and reference contract for distributed rate limiting across horizontal workers.

    Production implementations synchronize state using an external shared store
    (such as Redis sorted sets or token bucket Lua scripts):
    - Key: `{key_prefix}{client_id}`
    - Score: UNIX timestamp in seconds/microseconds
    - Atomic window: Remove entries older than `now - 60.0`, count remaining entries,
      and append current timestamp if under quota.

    If no external client is provided or the connection is unavailable, it gracefully
    falls back to process-local sliding window tracking.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        requests_per_minute: int = 120,
        key_prefix: str = "inferpath:ratelimit:",
    ):
        self.redis_client = redis_client
        self.requests_per_minute = requests_per_minute
        self.key_prefix = key_prefix
        self._fallback_limiter = RateLimiter(requests_per_minute=requests_per_minute)

    def is_allowed(self, client_id: str) -> bool:
        """Determines if request is allowed using distributed backend or fallback."""
        if self.redis_client is None:
            return self._fallback_limiter.is_allowed(client_id)

        try:
            now = time.time()
            cutoff = now - 60.0
            key = f"{self.key_prefix}{client_id}"

            # Atomic evaluation via Lua script if supported by the Redis client
            if hasattr(self.redis_client, "eval") and callable(self.redis_client.eval):
                lua_script = (
                    "local key = KEYS[1]\n"
                    "local now = tonumber(ARGV[1])\n"
                    "local cutoff = tonumber(ARGV[2])\n"
                    "local limit = tonumber(ARGV[3])\n"
                    "local ttl = tonumber(ARGV[4])\n"
                    "redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)\n"
                    "local current_count = redis.call('ZCARD', key)\n"
                    "if current_count < limit then\n"
                    "    redis.call('ZADD', key, now, tostring(now))\n"
                    "    redis.call('EXPIRE', key, ttl)\n"
                    "    return 1\n"
                    "else\n"
                    "    return 0\n"
                    "end"
                )
                res = self.redis_client.eval(
                    lua_script, 1, key, now, cutoff, self.requests_per_minute, 65
                )
                return bool(res)

            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.zcard(key)
            results = pipe.execute()
            count = results[1] if len(results) > 1 else (results[0] if results else 0)

            if count < self.requests_per_minute:
                add_pipe = self.redis_client.pipeline()
                add_pipe.zadd(key, {str(now): now})
                add_pipe.expire(key, 65)
                add_pipe.execute()
                return True
            return False
        except Exception:
            # Degrade gracefully to process-local tracking on external store failure
            return self._fallback_limiter.is_allowed(client_id)

    def reset(self) -> None:
        """Resets all rate limiting state across the distributed store or fallback."""
        if self.redis_client is None:
            self._fallback_limiter.reset()
            return

        try:
            keys = self.redis_client.keys(f"{self.key_prefix}*")
            if keys:
                self.redis_client.delete(*keys)
        except Exception:
            self._fallback_limiter.reset()
