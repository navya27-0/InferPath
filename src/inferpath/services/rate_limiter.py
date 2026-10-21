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
import uuid
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

    @abstractmethod
    def get_remaining(self, client_id: str) -> int:
        """Returns the number of remaining requests allowed for client_id in the current window."""
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

    def get_remaining(self, client_id: str) -> int:
        """Returns the number of remaining requests allowed for client_id in the current window."""
        now = time.monotonic()
        cutoff = now - 60.0
        with self._lock:
            history = self._history.get(client_id, [])
            valid_history = [t for t in history if t > cutoff]
            return max(0, self.requests_per_minute - len(valid_history))

    def get_stats(self) -> dict[str, int]:
        """Returns diagnostic metadata about active rate-limited clients."""
        with self._lock:
            return {
                "active_clients": len(self._history),
                "max_clients": self.max_clients,
                "requests_per_minute": self.requests_per_minute,
            }


class RateLimiterBackendError(Exception):
    """Raised when the distributed rate limiter backend fails in fail-closed mode."""

    pass


class DistributedRateLimiter(BaseRateLimiter):
    """
    Interface and reference contract for distributed rate limiting across horizontal workers.

    Production implementations synchronize state using an external shared store
    (such as Redis sorted sets or token bucket Lua scripts):
    - Key: `{key_prefix}{client_id}`
    - Score: UNIX timestamp in seconds/microseconds
    - Atomic window: Remove entries older than `now - 60.0`, count remaining entries,
      and append current timestamp if under quota.

    Operational Controls:
    - fail_closed=True: In production, requires operational Redis backend; raises
      RateLimiterBackendError on outages to prevent quota evasion.
    - fail_closed=False: Degrades to bounded process-local tracking while recording
      `is_degraded=True` and incrementing failure count for health telemetry.
    """

    def __init__(
        self,
        redis_client: Any | None = None,
        requests_per_minute: int = 120,
        key_prefix: str = "inferpath:ratelimit:",
        fail_closed: bool = False,
    ):
        self.redis_client = redis_client
        self.requests_per_minute = requests_per_minute
        self.key_prefix = key_prefix
        self.fail_closed = fail_closed
        self.is_degraded: bool = False
        self.failure_count: int = 0
        self._fallback_limiter = RateLimiter(requests_per_minute=requests_per_minute)

    def is_allowed(self, client_id: str) -> bool:
        """Determines if request is allowed using distributed backend or fallback."""
        if self.redis_client is None:
            if self.fail_closed:
                self.is_degraded = True
                self.failure_count += 1
                raise RateLimiterBackendError(
                    "Distributed rate limiter requires active Redis backend when fail_closed=True."
                )
            self.is_degraded = True
            return self._fallback_limiter.is_allowed(client_id)

        try:
            now = time.time()
            cutoff = now - 60.0
            key = f"{self.key_prefix}{client_id}"

            member_id = f"{now}:{uuid.uuid4().hex}"

            # Atomic evaluation via Lua script if supported by the Redis client
            if hasattr(self.redis_client, "eval") and callable(self.redis_client.eval):
                lua_script = (
                    "local key = KEYS[1]\n"
                    "local now = tonumber(ARGV[1])\n"
                    "local cutoff = tonumber(ARGV[2])\n"
                    "local limit = tonumber(ARGV[3])\n"
                    "local ttl = tonumber(ARGV[4])\n"
                    "local member = ARGV[5]\n"
                    "redis.call('ZREMRANGEBYSCORE', key, '-inf', cutoff)\n"
                    "local current_count = redis.call('ZCARD', key)\n"
                    "if current_count < limit then\n"
                    "    redis.call('ZADD', key, now, member)\n"
                    "    redis.call('EXPIRE', key, ttl)\n"
                    "    return 1\n"
                    "else\n"
                    "    return 0\n"
                    "end"
                )
                res = self.redis_client.eval(
                    lua_script, 1, key, now, cutoff, self.requests_per_minute, 65, member_id
                )
                self.is_degraded = False
                return bool(res)

            try:
                pipe = self.redis_client.pipeline(transaction=True)
            except TypeError:
                pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.zcard(key)
            results = pipe.execute()
            count = results[1] if len(results) > 1 else (results[0] if results else 0)

            if count < self.requests_per_minute:
                try:
                    add_pipe = self.redis_client.pipeline(transaction=True)
                except TypeError:
                    add_pipe = self.redis_client.pipeline()
                add_pipe.zadd(key, {member_id: now})
                add_pipe.expire(key, 65)
                add_pipe.execute()
                self.is_degraded = False
                return True
            self.is_degraded = False
            return False
        except Exception as err:
            self.is_degraded = True
            self.failure_count += 1
            if self.fail_closed:
                raise RateLimiterBackendError(
                    f"Distributed rate limiter backend error: {err}"
                ) from err
            # Degrade gracefully to process-local tracking on external store failure
            return self._fallback_limiter.is_allowed(client_id)

    def reset(self) -> None:
        """Resets all rate limiting state across the distributed store or fallback."""
        if self.redis_client is None:
            if self.fail_closed:
                raise RateLimiterBackendError("No Redis backend available to reset.")
            self._fallback_limiter.reset()
            return

        try:
            keys = self.redis_client.keys(f"{self.key_prefix}*")
            if keys:
                self.redis_client.delete(*keys)
            self.is_degraded = False
        except Exception as err:
            self.is_degraded = True
            self.failure_count += 1
            if self.fail_closed:
                raise RateLimiterBackendError(f"Backend reset failure: {err}") from err
            self._fallback_limiter.reset()

    def get_remaining(self, client_id: str) -> int:
        """Returns the number of remaining requests allowed for client_id in the current window."""
        if self.redis_client is None:
            if self.fail_closed:
                self.is_degraded = True
                self.failure_count += 1
                raise RateLimiterBackendError(
                    "Distributed rate limiter requires active Redis backend when fail_closed=True."
                )
            return self._fallback_limiter.get_remaining(client_id)

        try:
            now = time.time()
            cutoff = now - 60.0
            key = f"{self.key_prefix}{client_id}"
            pipe = self.redis_client.pipeline()
            pipe.zremrangebyscore(key, "-inf", cutoff)
            pipe.zcard(key)
            results = pipe.execute()
            count = results[1] if len(results) > 1 else (results[0] if results else 0)
            self.is_degraded = False
            return max(0, self.requests_per_minute - count)
        except Exception as err:
            self.is_degraded = True
            self.failure_count += 1
            if self.fail_closed:
                raise RateLimiterBackendError(
                    f"Distributed rate limiter backend error: {err}"
                ) from err
            return self._fallback_limiter.get_remaining(client_id)
