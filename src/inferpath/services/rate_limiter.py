"""Shared production rate limiter service for InferPath API and routing pipelines."""

from __future__ import annotations

import threading
import time
from abc import ABC, abstractmethod


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
