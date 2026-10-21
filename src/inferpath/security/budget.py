"""Per-run budget tracking, reservation, and reconciliation."""

import collections
import math
import threading
import uuid
from enum import StrEnum

from inferpath.config.settings import PricingTable


class ReservationState(StrEnum):
    """Lifecycle state for a budget reservation."""

    RESERVED = "RESERVED"
    RECONCILED = "RECONCILED"
    RELEASED = "RELEASED"
    EXCEEDED = "EXCEEDED"


class UnknownModelPricingError(ValueError):
    """Raised when a model is not configured in the active pricing table."""

    pass


class BudgetExceededError(ValueError):
    """Raised when an estimated model invocation exceeds the remaining per-run budget cap."""

    def __init__(self, message: str, estimated_cost: float, remaining_budget: float, cap: float):
        super().__init__(message)
        self.estimated_cost = estimated_cost
        self.remaining_budget = remaining_budget
        self.cap = cap


class RunBudgetTracker:
    """
    Manages budget enforcement at run / session scope (MAX_USD_PER_RUN).

    Scope:
    - Experiment Run / Pipeline Instance Scope.
    - Tracks and enforces cumulative budget limits across all requests handled by
      a single pipeline or experiment run instance.
    - When reused sequentially or concurrently within an application instance,
      it enforces the cumulative ceiling across all processed calls.

    Policy:
    - Tracks cumulative calculated cost across all attempts (including retries and fallback).
    - Thread-safe reservations and reconciliations via RLock.
    - Reserves estimated maximum cost prior to dispatch to prevent overspending.
    - Reconciles reservation with actual calculated cost upon response.
    - Rejection of invalid negative parameters.
    - Bounded retention of completed reservation history (evicts oldest settled entries to prevent unbounded memory growth).
    - Fails fast with BudgetExceededError when budget is insufficient.

    Notice:
    The cap is enforced against the internal calculated cost model (dated price table).
    It may differ from external provider billing if provider rates change.
    """

    def __init__(
        self,
        max_usd: float | None = None,
        pricing_table: PricingTable | None = None,
        *,
        max_budget_usd: float | None = None,
        history_limit: int = 1000,
    ):
        target_max = max_usd if max_usd is not None else max_budget_usd
        if target_max is None:
            target_max = 0.0
        if not math.isfinite(target_max):
            raise ValueError(f"max_usd must be a finite number, got {target_max}")
        if target_max < 0.0:
            raise ValueError(f"max_usd cannot be negative, got {target_max}")
        self.max_usd = float(target_max)
        self.pricing_table = pricing_table
        self.spent_usd: float = 0.0
        self.reserved_usd: float = 0.0
        self._active_reservations: dict[str, float] = {}
        self._settled_history: collections.deque[tuple[str, float, ReservationState]] = (
            collections.deque(maxlen=history_limit)
        )
        self._settled_tokens: set[str] = set()
        self._lock = threading.RLock()

    @property
    def _reservations(self) -> dict[str, dict[str, float | ReservationState]]:
        """Backward-compatible audit dictionary of active and recent settled reservations."""
        with self._lock:
            view: dict[str, dict[str, float | ReservationState]] = {
                t: {"cost": c, "state": ReservationState.RESERVED}
                for t, c in self._active_reservations.items()
            }
            for t, c, state in self._settled_history:
                view[t] = {"cost": c, "state": state}
            return view

    @property
    def remaining_budget(self) -> float:
        """Remaining uncommitted budget in USD."""
        with self._lock:
            return max(0.0, self.max_usd - (self.spent_usd + self.reserved_usd))

    @property
    def total_spent_usd(self) -> float:
        """Total spent budget in USD (alias for spent_usd)."""
        with self._lock:
            return self.spent_usd

    @property
    def total_cap_usd(self) -> float:
        """Total budget cap in USD."""
        return self.max_usd

    @property
    def max_budget_usd(self) -> float:
        """Total budget cap in USD (alias for max_usd)."""
        return self.max_usd

    def estimate_call_cost(
        self, model_id: str, prompt: str, max_tokens: int | None = None
    ) -> float:
        """Estimates maximum plausible cost using configured price table and prompt length."""
        if self.pricing_table is not None and model_id not in self.pricing_table.rates_per_1m:
            table_date = getattr(self.pricing_table, "effective_date", "unconfigured")
            raise UnknownModelPricingError(
                f"Cannot estimate call cost: model '{model_id}' is unpriced in active pricing table ({table_date})."
            )

        if not self.pricing_table:
            # Safe default fallback rate if price table not loaded in isolated unit tests
            return 0.005

        rates = self.pricing_table.rates_per_1m[model_id]
        prompt_rate = rates.get("prompt", 0.0)
        completion_rate = rates.get("completion", 0.0)

        # Conservative input token estimate: 1 token per 3 characters or word count + overhead
        input_tokens = max(1, len(prompt.split()) + 8, len(prompt) // 3)
        # Conservative output token estimate: requested max_tokens or standard 1024 bound
        output_tokens = max_tokens if max_tokens and max_tokens > 0 else 1024

        cost = (input_tokens / 1_000_000.0) * prompt_rate + (
            output_tokens / 1_000_000.0
        ) * completion_rate
        return round(cost, 6)

    def reserve(self, estimated_cost: float) -> str:
        """
        Reserves estimated cost prior to dispatch.
        Raises BudgetExceededError if remaining budget cannot cover the call.
        """
        if not math.isfinite(estimated_cost):
            raise ValueError(f"Estimated cost must be a finite number, got {estimated_cost}")
        if estimated_cost < 0:
            raise ValueError("Estimated cost cannot be negative.")

        with self._lock:
            if estimated_cost > self.remaining_budget:
                raise BudgetExceededError(
                    f"Estimated call cost ${estimated_cost:.6f} exceeds remaining budget "
                    f"${self.remaining_budget:.6f} (run cap: ${self.max_usd:.6f}).",
                    estimated_cost=estimated_cost,
                    remaining_budget=self.remaining_budget,
                    cap=self.max_usd,
                )

            token = uuid.uuid4().hex
            self._active_reservations[token] = estimated_cost
            self.reserved_usd += estimated_cost
            return token

    def reconcile(self, token: str, actual_cost: float | None = None) -> None:
        """Reconciles reservation with actual calculated cost.

        Guarantees:
        - If actual_cost is None (e.g. usage unavailable from provider), falls back to the reserved estimate.
        - Validates that actual_cost is finite and non-negative.
        - Releases active reservation.
        - Increments cumulative spent_usd.
        - If cumulative spend exceeds max_usd (when max_usd > 0), marks state as EXCEEDED
          and raises BudgetExceededError to guarantee a hard cap.
        """
        with self._lock:
            if actual_cost is None:
                actual_cost = self._active_reservations.get(token, 0.0)

            if not math.isfinite(actual_cost):
                raise ValueError(f"Actual cost must be a finite number, got {actual_cost}")
            if actual_cost < 0:
                raise ValueError("Actual cost cannot be negative.")

            reserved = self._active_reservations.pop(token, None)
            if reserved is None:
                return

            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            self.spent_usd += actual_cost

            # Enforce hard budget cap: check if cumulative spend exceeds max_usd
            over_cap = self.max_usd > 0.0 and self.spent_usd > self.max_usd
            state = ReservationState.EXCEEDED if over_cap else ReservationState.RECONCILED

            if (
                self._settled_history.maxlen is not None
                and len(self._settled_history) == self._settled_history.maxlen
                and self._settled_history
            ):
                evicted_token, _, _ = self._settled_history[0]
                self._settled_tokens.discard(evicted_token)

            self._settled_history.append((token, actual_cost, state))
            self._settled_tokens.add(token)

            if over_cap:
                raise BudgetExceededError(
                    f"Reconciled cost ${actual_cost:.6f} caused cumulative spend ${self.spent_usd:.6f} "
                    f"to exceed run budget cap ${self.max_usd:.6f}.",
                    estimated_cost=actual_cost,
                    remaining_budget=0.0,
                    cap=self.max_usd,
                )

    def release(self, token: str) -> None:
        """Releases reservation without adding cost (e.g. on call cancellation)."""
        with self._lock:
            reserved = self._active_reservations.pop(token, None)
            if reserved is None:
                return

            self.reserved_usd = max(0.0, self.reserved_usd - reserved)

            if (
                self._settled_history.maxlen is not None
                and len(self._settled_history) == self._settled_history.maxlen
                and self._settled_history
            ):
                evicted_token, _, _ = self._settled_history[0]
                self._settled_tokens.discard(evicted_token)

            self._settled_history.append((token, 0.0, ReservationState.RELEASED))
            self._settled_tokens.add(token)
