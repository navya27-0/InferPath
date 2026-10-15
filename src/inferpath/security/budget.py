"""Per-run budget tracking, reservation, and reconciliation."""

import threading
import uuid
from enum import StrEnum

from inferpath.config.settings import PricingTable


class ReservationState(StrEnum):
    """Lifecycle state for a budget reservation."""

    RESERVED = "RESERVED"
    RECONCILED = "RECONCILED"
    RELEASED = "RELEASED"


class BudgetExceededError(ValueError):
    """Raised when an estimated model invocation exceeds the remaining per-run budget cap."""

    def __init__(self, message: str, estimated_cost: float, remaining_budget: float, cap: float):
        super().__init__(message)
        self.estimated_cost = estimated_cost
        self.remaining_budget = remaining_budget
        self.cap = cap


class RunBudgetTracker:
    """
    Manages per-run calculated budget enforcement (MAX_USD_PER_RUN).

    Policy:
    - Tracks cumulative calculated cost across all attempts (including retries and fallback).
    - Thread-safe reservations and reconciliations via RLock.
    - Reserves estimated maximum cost prior to dispatch to prevent overspending.
    - Reconciles reservation with actual calculated cost upon response.
    - Rejection of invalid negative parameters.
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
    ):
        target_max = max_usd if max_usd is not None else max_budget_usd
        if target_max is None:
            target_max = 0.0
        if target_max < 0.0:
            raise ValueError(f"max_usd cannot be negative, got {target_max}")
        self.max_usd = float(target_max)
        self.pricing_table = pricing_table
        self.spent_usd: float = 0.0
        self.reserved_usd: float = 0.0
        self._reservations: dict[str, dict[str, float | ReservationState]] = {}
        self._lock = threading.RLock()

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
        if not self.pricing_table:
            # Safe default fallback rate if price table not loaded
            return 0.005

        rates = self.pricing_table.rates_per_1m.get(model_id, {"prompt": 3.0, "completion": 15.0})
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
            self._reservations[token] = {
                "cost": estimated_cost,
                "state": ReservationState.RESERVED,
            }
            self.reserved_usd += estimated_cost
            return token

    def reconcile(self, token: str, actual_cost: float) -> None:
        """Reconciles reservation with actual calculated cost."""
        if actual_cost < 0:
            raise ValueError("Actual cost cannot be negative.")

        with self._lock:
            res = self._reservations.get(token)
            if not res or res["state"] != ReservationState.RESERVED:
                return

            res["state"] = ReservationState.RECONCILED
            reserved = float(res["cost"])
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
            self.spent_usd += actual_cost

    def release(self, token: str) -> None:
        """Releases reservation without adding cost (e.g. on call cancellation)."""
        with self._lock:
            res = self._reservations.get(token)
            if not res or res["state"] != ReservationState.RESERVED:
                return

            res["state"] = ReservationState.RELEASED
            reserved = float(res["cost"])
            self.reserved_usd = max(0.0, self.reserved_usd - reserved)
