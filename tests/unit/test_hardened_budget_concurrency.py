"""Unit tests for RunBudgetTracker thread-safety, concurrency, and validation."""

import threading

import pytest

from inferpath.security.budget import BudgetExceededError, RunBudgetTracker


def test_budget_concurrency_thread_safety():
    """Concurrent threads reserving and reconciling must preserve exact balance invariants."""
    tracker = RunBudgetTracker(max_usd=10.0)
    num_threads = 20
    cost_per_call = 0.25

    def worker():
        try:
            token = tracker.reserve(cost_per_call)
            tracker.reconcile(token, cost_per_call)
        except BudgetExceededError:
            pass

    threads = [threading.Thread(target=worker) for _ in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # Total spent should equal 20 * 0.25 = 5.0
    assert tracker.total_spent_usd == pytest.approx(5.0)
    assert tracker.remaining_budget == pytest.approx(5.0)
    assert tracker.reserved_usd == pytest.approx(0.0)


def test_budget_negative_inputs_rejected():
    """Negative values for budget, reserve cost, or reconcile cost must raise ValueError."""
    # Negative initial cap
    with pytest.raises(ValueError, match="max_usd cannot be negative"):
        RunBudgetTracker(max_usd=-1.0)

    tracker = RunBudgetTracker(max_usd=5.0)

    # Negative reservation
    with pytest.raises(ValueError, match="Estimated cost cannot be negative"):
        tracker.reserve(-0.1)

    # Negative reconcile
    tok = tracker.reserve(1.0)
    with pytest.raises(ValueError, match="Actual cost cannot be negative"):
        tracker.reconcile(tok, -0.5)


def test_budget_idempotent_reconciliation():
    """Reconciling or releasing an already-settled token must be an idempotent no-op."""
    tracker = RunBudgetTracker(max_usd=5.0)
    tok = tracker.reserve(1.0)

    assert tracker.reserved_usd == 1.0
    assert tracker.remaining_budget == 4.0

    # First reconcile
    tracker.reconcile(tok, 0.8)
    assert tracker.spent_usd == 0.8
    assert tracker.reserved_usd == 0.0
    assert tracker.remaining_budget == 4.2

    # Second reconcile with same token -> no-op
    tracker.reconcile(tok, 0.8)
    assert tracker.spent_usd == 0.8
    assert tracker.remaining_budget == 4.2

    # Release after reconcile -> no-op
    tracker.release(tok)
    assert tracker.spent_usd == 0.8
    assert tracker.remaining_budget == 4.2


def test_budget_alias_and_keyword_arguments():
    """Verify max_budget_usd keyword argument and total_spent_usd property."""
    tracker = RunBudgetTracker(max_budget_usd=2.5)
    assert tracker.max_budget_usd == 2.5
    assert tracker.total_spent_usd == 0.0
    assert tracker.remaining_budget == 2.5
