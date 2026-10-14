"""OpenRouter API adapter with measured latency, calculated cost, retries, and budget checks."""

import asyncio
import os
import time
from typing import Any

import httpx

from inferpath.config.settings import PricingTable
from inferpath.models.domain import ModelResult
from inferpath.models.enums import ProvenanceLabel
from inferpath.security.budget import RunBudgetTracker
from inferpath.security.hashing import scrub_api_keys


class OpenRouterError(Exception):
    """Base exception for OpenRouter API interactions."""

    pass


class OpenRouterAuthError(OpenRouterError):
    """Raised on HTTP 401 or 403 authentication failures."""

    pass


class OpenRouterRateLimitError(OpenRouterError):
    """Raised on HTTP 429 rate limit errors when retries are exhausted."""

    pass


class OpenRouterProviderError(OpenRouterError):
    """Raised on HTTP 5xx server or gateway errors when retries are exhausted."""

    pass


class OpenRouterTimeoutError(OpenRouterError):
    """Raised when an HTTP connection or read times out."""

    pass


class OpenRouterResponseError(OpenRouterError):
    """Raised when provider returns an unparseable or malformed response payload."""

    pass


class CircuitBreakerOpenError(OpenRouterError):
    """Raised when request is rejected immediately because circuit breaker is OPEN."""

    pass


class CircuitBreaker:
    """
    Standard state-machine circuit breaker.

    States:
      - CLOSED: normal dispatch
      - OPEN: fail-fast on consecutive provider errors (5xx / 429 / timeouts)
      - HALF_OPEN: test probe after recovery timeout
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_seconds: float = 30.0,
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout_seconds = recovery_timeout_seconds
        self.state = "CLOSED"
        self.consecutive_failures = 0
        self.last_failure_time: float = 0.0

    def can_execute(self) -> bool:
        now = time.monotonic()
        if self.state == "OPEN":
            if now - self.last_failure_time >= self.recovery_timeout_seconds:
                self.state = "HALF_OPEN"
                return True
            return False
        return True

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.state = "CLOSED"

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.last_failure_time = time.monotonic()
        if self.consecutive_failures >= self.failure_threshold:
            self.state = "OPEN"


class OpenRouterAdapter:
    """
    HTTP client adapter connecting to OpenRouter (OpenAI-compatible chat completions).

    Guarantees:
    - Implements ModelAdapter protocol (generate, generate_async).
    - Monotonic wall-clock latency measurement.
    - Calculated cost derived from measured token usage and dated price table.
    - Provider-reported cost stored separately as provider_reported_cost_usd.
    - Zero API key or Authorization header leakage in logs or exceptions.
    - Bounded exponential backoff retries for transient errors (429, 5xx, timeouts).
    - Optional RunBudgetTracker reservation and reconciliation prior to dispatch.
    - Supports dependency-injected httpx client/transport for hermetic offline testing.
    """

    def __init__(
        self,
        model_id: str,
        pricing_table: PricingTable,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        timeout_seconds: float = 30.0,
        max_retries: int = 2,
        client: httpx.Client | None = None,
        async_client: httpx.AsyncClient | None = None,
        budget_tracker: RunBudgetTracker | None = None,
        circuit_breaker: CircuitBreaker | None = None,
    ):
        self.model_id = model_id
        self.pricing_table = pricing_table
        self.price_table_date = pricing_table.effective_date
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._client = client
        self._async_client = async_client
        self.budget_tracker = budget_tracker
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

    def _get_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "HTTP-Referer": "https://inferpath.research",
            "X-Title": "InferPath Research Prototype",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _prepare_payload(self, prompt: str, **params: Any) -> dict[str, Any]:
        return {
            "model": self.model_id,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": params.get("temperature", 0.0),
            "max_tokens": params.get("max_tokens", 1024),
        }

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        rates = self.pricing_table.rates_per_1m.get(
            self.model_id, {"prompt": 0.0, "completion": 0.0}
        )
        prompt_rate = rates.get("prompt", 0.0)
        completion_rate = rates.get("completion", 0.0)
        cost = (input_tokens / 1_000_000.0) * prompt_rate + (
            output_tokens / 1_000_000.0
        ) * completion_rate
        return round(cost, 6)

    def _parse_response_body(self, data: dict[str, Any], latency_ms: float) -> ModelResult:
        choices = data.get("choices", [])
        if not choices:
            raise OpenRouterResponseError("Provider response contains no choices.")

        text = choices[0].get("message", {}).get("content", "")
        usage = data.get("usage", {})

        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        # Calculated cost from verified/dated price table
        cost_calculated = self._calculate_cost(input_tokens, output_tokens)

        # Provider-reported cost (if reported by provider or OpenRouter usage header)
        provider_cost = usage.get("cost")
        if provider_cost is not None:
            try:
                provider_cost = float(provider_cost)
            except (ValueError, TypeError):
                provider_cost = None

        return ModelResult(
            text=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            cost_usd_calculated=cost_calculated,
            provider_reported_cost_usd=provider_cost,
            price_table_date=self.price_table_date,
            source=ProvenanceLabel.MEASURED,
            model_id=self.model_id,
        )

    def generate(self, prompt: str, **params: Any) -> ModelResult:
        """Synchronous model completion with retries, timeout, and budget tracking."""
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN for model '{self.model_id}'. Dispatch halted to prevent cascading failure."
            )

        endpoint = f"{self.base_url}/chat/completions"
        payload = self._prepare_payload(prompt, **params)
        headers = self._get_headers()

        # 1. Budget reservation
        reservation_token: str | None = None
        if self.budget_tracker:
            est_cost = self.budget_tracker.estimate_call_cost(
                self.model_id, prompt, params.get("max_tokens")
            )
            reservation_token = self.budget_tracker.reserve(est_cost)

        client = self._client or httpx.Client(timeout=self.timeout_seconds)
        close_client = self._client is None

        start_time = time.perf_counter()
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = client.post(endpoint, json=payload, headers=headers)
                    if response.status_code == 200:
                        self.circuit_breaker.record_success()
                        latency_ms = (time.perf_counter() - start_time) * 1000.0
                        data = response.json()
                        result = self._parse_response_body(data, latency_ms)

                        # Reconcile budget
                        if self.budget_tracker and reservation_token:
                            self.budget_tracker.reconcile(
                                reservation_token, result.cost_usd_calculated
                            )
                        return result

                    # Non-retryable auth errors
                    if response.status_code in (401, 403):
                        raise OpenRouterAuthError(
                            f"Authentication failure (HTTP {response.status_code}). Verify credentials."
                        )

                    # Retryable errors: 429, 5xx
                    if (
                        response.status_code in (429, 500, 502, 503, 504)
                        and attempt < self.max_retries
                    ):
                        time.sleep(0.2 * (2**attempt))
                        continue

                    if response.status_code == 429:
                        raise OpenRouterRateLimitError("Rate limit exceeded on OpenRouter API.")

                    raise OpenRouterProviderError(
                        f"OpenRouter HTTP error {response.status_code}: {scrub_api_keys(response.text)}"
                    )

                except httpx.TimeoutException as err:
                    if attempt < self.max_retries:
                        time.sleep(0.2 * (2**attempt))
                        continue
                    raise OpenRouterTimeoutError(
                        f"Request timed out after {self.timeout_seconds}s: {err}"
                    ) from err
                except httpx.TransportError as err:
                    if attempt < self.max_retries:
                        time.sleep(0.2 * (2**attempt))
                        continue
                    raise OpenRouterProviderError(
                        f"Transport error connecting to OpenRouter: {err}"
                    ) from err

            raise OpenRouterProviderError("Exhausted retries calling OpenRouter API.")

        except (OpenRouterRateLimitError, OpenRouterProviderError, OpenRouterTimeoutError):
            self.circuit_breaker.record_failure()
            if self.budget_tracker and reservation_token:
                self.budget_tracker.release(reservation_token)
            raise
        except Exception:
            if self.budget_tracker and reservation_token:
                self.budget_tracker.release(reservation_token)
            raise
        finally:
            if close_client:
                client.close()

    async def generate_async(self, prompt: str, **params: Any) -> ModelResult:
        """Asynchronous model completion with bounded retries and budget reservation."""
        if not self.circuit_breaker.can_execute():
            raise CircuitBreakerOpenError(
                f"Circuit breaker is OPEN for model '{self.model_id}'. Dispatch halted to prevent cascading failure."
            )

        endpoint = f"{self.base_url}/chat/completions"
        payload = self._prepare_payload(prompt, **params)
        headers = self._get_headers()

        reservation_token: str | None = None
        if self.budget_tracker:
            est_cost = self.budget_tracker.estimate_call_cost(
                self.model_id, prompt, params.get("max_tokens")
            )
            reservation_token = self.budget_tracker.reserve(est_cost)

        async_client = self._async_client or httpx.AsyncClient(timeout=self.timeout_seconds)
        close_client = self._async_client is None

        start_time = time.perf_counter()
        try:
            for attempt in range(self.max_retries + 1):
                try:
                    response = await async_client.post(endpoint, json=payload, headers=headers)
                    if response.status_code == 200:
                        self.circuit_breaker.record_success()
                        latency_ms = (time.perf_counter() - start_time) * 1000.0
                        data = response.json()
                        result = self._parse_response_body(data, latency_ms)

                        if self.budget_tracker and reservation_token:
                            self.budget_tracker.reconcile(
                                reservation_token, result.cost_usd_calculated
                            )
                        return result

                    if response.status_code in (401, 403):
                        raise OpenRouterAuthError(
                            f"Authentication failure (HTTP {response.status_code}). Verify credentials."
                        )

                    if (
                        response.status_code in (429, 500, 502, 503, 504)
                        and attempt < self.max_retries
                    ):
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue

                    if response.status_code == 429:
                        raise OpenRouterRateLimitError("Rate limit exceeded on OpenRouter API.")

                    raise OpenRouterProviderError(
                        f"OpenRouter HTTP error {response.status_code}: {scrub_api_keys(response.text)}"
                    )

                except httpx.TimeoutException as err:
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue
                    raise OpenRouterTimeoutError(
                        f"Request timed out after {self.timeout_seconds}s: {err}"
                    ) from err
                except httpx.TransportError as err:
                    if attempt < self.max_retries:
                        await asyncio.sleep(0.2 * (2**attempt))
                        continue
                    raise OpenRouterProviderError(
                        f"Transport error connecting to OpenRouter: {err}"
                    ) from err

            raise OpenRouterProviderError("Exhausted retries calling OpenRouter API.")

        except (OpenRouterRateLimitError, OpenRouterProviderError, OpenRouterTimeoutError):
            self.circuit_breaker.record_failure()
            if self.budget_tracker and reservation_token:
                self.budget_tracker.release(reservation_token)
            raise
        except Exception:
            if self.budget_tracker and reservation_token:
                self.budget_tracker.release(reservation_token)
            raise
        finally:
            if close_client:
                await async_client.aclose()
