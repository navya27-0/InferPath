"""Deterministic mock model adapters for offline testing and simulated execution."""

import hashlib
from typing import Any, Protocol, runtime_checkable

from inferpath.models.domain import ModelResult
from inferpath.models.enums import ProvenanceLabel


@runtime_checkable
class ModelAdapter(Protocol):
    """Abstract protocol for model execution adapters (Prompt 1 specification)."""

    model_id: str
    price_table_date: str

    def generate(self, prompt: str, **params: Any) -> ModelResult:
        """Generates a model completion synchronously."""
        ...

    async def generate_async(self, prompt: str, **params: Any) -> ModelResult:
        """Generates a model completion asynchronously."""
        ...


class BaseMockModel:
    """Base class providing deterministic canned generation and simulated provenance."""

    def __init__(
        self,
        model_id: str,
        prompt_price_per_1m: float,
        completion_price_per_1m: float,
        simulated_base_latency_ms: float,
        price_table_date: str = "2026-10-01",
    ):
        self.model_id = model_id
        self.prompt_price_per_1m = prompt_price_per_1m
        self.completion_price_per_1m = completion_price_per_1m
        self.simulated_base_latency_ms = simulated_base_latency_ms
        self.price_table_date = price_table_date

    def _determine_output_text(self, prompt: str) -> str:
        """Produces deterministic canned output based on prompt content and digest."""
        clean_prompt = prompt.strip().lower()

        if "25" in clean_prompt and "fahrenheit" in clean_prompt:
            return "25°C is equal to 77°F."
        if "boiling point" in clean_prompt and "nitrogen" in clean_prompt:
            return "The boiling point of nitrogen is -195.79°C (77.36 K)."
        if "summarize" in clean_prompt:
            return "Summary: The provided text outlines core resource-aware inference routing objectives."
        if "france" in clean_prompt and "capital" in clean_prompt:
            return "The capital of France is Paris."

        # General deterministic fallback using prompt digest
        digest_suffix = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        return f"[{self.model_id}] Deterministic simulated completion for request {digest_suffix}."

    def generate(self, prompt: str, **params: Any) -> ModelResult:
        """Synchronously produces deterministic canned output tagged strictly as simulated."""
        output_text = self._determine_output_text(prompt)

        # Deterministic simulated token counts (approx 1 token per word + overhead)
        prompt_words = len(prompt.split())
        output_words = len(output_text.split())

        simulated_input_tokens = max(1, prompt_words + 4)
        simulated_output_tokens = max(1, output_words + 2)

        # Calculated cost from dated rates
        cost_usd = (simulated_input_tokens / 1_000_000.0) * self.prompt_price_per_1m + (
            simulated_output_tokens / 1_000_000.0
        ) * self.completion_price_per_1m

        # Deterministic simulated latency
        simulated_latency_ms = self.simulated_base_latency_ms + (simulated_output_tokens * 1.5)

        return ModelResult(
            text=output_text,
            input_tokens=simulated_input_tokens,
            output_tokens=simulated_output_tokens,
            latency_ms=round(simulated_latency_ms, 2),
            cost_usd_calculated=round(cost_usd, 6),
            provider_reported_cost_usd=None,
            price_table_date=self.price_table_date,
            source=ProvenanceLabel.SIMULATED,  # Strictly SIMULATED
            model_id=self.model_id,
        )

    async def generate_async(self, prompt: str, **params: Any) -> ModelResult:
        """Asynchronous execution wrapper returning deterministic output."""
        return self.generate(prompt, **params)


class MockSmallModel(BaseMockModel):
    """
    Mock adapter representing Tier C (Small Model).
    Low simulated latency, low token price, deterministic canned responses.
    Operates strictly offline without network access or API keys.
    """

    def __init__(self, price_table_date: str = "2026-10-01"):
        super().__init__(
            model_id="mock-small",
            prompt_price_per_1m=0.05,
            completion_price_per_1m=0.20,
            simulated_base_latency_ms=35.0,
            price_table_date=price_table_date,
        )


class MockLargeModel(BaseMockModel):
    """
    Mock adapter representing Tier D (Large Model).
    Higher simulated latency, higher token price, richer deterministic responses.
    Operates strictly offline without network access or API keys.
    """

    def __init__(self, price_table_date: str = "2026-10-01"):
        super().__init__(
            model_id="mock-large",
            prompt_price_per_1m=2.50,
            completion_price_per_1m=10.00,
            simulated_base_latency_ms=110.0,
            price_table_date=price_table_date,
        )

    def _determine_output_text(self, prompt: str) -> str:
        base_output = super()._determine_output_text(prompt)
        return f"{base_output} Detailed analysis: The multi-tier routing architecture safely validates outputs."
