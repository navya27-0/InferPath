"""Factory service for constructing model adapters based on deployment mode and credentials."""

from __future__ import annotations

from typing import TYPE_CHECKING

from inferpath.handlers.mock_adapter import MockLargeModel, MockSmallModel, ModelAdapter
from inferpath.handlers.openrouter_adapter import OpenRouterAdapter

if TYPE_CHECKING:
    from inferpath.config.settings import PricingTable, Settings
    from inferpath.security.budget import RunBudgetTracker


class AdapterFactory:
    """Constructs small and large model adapters according to deployment mode and credentials."""

    @staticmethod
    def create_small_adapter(
        settings: Settings,
        pricing_table: PricingTable,
        budget_tracker: RunBudgetTracker | None = None,
        adapter_mode: str | None = None,
        allow_mock_fallback: bool = True,
    ) -> ModelAdapter:
        """Creates small model adapter explicitly based on mode, credentials, and fallback policy."""
        has_key = bool(settings.openrouter.api_key)
        if adapter_mode in ("simulated", "fixture"):
            return AdapterFactory.build_mock_small(settings, pricing_table)

        if adapter_mode == "real":
            if has_key:
                return AdapterFactory.build_openrouter_small(
                    settings, pricing_table, budget_tracker
                )
            if allow_mock_fallback:
                return AdapterFactory.build_mock_small(settings, pricing_table)
            raise RuntimeError(
                "adapter_mode is 'real' and allow_mock_fallback=False, but no OpenRouter API key is available."
            )

        if has_key:
            return AdapterFactory.build_openrouter_small(settings, pricing_table, budget_tracker)
        return AdapterFactory.build_mock_small(settings, pricing_table)

    @staticmethod
    def create_large_adapter(
        settings: Settings,
        pricing_table: PricingTable,
        budget_tracker: RunBudgetTracker | None = None,
        adapter_mode: str | None = None,
        allow_mock_fallback: bool = True,
    ) -> ModelAdapter:
        """Creates large model adapter explicitly based on mode, credentials, and fallback policy."""
        has_key = bool(settings.openrouter.api_key)
        if adapter_mode in ("simulated", "fixture"):
            return AdapterFactory.build_mock_large(settings, pricing_table)

        if adapter_mode == "real":
            if has_key:
                return AdapterFactory.build_openrouter_large(
                    settings, pricing_table, budget_tracker
                )
            if allow_mock_fallback:
                return AdapterFactory.build_mock_large(settings, pricing_table)
            raise RuntimeError(
                "adapter_mode is 'real' and allow_mock_fallback=False, but no OpenRouter API key is available."
            )

        if has_key:
            return AdapterFactory.build_openrouter_large(settings, pricing_table, budget_tracker)
        return AdapterFactory.build_mock_large(settings, pricing_table)

    @staticmethod
    def build_mock_small(settings: Settings, pricing_table: PricingTable) -> MockSmallModel:
        """Constructs deterministic MockSmallModel with current pricing schedule."""
        rates = pricing_table.rates_per_1m.get(
            settings.openrouter.small_model, {"prompt": 0.075, "completion": 0.30}
        )
        return MockSmallModel(
            model_id=settings.openrouter.small_model,
            prompt_price_per_1m=rates["prompt"],
            completion_price_per_1m=rates["completion"],
            simulated_base_latency_ms=120.0,
            price_table_date=pricing_table.effective_date,
        )

    @staticmethod
    def build_openrouter_small(
        settings: Settings,
        pricing_table: PricingTable,
        budget_tracker: RunBudgetTracker | None = None,
    ) -> OpenRouterAdapter:
        """Constructs live OpenRouterAdapter for small model tier."""
        return OpenRouterAdapter(
            model_id=settings.openrouter.small_model,
            pricing_table=pricing_table,
            api_key=settings.openrouter.api_key,
            base_url=settings.openrouter.base_url,
            timeout_seconds=settings.openrouter.timeout_seconds,
            max_retries=settings.openrouter.max_retries,
            budget_tracker=budget_tracker,
        )

    @staticmethod
    def build_mock_large(settings: Settings, pricing_table: PricingTable) -> MockLargeModel:
        """Constructs deterministic MockLargeModel with current pricing schedule."""
        rates = pricing_table.rates_per_1m.get(
            settings.openrouter.large_model, {"prompt": 3.0, "completion": 15.0}
        )
        return MockLargeModel(
            model_id=settings.openrouter.large_model,
            prompt_price_per_1m=rates["prompt"],
            completion_price_per_1m=rates["completion"],
            simulated_base_latency_ms=450.0,
            price_table_date=pricing_table.effective_date,
        )

    @staticmethod
    def build_openrouter_large(
        settings: Settings,
        pricing_table: PricingTable,
        budget_tracker: RunBudgetTracker | None = None,
    ) -> OpenRouterAdapter:
        """Constructs live OpenRouterAdapter for large model tier."""
        return OpenRouterAdapter(
            model_id=settings.openrouter.large_model,
            pricing_table=pricing_table,
            api_key=settings.openrouter.api_key,
            base_url=settings.openrouter.base_url,
            timeout_seconds=settings.openrouter.timeout_seconds,
            max_retries=settings.openrouter.max_retries,
            budget_tracker=budget_tracker,
        )
