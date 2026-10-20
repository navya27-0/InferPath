"""Configuration schemas and loaders for InferPath settings and reference tables."""

import contextlib
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from inferpath.models.telemetry import BoundaryDefinition

# Load .env file explicitly if present
load_dotenv()


class OpenRouterSettings(BaseModel):
    """OpenRouter API connectivity settings."""

    api_key: str = Field(default="", description="Secret OpenRouter API key; never logged")
    base_url: str = Field(default="https://openrouter.ai/api/v1")
    timeout_seconds: float = Field(default=30.0)
    max_retries: int = Field(default=2)
    small_model: str = Field(default="google/gemini-2.5-flash")
    large_model: str = Field(default="anthropic/claude-3.7-sonnet")


class RoutingSettings(BaseModel):
    """Operational toggles, cache TTLs, and thresholds for the routing pipeline."""

    enable_cache: bool = Field(default=True)
    enable_fallback: bool = Field(default=True)
    default_cache_ttl_seconds: int = Field(default=86400)
    cache_ttls: dict[str, int] = Field(
        default_factory=lambda: {
            "deterministic_a": 0,
            "template_b": 86400,
            "small_model_c": 86400,
            "large_model_d": 86400,
        }
    )
    thresholds: dict[str, float] = Field(
        default_factory=lambda: {
            "confidence_threshold": 0.80,
            "q_min": 0.70,
        }
    )
    complexity_keywords: list[str] = Field(
        default_factory=lambda: [
            "compare",
            "synthesize",
            "multi-step",
            "evaluate evidence",
            "proof",
            "prove",
            "step-by-step",
            "trade-offs",
            "counterexample",
            "in-depth analysis",
        ]
    )
    max_usd_per_run: float = Field(
        default=0.10, description="Per-run maximum calculated USD budget cap"
    )


class PricingTable(BaseModel):
    """Dated pricing table loading rates from YAML."""

    effective_date: str
    source: str
    verification_status: str = "unverified_illustrative"
    limitations: str | None = None
    rates_per_1m: dict[str, dict[str, float]]  # model -> {"prompt": ..., "completion": ...}


class EnvironmentalAssumption(BaseModel):
    """Structured representation of an environmental estimation factor."""

    unit: str
    low: float
    central: float
    high: float
    source_note: str
    verification_status: str = "unverified_illustrative"

    @model_validator(mode="after")
    def validate_bounds(self) -> "EnvironmentalAssumption":
        if self.low > self.central or self.central > self.high:
            raise ValueError(
                f"Bounds must satisfy low <= central <= high. Got low={self.low}, central={self.central}, high={self.high}"
            )
        return self


class CarbonBoundConfig(BaseModel):
    """Hardware energy coefficients and explicit boundary documentation."""

    effective_date: str
    verification_status: str = "unverified_illustrative"
    limitations: str | None = None
    system_boundary_energy: BoundaryDefinition
    system_boundary_water: BoundaryDefinition
    datacenter_pue: EnvironmentalAssumption
    coefficients_per_1000_tokens: dict[str, EnvironmentalAssumption]
    water_intensity: EnvironmentalAssumption

    @property
    def pue(self) -> dict[str, float]:
        """Backward-compatible mapping of PUE bounds."""
        return {
            "low": self.datacenter_pue.low,
            "central": self.datacenter_pue.central,
            "high": self.datacenter_pue.high,
        }

    @property
    def water_intensity_ml_per_wh(self) -> dict[str, float]:
        """Backward-compatible mapping of water intensity bounds."""
        return {
            "low": self.water_intensity.low,
            "central": self.water_intensity.central,
            "high": self.water_intensity.high,
        }


class Settings(BaseSettings):
    """InferPath root application settings."""

    model_config = SettingsConfigDict(
        env_prefix="INFERPATH_",
        env_file=".env",
        extra="ignore",
    )

    environment: str = Field(default="research")
    database_path: str = Field(default="data/storage/inferpath.db")
    config_dir: str = Field(default="configs")
    log_level: str = Field(default="INFO")
    strict_privacy: bool = Field(default=True)

    openrouter: OpenRouterSettings = Field(default_factory=OpenRouterSettings)
    routing: RoutingSettings = Field(default_factory=RoutingSettings)

    @model_validator(mode="before")
    @classmethod
    def apply_env_fallbacks(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "environment" not in data:
                env = (
                    os.getenv("INFERPATH_ENV")
                    or os.getenv("INFERPATH_ENVIRONMENT")
                    or os.getenv("ENVIRONMENT")
                )
                if env:
                    data["environment"] = env
            if "database_path" not in data:
                db = (
                    os.getenv("INFERPATH_DB_PATH")
                    or os.getenv("INFERPATH_DATABASE_PATH")
                    or os.getenv("DATABASE_PATH")
                )
                if db:
                    data["database_path"] = db
            if "log_level" not in data:
                lvl = os.getenv("INFERPATH_LOG_LEVEL") or os.getenv("LOG_LEVEL")
                if lvl:
                    data["log_level"] = lvl
            if "strict_privacy" not in data:
                sp = os.getenv("INFERPATH_STRICT_PRIVACY")
                if sp is not None:
                    data["strict_privacy"] = sp.strip().lower() in ("true", "1", "yes")
        return data


def load_settings(base_dir: Path | None = None) -> Settings:
    """Loads root configuration, populating OpenRouter API key from .env."""
    root = base_dir or Path.cwd()
    settings_yaml = root / "configs" / "settings.yaml"

    yaml_data: dict[str, Any] = {}
    if settings_yaml.exists():
        with open(settings_yaml, encoding="utf-8") as f:
            yaml_data = yaml.safe_load(f) or {}

    settings = Settings()

    # Apply YAML overrides if present
    if "app" in yaml_data:
        app_cfg = yaml_data["app"]
        settings.environment = app_cfg.get("environment", settings.environment)
        settings.database_path = app_cfg.get("database_path", settings.database_path)
        settings.log_level = app_cfg.get("log_level", settings.log_level)

    if "openrouter" in yaml_data:
        or_cfg = yaml_data["openrouter"]
        settings.openrouter.base_url = or_cfg.get("base_url", settings.openrouter.base_url)
        settings.openrouter.timeout_seconds = or_cfg.get(
            "timeout_seconds", settings.openrouter.timeout_seconds
        )
        settings.openrouter.max_retries = or_cfg.get("max_retries", settings.openrouter.max_retries)
        settings.openrouter.small_model = or_cfg.get("small_model", settings.openrouter.small_model)
        settings.openrouter.large_model = or_cfg.get("large_model", settings.openrouter.large_model)

    if "routing" in yaml_data:
        rt_cfg = yaml_data["routing"]
        settings.routing.enable_cache = rt_cfg.get("enable_cache", settings.routing.enable_cache)
        settings.routing.enable_fallback = rt_cfg.get(
            "enable_fallback", settings.routing.enable_fallback
        )
        settings.routing.default_cache_ttl_seconds = rt_cfg.get(
            "default_cache_ttl_seconds", settings.routing.default_cache_ttl_seconds
        )
        if "cache_ttls" in rt_cfg:
            settings.routing.cache_ttls = rt_cfg["cache_ttls"]
        if "thresholds" in rt_cfg:
            settings.routing.thresholds = rt_cfg["thresholds"]
        if "max_usd_per_run" in rt_cfg:
            settings.routing.max_usd_per_run = float(rt_cfg["max_usd_per_run"])

    # OpenRouter API key, base URL, and model selection from environment variables
    env_api_key = os.getenv("OPENROUTER_API_KEY")
    if env_api_key:
        settings.openrouter.api_key = env_api_key

    env_base_url = os.getenv("OPENROUTER_BASE_URL") or os.getenv("INFERPATH_OPENROUTER_BASE_URL")
    if env_base_url:
        settings.openrouter.base_url = env_base_url

    env_small_model = os.getenv("INFERPATH_SMALL_MODEL") or os.getenv("OPENROUTER_SMALL_MODEL")
    if env_small_model:
        settings.openrouter.small_model = env_small_model

    env_large_model = os.getenv("INFERPATH_LARGE_MODEL") or os.getenv("OPENROUTER_LARGE_MODEL")
    if env_large_model:
        settings.openrouter.large_model = env_large_model

    # Budget cap from environment
    env_budget = os.getenv("MAX_USD_PER_RUN") or os.getenv("INFERPATH_MAX_USD_PER_RUN")
    if env_budget:
        with contextlib.suppress(ValueError):
            settings.routing.max_usd_per_run = float(env_budget)

    # Environment
    env_environment = (
        os.getenv("INFERPATH_ENV") or os.getenv("INFERPATH_ENVIRONMENT") or os.getenv("ENVIRONMENT")
    )
    if env_environment:
        settings.environment = env_environment

    # Log level
    env_log_level = os.getenv("INFERPATH_LOG_LEVEL") or os.getenv("LOG_LEVEL")
    if env_log_level:
        settings.log_level = env_log_level

    # Database path
    env_db = (
        os.getenv("INFERPATH_DB_PATH")
        or os.getenv("INFERPATH_DATABASE_PATH")
        or os.getenv("DATABASE_PATH")
    )
    if env_db:
        settings.database_path = env_db

    # Routing toggles
    env_enable_cache = os.getenv("INFERPATH_ENABLE_CACHE")
    if env_enable_cache is not None:
        settings.routing.enable_cache = env_enable_cache.strip().lower() in ("true", "1", "yes")

    env_enable_fallback = os.getenv("INFERPATH_ENABLE_FALLBACK")
    if env_enable_fallback is not None:
        settings.routing.enable_fallback = env_enable_fallback.strip().lower() in (
            "true",
            "1",
            "yes",
        )

    # Strict privacy toggle
    env_strict_privacy = os.getenv("INFERPATH_STRICT_PRIVACY")
    if env_strict_privacy is not None:
        settings.strict_privacy = env_strict_privacy.strip().lower() in ("true", "1", "yes")

    return settings


def load_pricing_table(config_path: Path | None = None) -> PricingTable:
    """Loads dated price reference table."""
    if config_path is None:
        cand = Path.cwd() / "configs" / "pricing_table_2026_10.yaml"
        if cand.exists():
            config_path = cand
        else:
            config_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "configs"
                / "pricing_table_2026_10.yaml"
            )
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    meta = data.get("table_metadata", {})
    models = data.get("models", {})
    rates = {
        m: {
            "prompt": info.get("prompt_price_per_1m", 0.0),
            "completion": info.get("completion_price_per_1m", 0.0),
        }
        for m, info in models.items()
    }
    return PricingTable(
        effective_date=meta.get("date", "unknown"),
        source=meta.get("source", "unknown"),
        verification_status=meta.get("verification_status", "unverified_illustrative"),
        limitations=meta.get("limitations"),
        rates_per_1m=rates,
    )


def load_carbon_coefficients(config_path: Path | None = None) -> CarbonBoundConfig:
    """Loads energy and water estimation bounds and system boundary documentation."""
    if config_path is None:
        cand = Path.cwd() / "configs" / "carbon_coefficients.yaml"
        if cand.exists():
            config_path = cand
        else:
            config_path = (
                Path(__file__).resolve().parent.parent.parent.parent
                / "configs"
                / "carbon_coefficients.yaml"
            )
    with open(config_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    meta = data.get("metadata", {})

    pue_raw = data.get(
        "datacenter_pue",
        {
            "unit": "ratio",
            "low": 1.1,
            "central": 1.2,
            "high": 1.35,
            "source_note": "Unverified heuristic assumption",
        },
    )
    pue_assumption = EnvironmentalAssumption(**pue_raw)

    raw_coeffs = data.get("coefficients_per_1000_tokens", {})
    coeff_assumptions = {m: EnvironmentalAssumption(**vals) for m, vals in raw_coeffs.items()}

    water_raw = data.get(
        "water_intensity",
        {
            "unit": "mL / Wh",
            "low": 1.2,
            "central": 2.4,
            "high": 4.5,
            "source_note": "Unverified heuristic assumption",
        },
    )
    water_assumption = EnvironmentalAssumption(**water_raw)

    return CarbonBoundConfig(
        effective_date=meta.get("effective_date", "unknown"),
        verification_status=meta.get("verification_status", "unverified_illustrative"),
        limitations=meta.get("limitations"),
        system_boundary_energy=BoundaryDefinition(
            included=meta.get(
                "system_boundary_energy",
                "Scope 2 Datacenter operational compute (server dynamic + idle power per token multiplied by datacenter PUE)",
            ),
            excluded="Client device energy, network transit, Scope 3 embodied hardware manufacturing",
        ),
        system_boundary_water=BoundaryDefinition(
            included=meta.get(
                "system_boundary_water",
                "Datacenter on-site cooling water (WUE) and regional electricity generation water footprint",
            ),
            excluded="Supply-chain embodied manufacturing water footprint",
            notes=meta.get(
                "guidance", "Water is never a headline claim; reported as secondary metric only."
            ),
        ),
        datacenter_pue=pue_assumption,
        coefficients_per_1000_tokens=coeff_assumptions,
        water_intensity=water_assumption,
    )


# Alias for backward and framework naming compatibility
InferPathSettings = Settings
