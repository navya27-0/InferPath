"""Deployment profiles configuring security boundaries, rates, and operational defaults."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum


class ProfileName(StrEnum):
    """Canonical supported deployment profile identifiers."""

    LOCAL_DEMO = "local-demo"
    STAGING = "staging"
    PRODUCTION = "production"
    BENCHMARK_OFFLINE = "benchmark-offline"


@dataclass(frozen=True)
class DeploymentProfileConfig:
    """Security, concurrency, and telemetry defaults for a deployment tier."""

    name: ProfileName
    require_api_key: bool
    rate_limit_rpm: int
    adapter_mode: str
    log_level: str
    structured_json_logging: bool
    enable_tracing: bool
    default_bind_host: str
    allow_mock_fallback: bool


PROFILES: dict[ProfileName, DeploymentProfileConfig] = {
    ProfileName.LOCAL_DEMO: DeploymentProfileConfig(
        name=ProfileName.LOCAL_DEMO,
        require_api_key=False,
        rate_limit_rpm=60,
        adapter_mode="simulated",
        log_level="INFO",
        structured_json_logging=False,
        enable_tracing=True,
        default_bind_host="127.0.0.1",
        allow_mock_fallback=True,
    ),
    ProfileName.STAGING: DeploymentProfileConfig(
        name=ProfileName.STAGING,
        require_api_key=True,
        rate_limit_rpm=120,
        adapter_mode="real",
        log_level="INFO",
        structured_json_logging=True,
        enable_tracing=True,
        default_bind_host="0.0.0.0",
        allow_mock_fallback=True,
    ),
    ProfileName.PRODUCTION: DeploymentProfileConfig(
        name=ProfileName.PRODUCTION,
        require_api_key=True,
        rate_limit_rpm=240,
        adapter_mode="real",
        log_level="WARNING",
        structured_json_logging=True,
        enable_tracing=True,
        default_bind_host="0.0.0.0",
        allow_mock_fallback=False,
    ),
    ProfileName.BENCHMARK_OFFLINE: DeploymentProfileConfig(
        name=ProfileName.BENCHMARK_OFFLINE,
        require_api_key=False,
        rate_limit_rpm=10_000,
        adapter_mode="simulated",
        log_level="ERROR",
        structured_json_logging=False,
        enable_tracing=False,
        default_bind_host="127.0.0.1",
        allow_mock_fallback=True,
    ),
}


def get_deployment_profile(profile_str: str | None = None) -> DeploymentProfileConfig:
    """
    Resolves the active deployment profile.

    Lookup priority:
    1. Explicit function argument
    2. INFERPATH_DEPLOYMENT_PROFILE environment variable
    3. INFERPATH_PROFILE environment variable
    4. INFERPATH_DEPLOYMENT_MODE environment variable
    5. Environment detection: if INFERPATH_ENV or ENVIRONMENT is 'production'/'prod', defaults to production
    6. Default: local-demo
    """
    env_signal = (os.getenv("INFERPATH_ENV") or os.getenv("ENVIRONMENT") or "").strip().lower()

    raw = (
        profile_str
        or os.getenv("INFERPATH_DEPLOYMENT_PROFILE")
        or os.getenv("INFERPATH_PROFILE")
        or os.getenv("INFERPATH_DEPLOYMENT_MODE")
    )
    if not raw:
        raw = "production" if env_signal in ("production", "prod") else "local-demo"

    cleaned = raw.strip().lower()

    # Map standardized aliases
    alias_map = {
        "protected": ProfileName.STAGING,
        "prod": ProfileName.PRODUCTION,
        "production": ProfileName.PRODUCTION,
        "demo": ProfileName.LOCAL_DEMO,
        "local": ProfileName.LOCAL_DEMO,
        "local-demo": ProfileName.LOCAL_DEMO,
        "dev": ProfileName.LOCAL_DEMO,
        "development": ProfileName.LOCAL_DEMO,
        "staging": ProfileName.STAGING,
        "stage": ProfileName.STAGING,
        "benchmark": ProfileName.BENCHMARK_OFFLINE,
        "benchmark-offline": ProfileName.BENCHMARK_OFFLINE,
        "offline": ProfileName.BENCHMARK_OFFLINE,
        "eval": ProfileName.BENCHMARK_OFFLINE,
        "evaluation": ProfileName.BENCHMARK_OFFLINE,
    }

    if cleaned not in alias_map:
        valid_profiles = sorted({p.value for p in ProfileName} | {"development", "evaluation"})
        raise ValueError(
            f"Invalid deployment profile '{raw}'. Supported profiles: {valid_profiles}"
        )

    selected_enum = alias_map[cleaned]

    # Reject unauthenticated profile when environment signal is production
    if env_signal in ("production", "prod") and selected_enum == ProfileName.LOCAL_DEMO:
        raise ValueError(
            f"Conflicting deployment configuration: ENVIRONMENT='{env_signal}' cannot be combined "
            f"with unauthenticated '{raw}' profile. Set ENVIRONMENT=development for local testing "
            "or select a protected/production profile."
        )

    return PROFILES[selected_enum]
