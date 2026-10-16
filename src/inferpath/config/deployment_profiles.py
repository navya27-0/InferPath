"""Explicit production deployment profiles for InferPath service operations."""

from __future__ import annotations

import os
from enum import StrEnum

from pydantic import BaseModel, Field


class ProfileName(StrEnum):
    """Supported standardized deployment profiles."""

    LOCAL_DEMO = "local-demo"
    STAGING = "staging"
    PRODUCTION = "production"
    BENCHMARK_OFFLINE = "benchmark-offline"


class DeploymentProfileConfig(BaseModel):
    """Configuration constraints and defaults defined for a deployment profile."""

    name: ProfileName
    require_api_key: bool = Field(
        description="Enforces mandatory X-API-Key / Bearer authentication"
    )
    adapter_mode: str = Field(default="simulated", description="'simulated' or 'real'")
    rate_limit_rpm: int = Field(
        default=120, description="Sliding window requests per minute per client"
    )
    log_level: str = Field(default="INFO")
    structured_json_logging: bool = Field(default=True)
    enable_tracing: bool = Field(default=True)
    default_bind_host: str = Field(default="127.0.0.1")
    allow_mock_fallback: bool = Field(default=True)
    trusted_proxies: set[str] = Field(default_factory=set)


# Predefined canonical profiles
PROFILES: dict[ProfileName, DeploymentProfileConfig] = {
    ProfileName.LOCAL_DEMO: DeploymentProfileConfig(
        name=ProfileName.LOCAL_DEMO,
        require_api_key=False,
        adapter_mode="simulated",
        rate_limit_rpm=120,
        log_level="DEBUG",
        structured_json_logging=False,
        enable_tracing=True,
        default_bind_host="127.0.0.1",
        allow_mock_fallback=True,
    ),
    ProfileName.STAGING: DeploymentProfileConfig(
        name=ProfileName.STAGING,
        require_api_key=True,
        adapter_mode="simulated",
        rate_limit_rpm=300,
        log_level="INFO",
        structured_json_logging=True,
        enable_tracing=True,
        default_bind_host="127.0.0.1",
        allow_mock_fallback=True,
    ),
    ProfileName.PRODUCTION: DeploymentProfileConfig(
        name=ProfileName.PRODUCTION,
        require_api_key=True,
        adapter_mode="real",
        rate_limit_rpm=600,
        log_level="INFO",
        structured_json_logging=True,
        enable_tracing=True,
        default_bind_host="0.0.0.0",
        allow_mock_fallback=False,
    ),
    ProfileName.BENCHMARK_OFFLINE: DeploymentProfileConfig(
        name=ProfileName.BENCHMARK_OFFLINE,
        require_api_key=False,
        adapter_mode="simulated",
        rate_limit_rpm=10_000,
        log_level="WARNING",
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
    2. INFERPATH_PROFILE environment variable
    3. INFERPATH_DEPLOYMENT_MODE environment variable
    4. Default: local-demo
    """
    raw = (
        profile_str
        or os.getenv("INFERPATH_PROFILE")
        or os.getenv("INFERPATH_DEPLOYMENT_MODE")
        or "local-demo"
    )
    cleaned = raw.strip().lower()

    # Map legacy aliases
    alias_map = {
        "protected": ProfileName.PRODUCTION,
        "prod": ProfileName.PRODUCTION,
        "production": ProfileName.PRODUCTION,
        "demo": ProfileName.LOCAL_DEMO,
        "local": ProfileName.LOCAL_DEMO,
        "local-demo": ProfileName.LOCAL_DEMO,
        "staging": ProfileName.STAGING,
        "stage": ProfileName.STAGING,
        "benchmark": ProfileName.BENCHMARK_OFFLINE,
        "benchmark-offline": ProfileName.BENCHMARK_OFFLINE,
        "offline": ProfileName.BENCHMARK_OFFLINE,
    }

    selected_enum = alias_map.get(cleaned, ProfileName.LOCAL_DEMO)
    return PROFILES[selected_enum]
