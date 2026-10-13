"""Pytest configuration, shared fixtures, and network isolation for InferPath test suite."""

import socket
from pathlib import Path

import pytest

from inferpath.cache.exact_cache import ExactCache
from inferpath.config.settings import (
    CarbonBoundConfig,
    PricingTable,
    Settings,
    load_carbon_coefficients,
    load_pricing_table,
    load_settings,
)
from inferpath.security.privacy import PrivacyPolicyEnforcer
from inferpath.telemetry.database import DatabaseManager

_original_socket_connect = socket.socket.connect


@pytest.fixture(autouse=True)
def enforce_test_network_isolation(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch):
    """
    Automated safeguard blocking outbound network connections in all default tests.
    Ensures tests run completely hermetic and offline.
    Permits local loopback IPC (127.0.0.1, localhost) required for internal asyncio event loop wakeups.
    Only tests decorated with @pytest.mark.live are exempt.
    """
    if "live" in request.keywords:
        return

    def blocked_connect(self, address, *args, **kwargs):
        # Allow loopback connections for local async event loops
        if isinstance(address, tuple) and len(address) >= 2:
            host = str(address[0])
            if host in ("127.0.0.1", "localhost", "::1", "0.0.0.0") or host.startswith("127."):
                return _original_socket_connect(self, address, *args, **kwargs)

        raise RuntimeError(
            f"Outbound network access to '{address}' is strictly blocked during default test suite execution. "
            "Tests must execute offline using local deterministic logic or mock adapters."
        )

    monkeypatch.setattr(socket.socket, "connect", blocked_connect)


@pytest.fixture
def temp_db_path(tmp_path: Path) -> str:
    """Provides an isolated SQLite database path for testing."""
    return str(tmp_path / "test_inferpath.db")


@pytest.fixture
def db_manager(temp_db_path: str) -> DatabaseManager:
    """Provides an initialized test SQLite database manager."""
    mgr = DatabaseManager(temp_db_path)
    mgr.init_schema_sync()
    return mgr


@pytest.fixture
def exact_cache(temp_db_path: str) -> ExactCache:
    """Provides an initialized test exact cache engine."""
    return ExactCache(temp_db_path)


@pytest.fixture
def project_root() -> Path:
    """Returns the project root directory."""
    return Path(__file__).resolve().parent.parent


@pytest.fixture
def real_settings(project_root: Path) -> Settings:
    """Loads system settings from project root configs."""
    return load_settings(project_root)


@pytest.fixture
def real_pricing(project_root: Path) -> PricingTable:
    """Loads dated price reference table."""
    return load_pricing_table(project_root / "configs" / "pricing_table_2026_10.yaml")


@pytest.fixture
def real_carbon_bounds(project_root: Path) -> CarbonBoundConfig:
    """Loads carbon and water coefficient bounds."""
    return load_carbon_coefficients(project_root / "configs" / "carbon_coefficients.yaml")


@pytest.fixture
def privacy_enforcer(project_root: Path) -> PrivacyPolicyEnforcer:
    """Provides privacy policy enforcer initialized with configuration."""
    policy_path = project_root / "configs" / "privacy_policies.yaml"
    return PrivacyPolicyEnforcer(str(policy_path))
