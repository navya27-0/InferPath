"""Production secret manager abstraction supporting container file secrets and env fallback."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path


class SecretProvider(ABC):
    """Abstract interface for secret resolution backends."""

    @abstractmethod
    def get_secret(self, key: str) -> str | None:
        """Retrieves raw secret value by key, or returns None if not found."""
        raise NotImplementedError


class FileSecretProvider(SecretProvider):
    """
    Resolves secrets from a directory containing files named after the secret keys.
    Compatible with Docker Secrets, Kubernetes Secret Volume mounts, and local secret dirs.
    """

    def __init__(self, secrets_dir: str | Path | None = None):
        self.secrets_dir = Path(secrets_dir or os.getenv("INFERPATH_SECRETS_DIR", "/run/secrets"))

    def get_secret(self, key: str) -> str | None:
        if not self.secrets_dir.exists() or not self.secrets_dir.is_dir():
            return None

        # Check exact key, lowercase, and lowercase with underscores replaced by hyphens
        candidates = [
            self.secrets_dir / key,
            self.secrets_dir / key.lower(),
            self.secrets_dir / key.lower().replace("_", "-"),
        ]
        for candidate in candidates:
            if candidate.is_file():
                try:
                    val = candidate.read_text(encoding="utf-8").strip()
                    if val:
                        return val
                except OSError:
                    continue
        return None


class EnvironmentSecretProvider(SecretProvider):
    """Resolves secrets from process environment variables and loaded .env."""

    def __init__(self, prefix: str | None = None):
        self.prefix = prefix

    def get_secret(self, key: str) -> str | None:
        if self.prefix:
            val_custom = os.getenv(f"{self.prefix}{key}")
            if val_custom is not None and val_custom.strip():
                return val_custom.strip()

        val = os.getenv(key)
        if val is not None and val.strip():
            return val.strip()
        # Also check with INFERPATH_ prefix
        prefixed = f"INFERPATH_{key}"
        val_prefixed = os.getenv(prefixed)
        if val_prefixed is not None and val_prefixed.strip():
            return val_prefixed.strip()
        return None


class MockSecretProvider(SecretProvider):
    """In-memory secret provider for hermetic testing and isolated unit verification."""

    def __init__(self, initial_secrets: dict[str, str] | None = None):
        self._secrets: dict[str, str] = initial_secrets.copy() if initial_secrets else {}

    def set_secret(self, key: str, value: str) -> None:
        self._secrets[key] = value

    def get_secret(self, key: str) -> str | None:
        return self._secrets.get(key)


class SecretManager:
    """
    Production-grade composite secret manager.

    Search priority:
    1. Mounted container files (Docker / Kubernetes secrets)
    2. Environment variables (process environment / .env)
    3. In-memory / Mock fallback (for testing)
    """

    def __init__(self, providers: list[SecretProvider] | None = None):
        self.providers: list[SecretProvider] = providers or [
            FileSecretProvider(),
            EnvironmentSecretProvider(),
        ]
        self._cache: dict[str, str] = {}

    def get(self, key: str, default: str | None = None) -> str | None:
        """Resolves secret value across registered providers with local caching."""
        if key in self._cache:
            return self._cache[key]

        for provider in self.providers:
            val = provider.get_secret(key)
            if val is not None:
                self._cache[key] = val
                return val

        return default

    def require(self, key: str) -> str:
        """Retrieves secret value or raises ValueError if missing or empty."""
        val = self.get(key)
        if not val:
            raise ValueError(
                f"Required secret '{key}' was not found in any registered secret provider."
            )
        return val

    def mask_secret(self, key_or_val: str, show_prefix: int = 4, show_suffix: int = 4) -> str:
        """Returns safe masked representation for logs (e.g. sk-o...89ab)."""
        val = self.get(key_or_val) or key_or_val
        if len(val) <= show_prefix + show_suffix:
            return "***"
        return f"{val[:show_prefix]}...{val[-show_suffix:]}"

    def clear_cache(self) -> None:
        """Invalidates cached secrets (useful for test isolation or secret rotation)."""
        self._cache.clear()


# Default singleton secret manager
GLOBAL_SECRET_MANAGER = SecretManager()
