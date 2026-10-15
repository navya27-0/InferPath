"""Test-set lock and experiment configuration registration and verification."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class FrozenExperimentConfig(BaseModel):
    """Immutable registered experiment protocol configuration for test evaluation."""

    config_version: str = "1.0.0"
    config_hash: str = Field(description="SHA-256 hash of configuration parameters")
    q_min: float = 0.70
    dataset_version: str = "1.0.0"
    dataset_sha256: str
    price_table_date: str = "2026-10-01"
    carbon_config_date: str = "2026-10-01"
    cache_policy_version: str = "1.0.0"
    rubric_version: str = "1.0.0"
    small_model_id: str = "google/gemini-2.5-flash"
    large_model_id: str = "anthropic/claude-3.7-sonnet"
    registered_at: str
    registered_commit: str


class ConfigRegistry(BaseModel):
    """Persistent registry of authorized frozen test configurations."""

    registered_configs: dict[str, FrozenExperimentConfig] = Field(default_factory=dict)


def compute_config_hash(config_dict: dict[str, Any] | None = None) -> str:
    """Computes a deterministic SHA-256 hash over canonical JSON representation."""
    if config_dict is None:
        config_dict = {
            "q_min": 0.70,
            "dataset_version": "1.0.0",
            "price_table_date": "2026-10-01",
            "carbon_config_date": "2026-10-01",
            "cache_policy_version": "1.0.0",
            "rubric_version": "1.0.0",
            "small_model_id": "google/gemini-2.5-flash",
            "large_model_id": "anthropic/claude-3.7-sonnet",
        }
    canonical = json.dumps(config_dict, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def get_git_commit(cwd: Path | str | None = None, ignore_results: bool = True) -> str:
    """Retrieves current HEAD commit hash via git rev-parse, appending -dirty if uncommitted."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        commit = res.stdout.strip()
        if not check_git_tree_clean(cwd, ignore_results=ignore_results):
            commit = f"{commit}-dirty"
        return commit
    except Exception:
        return "uncommitted_git_fallback"


def check_git_tree_clean(cwd: Path | str | None = None, ignore_results: bool = True) -> bool:
    """Returns True if the git working tree has no uncommitted changes (optionally ignoring results)."""
    try:
        res = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd or Path.cwd(),
            capture_output=True,
            text=True,
            check=True,
        )
        raw_lines = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        if not ignore_results:
            return len(raw_lines) == 0

        filtered: list[str] = []
        for line in raw_lines:
            parts = line.split(maxsplit=1)
            if len(parts) < 2:
                continue
            path_str = parts[1].strip('"').replace("\\", "/")
            if (
                path_str.startswith("data/results/")
                or path_str.startswith("data/storage/")
                or path_str.startswith("data/logs/")
                or path_str.startswith("docs/img/")
            ):
                continue
            filtered.append(line)
        return len(filtered) == 0
    except Exception:
        return False


def get_registry_path() -> Path:
    return Path("configs") / "experiments" / "registered_test_configs.json"


def load_registry(registry_path: Path | None = None) -> ConfigRegistry:
    path = registry_path or get_registry_path()
    if not path.exists():
        return ConfigRegistry()
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return ConfigRegistry(**data)


def save_registry(registry: ConfigRegistry, registry_path: Path | None = None) -> None:
    path = registry_path or get_registry_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry.model_dump(mode="json"), f, indent=2, ensure_ascii=False)


def register_frozen_config(
    q_min: float = 0.70,
    dataset_path: Path | None = None,
    registry_path: Path | None = None,
    description: str = "",
    force: bool = False,
) -> FrozenExperimentConfig:
    """Computes and registers the frozen test experiment configuration hash."""
    from datetime import UTC, datetime

    from inferpath.evaluation.dataset import load_pilot_dataset

    ds = load_pilot_dataset(dataset_path)
    ds_hash = ds.compute_sha256()
    commit = get_git_commit()

    config_core = {
        "q_min": q_min,
        "dataset_version": ds.version,
        "dataset_sha256": ds_hash,
        "price_table_date": "2026-10-01",
        "carbon_config_date": "2026-10-01",
        "cache_policy_version": "1.0.0",
        "rubric_version": "1.0.0",
        "small_model_id": "google/gemini-2.5-flash",
        "large_model_id": "anthropic/claude-3.7-sonnet",
    }
    cfg_hash = compute_config_hash(config_core)

    frozen_cfg = FrozenExperimentConfig(
        config_version="1.0.0",
        config_hash=cfg_hash,
        q_min=q_min,
        dataset_version=ds.version,
        dataset_sha256=ds_hash,
        price_table_date="2026-10-01",
        carbon_config_date="2026-10-01",
        cache_policy_version="1.0.0",
        rubric_version="1.0.0",
        small_model_id="google/gemini-2.5-flash",
        large_model_id="anthropic/claude-3.7-sonnet",
        registered_at=datetime.now(UTC).isoformat(),
        registered_commit=commit,
    )

    registry = load_registry(registry_path=registry_path)
    registry.registered_configs[cfg_hash] = frozen_cfg
    save_registry(registry, registry_path=registry_path)

    return frozen_cfg


def verify_test_config_locked(
    config_hash: str,
    require_clean_tree: bool = True,
    registry_path: Path | None = None,
) -> tuple[bool, str]:
    """
    Verifies that:
    1. Git working tree is completely clean (if require_clean_tree is True).
    2. Configuration hash is registered in registered_test_configs.json.
    """
    if require_clean_tree and not check_git_tree_clean():
        return (
            False,
            "Git working tree is dirty! Test evaluation is locked against uncommitted changes.",
        )

    registry = load_registry(registry_path=registry_path)
    if config_hash not in registry.registered_configs:
        return (
            False,
            f"Configuration hash '{config_hash}' is not registered! Test evaluation requires pre-registered frozen protocol.",
        )

    return True, "Configuration verified and locked."


def verify_test_set_lock(
    split: str,
    config_hash: str,
    registry_path: Path | None = None,
    allow_dirty_git: bool = False,
) -> None:
    """Verifies test-set lock: dev is allowed unlocked; test requires registered hash and clean tree."""
    if split != "test":
        return
    ok, msg = verify_test_config_locked(
        config_hash=config_hash,
        require_clean_tree=not allow_dirty_git,
        registry_path=registry_path,
    )
    if not ok:
        raise RuntimeError(f"Test evaluation lock check failed: {msg}")


def main():
    parser = argparse.ArgumentParser(
        description="Register and freeze InferPath test experiment configuration."
    )
    parser.add_argument(
        "--register", action="store_true", help="Register current configuration hash"
    )
    parser.add_argument("--q-min", type=float, default=0.70, help="Frozen Q_min threshold")
    parser.add_argument("--verify", type=str, help="Verify specific configuration hash")
    args = parser.parse_args()

    if args.register:
        cfg = register_frozen_config(q_min=args.q_min)
        print(f"Registered frozen test config hash: {cfg.config_hash}")
        print(f"Dataset SHA-256: {cfg.dataset_sha256}")
    elif args.verify:
        ok, msg = verify_test_config_locked(args.verify, require_clean_tree=False)
        print(f"Verification: {ok} ({msg})")


if __name__ == "__main__":
    main()
