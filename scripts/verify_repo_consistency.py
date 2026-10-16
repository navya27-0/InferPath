"""Static repository consistency checker.

Verifies:
1. Version alignment across pyproject.toml, inferpath.__init__, Dockerfile, docker-compose.yml, and CHANGELOG.md.
2. Result artifacts exist, are non-empty, and contain valid JSON / Markdown.
3. Core documented Python modules exist and can be imported.
4. Relative file links in README.md and documentation exist on disk.
"""

from __future__ import annotations

import importlib
import json
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def check_version_consistency() -> list[str]:
    """Verify version numbers match across all project manifests and source files."""
    errors: list[str] = []

    # 1. pyproject.toml
    pyproject_file = REPO_ROOT / "pyproject.toml"
    if not pyproject_file.exists():
        return ["pyproject.toml does not exist"]

    pyproject_content = pyproject_file.read_text(encoding="utf-8")
    m = re.search(r'version\s*=\s*"([^"]+)"', pyproject_content)
    if not m:
        return ["version not found in pyproject.toml"]
    canonical_version = m.group(1)

    # 2. src/inferpath/__init__.py
    init_file = REPO_ROOT / "src" / "inferpath" / "__init__.py"
    if not init_file.exists():
        errors.append(f"{init_file} does not exist")
    else:
        init_content = init_file.read_text(encoding="utf-8")
        m_init = re.search(r'__version__\s*=\s*"([^"]+)"', init_content)
        if not m_init:
            errors.append(f"__version__ not found in {init_file}")
        elif m_init.group(1) != canonical_version:
            errors.append(
                f"Version mismatch in {init_file}: {m_init.group(1)} != {canonical_version}"
            )

    # 3. Dockerfile
    dockerfile = REPO_ROOT / "Dockerfile"
    if dockerfile.exists():
        df_content = dockerfile.read_text(encoding="utf-8")
        if f'version="{canonical_version}"' not in df_content:
            errors.append(
                f'Dockerfile missing org.opencontainers.image.version="{canonical_version}"'
            )
        if f"INFERPATH_VERSION={canonical_version}" not in df_content:
            errors.append(f"Dockerfile missing INFERPATH_VERSION={canonical_version}")

    # 4. docker-compose.yml
    dc_file = REPO_ROOT / "docker-compose.yml"
    if dc_file.exists():
        dc_content = dc_file.read_text(encoding="utf-8")
        if f":-{canonical_version}}}" not in dc_content:
            errors.append(f"docker-compose.yml missing default version :-{canonical_version}}}")

    # 5. CHANGELOG.md
    changelog_file = REPO_ROOT / "CHANGELOG.md"
    if changelog_file.exists():
        cl_content = changelog_file.read_text(encoding="utf-8")
        if f"## [{canonical_version}]" not in cl_content:
            errors.append(f"CHANGELOG.md missing release section for ## [{canonical_version}]")

    return errors


def check_result_artifacts() -> list[str]:
    """Verify that expected result directories and artifacts exist and parse cleanly."""
    errors: list[str] = []
    required_files = [
        REPO_ROOT / "data" / "pilot_dataset.json",
        REPO_ROOT / "configs" / "pricing_table_2026_10.yaml",
        REPO_ROOT / "data" / "results" / "simulated" / "benchmark_summary.json",
        REPO_ROOT / "data" / "results" / "simulated" / "benchmark_summary.md",
        REPO_ROOT / "data" / "results" / "simulated_test" / "benchmark_summary.json",
        REPO_ROOT / "data" / "results" / "simulated_test" / "benchmark_summary.md",
        REPO_ROOT / "data" / "results" / "fixtures" / "fixture_pilot_summary.json",
        REPO_ROOT / "data" / "results" / "fixtures" / "manifest_run_fixture_evaluation.json",
        REPO_ROOT / "data" / "results" / "fixtures" / "benchmark_summary.md",
        REPO_ROOT / "data" / "results" / "benchmarks" / "concurrency_benchmark.json",
    ]

    for p in required_files:
        if not p.exists():
            errors.append(f"Missing required artifact: {p.relative_to(REPO_ROOT)}")
        elif p.stat().st_size == 0:
            errors.append(f"Empty artifact: {p.relative_to(REPO_ROOT)}")
        elif p.suffix == ".json":
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
                if not isinstance(data, (dict, list)):
                    errors.append(f"Invalid JSON root in {p.relative_to(REPO_ROOT)}")
            except Exception as e:
                errors.append(f"Malformed JSON in {p.relative_to(REPO_ROOT)}: {e}")

    return errors


def check_core_modules() -> list[str]:
    """Verify core documented modules exist and import cleanly."""
    errors: list[str] = []
    modules = [
        "inferpath",
        "inferpath.pipeline",
        "inferpath.router",
        "inferpath.cache",
        "inferpath.security.privacy",
        "inferpath.security.secret_manager",
        "inferpath.services.rate_limiter",
        "inferpath.telemetry.structured_logging",
        "inferpath.telemetry.tracing",
        "inferpath.config.deployment_profiles",
        "inferpath.evaluation.router_eval",
        "inferpath.evaluation.harness",
    ]

    for mod in modules:
        try:
            importlib.import_module(mod)
        except Exception as e:
            errors.append(f"Failed to import core module {mod}: {e}")

    return errors


def check_documented_relative_links() -> list[str]:
    """Verify relative file links in README.md point to real existing files."""
    errors: list[str] = []
    readme_path = REPO_ROOT / "README.md"
    if not readme_path.exists():
        return ["README.md not found"]

    readme_content = readme_path.read_text(encoding="utf-8")
    # Match markdown links: [text](path) where path does not start with http/https/#/mailto
    links = re.findall(r"\[[^\]]+\]\(([^)]+)\)", readme_content)
    for link in links:
        # Strip potential query strings or anchor tags
        clean_link = link.split("#")[0].split("?")[0].strip()
        if not clean_link or clean_link.startswith(("http://", "https://", "mailto:")):
            continue
        # Convert forward slashes to local path
        target = REPO_ROOT / clean_link
        if not target.exists():
            errors.append(f"Broken link in README.md -> {clean_link}")

    return errors


def run_all_checks() -> dict[str, list[str]]:
    """Execute all static consistency checks and return report."""
    results: dict[str, list[str]] = {
        "version_consistency": check_version_consistency(),
        "result_artifacts": check_result_artifacts(),
        "core_modules": check_core_modules(),
        "documented_relative_links": check_documented_relative_links(),
    }
    return results


def main() -> int:
    report = run_all_checks()
    total_errors = sum(len(errs) for errs in report.values())

    print("=== InferPath Static Repository Consistency Check ===")
    for check_name, errs in report.items():
        if not errs:
            print(f"  [PASS] {check_name}")
        else:
            print(f"  [FAIL] {check_name} ({len(errs)} issues):")
            for e in errs:
                print(f"    - {e}")

    if total_errors > 0:
        print(f"\nConsistency check failed with {total_errors} issue(s).")
        return 1

    print("\nAll repository consistency checks passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
