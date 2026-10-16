"""Unit test executing static repository consistency checks."""

import sys
from pathlib import Path

# Ensure repo root is on sys.path for scripts import
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.verify_repo_consistency import run_all_checks  # noqa: E402


def test_repository_static_consistency():
    """Verify repository metadata, artifacts, core modules, and links are completely consistent."""
    report = run_all_checks()
    for check_name, errors in report.items():
        assert not errors, f"Consistency check failed on '{check_name}': {errors}"
