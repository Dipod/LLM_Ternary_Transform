"""Contract checks for the project scaffolding.

They guard assumptions that AGENTS.md declares about the environment: the Python
version floor and the communication mode read from ``.dev.env``.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
VALID_CAVEMAN_MODES = {"off", "auto", "on"}


def _read_dev_env() -> dict[str, str]:
    """Return the ``KEY=value`` pairs of ``.dev.env`` ignoring comments."""
    env_path = REPO_ROOT / ".dev.env"
    assert env_path.is_file(), f"missing {env_path}"
    values: dict[str, str] = {}
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        values[key.strip()] = value.strip()
    return values


def test_python_version_meets_project_minimum() -> None:
    """AGENTS.md -> Project info requires Python 3.10 or newer."""
    assert sys.version_info >= (3, 10), f"Python >= 3.10 required, got {sys.version.split()[0]}"


def test_caveman_mode_is_valid() -> None:
    """The caveman skill reads CAVEMAN from .dev.env; only off/auto/on are valid.

    An absent key defaults to ``auto``, matching the skill (CAVEMAN=auto).
    """
    mode = _read_dev_env().get("CAVEMAN", "auto")
    assert mode in VALID_CAVEMAN_MODES, f"CAVEMAN={mode!r} not in {sorted(VALID_CAVEMAN_MODES)}"
