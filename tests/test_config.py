"""Contract checks for the single config loader (tasks 1.2-1.3).

The loader must return the expected frozen dataclass for the checked-in
``config.yaml`` and must consume every key in that file, so no algorithm can
carry a stray hardcoded default (AGENTS.md -> Python standards -> Configuration).
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from phase0_feasibility.config import Config, Sweep, Tolerance, load_config

CONFIG_PATH = Path(__file__).resolve().parents[1] / "phase0_feasibility" / "config.yaml"
RECHECK_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "phase0_feasibility" / "config_recheck.yaml"
)


def _flatten(mapping: dict[str, Any], prefix: str = "") -> set[str]:
    """Return dotted leaf-key paths of a nested mapping (dicts recurse only)."""
    keys: set[str] = set()
    for key, value in mapping.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            keys |= _flatten(value, f"{path}.")
        else:
            keys.add(path)
    return keys


def test_load_config_returns_expected_dataclass() -> None:
    """The packaged config parses into the typed Config structure."""
    config = load_config(CONFIG_PATH)
    assert isinstance(config, Config)
    assert isinstance(config.tolerance, Tolerance)
    assert isinstance(config.sweep, Sweep)
    assert config.model_id == "Qwen/Qwen2.5-0.5B"
    assert config.layers == (12,)
    assert config.target_module == "gate_proj"
    assert config.slices == ("docs128",)
    assert config.dense_windows_count == 128
    assert config.many_documents_count == 1024
    assert config.eval_fraction == 0.2
    assert config.sweep.mu == (2, 3)
    assert config.sweep.tau == (0.7, 1.0)


def test_recheck_config_parses_into_the_same_schema() -> None:
    """The re-check config selects the new slices, three layers and its own output dir."""
    config = load_config(RECHECK_CONFIG_PATH)
    assert config.layers == (3, 12, 21)
    assert config.slices == ("dense_windows", "many_documents")
    assert config.eval_fraction == 0.2
    assert config.results_dir == "results/recheck"


def test_every_recheck_yaml_key_is_consumed_by_the_loader() -> None:
    """The re-check config also consumes every key it declares."""
    raw = yaml.safe_load(RECHECK_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    config = load_config(RECHECK_CONFIG_PATH)
    assert _flatten(asdict(config)) == _flatten(raw)


def test_every_yaml_key_is_consumed_by_the_loader() -> None:
    """No YAML key is ignored and no dataclass field is a hidden default."""
    raw = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    config = load_config(CONFIG_PATH)
    assert _flatten(asdict(config)) == _flatten(raw)
