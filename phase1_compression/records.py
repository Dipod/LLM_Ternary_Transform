"""Run records for Phase 1 Stage A.

Every recorded run uses the three-file layout of
``.kilo/rules/research-discipline.md``: ``config.yaml``, ``metrics.json``,
``report.md``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml


def write_run(
    results_dir: Path,
    experiment_id: str,
    *,
    config_dump: Mapping[str, Any],
    metrics: Mapping[str, Any],
    provenance: Mapping[str, Any],
    report: str,
) -> Path:
    """Write one recorded run.

    Args:
        results_dir: Repository ``results`` directory.
        experiment_id: Run id; becomes the directory name.
        config_dump: Configuration values that influenced the run.
        metrics: Raw metrics, machine-readable.
        provenance: Environment and slice facts.
        report: Markdown report body.

    Returns:
        The created run directory.
    """
    run_dir = results_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(dict(config_dump), sort_keys=False), encoding="utf-8"
    )
    payload = {"experiment_id": experiment_id, "provenance": dict(provenance), **dict(metrics)}
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(report, encoding="utf-8")
    return run_dir


def read_metrics(results_dir: Path, experiment_id: str) -> dict[str, Any]:
    """Read a recorded run's metrics.

    Args:
        results_dir: Repository ``results`` directory.
        experiment_id: Run id.

    Returns:
        The parsed ``metrics.json`` payload.

    Raises:
        FileNotFoundError: If the run does not exist.
    """
    path = results_dir / experiment_id / "metrics.json"
    if not path.is_file():
        raise FileNotFoundError(f"no recorded run at {path}")
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"metrics.json root must be a mapping, got {type(payload).__name__}")
    return payload
