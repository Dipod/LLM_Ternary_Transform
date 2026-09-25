"""Record layout for the redundancy map (spec phase1/redundancy-cartography R7).

The map is written with the standard three-file run layout before any probe exists
so the transformation order cannot be chosen after seeing probe results.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from phase1_compression.config import Config


@dataclass(frozen=True)
class Step:
    """One recorded removal checkpoint of a cartography axis.

    Attributes:
        kind: Unit kind, e.g. ``block``, ``head``, ``ffn_dim``.
        units: Removed unit identifiers at this checkpoint.
        removed: Cumulative number of removed units.
        kl: KL divergence in nats at this checkpoint.
        extra: Axis-specific numbers, e.g. ``e_x`` or ``zero_fraction``.
    """

    kind: str
    units: tuple[str, ...]
    removed: int
    kl: float
    extra: Mapping[str, float]


@dataclass(frozen=True)
class AxisMap:
    """Recorded result of one cartography axis.

    Attributes:
        axis: Axis name.
        proxy: Primary proxy metric name.
        budget: Proxy budget the axis was scored against.
        steps: Recorded checkpoints, in removal order.
        stopped_on_budget: Whether the axis stopped on its time budget.
        notes: Deviations and interpretation notes.
        e_x_cross_check: Layer-local E_x cross-check at the last checkpoint.
    """

    axis: str
    proxy: str
    budget: float
    steps: tuple[Step, ...]
    stopped_on_budget: bool
    notes: tuple[str, ...]
    e_x_cross_check: float | None


def write_map(
    results_dir: Path,
    map_id: str,
    axes: Sequence[AxisMap],
    *,
    config: Config,
    provenance: Mapping[str, Any],
) -> Path:
    """Write the cartography map in the standard run layout.

    Args:
        results_dir: Repository ``results`` directory.
        map_id: Experiment id of the map.
        axes: Axis results in the order they were measured.
        config: Run configuration, copied into the run directory.
        provenance: Environment and slice facts.

    Returns:
        The created run directory.
    """
    run_dir = results_dir / map_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(_config_dump(config), sort_keys=False), encoding="utf-8"
    )
    payload = {
        "experiment_id": map_id,
        "provenance": dict(provenance),
        "axes": [
            {
                "axis": axis.axis,
                "proxy": axis.proxy,
                "budget": axis.budget,
                "stopped_on_budget": axis.stopped_on_budget,
                "e_x_cross_check": axis.e_x_cross_check,
                "notes": list(axis.notes),
                "steps": [
                    {
                        "kind": step.kind,
                        "units": list(step.units),
                        "removed": step.removed,
                        "kl": step.kl,
                        "extra": dict(step.extra),
                    }
                    for step in axis.steps
                ],
            }
            for axis in axes
        ],
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(_report(map_id, axes, provenance), encoding="utf-8")
    return run_dir


def _config_dump(config: Config) -> dict[str, Any]:
    """Return a JSON-serialisable view of a config section for the record.

    Args:
        config: Loaded configuration.

    Returns:
        A flat mapping of the values that influenced the map.
    """
    return {
        "cartography_model_id": config.cartography_model_id,
        "target_model_id": config.target_model_id,
        "dtype": config.dtype,
        "device": config.device,
        "seed": config.seed,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "calibration_split": config.calibration_split,
        "seq_len": config.seq_len,
        "dense_windows_count": config.dense_windows_count,
        "cartography": {
            "structured": {
                "kinds": list(config.cartography.structured.kinds),
                "kl_budget": config.cartography.structured.kl_budget,
                "max_removals": config.cartography.structured.max_removals,
            },
            "directions": {
                "n_directions": list(config.cartography.directions.n_directions),
                "layer_bands": [list(band) for band in config.cartography.directions.layer_bands],
                "kl_budget": config.cartography.directions.kl_budget,
            },
            "ternary": {
                "layer": config.cartography.ternary.layer,
                "mu": list(config.cartography.ternary.mu),
                "tau": config.cartography.ternary.tau,
                "nm_patterns": [list(p) for p in config.cartography.ternary.nm_patterns],
                "rank_fractions": list(config.cartography.ternary.rank_fractions),
                "e_x_budget": config.cartography.ternary.e_x_budget,
            },
            "budget_gpu_hours_per_axis": config.cartography.budget_gpu_hours_per_axis,
        },
    }


def _report(map_id: str, axes: Sequence[AxisMap], provenance: Mapping[str, Any]) -> str:
    """Render the human-readable map report.

    Args:
        map_id: Experiment id.
        axes: Axis results.
        provenance: Environment and slice facts.

    Returns:
        Markdown report text.
    """
    lines = [
        f"# Redundancy map `{map_id}` (Phase 1 Stage A)",
        "",
        "Cartography of removable structure before the combined probe "
        "(spec phase1/redundancy-cartography). No probe result existed when this "
        "map was written.",
        "",
        "## Provenance",
        "",
    ]
    lines += [f"- {key}: {value}" for key, value in provenance.items()]
    lines += ["", "## Axes", ""]
    for axis in axes:
        lines += [
            f"### {axis.axis}",
            "",
            f"- proxy: `{axis.proxy}`, budget: {axis.budget}",
            f"- stopped on budget: {axis.stopped_on_budget}",
            f"- E_x cross-check at the last checkpoint: {axis.e_x_cross_check}",
        ]
        lines += [f"- note: {note}" for note in axis.notes]
        lines += ["", "| removed | kind | KL | extra |", "|---|---|---|---|"]
        for step in axis.steps:
            extra = ", ".join(f"{k}={v:.4g}" for k, v in step.extra.items())
            lines.append(f"| {step.removed} | {step.kind} | {step.kl:.4f} | {extra} |")
        lines.append("")
    return "\n".join(lines) + "\n"
