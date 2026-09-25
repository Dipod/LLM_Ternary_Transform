"""Probe plan derived from the recorded cartography map (spec R7).

The transformation order and the removal fractions are read back from the map
rather than chosen when the probe runs, so the plan cannot be tuned after seeing
probe results.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

__all__ = ["ProbePlan", "derive_probe_plan"]


@dataclass(frozen=True)
class ProbePlan:
    """Frozen transformation plan for the combined probe.

    Attributes:
        map_id: Experiment id of the map the plan was derived from.
        ffn_fraction: Fraction of FFN intermediate dimensions to remove.
        head_fraction: Fraction of attention heads to remove.
        block_layers: Decoder layers removed by block skipping.
        notes: How each fraction was derived.
    """

    map_id: str
    ffn_fraction: float
    head_fraction: float
    block_layers: tuple[int, ...]
    notes: tuple[str, ...]


def _axis(payload: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return one axis entry from a map payload.

    Args:
        payload: Map ``metrics.json`` payload.
        name: Axis name.

    Returns:
        The axis mapping.

    Raises:
        KeyError: If the axis is missing.
    """
    axes = payload.get("axes")
    if not isinstance(axes, list):
        raise KeyError("map payload has no 'axes' list")
    for axis in axes:
        if isinstance(axis, Mapping) and axis.get("axis") == name:
            return axis
    raise KeyError(f"map payload has no axis {name!r}")


def _passing_fraction(axis: Mapping[str, Any], kind: str) -> tuple[float, float]:
    """Return the largest passing removal fraction of one unit kind.

    Args:
        axis: Axis entry.
        kind: Step kind, e.g. ``ffn_dim``.

    Returns:
        ``(fraction, budget)`` where ``fraction`` is 0.0 when the first
        checkpoint already exceeds the budget.
    """
    budget = float(axis.get("budget", 0.0))
    fraction = 0.0
    steps = axis.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if not isinstance(step, Mapping) or step.get("kind") != kind:
                continue
            extra = step.get("extra") or {}
            total = float(extra.get("n_total", 0.0))
            removed = float(step.get("removed", 0.0))
            if total <= 0.0:
                continue
            if float(step.get("kl", 0.0)) <= budget:
                fraction = max(fraction, removed / total)
    return fraction, budget


def derive_probe_plan(payload: Mapping[str, Any]) -> ProbePlan:
    """Derive the probe plan from a recorded cartography map.

    Args:
        payload: Map ``metrics.json`` payload.

    Returns:
        The frozen :class:`ProbePlan`.

    Raises:
        KeyError: If the structured axis is missing from the payload.
    """
    map_id = str(payload.get("experiment_id", "unknown"))
    structured = _axis(payload, "structured")
    ffn_fraction, ffn_budget = _passing_fraction(structured, "ffn_dim")
    head_fraction, head_budget = _passing_fraction(structured, "head")
    block_layers: list[int] = []
    steps = structured.get("steps")
    if isinstance(steps, list):
        for step in steps:
            if isinstance(step, Mapping) and step.get("kind") == "block":
                for unit in step.get("units", []):
                    text = str(unit)
                    if text.startswith("layer"):
                        block_layers.append(int(text.removeprefix("layer")))
    notes = (
        f"ffn_dim fraction {ffn_fraction:.4f} = largest cumulative checkpoint with "
        f"KL <= budget {ffn_budget} in map {map_id}",
        f"head fraction {head_fraction:.4f} under the same budget {head_budget}",
        "directions are recorded by the map but remove no stored bytes without a "
        "materialised low-rank factor, so they are excluded from the byte claim",
    )
    return ProbePlan(
        map_id=map_id,
        ffn_fraction=ffn_fraction,
        head_fraction=head_fraction,
        block_layers=tuple(sorted(set(block_layers))),
        notes=notes,
    )
