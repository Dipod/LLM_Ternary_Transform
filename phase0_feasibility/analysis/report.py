"""Verdict and result recording for Phase 0 (design.md -> Metrics and gate).

The single canonical decision rule is fixed before the run and lives only here.
Every recorded run writes ``config.yaml``, ``metrics.json`` and ``report.md``
under ``<results_dir>/<experiment-id>/`` (.kilo/rules/research-discipline.md).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from phase0_feasibility.analysis.metrics import (
    GREEN_MAX_BPW,
    GREEN_MAX_E_X,
    GREEN_MIN_SPARSITY,
    YELLOW_MAX_BPW,
    YELLOW_MAX_E_X,
    LayerMetrics,
)
from phase0_feasibility.config import Config

VERDICT_GREEN = "GREEN"
VERDICT_YELLOW = "YELLOW"
VERDICT_RED = "RED"

PERPLEXITY_STATEMENT = (
    "End-to-end perplexity was not measured: a single converted projection out "
    "of many layers is not a sound end-to-end measurement (feasibility-gate spec -> "
    "Perplexity deferred). The verdict rests on layer-local metrics only."
)


def decide_verdict(metrics: LayerMetrics) -> str:
    """Apply the single canonical GREEN/YELLOW/RED rule.

    GREEN when ``E_x <= 0.05`` and ``sparsity >= 0.40`` and ``bpw_eff < 6.0``;
    YELLOW when ``E_x <= 0.10`` and ``bpw_eff < 8.0``; RED otherwise.

    Args:
        metrics: The layer metric set.

    Returns:
        One of ``"GREEN"``, ``"YELLOW"``, ``"RED"``.
    """
    if (
        metrics.e_x <= GREEN_MAX_E_X
        and metrics.sparsity >= GREEN_MIN_SPARSITY
        and metrics.bpw_eff < GREEN_MAX_BPW
    ):
        return VERDICT_GREEN
    if metrics.e_x <= YELLOW_MAX_E_X and metrics.bpw_eff < YELLOW_MAX_BPW:
        return VERDICT_YELLOW
    return VERDICT_RED


def verdict_from_metrics_dict(payload: dict[str, float]) -> str:
    """Recompute the verdict from a raw ``metrics.json`` metric mapping.

    Args:
        payload: Mapping with the keys produced by :meth:`LayerMetrics.as_dict`.

    Returns:
        The verdict string.
    """
    return decide_verdict(LayerMetrics(**payload))


def write_run(
    results_dir: str | Path,
    experiment_id: str,
    config: Config,
    metrics_fit: LayerMetrics,
    *,
    provenance: dict[str, Any],
    heldout_metrics: LayerMetrics | None = None,
    extra: dict[str, Any] | None = None,
    config_overrides: dict[str, Any] | None = None,
) -> Path:
    """Write one recorded run to ``<results_dir>/<experiment-id>/``.

    The verdict is computed from ``metrics_fit`` (``E_x`` on the fit portion),
    identically to the archived gate; ``heldout_metrics`` is recorded as a
    diagnostic only (design.md D6).

    Args:
        results_dir: Root results directory (repository-root ``results``).
        experiment_id: Run identifier; must not contain path separators.
        config: The configuration used for the run.
        metrics_fit: The fit-portion metric set; basis of the verdict.
        provenance: Environment and slice facts (model, dtype, slice, versions,
            hardware, seeds).
        heldout_metrics: Optional held-out metric set, recorded as a diagnostic.
        extra: Optional additional machine-readable fields (for example the
            contrast baseline metrics).
        config_overrides: Per-run values that override the base configuration in
            the recorded ``config.yaml`` (the swept ``mu``, ``tau``, ``reorder``
            and ``importance``), so the record carries every swept value.

    Returns:
        The path of the created run directory.

    Raises:
        ValueError: If ``experiment_id`` is empty or contains a path separator.
    """
    if not experiment_id or any(sep in experiment_id for sep in ("/", "\\")):
        raise ValueError(f"invalid experiment_id: {experiment_id!r}")
    run_dir = Path(results_dir) / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)

    verdict = decide_verdict(metrics_fit)
    payload: dict[str, Any] = {
        "experiment_id": experiment_id,
        "verdict": verdict,
        "verdict_basis": "metrics_fit",
        "metrics_fit": metrics_fit.as_dict(),
        "provenance": provenance,
    }
    if heldout_metrics is not None:
        payload["metrics_heldout"] = heldout_metrics.as_dict()
    if extra is not None:
        payload.update(extra)

    config_dict: dict[str, Any] = asdict(config)
    if config_overrides:
        config_dict.update(config_overrides)

    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8"
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (run_dir / "report.md").write_text(
        _report_markdown(
            experiment_id, verdict, metrics_fit, heldout_metrics, config_dict, provenance, extra
        ),
        encoding="utf-8",
    )
    return run_dir


def _report_markdown(
    experiment_id: str,
    verdict: str,
    metrics_fit: LayerMetrics,
    heldout_metrics: LayerMetrics | None,
    config_dict: dict[str, Any],
    provenance: dict[str, Any],
    extra: dict[str, Any] | None,
) -> str:
    runtime_ok = metrics_fit.runtime_s <= float(config_dict["runtime_budget_s"])
    lines = [
        f"# Phase 0 run — {experiment_id}",
        "",
        f"**Verdict: {verdict}**",
        "",
        "The verdict is computed from `E_x_fit` on the fit portion under the "
        "unchanged gate rule; `E_x_heldout` is a diagnostic and does not drive "
        "the verdict (design.md D6).",
        "",
        "## Metrics",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Output error `E_x_fit` (verdict basis) | {metrics_fit.e_x:.6f} |",
    ]
    if heldout_metrics is not None:
        lines.append(
            "| Output error `E_x_heldout` (diagnostic, not the verdict basis) | "
            f"{heldout_metrics.e_x:.6f} |"
        )
    lines += [
        f"| Energy preserved | {metrics_fit.energy:.6f} |",
        f"| Sparsity (zeros in B, C) | {metrics_fit.sparsity:.6f} |",
        f"| Effective BPW | {metrics_fit.bpw_eff:.6f} |",
        f"| Runtime (s) | {metrics_fit.runtime_s:.3f} |",
        "",
        f"Runtime budget of {float(config_dict['runtime_budget_s']):.0f} s: "
        f"{'met' if runtime_ok else 'NOT met'}.",
        "",
        "## Configuration (run parameters included)",
        "",
        "```yaml",
        yaml.safe_dump(config_dict, sort_keys=False).rstrip(),
        "```",
        "",
        "## Provenance",
        "",
    ]
    for key in sorted(provenance):
        lines.append(f"- `{key}`: {provenance[key]}")
    if extra:
        lines += ["", "## Additional fields", "", "```json", json.dumps(extra, indent=2), "```"]
    lines += ["", "## Interpretation", "", PERPLEXITY_STATEMENT, ""]
    return "\n".join(lines)
