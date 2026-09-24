"""Tests for the verdict rule and the result writer (tasks 4.2, 4.3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from phase0_feasibility.analysis.metrics import LayerMetrics
from phase0_feasibility.analysis.report import (
    VERDICT_GREEN,
    VERDICT_RED,
    VERDICT_YELLOW,
    decide_verdict,
    verdict_from_metrics_dict,
    write_run,
)
from phase0_feasibility.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "phase0_feasibility" / "config.yaml"


def _metrics(e_x: float, sparsity: float, bpw: float) -> LayerMetrics:
    return LayerMetrics(e_x=e_x, energy=0.9, sparsity=sparsity, bpw_eff=bpw, runtime_s=1.0)


def test_verdict_green() -> None:
    assert decide_verdict(_metrics(0.03, 0.5, 5.0)) == VERDICT_GREEN


def test_verdict_yellow_when_sparsity_fails() -> None:
    assert decide_verdict(_metrics(0.08, 0.2, 7.0)) == VERDICT_YELLOW


def test_verdict_red_on_high_error() -> None:
    assert decide_verdict(_metrics(0.3, 0.9, 2.0)) == VERDICT_RED


def test_verdict_red_on_high_bpw() -> None:
    assert decide_verdict(_metrics(0.01, 0.9, 9.0)) == VERDICT_RED


def test_verdict_is_one_of_three() -> None:
    for metrics in (_metrics(0.03, 0.5, 5.0), _metrics(0.08, 0.2, 7.0), _metrics(1.0, 0.0, 12.0)):
        assert decide_verdict(metrics) in {VERDICT_GREEN, VERDICT_YELLOW, VERDICT_RED}


def test_write_run_round_trips_all_three_files(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    metrics = _metrics(0.04, 0.55, 5.5)
    provenance = {"model_id": config.model_id, "seed": config.seed, "device": "cpu"}
    run_dir = write_run(tmp_path, "unit-test-run", config, metrics, provenance=provenance)

    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    assert (run_dir / "report.md").is_file()

    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["verdict"] == VERDICT_GREEN
    assert payload["verdict_basis"] == "metrics_fit"
    assert verdict_from_metrics_dict(payload["metrics_fit"]) == payload["verdict"]
    assert payload["metrics_fit"]["e_x"] == pytest.approx(0.04)

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert VERDICT_GREEN in report
    assert "E_x_fit" in report
    assert "perplexity" in report.lower()
    assert "not measured" in report


def test_write_run_records_heldout_as_diagnostic(tmp_path: Path) -> None:
    """The held-out block is recorded and named a diagnostic, not the verdict basis."""
    config = load_config(CONFIG_PATH)
    fit = _metrics(0.04, 0.55, 5.5)
    heldout = _metrics(0.09, 0.55, 5.5)
    provenance = {"model_id": config.model_id, "seed": config.seed, "device": "cpu"}
    run_dir = write_run(
        tmp_path,
        "unit-test-heldout",
        config,
        fit,
        heldout_metrics=heldout,
        provenance=provenance,
    )

    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["metrics_fit"]["e_x"] == pytest.approx(0.04)
    assert payload["metrics_heldout"]["e_x"] == pytest.approx(0.09)
    assert payload["verdict"] == decide_verdict(fit)

    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "E_x_fit" in report
    assert "E_x_heldout" in report
    assert "verdict basis" in report


def test_write_run_rejects_bad_experiment_id(tmp_path: Path) -> None:
    config = load_config(CONFIG_PATH)
    with pytest.raises(ValueError, match="experiment_id"):
        write_run(tmp_path, "bad/id", config, _metrics(0.1, 0.1, 1.0), provenance={})
