"""Quality-bar and decision-rule tests (tasks 2.4, 4.3)."""

from __future__ import annotations

import pytest

from phase1_compression.config import Gate
from phase1_compression.evaluation.gate import (
    VERDICT_GO,
    VERDICT_RED,
    VERDICT_YELLOW,
    Candidate,
    QualityResult,
    compression_ratio,
    decide_verdict,
    harness_is_valid,
    quality_bar,
    relative_ppl_delta,
    smallest_matched_quality_bytes,
)


def gate() -> Gate:
    return Gate(
        ppl_rel_tol=0.005,
        bench_delta_tol=0.5,
        go_bytes_ratio=0.9,
        yellow_bytes_ratio=1.1,
        min_compression=2.0,
    )


def quality(passed: bool, name: str) -> QualityResult:
    return QualityResult(
        ppl_rel_delta=0.001 if passed else 0.02,
        bench_deltas={"arc_easy": 0.0 if passed else -3.0},
        passed=passed,
        failed_on=() if passed else (name,),
    )


def test_relative_ppl_delta_known_answer() -> None:
    assert relative_ppl_delta(10.0, 11.0) == pytest.approx(0.1)
    with pytest.raises(ValueError, match="positive"):
        relative_ppl_delta(0.0, 1.0)


def test_quality_bar_passes_within_tolerance() -> None:
    result = quality_bar(10.0, 10.04, {"a": 50.0}, {"a": 49.6}, gate())
    assert result.passed
    assert result.ppl_rel_delta == pytest.approx(0.004)


def test_quality_bar_fails_on_ppl_alone() -> None:
    result = quality_bar(10.0, 10.06, {"a": 50.0}, {"a": 50.0}, gate())
    assert not result.passed
    assert any("ppl_rel_delta" in item for item in result.failed_on)


def test_quality_bar_fails_on_one_benchmark() -> None:
    result = quality_bar(10.0, 10.01, {"a": 50.0, "b": 40.0}, {"a": 49.4, "b": 40.0}, gate())
    assert not result.passed
    assert any("a_delta" in item for item in result.failed_on)


def test_quality_bar_requires_all_tasks() -> None:
    with pytest.raises(ValueError, match="missing tasks"):
        quality_bar(10.0, 10.0, {"a": 1.0, "b": 2.0}, {"a": 1.0}, gate())


def test_smallest_matched_quality_bytes_is_not_best_quality() -> None:
    candidates = [
        Candidate("q4", quality(False, "q4"), 5_000),
        Candidate("q5", quality(True, "q5"), 7_000),
        Candidate("q6", quality(True, "q6"), 9_000),
    ]
    assert smallest_matched_quality_bytes(candidates, gate()) == 7_000
    assert (
        smallest_matched_quality_bytes([Candidate("q4", quality(False, "q4"), 5)], gate()) is None
    )


def test_compression_ratio() -> None:
    assert compression_ratio(100, 50) == 2.0
    with pytest.raises(ValueError, match="compressed_bytes"):
        compression_ratio(100, 0)


def test_verdict_go_below_nine_tenths() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(True, "m"), 8_900)],
        reference_bytes=20_000,
        baseline_bytes=10_000,
        gate=gate(),
    )
    assert verdict.label == VERDICT_GO


def test_verdict_yellow_between_ratios() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(True, "m"), 9_500)],
        reference_bytes=20_000,
        baseline_bytes=10_000,
        gate=gate(),
    )
    assert verdict.label == VERDICT_YELLOW


def test_verdict_red_above_ratio() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(True, "m"), 12_000)],
        reference_bytes=20_000,
        baseline_bytes=10_000,
        gate=gate(),
    )
    assert verdict.label == VERDICT_RED


def test_verdict_red_below_min_compression() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(True, "m"), 20_000)],
        reference_bytes=20_000,
        baseline_bytes=None,
        gate=gate(),
    )
    assert verdict.label == VERDICT_RED


def test_verdict_red_when_nothing_passes() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(False, "m"), 1_000)],
        reference_bytes=20_000,
        baseline_bytes=10_000,
        gate=gate(),
    )
    assert verdict.label == VERDICT_RED


def test_verdict_go_without_matched_quality_baseline() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(True, "m"), 1_000)],
        reference_bytes=20_000,
        baseline_bytes=None,
        gate=gate(),
    )
    assert verdict.label == VERDICT_GO


def test_harness_sanity_check() -> None:
    valid, delta = harness_is_valid(10.0, 10.008, 0.001)
    assert valid and delta == pytest.approx(0.0008)
    valid, delta = harness_is_valid(10.0, 10.02, 0.001)
    assert not valid and delta == pytest.approx(0.002)


def test_rule_is_sensitive_to_every_threshold() -> None:
    from dataclasses import replace

    base = gate()

    def label(compressed: int, g: Gate, *, reference: int = 40_000, baseline: int = 10_000) -> str:
        return decide_verdict(
            [Candidate("m", quality(True, "m"), compressed)],
            reference_bytes=reference,
            baseline_bytes=baseline,
            gate=g,
        ).label

    # reference 40,000 keeps compression (4.5x) above the floor everywhere below.
    assert label(8_900, base) == VERDICT_GO
    assert label(8_900, replace(base, go_bytes_ratio=0.8)) == VERDICT_YELLOW
    assert label(10_500, base) == VERDICT_YELLOW
    assert label(10_500, replace(base, yellow_bytes_ratio=1.0)) == VERDICT_RED
    # min_compression isolated by removing the baseline comparison.
    assert (
        decide_verdict(
            [Candidate("m", quality(True, "m"), 19_000)],
            reference_bytes=20_000,
            baseline_bytes=None,
            gate=base,
        ).label
        == VERDICT_RED
    )
    assert (
        decide_verdict(
            [Candidate("m", quality(True, "m"), 19_000)],
            reference_bytes=20_000,
            baseline_bytes=None,
            gate=replace(base, min_compression=1.0),
        ).label
        == VERDICT_GO
    )


def test_healing_caps_go_at_yellow() -> None:
    verdict = decide_verdict(
        [Candidate("m", quality(True, "m"), 8_000, only_after_healing=True)],
        reference_bytes=20_000,
        baseline_bytes=10_000,
        gate=gate(),
    )
    assert verdict.label == VERDICT_YELLOW
    assert "healing" in verdict.reason
