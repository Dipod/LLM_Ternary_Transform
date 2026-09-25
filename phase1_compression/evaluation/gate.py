"""Stage A quality bar and the single GO / YELLOW / RED decision rule.

Spec: ``phase1/compression-gate`` R1 (quality bar), R3 (matched-quality baseline),
R6/R7 (healing attribution and the decision rule). Thresholds come from
``config.gate`` and are fixed before the run.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from phase1_compression.config import Gate

VERDICT_GO = "GO"
VERDICT_YELLOW = "YELLOW"
VERDICT_RED = "RED"


@dataclass(frozen=True)
class QualityResult:
    """Outcome of the strict quality bar for one configuration.

    Attributes:
        ppl_rel_delta: Relative perplexity increase versus the BF16 reference.
        bench_deltas: Benchmark delta in points per task, compressed minus reference.
        passed: Whether the configuration satisfies both conditions.
        failed_on: Names of the conditions that failed, empty when ``passed``.
    """

    ppl_rel_delta: float
    bench_deltas: Mapping[str, float]
    passed: bool
    failed_on: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    """One scored configuration offered to the decision rule.

    Attributes:
        name: Configuration identifier.
        quality: Quality-bar outcome.
        stored_bytes: Stored bytes of the compressed model.
        only_after_healing: Whether the configuration passes only after healing.
    """

    name: str
    quality: QualityResult
    stored_bytes: int
    only_after_healing: bool = False


@dataclass(frozen=True)
class Verdict:
    """The single Stage A verdict.

    Attributes:
        label: ``GO``, ``YELLOW`` or ``RED``.
        reason: One-line explanation naming the deciding numbers.
    """

    label: str
    reason: str


def relative_ppl_delta(reference_ppl: float, candidate_ppl: float) -> float:
    """Return the relative perplexity increase over the reference.

    Args:
        reference_ppl: BF16 reference perplexity, strictly positive.
        candidate_ppl: Compressed-model perplexity.

    Returns:
        ``(candidate - reference) / reference``.

    Raises:
        ValueError: If the reference perplexity is not strictly positive.
    """
    if reference_ppl <= 0.0:
        raise ValueError(f"reference perplexity must be positive, got {reference_ppl}")
    return (candidate_ppl - reference_ppl) / reference_ppl


def quality_bar(
    reference_ppl: float,
    candidate_ppl: float,
    reference_bench: Mapping[str, float],
    candidate_bench: Mapping[str, float],
    gate: Gate,
) -> QualityResult:
    """Apply the strict ±0.5 quality bar (spec R1).

    Args:
        reference_ppl: BF16 reference perplexity.
        candidate_ppl: Compressed-model perplexity.
        reference_bench: BF16 benchmark scores, in points.
        candidate_bench: Compressed-model benchmark scores for the same tasks.
        gate: Gate thresholds.

    Returns:
        The :class:`QualityResult`.

    Raises:
        ValueError: If a benchmark task is missing on either side.
    """
    missing = sorted(set(reference_bench) - set(candidate_bench))
    if missing:
        raise ValueError(f"candidate benchmark scores missing tasks: {missing}")
    ppl_delta = relative_ppl_delta(reference_ppl, candidate_ppl)
    deltas = {
        task: float(candidate_bench[task]) - float(reference_bench[task])
        for task in reference_bench
    }
    failed: list[str] = []
    if not ppl_delta <= gate.ppl_rel_tol:
        failed.append(f"ppl_rel_delta={ppl_delta:+.5f} > {gate.ppl_rel_tol}")
    for task, delta in deltas.items():
        if not delta >= -gate.bench_delta_tol:
            failed.append(f"{task}_delta={delta:+.3f} < {-gate.bench_delta_tol}")
    return QualityResult(
        ppl_rel_delta=ppl_delta,
        bench_deltas=deltas,
        passed=not failed,
        failed_on=tuple(failed),
    )


def compression_ratio(reference_bytes: int, compressed_bytes: int) -> float:
    """Return the compression ratio versus the reference model.

    Args:
        reference_bytes: Stored bytes of the BF16 reference model.
        compressed_bytes: Stored bytes of the compressed model.

    Returns:
        ``reference_bytes / compressed_bytes``.

    Raises:
        ValueError: If either size is not strictly positive.
    """
    if reference_bytes <= 0:
        raise ValueError(f"reference_bytes must be positive, got {reference_bytes}")
    if compressed_bytes <= 0:
        raise ValueError(f"compressed_bytes must be positive, got {compressed_bytes}")
    return reference_bytes / compressed_bytes


def harness_is_valid(
    reference_ppl: float, sanity_ppl: float, tolerance: float
) -> tuple[bool, float]:
    """Apply the 8-bit sanity check of spec R4.

    Args:
        reference_ppl: BF16 reference perplexity.
        sanity_ppl: Perplexity of the 8-bit group-128 baseline.
        tolerance: Maximum relative perplexity increase accepted.

    Returns:
        ``(valid, delta)`` where ``delta`` is the relative increase; when ``valid``
        is ``False`` the run must be recorded as void and no verdict issued.
    """
    delta = relative_ppl_delta(reference_ppl, sanity_ppl)
    return delta <= tolerance, delta


def smallest_matched_quality_bytes(candidates: Sequence[Candidate], gate: Gate) -> int | None:
    """Return the smallest stored bytes among candidates meeting the quality bar.

    This is the matched-quality integer baseline of spec R3: the strongest
    (smallest-byte) configuration that achieves the same quality, regardless of
    which candidate had the best quality.

    Args:
        candidates: Scored configurations, typically the integer baselines.
        gate: Gate thresholds; only the quality bar is used here.

    Returns:
        The smallest stored bytes among passing candidates, or ``None`` when no
        candidate meets the quality bar.
    """
    passing = [c for c in candidates if c.quality.passed]
    if not passing:
        return None
    return min(c.stored_bytes for c in passing)


def decide_verdict(
    candidates: Sequence[Candidate],
    *,
    reference_bytes: int,
    baseline_bytes: int | None,
    gate: Gate,
) -> Verdict:
    """Apply the single Stage A decision rule (spec R7).

    GO when a configuration meets the quality bar with stored bytes at most
    ``go_bytes_ratio`` of the matched-quality baseline and at least
    ``min_compression`` versus the reference; YELLOW when it meets the quality bar
    within ``yellow_bytes_ratio`` of that baseline; RED otherwise. A configuration
    that meets the bar only after healing scores YELLOW at best (spec R6).

    Args:
        candidates: Every scored configuration of the stage.
        reference_bytes: Stored bytes of the BF16 reference model.
        baseline_bytes: Smallest stored bytes among matched-quality integer
            baselines, or ``None`` when no baseline meets the quality bar.
        gate: Gate thresholds.

    Returns:
        The :class:`Verdict`.

    Raises:
        ValueError: If a candidate size is not positive.
    """
    passing = [c for c in candidates if c.quality.passed]
    if not passing:
        return Verdict(VERDICT_RED, "no configuration meets the quality bar")
    best = min(passing, key=lambda c: c.stored_bytes)
    ratio = compression_ratio(reference_bytes, best.stored_bytes)
    if ratio < gate.min_compression:
        return Verdict(
            VERDICT_RED,
            f"{best.name}: compression {ratio:.2f}x < {gate.min_compression}x at the quality bar",
        )
    if baseline_bytes is None:
        label = VERDICT_GO
        reason = (
            f"{best.name}: no matched-quality integer baseline exists; "
            f"{ratio:.2f}x versus the BF16 reference"
        )
    elif best.stored_bytes <= gate.go_bytes_ratio * baseline_bytes:
        label = VERDICT_GO
        reason = (
            f"{best.name}: {best.stored_bytes} B <= {gate.go_bytes_ratio} x {baseline_bytes} B "
            f"(baseline), {ratio:.2f}x versus the BF16 reference"
        )
    elif best.stored_bytes <= gate.yellow_bytes_ratio * baseline_bytes:
        label = VERDICT_YELLOW
        reason = (
            f"{best.name}: {best.stored_bytes} B is parity with the {baseline_bytes} B "
            f"baseline ({ratio:.2f}x versus the BF16 reference)"
        )
    else:
        return Verdict(
            VERDICT_RED,
            f"{best.name}: {best.stored_bytes} B above {gate.yellow_bytes_ratio} x "
            f"{baseline_bytes} B (baseline)",
        )
    if label == VERDICT_GO and best.only_after_healing:
        return Verdict(
            VERDICT_YELLOW,
            f"{best.name}: passes only after healing, capped at YELLOW (spec R6); {reason}",
        )
    return Verdict(label, reason)


def approx_equal(left: float, right: float, *, atol: float, rtol: float = 0.0) -> bool:
    """Return whether two floats agree within the given tolerances.

    Args:
        left: First value.
        right: Second value.
        atol: Absolute tolerance.
        rtol: Relative tolerance.

    Returns:
        ``True`` when the values agree.
    """
    return math.isclose(left, right, abs_tol=atol, rel_tol=rtol)
