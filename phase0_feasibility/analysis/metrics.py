"""Layer-local metrics for the Phase 0 gate (design.md -> Metrics and gate).

All computations are float32 and use explicit transposes so the shapes match
``PROJECT_PLAN.md`` conventions: ``X [tokens, in]``, ``W [out, in]``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from phase0_feasibility.decomposition.core import TernaryFactors

# Canonical thresholds, fixed before the run (feasibility-gate spec).
GREEN_MAX_E_X = 0.05
GREEN_MIN_SPARSITY = 0.40
GREEN_MAX_BPW = 6.0
YELLOW_MAX_E_X = 0.10
YELLOW_MAX_BPW = 8.0


@dataclass(frozen=True)
class LayerMetrics:
    """Metric set recorded for one layer configuration.

    Attributes:
        e_x: Relative layer output error ``E_x``.
        energy: Fraction of weight energy preserved (1 = perfect).
        sparsity: Fraction of zeros across ``B`` and ``C``.
        bpw_eff: Effective bits per weight (ExTernD Eq. (5) counting rule).
        runtime_s: Wall-clock decomposition runtime in seconds.
    """

    e_x: float
    energy: float
    sparsity: float
    bpw_eff: float
    runtime_s: float

    def as_dict(self) -> dict[str, float]:
        """Return a JSON-serialisable mapping."""
        return {
            "e_x": self.e_x,
            "energy": self.energy,
            "sparsity": self.sparsity,
            "bpw_eff": self.bpw_eff,
            "runtime_s": self.runtime_s,
        }


def layer_output_error(X: Tensor, W: Tensor, W_hat: Tensor) -> float:
    """Relative layer output error ``E_x`` (feasibility-gate spec).

    ``E_x = ||X W^T - X W_hat^T||_F / ||X W^T||_F`` computed in float32 with
    explicit transposes.

    Args:
        X: Calibration activations ``[tokens, in]``.
        W: Original weight ``[out, in]``.
        W_hat: Reconstruction ``[out, in]``.

    Returns:
        The relative error as a Python float.

    Raises:
        ValueError: If shapes disagree, values are non-finite, or the reference
            output ``X W^T`` has zero norm (the metric would be undefined).
    """
    if X.ndim != 2 or W.ndim != 2 or W_hat.ndim != 2:
        raise ValueError("layer_output_error expects three 2-D tensors")
    if X.shape[1] != W.shape[1] or W_hat.shape != W.shape:
        raise ValueError(
            f"shape mismatch: X {tuple(X.shape)}, W {tuple(W.shape)}, W_hat {tuple(W_hat.shape)}"
        )
    if not all(bool(torch.isfinite(t).all()) for t in (X, W, W_hat)):
        raise ValueError("layer_output_error received non-finite values")

    x = X.detach().to(torch.float32)
    w = W.detach().to(torch.float32)
    w_hat = W_hat.detach().to(torch.float32)
    reference = x @ w.t()
    reference_norm = float(torch.linalg.norm(reference))
    if reference_norm == 0.0:
        raise ValueError("layer_output_error undefined: ||X W^T||_F is zero")
    difference = reference - x @ w_hat.t()
    return float(torch.linalg.norm(difference) / reference_norm)


def energy_preserved(W: Tensor, W_hat: Tensor) -> float:
    """Fraction of weight energy preserved: ``1 - ||W_hat-W||_F^2 / ||W||_F^2``.

    Args:
        W: Original weight ``[out, in]``.
        W_hat: Reconstruction ``[out, in]``.

    Returns:
        The preserved-energy fraction.

    Raises:
        ValueError: If shapes disagree, values are non-finite, or ``||W||_F``
            is zero.
    """
    if W.shape != W_hat.shape:
        raise ValueError(f"shape mismatch: W {tuple(W.shape)}, W_hat {tuple(W_hat.shape)}")
    if not all(bool(torch.isfinite(t).all()) for t in (W, W_hat)):
        raise ValueError("energy_preserved received non-finite values")
    w = W.detach().to(torch.float32)
    w_hat = W_hat.detach().to(torch.float32)
    denominator = float(torch.linalg.norm(w) ** 2)
    if denominator == 0.0:
        raise ValueError("energy_preserved undefined: ||W||_F is zero")
    return 1.0 - float(torch.linalg.norm(w_hat - w) ** 2) / denominator


def factor_sparsity(B: Tensor, C: Tensor) -> float:
    """Fraction of zero entries across the two ternary factors.

    Args:
        B: Ternary factor ``[m, k]``.
        C: Ternary factor ``[k, n]``.

    Returns:
        ``zeros / (numel(B) + numel(C))``.

    Raises:
        ValueError: If either factor is empty.
    """
    total = int(B.numel()) + int(C.numel())
    if total == 0:
        raise ValueError("factor_sparsity undefined for empty factors")
    zeros = int((B == 0).sum()) + int((C == 0).sum())
    return zeros / total


def effective_bpw(m: int, n: int, k: int, mu: float, sparsity: float) -> float:
    """Effective bits per weight (ExTernD Eq. (5) counting rule).

    One bit of zero/nonzero mask plus one bit of sign per stored trit, over
    ``k * (m + n)`` stored trits, normalised by the original ``m * n``
    parameters: ``bpw = mu * (m + n) / max(m, n) * (2 - sparsity)``.

    Args:
        m: Original row count (``out_features``).
        n: Original column count (``in_features``).
        k: Stored rank; must satisfy ``k == round(mu * min(m, n))``.
        mu: Rank multiplier used for the fit.
        sparsity: Fraction of zeros across ``B`` and ``C``, in [0, 1].

    Returns:
        Effective bits per weight.

    Raises:
        ValueError: If ``k`` is inconsistent with ``mu`` and the shape, or
            ``sparsity`` is outside [0, 1].
    """
    if m < 1 or n < 1:
        raise ValueError(f"m and n must be positive, got m={m}, n={n}")
    if not 0.0 <= sparsity <= 1.0:
        raise ValueError(f"sparsity must be in [0, 1], got {sparsity}")
    expected_k = round(mu * min(m, n))
    if k != expected_k:
        raise ValueError(
            f"k={k} inconsistent with mu={mu} and shape ({m}, {n}); expected {expected_k}"
        )
    return mu * (m + n) / max(m, n) * (2.0 - sparsity)


def compute_metrics(
    X: Tensor,
    W: Tensor,
    factors: TernaryFactors,
    mu: float,
    runtime_s: float,
) -> LayerMetrics:
    """Compute the full metric set for one decomposed layer.

    Args:
        X: Calibration activations ``[tokens, in]``.
        W: Original weight ``[out, in]``.
        factors: Decomposition output for ``W``.
        mu: Rank multiplier used for the fit.
        runtime_s: Wall-clock decomposition runtime.

    Returns:
        The :class:`LayerMetrics` for this configuration.
    """
    out_features, in_features = int(W.shape[0]), int(W.shape[1])
    w_hat = factors.reconstruct()
    sparsity = factor_sparsity(factors.B, factors.C)
    return LayerMetrics(
        e_x=layer_output_error(X, W, w_hat),
        energy=energy_preserved(W, w_hat),
        sparsity=sparsity,
        bpw_eff=effective_bpw(out_features, in_features, factors.k, mu, sparsity),
        runtime_s=runtime_s,
    )


def metric_validity_check(
    X: Tensor,
    W: Tensor,
    *,
    seed: int,
    good_tol: float,
    ceiling_tol: float,
) -> tuple[bool, bool, float, float]:
    """Known-answer validity check on the real metric ``E_x``.

    A perfect reconstruction must score ~0 and an independent random
    reconstruction must score at or above the non-informative ceiling.

    Args:
        X: Calibration activations ``[tokens, in]``.
        W: Original weight ``[out, in]``.
        seed: Seed for the destroyed reconstruction.
        good_tol: Maximum ``E_x`` accepted for the perfect reconstruction.
        ceiling_tol: Tolerance subtracted from 1.0 for the destroyed case.

    Returns:
        ``(good_passes, bad_fails, good_e_x, bad_e_x)``.
    """
    good_e_x = layer_output_error(X, W, W)
    generator = torch.Generator(device=W.device).manual_seed(seed)
    destroyed = torch.randn(W.shape, generator=generator, dtype=torch.float32, device=W.device)
    bad_e_x = layer_output_error(X, W, destroyed)
    return good_e_x <= good_tol, bad_e_x >= 1.0 - ceiling_tol, good_e_x, bad_e_x
