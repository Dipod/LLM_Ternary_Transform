"""Cartography axis helper tests (tasks 3.1-3.4, pure parts)."""

from __future__ import annotations

from pathlib import Path

import torch

from phase0_feasibility.analysis.metrics import layer_output_error
from phase0_feasibility.decomposition.core import TernaryFactors
from phase1_compression.cartography.axes import (
    _factor_zero_fraction,
    _nm_mask,
    _plane_drop_steps,
    _unit_name,
)
from phase1_compression.cartography.record import AxisMap, Step, write_map
from phase1_compression.config import load_config


def _factors() -> TernaryFactors:
    B = torch.tensor([[1, 0], [-1, 1]], dtype=torch.int8)
    C = torch.tensor([[1, -1], [0, -1]], dtype=torch.int8)
    D = torch.tensor([2.0, 1.0], dtype=torch.float32)
    return TernaryFactors(B=B, D=D, C=C, k=2)


def test_zero_fraction_known_answer() -> None:
    factors = _factors()
    zeros = int((factors.B == 0).sum() + (factors.C == 0).sum())
    total = factors.B.numel() + factors.C.numel()
    assert _factor_zero_fraction(factors) == zeros / total


def test_nm_mask_keeps_the_largest_entries() -> None:
    factor = torch.tensor([[1, 0, -1, 0, 1, 0, 0, 1]], dtype=torch.int8)
    scales = torch.arange(1, 9, dtype=torch.float32)
    mask = _nm_mask(factor, scales, n_keep=2, m_group=8)
    assert mask.sum().item() == 2
    kept = factor.mul(mask).abs().reshape(-1).nonzero().flatten().tolist()
    assert all(index in {4, 7} for index in kept)


def test_nm_mask_rejects_bad_pattern() -> None:
    import pytest

    with pytest.raises(ValueError, match="invalid N:M"):
        _nm_mask(torch.ones(1, 4, dtype=torch.int8), torch.ones(4), n_keep=5, m_group=4)


def test_plane_drop_reproduces_a_known_reconstruction() -> None:
    torch.manual_seed(0)
    W = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    activations = torch.eye(2)
    factors = _factors()
    reconstruction = factors.reconstruct()
    steps = _plane_drop_steps(activations, reconstruction, factors, (1.0, 0.5))
    assert steps[0].extra["k"] == 2.0
    assert steps[0].extra["e_x"] < 1e-6
    assert steps[1].extra["k"] == 1.0
    assert steps[1].removed == 1
    # Dropping the smaller plane leaves a rank-1 approximation with nonzero error.
    assert steps[1].extra["e_x"] > 0.0
    assert layer_output_error(activations, W, W) < 1e-6


def test_unit_name_formats() -> None:
    assert _unit_name("head", (3, 5, 64, 1.0)) == "l3h5"
    assert _unit_name("ffn_dim", (3, 5, 1.0)) == "l3n5"


def test_write_map_layout(tmp_path: Path) -> None:
    config = load_config()
    axis = AxisMap(
        axis="structured",
        proxy="kl_nats",
        budget=0.1,
        steps=(Step("block", ("layer0",), 1, 0.01, {}),),
        stopped_on_budget=False,
        notes=("note",),
        e_x_cross_check=0.02,
    )
    run_dir = write_map(tmp_path, "p1_map_test", [axis], config=config, provenance={"seed": 42})
    assert (run_dir / "config.yaml").is_file()
    assert (run_dir / "metrics.json").is_file()
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert "p1_map_test" in report
    assert "structured" in report
    assert "0.0100" in report
