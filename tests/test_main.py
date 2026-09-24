"""Tests for the run id format used by the re-check records (task 3.3 / D8)."""

from __future__ import annotations

from phase0_feasibility.main import _experiment_id


def test_experiment_id_encodes_slice_and_layer() -> None:
    assert (
        _experiment_id("dense_windows", 3, 3, 0.7, False, False, 15)
        == "s-dense_windows_l3_mu3_tau0.7_impoff_reoff_ni15"
    )


def test_experiment_id_marks_importance_and_reorder_on() -> None:
    assert (
        _experiment_id("many_documents", 21, 2, 1.0, True, True, 15)
        == "s-many_documents_l21_mu2_tau1_impon_reon_ni15"
    )
