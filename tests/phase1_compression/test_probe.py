"""Probe plan derivation tests (task 4.1: the order comes from the map)."""

from __future__ import annotations

from typing import Any

import pytest

from phase1_compression.probe import derive_probe_plan


def _payload() -> dict[str, Any]:
    return {
        "experiment_id": "p1_map_test",
        "axes": [
            {
                "axis": "structured",
                "proxy": "kl_nats",
                "budget": 0.1,
                "steps": [
                    {"kind": "block", "units": ["layer3"], "removed": 1, "kl": 0.05, "extra": {}},
                    {
                        "kind": "head",
                        "units": ["l0h1"],
                        "removed": 1,
                        "kl": 2.88,
                        "extra": {"n_ranked": 1.0, "n_total": 20.0},
                    },
                    {
                        "kind": "ffn_dim",
                        "units": ["l0n1"],
                        "removed": 5,
                        "kl": 0.09,
                        "extra": {"n_ranked": 5.0, "n_total": 100.0},
                    },
                    {
                        "kind": "ffn_dim",
                        "units": ["l0n2"],
                        "removed": 10,
                        "kl": 0.20,
                        "extra": {"n_ranked": 10.0, "n_total": 100.0},
                    },
                ],
            },
            {"axis": "ternary", "proxy": "e_x_relative", "budget": 0.05, "steps": []},
        ],
    }


def test_plan_uses_the_largest_passing_fraction() -> None:
    plan = derive_probe_plan(_payload())
    assert plan.map_id == "p1_map_test"
    assert plan.ffn_fraction == pytest.approx(0.05)
    assert plan.head_fraction == 0.0
    assert plan.block_layers == (3,)


def test_plan_without_structured_axis_raises() -> None:
    with pytest.raises(KeyError, match="structured"):
        derive_probe_plan({"axes": []})


def test_plan_with_no_axes_list_raises() -> None:
    with pytest.raises(KeyError, match="axes"):
        derive_probe_plan({})
