"""Benchmark result parsing tests (task 2.6, harness fact from task 1.5)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from phase1_compression.evaluation.benchmarks import (
    _locate_results,
    build_command,
    parse_results,
)


def _payload() -> dict[str, Any]:
    return {
        "lm_eval_version": "0.4.13",
        "results": {
            "arc_easy": {"acc,none": 0.4, "acc_norm,none": 0.42, "alias": "arc_easy"},
            "gsm8k": {"exact_match,strict-match": 0.01, "exact_match,flexible-extract": 0.02},
            "hellaswag": {"acc,none": 0.3},
        },
    }


def test_parse_results_picks_the_primary_metric() -> None:
    parsed = parse_results(_payload())
    assert parsed["arc_easy"] == pytest.approx(42.0)
    assert parsed["gsm8k"] == pytest.approx(1.0)
    assert parsed["hellaswag"] == pytest.approx(30.0)


def test_parse_results_requires_results_mapping() -> None:
    with pytest.raises(ValueError, match="results"):
        parse_results({"nope": 1})


def test_locate_results_uses_the_timestamped_sibling(tmp_path: Path) -> None:
    requested = tmp_path / "harness.json"
    newer = tmp_path / "harness_2026-09-25T01-27-51.json"
    older = tmp_path / "harness_2026-09-24T01-27-51.json"
    older.write_text(json.dumps({"old": True}), encoding="utf-8")
    newer.write_text(json.dumps({"new": True}), encoding="utf-8")
    assert _locate_results(requested) == newer
    requested.write_text(json.dumps({"exact": True}), encoding="utf-8")
    assert _locate_results(requested) == requested


def test_locate_results_raises_when_absent(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no result file"):
        _locate_results(tmp_path / "missing.json")


def test_build_command_pins_the_evaluation_protocol() -> None:
    command = build_command(
        "Qwen/Qwen3-8B",
        tasks=("arc_easy", "gsm8k"),
        num_fewshot=(0, 4),
        limit=None,
        batch_size=8,
        device="cuda:0",
        output_path=Path("out/harness.json"),
        python_executable="python",
    )
    assert command[:3] == ["python", "-m", "lm_eval"]
    assert "arc_easy,gsm8k" in command
    assert "0,4" in command
    assert "--limit" not in command


def test_build_command_rejects_misaligned_fewshot() -> None:
    with pytest.raises(ValueError, match="align"):
        build_command(
            "m",
            tasks=("a", "b"),
            num_fewshot=(0,),
            limit=5,
            batch_size=1,
            device="cpu",
            output_path=Path("out.json"),
            python_executable="python",
        )
