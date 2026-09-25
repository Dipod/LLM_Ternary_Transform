"""Import smoke test and packaged-config validity (task 1.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from phase1_compression import config as config_module


def test_package_imports() -> None:
    import phase1_compression
    import phase1_compression.calibration.slices
    import phase1_compression.evaluation.accounting
    import phase1_compression.evaluation.benchmarks
    import phase1_compression.evaluation.gate
    import phase1_compression.evaluation.kl
    import phase1_compression.evaluation.perplexity
    import phase1_compression.evaluation.quantization
    import phase1_compression.runtime  # noqa: F401


def test_packaged_config_loads() -> None:
    config = config_module.load_config()
    assert config.cartography_model_id == "Qwen/Qwen2.5-0.5B"
    assert config.target_model_id == "Qwen/Qwen3-8B"
    assert config.seed == 42
    assert config.benchmarks.tasks == ("arc_easy", "hellaswag", "lambada_openai", "gsm8k")
    assert len(config.benchmarks.tasks) == len(config.benchmarks.num_fewshot)
    assert config.dense_windows_count * config.seq_len == 65536
    assert config.baselines.sanity_bits in config.baselines.bits
    assert config.gate.min_compression == 2.0
    assert config.cartography.ternary.layer == 12


def test_config_round_trips_a_copy(tmp_path: Path) -> None:
    text = config_module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    copy = tmp_path / "copy.yaml"
    copy.write_text(text, encoding="utf-8")
    assert config_module.load_config(copy) == config_module.load_config()


def test_missing_config_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        config_module.load_config(tmp_path / "missing.yaml")


def test_misaligned_fewshot_raises(tmp_path: Path) -> None:
    text = config_module.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
    broken = tmp_path / "broken.yaml"
    broken.write_text(
        text.replace("num_fewshot: [0, 0, 0, 4]", "num_fewshot: [0, 0, 4]"), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="align"):
        config_module.load_config(broken)
