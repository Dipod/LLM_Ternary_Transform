"""Network smoke tests for the real model and calibration slices (task 3.1).

Marked ``network``: they download ``Qwen/Qwen2.5-0.5B`` and WikiText-2 and are
excluded from the default gate chain (pyproject addopts). Run them explicitly
with ``pytest -m network``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from phase0_feasibility.calibration.hooks import capture_gate_inputs
from phase0_feasibility.calibration.loader import (
    DOCS128_COUNT,
    build_dense_windows,
    encode_documents,
    layer_module_path,
    load_documents,
    load_model,
    load_text_stream,
    load_tokenizer,
)
from phase0_feasibility.config import load_config

CONFIG_PATH = Path(__file__).resolve().parents[2] / "phase0_feasibility" / "config.yaml"


@pytest.mark.network
def test_model_architecture_and_forward_smoke() -> None:
    """24 layers, hidden 896, intermediate 4864, and a working capture pass."""
    config = load_config(CONFIG_PATH)
    model = load_model(config.model_id, config.dtype)
    assert len(model.model.layers) == 24
    assert model.config.hidden_size == 896
    assert model.config.intermediate_size == 4864

    tokenizer = load_tokenizer(config.model_id)
    input_ids, attention_mask = encode_documents(
        tokenizer, ["Ternary networks replace weights with {-1, 0, 1}."], config.seq_len
    )
    activations = capture_gate_inputs(
        model,
        input_ids,
        attention_mask,
        layer_path=layer_module_path(config.layers[0], config.target_module),
        batch_size=1,
    )
    assert activations.shape[1] == 896
    assert int(activations.shape[0]) == int(attention_mask.sum())


@pytest.mark.network
def test_calibration_slice_token_accounting() -> None:
    """``docs128`` reproduces the archived 128-document, 11,426-token slice."""
    config = load_config(CONFIG_PATH)
    documents = load_documents(
        config.dataset_id, config.dataset_config, config.dataset_split, DOCS128_COUNT
    )
    assert len(documents) == 128
    tokenizer = load_tokenizer(config.model_id)
    input_ids, attention_mask = encode_documents(tokenizer, documents, config.seq_len)
    assert input_ids.shape == (128, 512)
    assert int(attention_mask.sum()) == 11_426


@pytest.mark.network
def test_dense_windows_slice_token_accounting() -> None:
    """``dense_windows`` yields 128 full windows with no padding (65,536 tokens)."""
    config = load_config(CONFIG_PATH)
    tokenizer = load_tokenizer(config.model_id)
    text = load_text_stream(config.dataset_id, config.dataset_config, config.dataset_split)
    input_ids, attention_mask = build_dense_windows(
        tokenizer, text, seq_len=config.seq_len, n_windows=config.dense_windows_count
    )
    assert input_ids.shape == (128, 512)
    assert int(attention_mask.sum()) == 128 * 512 == 65_536
    assert bool((attention_mask == 1).all())
