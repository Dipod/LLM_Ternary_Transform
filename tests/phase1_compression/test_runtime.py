"""Seed and device helper tests (task 1.2)."""

from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from phase1_compression.runtime import resolve_device, set_seed


def test_seed_reproduces_tensors_bit_for_bit() -> None:
    set_seed(1234)
    first = torch.randn(8)
    second_numpy = np.random.rand(3)
    third_python = random.random()
    set_seed(1234)
    assert torch.equal(first, torch.randn(8))
    assert np.array_equal(second_numpy, np.random.rand(3))
    assert third_python == random.random()


def test_device_resolution() -> None:
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("auto") in {"cpu", "cuda"}
    if torch.cuda.is_available():
        assert resolve_device("cuda:0") == "cuda:0"
    else:
        with pytest.raises(ValueError, match="cuda"):
            resolve_device("cuda:0")
    with pytest.raises(ValueError, match="unknown device"):
        resolve_device("tpu")
