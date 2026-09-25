"""Stored-byte accounting tests (task 2.3)."""

from __future__ import annotations

from phase1_compression.evaluation.accounting import (
    integer_group_storage,
    reference_storage,
    stored_bytes,
    vector_quant_storage,
)


def test_reference_storage_is_sixteen_bits() -> None:
    assert stored_bytes([reference_storage("w", 1000)]) == 2000


def test_group_quantization_counts_scales() -> None:
    import math

    tensors = integer_group_storage("w", numel=1_000_000, bits=4, group_size=128)
    payload = 1_000_000 * 4 / 8
    scales = math.ceil(1_000_000 / 128) * 2
    assert stored_bytes(tensors) == int(payload + scales)
    assert sum(t.name.endswith(".scales") for t in tensors) == 1


def test_low_bits_with_large_codebook_does_not_score_lower() -> None:
    numel = 1_000_000
    int4 = stored_bytes(integer_group_storage("w", numel=numel, bits=4, group_size=128))
    vq2 = stored_bytes(
        vector_quant_storage("w", numel=numel, dim=8, codebook_size=65536, index_bits=16)
    )
    assert vq2 > int4


def test_small_codebook_vector_quantization_beats_four_bit() -> None:
    numel = 1_000_000
    int4 = stored_bytes(integer_group_storage("w", numel=numel, bits=4, group_size=128))
    vq2 = stored_bytes(
        vector_quant_storage("w", numel=numel, dim=8, codebook_size=256, index_bits=8)
    )
    assert vq2 < int4


def test_bad_inputs_raise() -> None:
    import pytest

    from phase1_compression.evaluation.accounting import StoredTensor

    with pytest.raises(ValueError, match="group_size"):
        integer_group_storage("w", numel=10, bits=4, group_size=0)
    with pytest.raises(ValueError, match="numel"):
        vector_quant_storage("w", numel=-1, dim=8, codebook_size=16, index_bits=8)
    with pytest.raises(ValueError, match="non-positive bits"):
        stored_bytes([StoredTensor("w", 1, 0.0)])
