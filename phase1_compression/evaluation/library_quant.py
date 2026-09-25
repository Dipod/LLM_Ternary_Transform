"""Library weight-only quantization baselines via ``torchao`` (spec R3, R4).

``torchao`` is an optional dependency: it exists in the ROCm experiment venv but
not in the dev toolchain, so it is imported lazily and its absence raises a clear
error instead of breaking the package import.
"""

from __future__ import annotations

from typing import Any

__all__ = ["LIBRARY_BITS", "library_available", "quantize_library_"]

#: Bit widths supported by the library arm.
LIBRARY_BITS: tuple[int, ...] = (8, 4)


def library_available() -> bool:
    """Return whether ``torchao`` can be imported.

    Returns:
        ``True`` when the optional dependency is importable.
    """
    try:
        import torchao  # noqa: F401
    except Exception:  # pragma: no cover - depends on the experiment venv
        return False
    return True


def quantize_library_(model: Any, *, bits: int, group_size: int) -> None:
    """Apply ``torchao`` weight-only quantization to a model, in place.

    Args:
        model: Model mutated in place.
        bits: ``8`` for ``int8_weight_only``, ``4`` for ``int4_weight_only``.
        group_size: Elements per quantization group.

    Raises:
        RuntimeError: If ``torchao`` is not importable.
        ValueError: If ``bits`` is not supported.
    """
    if bits not in LIBRARY_BITS:
        raise ValueError(f"library arm supports bits {LIBRARY_BITS}, got {bits}")
    try:
        from torchao.quantization import int4_weight_only, int8_weight_only, quantize_
    except Exception as exc:  # pragma: no cover - depends on the experiment venv
        raise RuntimeError(
            f"torchao is required for the library arm but could not be imported: {exc}"
        ) from exc
    config = (
        int8_weight_only(group_size=group_size)
        if bits == 8
        else int4_weight_only(group_size=group_size)
    )
    quantize_(model, config)
