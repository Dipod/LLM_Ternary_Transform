"""Determinism and device helpers (spec: AGENTS.md -> Python standards).

Every algorithm receives its seed through :func:`set_seed`; no algorithm calls a
seed function itself (task 1.2).
"""

from __future__ import annotations

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed torch, numpy and random from the configuration.

    Args:
        seed: Master seed.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def enable_determinism() -> str:
    """Enable deterministic torch algorithms where the platform supports them.

    Returns:
        A short note describing whether the strict mode was enabled; the note is
        recorded in the run provenance instead of claiming determinism.
    """
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:  # pragma: no cover - platform dependent
        return f"deterministic algorithms unavailable: {exc}"
    return "torch.use_deterministic_algorithms(True)"


def resolve_device(requested: str) -> str:
    """Resolve a configured device string to a concrete torch device.

    Args:
        requested: ``auto`` (cuda when available, else cpu), ``cpu``, ``cuda`` or
            ``cuda:N``. ROCm exposes the AMD card through the ``cuda`` API.

    Returns:
        The resolved device string.

    Raises:
        ValueError: If an explicit cuda device is requested but unavailable, or
            the string is unknown.
    """
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cpu":
        return "cpu"
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise ValueError(
                f"device {requested!r} requested but torch.cuda.is_available() is False"
            )
        return requested
    raise ValueError(f"unknown device {requested!r}; expected auto, cpu, cuda or cuda:N")
