"""Single configuration loader for Phase 0.

Every tunable value lives in ``config.yaml`` and reaches the algorithms as an
explicit argument; no algorithm hardcodes a hyperparameter (AGENTS.md -> Python
standards -> Configuration). The loader is the only place that reads the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")

# Selectable calibration slice modes (change phase0-calibration-robustness D1-D3).
# ``docs128`` is the archived 128-document definition, ``dense_windows`` a
# concatenated token stream cut into windows, ``many_documents`` the archived
# rule with a longer document count.
SLICE_MODES = frozenset({"dense_windows", "many_documents", "docs128"})


@dataclass(frozen=True)
class Tolerance:
    """Tolerances for tests, cross-checks and reproducibility comparisons.

    Attributes:
        recon_rtol: Relative tolerance for reconstruction cross-checks.
        recon_atol: Absolute tolerance for reconstruction cross-checks.
        cross_check_rtol: Tolerance for the batched-vs-sequential cross-check.
        e_x_atol: Absolute tolerance for ``E_x`` near zero.
        reproduce_rtol: Relative tolerance for repeat-run metric comparison.
    """

    recon_rtol: float
    recon_atol: float
    cross_check_rtol: float
    e_x_atol: float
    reproduce_rtol: float


@dataclass(frozen=True)
class Sweep:
    """Reduced sweep grid (design.md -> Metrics and gate).

    Attributes:
        mu: Rank multipliers to sweep.
        tau: Ternary thresholds to sweep.
        importance: Importance-weighting switches to sweep.
        reorder: Column-reordering switches to sweep.
        n_inner: Inner alternating iteration counts to sweep.
    """

    mu: tuple[int, ...]
    tau: tuple[float, ...]
    importance: tuple[bool, ...]
    reorder: tuple[bool, ...]
    n_inner: tuple[int, ...]


@dataclass(frozen=True)
class Config:
    """Fully resolved Phase 0 configuration.

    Attributes:
        model_id: HuggingFace model identifier.
        dtype: Model dtype for the calibration forward pass.
        device: Compute device: ``auto`` (cuda when available, else cpu), ``cpu``,
            ``cuda`` or ``cuda:N`` (ROCm exposes GPUs through the ``cuda`` API).
        layers: Decoder layer indices whose MLPs are decomposed (each scored as
            its own run).
        target_module: Module name inside the MLP block.
        mu: Default rank multiplier (``k = round(mu * min(out, in))``).
        tau: Default ternary threshold.
        b: Batched-path block size; ``None`` selects ``min(256, min(out,in)//8)``.
        n_inner: Inner alternating iterations per component block.
        eps: Numerical epsilon for least-squares inverses.
        seed: Master seed for torch, numpy and random.
        dataset_id: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        dataset_split: Dataset split name.
        slices: Calibration slice modes to build; subset of :data:`SLICE_MODES`.
        dense_windows_count: Complete windows kept in the ``dense_windows`` mode.
        many_documents_count: Non-empty documents kept in the ``many_documents`` mode.
        seq_len: Token truncation length per document and window length.
        eval_fraction: Fraction of slice items held out; contiguous split, no RNG.
        calibration_batch_size: Documents per forward pass during capture.
        importance_lambda: Ridge added to the normalised importance weights.
        reorder: Whether column reordering is applied in the default run.
        results_dir: Output directory for recorded runs (relative to repo root).
        runtime_budget_s: Per-layer runtime budget in seconds (spec: 600).
        tolerance: Numerical tolerances.
        sweep: Reduced sweep grid.
    """

    model_id: str
    dtype: str
    device: str
    layers: tuple[int, ...]
    target_module: str
    mu: int
    tau: float
    b: int | None
    n_inner: int
    eps: float
    seed: int
    dataset_id: str
    dataset_config: str
    dataset_split: str
    slices: tuple[str, ...]
    dense_windows_count: int
    many_documents_count: int
    seq_len: int
    eval_fraction: float
    calibration_batch_size: int
    importance_lambda: float
    reorder: bool
    results_dir: str
    runtime_budget_s: float
    tolerance: Tolerance
    sweep: Sweep


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a Phase 0 configuration file.

    Args:
        path: Path to the YAML configuration. Defaults to the packaged
            ``config.yaml``.

    Returns:
        The parsed :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML root is not a mapping.
        KeyError: If a required key is missing.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}")
    b_raw = raw["b"]
    tolerance_raw = raw["tolerance"]
    sweep_raw = raw["sweep"]
    layers = tuple(int(v) for v in raw["layers"])
    if not layers:
        raise ValueError("layers must list at least one decoder layer index")
    slices = tuple(str(v) for v in raw["slices"])
    unknown = [mode for mode in slices if mode not in SLICE_MODES]
    if unknown:
        raise ValueError(
            f"slices contains unknown modes {unknown}; expected a subset of {sorted(SLICE_MODES)}"
        )
    eval_fraction = float(raw["eval_fraction"])
    if not 0.0 <= eval_fraction < 1.0:
        raise ValueError(f"eval_fraction must be in [0, 1), got {eval_fraction}")
    return Config(
        model_id=str(raw["model_id"]),
        dtype=str(raw["dtype"]),
        device=str(raw["device"]),
        layers=layers,
        target_module=str(raw["target_module"]),
        mu=int(raw["mu"]),
        tau=float(raw["tau"]),
        b=None if b_raw is None else int(b_raw),
        n_inner=int(raw["n_inner"]),
        eps=float(raw["eps"]),
        seed=int(raw["seed"]),
        dataset_id=str(raw["dataset_id"]),
        dataset_config=str(raw["dataset_config"]),
        dataset_split=str(raw["dataset_split"]),
        slices=slices,
        dense_windows_count=int(raw["dense_windows_count"]),
        many_documents_count=int(raw["many_documents_count"]),
        seq_len=int(raw["seq_len"]),
        eval_fraction=eval_fraction,
        calibration_batch_size=int(raw["calibration_batch_size"]),
        importance_lambda=float(raw["importance_lambda"]),
        reorder=bool(raw["reorder"]),
        results_dir=str(raw["results_dir"]),
        runtime_budget_s=float(raw["runtime_budget_s"]),
        tolerance=Tolerance(
            recon_rtol=float(tolerance_raw["recon_rtol"]),
            recon_atol=float(tolerance_raw["recon_atol"]),
            cross_check_rtol=float(tolerance_raw["cross_check_rtol"]),
            e_x_atol=float(tolerance_raw["e_x_atol"]),
            reproduce_rtol=float(tolerance_raw["reproduce_rtol"]),
        ),
        sweep=Sweep(
            mu=tuple(int(v) for v in sweep_raw["mu"]),
            tau=tuple(float(v) for v in sweep_raw["tau"]),
            importance=tuple(bool(v) for v in sweep_raw["importance"]),
            reorder=tuple(bool(v) for v in sweep_raw["reorder"]),
            n_inner=tuple(int(v) for v in sweep_raw["n_inner"]),
        ),
    )
