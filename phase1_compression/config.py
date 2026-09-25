"""Single configuration loader for Phase 1 Stage A.

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


@dataclass(frozen=True)
class Benchmarks:
    """Benchmark evaluation settings (design.md D4).

    Attributes:
        tasks: Harness task names.
        num_fewshot: Few-shot count per task, aligned with ``tasks``.
        limit: Item cap per task; ``None`` evaluates the full task.
        batch_size: Harness batch size.
    """

    tasks: tuple[str, ...]
    num_fewshot: tuple[int, ...]
    limit: int | None
    batch_size: int


@dataclass(frozen=True)
class Baselines:
    """Integer group-quantization baseline settings (spec R3).

    Attributes:
        bits: Bit widths to evaluate.
        group_size: Elements per quantization group.
        n_clip: Clip-ratio search resolution for the MSE-optimal scale.
        sanity_bits: Bit width of the harness sanity baseline.
        sanity_ppl_rel_tol: Maximum relative perplexity increase accepted for the
            sanity baseline before the harness is declared invalid.
    """

    bits: tuple[int, ...]
    group_size: int
    n_clip: int
    sanity_bits: int
    sanity_ppl_rel_tol: float


@dataclass(frozen=True)
class Gate:
    """Stage A gate thresholds (spec R1, R7).

    Attributes:
        ppl_rel_tol: Maximum relative perplexity increase.
        bench_delta_tol: Minimum allowed benchmark delta, in points.
        go_bytes_ratio: Stored-byte ratio versus the matched-quality baseline for GO.
        yellow_bytes_ratio: Upper stored-byte ratio that still scores YELLOW.
        min_compression: Minimum compression ratio versus the BF16 reference.
    """

    ppl_rel_tol: float
    bench_delta_tol: float
    go_bytes_ratio: float
    yellow_bytes_ratio: float
    min_compression: float


@dataclass(frozen=True)
class StructuredAxis:
    """Axis A settings (spec redundancy-cartography R3).

    Attributes:
        kinds: Unit kinds to ablate, ordered.
        kl_budget: KL divergence budget in nats.
        max_removals: Cap on removal steps per kind.
    """

    kinds: tuple[str, ...]
    kl_budget: float
    max_removals: int


@dataclass(frozen=True)
class DirectionsAxis:
    """Axis B settings (spec redundancy-cartography R4).

    Attributes:
        n_directions: Subspace sizes to sweep.
        layer_bands: Inclusive layer ranges to sweep, as ``(start, end)`` pairs.
        kl_budget: KL divergence budget in nats.
    """

    n_directions: tuple[int, ...]
    layer_bands: tuple[tuple[int, int], ...]
    kl_budget: float


@dataclass(frozen=True)
class TernaryAxis:
    """Axis C settings (spec redundancy-cartography R5).

    Attributes:
        layer: Decoder layer whose MLP is decomposed.
        mu: Rank multipliers to sweep.
        tau: Ternary threshold.
        nm_patterns: N:M patterns applied over the natural zeros.
        rank_fractions: Fractions of the ternary-core rank to keep.
        e_x_budget: Relative output-error budget.
    """

    layer: int
    mu: tuple[int, ...]
    tau: float
    nm_patterns: tuple[tuple[int, int], ...]
    rank_fractions: tuple[float, ...]
    e_x_budget: float


@dataclass(frozen=True)
class Cartography:
    """Cartography settings across the three axes.

    Attributes:
        structured: Axis A settings.
        directions: Axis B settings.
        ternary: Axis C settings.
        budget_gpu_hours_per_axis: Finite per-axis GPU budget (spec R9).
        kl_probe_batch_size: Batch size for the KL probe forward passes.
    """

    structured: StructuredAxis
    directions: DirectionsAxis
    ternary: TernaryAxis
    budget_gpu_hours_per_axis: float
    kl_probe_batch_size: int


@dataclass(frozen=True)
class Probe:
    """Probe settings (design.md D7, probe plan).

    Attributes:
        ffn_sweep_fractions: Cumulative FFN-removal fractions for the end-to-end
            degradation sweep; strictly increasing.
    """

    ffn_sweep_fractions: tuple[float, ...]


@dataclass(frozen=True)
class Healing:
    """Bounded recovery-training settings (spec R6, design.md D5).

    Attributes:
        enabled: Whether healing runs at all.
        max_gpu_seconds: Wall-clock cap for the single run.
        lora_rank: Adapter rank cap.
        learning_rate: Optimizer learning rate.
        max_steps: Optimizer step cap.
        target_tokens: Training tokens drawn from the calibration slice.
    """

    enabled: bool
    max_gpu_seconds: float
    lora_rank: int
    learning_rate: float
    max_steps: int
    target_tokens: int


@dataclass(frozen=True)
class Tolerance:
    """Numerical tolerances used by tests and comparisons.

    Attributes:
        ppl_atol: Absolute tolerance for perplexity equality.
        kl_atol: Absolute tolerance for KL equality.
        bytes_rtol: Relative tolerance for stored-byte comparisons.
    """

    ppl_atol: float
    kl_atol: float
    bytes_rtol: float


@dataclass(frozen=True)
class Smoke:
    """Reduced settings for a cheap smoke run of the cartography axes.

    Attributes:
        structured_max_removals: Removal cap for the smoke run.
        directions_n: Subspace sizes for the smoke run.
        ternary_mu: Rank multipliers for the smoke run.
        limit: Benchmark item cap for the smoke run.
    """

    structured_max_removals: int
    directions_n: tuple[int, ...]
    ternary_mu: tuple[int, ...]
    limit: int


@dataclass(frozen=True)
class Config:
    """Fully resolved Phase 1 Stage A configuration.

    Attributes:
        cartography_model_id: Small model used for the redundancy map.
        target_model_id: Model the combined probe is applied to.
        dtype: Model dtype for forward passes.
        device: Compute device; ROCm exposes the GPU through the ``cuda`` API.
        seed: Master seed for torch, numpy and random.
        results_dir: Output directory for recorded runs, relative to the repo root.
        dataset_id: HuggingFace dataset identifier.
        dataset_config: Dataset configuration name.
        calibration_split: Split used for the calibration slice.
        seq_len: Window length in tokens.
        dense_windows_count: Complete windows kept in the calibration slice.
        ppl_split: Split used for the perplexity slice.
        ppl_windows_count: Windows kept in the perplexity slice.
        ppl_batch_size: Batch size for the perplexity forward passes.
        benchmarks: Benchmark evaluation settings.
        baselines: Integer baseline settings.
        gate: Gate thresholds.
        cartography: Cartography settings.
        probe: Probe settings.
        healing: Healing settings.
        tolerance: Numerical tolerances.
        smoke: Reduced smoke-run settings.
    """

    cartography_model_id: str
    target_model_id: str
    dtype: str
    device: str
    seed: int
    results_dir: str
    dataset_id: str
    dataset_config: str
    calibration_split: str
    seq_len: int
    dense_windows_count: int
    ppl_split: str
    ppl_windows_count: int
    ppl_batch_size: int
    benchmarks: Benchmarks
    baselines: Baselines
    gate: Gate
    cartography: Cartography
    probe: Probe
    healing: Healing
    tolerance: Tolerance
    smoke: Smoke


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load and validate a Phase 1 Stage A configuration file.

    Args:
        path: Path to the YAML configuration. Defaults to the packaged
            ``config.yaml``.

    Returns:
        The parsed :class:`Config`.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        TypeError: If the YAML root is not a mapping.
        KeyError: If a required key is missing.
        ValueError: If a value is outside its allowed range.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"config not found: {config_path}")
    raw: Any = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TypeError(f"config root must be a mapping, got {type(raw).__name__}")

    benchmarks_raw = raw["benchmarks"]
    tasks = tuple(str(v) for v in benchmarks_raw["tasks"])
    num_fewshot = tuple(int(v) for v in benchmarks_raw["num_fewshot"])
    if len(tasks) != len(num_fewshot):
        raise ValueError(
            f"benchmarks.num_fewshot must align with tasks: {len(tasks)} tasks, "
            f"{len(num_fewshot)} few-shot entries"
        )

    baselines_raw = raw["baselines"]
    gate_raw = raw["gate"]
    cart_raw = raw["cartography"]
    structured_raw = cart_raw["structured"]
    directions_raw = cart_raw["directions"]
    ternary_raw = cart_raw["ternary"]
    healing_raw = raw["healing"]
    probe_raw = raw["probe"]
    tolerance_raw = raw["tolerance"]
    smoke_raw = raw["smoke"]

    ppl_rel_tol = float(gate_raw["ppl_rel_tol"])
    if not ppl_rel_tol > 0.0:
        raise ValueError(f"gate.ppl_rel_tol must be positive, got {ppl_rel_tol}")
    bits = tuple(int(v) for v in baselines_raw["bits"])
    if not bits or any(b < 2 for b in bits):
        raise ValueError(f"baselines.bits must be >= 2, got {bits}")

    return Config(
        cartography_model_id=str(raw["cartography_model_id"]),
        target_model_id=str(raw["target_model_id"]),
        dtype=str(raw["dtype"]),
        device=str(raw["device"]),
        seed=int(raw["seed"]),
        results_dir=str(raw["results_dir"]),
        dataset_id=str(raw["dataset_id"]),
        dataset_config=str(raw["dataset_config"]),
        calibration_split=str(raw["calibration_split"]),
        seq_len=int(raw["seq_len"]),
        dense_windows_count=int(raw["dense_windows_count"]),
        ppl_split=str(raw["ppl_split"]),
        ppl_windows_count=int(raw["ppl_windows_count"]),
        ppl_batch_size=int(raw["ppl_batch_size"]),
        benchmarks=Benchmarks(
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=None if benchmarks_raw["limit"] is None else int(benchmarks_raw["limit"]),
            batch_size=int(benchmarks_raw["batch_size"]),
        ),
        baselines=Baselines(
            bits=bits,
            group_size=int(baselines_raw["group_size"]),
            n_clip=int(baselines_raw["n_clip"]),
            sanity_bits=int(baselines_raw["sanity_bits"]),
            sanity_ppl_rel_tol=float(baselines_raw["sanity_ppl_rel_tol"]),
        ),
        gate=Gate(
            ppl_rel_tol=ppl_rel_tol,
            bench_delta_tol=float(gate_raw["bench_delta_tol"]),
            go_bytes_ratio=float(gate_raw["go_bytes_ratio"]),
            yellow_bytes_ratio=float(gate_raw["yellow_bytes_ratio"]),
            min_compression=float(gate_raw["min_compression"]),
        ),
        cartography=Cartography(
            structured=StructuredAxis(
                kinds=tuple(str(v) for v in structured_raw["kinds"]),
                kl_budget=float(structured_raw["kl_budget"]),
                max_removals=int(structured_raw["max_removals"]),
            ),
            directions=DirectionsAxis(
                n_directions=tuple(int(v) for v in directions_raw["n_directions"]),
                layer_bands=tuple(
                    (int(pair[0]), int(pair[1])) for pair in directions_raw["layer_bands"]
                ),
                kl_budget=float(directions_raw["kl_budget"]),
            ),
            ternary=TernaryAxis(
                layer=int(ternary_raw["layer"]),
                mu=tuple(int(v) for v in ternary_raw["mu"]),
                tau=float(ternary_raw["tau"]),
                nm_patterns=tuple(
                    (int(pair[0]), int(pair[1])) for pair in ternary_raw["nm_patterns"]
                ),
                rank_fractions=tuple(float(v) for v in ternary_raw["rank_fractions"]),
                e_x_budget=float(ternary_raw["e_x_budget"]),
            ),
            budget_gpu_hours_per_axis=float(cart_raw["budget_gpu_hours_per_axis"]),
            kl_probe_batch_size=int(cart_raw["kl_probe_batch_size"]),
        ),
        probe=Probe(
            ffn_sweep_fractions=tuple(float(v) for v in probe_raw["ffn_sweep_fractions"]),
        ),
        healing=Healing(
            enabled=bool(healing_raw["enabled"]),
            max_gpu_seconds=float(healing_raw["max_gpu_seconds"]),
            lora_rank=int(healing_raw["lora_rank"]),
            learning_rate=float(healing_raw["learning_rate"]),
            max_steps=int(healing_raw["max_steps"]),
            target_tokens=int(healing_raw["target_tokens"]),
        ),
        tolerance=Tolerance(
            ppl_atol=float(tolerance_raw["ppl_atol"]),
            kl_atol=float(tolerance_raw["kl_atol"]),
            bytes_rtol=float(tolerance_raw["bytes_rtol"]),
        ),
        smoke=Smoke(
            structured_max_removals=int(smoke_raw["structured_max_removals"]),
            directions_n=tuple(int(v) for v in smoke_raw["directions_n"]),
            ternary_mu=tuple(int(v) for v in smoke_raw["ternary_mu"]),
            limit=int(smoke_raw["limit"]),
        ),
    )
