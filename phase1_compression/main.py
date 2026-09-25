"""Phase 1 Stage A entry point.

Subcommands mirror the task groups of the change
``phase1-redundancy-cartography``:

``cartography``
    Run the three redundancy axes on the small model and write the map.
``ppl``
    Evaluate the perplexity slice for the BF16 reference or an integer baseline
    arm and write a recorded run.

Every tunable comes from ``config.yaml`` (AGENTS.md -> Configuration).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import torch

from phase0_feasibility.calibration.loader import load_model, load_tokenizer
from phase1_compression.calibration.slices import calibration_slice, ppl_slice
from phase1_compression.cartography.axes import (
    run_directions_axis,
    run_structured_axis,
    run_ternary_axis,
)
from phase1_compression.cartography.record import write_map
from phase1_compression.config import Config, load_config
from phase1_compression.evaluation.accounting import stored_bytes
from phase1_compression.evaluation.gate import compression_ratio, harness_is_valid, quality_bar
from phase1_compression.evaluation.library_quant import LIBRARY_BITS, quantize_library_
from phase1_compression.evaluation.perplexity import perplexity
from phase1_compression.evaluation.quantization import (
    fake_quantize_,
    quantized_model_storage,
    reference_model_storage,
)
from phase1_compression.probe import derive_probe_plan
from phase1_compression.records import read_metrics, write_run
from phase1_compression.runtime import enable_determinism, resolve_device, set_seed
from phase1_compression.transform.selection import (
    rank_ffn_dims,
    select_lowest_fraction,
    total_units,
)
from phase1_compression.transform.structural import remove_ffn_dims_

REPO_ROOT = Path(__file__).resolve().parents[1]


def _provenance(
    config: Config, device: str, determinism: str, extra: dict[str, Any]
) -> dict[str, Any]:
    """Collect environment facts shared by every record.

    Args:
        config: Run configuration.
        device: Resolved device string.
        determinism: Determinism note from :func:`enable_determinism`.
        extra: Additional run-specific facts.

    Returns:
        The provenance mapping.
    """
    if device.startswith("cuda") and torch.cuda.is_available():
        device_name = torch.cuda.get_device_name(device)
        backend = "rocm" if torch.version.hip else "cuda"
    else:
        device_name = device
        backend = "cpu"
    return {
        "seed": config.seed,
        "device": device,
        "device_name": device_name,
        "torch_backend": backend,
        "torch_version": torch.__version__,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "determinism": determinism,
        **extra,
    }


def _config_dump(config: Config, section: str) -> dict[str, Any]:
    """Return the configuration values relevant to one subcommand.

    Args:
        config: Run configuration.
        section: ``cartography`` or ``ppl``.

    Returns:
        A flat mapping for the run record.
    """
    common = {
        "dtype": config.dtype,
        "device": config.device,
        "seed": config.seed,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "seq_len": config.seq_len,
    }
    if section == "cartography":
        return {
            **common,
            "model_id": config.cartography_model_id,
            "dense_windows_count": config.dense_windows_count,
            "kl_probe_batch_size": config.cartography.kl_probe_batch_size,
            "structured": {
                "kinds": list(config.cartography.structured.kinds),
                "kl_budget": config.cartography.structured.kl_budget,
                "max_removals": config.cartography.structured.max_removals,
            },
            "directions": {
                "n_directions": list(config.cartography.directions.n_directions),
                "layer_bands": [list(band) for band in config.cartography.directions.layer_bands],
                "kl_budget": config.cartography.directions.kl_budget,
            },
            "ternary": {
                "layer": config.cartography.ternary.layer,
                "mu": list(config.cartography.ternary.mu),
                "tau": config.cartography.ternary.tau,
                "e_x_budget": config.cartography.ternary.e_x_budget,
            },
        }
    return {
        **common,
        "model_id": config.target_model_id,
        "ppl_split": config.ppl_split,
        "ppl_windows_count": config.ppl_windows_count,
        "group_size": config.baselines.group_size,
        "bits": list(config.baselines.bits),
        "n_clip": config.baselines.n_clip,
    }


def cmd_cartography(config: Config, experiment_id: str) -> int:
    """Run the three cartography axes and write the map.

    Args:
        config: Run configuration.
        experiment_id: Map experiment id.

    Returns:
        Process exit code.
    """
    device = resolve_device(config.device)
    determinism = enable_determinism()
    tokenizer = load_tokenizer(config.cartography_model_id)
    print(f"Loading cartography model {config.cartography_model_id} on {device} ...")
    model = load_model(config.cartography_model_id, config.dtype).to(device)
    budget_s = config.cartography.budget_gpu_hours_per_axis * 3600.0
    axes = []
    for name, runner in (
        ("structured", run_structured_axis),
        ("directions", run_directions_axis),
    ):
        print(f"Axis {name} ...")
        axis = runner(config, model, tokenizer, device=device, time_budget_s=budget_s)
        print(
            f"  {name}: {len(axis.steps)} checkpoints, stopped_on_budget={axis.stopped_on_budget}"
        )
        axes.append(axis)
    print("Axis ternary ...")
    axes.append(run_ternary_axis(config, model, tokenizer, device=device))
    print(f"  ternary: {len(axes[-1].steps)} checkpoints")
    provenance = _provenance(
        config,
        device,
        determinism,
        {
            "model_id": config.cartography_model_id,
            "slice": "dense_windows",
            "windows": config.dense_windows_count,
            "probe_windows": config.cartography.kl_probe_batch_size,
            "tokens": config.dense_windows_count * config.seq_len,
        },
    )
    run_dir = write_map(
        REPO_ROOT / config.results_dir,
        experiment_id,
        axes,
        config=config,
        provenance=provenance,
    )
    print(f"Map written to {run_dir}")
    return 0


def cmd_ppl(config: Config, arm: str, n_clip: int | None = None) -> int:
    """Evaluate the perplexity slice for the reference or one baseline arm.

    Args:
        config: Run configuration.
        arm: ``reference`` or a bit width as a string.
        n_clip: Optional override of the clip-search resolution; ``1`` is pure
            absmax and is used to attribute a sanity failure to the clip search
            rather than to the bit width.

    Returns:
        Process exit code.

    Raises:
        ValueError: If ``arm`` is neither ``reference`` nor a configured bit width.
    """
    bits = None if arm == "reference" else int(arm[3:] if arm.startswith("lib") else arm)
    library = arm.startswith("lib")
    clip_steps = config.baselines.n_clip if n_clip is None else n_clip
    allowed = set(config.baselines.bits) | (set(LIBRARY_BITS) if library else set())
    if bits is not None and bits not in allowed:
        raise ValueError(f"arm {arm!r} is not in baselines.bits {config.baselines.bits}")
    device = resolve_device(config.device)
    determinism = enable_determinism()
    tokenizer = load_tokenizer(config.target_model_id)
    print(f"Loading {config.target_model_id} on {device} ...")
    if device.startswith("cuda") and torch.cuda.is_available():
        # ROCm's torch build rejects an explicit device argument here and accepts
        # only the current-device form; the machine exposes a single GPU to torch.
        torch.cuda.reset_peak_memory_stats()
    model = load_model(config.target_model_id, config.dtype).to(device)
    peak_vram_mb = (
        round(torch.cuda.max_memory_allocated() / 1024**2, 1)
        if device.startswith("cuda") and torch.cuda.is_available()
        else 0.0
    )
    reference_bytes = stored_bytes(reference_model_storage(model))
    model_config = model.config
    model_facts = {
        "num_hidden_layers": int(model_config.num_hidden_layers),
        "num_attention_heads": int(model_config.num_attention_heads),
        "num_key_value_heads": int(model_config.num_key_value_heads),
        "intermediate_size": int(model_config.intermediate_size),
        "hidden_size": int(model_config.hidden_size),
        "vocab_size": int(model_config.vocab_size),
        "max_position_embeddings": int(model_config.max_position_embeddings),
    }
    print(f"model config facts: {model_facts}")
    report = None
    if bits is not None:
        if library:
            quantize_library_(model, bits=bits, group_size=config.baselines.group_size)
        else:
            report = fake_quantize_(
                model,
                bits=bits,
                group_size=config.baselines.group_size,
                n_clip=clip_steps,
            )
        compressed_bytes = stored_bytes(
            quantized_model_storage(model, bits=bits, group_size=config.baselines.group_size)
        )
    else:
        compressed_bytes = reference_bytes
    ids, mask = ppl_slice(config, tokenizer)
    value = perplexity(model, ids, mask, batch_size=config.ppl_batch_size, device=device)
    suffix = "" if n_clip is None else f"_clip{clip_steps}"
    label = f"lib{bits}" if library else (f"b{bits}" if bits is not None else "ref")
    experiment_id = f"p1_ppl_{label}{suffix}"
    metrics: dict[str, Any] = {
        "ppl": value,
        "stored_bytes": compressed_bytes,
        "reference_bytes": reference_bytes,
        "compression_ratio": compression_ratio(reference_bytes, compressed_bytes),
        "tokens": int(mask.sum()),
        "arm": arm,
        "peak_vram_mb": peak_vram_mb,
        "model_config": model_facts,
        "library": "torchao" if library else None,
        "storage_layout_assumption": (
            "symmetric integer, group "
            f"{config.baselines.group_size}, 16-bit scales, lm_head and embeddings 16-bit"
            if bits is not None
            else None
        ),
    }
    if report is not None:
        metrics["quantized_modules"] = len(report.quantized)
        metrics["skipped_modules"] = len(report.skipped)
    provenance = _provenance(
        config,
        device,
        determinism,
        {
            "model_id": config.target_model_id,
            "split": config.ppl_split,
            "windows": config.ppl_windows_count,
            "batch_size": config.ppl_batch_size,
        },
    )
    lines = [
        f"# Perplexity run `{experiment_id}`",
        "",
        f"- arm: {arm}",
        f"- model: {config.target_model_id} ({config.dtype})",
        f"- slice: {config.dataset_id}/{config.dataset_config} {config.ppl_split}, "
        f"{config.ppl_windows_count} windows x {config.seq_len} tokens",
        f"- perplexity: {value:.4f}",
        f"- stored bytes: {compressed_bytes} "
        f"({metrics['compression_ratio']:.3f}x versus reference)",
        "",
    ]
    if report is not None:
        lines += [
            f"- quantized linear modules: {len(report.quantized)}",
            f"- skipped modules: {len(report.skipped)}",
            "",
        ]
    run_dir = write_run(
        REPO_ROOT / config.results_dir,
        experiment_id,
        config_dump={**_config_dump(config, "ppl"), "n_clip": clip_steps},
        metrics=metrics,
        provenance=provenance,
        report="\n".join(lines),
    )
    print(f"{experiment_id}: ppl={value:.4f} bytes={compressed_bytes} -> {run_dir}")
    return 0


def cmd_probe(config: Config, map_id: str, experiment_id: str) -> int:
    """Apply the frozen plan to the target model and measure quality and bytes.

    Args:
        config: Run configuration.
        map_id: Recorded cartography map the plan is derived from.
        experiment_id: Run id for the probe record.

    Returns:
        Process exit code.

    Raises:
        FileNotFoundError: If the map or the reference perplexity record is absent.
    """
    results_dir = REPO_ROOT / config.results_dir
    plan = derive_probe_plan(read_metrics(results_dir, map_id))
    print(
        f"Probe plan from {plan.map_id}: ffn_fraction={plan.ffn_fraction:.4f} "
        f"head_fraction={plan.head_fraction:.4f} blocks={plan.block_layers}"
    )
    reference_record = read_metrics(results_dir, "p1_ppl_ref")
    reference_ppl = float(reference_record["ppl"])
    reference_bytes = int(reference_record["stored_bytes"])

    device = resolve_device(config.device)
    determinism = enable_determinism()
    tokenizer = load_tokenizer(config.target_model_id)
    print(f"Loading {config.target_model_id} on {device} ...")
    model = load_model(config.target_model_id, config.dtype).to(device)
    calibration_ids, calibration_mask = calibration_slice(config, tokenizer)
    ranking = rank_ffn_dims(
        model,
        calibration_ids,
        calibration_mask,
        batch_size=config.cartography.kl_probe_batch_size,
        device=device,
    )
    selection = select_lowest_fraction(ranking, plan.ffn_fraction)
    selected_units = total_units(selection)
    if selected_units == 0:
        raise ValueError(
            f"probe plan from map {plan.map_id!r} removes zero units "
            f"(ffn_fraction={plan.ffn_fraction}); a no-op probe must not be recorded — "
            "re-derive the map with n_total recorded per step"
        )
    print(f"Selected {selected_units} FFN dimensions across {len(selection)} layers")
    summary = remove_ffn_dims_(model, selection)
    compressed_bytes = stored_bytes(reference_model_storage(model))
    ids, mask = ppl_slice(config, tokenizer)
    probe_ppl = perplexity(model, ids, mask, batch_size=config.ppl_batch_size, device=device)
    quality = quality_bar(
        reference_ppl,
        probe_ppl,
        {"perplexity": reference_ppl},
        {"perplexity": probe_ppl},
        config.gate,
    )
    metrics: dict[str, Any] = {
        "arm": "probe",
        "map_id": plan.map_id,
        "ffn_fraction": plan.ffn_fraction,
        "head_fraction": plan.head_fraction,
        "block_layers": list(plan.block_layers),
        "dims_removed": summary["dims_removed"],
        "params_removed": summary["params_removed"],
        "reference_ppl": reference_ppl,
        "probe_ppl": probe_ppl,
        "ppl_rel_delta": quality.ppl_rel_delta,
        "stored_bytes": compressed_bytes,
        "reference_bytes": reference_bytes,
        "compression_ratio": compression_ratio(reference_bytes, compressed_bytes),
        "ppl_quality_passed": quality.passed,
        "benchmarks_pending": True,
    }
    provenance = _provenance(
        config,
        device,
        determinism,
        {
            "model_id": config.target_model_id,
            "map_id": plan.map_id,
            "split": config.ppl_split,
            "windows": config.ppl_windows_count,
        },
    )
    lines = [
        f"# Combined probe `{experiment_id}` (Phase 1 Stage A)",
        "",
        f"Plan derived from map `{plan.map_id}` and frozen before the run:",
        "",
        *(f"- {note}" for note in plan.notes),
        "",
        f"- FFN dimensions removed: {summary['dims_removed']} "
        f"({summary['params_removed']} parameters)",
        f"- stored bytes: {compressed_bytes} of {reference_bytes} "
        f"({metrics['compression_ratio']:.4f}x)",
        f"- perplexity: {reference_ppl:.4f} -> {probe_ppl:.4f} "
        f"(relative {quality.ppl_rel_delta:+.5f}, gate {config.gate.ppl_rel_tol})",
        f"- perplexity-only quality bar: {'pass' if quality.passed else 'fail'}",
        "",
        "Benchmarks were not measured in this run, so no Stage A verdict is issued; "
        "the benchmark half of the quality bar and the matched-quality integer baseline "
        "comparison remain open (spec phase1/compression-gate R1, R3).",
        "",
    ]
    run_dir = write_run(
        results_dir,
        experiment_id,
        config_dump={**_config_dump(config, "ppl"), "probe": {"ffn_fraction": plan.ffn_fraction}},
        metrics=metrics,
        provenance=provenance,
        report="\n".join(lines),
    )
    print(
        f"{experiment_id}: ppl {reference_ppl:.4f} -> {probe_ppl:.4f} "
        f"({quality.ppl_rel_delta:+.5f}), bytes {compressed_bytes} "
        f"({metrics['compression_ratio']:.4f}x) -> {run_dir}"
    )
    return 0


def cmd_sweep(config: Config, experiment_id: str) -> int:
    """Sweep cumulative FFN-dimension removal end to end on the target model.

    Each fraction is measured on a freshly loaded model so the removed indices are
    the pristine model's; the ranking is computed once. The recorded points are the
    end-to-end arbiter the cartography KL proxy is checked against.

    Args:
        config: Run configuration.
        experiment_id: Run id for the sweep record.

    Returns:
        Process exit code.

    Raises:
        FileNotFoundError: If the reference perplexity record is absent.
        ValueError: If the configured fractions are not strictly increasing.
    """
    results_dir = REPO_ROOT / config.results_dir
    reference_record = read_metrics(results_dir, "p1_ppl_ref")
    reference_ppl = float(reference_record["ppl"])
    reference_bytes = int(reference_record["stored_bytes"])
    device = resolve_device(config.device)
    determinism = enable_determinism()
    tokenizer = load_tokenizer(config.target_model_id)
    calibration_ids, calibration_mask = calibration_slice(config, tokenizer)
    ids, mask = ppl_slice(config, tokenizer)
    points: list[dict[str, Any]] = []
    ranking: list[tuple[int, int, float]] | None = None
    previous = 0.0
    for fraction in config.probe.ffn_sweep_fractions:
        if fraction <= previous:
            raise ValueError(
                f"probe.ffn_sweep_fractions must be strictly increasing, got "
                f"{config.probe.ffn_sweep_fractions}"
            )
        previous = fraction
        print(f"Loading {config.target_model_id} for fraction {fraction} ...")
        model = load_model(config.target_model_id, config.dtype).to(device)
        if ranking is None:
            ranking = rank_ffn_dims(
                model,
                calibration_ids,
                calibration_mask,
                batch_size=config.cartography.kl_probe_batch_size,
                device=device,
            )
        selection = select_lowest_fraction(ranking, fraction)
        summary = remove_ffn_dims_(model, selection)
        stored = stored_bytes(reference_model_storage(model))
        value = perplexity(model, ids, mask, batch_size=config.ppl_batch_size, device=device)
        delta = (value - reference_ppl) / reference_ppl
        points.append(
            {
                "fraction": fraction,
                "dims_removed": summary["dims_removed"],
                "params_removed": summary["params_removed"],
                "stored_bytes": stored,
                "compression_ratio": compression_ratio(reference_bytes, stored),
                "ppl": value,
                "ppl_rel_delta": delta,
                "ppl_quality_passed": delta <= config.gate.ppl_rel_tol,
            }
        )
        print(
            f"  fraction={fraction}: dims={summary['dims_removed']} ratio="
            f"{points[-1]['compression_ratio']:.4f}x ppl={value:.4f} delta={delta:+.5f}"
        )
        del model
        if device.startswith("cuda") and torch.cuda.is_available():
            # ROCm's caching allocator holds the freed block otherwise, and the next
            # model load then exceeds VRAM (observed as an OOM between points).
            torch.cuda.empty_cache()
        _write_partial_sweep(results_dir, experiment_id, points, reference_ppl, reference_bytes)
    metrics: dict[str, Any] = {
        "arm": "ffn_sweep",
        "reference_ppl": reference_ppl,
        "reference_bytes": reference_bytes,
        "gate_min_compression": config.gate.min_compression,
        "points": points,
    }
    provenance = _provenance(
        config,
        device,
        determinism,
        {
            "model_id": config.target_model_id,
            "split": config.ppl_split,
            "windows": config.ppl_windows_count,
        },
    )
    lines = [
        f"# FFN-dimension removal sweep `{experiment_id}` (Phase 1 Stage A)",
        "",
        "Cumulative removal of the lowest-scoring FFN intermediate dimensions, each "
        "point measured on a freshly loaded model. The cartography KL proxy stops at "
        "5%; this sweep is the end-to-end arbiter.",
        "",
        f"- reference perplexity: {reference_ppl:.4f} over {reference_bytes} stored bytes",
        f"- pre-registered compression floor: {config.gate.min_compression}x",
        f"- perplexity bar: relative increase <= {config.gate.ppl_rel_tol}",
        "",
        "| fraction | dims removed | bytes | ratio | ppl | ppl delta | ppl bar |",
        "|---|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {point['fraction']:.2f} | {point['dims_removed']} | {point['stored_bytes']} | "
        f"{point['compression_ratio']:.4f}x | {point['ppl']:.4f} | "
        f"{point['ppl_rel_delta']:+.5f} | "
        f"{'pass' if point['ppl_quality_passed'] else 'fail'} |"
        for point in points
    ]
    lines += [
        "",
        "Benchmarks were not measured, so these are perplexity-only quality reads; "
        "the Stage A verdict additionally requires the benchmark half of the quality "
        "bar and the matched-quality integer baseline (spec R1, R3).",
        "",
    ]
    run_dir = write_run(
        results_dir,
        experiment_id,
        config_dump={
            **_config_dump(config, "ppl"),
            "probe": {"ffn_sweep_fractions": list(config.probe.ffn_sweep_fractions)},
        },
        metrics=metrics,
        provenance=provenance,
        report="\n".join(lines),
    )
    print(f"Sweep written to {run_dir}")
    return 0


def _write_partial_sweep(
    results_dir: Path,
    experiment_id: str,
    points: list[dict[str, Any]],
    reference_ppl: float,
    reference_bytes: int,
) -> None:
    """Persist sweep points as they are produced, so a crash keeps evidence.

    Args:
        results_dir: Repository ``results`` directory.
        experiment_id: Run id.
        points: Points measured so far.
        reference_ppl: BF16 reference perplexity.
        reference_bytes: BF16 stored bytes.
    """
    run_dir = results_dir / experiment_id
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "experiment_id": experiment_id,
        "arm": "ffn_sweep",
        "complete": False,
        "reference_ppl": reference_ppl,
        "reference_bytes": reference_bytes,
        "points": points,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def cmd_verdict(config: Config, experiment_id: str) -> int:
    """Apply spec R4 and, when the harness is valid, the single decision rule.

    Args:
        config: Run configuration.
        experiment_id: Run id for the verdict record.

    Returns:
        Process exit code.

    Raises:
        FileNotFoundError: If the reference or sanity run is absent.
    """
    results_dir = REPO_ROOT / config.results_dir
    reference = read_metrics(results_dir, "p1_ppl_ref")
    reference_ppl = float(reference["ppl"])
    reference_bytes = int(reference["stored_bytes"])
    sanity_id = f"p1_ppl_b{config.baselines.sanity_bits}"
    sanity = read_metrics(results_dir, sanity_id)
    valid, sanity_delta = harness_is_valid(
        reference_ppl, float(sanity["ppl"]), config.baselines.sanity_ppl_rel_tol
    )
    arms: list[dict[str, Any]] = []
    for bits in config.baselines.bits:
        record = read_metrics(results_dir, f"p1_ppl_b{bits}")
        arms.append(
            {
                "arm": f"g128_{bits}bit",
                "stored_bytes": int(record["stored_bytes"]),
                "ppl": float(record["ppl"]),
                "ppl_rel_delta": (float(record["ppl"]) - reference_ppl) / reference_ppl,
            }
        )
    if not valid:
        label = "VOID"
        reason = (
            f"spec R4: the {config.baselines.sanity_bits}-bit sanity baseline increased "
            f"perplexity by {sanity_delta:+.5f} > {config.baselines.sanity_ppl_rel_tol}; "
            "the harness is invalid for this stage and no GO/YELLOW/RED may be issued"
        )
    else:
        label = "PENDING"
        reason = "harness valid, but the benchmark half of the quality bar was not measured"
    metrics: dict[str, Any] = {
        "arm": "verdict",
        "verdict": label,
        "reason": reason,
        "sanity_delta": sanity_delta,
        "sanity_tolerance": config.baselines.sanity_ppl_rel_tol,
        "reference_ppl": reference_ppl,
        "reference_bytes": reference_bytes,
        "baseline_arms": arms,
    }
    provenance = _provenance(config, "cpu", "not applicable: reads recorded runs only", {})
    lines = [
        f"# Stage A verdict record `{experiment_id}`",
        "",
        f"- **{label}** — {reason}",
        f"- reference perplexity: {reference_ppl:.4f} ({reference_bytes} bytes)",
        f"- sanity delta: {sanity_delta:+.5f} against tolerance "
        f"{config.baselines.sanity_ppl_rel_tol}",
        "",
        "| arm | bytes | ratio | ppl | ppl delta |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {arm['arm']} | {arm['stored_bytes']} | "
        f"{compression_ratio(reference_bytes, int(arm['stored_bytes'])):.3f}x | "
        f"{arm['ppl']:.4f} | {arm['ppl_rel_delta']:+.5f} |"
        for arm in arms
    ]
    lines.append("")
    run_dir = write_run(
        results_dir,
        experiment_id,
        config_dump=_config_dump(config, "ppl"),
        metrics=metrics,
        provenance=provenance,
        report="\n".join(lines),
    )
    print(f"{experiment_id}: {label} — {reason}")
    print(f"recorded at {run_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and dispatch a Phase 1 Stage A subcommand.

    Args:
        argv: Argument list; defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = argparse.ArgumentParser(description="Phase 1 Stage A harness")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)
    cart = sub.add_parser("cartography", help="run the three redundancy axes")
    cart.add_argument("--experiment-id", default="p1_map")
    ppl = sub.add_parser("ppl", help="evaluate the perplexity slice for one arm")
    ppl.add_argument("--arm", required=True, help="reference, or a bit width such as 4")
    ppl.add_argument(
        "--n-clip",
        type=int,
        default=None,
        help="clip-search resolution override; 1 selects pure absmax (diagnostic)",
    )
    probe = sub.add_parser("probe", help="apply the frozen map plan to the target model")
    probe.add_argument("--map", required=True, help="recorded cartography map experiment id")
    probe.add_argument("--experiment-id", default="p1_probe")
    sweep = sub.add_parser("sweep", help="sweep cumulative FFN removal end to end")
    sweep.add_argument("--experiment-id", default="p1_ffn_sweep")
    verdict = sub.add_parser("verdict", help="apply spec R4 and the decision rule")
    verdict.add_argument("--experiment-id", default="p1_verdict")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    set_seed(config.seed)
    if args.command == "cartography":
        return cmd_cartography(config, str(args.experiment_id))
    if args.command == "probe":
        return cmd_probe(config, str(args.map), str(args.experiment_id))
    if args.command == "sweep":
        return cmd_sweep(config, str(args.experiment_id))
    if args.command == "verdict":
        return cmd_verdict(config, str(args.experiment_id))
    return cmd_ppl(config, str(args.arm), args.n_clip)


if __name__ == "__main__":
    sys.exit(main())
