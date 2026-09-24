"""Phase 0 sweep entry point (change phase0-calibration-robustness, task 4.1).

For every configured calibration slice mode and decoder layer the harness:

1. builds the slice (``dense_windows`` / ``many_documents`` / ``docs128``),
2. splits it deterministically into a contiguous fit and held-out portion,
3. captures the target projection inputs for both portions,
4. runs the reduced sweep ``mu x tau x importance`` (reorder off) plus the
   ``mu=1`` baseline, and writes one recorded run per configuration under
   ``<results_dir>/<experiment-id>/``.

The GREEN/YELLOW/RED verdict is computed from ``E_x`` on the fit portion only;
``E_x_heldout`` is recorded as a diagnostic and does not drive the verdict
(design.md D6).
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import sys
import time
from collections.abc import Callable
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch

from phase0_feasibility.analysis.metrics import (
    LayerMetrics,
    compute_metrics,
    metric_validity_check,
)
from phase0_feasibility.analysis.report import (
    decide_verdict,
    verdict_from_metrics_dict,
    write_run,
)
from phase0_feasibility.calibration.hooks import (
    activation_second_moment,
    capture_gate_inputs,
)
from phase0_feasibility.calibration.loader import (
    DOCS128_COUNT,
    build_dense_windows,
    encode_documents,
    layer_module_path,
    load_documents,
    load_model,
    load_text_stream,
    load_tokenizer,
    split_slice,
)
from phase0_feasibility.config import Config, load_config
from phase0_feasibility.decomposition.baseline import symmetric_baseline
from phase0_feasibility.decomposition.core import (
    TernaryFactors,
    decompose_ternary,
    normalized_importance,
)
from phase0_feasibility.decomposition.reordering import cosine_greedy_order

REPO_ROOT = Path(__file__).resolve().parents[1]


def set_seed(seed: int) -> None:
    """Seed torch, numpy and random from the config (determinism policy)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _enable_determinism() -> str:
    """Enable deterministic algorithms where the CPU path supports it.

    Returns:
        A short note describing whether the strict mode was enabled.
    """
    try:
        torch.use_deterministic_algorithms(True)
    except RuntimeError as exc:  # pragma: no cover - platform dependent
        return f"deterministic algorithms unavailable: {exc}"
    return "torch.use_deterministic_algorithms(True)"


def _experiment_id(
    slice_mode: str,
    layer: int,
    mu: int,
    tau: float,
    importance: bool,
    reorder: bool,
    n_inner: int,
) -> str:
    """Return the run id ``s-<mode>_l<layer>_mu<mu>_tau<tau>_imp..._re..._ni<n>``."""
    imp = "on" if importance else "off"
    reo = "on" if reorder else "off"
    return f"s-{slice_mode}_l{layer}_mu{mu}_tau{tau:g}_imp{imp}_re{reo}_ni{n_inner}"


def _time_decomposition(fit: Callable[[], TernaryFactors]) -> tuple[TernaryFactors, float]:
    """Run a decomposition callable and return its factors and runtime."""
    start = time.perf_counter()
    factors = fit()
    runtime_s = time.perf_counter() - start
    return factors, runtime_s


def _build_slice(
    config: Config, slice_mode: str, tokenizer: Any
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the token id and attention mask tensors for one slice mode.

    Args:
        config: Run configuration; supplies the dataset, counts and ``seq_len``.
        slice_mode: One of ``dense_windows``, ``many_documents``, ``docs128``.
        tokenizer: HuggingFace tokenizer (or duck-typed equivalent).

    Returns:
        ``(input_ids, attention_mask)``, both int64 ``[n_items, seq_len]``.

    Raises:
        ValueError: If ``slice_mode`` is unknown (unknown modes are also rejected
            by the config loader).
    """
    if slice_mode == "docs128":
        documents = load_documents(
            config.dataset_id, config.dataset_config, config.dataset_split, DOCS128_COUNT
        )
        return encode_documents(tokenizer, documents, config.seq_len)
    if slice_mode == "many_documents":
        documents = load_documents(
            config.dataset_id,
            config.dataset_config,
            config.dataset_split,
            config.many_documents_count,
        )
        return encode_documents(tokenizer, documents, config.seq_len)
    if slice_mode == "dense_windows":
        text = load_text_stream(config.dataset_id, config.dataset_config, config.dataset_split)
        return build_dense_windows(
            tokenizer, text, seq_len=config.seq_len, n_windows=config.dense_windows_count
        )
    raise ValueError(f"unknown slice mode {slice_mode!r}")


def _capture_layer(
    model: Any,
    config: Config,
    layer: int,
    *,
    fit_ids: torch.Tensor,
    fit_mask: torch.Tensor,
    heldout_ids: torch.Tensor,
    heldout_mask: torch.Tensor,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Capture fit/held-out activations and the weight for one decoder layer.

    Args:
        model: Causal LM (HuggingFace boundary type) on ``device``.
        config: Run configuration (module name, batch size).
        layer: Decoder layer index.
        fit_ids: Fit token ids ``[fit_items, seq_len]`` int64.
        fit_mask: Fit attention mask ``[fit_items, seq_len]`` int64.
        heldout_ids: Held-out token ids ``[heldout_items, seq_len]`` int64.
        heldout_mask: Held-out attention mask ``[heldout_items, seq_len]`` int64.
        device: Resolved device string.

    Returns:
        ``(fit_activations, heldout_activations, weight)``, all float32 on
        ``device``; activations ``[tokens, in]``, weight ``[out, in]``.
    """
    path = layer_module_path(layer, config.target_module)
    fit_activations = capture_gate_inputs(
        model,
        fit_ids.to(device),
        fit_mask.to(device),
        layer_path=path,
        batch_size=config.calibration_batch_size,
    )
    heldout_activations = capture_gate_inputs(
        model,
        heldout_ids.to(device),
        heldout_mask.to(device),
        layer_path=path,
        batch_size=config.calibration_batch_size,
    )
    weight = model.get_submodule(path).weight.detach().to(torch.float32).to(device)
    return fit_activations, heldout_activations, weight


def decompose_for_run(
    weight: torch.Tensor,
    *,
    config: Config,
    mu: int,
    tau: float,
    importance: torch.Tensor | None,
    n_inner: int,
) -> TernaryFactors:
    """Decompose one layer under a single sweep configuration.

    Args:
        weight: Weight matrix in the evaluation space, ``[out, in]`` float32.
        config: Run configuration.
        mu: Rank multiplier for this run.
        tau: Ternary threshold for this run.
        importance: Optional normalised importance weights ``[in]``.
        n_inner: Inner alternating iterations for this run.

    Returns:
        The fitted :class:`TernaryFactors`.
    """
    return decompose_ternary(
        weight,
        mu=float(mu),
        tau=tau,
        batched=True,
        b=config.b,
        importance=importance,
        n_inner=n_inner,
        eps=config.eps,
    )


def _prepare_space(
    weight: torch.Tensor,
    activations: torch.Tensor,
    reorder: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Apply the column permutation to both the weight and the activations.

    Args:
        weight: Weight ``[out, in]`` in original column order.
        activations: Activations ``[tokens, in]`` in original column order.
        reorder: Whether to reorder.

    Returns:
        ``(weight, activations, perm)``; ``perm`` is ``None`` when reorder is off.
    """
    if not reorder:
        return weight, activations, None
    perm, _inv_perm = cosine_greedy_order(weight)
    return weight[:, perm], activations[:, perm], perm


def _provenance(
    config: Config,
    *,
    layer: int,
    slice_mode: str,
    slice_items: int,
    slice_tokens: int,
    fit_items: int,
    heldout_items: int,
    device: str,
    determinism_note: str,
) -> dict[str, object]:
    """Collect environment and slice facts for the run record."""
    import datasets
    import transformers

    if device.startswith("cuda"):
        device_name = torch.cuda.get_device_name(device)
        backend = "rocm" if torch.version.hip else "cuda"
    else:
        device_name = device
        backend = "cpu"

    return {
        "model_id": config.model_id,
        "dtype": config.dtype,
        "layer": layer,
        "target_module": config.target_module,
        "seed": config.seed,
        "dataset_id": config.dataset_id,
        "dataset_config": config.dataset_config,
        "dataset_split": config.dataset_split,
        "slice_mode": slice_mode,
        "dense_windows_count": config.dense_windows_count,
        "many_documents_count": config.many_documents_count,
        "seq_len": config.seq_len,
        "eval_fraction": config.eval_fraction,
        "slice_items": slice_items,
        "slice_tokens": slice_tokens,
        "fit_items": fit_items,
        "heldout_items": heldout_items,
        "device": device,
        "device_name": device_name,
        "torch_backend": backend,
        "determinism": determinism_note,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "datasets_version": datasets.__version__,
        "numpy_version": np.__version__,
    }


def _resolve_device(requested: str) -> str:
    """Resolve the configured device to a concrete torch device string.

    Args:
        requested: ``auto`` (cuda when available, else cpu), or an explicit
            ``cpu`` / ``cuda`` / ``cuda:N`` string.

    Returns:
        The resolved device string.

    Raises:
        ValueError: If an explicit cuda device is requested but unavailable or
            unknown.
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


def _run_configuration(
    config: Config,
    *,
    weight: torch.Tensor,
    fit_activations: torch.Tensor,
    heldout_activations: torch.Tensor,
    slice_mode: str,
    layer: int,
    mu: int,
    tau: float,
    importance_on: bool,
    reorder: bool,
    n_inner: int,
    perm: torch.Tensor | None,
    results_dir: Path,
    provenance: dict[str, object],
    validity: dict[str, object],
    baseline_e_x: float,
    baseline_verdict: str,
) -> LayerMetrics:
    """Fit, score and record one sweep configuration.

    The reconstruction is fitted on the fit activations; ``E_x_fit`` and
    ``E_x_heldout`` are both computed from that single reconstruction and both
    recorded. The verdict is taken from the fit portion.
    """
    importance = None
    if importance_on:
        importance = normalized_importance(
            activation_second_moment(fit_activations), config.importance_lambda
        )

    experiment_id = _experiment_id(slice_mode, layer, mu, tau, importance_on, reorder, n_inner)
    factors, runtime_s = _time_decomposition(
        lambda: decompose_for_run(
            weight, config=config, mu=mu, tau=tau, importance=importance, n_inner=n_inner
        )
    )
    metrics_fit = compute_metrics(fit_activations, weight, factors, float(mu), runtime_s)
    metrics_heldout = compute_metrics(heldout_activations, weight, factors, float(mu), runtime_s)
    run_dir = write_run(
        results_dir,
        experiment_id,
        config,
        metrics_fit,
        heldout_metrics=metrics_heldout,
        provenance=provenance,
        extra={
            "mu": mu,
            "tau": tau,
            "importance": importance_on,
            "reorder": reorder,
            "n_inner": n_inner,
            "slice_mode": slice_mode,
            "layer": layer,
            "evaluation_space": "reordered" if reorder else "original",
            "k": factors.k,
            "stall_steps": list(factors.stall_steps),
            "validity": validity,
            "baseline_e_x": baseline_e_x,
            "baseline_verdict": baseline_verdict,
        },
        config_overrides={
            "mu": mu,
            "tau": tau,
            "reorder": reorder,
            "importance": importance_on,
            "n_inner": n_inner,
        },
    )
    if perm is not None:
        torch.save(perm, run_dir / "permutation.pt")
    print(
        f"  {experiment_id}: verdict={decide_verdict(metrics_fit)} "
        f"E_x_fit={metrics_fit.e_x:.4f} E_x_heldout={metrics_heldout.e_x:.4f} "
        f"sparsity={metrics_fit.sparsity:.3f} bpw={metrics_fit.bpw_eff:.3f} "
        f"runtime={runtime_s:.1f}s"
    )
    return metrics_fit


def verify_reproducibility(config: Config, experiment_id: str) -> int:
    """Re-run one recorded configuration and compare against its ``metrics.json``.

    The verdict is first recomputed from the recorded fit metric mapping (cheap,
    no forward pass), then the configuration is re-run from a fresh process and
    its metrics are compared within ``config.tolerance.reproduce_rtol``.

    Args:
        config: Loaded configuration.
        experiment_id: Recorded run directory name under ``results_dir``.

    Returns:
        Process exit code (0 on success).

    Raises:
        FileNotFoundError: If the recorded run does not exist.
        AssertionError: If the verdict or a metric diverges beyond tolerance.
    """
    run_dir = REPO_ROOT / config.results_dir / experiment_id
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"no recorded run at {metrics_path}")
    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    recorded = payload["metrics_fit"]
    recorded_verdict = payload["verdict"]

    recomputed_verdict = verdict_from_metrics_dict(recorded)
    if recomputed_verdict != recorded_verdict:
        raise AssertionError(
            f"verdict drift for {experiment_id}: recorded={recorded_verdict} "
            f"recomputed={recomputed_verdict}"
        )
    print(f"verdict recomputed from metrics.json: {recomputed_verdict}")

    slice_mode = str(payload["slice_mode"])
    layer = int(payload["layer"])
    mu = int(payload["mu"])
    tau = float(payload["tau"])
    importance_on = bool(payload["importance"])
    reorder = bool(payload["reorder"])
    n_inner = int(payload.get("n_inner", config.n_inner))

    set_seed(config.seed)
    _enable_determinism()
    device = _resolve_device(config.device)
    tokenizer = load_tokenizer(config.model_id)
    model = load_model(config.model_id, config.dtype).to(device)
    input_ids, attention_mask = _build_slice(config, slice_mode, tokenizer)
    fit_ids, fit_mask, heldout_ids, heldout_mask = split_slice(
        input_ids, attention_mask, config.eval_fraction
    )
    fit_activations, _heldout_activations, weight = _capture_layer(
        model,
        config,
        layer,
        fit_ids=fit_ids,
        fit_mask=fit_mask,
        heldout_ids=heldout_ids,
        heldout_mask=heldout_mask,
        device=device,
    )
    if reorder:
        perm, _inv_perm = cosine_greedy_order(weight)
        space_weight, space_fit = weight[:, perm], fit_activations[:, perm]
    else:
        space_weight, space_fit = weight, fit_activations
    importance = None
    if importance_on:
        importance = normalized_importance(
            activation_second_moment(space_fit), config.importance_lambda
        )
    factors, runtime_s = _time_decomposition(
        lambda: decompose_for_run(
            space_weight, config=config, mu=mu, tau=tau, importance=importance, n_inner=n_inner
        )
    )
    metrics = compute_metrics(space_fit, space_weight, factors, float(mu), runtime_s)

    rtol = config.tolerance.reproduce_rtol
    mismatches: list[str] = []
    for key in ("e_x", "energy", "sparsity", "bpw_eff"):
        old = float(recorded[key])
        new = float(getattr(metrics, key))
        if not math.isclose(new, old, rel_tol=rtol, abs_tol=1e-9):
            mismatches.append(f"{key}: recorded={old!r} new={new!r}")
    if mismatches:
        raise AssertionError(
            f"reproducibility mismatch for {experiment_id}: " + "; ".join(mismatches)
        )
    print(
        f"reproduced {experiment_id}: E_x_fit={metrics.e_x!r} sparsity={metrics.sparsity!r} "
        f"bpw={metrics.bpw_eff!r} (rtol={rtol})"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the Phase 0 reduced sweep and record every configuration."""
    parser = argparse.ArgumentParser(description="Phase 0 feasibility sweep")
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--verify",
        metavar="EXPERIMENT_ID",
        help="re-run one recorded configuration from a fresh process and compare metrics",
    )
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.verify is not None:
        return verify_reproducibility(config, args.verify)

    set_seed(config.seed)
    determinism_note = _enable_determinism()
    device = _resolve_device(config.device)
    if device.startswith("cuda"):
        backend = "rocm" if torch.version.hip else ("cuda" if torch.version.cuda else "unknown")
        print(f"Device: {device} — {torch.cuda.get_device_name(device)} (torch backend: {backend})")

    print(f"Loading {config.model_id} ...")
    tokenizer = load_tokenizer(config.model_id)
    model = load_model(config.model_id, config.dtype).to(device)
    results_dir = REPO_ROOT / config.results_dir

    for slice_mode in config.slices:
        input_ids, attention_mask = _build_slice(config, slice_mode, tokenizer)
        slice_items = int(input_ids.shape[0])
        slice_tokens = int(attention_mask.sum())
        fit_ids, fit_mask, heldout_ids, heldout_mask = split_slice(
            input_ids, attention_mask, config.eval_fraction
        )
        fit_items = int(fit_ids.shape[0])
        heldout_items = int(heldout_ids.shape[0])
        print(
            f"Slice {slice_mode}: {slice_items} items, {slice_tokens} tokens; "
            f"fit {fit_items} items, held-out {heldout_items} items"
        )
        for layer in config.layers:
            fit_activations, heldout_activations, weight = _capture_layer(
                model,
                config,
                layer,
                fit_ids=fit_ids,
                fit_mask=fit_mask,
                heldout_ids=heldout_ids,
                heldout_mask=heldout_mask,
                device=device,
            )
            provenance = _provenance(
                config,
                layer=layer,
                slice_mode=slice_mode,
                slice_items=slice_items,
                slice_tokens=slice_tokens,
                fit_items=fit_items,
                heldout_items=heldout_items,
                device=device,
                determinism_note=determinism_note,
            )
            print(
                f"Layer {layer}: fit {int(fit_activations.shape[0])} tokens, "
                f"held-out {int(heldout_activations.shape[0])} tokens; "
                f"weight {tuple(weight.shape)}"
            )

            good_passes, bad_fails, good_e_x, bad_e_x = metric_validity_check(
                fit_activations,
                weight,
                seed=config.seed,
                good_tol=config.tolerance.e_x_atol,
                ceiling_tol=1e-6,
            )
            validity: dict[str, object] = {
                "good_e_x": good_e_x,
                "bad_e_x": bad_e_x,
                "good_passes": good_passes,
                "bad_fails": bad_fails,
            }
            if not (good_passes and bad_fails):
                raise RuntimeError(f"metric validity check failed: {validity}")
            print(f"Validity: good E_x={good_e_x:.2e}, destroyed E_x={bad_e_x:.3f}")

            print(f"Running mu=1 baseline for {slice_mode}/layer {layer} ...")
            baseline_factors, baseline_runtime = _time_decomposition(
                partial(
                    symmetric_baseline,
                    weight,
                    tau=config.tau,
                    batched=True,
                    b=config.b,
                    n_inner=config.n_inner,
                    eps=config.eps,
                )
            )
            baseline_fit = compute_metrics(
                fit_activations, weight, baseline_factors, 1.0, baseline_runtime
            )
            baseline_heldout = compute_metrics(
                heldout_activations, weight, baseline_factors, 1.0, baseline_runtime
            )
            baseline_verdict = decide_verdict(baseline_fit)
            baseline_id = _experiment_id(
                slice_mode, layer, 1, config.tau, False, False, config.n_inner
            )
            write_run(
                results_dir,
                baseline_id,
                config,
                baseline_fit,
                heldout_metrics=baseline_heldout,
                provenance=provenance,
                extra={
                    "mu": 1,
                    "tau": config.tau,
                    "importance": False,
                    "reorder": False,
                    "n_inner": config.n_inner,
                    "slice_mode": slice_mode,
                    "layer": layer,
                    "evaluation_space": "original",
                    "k": baseline_factors.k,
                    "stall_steps": list(baseline_factors.stall_steps),
                    "validity": validity,
                },
                config_overrides={
                    "mu": 1,
                    "tau": config.tau,
                    "reorder": False,
                    "importance": False,
                    "n_inner": config.n_inner,
                },
            )
            print(
                f"  {baseline_id}: verdict={baseline_verdict} "
                f"E_x_fit={baseline_fit.e_x:.4f} E_x_heldout={baseline_heldout.e_x:.4f} "
                f"sparsity={baseline_fit.sparsity:.3f} bpw={baseline_fit.bpw_eff:.3f}"
            )

            print(f"Running reduced sweep for {slice_mode}/layer {layer} ...")
            for reorder_on in config.sweep.reorder:
                space_weight, space_fit, perm = _prepare_space(weight, fit_activations, reorder_on)
                if perm is None:
                    space_heldout = heldout_activations
                else:
                    space_heldout = heldout_activations[:, perm]
                for mu in config.sweep.mu:
                    for tau in config.sweep.tau:
                        for importance_on in config.sweep.importance:
                            for n_inner in config.sweep.n_inner:
                                _run_configuration(
                                    config,
                                    weight=space_weight,
                                    fit_activations=space_fit,
                                    heldout_activations=space_heldout,
                                    slice_mode=slice_mode,
                                    layer=layer,
                                    mu=mu,
                                    tau=tau,
                                    importance_on=importance_on,
                                    reorder=reorder_on,
                                    n_inner=n_inner,
                                    perm=perm,
                                    results_dir=results_dir,
                                    provenance=provenance,
                                    validity=validity,
                                    baseline_e_x=baseline_fit.e_x,
                                    baseline_verdict=baseline_verdict,
                                )
    print(f"Done. Records under {results_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
