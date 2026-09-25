"""The three cartography axes (spec phase1/redundancy-cartography R3-R5).

Axis A ablates whole units (deliberately deviating from literal per-unit removal
for the fine-grained kinds: with 4864 intermediate dimensions per layer a
per-step curve is unaffordable, so those kinds are ranked once by a stated
importance score and the proxy is recorded at cumulative checkpoints; the block
kind is measured greedily per step). Axis B removes least-significant activation
directions by weight orthogonalization. Axis C probes the TCD ternary
representation itself.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from typing import Any

import torch
from torch import Tensor, nn

from phase0_feasibility.analysis.metrics import layer_output_error
from phase0_feasibility.calibration.hooks import capture_gate_inputs
from phase0_feasibility.decomposition.core import TernaryFactors, decompose_ternary
from phase1_compression.calibration.slices import calibration_slice
from phase1_compression.cartography.record import AxisMap, Step
from phase1_compression.evaluation.kl import mean_kl

# Cumulative removal checkpoints for the ranked fine-grained kinds, as fractions
# of the total number of units of that kind.
_RANKED_FRACTIONS: tuple[float, ...] = (0.05, 0.10, 0.20, 0.40)


def probe_slice(config: Any, tokenizer: Any) -> tuple[Tensor, Tensor]:
    """Return the benign probe slice used by the KL proxy.

    Args:
        config: Run configuration.
        tokenizer: HuggingFace tokenizer.

    Returns:
        ``(input_ids, attention_mask)`` with ``kl_probe_batch_size`` windows.
    """
    ids, mask = calibration_slice(config, tokenizer)
    n = min(int(config.cartography.kl_probe_batch_size), int(ids.shape[0]))
    return ids[:n], mask[:n]


def residual_e_x(
    reference_model: Any,
    candidate_model: Any,
    input_ids: Tensor,
    attention_mask: Tensor,
    *,
    layer: int,
    batch_size: int,
    device: str,
) -> float:
    """Return the relative residual-stream error at ``layer`` between two models.

    This is the Phase 0 ``E_x`` definition applied to the residual stream: the
    cross-check required by spec R2.

    Args:
        reference_model: Unmodified model.
        candidate_model: Model with removals applied.
        input_ids: Token ids ``[items, seq_len]``.
        attention_mask: Mask ``[items, seq_len]``.
        layer: Decoder layer whose input residual stream is compared.
        batch_size: Items per forward pass.
        device: Torch device string.

    Returns:
        ``||H_cand - H_ref||_F / ||H_ref||_F`` over the probe slice.

    Raises:
        ValueError: If the models expose no hidden states.
    """
    diffs_sq = 0.0
    ref_sq = 0.0
    for start in range(0, input_ids.shape[0], batch_size):
        ids = input_ids[start : start + batch_size].to(device)
        mask = attention_mask[start : start + batch_size].to(device)
        with torch.inference_mode():
            ref_out = reference_model(input_ids=ids, attention_mask=mask, output_hidden_states=True)
            cand_out = candidate_model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True
            )
        ref_states = ref_out.hidden_states
        cand_states = cand_out.hidden_states
        if ref_states is None or cand_states is None:
            raise ValueError("models did not return hidden states")
        ref = ref_states[layer].float()
        cand = cand_states[layer].float()
        diffs_sq += float(((cand - ref) ** 2).sum())
        ref_sq += float((ref**2).sum())
    if ref_sq == 0.0:
        raise ValueError("reference residual stream is identically zero")
    return (diffs_sq / ref_sq) ** 0.5


def _skip_hook(_module: nn.Module, args: tuple[Any, ...], output: Any) -> Any:
    """Forward hook that replaces a decoder layer's output with its input."""
    if isinstance(output, tuple):
        return (args[0], *tuple(output[1:]))
    return args[0]


def _decoder_layers(model: Any) -> Sequence[Any]:
    """Return the decoder layer modules in order.

    Args:
        model: Causal LM.

    Returns:
        The ``model.model.layers`` sequence.
    """
    layers: Sequence[Any] = model.model.layers
    return layers


def run_structured_axis(
    config: Any,
    model: Any,
    tokenizer: Any,
    *,
    device: str,
    time_budget_s: float,
) -> AxisMap:
    """Measure removable structured units (axis A).

    Args:
        config: Run configuration.
        model: Small cartography model, moved to ``device``.
        tokenizer: HuggingFace tokenizer.
        device: Torch device string.
        time_budget_s: Wall-clock budget for the axis.

    Returns:
        The recorded :class:`AxisMap`.
    """
    ids, mask = probe_slice(config, tokenizer)
    budget = config.cartography.structured.kl_budget
    max_removals = int(config.cartography.structured.max_removals)
    layers = _decoder_layers(model)
    n_layers = len(layers)
    reference = copy.deepcopy(model)
    steps: list[Step] = []
    notes: list[str] = []
    started = time.perf_counter()
    stopped_on_budget = False
    last_candidate: Any | None = None

    # Block kind: greedy per-step removal, scored by KL against the pristine model.
    if "block" in config.cartography.structured.kinds:
        removed: list[int] = []
        handles: list[Any] = []
        limit = min(max_removals, max(n_layers - 1, 0))
        while len(removed) < limit:
            if time.perf_counter() - started > time_budget_s:
                stopped_on_budget = True
                break
            best: tuple[float, int] | None = None
            for index in range(n_layers):
                if index in removed:
                    continue
                handle = layers[index].register_forward_hook(_skip_hook)
                kl = mean_kl(
                    reference,
                    model,
                    ids,
                    mask,
                    batch_size=config.cartography.kl_probe_batch_size,
                    device=device,
                )
                handle.remove()
                if best is None or kl < best[0]:
                    best = (kl, index)
            if best is None or best[0] > budget:
                best_kl = None if best is None else round(best[0], 4)
                notes.append(
                    f"block: stopped after {len(removed)} removals; "
                    f"best remaining KL={best_kl} > budget {budget}"
                )
                break
            kl, index = best
            handles.append(layers[index].register_forward_hook(_skip_hook))
            removed.append(index)
            steps.append(
                Step(
                    kind="block",
                    units=(f"layer{index}",),
                    removed=len(removed),
                    kl=kl,
                    extra={},
                )
            )

    # Fine-grained kinds: ranked once, then cumulative checkpoints.
    if "head" in config.cartography.structured.kinds or (
        "ffn_dim" in config.cartography.structured.kinds
    ):
        head_scores = _head_scores(model)
        if "head" in config.cartography.structured.kinds:
            head_steps, head_candidate = _ranked_curve(
                config,
                model=model,
                reference=reference,
                tokenizer=tokenizer,
                device=device,
                kind="head",
                total=len(head_scores),
                order=head_scores,
                notes=notes,
            )
            steps += head_steps
            last_candidate = head_candidate or last_candidate
        if "ffn_dim" in config.cartography.structured.kinds:
            ffn_scores = _ffn_scores(config, model, tokenizer, ids, mask, device=device)
            ffn_steps, ffn_candidate = _ranked_curve(
                config,
                model=model,
                reference=reference,
                tokenizer=tokenizer,
                device=device,
                kind="ffn_dim",
                total=len(ffn_scores),
                order=ffn_scores,
                notes=notes,
            )
            steps += ffn_steps
            last_candidate = ffn_candidate or last_candidate

    cross = None
    if steps:
        final = model if last_candidate is None else last_candidate
        cross = _last_cross_check(config, reference, final, ids, mask, device=device)
    notes.append(
        "axis A is cumulative: the head and ffn_dim checkpoints are measured on top of the "
        "block removals already applied, and the E_x cross-check reflects the final state"
    )
    notes.append(
        "ranking rule (reproducible from this config): head = sum of |o_proj output rows| per "
        "head; ffn_dim = Wanda score sum_i |down_proj[i, j]| * ||X[:, j]|| / sqrt(N); both "
        "ascending, so the least important unit is removed first; the record keeps a 32-name "
        "preview per checkpoint"
    )
    return AxisMap(
        axis="structured",
        proxy="kl_nats",
        budget=budget,
        steps=tuple(steps),
        stopped_on_budget=stopped_on_budget,
        notes=tuple(notes),
        e_x_cross_check=cross,
    )


def _head_scores(model: Any) -> list[tuple[int, int, int, float]]:
    """Rank attention heads by output-row magnitude (least important first).

    Args:
        model: Causal LM.

    Returns:
        Tuples ``(layer, head, head_dim, score)`` sorted ascending by score.
    """
    config = model.config
    n_heads = int(config.num_attention_heads)
    hidden = int(config.hidden_size)
    head_dim = hidden // n_heads
    scored: list[tuple[int, int, int, float]] = []
    for layer_index, layer in enumerate(_decoder_layers(model)):
        weight = layer.self_attn.o_proj.weight.detach().float()
        for head in range(n_heads):
            rows = weight[head * head_dim : (head + 1) * head_dim]
            scored.append((layer_index, head, head_dim, float(rows.abs().sum())))
    scored.sort(key=lambda item: item[3])
    return scored


def _ffn_scores(
    config: Any,
    model: Any,
    tokenizer: Any,
    ids: Tensor,
    mask: Tensor,
    *,
    device: str,
) -> list[tuple[int, int, float]]:
    """Rank FFN intermediate dimensions by an activation-weighted magnitude.

    The score is the Wanda-style ``sum_i |W[i, j]| * ||X[:, j]|| / sqrt(N)`` on the
    ``down_proj`` weight, which is exactly the neuron's contribution magnitude.

    Args:
        config: Run configuration (batch size, probe size).
        model: Causal LM.
        tokenizer: Unused; kept for symmetry with the other axis helpers.
        ids: Probe token ids.
        mask: Probe attention mask.
        device: Torch device string.

    Returns:
        Tuples ``(layer, unit, score)`` sorted ascending by score.
    """
    del tokenizer
    scored: list[tuple[int, int, float]] = []
    for layer_index, _layer in enumerate(_decoder_layers(model)):
        path = f"model.layers.{layer_index}.mlp.down_proj"
        activations = capture_gate_inputs(
            model,
            ids.to(device),
            mask.to(device),
            layer_path=path,
            batch_size=config.cartography.kl_probe_batch_size,
        ).float()
        col_norm = activations.pow(2).mean(dim=0).clamp_min(0.0).sqrt()
        weight = model.get_submodule(path).weight.detach().float()
        score = (weight.abs() * col_norm.unsqueeze(0)).sum(dim=0)
        for unit in range(score.numel()):
            scored.append((layer_index, unit, float(score[unit])))
    scored.sort(key=lambda item: item[2])
    return scored


def _ranked_curve(
    config: Any,
    *,
    model: Any,
    reference: Any,
    tokenizer: Any,
    device: str,
    kind: str,
    total: int,
    order: Sequence[Any],
    notes: list[str],
) -> tuple[list[Step], Any | None]:
    """Measure the cumulative removal curve of a ranked fine-grained kind.

    Args:
        config: Run configuration.
        model: Pristine small model.
        reference: Pristine copy used as the KL reference.
        tokenizer: HuggingFace tokenizer.
        device: Torch device string.
        kind: ``head`` or ``ffn_dim``.
        total: Total number of units of this kind.
        order: Unit ranking, least important first.
        notes: Note accumulator.

    Returns:
        ``(steps, last_candidate)`` where ``last_candidate`` is the last evaluated
        model, kept for the residual E_x cross-check.
    """
    ids, mask = probe_slice(config, tokenizer)
    budget = config.cartography.structured.kl_budget
    steps: list[Step] = []
    last_candidate: Any | None = None
    planned = sorted({max(1, round(fraction * total)) for fraction in _RANKED_FRACTIONS})
    notes.append(
        f"{kind}: ranked by importance; checkpoints at {planned} removals of {total} units "
        "(cumulative, not per-step: a per-step curve is unaffordable at this unit count)"
    )
    for count in planned:
        candidate = copy.deepcopy(model)
        selected = list(order[:count])
        if kind == "head":
            _apply_head_removal(candidate, selected)
        else:
            _apply_ffn_removal(candidate, selected)
        kl = mean_kl(
            reference,
            candidate,
            ids,
            mask,
            batch_size=config.cartography.kl_probe_batch_size,
            device=device,
        )
        steps.append(
            Step(
                kind=kind,
                units=_preview(kind, selected),
                removed=count,
                kl=kl,
                extra={"n_ranked": float(count), "n_total": float(total)},
            )
        )
        last_candidate = candidate
        if kl > budget:
            notes.append(
                f"{kind}: budget {budget} exceeded at {count}/{total} removals (KL={kl:.4f})"
            )
            break
    return steps, last_candidate


def _preview(kind: str, selected: Sequence[Any], limit: int = 32) -> tuple[str, ...]:
    """Return a bounded removal-order preview for the record.

    Args:
        kind: Unit kind.
        selected: Removed units in removal order.
        limit: Maximum names kept.

    Returns:
        The first ``limit`` unit names, plus a count marker when truncated.
    """
    names = tuple(_unit_name(kind, item) for item in selected[:limit])
    remaining = len(selected) - len(names)
    return (*names, f"...+{remaining} more") if remaining > 0 else names


def _unit_name(kind: str, item: Any) -> str:
    """Render a unit identifier for the record.

    Args:
        kind: Unit kind.
        item: Ranking entry.

    Returns:
        A short identifier.
    """
    if kind == "head":
        layer, head, _head_dim, _score = item
        return f"l{layer}h{head}"
    layer, unit, _score = item
    return f"l{layer}n{unit}"


def _apply_head_removal(model: Any, selected: Sequence[Any]) -> None:
    """Zero the output rows of the given attention heads in place.

    Args:
        model: Model mutated in place.
        selected: Entries ``(layer, head, head_dim, score)``.
    """
    layers = _decoder_layers(model)
    with torch.no_grad():
        for layer_index, head, head_dim, _score in selected:
            weight = layers[layer_index].self_attn.o_proj.weight
            weight[head * head_dim : (head + 1) * head_dim] = 0


def _apply_ffn_removal(model: Any, selected: Sequence[Any]) -> None:
    """Zero the ``down_proj`` input columns of the given FFN dimensions in place.

    Args:
        model: Model mutated in place.
        selected: Entries ``(layer, unit, score)``.
    """
    layers = _decoder_layers(model)
    with torch.no_grad():
        for layer_index, unit, _score in selected:
            layers[layer_index].mlp.down_proj.weight[:, unit] = 0


def _last_cross_check(
    config: Any,
    reference: Any,
    candidate: Any,
    ids: Tensor,
    mask: Tensor,
    *,
    device: str,
) -> float:
    """Return the residual-stream E_x cross-check on the final hidden state.

    The final residual stream is used rather than the configured ternary layer
    because removals in any band must be visible to the cross-check; a mid-model
    layer is blind to edits in deeper bands.

    Args:
        config: Run configuration.
        reference: Pristine model.
        candidate: Model with removals applied.
        ids: Probe ids.
        mask: Probe mask.
        device: Torch device string.

    Returns:
        The relative error of the final residual stream.
    """
    return residual_e_x(
        reference,
        candidate,
        ids,
        mask,
        layer=-1,
        batch_size=config.cartography.kl_probe_batch_size,
        device=device,
    )


def run_directions_axis(
    config: Any,
    model: Any,
    tokenizer: Any,
    *,
    device: str,
    time_budget_s: float,
) -> AxisMap:
    """Measure removable activation directions (axis B).

    Args:
        config: Run configuration.
        model: Small cartography model.
        tokenizer: HuggingFace tokenizer.
        device: Torch device string.
        time_budget_s: Wall-clock budget for the axis.

    Returns:
        The recorded :class:`AxisMap`.
    """
    del time_budget_s
    ids, mask = probe_slice(config, tokenizer)
    reference = copy.deepcopy(model)
    budget = config.cartography.directions.kl_budget
    steps: list[Step] = []
    notes: list[str] = [
        "directions: least-significant residual-stream principal directions, removed by "
        "weight orthogonalization of o_proj and down_proj inside the band (weight edit, no hook)"
    ]
    started = time.perf_counter()
    last_candidate: Any | None = None
    for band_start, band_end in config.cartography.directions.layer_bands:
        path = f"model.layers.{band_start}"
        activations = capture_gate_inputs(
            model,
            ids.to(device),
            mask.to(device),
            layer_path=path,
            batch_size=config.cartography.kl_probe_batch_size,
        ).float()
        covariance = activations.t() @ activations / activations.shape[0]
        _values, vectors = torch.linalg.eigh(covariance)
        for n_directions in config.cartography.directions.n_directions:
            candidate = copy.deepcopy(model)
            basis = vectors[:, : int(n_directions)]
            _orthogonalize_band(candidate, band=range(band_start, band_end + 1), basis=basis)
            kl = mean_kl(
                reference,
                candidate,
                ids,
                mask,
                batch_size=config.cartography.kl_probe_batch_size,
                device=device,
            )
            steps.append(
                Step(
                    kind="direction",
                    units=(f"band{band_start}-{band_end}_n{n_directions}",),
                    removed=int(n_directions),
                    kl=kl,
                    extra={},
                )
            )
            last_candidate = candidate
            if kl > budget:
                notes.append(
                    f"band {band_start}-{band_end}: budget {budget} exceeded at "
                    f"{n_directions} directions (KL={kl:.4f})"
                )
    stopped = time.perf_counter() - started > config.cartography.budget_gpu_hours_per_axis * 3600.0
    final = model if last_candidate is None else last_candidate
    cross = _last_cross_check(config, reference, final, ids, mask, device=device)
    return AxisMap(
        axis="directions",
        proxy="kl_nats",
        budget=budget,
        steps=tuple(steps),
        stopped_on_budget=stopped,
        notes=tuple(notes),
        e_x_cross_check=cross,
    )


def _orthogonalize_band(model: Any, *, band: range, basis: Tensor) -> None:
    """Remove the subspace spanned by ``basis`` from the residual writers in a band.

    Args:
        model: Model mutated in place.
        band: Decoder layer indices to edit.
        basis: Orthonormal directions ``[hidden, n]``.
    """
    layers = _decoder_layers(model)
    with torch.no_grad():
        for layer_index in band:
            for module in (layers[layer_index].self_attn.o_proj, layers[layer_index].mlp.down_proj):
                weight = module.weight
                projection = basis.t().to(weight.dtype) @ weight
                module.weight.copy_(weight - basis.to(weight.dtype) @ projection)


def run_ternary_axis(
    config: Any,
    model: Any,
    tokenizer: Any,
    *,
    device: str,
) -> AxisMap:
    """Probe the TCD ternary representation itself (axis C).

    Args:
        config: Run configuration.
        model: Small cartography model.
        tokenizer: HuggingFace tokenizer.
        device: Torch device string.

    Returns:
        The recorded :class:`AxisMap`.
    """
    settings = config.cartography.ternary
    ids, mask = probe_slice(config, tokenizer)
    path = f"model.layers.{settings.layer}.mlp.gate_proj"
    activations = capture_gate_inputs(
        model,
        ids.to(device),
        mask.to(device),
        layer_path=path,
        batch_size=config.cartography.kl_probe_batch_size,
    ).float()
    weight = model.get_submodule(path).weight.detach().float()
    steps: list[Step] = []
    notes: list[str] = [
        "ternary: E_x is the layer output error of the TCD reconstruction "
        "(spec phase1/compression-gate R1 definition), budget "
        f"{settings.e_x_budget}"
    ]
    best: tuple[int, TernaryFactors, float] | None = None
    for mu in settings.mu:
        factors = decompose_ternary(
            weight,
            mu=float(mu),
            tau=settings.tau,
            batched=True,
            b=None,
            importance=None,
            n_inner=15,
        )
        e_x = layer_output_error(activations, weight, factors.reconstruct())
        zero_fraction = _factor_zero_fraction(factors)
        steps.append(
            Step(
                kind="tcd_baseline",
                units=(f"mu{mu}",),
                removed=0,
                kl=0.0,
                extra={"e_x": e_x, "zero_fraction": zero_fraction, "k": float(factors.k)},
            )
        )
        if best is None or e_x < best[2]:
            best = (mu, factors, e_x)
    if best is None:
        raise ValueError("ternary axis requires at least one mu value")
    _mu, factors, base_e_x = best
    if base_e_x > settings.e_x_budget:
        notes.append(
            f"ternary: the best TCD baseline already exceeds the E_x budget "
            f"({base_e_x:.4f} > {settings.e_x_budget})"
        )
    steps += _plane_drop_steps(activations, weight, factors, settings.rank_fractions)
    steps += _nm_steps(activations, weight, factors, settings.nm_patterns)
    steps += _core_rank_steps(activations, weight, factors, settings.rank_fractions)
    return AxisMap(
        axis="ternary",
        proxy="e_x_relative",
        budget=settings.e_x_budget,
        steps=tuple(steps),
        stopped_on_budget=False,
        notes=tuple(notes),
        e_x_cross_check=base_e_x,
    )


def _factor_zero_fraction(factors: TernaryFactors) -> float:
    """Return the zero fraction of the stored ternary factors.

    Args:
        factors: TCD factors.

    Returns:
        Fraction of zeros over ``B`` and ``C`` combined.
    """
    zeros = int((factors.B == 0).sum() + (factors.C == 0).sum())
    total = int(factors.B.numel() + factors.C.numel())
    return zeros / total


def _plane_drop_steps(
    activations: Tensor,
    weight: Tensor,
    factors: TernaryFactors,
    fractions: Sequence[float],
) -> list[Step]:
    """Measure E_x after keeping only the largest planes by ``|D|``.

    Args:
        activations: Calibration activations ``[tokens, in]``.
        weight: Original weight ``[out, in]``.
        factors: TCD factors.
        fractions: Fractions of planes to keep.

    Returns:
        One step per fraction.
    """
    order = torch.argsort(factors.D.abs(), descending=True)
    steps: list[Step] = []
    for fraction in fractions:
        keep = max(1, round(fraction * factors.k))
        index = order[:keep]
        reduced = TernaryFactors(
            B=factors.B[:, index],
            D=factors.D[index],
            C=factors.C[index, :],
            k=keep,
        )
        e_x = layer_output_error(activations, weight, reduced.reconstruct())
        steps.append(
            Step(
                kind="plane_drop",
                units=(f"keep{keep}of{factors.k}",),
                removed=factors.k - keep,
                kl=0.0,
                extra={"e_x": e_x, "k": float(keep)},
            )
        )
    return steps


def _nm_steps(
    activations: Tensor,
    weight: Tensor,
    factors: TernaryFactors,
    patterns: Sequence[tuple[int, int]],
) -> list[Step]:
    """Measure E_x under entry-level N:M masking of the ternary factors.

    Args:
        activations: Calibration activations.
        weight: Original weight.
        factors: TCD factors.
        patterns: ``(n_keep, m_group)`` patterns.

    Returns:
        One step per pattern.
    """
    steps: list[Step] = []
    for n_keep, m_group in patterns:
        masked_b = _nm_mask(factors.B, factors.D, n_keep, m_group)
        masked_c = _nm_mask(factors.C.t().contiguous(), factors.D, n_keep, m_group).t()
        reconstructed = (
            (factors.B * masked_b).to(factors.D.dtype)
            @ torch.diag(factors.D)
            @ (factors.C * masked_c).to(factors.D.dtype)
        )
        e_x = layer_output_error(activations, weight, reconstructed)
        kept = float(masked_b.float().mean() * masked_c.float().mean())
        steps.append(
            Step(
                kind="nm_mask",
                units=(f"{n_keep}:{m_group}",),
                removed=0,
                kl=0.0,
                extra={"e_x": e_x, "mask_keep_fraction": kept},
            )
        )
    return steps


def _nm_mask(factor: Tensor, scales: Tensor, n_keep: int, m_group: int) -> Tensor:
    """Build an entry-level N:M keep-mask ranked by ``|factor| * |scale|``.

    Args:
        factor: Ternary factor ``[rows, k]`` int8.
        scales: Plane scales ``[k]``.
        n_keep: Entries kept per group.
        m_group: Group size along the ``k`` axis.

    Returns:
        A 0/1 mask with the same shape as ``factor``.

    Raises:
        ValueError: If ``n_keep`` exceeds ``m_group`` or ``m_group`` is not positive.
    """
    if not 0 < n_keep <= m_group:
        raise ValueError(f"invalid N:M pattern {n_keep}:{m_group}")
    rows, k = factor.shape
    pad = (-k) % m_group
    if pad:
        factor = torch.nn.functional.pad(factor, (0, pad))
        scales = torch.nn.functional.pad(scales, (0, pad))
    groups = factor.reshape(rows, -1, m_group).abs().to(torch.float32)
    score = groups * scales.abs().reshape(1, -1, m_group).to(torch.float32)
    index = score.topk(n_keep, dim=-1).indices
    mask = torch.zeros_like(groups)
    mask.scatter_(-1, index, 1.0)
    mask = mask.reshape(rows, -1)
    if pad:
        mask = mask[:, :k]
    return mask.to(factor.dtype)


def _core_rank_steps(
    activations: Tensor,
    weight: Tensor,
    factors: TernaryFactors,
    fractions: Sequence[float],
) -> list[Step]:
    """Measure E_x after truncating the SVD rank of the ternary reconstruction.

    Args:
        activations: Calibration activations.
        weight: Original weight.
        factors: TCD factors.
        fractions: Fractions of ``min(out, in)`` ranks to keep.

    Returns:
        One step per fraction.
    """
    reconstruction = factors.reconstruct()
    u, s, vh = torch.linalg.svd(reconstruction, full_matrices=False)
    steps: list[Step] = []
    for fraction in fractions:
        rank = max(1, round(fraction * min(reconstruction.shape)))
        truncated = (u[:, :rank] * s[:rank]) @ vh[:rank, :]
        e_x = layer_output_error(activations, weight, truncated)
        steps.append(
            Step(
                kind="core_rank",
                units=(f"rank{rank}",),
                removed=0,
                kl=0.0,
                extra={"e_x": e_x, "rank": float(rank)},
            )
        )
    return steps
