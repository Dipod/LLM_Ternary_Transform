# Proposal

## Why

Phase 0 ended in a **STOP** for its premise: at matched effective bits, ordinary group-128 integer quantization dominates the factorized ternary frontier on the layer-local metric (`results/decision_report.md`; the same law is published — AQLM `arXiv:2401.06118` and ParetoQ `arXiv:2502.02631` both find 2-bit at matched bytes worse than a smaller model at 3–4 bits). **Bit-width is an exhausted axis, and the project has no active direction.**

The unexploited axis is **structure**. Removing whole units — layers, attention heads, FFN intermediate dimensions, experts, rank directions, or whole factorization planes — lowers stored bytes *and* compute, which no bit-width change can do. The literature says the redundancy is real but task-dependent (ShortGPT, `arXiv:2403.03853`; the ICLR 2025 deeper-layer result) and that single-technique methods plateau (Bonsai `arXiv:2402.05406`; GISP, ACL 2026; REAP, ICLR 2026), while *combinations* help (SPQ `arXiv:2602.18420`; "Compression Trinity" `arXiv:2608.24070`). What is missing is a cheap, pre-registered **map of where redundancy actually lives** in our target model, measured under a quality bar strict enough to mean something.

Why now: the Phase 0 harness, calibration slice builder and ROCm runtime already exist, so the cartography is cheap; and our own TCD output already sits on the ~42 % zero "quantization valley" that Sparse-BitNet (`arXiv:2603.05168`) identifies as naturally sparsity-friendly — an asset nobody has tested in a post-hoc pipeline.

The eventual goal that motivates all of this — a model that normally needs ~2× the VRAM running in 24 GB at ≥ 20 tok/s with a wide context — is **not** gated by this change. Stage A gates *quality at reduced stored bytes*; speed, VRAM and context length are acceptance criteria of later stages.

## What Changes

- **New Phase 1, Stage A** with two deliverables in one change:
  1. **Redundancy cartography** on the cached small model (`Qwen/Qwen2.5-0.5B`): measure, along three independent axes, how much each unit can be removed before a pre-registered cheap proxy degrades — (a) structured components (transformer blocks, attention heads, FFN intermediate dimensions), (b) directions/subspaces (KL-guided importance and abliteration-style orthogonalization), (c) the model's own ternary representation (factor planes, N:M patterns over its natural zeros, low-rank structure of the ternary core).
  2. **One combined-transformation probe** on a mid model (`Qwen/Qwen3-8B`, BF16) that chains the three axes in the order the cartography fixed, scored end-to-end against the strict gate.
- **Strict quality gate (+0.5 / −0.5)**: relative perplexity increase ≤ 0.5 % and every benchmark delta ≥ −0.5 points versus the same model at BF16, at equal-or-lower stored bytes. Fixed before the run.
- **Matched-quality comparison against the strongest baseline**: a compression claim is compared against integer group-128 4/5/6-bit quantization on the same model/slice at *equal quality* (the inverse of the Phase 0 matched-bits rule), not against FP16.
- **An 8-bit sanity baseline** that must show near-zero degradation, otherwise the evaluation harness is declared broken and the Stage A verdict is void.
- **Fixed, budgeted healing**: the removal decision is training-free; after the units are chosen, exactly one pre-registered mini-LoRA recovery step (≤ 1 GPU-hour, rank ≤ 8, seeds from config) is allowed, and the verdict is reported both before and after healing.
- **Roadmap, not implementation**: Stage B tracks (learned/adaptive-skip architectures, MoE expert removal, direction-subspace storage format, tiny-pretrain architecture experiments) are described in `design.md` and become separate changes if this stage returns GO.
- No existing behaviour is removed or changed (**no breaking changes**).

## Capabilities

### New Capabilities
- `phase1/compression-gate`: the strict ±0.5 gate and its metric definitions (relative perplexity delta, benchmark deltas, stored-byte accounting), the matched-quality integer baseline rule, the 8-bit harness sanity check, the fixed mini-LoRA healing budget, and the single GO / YELLOW / RED decision rule.
- `phase1/redundancy-cartography`: the protocol for measuring removable structure on the three axes, the importance/search procedures, the rule that the combination order is frozen from the map before the probe runs, and the requirement that the map is recorded on disk before the probe starts.

### Modified Capabilities
- None. The `phase0/*` contracts describe Phase 0 behaviour and are not reinterpreted; the Phase 0 results remain valid for their own slice and metric.

## Impact

- New package `phase1_compression/` (mirroring `phase0_feasibility/`) with `cartography/`, `transform/`, `evaluation/`, `main.py` and `config.yaml`; tests under `tests/phase1_compression/`.
- Reuses the Phase 0 calibration slice definition and the recorded `E_x` proxy as a cheap cartography metric; the recorded Phase 0 and STOP artifacts are not reinterpreted.
- New model acquisition: `Qwen/Qwen3-8B` BF16 (~16 GB); cartography uses the already cached `Qwen/Qwen2.5-0.5B`. `C:` has ~37 GB free, so exactly one mid model is downloaded in this change.
- New evaluation dependency: `lm-evaluation-harness` for the benchmark suite, plus the project's own perplexity script; the exact version is pinned in `design.md` (D4) and verified by a spike task before it is used for a gate.
- Updates `openspec/project.md` (Phase 0 closed, Phase 1 Stage A open) and `memory.md` with the Stage A verdict.
- Runtime: ROCm-on-WSL on the RX 7900 XTX; cartography is forward-only so 8B BF16 fits in 24 GB.

## Context sources

Checked before writing:

- Repository state: `openspec list --specs` (three `phase0/*` capabilities), `openspec/project.md` (STOP verdict, matched-bits convention), `results/decision_report.md`, `memory.md`, `AGENTS.md`, `.kilo/rules/{sdd-integrations,research-discipline,verification-gates}.md`.
- Ours: Phase 0/STOP records in `results/` (matched-bits comparison, µ and τ probes, sparsity 0.425 at µ=3) — cited as the basis for "bit-width is exhausted".
- Literature (web, 2026-09-24): `arXiv:2401.06118` (AQLM), `arXiv:2502.02631` (ParetoQ), `arXiv:2402.05406` (Bonsai), ACL 2026 GISP, `arXiv:2403.03853` (ShortGPT), ICLR 2025 deeper-layer pruning, `arXiv:2602.01997` (layer-pruning limits on generative tasks), `arXiv:2510.13999` (REAP), `arXiv:2606.05538` (Fisher-MoE), `arXiv:2603.05168` (Sparse-BitNet), `arXiv:2406.11717` and `arXiv:2602.02132` (directional ablation), `arXiv:2602.18420` (SPQ), `arXiv:2608.24070` (Compression Trinity).
- Deliberately not checked: exact benchmark numbers of the cited papers (they are context, not targets); whether `lm-evaluation-harness` currently installs against our pins (closed by task T2, before any gate run).
