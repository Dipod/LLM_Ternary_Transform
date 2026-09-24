# Project context

Hand-maintained context used by the OpenSpec propose and apply phases. Keep it current: it is the first thing a fresh agent reads about the project.

## Goal

Convert existing FP32/BF16 open-weight LLMs into **pure ternary networks** (weights and activations in {-1, 0, 1}) **without retraining**, by algorithmically reconstructing the original activation pathways — the TCD (Topological Mapping and Ternary Decomposition) method.

The Phase 0 algorithm is **ExTernD-Lite**: a factorized decomposition `W ≈ B diag(D) C` with `B, C ∈ {-1, 0, 1}` and `D` real, fitted against real calibration activations with a symmetric adaptive threshold `T_τ` and no gradients.

## Status

Phase 0 is implemented and its change is **archived** (`openspec/changes/archive/2026-09-24-phase0-ternary-feasibility/`); the behavioural contracts now live in `openspec/specs/phase0/ternary-decomposition/spec.md` and `openspec/specs/phase0/feasibility-gate/spec.md`. Code is in `phase0_feasibility/`, tests under `tests/`.

**Gate outcome (2026-09-24, AMD RX 7900 XTX / ROCm): YELLOW, not GREEN.** Best configuration μ=3, τ=0.7, importance off gives `E_x = 0.0716` (GREEN needs ≤ 0.05), sparsity 0.425, effective BPW 5.595, runtime 1.4 s; the μ=1 baseline is RED (`E_x = 0.303`). Records are under `results/<experiment-id>/` (git-ignored).

**Premise verdict (2026-09-24): STOP — the premise is not confirmed.** The gate compares against FP16, not against the real competitor. A pre-registered comparison at **matched bits** on the same layer, model, slice and activations shows ordinary group-wise integer quantization dominating the factorized ternary frontier: at `bpw ≈ 6.13` an integer 6-bit group-128 quantizer gives `E_x` 0.018–0.019 vs ternary 0.046–0.048 (~2.5× better), and integer 5-bit (bpw 5.125, 17% fewer bits) still gives 0.035–0.039; ternary at bpw 4.4–4.6 (0.088–0.115) also loses to integer 4-bit at bpw 4.125 (0.070–0.085). Three independent literature surveys agree: post-training ternary loses to 4-bit on both accuracy and bits per quality, has no native tensor-core datapath on the target GPUs, and its zeros are not skipped by production kernels. Ternary keeps a niche on multiplier-free hardware (CPU/edge batch-1, FPGA/ASIC), extreme-memory MoE, and **natively trained** ternary — none of which is the current stand. Evidence: `results/bit_frontier/`, `results/decision_report.md`, `memory.md`.

`inputs/PROJECT_PLAN.md`, `inputs/PHASE0_PREREQUISITES.md`, `inputs/PHASE0_ACQUISITION_PLAN.md`, `inputs/PHASE0_REVIEW.md` and `inputs/idea source.txt` remain on disk as historical AI-generated inputs. They are **not** the plan of record: their algorithmic and gate claims were corrected in the archived change and are superseded by `openspec/specs/` and the change's `design.md`. Do not derive thresholds, ranks or shapes from them.

## Stack

Python 3.12+ (raised by the pinned `numpy 2.5.3` `requires-python`); PyTorch for Phase 0 (`transformers`, `datasets`, `scikit-learn`, `matplotlib`, `numpy`). Phase 4+ adds a C++/AVX-512 kernel with Python bindings.

Toolchain for gates: `ruff check`, `ruff format --check`, `mypy`, `pytest` (network tests excluded by default via the `network` marker), GitHub Actions CI.

Hardware: CPU with 16 GB+ RAM; an AMD Radeon RX 7900 XTX (24 GB) via ROCm-on-WSL is the compute device for Phase 0 (`design.md` D9, updated 2026-09-24). The CPU path remains valid (`device: cpu`); GPU with 8 GB+ VRAM recommended for later sweeps.

## Phase 0 gate (fixed before the run)

The unit is one Linear layer: `gate_proj` of layer 12 of `Qwen/Qwen2.5-0.5B`, weight shape `[out, in] = [4864, 896]`.

| Metric | Definition | GREEN |
|---|---|---|
| Output error `E_x` | `‖X Wᵀ − X Ŵᵀ‖_F / ‖X Wᵀ‖_F` on calibration activations | ≤ 0.05 |
| Sparsity | fraction of zeros in `B` and `C` | ≥ 0.40 |
| Effective BPW | `μ·(out+in)/max(out,in)·(2−sparsity)` (mask + sign per stored trit) | < 6.0 |
| Runtime per layer | wall-clock seconds | < 600 s |

YELLOW: `E_x ≤ 0.10` and effective BPW < 8.0. RED otherwise. End-to-end perplexity is **not** part of the Phase 0 gate because a single converted projection is not a sound end-to-end measurement; it is deferred to a later phase and its absence is stated in every report. The `μ=1` baseline (same procedure, no importance weighting) is the contrast expected to fail.

`E_x`, sparsity and effective BPW are defined in `openspec/changes/phase0-ternary-feasibility/specs/phase0/feasibility-gate/spec.md`.

## Critical path

Naive per-block fitting and the per-block rank formula from the historical docs were dropped: they are shape-invalid and yield no compression. Phase 0 fits a **global** `B diag(D) C` with `k = round(μ·min(out, in))`; column blocks are a compute strategy only (`design.md` D4). Escalation (larger model, more layers, AGA / activation-space fitting, kernels) does **not** start: the premise failed the matched-bit comparison against integer quantization. Any future work restarts from the niche framing in `results/decision_report.md`, not from "beat 4-bit PTQ".

## Conventions

- Reduced sweep: μ ∈ {2, 3} × τ ∈ {0.7, 1.0} × importance ∈ {off, on}, reorder off, plus the `μ=1` baseline; reorder-on variants only if the result is ambiguous.
- Seeds fixed and taken from `phase0_feasibility/config.yaml` (`torch`, `numpy`, `random`, `torch.use_deterministic_algorithms(True)` where feasible).
- Calibration slice: WikiText-2 (`wikitext-2-raw-v1`, train), first 128 non-empty documents, `seq_len=512` (up to 65,536 tokens), single pass, pad positions dropped.
- Every recorded experiment lives in `results/<experiment-id>/` with `config.yaml`, `metrics.json`, `report.md` (`.kilo/rules/research-discipline.md`).
- **A compression claim must be compared at matched effective bits against an integer quantizer baseline** (RTN/GPTQ group-128 at 4/5/6-bit), not against FP16; the decision rule is pre-registered before the run.
- No custom kernels, no packaging work before the Phase 0 gate passes.
