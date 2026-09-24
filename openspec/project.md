# Project context

Hand-maintained context used by the OpenSpec propose and apply phases. Keep it current: it is the first thing a fresh agent reads about the project.

## Goal

Convert existing FP32/BF16 open-weight LLMs into **pure ternary networks** (weights and activations in {-1, 0, 1}) **without retraining**, by algorithmically reconstructing the original activation pathways — the TCD (Topological Mapping and Ternary Decomposition) method.

The Phase 0 algorithm is **ExTernD-Lite**: a factorized decomposition `W ≈ B diag(D) C` with `B, C` ternary and `D` real, fitted per block of columns against real calibration activations, with asymmetric per-row ternary quantization.

## Status

Documentation and plan of record only; no code yet. Sources of truth:

- `PROJECT_PLAN.md` — phases 0–5, the Phase 0 algorithm, the GREEN/YELLOW/RED decision framework.
- `PHASE0_PREREQUISITES.md` — prerequisites A–F, the critical path, correctness tests F1–F4.
- `PHASE0_ACQUISITION_PLAN.md` — how each prerequisite is obtained, in critical-path order.
- `PHASE0_REVIEW.md` — gap analysis against 2024–2026 SOTA (ExTernD, PTQTP, PT2-LLM, TWLA, CAT-Q).
- `idea source.txt` — the original idea and the reasoning it came from.

## Stack

Python 3.10+; PyTorch (CPU or CUDA); `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `numpy`. Phase 4+ adds a C++/AVX-512 kernel with Python bindings.

Toolchain for gates: `ruff check`, `ruff format --check`, `mypy`, `pytest`, GitHub Actions CI (configs land with the first code change).

Hardware: CPU with 16 GB+ RAM minimum; GPU with 8 GB+ VRAM recommended for the sweep and imatrix work.

## Phase 0 gate (fixed before the run)

| Metric | GREEN threshold | SOTA reference |
|---|---|---|
| Perplexity delta vs FP16 (WikiText-2) | < 5 % | ExTernD μ=3: +3.2 % |
| Effective BPW (Shannon entropy) | < 6.0 | PTQTP K=4: ~6.8 |
| Sparsity (zeros in B, C) | > 40 % | ExTernD τ=1.0: 57 % |
| Max layer output error `‖XW − XŴ‖_F / ‖XW‖_F` | < 5 % | PT2-LLM AGA |
| Runtime per layer | < 10 min | — |

YELLOW: perplexity delta < 10 % and BPW < 8. RED otherwise — the project stops. The symmetric sign-threshold baseline is expected to produce RED and exists to validate the test itself (F4).

## Critical path

`C2` (`asymmetric_ternary_fit`) → `C6` (fix `k = μ · min(m, G)`, not `min(m, n)`) → `F2` (monotone residual decrease) → Phase 0 can start. Then `C1` (SSR column reorder), `C3` (Hessian diagonal / imatrix, 128 samples), `C4` (full-block hooks), `C5` (softened ternarization), then F1–F4, the model and the calibration set.

## Conventions

- Reduced sweep first: μ ∈ {2, 3}, τ ∈ {0.7, 1.0}, G ∈ {64, 128}; expand only if the result is ambiguous.
- Seeds fixed and taken from the config (`torch.manual_seed(42)`, `numpy.random.seed(42)`, `torch.use_deterministic_algorithms(True)` where feasible).
- Every recorded experiment lives in `results/<experiment-id>/` with `config.yaml`, `metrics.json`, `report.md` (`.kilo/rules/research-discipline.md`).
- No custom kernels, no packaging work before the Phase 0 gate passes.
