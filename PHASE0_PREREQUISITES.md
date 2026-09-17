# Phase 0 Prerequisites — ExTernD-Lite Ternary Decomposition

> Generated from gap analysis of `PHASE0_REVIEW.md` applied to `PROJECT_PLAN.md`
> Date: 2026-09-17

---

## A. Environment & Dependencies

| # | Prerequisite | Specification | Status |
|---|-------------|---------------|--------|
| A1 | Python | 3.10+ | ☐ |
| A2 | PyTorch | Latest stable (CPU or CUDA) | ☐ |
| A3 | transformers | Latest stable (for AutoModel, hooks) | ☐ |
| A4 | datasets | Latest stable (for wikitext-2-raw-v1) | ☐ |
| A5 | scikit-learn | For cosine similarity (SSR column reorder) | ☐ |
| A6 | matplotlib | For analysis plots (Pareto frontiers) | ☐ |
| A7 | numpy | Array operations | ☐ |

**Hardware requirement**: CPU: 16GB+ RAM minimum. GPU (recommended): 8GB+ VRAM for imatrix computation and sweep.

---

## B. Data & Model

| # | Prerequisite | Specification | Status |
|---|-------------|---------------|--------|
| B1 | Model | `Qwen/Qwen2.5-0.5B` or `google/gemma-2-2b` in FP32/BF16 | ☐ |
| B2 | Calibration data | `wikitext-2-raw-v1`, **128-512 samples** (not 50) | ☐ |
| B3 | Synthetic distribution | Heavy-tailed (Student-t df=3 or Laplace — TBD) | ☐ |
| B4 | Target layer | `gate_proj`, layer index 12 (to be confirmed) | ☐ |

---

## C. Algorithm Components (To Implement)

| # | Prerequisite | Specification | Status |
|---|-------------|---------------|--------|
| C1 | `column_reorder_by_cosine(W)` | SSR: cosine similarity between all column pairs, sort/maximize adjacent similarity | ☐ |
| C2 | `asymmetric_ternary_fit(x, tau)` | Core: row-wise {−α+μ, μ, α+μ} ternary with soft threshold. Input: real vector; Output: ternary vector + scale. Based on PT2-LLM ITF + CAT-Q softened ternarization | ☐ |
| C3 | `compute_hessian_diagonal(model, dataset)` | Per-sample gradient through target layer, accumulate diag(∂L/∂W)² for all W. Requires autograd hooks | ☐ |
| C4 | `register_full_block_hooks(model, layer_idx)` | Capture input+output activations of target layer AND adjacent layers simultaneously | ☐ |
| C5 | Softened ternarization | Differentiable transition near τ (temperature-scaled sigmoid ternarization). Referenced as CAT-Q ST but **not yet in code** | ☐ |
| C6 | Fixed bug: k dimension | `k = mu * min(m, G)` NOT `mu * min(m, n)` — C is per-block with G columns, not full n. Original formula gives k=8192 which is infeasible | ☐ |

---

## D. Configuration

| # | Prerequisite | Specification | Status |
|---|-------------|---------------|--------|
| D1 | Hyperparameter sweep space | μ ∈ {1,2,3,4}, τ ∈ {0.5,0.7,1.0,1.5,2.0}, G ∈ {64,128,256}, symmetric/asymmetric, reorder on/off, importance on/off | ☐ |
| D2 | Random seed | Fixed for reproducibility (TBD — suggest 42) | ☐ |
| D3 | Layer index | gate_proj at layer 12 (confirm with model architecture) | ☐ |
| D4 | Baseline comparison | Symmetric sign-threshold baseline (expected RED) — must run alongside ExTernD-Lite | ☐ |

---

## E. Computational Constraints

| # | Prerequisite | Specification | Status |
|---|-------------|---------------|--------|
| E1 | Sweep prioritization | Phase 1: μ ∈ {2,3}, τ ∈ {0.7,1.0} (24 combos, not 1440). Expand only if ambiguous | ☐ |
| E2 | Perplexity budget | Use layer output error as proxy for all but top 2-3 configs. Full perplexity only on finalists | ☐ |
| E3 | Per-layer runtime target | < 10 minutes per layer configuration | ☐ |
| E4 | imatrix timing | Hessian diagonal for 512 samples through 45M-param layer: estimate 30-60 min on CPU, 5-10 min on GPU | ☐ |

---

## F. Correctness Verification

| # | Prerequisite | Specification | Status |
|---|-------------|---------------|--------|
| F1 | Unit test: asymmetric ternary | Verify `asymmetric_ternary_fit` produces exactly {−1, 0, 1} outputs with correct row-wise offsets | ☐ |
| F2 | Unit test: monotonic decrease | Verify ||R_{i+1}||_F² ≤ ||R_i||_F² on synthetic data with known weights | ☐ |
| F3 | Unit test: column reorder | Verify SSR preserves layer function (P-1 permutation, no information loss) | ☐ |
| F4 | Baseline sanity check | Symmetric sign-threshold baseline produces RED (confirms test validity, not false negative) | ☐ |

---

## Critical Path Items (Must Resolve Before Phase 0 Starts)

1. **C2** — `asymmetric_ternary_fit` is the core algorithm. Without it, nothing else matters. Requires literature implementation or novel design based on PT2-LLM/CAT-Q.
2. **C6** — k dimension bug must be fixed before any computation is run. Wrong k = wasted hours.
3. **C5** — Softened ternarization is listed as a key design decision but absent from code. Required for stable convergence of alternating fit.
4. **B1** — Model must be downloadable and loadable before any calibration or decomposition.
5. **C3** — imatrix computation is the most expensive prerequisite. Need to estimate and possibly simplify (e.g., use fewer calibration samples for Phase 0).
