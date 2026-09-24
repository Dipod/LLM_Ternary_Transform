# Proposal

## Why

Phase 0 is meant to be a fail-fast feasibility gate for factorized ternary weight
decomposition, but the repository's five planning documents
(`PROJECT_PLAN.md`, `PHASE0_PREREQUISITES.md`, `PHASE0_ACQUISITION_PLAN.md`,
`PHASE0_REVIEW.md`, `idea source.txt`) are AI-generated and, on independent
review, are unusable as a plan of record:

- The proposed algorithm does not execute. The alternating updates are
  shape-invalid (`u = ...(R.T @ X_block)` needs `m == t`; `R.T @ v` uses `v`
  before assignment), and the block `torch.cat` produces factors whose inner
  dimensions do not match, so `B diag(D) C` cannot close. One code block is
  self-labelled `WRONG` and then returns undefined names.
- It does not build the factorization it claims: with a per-block `B`, the
  asserted `k(m+n)` parameter count is false (≈93M entries vs 45M dense for the
  worked example — no compression at all).
- It optimizes weight-space residual while claiming activation-aware output error
  `E_x`; `E_x` is never the quantity being minimized.
- Its gate is vacuous: "effective BPW = Shannon entropy of B,C" is at most
  `log2(3) ≈ 1.585` bits/entry, so `< 6.0` can never fail; the gate is also
  defined twice with different GREEN/RED sets.
- Citations are partly fabricated or misattributed: `github.com/ExTernD` returns
  404; `Qwen2.5-0.5B` `gate_proj` is `[4864, 896]`, not `[4096, 11008]`
  (that is Llama-7B); TWLA is 2026 not 2024; Tequila is 2025 not 2024; AWQ is
  not Hessian/block-wise.
- "No gradients" is contradicted by the imatrix (autograd), CAT-Q softened
  ternarization, and "learned" offset the same documents require.

A corrected Phase 0 must therefore be re-established on verified method and
mathematics before any experiment is run, and the gate must be redefined so it
can actually fail.

## What Changes

- **BREAKING** Replace the decomposition design with a shape-explicit,
  gradient-free factorized ternary fit derived from the verified primary source
  (ExTernD, `arXiv:2607.13511`): a **global** `B ∈ {-1,0,1}^{m×k}`,
  `D ∈ ℝ^k`, `C ∈ {-1,0,1}^{k×n}` with `k = μ·min(m,n)`. The per-block
  `k = μ·min(m,G)` and the block concatenation are removed; column blocks become
  a compute strategy only, never a storage structure.
- Introduce a single canonical Phase 0 gate on **one** Linear layer using
  layer-local metrics (relative output error `E_x`, energy preservation,
  sparsity, effective BPW) with end-to-end perplexity explicitly deferred to a
  later phase.
- Redefine effective BPW to a counting rule normalised by `m·n` (stored trit bits
  of `B`/`C` plus `D`), replacing the always-true per-entry entropy metric.
- Constrain Phase 0 to strictly gradient-free optimization: closed-form and
  alternating least squares only; no autograd, no softened/learnable components;
  importance, if used, comes from activation second moments.
- Add a minimal Phase 0 harness (model/layer/calibration loading, decomposition
  core, metrics, a recorded run under `results/<experiment-id>/`) and a
  known-answer validity test on the real gate metric.
- Retain a symmetric sign-threshold baseline purely as the contrast expected to
  fail, with its criterion restated on the real gate metric rather than a
  weight-cosine number.

## Capabilities

### New Capabilities

- `phase0/ternary-decomposition`: factorized ternary decomposition of a single
  weight matrix fitted against calibration activations — the fit algorithm, its
  shape invariants, the column-reordering step, the baseline, and the
  determinism requirements.
- `phase0/feasibility-gate`: the Phase 0 metric definitions and the single
  canonical GREEN/YELLOW/RED decision rule, including the effective-BPW counting
  rule and the deferred-perplexity boundary.

### Modified Capabilities

None — no specs exist yet in `openspec/specs/`.

## Impact

- **New code** (all new; the repository has no source yet): a Phase 0 package
  implementing the decomposition, calibration/hooks, and metrics, plus tests.
- **Environment**: Phase 0 runs on CPU PyTorch; the ML stack (`torch`,
  `transformers`, `datasets`) must be installed for the first run. AMD ROCm
  acceleration is out of scope here and tracked separately.
- **Model/data**: `Qwen/Qwen2.5-0.5B` (`hidden=896`, `intermediate=4864`,
  24 layers) and a WikiText-2 calibration slice; weights are never committed.
- **Documents**: `PROJECT_PLAN.md` and `PHASE0_*.md` remain on disk as
  historical inputs but are no longer the plan of record; the delta specs and
  `design.md` supersede their algorithmic and gate claims. `openspec/project.md`
  and `.kilo/agents/py-architect.md` repeat several fabricated numbers and must
  be corrected in the same change.
- **Recorded results**: no prior `results/` exist, so nothing is invalidated.
