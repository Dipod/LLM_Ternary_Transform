# Design — Phase 0 ternary decomposition feasibility

## Context

The change replaces an unusable algorithmic design (see `proposal.md`) with a
specification derived from verified primary sources. The method of record is
ExTernD (`arXiv:2607.13511`, Eq. 1–5, §2.1–2.5, Appendix A), with PT2-LLM
(`arXiv:2510.03267`) used only for the optional structural-similarity reordering.
Repo planning documents were read as a claim list and are not a source.

Key correction: ExTernD ternarizes with a **symmetric adaptive** threshold
`T_τ` and carries the asymmetry in the real scale `D`, not in a per-row affine
grid. The repo's `{−α+μ, μ, α+μ}` grid is PT2-LLM/CAT-Q and is not representable
in `B diag(D) C` with a single scalar per plane; it is dropped.

## Goals and non-goals

**Goals.** A single gradient-free decomposition of one Linear weight matrix,
correct in shape and math, with one canonical fail-fast gate on one layer, and a
recorded reproducible run.

**Non-goals.** Full-model conversion; ternary activations; end-to-end perplexity;
inference kernels or speedups; AGA (activation-space fitting) and CAT-Q softened
ternarization; GPU acceleration (deferred to a separate change).

## Decisions

### D1 — Factorized form and rank

`A ≈ B diag(D) C` with `A ∈ ℝ^{m×n}`, `B ∈ {-1,0,1}^{m×k}`,
`D ∈ ℝ^k`, `C ∈ {-1,0,1}^{k×n}`, and `k = μ · min(m, n)`.
*Rationale:* ExTernD Eq. (1), §2.1. *Alternatives rejected:* per-row ensemble
(no low-rank sharing, does not compress); `k = μ·min(m,G)` with block-structured
`C` (contradicted by the source and yields no compression).

### D2 — Ternary operator

`T_τ(w)_i = sign(w_i)·1[|w_i| > τ·(1/L)Σ_j|w_j|]`, applied per vector, with the
safeguard that a single largest entry is kept if the result would be all zeros.
*Rationale:* ExTernD Eq. (2), §2.2; positively scale-invariant, so it is applied
to the raw residual targets without a separate normalization. *Alternative
rejected:* fixed `0.75·E|W|` threshold (TWN/PT2-LLM); the τ dial is the one that
controls sparsity in the source.

### D3 — Fit procedure and update order

Greedy sequential deflation. For component `i`: `v ← T_τ(Rᵀu)`, `u ← T_τ(Rv)`,
repeated **15** times; then `d_i = (uᵀRv)/(‖u‖²‖v‖²)`; then `R ← R − d_i u vᵀ`.
Scales of a batch are solved jointly by `[(UᵀU)⊙(VᵀV)]d = diag(UᵀRV)`.
*Rationale:* ExTernD §2.3 (15 inner iterations; `v` then `u`) and §2.4 Eq. (4),
which reduces to the scalar form at `b = 1`. *Alternative rejected:* joint solve
of all `k` components (source: "fails completely", 0–50% energy).

### D4 — Component batching is compute-only

The optional batched path processes components in blocks of
`b = min(256, ⌊min(m,n)/8⌋)` using the decorrelated least-squares targets
`V ← T_τ([(UᵀU+εI)⁻¹UᵀR]ᵀ)`, `U ← T_τ(RV(VᵀV+εI)⁻¹)`, and deflates the residual
sequentially across blocks. *Rationale:* ExTernD §2.4. It MUST be
cross-checked against the sequential reference on the same input (cross-check
gate) and MUST NOT change the stored factor shapes. *Alternative rejected:*
per-column batching without decorrelation (source: energy collapses 99.5%→50%
at `b=256`).

### D5 — Initialization of `u` (source does not state one)

Each component is initialized with the only sourced candidate: the best
single-entry pair at the largest-magnitude residual entry
(`u = e_{i*}`, and the first target `Rᵀu` drives `v`). The batched path
initializes `U` with the `b` largest-magnitude residual coordinates as one-hot
columns and `V = T_τ(RᵀU)`. *Rationale:* ExTernD Appendix A Prop. 2 specifies
exactly this pair as its alternative step; the §2.3 loop leaves the initial `u`
unspecified. The choice is deterministic and recorded as ours, not the source's.

### D6 — Gradient-free

No autograd anywhere in the decomposition. Importance weighting (D7), if enabled,
uses only activation second moments. *Rationale:* project decision; ExTernD
Appendix A needs no gradients, so the premise is achievable. *Removed:*
CAT-Q softened ternarization, "learned" offsets, and the autograd imatrix of the
old prerequisites doc. *Consequence:* the monotonicity proof (Prop. 1) holds for
any nonzero ternary `u,v` with the LS scale; the geometric bound
`(1−1/mn)^k` requires the augmented single-entry option of Prop. 2, which we do
not implement — we claim only **non-increase**, plus an explicit stall check.

### D7 — Importance weighting (optional sweep dimension)

`h_j = E[x_j²]` per input channel, computed from calibration activations;
`h̃ = h/max(h) + λ`. The weighted objective uses `R⊙h̃` as target in the `u` and
scale steps; the `v` step keeps the plain residual `R` (the weight cancels
there). *Rationale:* ExTernD §2.5; it is the only activation-aware signal that
needs no gradients. *Default:* λ=0, importance **off** in the primary run and a
swept variant.

### D8 — Column reordering (optional)

Structural-similarity ordering of input channels, applied identically to `A` and
`X` on the in_features axis, with the inverse permutation retained. Phase 0
implements a deterministic cosine-greedy ordering (our construction, since the
source's Eq. (16) is garbled). *Rationale:* PT2-LLM SSR (§3.3). *Default:* off
in the primary run; swept on/off.

### D9 — Compute backend

PyTorch on **CPU** (fp32 math). *Rationale:* the layer is small
(`[4864, 896]`), CPU is unblocked, and it keeps the fail-fast gate independent of
the AMD/ROCm-on-Windows setup, which needs Python 3.12 and a specific driver
(verified facts; out of scope here). *Alternative rejected:* AMD ROCm now
(setup risk on the critical path); NVIDIA T400 (4 GB, below the recommended 8 GB).

### D10 — Model and target layer

`Qwen/Qwen2.5-0.5B` (official config: `hidden_size=896`,
`intermediate_size=4864`, 24 layers), target `gate_proj` at layer 12,
weight shape `[out, in] = [4864, 896]`. *Rationale:* cheapest representative
MLP projection; ungated on HF; makes `k = 2·896 = 1792` at μ=2. *Alternative
rejected:* Llama-7B shapes from the repo docs (a different model) and
gemma-2-2b (gated, larger); escalation to a 7B-class model is a follow-up only
if Phase 0 passes.

### D11 — Precision and dtype policy

Load the model in bf16 for the calibration forward; compute all decomposition
math in fp32; store `B`,`C` as int8 in {-1,0,1} and `D` as fp32. *Rationale:*
the source states values, not dtypes, so this is our decision; fp32 avoids
accumulation error at μ=2 (`k=1792`), and int8 is the natural ternary container.

## Algorithm specification

```text
Input:  A ∈ ℝ^{m×n}, μ, τ, optional h̃ ∈ ℝⁿ, n_inner = 15, ε = 1e-8
Output: B ∈ {-1,0,1}^{m×k}, D ∈ ℝ^k, C ∈ {-1,0,1}^{k×n},  k = round(μ·min(m,n))

R ← A.clone()                                   # [m,n]
for component block s (size b; b=1 for the sequential reference):
    U, V ← init_block(R, b)                     # [m,b], [n,b]   (D5)
    repeat n_inner times:                       # exactly 15
        V ← T_τ_cols( (UᵀU+εI)⁻¹ Uᵀ R )ᵀ )       # [n,b]
        U ← T_τ_cols( R V (VᵀV+εI)⁻¹ )          # [m,b]
    G ← (UᵀU) ⊙ (VᵀV)                           # [b,b]
    d ← G⁻¹ diag(Uᵀ R V)                        # [b]
    R ← R − U diag(d) Vᵀ                         # [m,n]   (sequential across blocks)
    B[:,s:s+b] ← U; C[s:s+b,:] ← Vᵀ; D[s:s+b] ← d
```

Sequential reference: the same loop with `b=1`, `v ← T_τ(Rᵀu)`,
`u ← T_τ(Rv)`, `d = (uᵀRv)/((uᵀu)(vᵀv)+ε)`, `R ← R − d·u vᵀ`.

Reconstruction: `Â = B @ diag(D) @ C`, shape `[m,n]`.

**Baseline (contrast).** The same procedure at `μ = 1` (single-plane symmetric
ternary, no importance weighting), evaluated on the identical gate metrics; it is
expected to fail and exists to show the gate can fire.

## Metrics and gate

- **Output error (primary):** `E_x = ‖X Wᵀ − X Âᵀ‖_F / ‖X Wᵀ‖_F`, with
  `W = [out,in]` and `X = [tokens,in]`, computed in fp32 on the calibration
  activations.
- **Energy preserved:** `E = 1 − ‖Â − W‖_F² / ‖W‖_F²`.
- **Sparsity:** fraction of zeros in `B` and `C`.
- **Effective BPW:** `bpw_eff = μ·(m+n)/max(m,n)·(2 − sparsity)`
  (ExTernD Eq. (5); 1-bit mask + 1-bit sign per stored trit).
- **Runtime:** wall-clock seconds per layer configuration.
- **Canonical rule (fixed before the run):** GREEN if `E_x ≤ 0.05` and
  `sparsity ≥ 0.40` and `bpw_eff < 6.0`; YELLOW if `E_x ≤ 0.10` and
  `bpw_eff < 8.0`; RED otherwise. End-to-end perplexity is **not** part of this
  gate.

**Reduced sweep:** μ ∈ {2,3} × τ ∈ {0.7,1.0} × importance ∈ {off,on} = 8 runs
(reorder off), plus 8 with reorder on only if the result is ambiguous, plus the
`μ=1` baseline. Each run records all metrics.

## Calibration protocol

WikiText-2 (`wikitext-2-raw-v1`, `train`), first 128 non-empty documents,
tokenized with the model tokenizer, truncated to `seq_len = 512` (single pass,
no stride) → up to 65,536 tokens. Capture the input activations of the target
`gate_proj` (post-RMSNorm) with a forward hook on the calibration split. No
evaluation/held-out split is used at this gate (it is a layer-fit, not a
generalization test); this is stated in the report.

## Module layout and public surface

```text
phase0_feasibility/
├── config.yaml                 # model, layer, μ, τ, b, seeds, slices, tolerances
├── main.py                     # orchestrates the sweep and writes results/<id>/
├── decomposition/
│   ├── ternary.py              # T_tau, T_tau_cols
│   ├── core.py                 # decompose_ternary(A, ...) -> TernaryFactors
│   ├── reordering.py           # cosine_greedy_order(W) -> (perm, inv_perm)
│   └── baseline.py             # symmetric_baseline(...)
├── calibration/
│   ├── loader.py               # model + wikitext slice
│   └── hooks.py                # capture gate_proj inputs
├── analysis/
│   ├── metrics.py              # E_x, energy, sparsity, bpw_eff, runtime
│   └── report.py               # verdict + results/<id>/{config,metrics,report}
└── requirements.txt
```

Public signatures (stable within this change):

- `T_tau(w: Tensor) -> Tensor` — `[L] → [L]`, values in {-1,0,1}.
- `decompose_ternary(A: Tensor, *, mu: float, tau: float, batched: bool, b: int | None, importance: Tensor | None, n_inner: int = 15, eps: float = 1e-8) -> TernaryFactors`
  with `TernaryFactors(B: Tensor, D: Tensor, C: Tensor, k: int)`.
- `cosine_greedy_order(W: Tensor) -> tuple[Tensor, Tensor]` — `(perm, inv_perm)`.
- `layer_output_error(X: Tensor, W: Tensor, W_hat: Tensor) -> float`.
- `effective_bpw(m: int, n: int, k: int, mu: float, sparsity: float) -> float`.

## Determinism and reproducibility

Seeds (`torch`, `numpy`, `random`) come from `config.yaml`; `torch.use_deterministic_algorithms(True)`
is enabled where the CPU path supports it and any fallback is named in the
report. Every run writes `results/<experiment-id>/{config.yaml,metrics.json,report.md}`
with model, dtype, layer, dataset slice, calibration count, hyperparameters,
seeds, hardware, runtime and raw metrics. Library versions (`torch`,
`transformers`) are recorded per run.

## Risks

| Risk | Mitigation |
|---|---|
| Weight-space fit gives poor `E_x` | `E_x` is the gate; importance weighting is a sweep dimension; AGA is an explicit follow-up |
| Batched path diverges from reference | Cross-check gate on identical input |
| `μ=2` at `k=1792` is memory-heavy on CPU | Sequential/batched components, fp32, chunked matmul; layer is small |
| Calibration slice unrepresentative | Fixed slice recorded; slice stated in the report; escalation only after GREEN |
| Decomposition stalls (many `d=0`) | Explicit stall detection and reporting (spec requirement) |

## Out of scope / follow-ups

- AMD ROCm-on-Windows (Python 3.12 + driver) as a separate environment change.
- Vulkan for inference only (llama.cpp), not for PyTorch decomposition.
- AGA / activation-space fitting; CAT-Q softened ternarization; PTQTP trit-planes.
- Full MLP block, multi-layer error accumulation, perplexity, kernels.

## Context sources

- ExTernD, `arXiv:2607.13511v1` — Eq. (1)–(5), §2.1–2.5, Appendix A Prop. 1–2 (read in full).
- PT2-LLM, `arXiv:2510.03267v2` (ICLR 2026) — §3.1–3.3, SSR Eq. (14)–(16); code `github.com/XIANGLONGYAN/PT2-LLM`.
- PTQTP, `arXiv:2509.16989v3` — §2 (dual trit-planes) read only to bound scope.
- CAT-Q, `arXiv:2606.26650v1` (ICML 2026) — §2 read only to state what is dropped.
- Qwen2.5-0.5B `config.json` (HuggingFace) — `hidden_size=896`, `intermediate_size=4864`, 24 layers.
- Hardware facts: local inventory and vendor docs (PyTorch/Vulkan/ROCm), as recorded in `memory.md`.
- **Not checked:** ExTernD's exact weighted normal equations are quoted as derived, not printed in the source; ExTernD/PT2-LLM numbers were not reproduced (no code run).
