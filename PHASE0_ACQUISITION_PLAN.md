# Phase 0 Prerequisites — Acquisition Plan

> How to obtain each prerequisite. Prioritized by critical path.

---

## Critical Path First

### C2 — asymmetric_ternary_fit (BLOCKER: Phase 0 cannot start without this)

**Approach** (based on PT2-LLM ITF + CAT-Q softened ternarization):

```python
# Softened ternarization (CAT-Q style) — differentiable
def softened_ternarize(x, tau, beta=10.0):
    # beta = sharpness parameter (start 10, anneal to 1)
    z = torch.empty_like(x).zero_()
    pos = torch.where(x > tau, (1 + (1 - tau/beta) / (1 + exp(-beta*(x - tau)))) / 2, z)
    neg = torch.where(x < -tau, -(1 + (1 - tau/beta) / (1 + exp(beta*(x + tau)))) / 2, z)
    return pos + neg  # ∈ [-1, 1], approaches {-1,0,1} as beta→∞

def asymmetric_ternary_fit(x, tau, beta=10.0):
    """
    Input: x ∈ ℝ^d (real vector)
    Output: t ∈ {-1,0,1}^d, alpha (row-wise scale)
    1. Softened ternarize: z = softened_ternarize(x, tau, beta)
    2. Row-wise scale: alpha = <x, z> / ||z||²
    3. t = alpha * z (produces values in {-1, 0, 1} after rounding)
    """
```

**Implementation steps**:
1. Start with hard ternary (beta=∞, sign-threshold) — simplest correct version
2. Verify: does hard asymmetric ternary fit reduce residual monotonically on random data?
3. Then add softened ternarization (soft beta=10) for stable convergence
4. Implement alternating fit: fix scales → update ternary → fix ternary → update scales
5. **Verify monotonic decrease** (F2 unit test)

**Unit tests**: F1 (produces {−1,0,1}), F2 (monotonic decrease)

---

### C6 — Fix k dimension bug (BLOCKER: wrong k = 8192 → correct k = 256)

**Bug**: `k = int(mu * min(m, n))` — produces 2×4096 = 8192 planes (infeasible)
**Fix**: `k = int(mu * min(m, G))` — C is per-block with G columns, inner rank applies to block width

For gate_proj (4096×11008) with G=128: k = 2×128 = **256 planes** (feasible)

---

## Environment & Configuration

### A1-A7 — Environment & Dependencies
- Create venv, install: `pip install torch transformers datasets scikit-learn matplotlib numpy`

### D1 — Sweep configuration
- Define `config.yaml` with reduced sweep: μ∈{2,3}, τ∈{0.7,1.0}, G∈{64,128}

### D2 — Random seeds
- Lock: `torch.manual_seed(42)`, `np.random.seed(42)`, `torch.use_deterministic_algorithms(True)`

### E1 — Sweep prioritization
- Phase 0: μ∈{2,3}, τ∈{0.7,1.0} (reduced from full space). Expand only if ambiguous.

---

## Infrastructure Components (any order)

### C1 — column_reorder_by_cosine

**Algorithm**:
1. Compute cosine similarity matrix between all column pairs: `sim[i,j] = cos(W[:,i], W[:,j])`
2. Greedy ordering: start with column of max ||W[:,j]||, repeatedly pick next column with highest avg similarity to already-placed columns (TSP-like, O(n²))
3. For 11008 columns: similarity matrix = 11008² × 4 bytes = 484 MB. Use chunked computation.

**Unit test**: F3 (verify it's a valid permutation, no columns lost/duplicated)

### C4 — register_full_block_hooks

Capture input + output of target layer AND adjacent layers simultaneously. Exact hook implementation depends on model architecture (transformer_layers, attention/MLP submodules).

### C3 — compute_hessian_diagonal

**Approach** (simplified imatrix): diagonal of Hessian H_ii = E[grad_i²]

**Optimization**: Use 128 samples for Phase 0 (reduced from 512). Sensitivity analysis later.

---

## Refinement (after C2 works)

### C5 — Softened ternarization

**Depends on**: C2 (asymmetric_ternary_fit works with hard ternary first)

**Steps**:
1. Verify hard asymmetric ternary produces correct monotonic decrease (F2 test)
2. Add softened version: replace `sign()` with differentiable sigmoid-based soft threshold
3. Compare convergence speed: hard vs soft at beta = {1, 5, 10, 50, ∞}
4. Choose beta that gives stable convergence within alternating iterations

---

## Unit Tests (F1-F4)

| Test | What | Input | Expected |
|------|------|-------|----------|
| F1 | asymmetric_ternary_fit output values | Random vector [100] | Only {−1, 0, 1} |
| F2 | Monotonic residual decrease | Random matrix + algorithm | ||R_{i+1}||² ≤ ||R_i||² |
| F3 | Column reorder validity | Matrix [10, 100] | Permutation of 0..99, no dupes, no missing |
| F4 | Baseline produces RED | Random matrix | Cosine < 0.85 at K=8 (confirming test validity) |

---

## Model & Data

### B1 — Model
- `AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", torch_dtype=torch.bfloat16)`

### B2 — Calibration data
- `datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train").select(range(512))`

### B3 — Synthetic distribution
- Heavy-tailed: Student-t (df=3) or Laplace — to be decided after experimentation with real activation profiles

### D3 — Target layer verification
- Inspect model config, confirm gate_proj at layer 12 exists

### D4 — Symmetric baseline
- Implement original symmetric sign-threshold algorithm as contrast (expected RED, confirms test validity)

---

## Summary

**Critical path**: C2 → C6 → F2 → Phase 0 can start
**All items**: C1, C2, C3, C4, C5, C6, F1, F2, F3, F4, A1-A7, B1-B3, D1-D4, E1

**Order**: Environment → C2+C6 (critical path) → C1, C3, C4 (parallel) → C5 (after C2) → F1-F4 → B1, B2, D3, D4 → Phase 0 begins
