# Project Plan: Ternary Neural Network Conversion (TCD Method)

## Executive Summary
Convert existing FP32/BF16 models with open weights to pure ternary networks (weights and activations ∈ {-1, 0, 1}) without retraining, preserving activation pathways through algorithmic "magical" reconstruction.

---

## High-Level Project Plan

### Phase 0: Feasibility Validation (Weeks 1-2) — **FAIL FAST**
- **Goal**: Prove factorized ternary viability (ExTernD-Lite) on single layer
- **Scope**: One MLP layer from small model; factorized B diag(D) C decomposition
- **Success Criteria**: Perplexity < 5% vs FP16, output error < 5%, BPW < 6 (SOTA-aligned)
- **Deliverable**: Working Python script + decision gate
- **Key Change**: Replaced per-row symmetric sign-threshold with ExTernD-Lite factorized asymmetric ternary decomposition based on PHASE0_REVIEW.md gap analysis

### Phase 1: Single-Layer Conversion (Weeks 2-3)
- **Goal**: Convert full MLP block with ternary activations
- **Scope**: gate_proj + up_proj + down_proj in one transformer block
- **Success Criteria**: Layer output error < 5% vs FP16 reference
- **Deliverable**: Layer converter module

### Phase 2: Multi-Layer Chain (Weeks 3-4)
- **Goal**: Test error accumulation across 3+ sequential layers
- **Scope**: 3 consecutive transformer blocks
- **Success Criteria**: Output degradation < 10% vs single-layer error
- **Deliverable**: Chain validation + error correction strategy

### Phase 3: Full Model Conversion Pipeline (Weeks 4-6)
- **Goal**: Automated converter for entire model
- **Scope**: All Linear layers in transformer (Q/K/V/O, MLP)
- **Success Criteria**: Perplexity within 5% of original on validation set
- **Deliverable**: `ternary_converter.py` CLI tool

### Phase 4: Inference Engine Prototype (Weeks 6-8)
- **Goal**: Demonstrate CPU speedup
- **Scope**: Custom kernel for ternary matmul (AVX-512/NEON)
- **Success Criteria**: 10x speedup vs FP32 PyTorch on CPU
- **Deliverable**: `ternary_engine` C++ library + Python bindings

### Phase 5: Hardware Validation & Optimization (Weeks 8-12)
- **Goal**: Production-ready inference
- **Scope**: Full model inference, memory profiling, quantization-aware tuning
- **Success Criteria**: Llama-3-8B equivalent runs on 16GB RAM CPU at >20 tok/s
- **Deliverable**: Benchmark report + optimized engine

---

## Detailed Phase 0 Plan: Feasibility Validation — **FAIL FAST**

### Objective
Prove that FP32 weight matrices of LLM layers can be approximated by **factorized ternary decompositions** of the form **W ≈ B diag(D) C** (B, C ∈ {-1,0,1}, D ∈ ℝ) with asymmetric ternary quantization per row, activation-aware output error minimization, and block-wise processing. This is the **ExTernD-Lite** algorithm — the minimum viable approach identified by literature review as necessary to avoid a false-negative RED result.

### Scope
- Single Linear layer (MLP `gate_proj`) from a small model
- FP32/BF16 weights → factorized ternary {B, D, C}
- Real activation distributions (not random noise)
- Activation-aware output error calibration
- Pure Python/PyTorch (no custom kernels)

### Why This Revision Is Required
The original Phase 0 plan used per-output-neuron symmetric sign-threshold decomposition — a method that **provably fails** on LLM layers:
- Single-plane ternary (TNT) drops 8% Top-1 on VGG-16 ImageNet (CNN!) — transformers are more sensitive
- Unimodal Gaussian weight distributions are mismatched to symmetric {-1,0,1}: ~68% of weights quantize to zero within 1σ
- Weight-space-only calibration ignores the weight-activation interaction that determines actual output error
- No outlier handling: outliers dominate the residual when processing an entire layer as one matrix

See `PHASE0_REVIEW.md` for full evidence and gap analysis against 2024-2026 SOTA methods (ExTernD, PTQTP, PT2-LLM, TWLA, CAT-Q).

### Success Criteria (Gate to Phase 1)
| Metric | Threshold | Rationale | SOTA Reference |
|--------|-----------|-----------|----------------|
| Perplexity Delta | < 5% vs FP16 (WikiText-2) | End-to-end quality proxy | ExTernD μ=3: +3.2% |
| Effective BPW | < 6.0 bits/weight (Shannon entropy) | Memory efficiency | PTQTP K=4: ~6.8 bpw |
| Sparsity | > 40% zeros in B,C | Compression benefit | ExTernD τ=1.0: 57% |
| Max Layer Output Error | < 5% (\|XW−XŴ\|_F / \|XW\|_F) | Activation-aware quality | PT2-LLM AGA target |
| Runtime | < 10 min per layer | Practical for full model | — |

### Steps

#### Step 1: Column Reordering & Block Partitioning
```python
def preprocess_layer(W_fp, X_calib, G=128):
    """
    W_fp: [out_features, in_features]
    X_calib: [num_tokens, in_features]
    """
    # 1a. Column reordering by structural similarity (SSR)
    # Clusters columns by cosine similarity; groups outliers together
    # so they don't distort normal columns
    P = column_reorder_by_cosine(W_fp)
    W = W_fp[:, P]
    X = X_calib[:, P]

    # 1b. Block partition (G=128 columns per block)
    # Error compensation across blocks prevents unbounded drift
    num_blocks = (W.shape[1] + G - 1) // G
    return W, X, P, num_blocks
```

**Rationale**: PT2-LLM SSR shows that grouping outliers prevents them from distorting normal columns. Block-wise processing (GPTQ/PTQTP/PT2-LLM G=128) enables error compensation within manageable chunks.

#### Step 2: Asymmetric Ternary Factorized Decomposition (ExTernD-Lite)
```python
def extrend_lite_decompose(W_fp, X_calib, mu=2.0, tau=1.0, G=128):
    """
    W_fp: [out_features, in_features]
    X_calib: [num_tokens, in_features]
    Returns: B [out, k_total], D [k_total], C [k_total, in]
             with B,C ternary, D real
    """
    m, n = W_fp.shape
    k = int(mu * min(m, G))  # G columns per block — C is block-structured

    W, X, P, num_blocks = preprocess_layer(W_fp, X_calib, G)

    # B is shared (global) across blocks; C is per-block
    B_list, D_list, C_list = [], [], []

    for b in range(num_blocks):
        cols = slice(b * G, min((b + 1) * G, n))
        W_block = W[:, cols]
        X_block = X[:, cols]

        R = W_block.clone()
        B_block, C_block, D_block = [], [], []

        for i in range(k):
            # Alternating ternary fit (15 iters)
            # Asymmetric ternary: {−α+μ, μ, α+μ} per row
            u = asymmetric_ternary_fit(R.T @ X_block, tau)
            v = asymmetric_ternary_fit(R @ u, tau)

            # Optimal scale (least squares ridge regression)
            d = (u @ R @ v) / (u.norm()**2 * v.norm()**2 + 1e-8)

            # Deflate residual (proven monotonic decrease per ExTernD)
            R = R - d * u.outer(v)

            B_block.append(u)
            C_block.append(v)
            D_block.append(d)

        B_list.append(torch.stack(B_block, dim=1))
        C_list.append(torch.stack(C_block, dim=0))
        D_list.append(torch.stack(D_block, dim=0))

    B = torch.cat(B_list, dim=1)
    C = torch.cat(C_list, dim=1)
    D = torch.cat(D_list, dim=0)
    return B, D, C, P
```

**Key design decisions from SOTA:**
- **Factorized structure** B diag(D) C: reduces parameters from K×m×n to k×(m+n) (ExTernD)
- **Asymmetric ternary** per row: {−α_i + μ_i, μ_i, α_i + μ_i} handles non-zero mean weights (PT2-LLM, CAT-Q)
- **Softened ternarization**: differentiable transition near threshold for stable convergence (CAT-Q ST)
- **Monotonic residual decrease**: ||R_{i+1}||_F² ≤ ||R_i||_F² proven (ExTernD)

**Algorithm Variants to Test:**
1. **ExTernD-Lite** (recommended): factorized + asymmetric + activation-aware
2. **Symmetric baseline** (original plan): per-row sign-threshold — included as contrast
3. **PTQTP-style**: 2-4 trit-planes with progressive group-wise fitting
4. **Symmetric vs Asymmetric**: same K, compare with/without μ offset

#### Step 3: Activation-Aware Calibration & Importance Weighting
```python
def compute_calibration_statistics(model, layer_idx, dataset, hooks):
    """
    Collects: X (inputs), Y_orig (reference outputs), Hessian diagonal
    """
    # 3a. Register forward hooks for full block inputs + outputs
    # (not just single layer — error compensation needs adjacent layers)
    hooks.register_full_block_hooks(model, layer_idx)

    # 3b. Run calibration data through model
    # Dataset: wikitext-2-raw-v1, 128-512 samples (imatrix standard)
    # Collect: X [num_tokens, hidden_dim], Y_orig [num_tokens, intermediate_dim]

    # 3c. Compute importance weights (imatrix / Hessian diagonal)
    # llama.cpp imatrix statistics: diagonal of Hessian H_ii
    # Used to weight output error by importance
    importance = compute_hessian_diagonal(model, dataset)
    return X_calib, Y_orig, importance
```

**Rationale**: PT2-LLM AGA minimizes E_x = ||XW^T − XŴ^T||, not E_w = ||W − Ŵ||. ExTernD importance-weighted variant uses llama.cpp imatrix statistics for non-obvious correction to the V-step. Weight-space-only calibration (original plan) cannot detect output-degrading decompositions.

#### Step 4: Synthetic Validation (Pre-Calibration Sanity Check)
- Generate synthetic activations X from **heavy-tailed distribution** (not N(0,I)) — LLM activations have significant outliers
- Add controlled outlier columns to X to test robustness
- Compare Y_orig = XW^T vs Y_ternary = X @ (B diag(D) C)^T
- **Metric**: Output error E_x = ||XW^T − XŴ^T||_F / ||XW^T||_F (not weight-space cosine)
- Verify decomposition quality on both clean and outlier-corrupted inputs

**Rationale**: Original plan used X ~ N(0,I) which masks the outlier sensitivity that makes LLM ternary quantization difficult. TWLA and PT2-LLM both show that real activation distributions have heavy tails requiring special handling.

#### Step 5: Parameter Sweep Experiments
| Dimension | Values | SOTA Reference |
|-----------|--------|----------------|
| μ (inner rank multiplier) | 1.0, 2.0, 3.0, 4.0 | ExTernD: μ=3 optimal for Qwen3.5-4B |
| τ (sparsity threshold) | 0.5, 0.7, 1.0, 1.5, 2.0 | ExTernD: τ=0.7→41%, τ=2.0→87% |
| G (block size) | 64, 128, 256 | GPTQ/PTQTP/PT2-LLM: G=128 |
| Asymmetric vs symmetric | Both | PT2-LLM, CAT-Q |
| Row-wise offset μ | 0 (sym), learned (asym) | PT2-LLM |
| Column reordering | ON/OFF | PT2-LLM SSR |
| Importance weighting | ON/OFF | ExTernD, GPTQ |
| Number of ternary planes | 2, 3, 4 | PTQTP K=2,3,4 |

**Outputs per run:**
- Perplexity on WikiText-2 (or layer output error as proxy)
- Effective BPW (Shannon entropy of B,C including zeros)
- Sparsity (% zeros in B,C)
- Layer output error: ||XW − XŴ||_F / ||XW||_F
- Perplexity delta vs FP16 baseline
- Runtime per layer

#### Step 6: Analysis & Gate Decision
- Plot: μ vs Output Error (multiple τ values)
- Plot: τ vs Sparsity vs Output Error (Pareto frontier)
- Plot: With/without column reordering comparison
- Plot: Symmetric vs Asymmetric ternary comparison
- Table: Best config per layer type with full metrics
- **DECISION**: GREEN / YELLOW / RED (see framework below)

### Deliverables (Phase 0)
```
phase0_feasibility/
├── config.yaml
├── main.py                    # Orchestrates full sweep
├── decomposition/
│   ├── __init__.py
│   ├── core.py               # extrend_lite_decompose()
│   ├── asymmetric.py         # asymmetric_ternary_fit(), row-wise {−α+μ, μ, α+μ}
│   ├── reordering.py         # column_reorder_by_cosine() (SSR)
│   └── vectorized.py         # Batched block-wise implementation
├── calibration/
│   ├── __init__.py
│   ├── loader.py             # Model + dataset loading (128-512 samples)
│   ├── hooks.py              # Full block input/output hooks
│   └── importance.py         # Hessian diagonal / imatrix computation
├── analysis/
│   ├── metrics.py            # E_x, BPW, sparsity, perplexity delta
│   └── plotting.py           # Decision plots (Pareto frontiers)
├── results/
│   ├── sweep_results.csv
│   └── decision_report.md    # GREEN/YELLOW/RED + evidence
└── requirements.txt
```

### Risk Register & Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Factorized decomposition still insufficient | Medium | Project death | Fall back to PTQTP trit-planes (K=4) as alternative |
| Symmetric baseline confounds results | Medium | False conclusion | Always run symmetric baseline alongside ExTernD-Lite for contrast |
| Activation distributions differ from calibration | High | False positive | Use 512 samples minimum; test 3 layers (early/mid/late) |
| Importance weighting (imatrix) unstable | Medium | Noisy metrics | Cross-validate with plain output error |
| Memory OOM on large layers | High | Blocked | Block-wise processing (G=128); gradient checkpointing style |
| Computational cost of sweep is too high | Medium | Blocked | Start with μ ∈ {2,3}, τ ∈ {0.7,1.0}; expand only if ambiguous |
| Asymmetric ternary implementation buggy | Medium | Wrong results | Unit test: verify {−α+μ, μ, α+μ} produces correct 3-valued output |

### Go/No-Go Decision Framework
```python
# After running ExTernD-Lite on gate_proj (layer 12) with mu=2,3,4:
GREEN = (
    perplexity_delta_pct < 5%       # WikiText-2 perplexity vs FP16
    and effective_bpw < 6.0          # Shannon entropy bits/weight
    and sparsity > 0.4               # >40% zeros in B,C
    and max_layer_output_error < 0.05  # ||XW - XŴ||_F / ||XW||_F
)

YELLOW = (
    perplexity_delta_pct < 10%
    and effective_bpw < 8.0
)

RED = otherwise  # KILL PROJECT
```

**Critical note**: The symmetric sign-threshold baseline (original Phase 0 algorithm) is **expected to produce RED** — this is a *false negative risk*, not a true negative. The GREEN/YELLOW/RED decision applies only to the **ExTernD-Lite algorithm** (or better). If ExTernD-Lite also fails, the approach is genuinely non-viable.

---

## Detailed Phase 1 Plan: Atomic Test (Fail Fast)

### Objective
Validate that factorized asymmetric ternary decomposition (ExTernD-Lite) can approximate a single FP32 Linear layer's weight matrix with minimal inner rank expansion (μ=2-4).

### Prerequisites
- Python 3.10+
- 8GB+ RAM (CPU only, no GPU required)
- Internet for model download (~1-4GB)

### Environment Setup (30 min)
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch transformers datasets scikit-learn matplotlib numpy
```

### Step 1: Model & Layer Selection (15 min)
- **Model**: `google/gemma-2-2b` or `Qwen/Qwen2.5-0.5B` (HF Hub)
- **Target Layer**: MLP `gate_proj` at layer index 12 (middle layer)
- **Extract**: Weight matrix `W_orig` shape `[hidden_dim, intermediate_dim]`

### Step 2: Calibration Data Collection (30 min)
- **Dataset**: `wikitext-2-raw-v1`, **128-512 samples** (imatrix standard)
- **Hook**: Register forward hooks for **full block** inputs and outputs (adjacent layers needed for error compensation)
- **Capture**: X [num_tokens, hidden_dim], Y_orig [num_tokens, intermediate_dim] for each layer
- **Importance**: Compute Hessian diagonal (imatrix) for weighting output error

### Step 3: Asymmetric Ternary Factorized Decomposition (Core - 2-3 hours)

**Algorithm** — ExTernD-Lite: factorized B diag(D) C with asymmetric ternary quantization:
```
Input: W_fp ∈ ℝ^{m×n}, X_calib ∈ ℝ^{t×n}, μ (inner rank), τ (threshold)
Output: B ∈ {-1,0,1}^{m×k}, D ∈ ℝ^k, C ∈ {-1,0,1}^{k×n}

1. Column reorder W and X by structural similarity (SSR)
2. Partition into blocks of G=128 columns
3. For each block:
   R = W_block
   for i in 1..k:
       u = asymmetric_ternary_fit(R^T X, τ)    # [m], {−α+μ, μ, α+μ} per row
       v = asymmetric_ternary_fit(R u, τ)        # [G]
       d = argmin_d ||R − d·u⊗v||_F             # optimal scale
       R = R − d·u⊗v                             # deflate (monotonic)
       append u, v, d
4. Concatenate B, D, C across blocks
```

**Threshold**: τ is a tunable sparsity dial (ExTernD). Start with τ = 1.0.

**Key Properties (from SOTA literature):**
- **Monotonic residual decrease**: ||R_{i+1}||_F² ≤ ||R_i||_F² (ExTernD)
- **Asymmetric ternary**: {−α_i + μ_i, μ_i, α_i + μ_i} per row (PT2-LLM)
- **Activation-aware**: fit against X-calibration, not just W (PT2-LLM AGA)

**Algorithm Variants to Test:**
1. **ExTernD-Lite** (recommended): factorized + asymmetric + activation-aware
2. **Symmetric baseline**: per-row sign-threshold — included as contrast only

### Step 4: Ternary Forward Simulation (30 min)
For each ternary plane i:
- `Y_i = X @ (d_i * B_i C_i)^T` where B_i, C_i are ternary columns
- Accumulate: `Y_ternary = Σ d_i * (X @ (B_i C_i)^T)`
- Compute output error: E_x = ||Y_orig − Y_ternary||_F / ||Y_orig||_F

### Step 5: Metrics & Decision Gate (30 min)

| Metric | Formula | Target | SOTA Reference |
|--------|---------|--------|----------------|
| Perplexity Delta | (PP_light − PP_fp) / PP_fp | < 5% | ExTernD μ=3: +3.2% |
| Output Error | ||XW − XŴ||_F / ||XW||_F | < 5% | PT2-LLM AGA |
| Effective BPW | Shannon entropy of B,C | < 6.0 | PTQTP K=4: ~6.8 |
| Sparsity | % zeros in B,C | > 40% | ExTernD τ=1.0: 57% |

**Sweep**: μ ∈ {1, 2, 3, 4}, τ ∈ {0.7, 1.0, 1.5}, block size G ∈ {64, 128}
**Plot**: μ (x-axis) vs Output Error / Sparsity (y-axis)

### Decision Criteria

| Outcome | Condition | Action |
|---------|-----------|--------|
| **GREEN** | PP delta < 5% AND BPW < 6 AND error < 5% | Proceed to Phase 2 |
| **YELLOW** | PP delta < 10% AND BPW < 8 | Proceed with wider ensemble |
| **RED** | PP delta > 10% at μ=4 | **KILL PROJECT** |

### Code Structure (Single Script)
```
phase1_atomic_test/
├── config.yaml           # Model, layer, μ, τ, G values
├── main.py               # Orchestrates full pipeline
├── decomposition.py      # ExTernD-Lite (factorized asymmetric ternary)
├── calibration.py        # Data loading, hooks, imatrix computation
├── metrics.py            # E_x, BPW, sparsity, perplexity delta
└── requirements.txt
```

### Timeline: 1-2 Days Total
- Day 1: Setup + Calibration + Core Algorithm
- Day 2: Simulation + Metrics + Decision

### Risk Mitigation
- **Threshold sensitivity**: Sweep τ ∈ {0.7, 1.0, 1.5}
- **Rank sensitivity**: Test μ ∈ {1, 2, 3, 4}
- **Baseline comparison**: Always compare against symmetric sign-threshold baseline
- **Activation quantization**: Defer ternary activation test to Phase 2

---

## Success Definition for Phase 1
> "A single MLP Linear layer can be factorized into ternary matrices B diag(D) C (with asymmetric per-row ternary quantization) achieving <5% perplexity delta and <5% output error on real activation distributions, using only analytical computation (no gradients, no backprop)."

---

## Next Steps After Phase 1
1. If GREEN → Implement full MLP block conversion (gate+up+down proj)
2. Add ternary activation function (thresholded ReLU/GeLU)
3. Test 3-layer chain for error accumulation
4. Build automated layer converter module