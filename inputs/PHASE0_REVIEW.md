# Rigorous Review of Phase 0 Plan

**Verdict: Phase 0 as written will likely produce a RED (kill) result on LLMs. Fundamental algorithmic gaps vs. 2024-2026 SOTA PTQ ternary methods.**

---

## Executive Summary

| Aspect | Phase 0 Plan | 2024-2026 SOTA (ExTernD, PTQTP, PT2-LLM, TWLA, CAT-Q) | Gap Severity |
|--------|--------------|--------------------------------------------------------|--------------|
| **Decomposition Target** | Per-output-neuron (row-wise) ensemble | Per-matrix factorization (ExTernD: B diag(D) C, PTQTP: 2-4 trit-planes) | **CRITICAL** |
| **Ternary Quantization** | Symmetric sign(threshold) | Asymmetric (α⁺, α⁻, μ), learned thresholds, softened ternarization | **CRITICAL** |
| **Weight Distribution** | Assumed amenable to sign-threshold | Unimodal Gaussian → mismatched to {-1,0,1} (TWLA, PT2-LLM) | **CRITICAL** |
| **Calibration Usage** | Weight-space only (||W - W_tern||) | Activation-aware (output error E_x), Hessian/importance weighting | **HIGH** |
| **Outlier Handling** | None | Column reordering (SSR), rotation (KOTMS), block-wise (G=128) | **HIGH** |
| **Scale Factors** | Single α_k per ternary vector | Row-wise α⁺/α⁻/μ, group-wise adaptive ridge (PTQTP) | **HIGH** |
| **Convergence Guarantee** | Heuristic greedy | Proven monotonic residual decrease (ExTernD, TSVD) | **MEDIUM** |
| **Sparsity Control** | Implicit via threshold | Explicit τ parameter, 41-87% sparsity (ExTernD) | **MEDIUM** |

---

## Detailed Findings by Literature

### 1. Single-Plane Ternary is Provably Insufficient for LLMs

**Evidence:**
- **TNT (2019)**: Single optimal ternary vector per weight vector via cosine similarity. Works on CNNs (LeNet-5: -0.21%, VGG-7: -0.14% on MNIST/CIFAR) but **VGG-16 ImageNet: -8% Top-1, -5.34% Top-5**.
- **ExTernD (2026)**: "A single ternary plane carries at most 1.58 bits of information per weight, so post-training projection of a full-precision matrix onto one ternary plane loses too much: reasoning collapses." **Proves** monotonic residual decrease with expanded rank μ > 1. At μ=3 (3× full rank), Qwen3.5-4B reaches 10.10 perplexity vs 9.78 bf16 (+3.2%).
- **PTQTP (2025)**: Minimum 2 trit-planes. "Two planes is still a fixed capacity budget with a ceiling it cannot get past." At K=4 planes (∼6.8 bpw), approaches ExTernD performance.
- **Ternary Residual Networks (2018)**: FGQ with sub-tensors + residual ternarization. Different sub-tensors need different numbers of residuals (r_i).

**Phase 0 Flaw**: Algorithm decomposes **each output neuron independently** into K ternary vectors. This is equivalent to K "planes" but:
- No cross-neuron coordination (unlike ExTernD's B ∈ ℝ^{m×k}, C ∈ ℝ^{k×n})
- No low-rank structure exploitation
- Capacity scales as K × out_features × in_features (huge) vs ExTernD's μ × (m+n) × min(m,n)

### 2. LLM Weight Distributions are Unimodal Gaussian — Mismatched to {-1,0,1}

**Evidence:**
- **TWLA (2024)**: "Empirically, per-channel weight distributions are often close to a unimodal Gaussian. This shape mismatches the ternary codebook {-1,0,+1}, which leads to large quantization error when weights are projected into the ternary space. In contrast, a tri-modal distribution is more aligned with ternary representation."
- **PT2-LLM (2025)**: "Weight distributions in LLMs are not always symmetric, as many layers exhibit non-zero means." Introduces **asymmetric ternary quantization** with row-wise offset μ.
- **CAT-Q (2026)**: Learnable modulation (LM) transforms weight distribution via δ_μ, δ_α, δ_Δ before ternarization.
- **Tequila (2024)**: "Deadzone trapping" — aggressive ternarization creates vast deadzone (-Δ, Δ) where weights quantize to zero. These "dead" weights receive uninformative gradients.

**Phase 0 Flaw**: `t_k = sign(r) * (|r| > τ)` with τ = 0.5 * mean(|r|) assumes symmetric zero-centered distribution. On unimodal Gaussian weights, this quantizes ~68% (within 1σ) to zero → massive information loss. No row-wise offset μ, no asymmetric α⁺/α⁻, no distribution transformation.

### 3. Calibration Must Be Activation-Aware, Not Weight-Space Only

**Evidence:**
- **PT2-LLM**: "While Iterative Ternary Fitting effectively minimizes the weight quantization error E_w, the actual output of LLMs depends on the interaction between weights and activations." Introduces **Activation-aware Grid Alignment (AGA)** minimizing E_x = ||XW^T - XŴ^T||.
- **TWLA**: E2M-ATQ minimizes layer-output error via "Euclidean-to-Manifold" two-stage optimization.
- **ExTernD**: Importance-weighted variant using **llama.cpp imatrix statistics** (Hessian diagonal) — "non-obvious correction to the V-step".
- **GPTQ / AWQ / PTQTP**: All use Hessian/block-wise error compensation.

**Phase 0 Flaw**: Step 3 decomposes W_fp in weight space only (||w_fp - Σα_k t_k||). Step 4 simulates on calibration X but doesn't feed back into decomposition. No output-error minimization, no importance weighting.

### 4. Outlier Handling is Non-Negotiable for LLMs

**Evidence:**
- **PT2-LLM**: **Structural Similarity-based Reordering (SSR)** — clusters columns by cosine similarity before ternarization. "Grouping outliers together prevents them from distorting normal columns—outliers among outliers cease to be outliers."
- **TWLA**: **KOTMS** — Kronecker-structured orthogonal rotation reshapes weights to tri-modal distribution while suppressing activation outliers via shared rotation.
- **PTQTP / GPTQ / AWQ**: Block-wise processing (G=128 columns) with error compensation.
- **CAT-Q**: Learnable modulation suppresses outlier effects via δ_α, δ_μ.

**Phase 0 Flaw**: No column grouping, no reordering, no rotation, no block-wise processing. Processes entire layer as one matrix → outliers dominate residual.

### 5. Scale Factor Strategy is Oversimplified

**Evidence:**
- **TTQ (2016)**: Two scaling factors per layer (W⁺_l, W⁻_l) for positive/negative weights. "Outperforms full-precision AlexNet by 0.3% Top-1 on ImageNet."
- **PT2-LLM**: Row-wise asymmetric grid {−α_i + μ_i, μ_i, α_i + μ_i} — **three parameters per row**.
- **PTQTP**: Row-wise adaptive ridge regression for 2-plane scaling: α_i = A⁻¹_i b_i ∈ ℝ² per row.
- **CAT-Q**: Disentangled learning of δ_μ, δ_α, δ_Δ per layer.
- **TNT**: Two scalars (λ_p, λ_n) for positive/negative components of ternary vector.

**Phase 0 Flaw**: Single scalar α_k per ternary vector (entire output neuron). No asymmetry, no row-wise adaptation, no disentangled learning.

### 6. Threshold Selection is Ad-Hoc vs. Learned/Optimized

**Evidence:**
- **TWN (2016)**: Δ* ≈ 0.75 E(|W|) for normal, Δ* = α/3 for uniform.
- **TTQ (2016)**: Δ = 0.05 × max(|W|).
- **TNT (2019)**: Optimal threshold via cosine similarity maximization over N candidates (O(N log N)).
- **PT2-LLM**: Learns threshold via flexible rounding during ITF.
- **CAT-Q**: Learns δ_Δ via softened ternarization (differentiable transition).
- **Tequila**: Reactivates deadzone weights as dynamic biases (0₋, 0₊).

**Phase 0 Flaw**: τ = 0.5 * mean(|r|) is a heuristic with no theoretical grounding for LLMs. No sweep over learned/optimized thresholds in Step 5.

### 7. Convergence Guarantees Exist — Phase 0 Has None

**Evidence:**
- **ExTernD**: Proves ||R_{i+1}||_F² ≤ ||R_i||_F² (monotonic) and ||R_k||_F² ≤ (1 - 1/mn)^k ||A||_F² → 0. **Arbitrary accuracy achievable**.
- **TSVD (2023)**: Proves linear convergence rate √(1 - cos²(2θ)/min(M,N)) for θ > π/4.
- **Ternary Residual Networks**: Proves adding ternary residuals strictly reduces ℓ₂ error at every step.

**Phase 0 Flaw**: No convergence proof. Greedy sign-threshold on residual has no guarantee of monotonic decrease in output error.

---

## Specific Step-by-Step Critique

### Step 1: GTD Core Implementation
| Issue | Current | Required |
|-------|---------|----------|
| Algorithm | Basic greedy sign-threshold | ExTernD ALS (alternating ternary fit + optimal scale + deflate) OR PTQTP progressive group-wise |
| Asymmetry | Symmetric | Asymmetric ternary {−α+μ, μ, α+μ} per row (PT2-LLM) |
| Threshold | Fixed heuristic τ = 0.5×mean(|r|) | Learned via ITF (PT2-LLM) or optimized via cosine similarity (TNT) |
| Structure | Per-row independent | Factorized B diag(D) C (ExTernD) or trit-planes (PTQTP) |

### Step 2: Vectorized Batch Implementation
| Issue | Current | Required |
|-------|---------|----------|
| Chunking | Arbitrary 512-row chunks | Block-wise G=128 columns (GPTQ/PTQTP/PT2-LLM) |
| Decorrelation | None | Block decorrelation: [(U^TU)⊙(V^TV)]d = diag(U^TRV) (ExTernD Eq. 4) |
| Parallelism | Row-wise | Column-block-wise with within-block deflation |

### Step 3: Synthetic Validation
| Issue | Current | Required |
|-------|---------|----------|
| Distribution | X ~ N(0,I) | Real activation distributions (heavy-tailed, outliers) |
| Metric | Weight-space cosine | **Output-space error** E_x = ||XW^T - XŴ^T|| (PT2-LLM, TWLA) |

### Step 4: Model Loading & Calibration
| Issue | Current | Required |
|-------|---------|----------|
| Data | wikitext-2, 50 samples | Sufficient for imatrix (llama.cpp: 128-512 samples) |
| Hooks | Single layer input | **Full block inputs + outputs** for error compensation |
| Statistics | None | **Hessian diagonal / imatrix** for importance weighting (ExTernD, GPTQ) |

### Step 5: Parameter Sweep
| Missing Sweep Dimensions | Required by SOTA |
|--------------------------|------------------|
| Asymmetric vs symmetric ternary | PT2-LLM, TTQ, CAT-Q |
| Row-wise offset μ | PT2-LLM, CAT-Q |
| Column reordering (SSR) | PT2-LLM |
| Block size G | GPTQ, PTQTP, PT2-LLM (G=128) |
| Rotation (KOTMS) | TWLA |
| Importance weighting | ExTernD, GPTQ |
| Sparsity threshold τ | ExTernD (τ=0.7→41%, τ=1.0→57%, τ=2.0→87%) |

### Step 6: Analysis & Gate Decision
| Current Metric | SOTA Metric |
|----------------|-------------|
| Weight cosine similarity | **Perplexity on WikiText-2 / C4**, zero-shot accuracy (BoolQ, ARC-Challenge) |
| Layer MSE | **Output error E_x**, energy preservation (ExTernD) |
| Sparsity % | **Effective bits-per-weight** (Shannon entropy accounting for zeros) |

---

## Revised Phase 0 — Minimum Viable to Avoid False Negative

### Must Add (Non-Negotiable)
1. **Asymmetric ternary quantization** per row: {−α_i + μ_i, μ_i, α_i + μ_i}
2. **Block-wise processing** (G=128 columns) with group-wise error compensation
3. **Activation-aware alignment**: minimize ||XW^T - XŴ^T||_F not ||W - Ŵ||_F
4. **Column reordering** by structural similarity (cosine clustering) before ternarization
5. **Importance weighting** via calibration Hessian diagonal (imatrix)
6. **Factorized decomposition** (ExTernD-style B diag(D) C) not per-row ensemble

### Should Add (High Impact)
7. **Two-stage optimization**: Euclidean initialization → manifold relocation (E2M-ATQ)
8. **Softened ternarization** for stable convergence (CAT-Q ST)
9. **Sparsity threshold τ** as explicit dial (ExTernD)
10. **Per-layer adaptive μ** (inner rank multiplier)

### Can Defer
- Full orthogonal rotation (KOTMS) — adds complexity, marginal gain for weight-only
- End-to-end perplexity — use layer-wise output error as proxy

---

## Recommended Phase 0 Algorithm (ExTernD-Lite)

```python
def extrend_lite_decompose(W_fp, X_calib, mu=2.0, tau=1.0, G=128):
    """
    W_fp: [out_features, in_features]  (e.g., 4096, 11008)
    X_calib: [num_tokens, in_features]
    Returns: B [out, k], D [k], C [k, in] with B,C ternary, D real
    """
    m, n = W_fp.shape
    k = int(mu * min(m, n))
    
    # 1. Column reordering by structural similarity (SSR)
    P = column_reorder_by_cosine(W_fp)  # permutation
    W = W_fp[:, P]
    X = X_calib[:, P]
    
    # 2. Block-wise processing
    num_blocks = (n + G - 1) // G
    B_blocks, C_blocks, D_blocks = [], [], []
    
    for b in range(num_blocks):
        cols = slice(b*G, min((b+1)*G, n))
        W_block = W[:, cols]      # [m, G]
        X_block = X[:, cols]      # [num_tokens, G]
        
        # 3. ExTernD greedy ALS on this block
        R = W_block.clone()
        B_block, C_block, D_block = [], [], []
        
        for i in range(k):
            # Alternating ternary fit (15 iters)
            u = ternary_fit(R.T @ v, tau)  # [m]
            v = ternary_fit(R @ u, tau)    # [G]
            
            # Optimal scale (least squares)
            d = (u @ R @ v) / (u.norm()**2 * v.norm()**2)
            
            # Deflate
            R = R - d * u.outer(v)
            
            B_block.append(u)
            C_block.append(v)
            D_block.append(d)
        
        B_blocks.append(torch.stack(B_block, dim=1))  # [m, k]
        C_blocks.append(torch.stack(C_block, dim=0))  # [k, G]
        D_blocks.append(torch.tensor(D_block))        # [k]
    
    # 4. Concatenate blocks
    B = torch.cat(B_blocks, dim=1)  # [m, k*num_blocks] — WRONG, need to align
    # Actually: B is shared across blocks! C is per-block.
    # ExTernD: B [m, k], C [k, n] with block structure in C
    
    return B, D, C
```

**Key insight from ExTernD**: B ∈ {-1,0,1}^{m×k} is **global** (shared across all column blocks), C ∈ {-1,0,1}^{k×n} is **block-structured**. This reduces parameters from K×m×n to k×(m+n).

---

## Updated Go/No-Go Criteria (Aligned with SOTA)

```python
# After running ExTernD-Lite on gate_proj (layer 12) with mu=2,3,4:
GREEN = (
    perplexity_delta_pct < 5%      # WikiText-2 perplexity vs FP16
    and effective_bpw < 6.0        # Shannon entropy bits/weight
    and sparsity > 0.4             # >40% zeros in B,C
    and max_layer_output_error < 0.05  # ||XW - XŴ||_F / ||XW||_F
)

YELLOW = (
    perplexity_delta_pct < 10%
    and effective_bpw < 8.0
)

RED = otherwise  # KILL PROJECT
```

**Benchmark targets from literature**:
- ExTernD μ=3 on Qwen3.5-4B: 10.10 vs 9.78 perplexity (+3.2%)
- PTQTP K=4 on Qwen3: 82.4% math reasoning retention
- PT2-LLM: outperforms SOTA 2-bit PTQ at same memory
- TWLA W1.58A4: maintains high accuracy

---

## Conclusion

**Phase 0 as written tests a strawman algorithm** (per-row symmetric sign-threshold ensemble) that literature shows **cannot work on LLMs**. The 8% accuracy drop of TNT on VGG-16 ImageNet (CNN!) with single-plane ternary should be a warning: transformers are more sensitive.

**To get a meaningful GREEN/YELLOW/RED signal**, Phase 0 must implement at minimum:
1. Asymmetric ternary per row (α⁺, α⁻, μ)
2. Block-wise (G=128) with error compensation
3. Activation-aware output error minimization
4. Factorized decomposition (B diag(D) C) with μ ≥ 2

**Estimated effort to fix**: 3-5 additional steps in Phase 0. Without these, a RED result is a **false negative** — it kills a viable approach by testing an inadequate algorithm.

---

## References (Key Papers for Implementation)

| Method | Year | Key Innovation | Code |
|--------|------|----------------|------|
| **ExTernD** | 2026 | Expanded-rank B diag(D) C, provable convergence | [GitHub](https://github.com/ExTernD) |
| **PTQTP** | 2025 | 2-4 trit-planes, progressive group-wise | [GitHub](https://github.com/HeXiao-55/PTQTP) |
| **PT2-LLM** | 2025 | ATQ + ITF + AGA + SSR | [GitHub](https://github.com/XIANGLONGYAN/PT2-LLM) |
| **TWLA** | 2024 | KOTMS rotation + E2M-ATQ + ILA-AMP | [GitHub](https://github.com/Kishon-zzx/TWLA) |
| **CAT-Q** | 2026 | Learnable modulation + softened ternarization | — |
| **Tequila** | 2024 | Deadzone trapping fix via dynamic biases | — |
| **TNT** | 2019 | Cosine similarity O(N log N) optimal ternary | — |
| **TSVD** | 2023 | Ternary SVD with greedy algorithm | — |