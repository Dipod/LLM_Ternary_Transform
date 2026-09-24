# Phase 0 decision report — STOP (2026-09-24)

## Decision

**STOP** the current framing: "convert open-weight LLMs to ternary weights by post-training
algorithmic reconstruction, as a compression method competing with 4-bit quantization".
The premise is **not confirmed** on the target stack.

## What was built and measured

`phase0_feasibility/` implements a gradient-free factorized ternary decomposition
`W ≈ B diag(D) C` (`B, C ∈ {−1,0,1}`, `D` real, `k = round(μ·min(out,in))`), calibration
capture, the pre-registered `E_x`/sparsity/BPW gate, and a recorded sweep harness.
All measurements ran on `Qwen/Qwen2.5-0.5B` `gate_proj` (layers 12 and 21), WikiText-2
calibration, AMD Radeon RX 7900 XTX via ROCm-on-WSL.

Gate result: **YELLOW**. Best `E_x = 0.045–0.048` (fit and held-out) at sparsity ≈ 0.70 and
`bpw ≈ 6.16`; GREEN needs `E_x ≤ 0.05` AND `bpw < 6.0`, so accuracy passed and the bit
budget missed by ~2.6%. Cheap tuning, importance weighting, AGA scale re-solve (~3.5% gain,
transfers to held-out) and component pruning could not close it.

## The decisive comparison (pre-registered)

The gate compared against FP16 (16 bits), not against the real competitor. Rule fixed
before the run: compare at **matched effective bits**; STOP if the integer frontier lies
below the ternary frontier. Layer-local `E_x`, same model/slice/activations, 8-bit sanity
`E_x = 0.0045` confirming the pipeline.

| Method | bpw | `E_x` (layer 12 / 21) |
|---|---|---|
| integer 4-bit g128 | 4.125 | 0.083 / 0.070 |
| integer 5-bit g128 | 5.125 | 0.039 / 0.035 |
| integer 6-bit g128 | 6.125 | 0.019 / 0.018 |
| ternary μ=4, τ=1.3 | 6.160 | 0.048 / 0.046 |

At ~equal bit budget, ordinary integer group-128 quantization is **~2.5× better** on the
project's own metric; integer 5-bit with 17% fewer bits also wins. Ternary points at
bpw 4.4–4.6 (0.088–0.115) lose to integer 4-bit at 4.125 (0.070–0.085).

Evidence: `results/bit_frontier/` (prototypes + report), `results/recheck/`,
`results/tau_probe/`, `results/mu_probe/`, `results/explore_mu3/`.

## Why (independent literature check)

Three read-only surveys (accuracy landscape, hardware reality, premise review) converge:
- Ternary post-training loses to 4-bit on accuracy (5–11 points on task averages) and on
  bits-per-quality (ExTernD needs ~5.2–5.7 bpw to match `Q4_K` at ~4.5–4.9);
- batch-1 decode is memory-bandwidth-bound, so removing the multiply does not touch the
  limiting resource; datacenter GPUs have no native ternary tensor core; ternary zeros are
  unstructured and not skipped by production kernels;
- the factorized form costs `μ(m+n)/max ≈ 3.5–4.7×` more operations than a dense layer, and
  on GPUs an add costs about the same as a multiply.

## Where ternary remains defensible (not the current stand)

1. CPU / edge batch-1 with real ternary kernels (`bitnet.cpp` 2.4–6.2× vs FP16, large energy
   savings; T-MAC lookup kernels).
2. Multiplier-free FPGA/ASIC datapaths and extreme-memory (MoE in system RAM).
3. **Natively trained** ternary models (BitNet-style), which reach roughly fp16 parity at
   small size — outside a "no retraining" premise.

Any restart should pick one of these, name its hardware, and pre-register a matched-bits
comparison against the strongest integer baseline for that hardware.

## Limitations

Layer-local `E_x`, one 0.5B model, one calibration slice; the integer arm is a hand-rolled
RTN/GPTQ with clipping (8-bit sanity passed; a library AWQ/GPTQ would only be stronger);
end-to-end perplexity was not measured. All limitations favour the STOP verdict.

## Follow-ups if the project continues

- Optionally confirm with a library 4-bit baseline (AWQ/GPTQ via `llm-compressor`, or
  `Q4_K_M` in llama.cpp) plus end-to-end perplexity.
- Pivot options: 2-bit integer PTQ (hardware-friendly, ParetoQ path), or the CPU/edge
  ternary-kernel niche.
