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

---

# Phase 1 Stage A — structural redundancy and post-hoc removal (2026-09-25)

**Status: VOID — no GO/YELLOW/RED was issued.** Spec `phase1/compression-gate` R4 requires
the 8-bit group-128 sanity baseline to stay within +0.1 % perplexity; it measured **+0.167 %**
(clip-searched) and **+0.153 %** (pure absmax, excluding the clip search as the cause). The
harness is therefore invalid for this stage, the run is recorded as void, and the numbers
below are **diagnostics, not a Stage A verdict**. Verdict record: `results/p1_verdict/`.

## Target stack

`Qwen/Qwen3-8B` BF16 (36 layers, 32 heads, 8 KV heads, hidden 4096, intermediate 12288,
vocab 151936), WikiText-2 test, fixed slice 128 × 512 = 65 536 tokens, AMD RX 7900 XTX /
ROCm. Reference perplexity **12.6398** at 16 381 470 720 stored bytes; peak VRAM 15 624 MB.
Harness: `lm-evaluation-harness` 0.4.13 (installed against the pinned ROCm stack).

## Integer baselines (group 128, RTN with per-group MSE-optimal clipping)

| Arm | Stored bytes | Ratio | Perplexity | Relative delta |
|---|---|---|---|---|
| 4-bit | 6 071 347 200 | 2.698× | 13.8663 | +9.70 % |
| 5-bit | 6 939 568 128 | 2.361× | 12.9520 | +2.47 % |
| 6-bit | 7 807 789 056 | 2.098× | 12.7586 | +0.94 % |
| 8-bit (sanity) | 9 544 230 912 | 1.716× | 12.6609 | +0.167 % |
| 8-bit absmax | 9 544 230 912 | 1.716× | 12.6592 | +0.153 % |
| 8-bit library (`torchao` 0.13.0 `int8_weight_only`, g128) | 9 544 230 912 | 1.716× | 12.6878 | +0.38 % |

No arm meets the pre-registered ±0.5 % quality bar; the closest is 6-bit at +0.94 %.

**Resolution of the void.** The strong-baseline check settles it: a mainstream library
implementation (`torchao` 0.13.0, `int8_weight_only`, group 128) reaches only **+0.38 %** —
*worse* than the hand-rolled RTN arm (+0.167 %). The 8-bit cost is a property of this model on
this slice, not an artifact of our quantizer, and the pre-registered 0.1 % sanity tolerance was
simply below what any 8-bit weight-only scheme achieves on Qwen3-8B. The library 4-bit arm
cannot run on this GPU: `_convert_weight_to_int4pack_cuda is only supported on AMD gpu arch
greater than or equal to CDNA2` (the RX 7900 XTX is gfx1100/RDNA3), so the library 4-bit
comparison is unavailable on this hardware. Recalibrating the sanity tolerance from this
measurement is a threshold change and needs its own decision (rule R-005, updated 2026-09-25).

## Redundancy map (`results/p1_map_2026-09-25e/`)

- **Blocks: none removable.** The cheapest single-layer removal costs KL 0.1105 nats against
  the pre-registered budget 0.10.
- **Heads: none removable.** Removing the 17 least important heads (5 % of 336) costs
  KL 2.8884; attention heads are load-bearing on this model.
- **FFN intermediate dimensions: the only headroom.** 5 837 of 116 736 (5 %) cost KL 0.0903
  (inside budget); 11 674 (10 %) cost 0.2074 (outside).
- **Least-significant residual directions** (removed by weight orthogonalization): band 8–16
  gives KL 0.0203 / 0.0393 / 0.0806 / 0.1666 for 8 / 16 / 32 / 64 directions; band 16–23 gives
  0.0267 / 0.0557 / 0.1132 / 0.2210. About 32 directions per band are affordable.
- **Ternary representation (TCD on layer 12 `gate_proj`):** natural zero fraction 0.4227 /
  0.4252 / 0.4285 at μ = 2 / 3 / 4 with `E_x` 0.143 / 0.0713 / 0.0359 — our reconstruction
  lands exactly on the ~42 % zero valley Sparse-BitNet reports for trained ternary weights.
  **That valley does not extend post-hoc:** dropping 20 % of factorization planes raises
  `E_x` to 0.0623 (baseline 0.0359), and entry-level N:M masking over the ternary factors is
  catastrophic — 2:4 → 0.5617, 4:8 → 0.5263, 6:8 → 0.2235. The "42 % zeros can be pushed to
  N:M" premise fails without retraining.

## Combined probe and the end-to-end cliff

The frozen plan from the map (order fixed before the run) is FFN-only: blocks and heads
contribute zero removable units, and directions remove no stored bytes without a materialised
low-rank factor, so they are excluded from the byte claim.

| FFN dims removed | Equivalent | Stored bytes | Ratio | Perplexity | Delta |
|---|---|---|---|---|---|
| 5 % (22 119) | probe | 15 837 874 176 | 1.0343× | 12.5867 | **−0.42 %** |
| 6 % (26 542) | sweep | — | 1.0415× | 12.5942 | −0.36 % |
| 8 % (35 389) | sweep | — | 1.0561× | 13.3026 | +5.24 % |
| 10 % (44 237) | sweep | — | 1.0711× | 21.2005 | +67.73 % |
| 15 % (66 355) | sweep | — | 1.1106× | 1 026.1407 | +8 018 % |
| 25 % (110 592) | sweep | — | 1.1989× | 11 361.4296 | +89 786 % |

The removal is **free in perplexity terms up to ~6 %** of FFN dimensions, then falls off a
cliff between 6 % and 8 % and is destructive beyond 10 %. The cartography KL proxy was
conservative at 5 % (it predicted damage where end-to-end showed none) and optimistic at
10 %. Best achievable structural compression inside the quality bar: **≈1.04×**.

## What this means

1. **No configuration satisfies both gate halves.** The 2× floor is met only by the 4/5/6-bit
   integer arms, all of which fail the quality bar; structural removal reaches 1.03–1.07×.
   Post-hoc structural redundancy on Qwen3-8B is worth ~4 % of parameters, not 50 %.
2. **The ±0.5 % bar and the 0.1 % sanity bar are both stricter than a plain group-128 integer
   arm achieves on this model** (+0.94 % at 6-bit; +0.153 % at 8-bit absmax). Both are
   pre-registered thresholds; relaxing either needs an explicit, recorded change.
3. **The ternary post-hoc repair premise is dead** in this pipeline, independently of Phase 0.
4. A stronger integer baseline (library GPTQ/AWQ) would move the baseline, not the structural
   result: the FFN-dimension cliff is measured directly end to end.

## Limitations

Perplexity-only (the benchmark half of the quality bar was not measured, so the matched-quality
integer baseline of R3 is not established either); hand-rolled RTN+clip quantizer; one model
and one slice; no healing; N:M masking implemented at entry level with a magnitude ranking.

## Open questions for the user

**Direction closed on 2026-09-25 by the user's decision.** Post-hoc structural removal is not
pursued further. The recalibration of the sanity tolerance (a threshold change) and the Stage B
candidates below remain open.

1. **Threshold calibration** — the strong-baseline check shows a mainstream 8-bit library arm
   costs +0.38 %, so the 0.1 % sanity tolerance and the ±0.5 % quality bar both sit below the
   8-bit weight-only frontier of this model; both need a recorded change before any Stage A
   verdict could exist.
2. **Stage B candidates** (not started, each needs its own change): MoE expert removal
   (REAP / Fisher-MoE report near-lossless 50 % at large scale, and it is the only track that
   speaks to the 24 GB / 20 tok/s goal), the ternary codec, and scale testing (does redundancy
   grow with model size — only an 8B model was measured here).
3. **Benchmarks** — the four-task harness was not run.

