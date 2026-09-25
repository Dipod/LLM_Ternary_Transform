# Promising neural-network architectures — literature survey

Date: 2026-09-25. Read-only reconnaissance by a delegated literature agent
(`py-ml-researcher`); it wrote nothing and ran no benchmarks. Question: **which architectures
today show outstanding, measured results** — and which of them can help this project.

Filter that makes the survey actionable (from this repository's own measurements): the target is
**24 GB VRAM, >= 20 tok/s single-stream, wide context, on an AMD RX 7900 XTX (RDNA3/gfx1100)
via ROCm-on-WSL**. Batch-1 decode is **memory-bandwidth-bound** (VRAM 744 GB/s measured;
`memory.md`). Therefore an architecture helps only if it **reduces bytes moved per decode step**
(weight bytes, KV/state bytes, or both); compute-only or prefill-only wins do not transfer.

Evidence labels: `proven-and-reproduced` / `single-paper-claim` / `theory-only` /
`hype-no-result`. Items not confirmable from an opened source are marked `unverified`.
Companion survey on cross-architecture transfer: `results/cross_architecture_transfer_survey.md`.

## 1. Top architectures by evidence strength

| Architecture | Single strongest measured result (with its setting) | What is new | Label | Citation |
|---|---|---|---|---|
| Mamba-2 | 2.7B params / 300B tokens on the Pile beats Mamba-2.8B, Pythia-2.8B and Pythia-6.9B; SSD core 2–8× faster than Mamba's selective scan | SSM↔attention duality via semiseparable matrices; larger state; multi-value heads | proven-and-reproduced (adopted by NVIDIA Megatron hybrid, Zamba2, Codestral-Mamba) | arXiv:2405.21060v1, ICML 2024 |
| Gated DeltaNet | Surpasses Mamba-2 and DeltaNet on language modelling, commonsense, in-context retrieval, length extrapolation; hybrids better still (no single headline number) | Forgetting gate + delta rule; chunkwise parallel form | single-paper-claim (adopted in Qwen3-Next) | arXiv:2412.06464, ICLR 2025 |
| GLA | 340M/15B and 1.3B/100B tokens: on par with LLaMA-architecture Transformer and Mamba; trained at 2K, generalises to 18–20K | Data-dependent gating + block-parallel tensor-core training | single-paper-claim | arXiv:2312.06635v3 |
| RWKV-7 "Goose" | 2.9B on a 3.1T-token corpus: new 3B multilingual SoTA, matches 3B English SoTA; constant memory/time per token; recognises all regular languages (proof) | Vector-valued gating and in-context learning rate | single-paper-claim | arXiv:2503.14456v2 |
| Jamba | 12B active / 52B total, 256K context, fits one 80 GB GPU at int8; 8× smaller KV cache | Transformer + Mamba + MoE interleaved at scale | single-paper-claim (weights released) | arXiv:2403.19887v1 |
| Zamba / Zamba2 | Zamba2 1.2/2.7/7.4B up to 3T tokens: 30–50 % TTFT reduction, 6× KV-cache reduction | Mamba backbone + shared global attention block(s) | single-paper-claim (Apache-2.0 weights) | arXiv:2405.16712v1; arXiv:2411.15242v1 |
| Hymba | 1.5B / 1.5T tokens: +1.32 % average accuracy vs Llama-3.2-3B, 11.67× cache reduction, 3.49× throughput | Attention heads and SSM heads fused in the same layer; meta tokens; cross-layer KV sharing | single-paper-claim | arXiv:2411.13676v1, ICLR 2025 |
| Mamba-3 | 1.5B: +0.6 pts over Gated DeltaNet; MIMO +1.2 more (+1.8 total); Mamba-2 perplexity at half the state size | Exponential-trapezoidal discretisation; complex state; MIMO with no decode-latency increase | single-paper-claim | arXiv:2603.15569v1, ICLR 2026 |
| MLA | DeepSeek-V2 (236B/21B active, 8.1T tokens): KV cache −93.3 %, generation throughput 5.76× vs DeepSeek-67B, better quality | Low-rank joint KV compression; decoupled RoPE key; absorbtion into Q/O | proven-and-reproduced (shipped in V3/V3.2 and open serving stacks) | arXiv:2405.04434v2 |
| NSA | 27B / 3B active, 260–270B tokens: LongBench 0.469 (+0.032 over full attention); 9.0× forward / 6.0× backward, up to 11.6× decoding at 64K, A100 + Triton | Natively trainable hierarchical sparse attention | single-paper-claim | arXiv:2502.11089v2, ACL 2025 |
| MoBA | Llama-8B-1M, block 4096, top-12 (62.5 % sparsity at 128K): RULER 0.7818 vs 0.7849 full | MoE-style top-k block routing for attention | proven-and-reproduced (author-claimed Kimi production use) | arXiv:2502.13189v1, NeurIPS 2025 |
| DSA | Built on 685B V3.1-Terminus: on par with it on public benchmarks while cutting long-context cost; indexer keeps top-2048 tokens/query | Lightning indexer + fine-grained token selection inside MLA | single-paper-claim (official report) | DeepSeek announcement 2025-09-29; HF `deepseek_v32` |
| DeepSeekMoE | 16B / 2T tokens ≈ LLaMA2-7B at ~40 % of the compute; disabling the shared expert raises Pile loss 1.808→2.414 | Fine-grained expert segmentation + always-on shared expert | proven-and-reproduced (basis of V2/V3) | arXiv:2401.06066, ACL 2024 |
| DeepSeek-V3 | 671B / 37B active, 14.8T tokens, 2.788M H800 GPU-hours, quality comparable to leading closed models | Auxiliary-loss-free load balancing; multi-token prediction | proven-and-reproduced (open weights) | arXiv:2412.19437v1/v2 |
| Qwen3-Next | 80B total / 3B active: on par with Qwen3-235B-A22B and beats it on RULER within 256K | 3:1 Gated DeltaNet : gated attention hybrid; highly sparse MoE; MTP | single-paper-claim (open weights) | Qwen official blog 2025-10-14; HF `modeling_qwen3_next.py` |
| Qwen3.8-Flash-Next | 125B total / 6B active (+51B host n-gram tables): leads the 397B-A17B predecessor on 8/14 pretraining benchmarks at ~1/3 activated params, ~1/3 tokens, ~1/9 FLOPs | Qwen Sparse Attention micro-block indexer; gated four-branch residual | single-paper-claim | arXiv:2608.30320 (2026) |
| MiniMax-01 | 456B / 45.9B active, 1M-token training and 4M-token inference; matches GPT-4o/Claude-3.5-Sonnet with 20–32× longer context; 75 % MFU on H20 | First large-scale deployment of linear (lightning) attention | single-paper-claim | arXiv:2501.08313v1 |
| LoLCATs | Linearises Llama-3-8B / Mistral-7B: +20 pts 5-shot MMLU over prior linearising methods with 0.2 % of params and 0.04–0.2 % of tokens; first linearised 70B/405B, closes 77.8 % / 78.1 % of the gap | Attention-transfer MSE to a linear analogue, then LoRA only | single-paper-claim | arXiv:2410.10254v3, ICLR 2025 |
| HRM | 27M params, ~1000 training samples: ARC-AGI 40.3 % vs o3-mini-high 34.5 %, Claude 3.7 21.2 %; near-perfect Sudoku-Extreme | Two coupled recurrent modules at different timescales; one-step gradient | single-paper-claim (synthetic reasoning, not a chat LLM) | arXiv:2506.21734v3 |
| Mixture-of-Depths | isoFLOP +1.5 % log-prob, or loss parity with up to 50 % fewer FLOPs per forward | Per-token depth routing with a static compute graph | single-paper-claim | arXiv:2404.02258v1 |
| Ouro (looped LM) | 1.4B and 2.6B on 7.7T tokens match up to 12B models (2–3× parameter efficiency) | Pre-trained looped weight-tied depth + learned exit | single-paper-claim | arXiv:2510.25741v4 |
| TTT | 125M–1.3B: matches/exceeds Transformer and Mamba, keeps improving past 16K where Mamba stalls | Hidden state is a model, trained by self-supervision at inference | single-paper-claim | arXiv:2407.04620v1 |
| Titans | Outperforms Transformers at the same context window and scales beyond 2M on needle-in-haystack | Neural long-term memory learned at test time | single-paper-claim | arXiv:2501.00663v1 |
| LLaDA | 8B, 2.3T tokens, 0.13M H800 hours: comparable to LLaMA3-8B in-context learning over 15 tasks; beats GPT-4o on reversal-poem completion | Masked-diffusion LM trained from scratch | single-paper-claim | arXiv:2502.09992v3 |
| Mercury | Coder Mini 1109 tok/s, Small 737 tok/s on H100 (third-party evaluation); Mercury 2 1009 tok/s on Blackwell, 128K | Parallel token refinement instead of sequential decode | single-paper-claim (commercial API) | arXiv:2506.17298v1; Inception Labs blog |
| BLT | First FLOP-controlled byte-level study up to 8B params / 4T bytes: matches tokenisation at scale | Dynamic entropy-based byte patching, no fixed vocabulary | single-paper-claim | arXiv:2412.09871v1, ACL 2025 |
| MatMul-free LM | Up to 2.7B trained on 100B tokens: comparable to Transformer++ (better on ARC-C/OpenbookQA); batch-1 seq-2048 advantage with BitBLAS | Ternary dense layers + MLGRU token mixer | single-paper-claim (**>2.7B GPU numbers are simulated with random weights**) | arXiv:2406.02528v3 |

## 2. Efficiency reality check for batch-1 single-stream decode on 24 GB

| Architecture | Reduces memory or only compute? | Batch-1 decode relevant? | Why |
|---|---|---|---|
| MLA | Memory: KV bytes/token | Yes | KV is read every decode step; −93.3 % is a direct bytes cut. Shipped. |
| Gated DeltaNet / Mamba-2 / RWKV-7 hybrid | Memory: KV → O(1) state at long context | Yes at long context | Removes KV growth; weight bytes unchanged; a ~25 % attention hybrid keeps a KV on those layers. |
| Sparse attention (NSA / DSA) | Memory: KV blocks read per step | Yes (NSA reports 11.6× decode at 64K) | Cuts bytes read per decode step at long context; no gain at short context. |
| MoBA | Compute, at prefill only | No | Authors run full attention during generation. |
| Jamba / Zamba2 / Hymba | Memory: KV | Yes at long context | All trained from scratch; Jamba 52B total does not fit 24 GB usefully. |
| DeepSeek-V3 / Qwen3-Next MoE | Memory: active weight bytes/token | Yes in principle, not at this size | 671B→~335 GB at 4-bit; 80B-A3B→~40 GB at 4-bit; neither fits 24 GB. Only a ≤9–10B MoE would. |
| LoLCATs / linearised LLM | Memory: KV | Yes after conversion | Costs training; prior linearisations lose up to 42.4 pts MMLU before recovery. |
| MoD (adaptive depth) | Compute only | No | Every layer's weights stay resident and are still read. |
| MatMul-free LM | Memory (ternary weights) + compute | Partly | Real batch-1 GPU measurements only to 2.7B; larger figures are simulated. |
| TTT / Titans | State memory, but a per-token training step | Risk | Extra compute per token; TTT-MLP memory I/O acknowledged unsolved. |
| Diffusion LM (LLaDA / Mercury) | Compute / throughput | No for single stream | 1000+ tok/s is a server parallel-decode metric. |
| BLT | Compute | Neutral/negative | Shortens the sequence; does nothing for weight/state bytes at decode. |
| Ouro / looped LM | Parameters (2–3×) | Possible fit win, speed risk | Fewer weight bytes helps fit 24 GB; looped depth adds compute per token. |
| HRM | N/A (27M) | No | Synthetic reasoning, not a language model. |

Training-from-scratch vs reusable: trained from scratch — Jamba, Zamba/Zamba2, Hymba, Mamba-3,
Qwen3-Next, Qwen3.8-Next, MiniMax-01, RWKV-7, DeepSeek-V3, LLaDA, Mercury, BLT, MatMul-free,
Ouro, TTT, Titans, HRM, MoD. Conversion paths with published budgets — Mamba-2 / Gated DeltaNet
targets (MOHAWK, Llamba, RADLADS, ARWKV), linearisation (LoLCATs), MoBA (continued pretraining
of Llama-8B-1M for 100B tokens), DSA (continued training of V3.1-Terminus). **Zero-training
efficiency wins exist only for inference-time techniques on an existing checkpoint** —
weight-only quantisation and KV-cache quantisation/eviction; nothing architectural is free.

## 3. Shortlist ranked by evidence strength × fit to the target

1. **MLA.** The single strongest measured, shipped memory result attacking exactly the binding
   resource: KV bytes per decode step, −93.3 % (arXiv:2405.04434v2). It preserves attention
   quality rather than trading it away. **Blocker:** trained in — not retrofittable onto
   Qwen3-8B post-hoc; using it means adopting an MLA checkpoint (losing Qwen3-8B's knowledge, as
   in Route C of the transfer survey) or paying conversion training. Whether any ≤10B MLA
   checkpoint fits 24 GB at wide context was `unverified`.
2. **Gated DeltaNet / Mamba-2 hybrid with ~25 % full attention (Qwen3-Next 3:1 recipe).** The
   strongest evidence that a hybrid matches or beats a larger Transformer: Qwen3-Next-80B-A3B
   ≈ Qwen3-235B-A22B (and beats it on RULER within 256K), and the controlled 8B study where the
   Mamba-2 hybrid beats the Transformer on all 12 tasks (+2.65 avg, 3.5T tokens) with up to ~8×
   faster generation (arXiv:2406.07887v1). **Blocker:** converting Qwen3-8B costs 0.35–20B tokens
   (weeks to months on one RX 7900 XTX) and ROCm support for the chunked gated-delta-rule / Mamba
   kernels is `unverified`.
3. **Decode-time sparse attention (NSA / DSA).** The only candidates with a measured
   *decode-stage* speedup: NSA up to 11.6× decoding and 9.0× forward at 64K (ACL 2025), and DSA
   shipped in a 685B production model at claimed quality parity. **Blocker:** the indexer must be
   trained, and the fast paths are CUDA-specific (FlashMLA / DeepGEMM; HF notes `flash_mla` not
   supported yet). DSA also disclosed an indexer RoPE bug on 2025-11-17 — a reproduction hazard.

**Under a no-training constraint none of this is an architecture change:** it reduces to 4-bit
weight-only integer quantisation plus KV quantisation on Qwen3-8B (already the strongest frontier
in `results/decision_report.md`) or to adopting a small existing hybrid / latent-KV checkpoint.

## 4. Negative and cautionary results

1. **Pure SSM/linear loses copying and in-context recall.** At 8B, Mamba-2 is 17 points below the
   matched Transformer on 5-shot MMLU at 1.1T tokens and still 1.37 below at 3.5T; Phonebook and
   long-context benchmarks stay hard at any budget. The hybrid (43 % Mamba-2 / 7 % attention /
   50 % MLP) is the only configuration that wins (+2.65 avg) (arXiv:2406.07887v1).
2. **Fundamental recall↔state tradeoff.** Based reports linear attention alone "lacks the
   precision to perform local token shifts and comparisons"; its 24× throughput claim is at
   **batch size 128**, not batch 1 (arXiv:2402.18668).
3. **SSM expressivity conclusions may be confounded by optimisation:** success confined to a
   narrow learning-rate window over 3000+ runs and ~20k GPU-hours; SSMs favour width over depth
   (arXiv:2508.19029v2). A 2026 single-paper claim that DeltaNet collapses to 0.010 exact match
   at 24 pairs (arXiv:2605.11196) is `unverified`.
4. **MoBA's headline is a prefill number**; full attention is used during generation
   (arXiv:2502.13189).
5. **MatMul-free LM's large-scale GPU numbers are simulated** ("randomly initialized weights"
   above 2.7B) (arXiv:2406.02528v3).
6. **Compute-only designs do not fix batch-1 memory** (MoD, arXiv:2404.02258).
7. **LoLCATs is recovery, not parity** — up to 42.4 points of MMLU lost before recovery; 77.8 % /
   78.1 % of the gap closed on 70B/405B (arXiv:2410.10254v3).
8. **Diffusion-LM throughput is a server metric** (arXiv:2506.17298v1; Mercury 2 blog).
9. **DSA reproduction hazard:** official README (2025-11-17) reports a RoPE layout bug in the
   indexer that "potentially lead[s] to degraded model performance".
10. **This project's own negatives still bind:** post-hoc structural removal of Qwen3-8B ≈ 4 % of
    parameters; integer group-128 dominates ternary at matched bits (`results/decision_report.md`).

## 5. Citations

1. Dao, Gu. *Transformers are SSMs (Mamba-2).* arXiv:2405.21060v1, ICML 2024. 2–8× faster SSD core; 2.7B/300B tokens.
2. Yang et al. *Gated Delta Networks.* arXiv:2412.06464, ICLR 2025.
3. Yang et al. *Gated Linear Attention Transformers.* arXiv:2312.06635v3. 340M/15B; 1.3B/100B.
4. *RWKV-7 "Goose".* arXiv:2503.14456v2. 0.19B–2.9B on a 3.1T-token corpus.
5. AI21. *Jamba.* arXiv:2403.19887v1. 12B active / 52B total; 256K; 8× KV reduction.
6. Glorioso et al. *Zamba.* arXiv:2405.16712v1. 7B/1T tokens.
7. Zyphra. *Zamba2 Suite.* arXiv:2411.15242v1. 1.2/2.7/7.4B; 30–50 % TTFT cut; 6× KV cut.
8. NVIDIA. *Hymba.* arXiv:2411.13676v1, ICLR 2025. 1.5B/1.5T tokens.
9. Lahoti et al. *Mamba-3.* arXiv:2603.15569v1, ICLR 2026. 1.5B; +0.6/+1.8 pts; half state size.
10. DeepSeek-AI. *DeepSeek-V2 (MLA).* arXiv:2405.04434v2. 236B/21B; KV −93.3 %; 5.76× throughput.
11. DeepSeek-AI. *Native Sparse Attention.* arXiv:2502.11089v2, ACL 2025. 27B/3B; 260–270B tokens; 11.6× decode at 64K.
12. Moonshot. *MoBA.* arXiv:2502.13189v1, NeurIPS 2025. Llama-8B-1M; prefill-only.
13. DeepSeek. *DeepSeek-V3.2-Exp (DSA).* Announcement + GitHub README + HF docs, 2025-09-29 (bug note 2025-11-17). 685B; top-2048 indexer.
14. DeepSeek-AI. *DeepSeekMoE.* arXiv:2401.06066, ACL 2024. 16B/2T tokens ≈ LLaMA2-7B at ~40 % compute.
15. DeepSeek-AI. *DeepSeek-V3.* arXiv:2412.19437v1/v2. 671B/37B; 14.8T tokens; 2.788M H800 hours.
16. Alibaba Qwen. *Qwen3-Next* blog, 2025-10-14 (+ HF implementation). 80B-A3B; 3:1 GDN:attention.
17. Alibaba Qwen. *On the Design of Qwen3.8-Next Architecture.* arXiv:2608.30320. 125B/6B active.
18. MiniMax. *MiniMax-01.* arXiv:2501.08313v1. 456B/45.9B active; 1M/4M context; 75 % MFU on H20.
19. Liu et al. *LoLCATs.* arXiv:2410.10254v3, ICLR 2025. +20 pts MMLU over prior linearisations; 70B/405B gap closed 77.8 %/78.1 %.
20. Wang et al. *Hierarchical Reasoning Model.* arXiv:2506.21734v3. 27M params, ~1000 samples, ARC-AGI 40.3 %.
21. Raposo et al. *Mixture-of-Depths.* arXiv:2404.02258v1. isoFLOP +1.5 %; up to 50 % fewer FLOPs.
22. *Scaling Latent Reasoning via Looped Language Models (Ouro).* arXiv:2510.25741v4. 1.4B/2.6B on 7.7T tokens; 2–3× parameter efficiency.
23. Sun et al. *TTT.* arXiv:2407.04620v1. 125M–1.3B; long-context advantage past 16K.
24. Behrouz et al. *Titans.* arXiv:2501.00663v1. >2M context NIAH.
25. Nie et al. *LLaDA.* arXiv:2502.09992v3. 8B, 2.3T tokens, 0.13M H800 hours.
26. Inception Labs. *Mercury* arXiv:2506.17298v1; *Mercury 2* blog. 1109 / 737 tok/s on H100; 1009 tok/s on Blackwell.
27. Meta. *Byte Latent Transformer.* arXiv:2412.09871v1, ACL 2025. Up to 8B / 4T bytes.
28. Zhu et al. *Scalable MatMul-free Language Modeling.* arXiv:2406.02528v3. 2.7B trained; >2.7B simulated.
29. Waleffe et al. *An Empirical Study of Mamba-based Language Models.* arXiv:2406.07887v1. 8B, up to 3.5T tokens; hybrid +2.65.
30. Arora et al. *Simple linear attention LMs (Based).* arXiv:2402.18668. 1.3B; 24× throughput at batch 128.
31. *Revisiting associative recall in modern recurrent models.* arXiv:2508.19029v2. 3000+ runs; learning-rate brittleness.
32. *Variational Linear Attention.* arXiv:2605.11196. Single-paper, `unverified`.
33. This repository: `memory.md` (744 GB/s VRAM; batch-1 bandwidth-bound), `results/decision_report.md`, `results/cross_architecture_transfer_survey.md`.

## 6. Unverified / unavailable

- **ROCm (gfx1100) support for any of these kernels:** no verified source found. Mamba / GDN /
  NSA / DSA kernels are documented as CUDA or Triton-on-NVIDIA; the Qwen3-Next HF port uses
  `torch_recurrent_gated_delta_rule` / `torch_chunk_gated_delta_rule`, but whether those have a
  tuned ROCm path is `unverified`.
- Whether any **≤10B MLA checkpoint** fits 24 GB at wide context — `unverified`.
- DeepSeek-V3.2 full report and its batch-1 cost numbers; MiniMax-01 per-benchmark table —
  only announcement / abstract text was opened.
- **Independent replication** of Gated DeltaNet, RWKV-7, Hymba, Zamba2, Mamba-3, Qwen3-Next,
  Qwen3.8-Next, MiniMax-01, LoLCATs, HRM, MoD, Ouro, TTT, Titans, LLaDA, Mercury, BLT,
  MatMul-free: none found by a non-overlapping group.
