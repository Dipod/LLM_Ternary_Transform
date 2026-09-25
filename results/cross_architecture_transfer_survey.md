# Cross-architecture knowledge transfer — literature survey

Date: 2026-09-25. Read-only reconnaissance performed by a delegated literature agent
(`py-ml-researcher`); no code, no training runs, no files touched by it.

Question being answered: **are there proven, documented cases of migrating an already-trained
model's knowledge and skills into a *different* architecture — in particular into an
architecture that is cheaper in compute or memory yet performs comparably or better?**

Motivation in this repository: Phase 0 STOPped post-hoc ternary quantization
(`results/decision_report.md`, matched-bits comparison dominated by integer group-128
quantization) and Phase 1 Stage A was voided while finding that post-hoc structural redundancy
of `Qwen/Qwen3-8B` is about **4 % of parameters** (no removable layers, no removable attention
heads, ~5–6 % of FFN intermediate dimensions free, cliff between 6 % and 8 %). The idea under
test here is therefore whether a purpose-built cheaper architecture plus transfer from an
existing checkpoint is a demonstrated path.

Evidence labels used below: `proven-and-reproduced` (independently replicated or mainstream
adoption) / `single-paper-claim` (one group, no independent replication found) / `theory-only`
/ `tried-and-failed`. Sources were opened during the session; items that could not be confirmed
from an opened source are marked `unverified`.

## Summary answer

Cross-architecture transfer is real, peer-reviewed and demonstrated at 1.3B–72B — but it is
never free. Every mechanism includes a training phase, and the target architecture usually has
to be *co-designed* for convertibility. Pure linear/SSM conversions lose recall, so the strong
results keep a hybrid with a fraction of full attention. "Design an arbitrary cheaper
architecture and transplant the knowledge for free" is contradicted by the evidence; the
correct framing is "**re-initialise a purpose-built cheaper architecture from the source
checkpoint and pay a conversion-scale training bill**".

## 1. Migration categories with the strongest evidence

| Method | Transferred | Mechanism | Training budget actually used | Largest scale | Label | Source |
|---|---|---|---|---|---|---|
| MOHAWK | attention projections re-cast as Mamba-2 parameters; MLP/embedding weights; logits | three-stage progressive distillation (mixing matrices → block hidden units → end-to-end logit KD); **hybrid keeps 4 attention layers** | 3B tokens (Phi-Mamba-1.5B), 5B tokens (hybrid) | Phi-1.5 1.3B → Phi-Mamba 1.3B | single-paper-claim | arXiv:2408.10189v2, NeurIPS 2024 |
| Llamba | same MOHAWK recipe, Llama-3.x → Mamba | MOHAWK stages; Discrete-Mamba-2 variant | 8B / 10B / 12B tokens for 1B / 3B / 8B students | Llama-3.1-8B → Llamba-8B | single-paper-claim | arXiv:2502.14458v2 |
| Mamba in the Llama | attention linear projections reused; FFN frozen in stage 1 | attention→Mamba init + hybrid stepwise KD | 20B tokens, < 5 days on 8×A100-80G | Zephyr-7B, Llama-3-8B, 50/25/12.5/0 % attention | single-paper-claim | arXiv:2408.15237v4, NeurIPS 2024 |
| RADLADS | attention weights → RWKV-6/7; hidden-state alignment | target RWKV variants modified specifically to make conversion fit; **350–700M tokens, < 0.005 % of teacher tokens; 72B < $2 000** | 350–700M tokens | Qwen2.5 7B / 32B / 72B | single-paper-claim | arXiv:2505.03005v4, COLM 2025 |
| ARWKV | Qwen2.5 → RWKV-7 attention, RMSNorm + SwiGLU kept | attention alignment + KD | 7B on one A100-80G; token count `unverified` | Qwen2.5 7B (32B claimed) | single-paper-claim | arXiv:2501.15570v1 |
| Attention Bridge (CAB) | attention representations → token-dependent state projections | lightweight bridge, layer-wise alignment | "data-efficient"; exact budget `unverified` | LM + image classification | single-paper-claim | arXiv:2510.19266v4 |
| GQA up-training | K/V heads mean-pooled into fewer heads | up-train | **5 % of original pretraining compute** | T5-1.1 Base/Large/XXL | proven-and-reproduced | arXiv:2305.13245v3, EMNLP 2023 |
| Sparse Upcycling | dense FFN copied into every expert + router | continued training | ~50 % of the dense pretraining sunk cost | T5 Base/Large/XL, ViT | proven-and-reproduced | arXiv:2212.05055v2, ICLR 2023 |
| Branch-Train-MiX | domain experts branched from one seed, FFNs merged as experts | asynchronous expert training + MoE finetune | not stated uniformly | Llama-2 7B seed | single-paper-claim | arXiv:2403.07816v1 |
| Net2Net | function-preserving widening/deepening | exact re-parameterisation | ~0 for the transform | CNN / ImageNet | proven-and-reproduced | arXiv:1511.05641v4, ICLR 2016 |
| LiGO | learned linear width+depth growth operator | learn growth operator | 100 SGD steps; saves 44.7 % BERT-Base / 22.5 % GPT2-Medium FLOPs; ≤ 50 % overall | BERT-Base/Large, GPT-2 M/L, ViT | single-paper-claim | arXiv:2303.00980v1, ICLR 2023 |
| Stacking (G_stack) | depth-wise copy/stack + continued pretraining | growth | 300B → 194B tokens for equal loss (54.6 % speedup) | up to 7B | single-paper-claim | arXiv:2405.15319v2, NeurIPS 2024 |
| Depth Up-Scaling (SOLAR) | duplicate layers, slice at the seam, continue pretraining | growth | continued pretraining, token count `unverified` | Mistral 7B → 10.7B, released | proven-and-reproduced | arXiv:2312.15166v3 |
| DistilBERT | same-family half-depth student; triple loss | distillation at pretraining | exact GPU-hours `unverified` | BERT-base 110M → 66M (−40 %, 97 % GLUE, 60 % faster) | proven-and-reproduced | arXiv:1910.01108v4 |
| TinyBERT | attention+hidden+prediction distillation, 2 stages | distillation | `unverified` tokens | 7.5× smaller, 9.4× faster, > 96.8 % GLUE | proven-and-reproduced | arXiv:1909.10351v5 |
| MiniLM | last-layer self-attention + value-relation distillation | distillation | same-family | 50 % params, > 99 % SQuAD 2.0 / GLUE | proven-and-reproduced | arXiv:2002.10957v2, NeurIPS 2020 |
| Minitron | depth/width/attention/MLP pruning + KD retraining | prune then distill | **< 3 % of original data; up to 40× fewer tokens; 4B arm = 100B tokens** | Nemotron-4 15B → 8B/4B, open weights | proven-and-reproduced | arXiv:2407.14679v2 |
| Sheared LLaMA | targeted structured pruning + continued pretraining | prune then train | **50B tokens ≈ 3 % of from-scratch compute** | LLaMA2-7B → 1.3B/2.7B, open weights | proven-and-reproduced | arXiv:2310.06694v1 |
| Minitron-SSM | group-aware Mamba-head pruning + FFN/embedding pruning + KD | hybrid compression | ≤ 40× fewer tokens | Nemotron-H 8B → 4B, ~2× faster | single-paper-claim | arXiv:2504.11409v2, NeurIPS 2025 |
| CALM | cross-attention + projection bridges between **frozen** models | bridge-only training, no task data | bridge params only | PaLM2-S + PaLM2-XXS, up to +13 % absolute | single-paper-claim | arXiv:2401.02412v1, ICLR 2024 |
| Transport and Merge | optimal-transport alignment of activations → cross-architecture weight fusion | small unlabeled set | low-resource 1B targets, 25.26 → 27.44 (+2.18) | LLaMA-3-8B / Qwen2.5-7B source | single-paper-claim | arXiv:2602.05495v2, ICML 2026 |

Mechanistically, what is transferred is (i) tokenizer/embedding and FFN/MLP weights,
(ii) attention projection weights remapped into the SSM/linear parameterisation, and
(iii) output behaviour via knowledge distillation. It is parameter reuse **plus a real training
phase**, not extraction of "knowledge as an object".

Theory bridge: Dao & Gu show every structured SSM is equivalent to multiplication by a
semiseparable matrix (`arXiv:2405.21060v1`, ICML 2024, `theory-only` for the equivalence) —
this is what makes the mixing-matrix correspondence the conversion papers exploit.

Architecture families that required **from-scratch pretraining** rather than conversion:
Nemotron-H (`arXiv:2504.03624v1`), Samba (`arXiv:2406.07522v2`, 3.8B / 3.2T tokens), Hymba
(`arXiv:2411.13676v1`, ICLR 2025), Nemotron 3 Nano (`arXiv:2512.20848v1`, 25T tokens).

## 2. Negative results

1. **Pure recurrent/linear conversion degrades.** In Mamba in the Llama the 0 %-attention model
   "degrades significantly"; the 25 % model is only slightly worse than the teacher. MOHAWK
   always retains 4 attention layers (`2408.15237v4`).
2. **SSMs are provably limited at copying/recall.** A two-layer Transformer can copy
   exponentially long strings; any fixed-state GSSM cannot copy beyond its state size
   (`arXiv:2402.01032v2`, ICML 2024, plus empirical evidence).
3. **Controlled 8B comparison.** Mamba/Mamba-2 up to 3.5T tokens lag Transformers on 5-shot
   MMLU, Phonebook and long-context reasoning; the **hybrid** (43 % Mamba-2 / 7 % attention /
   50 % MLP) beats the Transformer by +2.65 average over 12 tasks (`arXiv:2406.07887v1`).
4. **Low-rank linearization is not lossless.** LoLCATs reports its own linearized LLMs perform
   worse than full-parameter alternatives and the original Transformers on 5/6 tasks, up to
   42.4 points on 5-shot MMLU in the baseline study (`arXiv:2410.10254v3`).
5. **Earlier linear attentions collapsed under conversion.** Hedgehog reports a 16.5 ROUGE-1
   drop for prior methods converting Llama-2 7B; its own LoRA conversion reaches 39.1 ROUGE
   against 43.5 for softmax+LoRA — recovery, not parity (`arXiv:2402.04347v1`, ICLR 2024).
6. **Naive MoE upcycling converges slower than from-scratch in the long term**, repaired only by
   statistical re-initialisation (Drop-Upcycling, ICLR 2025 proceedings).
7. **Distillation is not always cheaper.** With a fixed token budget distillation wins, but
   supervised learning beats distillation given enough student compute, and if the teacher must
   also be trained for a single student, supervised is generally preferable
   (`arXiv:2502.08606v2`, ICML 2025).
8. **This repository's own negative.** Post-hoc structural redundancy of Qwen3-8B is ~4 % of
   parameters; any route that hopes to obtain a cheaper architecture by pruning Qwen3-8B is
   capped near 1.04× (`results/decision_report.md`).
9. **Not found:** any peer-reviewed case of converting an arbitrary trained LLM into a
   *different* architecture with **zero training** and preserved quality. The closest is
   weight-transplant *initialisation* (RADLADS, ARWKV), always followed by distillation.

## 3. Cost floor

| Mechanism | Minimum reported budget | Scales with size? | Note |
|---|---|---|---|
| GQA / MQA up-training | 5 % of original pretraining compute | yes | Qwen3-8B is already GQA → no headroom |
| Sparse upcycling | ~50 % of dense pretraining cost | yes | increases total memory |
| Net2Net | ~0 for the transform | n/a | growth only |
| LiGO | 100 SGD steps; saves 22.5–50 % of from-scratch FLOPs | weak | still needs most of a training run |
| Depth stacking | 300B → 194B tokens (54.6 %) at 7B | yes (measured to 7B) | growth only |
| DUS / SOLAR | continued pretraining, `unverified` | yes | growth only |
| DistilBERT / TinyBERT / MiniLM | large-corpus distillation, `unverified` GPU-hours | yes | same family |
| Minitron | < 3 % of original data; up to 40× fewer tokens | sub-linear | open weights |
| Sheared LLaMA | 50B tokens ≈ 3 % compute | sub-linear | open weights |
| MOHAWK | 3B / 5B tokens | roughly flat | Mamba-2 target |
| Llamba | 8–12B tokens for 1B–8B | **flat in size** | open checkpoints |
| Mamba in the Llama | 20B tokens, ~40 A100-days | ~flat | open code/checkpoints |
| RADLADS | **350–700M tokens; 72B < $2 000** | **flat** (7B→72B same budget) | cheapest documented conversion |
| ARWKV | 7B on one A100-80G, `unverified` tokens | unclear | Qwen2.5 → RWKV-7 |
| Minitron-SSM | ≤ 40× fewer tokens | sub-linear | hybrid target |
| Samba / Hymba | from scratch (3.2T / `unverified`) | yes | not conversions |

The conversion budget is remarkably flat in model size; the expensive part is only relevant if
the target architecture must itself be pretrained from scratch.

## 4. Verdict

**Provable today.** Migrating a trained Transformer's parameters and behaviour into a different,
cheaper architecture is real and demonstrated at 1.3B–72B (`2408.10189`, `2408.15237`,
`2502.14458`, `2504.11409`, `2505.03005`, `2501.15570`). The conversion cost is small and
roughly flat in size: **~0.35–1B tokens / < $2 000** at the cheap end (RADLADS), 3B–20B tokens
for the SSM-specific recipes.

**Genuinely open.** Independent replication of RADLADS/ARWKV at 7–8B and on non-NVIDIA (ROCm)
hardware; whether a *pure* linear/SSM conversion preserves long-context recall; whether a
converted 8B fits 24 GB at useful speed (bf16 weights alone are ~16 GB, so a 4-bit weight path
plus a linear-state cache is required, and ROCm kernels for Mamba/RWKV are immature).

**Where the premise as stated fails.** The transplant always includes a training phase
(0.35B–20B tokens); the target architecture had to be co-designed for conversion (RADLADS
modified RWKV-6 into RAD-RWKV6/7); pure cheap architectures lose recall; post-hoc surgery on the
source is capped near 4 % on this exact model.

**Cheapest credible route from Qwen3-8B to 24 GB / ≥ 20 tok/s / wide context**

- **Route A — architecture change.** Qwen3-8B → hybrid Mamba-2 or RWKV-7 keeping 12.5–25 % of
  attention layers (Mamba-in-the-Llama / RADLADS recipe: reuse projections, attention transfer,
  logit KD). Budget: ~0.5–1B tokens for the RWKV family, ~20B tokens for Mamba-2. On the local
  RX 7900 XTX (47.2 TFLOPS fp16, ~6.6× below one A100-80G) that is weeks to months of dedicated
  GPU time. Risks: ROCm kernel maturity, recall/MMLU gap, mandatory attention subset.
- **Route B — fastest, not an architecture change.** Keep Qwen3-8B: 4-bit weight-only integer
  quantisation (already the stronger frontier in `results/decision_report.md`) plus KV-cache
  quantisation; GQA is already present. Fits 24 GB; ≥ 20 tok/s plausible. Cost: hours.
- **Route C — no local conversion.** Adopt an existing hybrid checkpoint that already fits 24 GB
  with wide context (for example Nemotron-Nano-9B-v2: 128k in bf16 on a 22 GiB A10G, up to 6×
  throughput versus Qwen3-8B, `arXiv:2508.14444v4`). Cost: download; Qwen3-8B's knowledge is not
  carried over.

**Under a "less than one day on one local GPU" budget, no route converts Qwen3-8B into a
genuinely different, cheaper architecture at comparable quality.**

## 5. Cheapest fail-cheap pre-check if the direction is ever opened

Convert a small cached model (`Qwen/Qwen2.5-0.5B`) to a hybrid: replace a fraction of attention
layers with RWKV-7/GLA modules, copy Q/K/V/O projections, alignment-train + logit-KD for
~50–100M tokens, keep at least 12.5 % full-attention layers. Measure: whether the ROCm kernel
runs at all, the perplexity gap and a long-context recall task (needle/passkey) against the
unmodified 0.5B, and the GPU-hours consumed. Cost: hours to days. It yields a transferability
number for this hardware before any 8B commitment is considered.

## 6. Unverified / not found

- Frankenstein/mergekit-style cross-architecture grafting: no peer-reviewed result opened.
- Exact continued-pretraining tokens for SOLAR/DUS; token budgets for ARWKV and CAB; GPU-hours
  for DistilBERT/TinyBERT/MiniLM — `unverified`.
- Independent replication of MOHAWK/Llamba/RADLADS/ARWKV by a non-overlapping group: none found.
- Numeric cross-architecture result for MergeME (NAACL 2025): abstract only, numbers `unverified`.
- No ROCm / consumer-GPU conversion result was found for any of these methods.

## 7. Citations

1. Bick, Li, Xing, Kolter, Gu. *Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models.* arXiv:2408.10189v2, NeurIPS 2024. 3B tokens (1.3B student) / 5B (hybrid); hybrid keeps 4 attention layers.
2. Bick, Katsch, Sohoni, Desai, Gu. *Llamba: Scaling Distilled Recurrent Models.* arXiv:2502.14458v2. 8B/10B/12B tokens for 1B/3B/8B; Llama-3.1-8B teacher.
3. Wang, Paliotta, May, Rush, Dao. *The Mamba in the Llama.* arXiv:2408.15237v4, NeurIPS 2024. 20B tokens; < 5 days on 8×A100-80G; 0 % attention degrades significantly.
4. Dao, Gu. *Transformers are SSMs.* arXiv:2405.21060v1, ICML 2024. Mamba-2 layer 2–8× faster.
5. Ainslie et al. *GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints.* arXiv:2305.13245v3, EMNLP 2023. 5 % of original pretraining compute; T5-1.1.
6. Komatsuzaki et al. *Sparse Upcycling.* arXiv:2212.05055v2, ICLR 2023. ~50 % of dense pretraining sunk cost.
7. Chen, Goodfellow, Shlens. *Net2Net.* arXiv:1511.05641v4, ICLR 2016.
8. Wang et al. *Learning to Grow Pretrained Models (LiGO).* arXiv:2303.00980v1, ICLR 2023. 44.7 % BERT-Base / 22.5 % GPT2-Medium FLOPs saved; 100 growth steps.
9. Du et al. *Stacking Your Transformers.* arXiv:2405.15319v2, NeurIPS 2024 Spotlight. 7B; 750B tokens; 300B→194B = 54.6 %.
10. Kim et al. *SOLAR 10.7B (Depth Up-Scaling).* arXiv:2312.15166v3, NAACL 2024 Industry. Mistral 7B → 48 layers; continued-pretraining tokens unverified.
11. Sanh, Debut, Chaumond, Wolf. *DistilBERT.* arXiv:1910.01108v4. 110M → 66M; −40 %, 97 % GLUE, 60 % faster.
12. Jiao et al. *TinyBERT.* arXiv:1909.10351v5, Findings of EMNLP 2020. 7.5× smaller, 9.4× faster, > 96.8 % GLUE.
13. Wang et al. *MiniLM.* arXiv:2002.10957v2, NeurIPS 2020. 50 % params, > 99 % SQuAD 2.0.
14. Muralidharan et al. *Compact Language Models via Pruning and Knowledge Distillation (Minitron).* arXiv:2407.14679v2. < 3 % of data; up to 40× fewer tokens; 15B → 8B/4B; 4B arm = 100B tokens.
15. Xia, Gao, Zeng, Chen. *Sheared LLaMA.* arXiv:2310.06694v1. 7B → 1.3B/2.7B; 50B tokens ≈ 3 % compute.
16. Wang et al. *The Hedgehog & the Porcupine.* arXiv:2402.04347v1, ICLR 2024. Prior linear attentions −16.5 ROUGE; own conversion 39.1 vs 43.5.
17. Zhang et al. *LoLCATs.* arXiv:2410.10254v3, ICLR 2025. 0.2 % of prior params; up to 42.4-point MMLU gap in the baseline study.
18. Jelassi, Brandfonbrener, Kakade, Malach. *Repeat After Me: Transformers are Better than State Space Models at Copying.* arXiv:2402.01032v2, ICML 2024.
19. Waleffe et al. *An Empirical Study of Mamba-based Language Models.* arXiv:2406.07887v1. 8B, up to 3.5T tokens; hybrid +2.65 average over 12 tasks.
20. Sukhbaatar et al. *Branch-Train-MiX.* arXiv:2403.07816v1.
21. Bansal et al. *CALM: LLM Augmented LLMs.* arXiv:2401.02412v1, ICLR 2024. PaLM2-S + PaLM2-XXS; up to +13 % absolute.
22. Cui et al. *Transport and Merge: Cross-Architecture Merging for LLMs.* arXiv:2602.05495v2, ICML 2026. 25.26 → 27.44 (+2.18).
23. Zhang et al. *Model Assembly Learning with Heterogeneous Layer Weight Merging.* arXiv:2503.21657v1, ICLR 2025 Workshop.
24. *Distillation Scaling Laws.* arXiv:2502.08606v2, ICML 2025. Teachers 143M–12.6B; distillation 3× more efficient at modest budgets; supervised wins at large budgets.
25. Zhang et al. *Data Efficient Any Transformer-to-Mamba Distillation via Attention Bridge (CAB).* arXiv:2510.19266v4. Exact token budget unverified.
26. *RADLADS: Rapid Attention Distillation to Linear Attention Decoders at Scale.* arXiv:2505.03005v4, COLM 2025. 350–700M tokens, < 0.005 % of teacher; Qwen2.5 7B/32B/72B; 72B < $2 000.
27. *ARWKV: Pretrain is not what we need, an RNN-Attention-Based Language Model Born from Transformer.* arXiv:2501.15570v1. Qwen2.5 → RWKV-7; 7B on one A100-80G.
28. Ren et al. *Samba: Simple Hybrid State Space Models.* arXiv:2406.07522v2, ICLR 2025. 3.8B / 3.2T tokens; 3.73× throughput at 128K.
29. Dong et al. *Hymba: A Hybrid-head Architecture for Small Language Models.* arXiv:2411.13676v1, ICLR 2025. 1.5B; 11.67× cache reduction; 3.49× throughput.
30. NVIDIA. *Nemotron-H family.* arXiv:2504.03624v1. MiniPuzzle 56B → 47B.
31. NVIDIA. *Minitron-SSM: Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning.* arXiv:2504.11409v2, NeurIPS 2025. Nemotron-H 8B → 4B; ≤ 40× fewer tokens.
32. NVIDIA. *Nemotron Nano 2.* arXiv:2508.14444v4. 12B pretrained on 20T tokens → 9B via Minitron; 128k on 22 GiB A10G bf16; up to 6× throughput versus Qwen3-8B.
33. NVIDIA. *Nemotron 3 Nano 30B-A3B.* arXiv:2512.20848v1. 25T tokens; 1M context.
34. LLM-jp. *Drop-Upcycling.* ICLR 2025 proceedings. Naive upcycling converges slower long-term.
35. This repository: `results/decision_report.md` (Phase 0 STOP; Phase 1 Stage A VOID; Qwen3-8B structural redundancy ≈ 4 %), `inputs/PROJECT_PLAN.md`, `memory.md`.
