# Design

## Context

See `proposal.md` for motivation. The constraints that shape this design:

- Phase 0 closed with STOP; the project has no active direction, and its surviving conventions are binding: the calibration slice definition, the `E_x` metric, the three-file run layout, seeds from config, and "a compression claim is compared at matched bits against an integer quantizer, not against FP16" (now inverted to matched *quality*, see D3).
- Runtime is ROCm-on-WSL on the RX 7900 XTX (24 GB), forward-only passes; vLLM/SGLang are unusable on this WSL host (UVA unavailable), so nothing may depend on a paged-attention engine.
- Disk on `C:` has ~37 GB free; the cache holds `Qwen/Qwen2.5-0.5B`, a Mistral-NeMo-12B GGUF and a Qwen3.8-27B GGUF. GGUF is not usable for weight surgery, so only one mid model may be downloaded.
- Rule R-005 (compare against the strongest baseline at matched budget, sanity-check that baseline, pre-register) and R-006 (plan and estimate before a sizeable direction) are active.

## Goals / Non-Goals

**Goals:**

- Produce a recorded redundancy map of the small model along three axes, each with a cheap pre-registered proxy and a finite budget.
- Produce one end-to-end verdict on an 8B model for a combined transformation, under a strict ±0.5 quality bar, with the compression claim stated against a matched-quality integer baseline.
- Keep the whole stage inside a fail-cheap envelope: at most one model download, at most 4 GPU-hours per cartography axis, and the recorded verdict stands whether it is GO or RED.

**Non-Goals:**

- Decode speed, peak VRAM fit at 24 GB, and context length are not measured or gated here (they are Stage B acceptance criteria).
- No new architecture, no pretraining, no custom kernels, no serving engine, no packaging.
- No reinterpretation of Phase 0/STOP records.

## Decisions

### D1 — One package per phase, mirroring Phase 0

New package `phase1_compression/` with `cartography/`, `transform/`, `evaluation/`, `main.py` and `config.yaml`; tests under `tests/phase1_compression/`. Rationale: the repository convention is one package per phase with a runnable entry point and no global state. Alternatives: extending `phase0_feasibility/` (rejected — Phase 0 is archived and its specs describe different behaviour), a single flat module (rejected — four responsibilities).

### D2 — Two models: cached 0.5B for cartography, `Qwen/Qwen3-8B` BF16 as the target

Cartography runs on `Qwen/Qwen2.5-0.5B` (already cached, so free). The probe runs on `Qwen/Qwen3-8B` BF16 (~16 GB, fits both the 37 GB disk and the 24 GB VRAM for forward-only passes). Rationale: the published post-hoc compression literature measures at 7–8B (Bonsai, ShortGPT, GISP), and small models are the worst case for compression (QuIP reports 2-bit failing below ~2B; AQLM finds 2-bit at matched bytes below a smaller 4-bit model). Alternatives: the cached 12B/27B GGUF (rejected — GGUF containers do not expose PyTorch weights for surgery), a 14B+ BF16 model (rejected — exceeds the disk budget for a single download and the forward-only VRAM budget). The model's exact shape fields (layers, heads, intermediate size) are read from its `config.json` at run time and never hardcoded.

### D3 — The gate is matched-quality, the inverse of the Phase 0 matched-bits rule

Every claim is compared against integer group-128 4/5/6-bit baselines at *equal or better quality*, and the reported number is the smallest-byte baseline that meets the same ±0.5 bar (GO needs ≤ 0.9 × those bytes, YELLOW is parity within 0.9–1.1 ×, RED otherwise). Rationale: structure removal reduces bytes *and* compute, which a bit-width comparison cannot credit; Phase 0 already showed matched-bits is the wrong frame for anything but pure quantization. Alternatives: matched bits again (rejected — repeats the frame that produced STOP), comparison against FP16 (rejected — the project's own convention forbids it).

### D4 — Evaluation: `lm-evaluation-harness` for benchmarks, the project's own script for perplexity

Benchmarks ARC-Easy, HellaSwag, LAMBADA and GSM8K via `lm-evaluation-harness`, pinned to one version recorded in the run config; WikiText-2 perplexity via the project's own script (fixed slice, `seq_len = 512`); few-shot counts fixed per benchmark in `config.yaml` (ARC-Easy 0-shot, HellaSwag 0-shot, LAMBADA 0-shot, GSM8K 4-shot). Rationale: the harness is the de-facto standard, so the numbers are comparable with the cited literature, and it reports every task from one invocation against the same HF model object. Alternatives: a self-written benchmark suite (rejected — re-implementing metric definitions is a correctness risk with no upside), `lighteval` (viable but less represented in the low-bit literature we compare against). The harness's installability against our pinned `torch`/`transformers` is verified by a spike task *before* it is used for any gate; if it cannot be installed, that is a recorded blocker, not a silent fallback.

### D5 — Healing is bounded and cannot upgrade the verdict

A single ≤ 1 GPU-hour LoRA run (rank ≤ 8, seeds from config) is allowed only after the removal mask is frozen; the verdict uses the **pre-healing** measurement, and a configuration that passes the bar only after healing is YELLOW at best. Rationale: "prune then LoRA" is standard practice, not a breakthrough; the claim must survive without it, while healing is still recorded because it is what a deployer would do. Alternatives: no healing at all (rejected — throws away a free diagnostic and diverges from the cited practice), full QLoRA (rejected — cost and it would blur attribution).

### D6 — Importance measures, one per axis

- Structured: magnitude × activation-norm (Wanda-style) with an iterative greedy schedule, as in GISP/Bonsai; the small-model cartography also runs a forward-only perturbative variant to cross-check the ranking.
- Directions/subspaces: difference-in-means directions plus a PCA/SOM multi-direction basis, scored by KL on a fixed benign slice with a grid over layer band and strength (the `ablate`-style search); removal is a weight orthogonalization.
- Ternary: on `W ≈ B diag(D) C`, three sub-probes — drop factorization planes, N:M-mask over the natural zeros of `B`/`C`, and reduce the rank of the ternary core — with any re-fit done by the same closed-form the decomposition uses (no gradients on factors).

Rationale: each measure has independent published support (GISP/Bonsai; `arXiv:2406.11717`/`arXiv:2602.02132`; Sparse-BitNet's ~42 % zero valley). Layer/block removal is included as a control because `arXiv:2602.01997` documents its collapse on generative tasks, so a null result there is expected and informative.

### D7 — The combination order is frozen from the map, and unit selection is recomputed on the target

The probe's ordered transformation list is written into its config from the recorded cartography ranking before the probe runs; the *procedure* transfers from 0.5B but the unit *selection* is recomputed on the 8B model with the same scores. Rationale: redundancy structure is scale- and model-dependent, but recomputing importance is cheap (forward-only) and removes the transfer assumption.

### D8 — No serving engine, no kernels, no GGUF

All work is forward-only PyTorch on ROCm; the compressed representation is validated as stored bytes and quality, not as a runtime. Rationale: engines are unavailable on this host and are Stage B work; including them would turn a research gate into an integration project. Consequence: a GO verdict is a statement about compression, not about deployability.

### D9 — Determinism and records

Seeds come from `config.yaml`; `torch.use_deterministic_algorithms(True)` is enabled where feasible and any nondeterministic source is named in the report; every run uses the three-file layout under `results/<experiment-id>/`, and the cartography map is recorded before the probe exists.

## Risks / Trade-offs

- **±0.5 % relative perplexity may be unreachable post-hoc** → Mitigation: RED is a pre-registered, valid outcome; the gate is not softened after seeing results.
- **Benchmark deltas at ±0.5 points are at the noise floor for GSM8K (~1.3 k items) and small for ARC-Easy** → Mitigation: perplexity is the primary arbiter and the report must state each benchmark's sample size and that a ±0.5 point delta is within noise for the small suites; benchmarks are corroborating, not decisive.
- **`lm-evaluation-harness` may not install against our pins** → Mitigation: spike task before any gate; failure is a recorded blocker, and the Stage A verdict is not issued on an unverified harness.
- **Redundancy found on 0.5B may not transfer to 8B** → Mitigation: D7 recomputes selection on the target; the small-model map is used only for ordering.
- **Healing masks a weak removal** → Mitigation: D5 — the verdict is pre-healing; healing is reported and capped at YELLOW.
- **Disk is 37 GB; one 16 GB download must succeed** → Mitigation: exactly one model download in the change; no dataset collection beyond the small `lm-eval` task data.
- **ROCm-on-WSL flakiness** → Mitigation: record the nondeterministic source per run rather than claiming determinism; the CPU path stays available for the cartography proxies if the GPU host fails.

## Stage B roadmap (documented, not implemented here)

Each track becomes its own change with its own pre-registered gate if Stage A returns GO or YELLOW:

1. **Ternary codec** — make the ternary axis a real storage format: N:M over the natural zeros, low-rank/plane sharing, closed-form re-fit, and a byte-exact accounting. Highest novelty, uses our own asset.
2. **Direction-subspace compression** — turn the KL-guided orthogonalization into a storage reduction (fewer effective rank directions) rather than a behavioural edit.
3. **Adaptive-depth / learned-skip architecture** — train a model that decides per token how many middle layers to run (building on the middle-layer redundancy literature); requires the budgeted tiny-pretrain envelope, so it is gated on a mechanism result from Stage A.
4. **MoE expert removal** — apply the REAP/Fisher-MoE evidence to a sparse MoE target, where expert removal is one-shot near-lossless at large scale; also the only track that speaks directly to the 24 GB / 20 tok/s goal.
5. **Tiny-pretrain architecture probe** (≤ 0.5B, single GPU) — only if Stage A shows the post-hoc route is empty but the mechanism (capability concentration) is real.

## Context sources

- Repository: `openspec/specs/phase0/*`, `openspec/project.md`, `results/decision_report.md`, `memory.md` (hardware measurements: VRAM 744 GB/s, fp16 47.2 TFLOPS; ROCm-on-WSL recipe), `.kilo/rules/{research-discipline,verification-gates,python-standards}.md`.
- Literature: `arXiv:2402.05406` (Bonsai), ACL 2026 GISP, `arXiv:2403.03853` (ShortGPT), ICLR 2025 deeper layers, `arXiv:2602.01997` (layer-pruning limits), `arXiv:2510.13999` (REAP), `arXiv:2606.05538` (Fisher-MoE), `arXiv:2401.06118` (AQLM), `arXiv:2502.02631` (ParetoQ), QuIP/QuIP#, `arXiv:2603.05168` (Sparse-BitNet), `arXiv:2406.11717` + `arXiv:2602.02132` (directional ablation), `arXiv:2602.18420` (SPQ), `arXiv:2608.24070` (Compression Trinity).
- Not verified here, closed by tasks: `lm-evaluation-harness` installability against our pins (T2); the exact shape fields and layer/head counts of `Qwen/Qwen3-8B` (T1, read from `config.json`); whether the model download succeeds within the disk budget (T1).

## Open Questions

None. Every decision that would change this design or the task breakdown is settled above; the remaining unknowns (harness installability, model shape fields, download success) are facts closed by tasks T1 and T2 before any gate run, not deferred choices.
