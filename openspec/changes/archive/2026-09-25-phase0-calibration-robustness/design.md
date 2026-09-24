# Design: Phase 0 calibration robustness re-check

## Context

The archived gate verdict is YELLOW (`E_x = 0.0716` at μ=3, τ=0.7, importance off) on a slice of **11,426 non-pad tokens** measured at **one layer**, with `E_x` computed on the same activations used for the fit. Cheap tuning plateaued at `E_x = 0.0650`. This change tests whether the verdict survives a denser/longer slice, three layers, and a held-out split.

## Goals

- Measure whether `E_x` at μ=3 is stable across slice construction and layer position.
- Separate fit quality from held-out quality so the YELLOW is not read as generalization.
- Keep the archived verdict basis and thresholds untouched, so results are comparable.

## Non-goals

- No change to the decomposition algorithm, the `E_x` formula, the GREEN/YELLOW/RED thresholds, the model, or the seed.
- No switch of the verdict basis to held-out (explicitly deferred; would be a gate change).
- No end-to-end perplexity.

## Slice definitions

### D1 — `dense_windows` (c1)

Concatenate the `text` field of the WikiText-2 (`wikitext-2-raw-v1`, `train`) split in order, tokenize the concatenated stream once with the model tokenizer (no padding), and cut non-overlapping windows of `seq_len = 512` tokens. Use the first `dense_windows_count = 128` complete windows → 65,536 tokens. Padding: none (every window is full). *Rationale:* standard LM calibration; full context length; removes the padding waste that shrank the archived slice.

### D2 — `many_documents` (c2)

Keep the archived rule: the first `many_documents_count = 1024` non-empty documents, each tokenized and padded/truncated to `seq_len = 512`. *Rationale:* same construction as the archived slice, only longer; isolates slice length from slice shape.

### D3 — `archived` mode stays available

`docs128` (128 documents, the archived definition) remains selectable so the recorded runs can be reproduced by the same harness.

## Fit / held-out split

### D4 — Deterministic contiguous split

Split each slice **by order**: the first `round((1 - eval_fraction) * n_items)` items are the **fit** portion, the remainder is the **held-out** portion. `eval_fraction = 0.2`. No shuffling and no RNG, so the split is reproducible and the difference from the archived setup is only the slice, not an extra random draw. *Rationale:* keeps the change attributable; a random split would add a seed-dependent variable.

## Layers

### D5 — Early / mid / late

Evaluate `layers = [3, 12, 21]` of the 24-layer model. Layer 12 is the archived target; 3 and 21 bracket it. The gate unit stays one layer: each (layer, configuration) is scored and recorded independently.

## Verdict basis

### D6 — Verdict unchanged

The verdict for each run is computed from `E_x` on the **fit** portion, identical to the archived gate. `E_x_heldout` is recorded and printed as a diagnostic only. *Rationale:* switching the verdict basis is a gate-definition change and needs its own decision; this change must remain comparable to the archived YELLOW.

## Configuration schema

### D7 — New keys

Replace `target_layer: int` with `layers: [int, ...]`; keep `seq_len`; add `slices: [str, ...]`, `dense_windows_count: int`, `many_documents_count: int`, `eval_fraction: float`. `n_documents` is removed in favour of the two explicit counts. All keys are consumed by the loader (the key-consumption test is updated accordingly).

## Recording

### D8 — Experiment id and metrics

Id: `s-<mode>_l<layer>_mu<mu>_tau<tau>_imp<on|off>_re<on|off>_ni<n>`. `metrics.json` gains `metrics_fit` and `metrics_heldout` blocks plus the slice/layer fields in `provenance`; `report.md` shows both `E_x` values and states the verdict basis. The three-file layout is unchanged.

## Determinism and reproducibility

Seeds are unchanged and come from `config.yaml`. Slicing does no RNG. `torch.use_deterministic_algorithms(True)` is requested as before; HIP is not guaranteed bit-deterministic, so the record carries `torch_backend` and a repeat run is verified within `reproduce_rtol`.

## Risks

| Risk | Mitigation |
|---|---|
| Held-out E_x misread as the gate | Report explicitly states the verdict uses the fit portion; held-out is diagnostic |
| `dense_windows` concatenation changes token statistics vs documents | Both modes are run; the archived mode stays available for comparison |
| More runs inflate the record set | Runs are seconds each; each is a full recorded run under `results/` |
| Slice change invalidates the archived YELLOW | Archived records are untouched; this change reports its own verdicts per (slice, layer) |

## Context sources

- `results/mu3_tau0.7_impoff_reoff/metrics.json` (YELLOW), `results/explore_mu3/` (plateau).
- Archived design "Calibration protocol" and `openspec/specs/phase0/feasibility-gate/spec.md`.
- Verified: `datasets 5.0.1` exposes WikiText-2 `text`; the tokenizer accepts a long string with `truncation=False` (checked in this environment).
- Not checked: which slice the model's gate activations are most sensitive to; the runs will show it.
