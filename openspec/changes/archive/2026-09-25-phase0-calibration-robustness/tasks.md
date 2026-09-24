# Tasks

## 1. Config schema for slices and layers

- [x] 1.1 Replace `target_layer` with `layers: [int, ...]` and add `slices: [str, ...]`, `dense_windows_count: int`, `many_documents_count: int`, `eval_fraction: float` to `Config` and `config.yaml`; verify the key-consumption test still passes (every YAML key consumed, no stray defaults).
- [x] 1.2 Update `phase0_feasibility/config.yaml` to the re-check grid (slices `[dense_windows, many_documents]`; `layers: [3, 12, 21]`; `eval_fraction: 0.2`) in a dedicated config file, leaving the canonical `config.yaml` on `docs128`/layer 12 defaults.

## 2. Calibration slice construction

- [x] 2.1 Implement `dense_windows`: concatenate the WikiText-2 train `text` in order, tokenize without padding, cut non-overlapping `seq_len` windows, keep the first `dense_windows_count`; verify 128 windows × 512 = 65,536 tokens and no padding.
- [x] 2.2 Implement `many_documents` (first `many_documents_count` non-empty documents, padded/truncated) and keep `docs128`; verify `docs128` reproduces the archived slice (128 documents, same token count 11,426).
- [x] 2.3 Implement the deterministic contiguous fit/held-out split (`eval_fraction`); verify the split is contiguous, reproducible and uses no RNG.

## 3. Multi-layer capture and metrics

- [x] 3.1 Capture gate_proj inputs for each configured layer into fit and held-out activation tensors; verify the token counts match the split for every layer.
- [x] 3.2 Compute `E_x_fit` and `E_x_heldout` per run and keep the verdict on `E_x_fit`; verify the verdict equals the fit-portion verdict under the unchanged rule.
- [x] 3.3 Extend the record: id `s-<mode>_l<layer>_...`, `metrics_fit`/`metrics_heldout` in `metrics.json`, slice/layer in provenance, and a report that names the verdict basis; verify the three-file layout and `metrics.json` round-trip.

## 4. Run and record the re-check

- [x] 4.1 Orchestrate the re-check: slices × layers × (μ∈{2,3} × τ∈{0.7,1.0} × importance∈{off,on}) plus the μ=1 baseline, recording every run.
- [x] 4.2 Run it on the AMD ROCm backend and record runtime and verdicts; verify each `report.md` states the verdict, both `E_x` values and that end-to-end perplexity was not measured.
- [x] 4.3 Verify the archived `results/` directories are unchanged and that new records live in their own directories.

## 5. Gates and memory

- [x] 5.1 Run the full gate chain `ruff check .`, `ruff format --check .`, `mypy .`, `pytest` on the final content.
- [x] 5.2 Record the outcome (per-slice/per-layer `E_x`, whether YELLOW survives) in `memory.md`, and state whether the verdict basis or method should change next.
