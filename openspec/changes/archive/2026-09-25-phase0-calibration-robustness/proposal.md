# Proposal: Phase 0 calibration robustness re-check

## Why

The archived change `phase0-ternary-feasibility` returned **YELLOW** on the Phase 0 gate: the best configuration (μ=3, τ=0.7, importance off) reached `E_x = 0.0716`, still above the GREEN bar of 0.05. A follow-up probe (`results/explore_mu3/`, 24 configurations at μ=3) showed that cheap in-algorithm tuning plateaus: more inner iterations (15→50), a finer τ grid and reordering move `E_x` only from 0.0716 to 0.0650, and nothing reaches 0.05.

Before changing the fitting method (AGA / activation weighting), we must rule out that the YELLOW verdict is an artifact of the **measurement**. Two properties of the current setup are suspect:

1. The calibration slice is small and unrepresentative: only **11,426 non-pad tokens**, because WikiText-2 raw "documents" are short lines truncated to `seq_len=512` and then padded.
2. `E_x` is measured **on the same activations used for the fit** (fit quality, not generalization), and only **one layer** (12) was evaluated.

## What Changes

- Add two alternative calibration slice definitions next to the archived one:
  - **`dense_windows`** (c1): concatenate the WikiText-2 train raw text and cut it into non-overlapping 512-token windows; use the first 128 windows (65,536 tokens).
  - **`many_documents`** (c2): keep the archived "first N non-empty documents" rule but with N = 1024 documents.
- Evaluate **three layers** (early / mid / late: 3, 12, 21) instead of layer 12 only.
- Split each slice into **fit (80%)** and **held-out (20%)** portions; report **both** `E_x_fit` and `E_x_heldout`.
- Re-run the reduced sweep (μ ∈ {2,3} × τ ∈ {0.7,1.0} × importance ∈ {off,on}) plus the μ=1 baseline for every (slice mode, layer) pair.
- Record every run under `results/` as before.

**Verdict basis is unchanged:** the GREEN/YELLOW/RED verdict still uses `E_x` on the fit portion, exactly as the archived gate defines; `E_x_heldout` is reported as a diagnostic and does **not** change the verdict. Changing the verdict basis to held-out would be a separate gate change and requires its own decision.

## Impact

- **Affected specs:**
  - `phase0/calibration-protocol` — new capability (slice definitions, fit/held-out split, multi-layer evaluation, record requirements).
  - `phase0/feasibility-gate` — no change; the metric and thresholds are untouched, and the verdict basis stays the fit portion.
  - `phase0/ternary-decomposition` — no change; the decomposition algorithm is untouched.
- **Affected code:** `phase0_feasibility/config.py`, `config.yaml`, `calibration/loader.py`, `main.py`, `analysis/report.py`, tests.
- **Archived results:** remain valid for their slice (128 documents, layer 12). This change adds new records; it does not overwrite or reinterpret the archived ones.
- **Not changing:** the model (`Qwen/Qwen2.5-0.5B`), the gate thresholds, the `E_x` formula, the decomposition algorithm, the random seed.

## Context sources

- `results/mu3_tau0.7_impoff_reoff/report.md` and `results/explore_mu3/` — the YELLOW verdict and the plateau.
- `openspec/changes/archive/2026-09-24-phase0-ternary-feasibility/design.md` → "Calibration protocol" — the archived slice definition.
- `openspec/specs/phase0/feasibility-gate/spec.md` — the unchanged metric and decision rule.
- Verified environment facts: ROCm-on-WSL on the AMD RX 7900 XTX; a full run of 9 configurations takes seconds (recorded runtimes ~1–5 s per configuration).
- Not checked: whether `dense_windows` or `many_documents` is more representative for this model — that is exactly what the runs will measure.
