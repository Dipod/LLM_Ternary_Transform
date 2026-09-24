# phase0/calibration-protocol Specification

## Purpose
Defines how Phase 0 builds its calibration slice, splits it into fit and held-out portions, and evaluates several layers — so the YELLOW verdict can be re-checked against slice construction and layer position without changing the metric or the decision rule.

## Requirements

### Requirement: Selectable calibration slice modes

The harness MUST support three calibration slice modes: `dense_windows` (concatenated token stream cut into windows), `many_documents` (first N non-empty documents, padded/truncated), and `docs128` (the archived 128-document definition). Each mode MUST be selectable independently and MUST report its token count.

#### Scenario: All three modes build a slice

- **WHEN** the harness runs with slice modes `[dense_windows, many_documents, docs128]`
- **THEN** each mode produces a tokenized slice and records its non-pad token count
- **AND** `docs128` reproduces the archived 128-document slice

#### Scenario: Dense windows are full-length

- **WHEN** the `dense_windows` slice is built with `seq_len = 512` and `dense_windows_count = 128`
- **THEN** the slice has 65,536 tokens and every window is exactly 512 tokens with no padding

### Requirement: Deterministic fit and held-out split

Every slice MUST be split by order into a fit portion and a held-out portion at a configured fraction, with no shuffling and no randomness, so the split is reproducible.

#### Scenario: Split is contiguous and reproducible

- **WHEN** a slice of N items is split with `eval_fraction = 0.2`
- **THEN** the first `round(0.8 * N)` items form the fit portion and the remaining items form the held-out portion
- **AND** repeating the split on the same slice yields identical portions

### Requirement: Multi-layer evaluation

The harness MUST evaluate a configured list of layers, scoring each layer independently and recording it as its own run.

#### Scenario: Early, mid and late layers

- **WHEN** the harness runs with `layers = [3, 12, 21]`
- **THEN** each layer produces its own recorded run for every configuration
- **AND** no run mixes activations from two layers

### Requirement: Held-out error is reported, not used for the verdict

Each run MUST report the layer output error on the fit portion (`E_x_fit`) and on the held-out portion (`E_x_heldout`). The GREEN/YELLOW/RED verdict MUST be computed from `E_x_fit` only, identically to the gate rule. The report MUST state that the held-out value is a diagnostic and does not drive the verdict.

#### Scenario: Verdict uses the fit portion

- **WHEN** a run is scored
- **THEN** its verdict equals the verdict of the fit-portion metric under the unchanged decision rule
- **AND** the report names `E_x_fit` as the verdict basis and `E_x_heldout` as a diagnostic

### Requirement: Slice and layer recorded in provenance

Each run MUST record its slice mode, slice token count, fit/held-out item counts and target layer in the provenance block, and MUST include both metric sets (`metrics_fit`, `metrics_heldout`) in `metrics.json`.

#### Scenario: Provenance identifies the measurement

- **WHEN** a run completes
- **THEN** `metrics.json` contains the slice mode, the layer and both metric sets
- **AND** the three-file run layout (`config.yaml`, `metrics.json`, `report.md`) is preserved

### Requirement: Archived results are not reinterpreted

The re-check MUST write its own records and MUST NOT overwrite or reinterpret the archived `results/` runs, which remain valid for their own slice (128 documents, layer 12).

#### Scenario: Separate records

- **WHEN** the re-check runs
- **THEN** every new record is a new run directory with its slice mode and layer in its id
- **AND** the archived run directories are left unchanged
