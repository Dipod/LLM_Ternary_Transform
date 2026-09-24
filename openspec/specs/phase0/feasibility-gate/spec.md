# phase0/feasibility-gate Specification

## Purpose
Defines the Phase 0 metric set and the single canonical GREEN/YELLOW/RED decision rule, so the feasibility verdict is fixed before any run and cannot be reinterpreted after results are seen.

## Requirements

### Requirement: Gate unit is one Linear layer

The Phase 0 gate MUST be evaluated on a single Linear layer (MLP gate_proj) of the configured model, converted in isolation with the remaining model left in its original precision.

#### Scenario: Layer-local evaluation

- **WHEN** a Phase 0 verdict is produced
- **THEN** it is computed from the converted gate_proj layer only
- **AND** no end-to-end model conversion is required to produce the verdict

### Requirement: Output error metric

The primary quality metric MUST be the relative layer output error `E_x = ||X W^T - X What^T||_F / ||X W^T||_F`, computed with explicit transposes on calibration activations, where W is the original weight and What is the reconstruction.

#### Scenario: E_x on a perfect reconstruction

- **WHEN** the reconstruction equals the original weight matrix
- **THEN** E_x equals 0 within numerical tolerance

#### Scenario: E_x on a destroyed reconstruction

- **WHEN** the reconstruction is an independent random matrix of the same shape
- **THEN** E_x is greater than or equal to 1 within numerical tolerance

### Requirement: Effective BPW counting rule

Effective BPW MUST equal `mu * (out + in) / max(out, in) * (2 - sparsity)`: one bit of zero/nonzero mask plus one bit of sign per stored trit of B and C, normalised by the original parameter count `out * in`. It MUST NOT be defined as the per-entry Shannon entropy of a ternary factor.

#### Scenario: BPW bounds

- **WHEN** a decomposition with a dense ternary B and C (no zeros) is scored
- **THEN** effective BPW is greater than 0 and finite
- **AND** at equal sparsity a decomposition with more planes k has a strictly larger effective BPW than one with fewer planes

### Requirement: Single canonical decision rule

Exactly one GREEN/YELLOW/RED rule MUST be used, fixed before the run: GREEN when E_x <= 0.05 and sparsity >= 0.40 and effective BPW < 6.0; YELLOW when E_x <= 0.10 and effective BPW < 8.0; RED otherwise. Changing any threshold MUST be recorded as a change, not applied by editing the rule after results are seen.

#### Scenario: Verdict is one of three

- **WHEN** metrics are scored
- **THEN** the verdict is exactly GREEN, YELLOW or RED, with RED as the default when no other condition holds

#### Scenario: Threshold change is recorded

- **WHEN** a threshold or metric definition is altered
- **THEN** the alteration is recorded as a change to this spec rather than applied silently

### Requirement: Perplexity deferred

The Phase 0 gate MUST NOT include end-to-end perplexity, because a single converted projection out of many layers is not a sound end-to-end measurement. Perplexity is deferred to a later phase and its absence MUST be stated in the Phase 0 report.

#### Scenario: Gate without perplexity

- **WHEN** a Phase 0 verdict is produced
- **THEN** it rests on layer-local metrics only
- **AND** the report states that end-to-end perplexity was not measured

### Requirement: Validity check on the real metric

The gate MUST include a known-answer validity check on the real gate metric: a rank-1-consistent construction MUST score near zero error and a deliberately destroyed construction MUST score near the non-informative ceiling, so the metric itself is shown to be able to fail.

#### Scenario: Metric can detect failure

- **WHEN** the validity check runs
- **THEN** the known-good case passes and the known-bad case fails

### Requirement: Runtime budget is recorded

Each run MUST record runtime per layer configuration, and the gate MUST report whether the per-layer budget of 10 minutes was met.

#### Scenario: Runtime recorded

- **WHEN** a run completes
- **THEN** its runtime per layer configuration is recorded and compared against the 10-minute budget
