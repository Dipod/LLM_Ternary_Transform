# Spec Delta

## Purpose

Defines how Stage A maps removable structure in a target model: three independent measurement axes, the cheap pre-registered proxy used on each, the finite search budget, and the rule that the map and the transformation order are frozen on disk before the combined probe runs.

## ADDED Requirements

### Requirement: Three independent cartography axes

Cartography MUST measure removable structure along three axes independently, each with its own recorded run: (a) structured components — transformer blocks, attention heads and FFN intermediate dimensions; (b) directions and subspaces — linear directions removed by orthogonalization; (c) the model's own ternary representation — factorization planes, N:M patterns over the natural zeros of the ternary factors, and low-rank structure of the ternary core. No axis MAY reuse another axis' removal results as its baseline.

#### Scenario: Axes are separate runs

- **WHEN** the cartography runs
- **THEN** each axis produces its own recorded run directory with its own configuration and metrics
- **AND** no axis' curve is computed on a model already modified by another axis

### Requirement: Cheap pre-registered proxy per axis

Each axis MUST be scored with a cheap proxy fixed before the run: the relative layer output error `E_x` shared with Phase 0 where the axis is measured at a layer, and the KL divergence between the candidate model's next-token distribution and the reference model's on a fixed benign slice where the axis changes the output distribution. The proxy and its degraded-state budget MUST be recorded in the axis configuration before the axis runs.

#### Scenario: Proxy is declared before the axis runs

- **WHEN** an axis is started
- **THEN** its proxy metric and budget are already present in its configuration file

#### Scenario: Proxy can be cross-checked

- **WHEN** an axis reports a collapse point
- **THEN** the same removal step is re-scored with the Phase 0 `E_x` definition on a fixed layer as a cross-check

### Requirement: Structured-component removal protocol

The structured axis MUST remove whole units (blocks, heads, FFN intermediate dimensions) using a stated importance score, iteratively and greedily, recording the proxy after every removal step until the fixed budget is exceeded. The run MUST record the unit ranking, the removal order and the first step at which the budget is exceeded.

#### Scenario: Collapse point is recorded

- **WHEN** the structured axis completes
- **THEN** its record contains the removal order and the first unit whose removal pushes the proxy past the budget

#### Scenario: Importance score is stated

- **WHEN** the structured axis ranks units
- **THEN** the score used is named in the configuration and applied identically to every unit of that type

### Requirement: Direction and subspace removal protocol

The direction/subspace axis MUST search over direction, layer band and ablation strength, scoring each candidate by KL divergence from the reference on the fixed benign slice, and MUST record the search grid and every evaluated candidate. Removal MUST be expressed as an orthogonality constraint on the affected weight matrices so that it is storage-free at inference.

#### Scenario: Search grid is recorded

- **WHEN** the direction axis completes
- **THEN** its record lists every evaluated (direction, layer band, strength) candidate with its KL score

#### Scenario: Removal is a weight edit

- **WHEN** a direction is selected for removal
- **THEN** it is applied as a weight orthogonalization and not as an inference-time hook

### Requirement: Ternary-representation removal protocol

The ternary axis MUST operate on a TCD decomposition `W ≈ B diag(D) C` with `B, C ∈ {-1, 0, 1}` and MUST measure, separately and at a fixed proxy budget: dropping factorization planes, applying N:M masks to the natural zeros of `B` and `C`, and reducing the rank of the ternary core. Any re-fit after a removal MUST use the same closed-form used by the decomposition and MUST NOT train the factors.

#### Scenario: Natural zeros are the starting point

- **WHEN** the ternary axis measures N:M masking
- **THEN** the observed zero fraction of the undecomposed TCD output is recorded first and the mask is applied to those zeros

#### Scenario: No gradient training

- **WHEN** the ternary axis re-fits after a removal
- **THEN** no gradient descent is used on the factors

### Requirement: Calibration slice is reused

Cartography MUST use the calibration slice definition of the Phase 0 calibration protocol (WikiText-2 `wikitext-2-raw-v1`, `dense_windows`, `seq_len = 512`, 128 windows, 65,536 tokens, pad positions dropped) so its measurements are comparable with the recorded Phase 0 and STOP artifacts.

#### Scenario: Same slice definition

- **WHEN** an axis builds its calibration slice
- **THEN** its configuration names the Phase 0 slice definition and the run records the non-pad token count

### Requirement: Map and transformation order are frozen before the probe

The combined-transformation order used by the probe MUST be derived from the recorded cartography map and written into the probe's configuration before the probe starts. The map MUST be recorded with the standard run layout (`config.yaml`, `metrics.json`, `report.md`) under `results/<experiment-id>/` before any probe result exists, and the probe record MUST reference the map's experiment id.

#### Scenario: Order is frozen in advance

- **WHEN** the probe is configured
- **THEN** its configuration contains the ordered list of transformations and the id of the cartography run that produced it

#### Scenario: Map exists first

- **WHEN** the probe runs
- **THEN** the cartography run directory already exists on disk with its verdict

### Requirement: Protocol transfers, unit selection is recomputed

The cartography procedure MAY be measured on the small model, but the unit selection used by the probe MUST be recomputed on the target model with the same importance score and removal protocol, and the probe MUST record both the small-model map and the recomputed target-model selection.

#### Scenario: Recompute on the target

- **WHEN** the probe is prepared on the target model
- **THEN** its removal selection is produced by running the same scoring procedure on the target model, not by copying the small model's unit list

### Requirement: Finite fail-cheap search budget

Each axis MUST run under a finite budget fixed before the run: at most 4 GPU-hours on the small model per axis, and a total cartography wall-clock of at most one working day. An axis MUST stop when its budget is exhausted and MUST record that it stopped on budget rather than extend the search.

#### Scenario: Budget is respected

- **WHEN** an axis reaches its 4 GPU-hour cap
- **THEN** it stops, records the cap as the stopping reason, and reports the curve collected so far

#### Scenario: Every candidate is recorded

- **WHEN** an axis evaluates a candidate
- **THEN** that candidate's configuration and proxy value appear in the run record
