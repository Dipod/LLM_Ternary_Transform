# phase0/ternary-decomposition Specification

## Purpose
Factorized ternary decomposition of a single weight matrix: it defines the factor structure, the gradient-free fit contract, the reordering and baseline behavior, and the determinism guarantees Phase 0 relies on.

## Requirements

### Requirement: Factorized ternary output

The decomposition of a weight matrix `W` of shape `[out, in]` SHALL output three factors: a ternary `B` of shape `[out, k]`, a real `D` of shape `[k]`, and a ternary `C` of shape `[k, in]`, such that the reconstruction `B @ diag(D) @ C` has shape `[out, in]`. Every element of `B` and `C` MUST be in {-1, 0, 1}; every element of `D` MUST be finite.

#### Scenario: Shapes and value domains

- **WHEN** a weight matrix of shape [4864, 896] is decomposed with rank multiplier mu=2
- **THEN** B is [4864, k], D is [k], C is [k, 896] with k = 2 * min(4864, 896) = 1792
- **AND** all entries of B and C are in {-1, 0, 1} and all entries of D are finite

### Requirement: Global rank with compute-only blocks

The rank `k` MUST equal `mu * min(out, in)`. Column blocks used to bound memory or time MUST NOT appear in the stored factors: the output MUST contain exactly one global `B`, `D` and `C` regardless of the number of blocks used.

#### Scenario: Block count does not change the factors

- **WHEN** the same matrix is decomposed with block size G=64 and with G=256
- **THEN** both runs return B, D, C of identical shapes ([out, k], [k], [k, in])
- **AND** neither run returns per-block stacks that multiply the rank by the number of blocks

### Requirement: Gradient-free fitting

The fit MUST be computed without automatic differentiation: no tensor in the decomposition path may require gradients and no backward pass may be executed. Every parameter update MUST be closed-form or alternating least squares.

#### Scenario: No autograd in the fit

- **WHEN** the decomposition runs with gradients disabled globally
- **THEN** the run completes without any gradient computation
- **AND** no code path in the decomposition sets requires_grad=True or calls backward()

### Requirement: Non-increasing residual with stall detection

Each rank-1 deflation step MUST use the optimal scalar for the current factors so that the fit residual (Frobenius norm of the weight residual) does not increase. The procedure MUST detect and report a step whose optimal scalar is zero or whose residual reduction is below a configured epsilon, instead of continuing silently.

#### Scenario: Monotone on random input

- **WHEN** the decomposition runs on a random matrix with a fixed seed
- **THEN** the residual at step i+1 is less than or equal to the residual at step i for every step
- **AND** if any step's reduction is below the configured epsilon, the run reports a stall

### Requirement: Reordering is consistent and reversible

When column reordering is enabled, the same permutation MUST be applied to the in_features axis of both the weight matrix and the calibration activations, and the inverse permutation MUST be available for deployment. Reordering MUST preserve the layer function up to the decomposition error.

#### Scenario: Reorder validity

- **WHEN** reordering is applied to a matrix with n columns
- **THEN** the result is a permutation of 0..n-1 with no duplicates and no missing indices
- **AND** applying the inverse permutation recovers the original column order

### Requirement: Symmetric baseline as contrast

A symmetric sign-threshold baseline MUST be available and MUST be evaluated on the same gate metrics as the primary decomposition, so the primary result is always reported against a contrast.

#### Scenario: Baseline uses the same metric

- **WHEN** the primary decomposition and the symmetric baseline are run on the same layer and calibration slice
- **THEN** both report the same metric set (output error, sparsity, effective BPW, runtime)

### Requirement: Determinism and reproducibility record

The decomposition MUST be deterministic for a fixed configuration, seed, dtype and hardware, and each run MUST record its configuration, seeds, data slice and raw metrics.

#### Scenario: Repeat run reproducibility

- **WHEN** the same configuration and seed are run twice on the same hardware
- **THEN** the reported metrics are identical to the stated tolerance
- **AND** a record exists containing config, seeds, data slice and raw metrics
