# Tasks

## 1. Environment and skeleton

- [ ] 1.1 Install CPU PyTorch plus `transformers` and `datasets` into `.venv`, then pin the exact resolved versions in `requirements.txt`; verify with `python -c "import torch, transformers, datasets; print(torch.__version__)"` from `.venv`.
- [ ] 1.2 Create the `phase0_feasibility/` package skeleton (`decomposition/`, `calibration/`, `analysis/`, `results/`) and a single config loader that reads `config.yaml`; verify the loader returns the expected dataclass for a checked-in example config.
- [ ] 1.3 Add `phase0_feasibility/config.yaml` with model, target layer, μ, τ, `b`, `n_inner`, seeds, dataset slice, `seq_len` and tolerances; verify a test asserts every key is consumed by the loader (no stray defaults in algorithms).

## 2. Ternary operator and decomposition core

- [ ] 2.1 Implement `T_tau`/`T_tau_cols`; verify a known-answer unit test with a hand-computed vector, including the all-zero safeguard.
- [ ] 2.2 Implement the sequential `decompose_ternary` (global `k = μ·min(m,n)`, 15 inner iterations, `v→u→d→deflate`, single-entry init); verify shapes `[m,k]`,`[k]`,`[k,n]`, values of `B`,`C` in {-1,0,1}, finite `D`, on a fixed-seed random matrix.
- [ ] 2.3 Add residual non-increase accounting plus stall detection (zero or sub-epsilon reduction); verify a test on a random matrix (non-increase every step) and a test on a matrix engineered to stall (stall reported).
- [ ] 2.4 Implement the batched component path (decorrelated targets, `b = min(256, ⌊min(m,n)/8⌋)`, sequential cross-block deflation); verify a cross-check test against the sequential path on identical input within the configured tolerance, with identical output shapes.
- [ ] 2.5 Implement `cosine_greedy_order`; verify validity (permutation of 0..n-1), inverse recovery, and function preservation `X[:,P] @ (W[:,P])ᵀ == X @ Wᵀ` within tolerance.
- [ ] 2.6 Implement the `μ=1` symmetric baseline; verify it returns factors on the same layer and computes the same metric set.

## 3. Calibration harness

- [ ] 3.1 Implement the model/dataset loader (`Qwen/Qwen2.5-0.5B`, first 128 WikiText-2 train documents, `seq_len=512`); verify a smoke test asserts 24 layers, `hidden=896`, `intermediate=4864`, and a successful forward pass.
- [ ] 3.2 Implement the forward hook capturing layer-12 `gate_proj` inputs; verify the captured tensor shape is `[tokens, 896]` and the token count matches the configured slice.
- [ ] 3.3 Implement the activation second-moment statistic `h_j = E[x_j²]`; verify a test computes it with gradients globally disabled (no `requires_grad`, no `backward`).

## 4. Metrics and gate

- [ ] 4.1 Implement `E_x` (explicit transposes), energy preserved, sparsity, `bpw_eff` (ExTernD formula) and runtime; verify known-answer tests: perfect reconstruction → `E_x≈0`; random reconstruction → `E_x≥1`; dense factors give `bpw>0`, and more planes give larger `bpw` at equal sparsity.
- [ ] 4.2 Implement the single verdict function and the validity check on the real metric; verify a known-good case passes and a known-bad case fails.
- [ ] 4.3 Implement the results writer producing `results/<experiment-id>/{config.yaml,metrics.json,report.md}`; verify a test run writes all three and `metrics.json` round-trips.

## 5. Sweep and recorded run

- [ ] 5.1 Implement `main.py` orchestrating the reduced sweep (μ∈{2,3} × τ∈{0.7,1.0} × importance∈{off,on}, reorder off) plus the baseline; verify every configuration produces a result directory with a parseable `metrics.json`.
- [ ] 5.2 Run the sweep on CPU and record runtime and verdicts; verify each `report.md` states the verdict, the metrics, and that end-to-end perplexity was not measured.

## 6. Correct stale inputs and documentation

- [ ] 6.1 Correct `openspec/project.md` (remove `[4096, 11008]`, the `μ·min(m,G)` rank, `128-sample` and reduced-sweep claims that contradict the verified facts); verify no occurrence of `4096`/`11008` remains.
- [ ] 6.2 Correct `.kilo/agents/py-architect.md` where it repeats the same fabricated shapes/thresholds; verify no stale number remains.
- [ ] 6.3 Add a memory entry recording the Phase 0 outcome and the gate verdict; verify the entry exists in `memory.md`.

## 7. Integration checks

- [ ] 7.1 Recreate one full run from a fresh process with the same seed and verify metrics match the earlier run within the stated tolerance, and that the verdict recomputes consistently from `metrics.json`.
- [ ] 7.2 Run the full gate chain `ruff check .`, `ruff format --check .`, `mypy .`, `pytest` and verify all four pass on the final content.
