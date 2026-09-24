# Research discipline

This repository exists to answer one question fast and cheaply: is factorized ternary decomposition (ExTernD-Lite) viable for open-weight LLMs? Everything here serves the fail-fast gates, so the rules below are hard gates, not preferences.

## Gate definitions are fixed before the run

The GREEN / YELLOW / RED thresholds for Phase 0 are defined in `PROJECT_PLAN.md` and mirrored in the OpenSpec delta spec. They are fixed **before** data is produced.

- Changing a threshold, a metric definition, a tolerance or the decision rule requires a `CONFUSION` block and the user's decision, and the change is recorded in the OpenSpec change (`design.md`) — never as a quiet post-hoc edit to a document after seeing results.
- A result that fails the gate is a valid, valuable outcome. It is reported as such.

## Every recorded experiment carries its context

An experiment that is not reproducible from its record is not evidence. Each run that anyone will cite later is recorded under `results/<experiment-id>/` with:

```text
results/<experiment-id>/
├── config.yaml    # every parameter that influenced the run, including swept values
├── metrics.json   # raw metrics, machine-readable
└── report.md      # verdict, interpretation, links to the change/design
```

`report.md` states at minimum: date; git commit; model and dtype; target layer; dataset and exact slice (e.g. `wikitext-2-raw-v1` train, first 512 rows); calibration sample count; all hyperparameters (μ, τ, G, reorder on/off, importance on/off, planes K); seeds; hardware; runtime; and the metrics: output error `E_x`, effective BPW, sparsity, perplexity delta, runtime per layer.

## Reproducibility

- Seeds are explicit and come from the config, never hardcoded inside algorithms: `torch.manual_seed`, `numpy.random.seed`, and `random.seed` for completeness.
- Where feasible, `torch.use_deterministic_algorithms(True)` is enabled; where it is not (a CUDA kernel without a deterministic path), the record names the nondeterministic source instead of pretending determinism.
- Library versions that affect numerics (`torch`, `transformers`, `numpy`) are recorded; the environment is recreated from `requirements.txt`/`pyproject.toml` plus the lock step used for the run.
- Floating-point comparisons state their tolerance. Cross-machine reproducibility is claimed only at a stated tolerance.

## No cherry-picking

- All swept configurations are reported, or the record states the reduction rule used to skip the rest (`PHASE0_PREREQUISITES.md → E1` defines the reduced grid).
- Failed and RED configurations are kept in the record, not deleted: they are what makes the baseline credible (`F4` exists exactly for this).
- Metrics are never compared across runs with different data slices, dtypes or seeds without saying so.
- A number quoted from literature carries its source. A SOTA threshold used as a target is cited (paper, method, configuration) in the doc that states it.

## Cost discipline

- State a runtime estimate before any run expected to exceed roughly ten minutes, and say what it costs (download size, RAM/VRAM).
- Prefer the reduced grid, the cheaper metric (layer output error as proxy) and the smaller calibration set first; expand only when the result is ambiguous (`PHASE0_PREREQUISITES.md → E1`, `E2`).
- Do not start a full sweep, a model download or a full-perplexity evaluation without the user's explicit go-ahead in this session.

## Phase discipline

- Phase 0 is a feasibility gate, not a product: no inference engine, no custom kernels, no packaging work unless the gate passed or the user asked explicitly.
- A phase gate's outcome (GREEN / YELLOW / RED) is recorded in the OpenSpec change and in `results/<experiment-id>/report.md` before any work of the next phase starts.
