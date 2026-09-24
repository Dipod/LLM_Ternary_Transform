# Python standards

Conventions for this repository. Follow the file you are editing first; where it is silent, these apply.

## Layout

- One package per phase, as laid out in `PROJECT_PLAN.md` — e.g. `phase0_feasibility/` with `decomposition/`, `calibration/`, `analysis/`, `results/`.
- Tests mirror the package layout under `tests/` (`tests/decomposition/test_asymmetric.py` ↔ `phase0_feasibility/decomposition/asymmetric.py`).
- A runnable entry point per phase (`main.py`) that reads `config.yaml` and orchestrates; library code never reads global state or environment variables directly.

## Typing

- `mypy` clean. Every public function and method has full annotations, including return types.
- Tensors are annotated as `torch.Tensor`; say the shape and dtype in the docstring.
- Avoid `Any`. Where a library forces it, keep it at the boundary and narrow immediately.

## Docstrings and comments

- Every public function carries a docstring: purpose, parameters with shapes and dtype, return value, raised errors.
- Tensor shapes are written as they are used in `PROJECT_PLAN.md`: `W_fp: [out_features, in_features]`.
- Comments explain *why*, not *what*. A non-obvious numerical trick (e.g. a deflation step, a softened threshold) states the property it relies on and where it comes from.

## Numerics

- Never compare floats for exact equality; use explicit tolerances and state them.
- Assert finiteness where a NaN or an infinity would silently poison a sweep (`torch.isfinite`), and fail loudly instead of propagating.
- Keep dtype and device handling explicit; no hidden `.float()` casts inside algorithms. Casting is a decision at the boundary.
- Prefer vectorized torch operations over Python loops over tokens or parameters; but keep a readable reference implementation next to an optimized one when it aids verification, and cross-check them (`.kilo/rules/verification-gates.md → Cross-check gate`).

## Configuration

- All tunable values (μ, τ, G, K, seeds, model name, layer index, dataset slice, tolerances) come from `config.yaml`, not from literals inside algorithms.
- Defaults live in one place (a dataclass or config loader) with a comment stating the SOTA or the reason for the default.
- A config change that affects a recorded result is a promotion trigger (`AGENTS.md → Triage`): rerun the affected experiment.

## Determinism

- One helper sets all seeds from the config; algorithms never call the seed functions themselves.
- State nondeterminism explicitly where the platform forces it rather than hiding it.

## Errors

- Fail loudly with a message that names the value, the expected range and the parameter it came from. A bare `except` is forbidden; a caught exception is either handled meaningfully or re-raised with context.
- No silent fallbacks: a decomposition that fails its internal check raises, it does not return a degraded result.

## Naming and style

- `snake_case` functions and variables, `PascalCase` classes, `UPPER_SNAKE` constants. Identifiers, comments and literals are English (see `AGENTS.md → Language`).
- Prefer a small module with one responsibility over a large "utils" module.
- `ruff` decides layout and import order; do not fight the formatter in review.

## Tests

- `pytest`; one behaviour per test; the failing-then-passing order is recorded when the test was written first.
- Known-answer fixtures for numerics: the expected value is derived independently and the tolerance stated.
- Property tests are welcome for invariants (output values strictly in {-1, 0, 1}, monotone residual decrease, permutation validity) — `hypothesis` may be added when the first such test lands.
