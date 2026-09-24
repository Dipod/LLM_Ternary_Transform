# Verification gates

Canon for what must be checked before any work here is called done. `AGENTS.md → Development Procedure → 4` owns the procedure; this file owns the gates, the evidence and the budgets.

## Gate chain

Run in this order, on the artifacts as they are after the last edit:

| # | Gate | Command | Catches |
|---|---|---|---|
| 1 | Lint | `ruff check .` | unused imports, undefined names, shadowing, dead code, import order |
| 2 | Style | `ruff format --check .` | formatting drift |
| 3 | Types | `mypy .` | type errors, wrong call signatures, missing annotations on public API |
| 4 | Tests | `pytest` | behaviour, regressions, shape/dtype assumptions |
| 5 | CI | GitHub Actions on push | the same four on a clean checkout |

Rules for the chain:

- A failure is **fixed**, never silenced. No blanket `# noqa`; `# type: ignore` only with a stated reason on the same line; a skipped test only with an explicit reason in the report.
- **Numerical gate (additional, mandatory):** any change to the decomposition, metrics or quantization logic requires a unit test against a known-answer fixture — an analytical case whose expected value is computed independently (hand-derivation, closed form, or reference implementation) and whose tolerance is stated (`rtol`/`atol`).
- **Experiment gate (additional, conditional):** a change that anything recorded in `results/` depends on requires rerunning the affected experiment and comparing with the recorded number. A mismatch is reported, never averaged away.
- **Cross-check gate (additional, conditional):** an optimization (vectorization, fusion, a new kernel path) requires a comparison against the straightforward reference implementation on the same input, with the deviation reported.

## Evidence

Evidence identifies the artifact state and what was actually run:

- the file(s) with a content fingerprint (hash or "unchanged since the edit in this session");
- the exact command and its result (`clean`, counts of findings, pass/fail counts);
- for tests, the failing-then-passing order when the test was written first;
- for experiments, the experiment id from `.kilo/rules/research-discipline.md`.

Evidence reuse: do not rerun a gate whose input content has not changed since the recorded run. A fingerprint change (including a change in `pyproject.toml`, seeds or the config that produced the artifact) invalidates the affected evidence — the stale gate is rerun.

## Budgets

- `QUICKFIX_MAX_LINES` (default 40 changed lines) bounds the quick-fix triage path.
- One clean pass per gate on the final content. A gate that fails on a substantive defect: fix, rerun **only** the affected gate, at most twice; a third failure is reported as blocked, not iterated further.
- No reruns against unchanged input, and no re-running a passing gate because a later, unrelated file changed.
- A gate that cannot run (missing interpreter, no network for a dependency) is **not** a pass: record the skip and the reason in the report and, if it is on the critical path of the task, say the task is unverified.

## Delivery contract

For non-trivial work the final report states:

- every modified file;
- each gate actually run, with its result and run count;
- the numerical/experiment evidence when those additional gates applied;
- real limitations: what was not checked, what was assumed, what remains uncertain;
- evidence lines: `Tests:`, `Lint:`, `Types:`, `Experiment:`, `Docs:`, `Memory:`.

Incomplete work is reported as incomplete. A green summary over an unrun gate is a defect of the same severity as a failing test.
