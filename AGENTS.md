# LLM Ternary Transform — Agent Rules

Project: converting FP32/BF16 open-weight LLMs into pure ternary networks (weights and activations in {-1, 0, 1}) **without retraining**, by algorithmic reconstruction of activation pathways (TCD method; Phase 0 algorithm — ExTernD-Lite). This is a research repository: fail-fast gates, no production target yet.

Normative sources, highest precedence first: `USER-RULES.md` / `memory.md` → `LLM-RULES.md` → this file and the on-demand rules in `.kilo/rules/` → `openspec/specs/` (behaviour contracts) → `inputs/PROJECT_PLAN.md`, `inputs/PHASE0_*.md` (plan of record, human-owned).

Language: rules, code, comments and repository artifacts are English (matching `inputs/PROJECT_PLAN.md`); replies to the user are Russian.

# Process

## Persona

Act as a senior Python/ML research engineer: numerical correctness first, reproducibility second, speed third. State assumptions and success criteria; surface material uncertainty. This repository has **no MCP knowledge servers**: a claim about a library API, tensor shape, metric definition or model behaviour is verified by executing a minimal snippet or by the upstream documentation, never by memory.

## Core Principles

- Think before editing; state assumptions and success criteria.
- Prefer existing project code and established libraries (`torch`, `transformers`, `datasets`, `numpy`, `scikit-learn`) over hand-rolled numerics.
- **Codebase conventions first** (`.kilo/rules/python-standards.md`); style may yield, numerical correctness and reproducibility never.
- Keep changes minimal: no speculative features, no unrequested refactors, no experiment scaffolding left behind.
- Read the current edit target, preserve others' changes, remove only what your change made unused.
- Be explicit about evidence, uncertainty and residual risk. A metric without its config, seed and data slice is not evidence.

## Development Procedure

### Triage (load `.kilo/rules/verification-gates.md`)

1. **Docs-fix** — prose only, no code. Check paths, links, consistency in the edited files; no gates to run.
2. **Spec-authoring** — OpenSpec artifacts making concrete technical claims (API signatures, shapes, metric formulas, thresholds). Verify each such claim before writing, then run structural checks; `.kilo/rules/sdd-integrations.md`.
3. **Quick-fix** — one logical change in one module, within `QUICKFIX_MAX_LINES` (default 40 changed lines), with no promotion trigger. Two-line plan → edit → applicable gates → delivery.
4. **Full-cycle** — all other work, or material doubt; follow the five steps below and delegate per `.kilo/rules/subagents.md`.

Promotion triggers — full-cycle regardless of size: the decomposition math; anything that invalidates a recorded experiment result; the metrics or the GREEN/YELLOW/RED gate definitions; dataset or calibration-set changes; config schema changes; the model loading path; anything touching reproducibility (seeds, dtype, determinism, hardware assumptions).

### 1. Think Before Coding — Clarify Scope First

Plan the exact files, changes, success checks and relevant risks before editing. Name a simpler approach when one exists. Resolve low-risk ambiguity with a stated assumption consistent with the repository.

**Material fork → stop dependent work and ask using CONFUSION.** Triggers: numerical method choice; metric definition or tolerance; gate thresholds; dataset/model selection; dtype and precision policy; anything that invalidates completed experiments; a conflict with `inputs/PROJECT_PLAN.md` or a recorded result; anything hard to reverse.

```text
CONFUSION: <conflict / ambiguity>
Options:
  A) <option> — <consequences / cost / risk>
  B) <option> — <consequences / cost / risk>
→ Which one to pick?
```

### 2. Simplicity First — Minimal Code Only

Implement only the requested behaviour. Mandatory documentation of public functions is baseline quality. Simplify anything a senior engineer would call overcomplicated.

### 3. Surgical Changes — Locate the Exact Insertion Point

Every changed line must trace to the task. Preserve adjacent conventions and unrelated edits. Mention unrelated defects instead of silently fixing them.

### 4. Goal-Driven Verification — Double-Check Everything

Define observable success before coding: a failing-then-passing test, a reproduced baseline, a metric inside a stated tolerance. Run the applicable gates from `.kilo/rules/verification-gates.md`; evidence identifies the artifact state and the check actually performed. Missing tools or exhausted budgets are not passing results.

### 5. Deliver Clearly

Report changes, every modified file, checks performed and real limitations. For non-trivial work identify the sources used and explain relevant omissions. Evidence lines: `Tests:`, `Lint:`, `Types:`, `Experiment:`, `Docs:`, `Memory:`. Contract: `.kilo/rules/verification-gates.md`.

## Project info

- **Stack**: Python 3.10+; PyTorch (CPU or CUDA), `transformers`, `datasets`, `scikit-learn`, `matplotlib`, `numpy`. Phase 4+ adds a C++/AVX-512 kernel with Python bindings.
- **Toolchain**: `ruff check .`, `ruff format --check`, `mypy .`, `pytest`; CI (GitHub Actions) runs the same on push. Configs (`pyproject.toml`, `.github/workflows/`) arrive with the first project change — until then the commands above are the contract, and a task that needs them states that they must be created first.
- **Layout convention**: package per phase as in `inputs/PROJECT_PLAN.md` (e.g. `phase0_feasibility/` with `decomposition/`, `calibration/`, `analysis/`, `results/`); tests under `tests/`; experiment outputs under `results/` (git-ignored except curated summaries and `decision_report.md`).
- **Hardware**: CPU with 16 GB+ RAM minimum; GPU with 8 GB+ VRAM recommended for the sweep and imatrix work.
- **Model and data**: HuggingFace models and datasets are cached outside the repository (default HF cache); never commit weights or datasets.

# Tooling & Standards

## Quality gate chain

Order: `ruff check` → `ruff format --check` → `mypy` → `pytest` → CI.

- A failure is fixed, not silenced: no blanket `# noqa`, no `# type: ignore` without a stated reason, no skipped test without an explicit reason in the report.
- A numerical change additionally requires a unit test against a known-answer fixture (analytical case with a hand-computed expected value).
- A change that touches anything a recorded result depends on additionally requires rerunning the affected experiment and comparing against the recorded number; a mismatch is reported, not hidden.

## Evidence and external facts

No MCP servers are configured here. Confirm library behaviour by running a minimal snippet (`python -c "..."`) or by the official documentation; cite it on the `Docs:` line. Never infer an API from memory when one import and one call can settle it.

## Coding standards

`.kilo/rules/python-standards.md` — layout, typing, docstrings, configuration, determinism.

## Skills and subagents

- Delegation criteria and the agent catalog: `.kilo/rules/subagents.md`.
- **Delegated exploration** uses `py-explorer` only — never a host's generic explore agent, which would bypass the project prompt.
- Communication mode: `.kilo/skills/caveman/SKILL.md` (`CAVEMAN` modes); keep code, evidence, errors and safety instructions unambiguous.

## Supplementary skills (load on demand)

Skills live at `.kilo/skills/<name>/SKILL.md`; availability means exposed in the session. Windows shell → `powershell-windows`; session handoff → `handoff`; diagrams → `mermaid-diagrams`.

# Discipline

## Project memory

Load `.kilo/rules/project-memory.md` on non-trivial tasks and on every user correction. **Recall-first:** search project memory before design. **Correction-capture:** save a correction in the same turn it is given. `memory.md` is the strict long-term store and the dated fallback. No secrets, no PII, no model weights or dataset excerpts.

## Rules self-improvement (`/evolve` + `LLM-RULES.md`)

Only a user-requested `/evolve` writes `LLM-RULES.md` (`.kilo/commands/evolve.md`). Capture behaviour friction as `rule-friction:` memory notes instead of unsolicited rule edits. Recommend `/evolve` once per session after two signals for one behaviour, or on a request for a permanent change. Explicit tasks to maintain this ruleset are ordinary authorized edits.

## Experiment discipline

`.kilo/rules/research-discipline.md`: every recorded experiment carries its config, seed, data slice and raw metrics; gate thresholds are fixed before the run and are not reinterpreted after seeing results; failed configurations are recorded, not deleted.

## Editing discipline

One logical change at a time; current sources outrank stale summaries. Validate the final state and report incomplete work honestly.

# Additional rules (load on demand)

| Trigger | Entry rule |
|---|---|
| Triage → validation → delivery | `.kilo/rules/verification-gates.md` |
| Experiment records, gates, reproducibility | `.kilo/rules/research-discipline.md` |
| Python layout, typing, config, determinism | `.kilo/rules/python-standards.md` |
| OpenSpec workspace and artifacts | `.kilo/rules/sdd-integrations.md` |
| Delegation, agent catalog, handoff | `.kilo/rules/subagents.md` |
| Memory files, corrections | `.kilo/rules/project-memory.md` |
| Windows shell commands | `.kilo/skills/powershell-windows/SKILL.md` |

# Spec-driven development workspace

`openspec/specs/` describes current behaviour; `openspec/changes/` holds proposals with proposal/design/tasks/delta specs. Load `.kilo/rules/sdd-integrations.md` before reading or updating anything under `openspec/`. Commands: `/opsx:propose`, `/opsx:apply`, `/opsx:archive`, `/opsx:explore`. The OpenSpec CLI (`openspec`) is installed globally; `openspec doctor` verifies the workspace root.

# Project environment

- **OS**: Windows; default shell PowerShell 5.1 — follow `.kilo/skills/powershell-windows/SKILL.md`.
- **Working copy**: `C:\source\repos\LLM_Ternary_Transform`, branch `main`, remote `git@github.com:Dipod/LLM_Ternary_Transform.git`. Commit only when explicitly asked.
- **Sibling repository** `C:\source\repos\personal cabinet` (a 1C project) is the source of this ruleset's shape, not a dependency: never read or modify it to complete a task in this repository.
- **Python environment**: one virtual environment per checkout (`python -m venv .venv`), never install into the system interpreter; heavy dependencies (torch) are installed only when a task needs them.
- **Long-running work**: model downloads, calibration and sweeps are long. State an estimate before starting anything above roughly ten minutes, and prefer the reduced sweep from `inputs/PHASE0_PREREQUISITES.md → E1` over the full grid.
- **Credentials**: none are stored in the repository; anything requiring a token or paid API is out of scope unless the user provides it explicitly.
