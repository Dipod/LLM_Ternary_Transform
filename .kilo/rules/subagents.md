# Subagents — catalog and delegation rules

Load this file when a task feels large, multi-module or research-heavy and delegation is a candidate.

## Delegation criteria

Delegate when at least one holds:

- the change touches **≥ 3 modules** or **≥ 2 packages** (a decomposition algorithm plus its calibration and analysis, for example);
- an **independent read-only track** exists (exploration, literature/pattern survey, impact listing) that can run while the parent continues;
- the task needs **≥ 5 files read** before the first edit, or a **mechanical edit across ≥ 5 files** — the parent's context is the bottleneck;
- an **independent review** is explicitly requested by the user.

Otherwise execute directly. The five-step procedure in `AGENTS.md` applies either way, and the gates in `.kilo/rules/verification-gates.md` are never waived.

Model choice: this repository stores no model settings. The parent inherits the client's model; a bounded read-only task (scouting, a metric summary, a smoke check) may deliberately be run on a cheaper model in the client, and its output is working material, never authority.

## Host-tool built-in explorers (hard ban)

Hosts ship a built-in generic explore helper with a fixed prompt that ignores the project's rules. For any delegated read-only exploration matching `py-explorer` in the catalog below:

1. launch **`py-explorer`**;
2. never launch the host's built-in Explore / generic scout for that work;
3. if the session cannot start `py-explorer`, either do the exploration on the parent or tell the user that `py-explorer` is unavailable — silent substitution is a defect.

## Common obligations (every subagent)

- **CONFUSION on material forks**: numerical method, metric definition, gate threshold, dataset/model choice, dtype policy, anything invalidating a recorded result. Never resolve such a fork by picking an interpretation silently; never paraphrase it into prose. Low-risk ambiguity: state the assumption in one line and proceed.
- **Evidence**: run the gates that apply to what you touched; report the command and the observed result. Never present an unrun gate as passing.
- **Scope**: edit only the assigned files; report an unrelated defect instead of fixing it; never revert another agent's edits; never delete files without an explicit instruction.
- **Done criteria**: every assigned item is implemented or explicitly listed as not done. A plan that turns out wrong goes back to the parent as a `CONFUSION`; the subagent does not re-plan.
- **Report vocabulary**: severity `critical` / `major` / `minor`; status `✅ DONE` / `⚠️ PARTIAL` / `❌ BLOCKED` for implementers, `✅ APPROVE` / `⚠️ CONCERNS` / `❌ BLOCK` for reviewers and testers.
- **Handoff**: when a change is split across several implementation subagents, the upstream one emits a `## Handoff for the next subagent` block (artifacts, evidence with fingerprints, public surface, open TODOs, locked decisions, open questions) at the top of its report; the next one reads it first and treats current file contents as the source of truth.

## Catalog

| Subagent | When to call | When not to call |
|---|---|---|
| `py-explorer` | Read-only exploration across many files or packages, dependency and call-site questions, "where is X / how does Y work" before planning or editing | A lookup the parent can settle with one read or one grep |
| `py-analytic` | A PRD/specification or an analysis of an existing area without writing code | The task is to write code |
| `py-planner` | A multi-step implementation or experiment plan before coding | The plan is one or two lines |
| `py-architect` | Designing a sizeable change: a new decomposition stage, a pipeline boundary, a kernel interface | A single-function change |
| `py-arch-reviewer` | The user asks to validate an architecture or design decision before implementation | No design exists yet |
| `py-developer` | Bulk code writing across several modules, or a change that would drain the parent's context | A small local edit (quick-fix path) |
| `py-ml-researcher` | Literature and method work: surveying SOTA, checking whether an approach is already published, turning a method paper into an implementable algorithm sketch | Writing production code (that is `py-developer`) |
| `py-refactoring` | Dead-code removal, consolidation, deduplication across modules | A refactor local to one function |
| `py-performance-optimizer` | A measured slowdown, or vectorization/kernel optimization is the explicit task | No performance concern was raised |
| `py-error-fixer` | Fixing failing tests, exceptions, lint/type errors with no architectural change | The fix requires redesign — escalate to `py-architect`/`py-developer` |
| `py-tester` | Writing and running tests for a change, and reporting results with evidence | Purely static work; a change with no executable behaviour |
| `py-code-reviewer` | **Only when the user explicitly asks for a code review** | Auto-triggering after edits is forbidden |
| `py-doc-writer` | User-facing documentation: method notes, experiment reports, codemaps, API references | Inline docstrings — that is the developer's responsibility |

## Bounded task templates

Every delegation prompt states: bounded responsibility (one verifiable goal), allowed and forbidden sources, read/write scope, expected report format, and that the subagent is not alone in the codebase.

```text
Read-only reconnaissance. Find <what>. Do not edit files.
Return: locations with file:line, symbol names, how each is used, risky dependencies, and gaps you could not verify.
```

```text
Bounded implementation. You are not alone in the repository; do not revert or overwrite edits outside your scope.
Edit only: <files>. Implement <change> per <artifact/task id>.
Run the gates from .kilo/rules/verification-gates.md on every touched module (ruff, ruff format --check, mypy, pytest).
Return: changed files, what changed against the plan, gates run with results, unresolved risks.
```

```text
Independent review of the current change for bugs, numerical errors, reproducibility gaps and rule violations.
Scope: <diff or explicit file list>. Do not edit files. High-confidence findings only, ordered by severity,
with file:line; then test gaps and residual risk.
```

```text
Experiment report. Run <experiment> with config <path>, seeds from the config.
Record results under results/<experiment-id>/ per .kilo/rules/research-discipline.md.
Return: the recorded metrics with the configuration and slice they belong to, the runtime, the verdict against the gate,
and anything that makes the run non-reproducible.
```
