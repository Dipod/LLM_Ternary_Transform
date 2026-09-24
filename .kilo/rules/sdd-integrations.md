# Spec-driven development (OpenSpec)

Load this file before reading or changing anything under `openspec/`.

## Workspace

```text
openspec/
├── project.md                    # project context (hand-maintained here)
├── config.yaml                   # OpenSpec project settings
├── specs/<capability>/spec.md    # current behaviour, one file per capability
└── changes/<change-id>/
    ├── proposal.md               # why and what
    ├── design.md                 # how, decisions, risks
    ├── tasks.md                  # implementation checklist
    └── specs/<capability>/spec.md  # delta: ADDED / MODIFIED / REMOVED / RENAMED
openspec/changes/archive/<date>-<change-id>/   # archived changes
```

The CLI is installed globally (`openspec`); `openspec doctor` verifies the workspace root, `openspec list` shows changes, `openspec status --change <id>` shows artifact progress, `openspec validate <id>` checks a change.

## Workflow

1. **Propose** (`/opsx:propose <idea>`) — the propose phase owns every architectural decision. Ask now; do not defer a decision to apply time. Required artifacts: `proposal.md`, delta `specs/`, `design.md`, `tasks.md`; `openspec validate` must pass.
2. **Apply** (`/opsx:apply`) — one consolidated preflight round at the start (genuine blockers only), then near-silence. No mid-loop questions except a live-state fact that contradicts a locked artifact decision.
3. **Archive** (`/opsx:archive`) — merge the deltas into `openspec/specs/`, move the change to `archive/`.

`/opsx:explore` is for thinking without artifacts.

## Delta format

- `## ADDED Requirements` / `## MODIFIED Requirements` / `## REMOVED Requirements` / `## RENAMED Requirements`.
- A new capability's delta opens with `## Purpose` (one paragraph, ≥ 50 characters).
- Each requirement: `### Requirement: <name>` with `SHALL`/`MUST` wording; each has **at least one** `#### Scenario:` with `- **WHEN**` / `- **THEN**` lines.
- `MODIFIED` carries the full updated requirement text, not a diff.

Forbidden in finalized artifacts: `TODO: clarify during apply`, "decide later", vague verbs ("appropriately", "if needed"), two equally weighted options without a stated default. Every one of these is a propose-phase defect.

## Facts before writing

This repository has no MCP knowledge servers: verify a technical claim before it becomes a requirement, a threshold or a task. Verify by running a minimal snippet, reading the upstream documentation, or citing an experiment id from `results/`. Non-trivial artifacts end with a `## Context sources` block naming what was checked (library + version, doc page, experiment id) and what was deliberately not checked.

Research-specific:

- Gate thresholds, metric definitions and tolerances belong in the delta spec, not only in prose elsewhere; a spec that cites a number from literature names the source.
- A spec that depends on a finished experiment cites its `results/<experiment-id>/` and the verdict.
- A change that invalidates a recorded result says which result and how it will be re-verified.

## Artifact ownership

| Artifact | Owner |
|---|---|
| `proposal.md`, delta `specs/**` | `py-analytic` |
| `design.md` | `py-architect` |
| `tasks.md` | `py-planner` |
| Experiment results, `tasks.md` ticks after a run | `py-tester` / `py-ml-researcher` |
| Read-only reconnaissance for any of the above | `py-explorer` |

## After implementation

- Map every requirement to the test or experiment that verifies it; unmapped requirements are declared explicitly, never left silent.
- Tick `tasks.md` only after the gates for that task passed (`.kilo/rules/verification-gates.md`).
- A change to a locked decision found during apply updates the artifact in the same session and is reported as an artifact change.
