# OpenSpec workspace

Spec-driven development workspace for this repository. The normative rule is `.kilo/rules/sdd-integrations.md`; this file is the map.

```text
openspec/
├── config.yaml                    # project name and description for the CLI
├── project.md                     # project context (hand-maintained here)
├── specs/<capability>/spec.md     # current behaviour, one file per capability
└── changes/<change-id>/
    ├── proposal.md                # why and what
    ├── design.md                  # how: decisions, alternatives, risks
    ├── tasks.md                   # implementation checklist
    └── specs/<capability>/spec.md # delta: ADDED / MODIFIED / REMOVED / RENAMED
openspec/changes/archive/<date>-<change-id>/
```

## Commands

| Command | Purpose |
|---|---|
| `/opsx:propose <idea>` | create a change with all artifacts |
| `/opsx:apply [change]` | implement the tasks of a change |
| `/opsx:archive [change]` | merge the deltas into `specs/` and move the change to `archive/` |
| `/opsx:explore` | think through an idea without creating artifacts |

CLI: `openspec doctor` (workspace root), `openspec list` (changes), `openspec status --change <id>`, `openspec validate <id>`.

## Rules

- Do not create a change for prose-only edits or a one-file fix that changes no behaviour (`AGENTS.md → Triage`).
- A requirement needs `SHALL`/`MUST` wording and at least one `#### Scenario:`; a new capability's delta opens with `## Purpose`.
- Facts behind a requirement are verified before writing (a run, an experiment id, or the upstream docs) and listed in the artifact's `## Context sources` block.
- Gate thresholds and metric definitions live in the delta spec, not only in prose.
- This workspace is hand-maintained: `project.md` is edited by hand when the project context changes (there is no installer in this repository).
