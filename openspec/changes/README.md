# Changes

Each change lives in its own directory: `changes/<change-id>/`.

```text
changes/<change-id>/
├── proposal.md                     # why, what, impact, context sources
├── design.md                       # how: decisions with alternatives, risks, migration
├── tasks.md                        # implementation checklist (- [ ] / - [x])
└── specs/<capability>/spec.md      # delta: ADDED / MODIFIED / REMOVED / RENAMED
```

Lifecycle: `/opsx:propose` → `/opsx:apply` → `/opsx:archive`.

- **Propose** settles every architectural decision — model, dataset slice, tolerances, gate thresholds, layout. Artifacts contain no `TODO: clarify during apply`.
- **Apply** implements the tasks; one consolidated preflight question round at the start, then near-silence.
- **Archive** merges the deltas into `../specs/`, moves the directory to `changes/archive/<YYYY-MM-DD>-<change-id>/`, and leaves the spec as the current contract.

Validation: `openspec validate <change-id>` must pass before apply, and every requirement must be matched by a test, an experiment or a declared exception before archiving.

No active changes yet.
