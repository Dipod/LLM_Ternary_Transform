# Specs

Current **behaviour contracts** of the project, one file per capability: `specs/<capability>/spec.md`.

- A spec describes what the system does now, not what someone plans to do.
- Specs are written and changed through a change in `../changes/`; archiving a change merges its deltas into the relevant spec here.
- Requirements use `SHALL`/`MUST`; each carries at least one `#### Scenario:`.
- Research specs may define metrics, tolerances and gate thresholds — those are behaviour contracts too, because a reinterpreted metric changes the verdict.

No capability is specified yet: the project is at plan stage, and the first change will add `specs/tcd-conversion/spec.md` (or the capability named by that change).
