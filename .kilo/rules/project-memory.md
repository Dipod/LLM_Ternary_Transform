# Project memory

Load this file on non-trivial tasks and on every user correction.

## Stores

1. **Harness project memory** — the client's own project-memory tools, when exposed in the session. Prefer it for short, structured facts and for `rule-friction:` notes.
2. **`memory.md`** (repository root) — the strict long-term store and the dated fallback when no provider can save. Entries are dated, one fact per line, no secrets, no PII, no dataset or model excerpts.

`USER-RULES.md` and `memory.md` outrank `LLM-RULES.md`, which outranks `AGENTS.md` and the on-demand rules.

## Recall first

Before designing anything, search project memory for: the same area of work, prior decisions about metrics, thresholds, datasets or model choice, and known pitfalls. A decision that contradicts saved memory is surfaced to the user, not silently overridden. Current repository state and the user's message outrank saved memory; when they conflict, say so in one line and treat the fresher source as authoritative.

## Correction capture

When the user corrects the agent — a naming rule, a method preference, a review remark, a process expectation — save it **in the same turn**:

- a fact or decision about the project goes to project memory under a stable key;
- a behaviour expectation ("always use X instead of Y") becomes a `rule-friction:` note, which `/evolve` later turns into an `LLM-RULES.md` entry with the user's approval.

A correction that only lives in the chat is lost at the next session boundary; treat capture as part of the task, not as a follow-up.

## What belongs here

Write: decisions with lasting effect (why a method, metric, dataset, layout or threshold was chosen); environment facts (interpreter, hardware, cache locations, how a heavy dependency is installed); conventions that are not written in the repository yet; the current phase and its gate status; corrections.

Do not write: transient task state that the repository already records; anything derivable by reading the repo; secrets, tokens, credentials; dataset or model excerpts; personal data.

## `rule-friction:` notes

Format: `rule-friction:<area>.<short-topic> — what happened, what behaviour is expected, date`. One note per behaviour; `/evolve note <text>` writes exactly one. `/evolve` clusters these into proposals for `LLM-RULES.md`, and only the user's approval writes that file.
