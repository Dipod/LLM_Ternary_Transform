---
name: handoff
description: "Compact the current conversation into a self-contained handoff document so a fresh agent (new chat, another machine, another AI client) can continue the work without re-discovering the context. References durable artifacts (`openspec/`, `memory.md`, commits, project-memory notes) instead of duplicating them. Use when the user says 'handoff', 'compact session', 'save context for continuation', 'brief the next session', 'сделай handoff', 'передай контекст', 'сохрани контекст для продолжения', or invokes `/handoff`."
argument-hint: "Optional: focus of the next session, or a target path/folder for the handoff file."
---

# handoff — session transfer to the next agent

Adapted from [`mattpocock/skills`](https://github.com/mattpocock/skills) (`skills/productivity/handoff`, MIT). Compresses the current conversation into a self-contained document for the next session. Principle: **reference durable artifacts, do not duplicate them**.

## Argument

- If the argument looks like a path (ends with `.md` or points to an existing directory) → **target path**.
- Otherwise → **focus** of the next session (insert it into the handoff header).
- If both are present, treat the first token as the path and the rest as focus.
- Without an argument, ask the user for the focus in one line and continue with the default path.

## Where to write

1. Default directory: `handoffs/` at the project root. Create it if missing.
2. Default file name: `handoff-<YYYYMMDD-HHMMSS>.md` (local time).
3. If the user provided a path, use it (overwrite without confirmation).
4. **Before writing**, read the target file through Read. The expected "does not exist" error is fine; this is protection against overwriting an unrelated existing file with the same name.
5. If the project has `.gitignore` and `handoffs/` is not mentioned there, **offer** to add it (handoffs are session artifacts, not code), but **do not add it automatically**.

PowerShell conventions (`\` in paths, quotes around paths with spaces) — see the `powershell-windows` skill.

## Document structure

```markdown
# Handoff: <one-line session goal>

**When**: <YYYY-MM-DD HH:MM local>
**Branch / commit**: <branch>, latest commit <short SHA + subject>
**Next session focus**: <argument focus, if provided>

## Current State
1-3 sentences: what was done last, what remains unfinished, what is blocked.

## Open Questions
Bulleted list of real unresolved questions (architectural forks, waiting for the user, unclear contract). If empty, omit the section.

## Files Changed In This Session
- `path/to/module.py` — what changed and why.
- `path/to/config.yaml` — same.
Only include the current session diff. If nothing changed, omit the section.

## Verification State
Which gates from `.kilo/rules/verification-gates.md` passed / failed / were skipped. Latest `ruff check` / `mypy` / `pytest` result in brief (finding counts, key messages). For an experiment, its `results/<experiment-id>/` and the verdict.

## Next Steps
1-5 imperative items ("Rerun the sweep for μ=3 with the reduced grid", "Add the known-answer test for `asymmetric_ternary_fit`").

## What To Load Next Session
- **Subagents**: `py-<name>` when the task matches their role (see `.kilo/rules/subagents.md`).
- **On-demand rules**: `<name>.md` based on the task trigger (see `AGENTS.md → Additional rules`).
- **Sources**: this repository has no MCP servers — the sources are the code, `openspec/`, `results/` and the upstream library documentation.
- **Slash commands**: `/opsx:apply` when there is an active OpenSpec change, `/opsx:propose` for new work, `/evolve` for behaviour rules.

## Links (DO NOT copy content)
- `openspec/changes/<id>/proposal.md`, `design.md`, `tasks.md`
- `results/<experiment-id>/report.md`
- `memory.md` — relevant sections
- Project-memory keys (harness memory) — `<key1>`, `<key2>`
- Commits / PR / issue
- Upstream documentation pages that were checked
```

## What NOT to write in the handoff

- Contents of existing artifacts (PRD, OpenSpec proposal/design/tasks, ADR, doc page, commit, PR description). Link only.
- Full module code. Only include a short change description and path.
- Secrets, tokens, passwords, `.dev.env` contents, connection strings.
- Long tool or log dumps. Include only the result and the parameters so the check can be repeated.

## After writing

1. Tell the user the absolute path of the created file and its line count.
2. If the session produced corrections / facts that may qualify for `memory.md` or a harness `remember` under `AGENTS.md → Project memory`, **list them separately** as candidates for long-term memory. Do not save automatically.

## Boundaries

- Handoff is a session artifact, not configuration and not code. Do not run `ruff` / `mypy` / `pytest` against it.
- Handoff **does not replace** an OpenSpec change. If the task requires one and it does not exist yet, additionally suggest `/opsx:propose` and reference the future ID from the handoff.
- Handoff **does not duplicate** `memory.md` and harness memory notes. Memory and handoff are different channels (see `AGENTS.md → Project memory`).
- Handoff is written in normal grammar, not caveman style, so the next agent can read it without ambiguity.
