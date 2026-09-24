---
description: "Check that this repository's ruleset, agent definitions, OpenSpec workspace and Python tooling are present and consistent. Read-only."
---

# doctor — repository readiness check

Read-only. Report findings grouped by section; never edit files from this command.

## 1. Ruleset files

| Path | Expected |
|---|---|
| `AGENTS.md` | present, references `.kilo/rules/*.md` paths that exist |
| `USER-RULES.md`, `memory.md`, `LLM-RULES.md` | present |
| `.dev.env` | present, contains `CAVEMAN` |
| `.kilo/kilo.json` | valid JSON |
| `.kilo/rules/` | `verification-gates.md`, `research-discipline.md`, `python-standards.md`, `sdd-integrations.md`, `subagents.md`, `project-memory.md` |
| `.kilo/agents/` | 13 `py-*.md` files, each with `name`, `description`, `mode` in the frontmatter |
| `.kilo/commands/` | `opsx-propose.md`, `opsx-apply.md`, `opsx-archive.md`, `opsx-explore.md`, `evolve.md`, `caveman.md` |
| `.kilo/skills/` | `caveman`, `handoff`, `mermaid-diagrams`, `powershell-windows`, each with `SKILL.md` |

Report every missing file and every reference in `AGENTS.md` that points at a path that does not exist. Do not report a missing `pyproject.toml`, `tests/` or `.github/workflows/` as a defect: those arrive with the first code change.

## 2. OpenSpec

- `openspec/config.yaml` parses as a YAML object (not a file of comments only).
- `openspec/project.md`, `openspec/specs/README.md`, `openspec/changes/README.md` exist.
- CLI: `openspec doctor` reports the workspace root as ok; `openspec list` runs without error.
- For every change under `openspec/changes/` (excluding `archive/`): `openspec status --change <id>` and `openspec validate <id>`.

## 3. Python environment (informational, never a failure)

- `python --version` (expect 3.10+), and whether a virtual environment is active.
- Availability of `ruff`, `mypy`, `pytest` (`--version` for each).
- Absent tooling is reported as `SKIP` with the install command, not as a defect: the gates only apply to code that exists.

## 4. Git state

- Branch and remote (`git status -sb`, `git remote -v`).
- Uncommitted changes (`git status --short`) — list them, do not commit.

## 5. Report

```text
Section: OK / ISSUES (n)
  - <path or check>: <what is wrong> — <the fix, one line>
Skipped: <checks that do not apply yet>
```

End with the single most important next action. Never modify anything while checking.
