# LLM Rules

Agent-maintained behaviour rules for this project, written **only** by the `/evolve` command with per-entry user approval. Contract, precedence, capture discipline and entry format — `AGENTS.md → Rules self-improvement` and the `/evolve` command.

Last /evolve run: 2026-09-24

## Active rules

### R-001 — Do not mix background work with a user question (2026-09-24)

- **Rule:** Never ask the user a question while a background task or other long-running signal is in flight: finish or wait for all of them first, then ask in a dedicated round. Treat a question that returns `dismissed` without an answer as an interrupted signal, not as a user refusal or an answer, and re-ask once no pending signals remain.
- **Scope:** Every use of the question tool in this project's sessions, before launching any background subagent and for the rest of the session.
- **Evidence:** 2026-09-24 — two question rounds (goal confirmation; the six-decision batch) were auto-closed by subagent completions and misread by the agent as refusals; the user corrected the interpretation.
- **Refines:** none directly (refines the clarification flow of `AGENTS.md → Development Procedure` and `.kilo/rules/subagents.md` delegation).

### R-002 — Plan and execute user-requested work in DeepSeek off-peak MSK (2026-09-24)

- **Rule:** The agent runs on a DeepSeek model, so its inference is billed at DeepSeek rates. Plan and execute all user-requested work during off-peak MSK even when the request arrives during peak. Peak (avoid): Mon–Fri 04:00–07:00 and 09:00–13:00 MSK. Off-peak (use): Mon–Fri 00:00–04:00, 07:00–09:00, 13:00–24:00 MSK; all weekend; Chinese public holidays. A request received during peak is answered with a short plan and scheduled for the next off-peak window; heavy work (docs study, spec authoring, calibration runs, sweeps and other long DeepSeek-billed tool chains) must not start during peak. Work that consumes no DeepSeek tokens (model and dataset downloads, filesystem operations, the local gate toolchain) may run during peak.
- **Scope:** All work in this repository's sessions that bills DeepSeek inference.
- **Evidence:** 2026-09-24 — the user asked whether an off-peak rule exists and directed copying it from the sibling repository `personal cabinet` (`AGENTS.md` §"Scheduling policy — DeepSeek off-peak only"); an explicit permanent-change request, so a single episode meets the threshold.
- **Refines:** `.kilo/rules/research-discipline.md → Cost discipline` (runtime estimate and explicit go-ahead for long runs) and `AGENTS.md → Project environment → Long-running work`.

## Superseded
