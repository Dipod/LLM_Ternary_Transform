# LLM Rules

Agent-maintained behaviour rules for this project, written **only** by the `/evolve` command with per-entry user approval. Contract, precedence, capture discipline and entry format — `AGENTS.md → Rules self-improvement` and the `/evolve` command.

Last /evolve run: 2026-09-25

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

### R-003 — Wait for long-running work via scheduled wakeups, not blocking shell sleeps (2026-09-24)

- **Rule:** When a task must wait for a long-running process (model/dataset downloads, sweeps, builds, environment setup), do not hold the turn with a blocking shell sleep or a poll loop. Register a scheduled wakeup (`schedule_wakeup`; `cron_create` for recurring) or rely on a tracked background-process notification, then end the turn so the user can interact. A blocking wait is allowed only for short waits (up to roughly two minutes) where it is the cheaper option.
- **Scope:** Any wait for a process expected to exceed roughly two minutes, in this project's sessions.
- **Evidence:** 2026-09-24 — repeated `Start-Sleep` loops held the session during the Phase 0 sweep and the ROCm setup; two user messages queued until a sleep ended, and the user asked to switch to scheduled Kilo actions.
- **Refines:** `AGENTS.md → Project environment → Long-running work`; complements R-001 (which governs questions while background work is in flight).

### R-004 — Adapt user-facing instructions to the actual host and do the setup instead of describing it (2026-09-24)

- **Rule:** When telling the user how to do something, target the host they are actually in and give concrete, copy-pasteable steps: absolute paths, exact file contents, and create the file or template yourself when a placeholder is needed. Never reference UI that exists only in another host (for example a CLI/TUI sidebar while the user works in the VS Code extension), and never hand over a bare CLI command without explaining it or performing it. When the progress of a long step matters to the user, mirror it to a log file they can open.
- **Scope:** Every instruction, path or UI reference addressed to the user in this project's sessions.
- **Evidence:** 2026-09-24 — (1) the HF-token answer gave `hf auth login`, and the user replied "Не понимаю как это сделать. Напиши абсолютный путь к файлу… создай его и предзаполни"; (2) the agent pointed a VS Code user at the CLI background-process sidebar, which the extension does not have.
- **Refines:** `.kilo/skills/powershell-windows/SKILL.md` (host-specific command presentation) and `AGENTS.md → Deliver Clearly`.

### R-005 — Compare against the strongest relevant baseline at matched budget, and sanity-check it (2026-09-25)

- **Rule:** Before concluding that a method wins or loses, compare it against the strongest relevant baseline for the same task and hardware at a matched budget (equal bits, FLOPs or memory) — not against a convenient weaker one (for example FP16 against a 4-bit competitor). Validate the baseline implementation itself with a known-answer sanity check (a clearly higher-precision setting) before drawing a verdict, and pre-register the comparison and its decision rule. Set any numeric sanity tolerance from the measured behaviour of the reference implementation rather than from intuition: measure the reference arm first (for example 8-bit quantization), choose a tolerance it demonstrably meets, and record both numbers. A tolerance that voids a legitimate run is itself a defect, not a finding.
- **Scope:** Any quality or performance comparison used to justify continuing or stopping a research direction in this project, and any numeric tolerance pre-registered for such a comparison.
- **Evidence:** 2026-09-24/25 — (1) the Phase 0 gate compared against FP16; the user challenged it ("4 bits already give ~99% … maybe the project is pointless"), which forced the matched-bit comparison; (2) the first hand-rolled 4-bit baseline used naive absmax and gave E_x 0.084, nearly supporting the wrong "ternary wins" conclusion, until MSE-optimal clipping and an 8-bit sanity check (E_x 0.0045) made the comparison fair — after which integer quantization dominated; (3) 2026-09-25 — the Stage A sanity tolerance (+0.1 % perplexity) proved unreachable by a genuine 8-bit group-128 RTN arm (+0.153 % absmax, +0.167 % with the clip search), which voided the whole stage under its own spec R4; the user's resolution was "сначала сильный baseline" — establish the reference arm, then set the tolerance.
- **Refines:** `.kilo/rules/research-discipline.md` (no cherry-picking, fixed gates) and `AGENTS.md → Development Procedure → 4. Goal-Driven Verification`.

### R-006 — For a sizeable or new direction, present the plan and an effort estimate first (2026-09-25)

- **Rule:** Before starting a sizeable or newly chosen work direction (a new method, an algorithm change, a multi-step experiment), present the concrete plan and an effort/cost estimate and wait for the user's confirmation. Do not jump from a chosen option straight into implementation.
- **Scope:** Any work direction beyond a bounded quick-fix, after the user picks an option and before edits or runs.
- **Evidence:** 2026-09-24 — the user had to ask twice: "Выбираю A, но сначала подробнее объясни что мы будем делать и оцени трудоёмкость" and "Сначала обсудить A2".
- **Refines:** `AGENTS.md → Development Procedure → 1. Think Before Coding — Clarify Scope First`.

### R-007 — Spend a user-granted autonomous-work time budget (2026-09-25)

- **Rule:** Spend a period of autonomous work only when the user has explicitly granted such a budget — an unavailability window with a delivery point ("I am asleep until 09:00", "you have the off-peak window until the deadline"), or an equivalent explicit statement that a budget of autonomous work time is allocated. When that grant exists, spend the budget instead of stopping at the first natural checkpoint: continue the next decision-free, evidence-producing step until the budget is exhausted or the agreed delivery point arrives, and deliver the summary at that point rather than earlier. Without an explicit grant, do not extend work beyond the user's request. The grant never widens the mandate: do not start work that requires user approval, do not relax a gate, a CONFUSION requirement or a verification step, and do not exceed the granted scope; if every remaining step is gated on a user decision, say so in the summary instead of inventing work.
- **Scope:** Sessions in which the user has explicitly stated an autonomous-work budget; the rule does not apply in the absence of that statement.
- **Evidence:** 2026-09-25 — (1) the user granted the night: "Я ухожу спать… сделай максимум что можешь до закрытия off-peak окна… напиши сводку"; (2) the Stage A summary was delivered at ~01:35 UTC with the 04:00–06:00 UTC window unused, and the user corrected: "off-peak с 7 до 9 утра тоже можешь использовать… отчёт мне нужен только к 9:00"; (3) the user first rejected this rule at the `/evolve` approval round ("Отклонить") and then reinstated it narrowed to require the explicit grant.
- **Refines:** R-002 (off-peak windows) and R-006 (plan and estimate before starting).

## Superseded
