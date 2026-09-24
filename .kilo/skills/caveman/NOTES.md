# caveman — maintainer notes

Companion of `SKILL.md` in this directory. Nothing here is a rule and nothing loads it at runtime: the skill file is the only text an agent reads. This file keeps what the audit found inert in the skill — the measurements behind the style, the deliberate deviations from upstream (so the next sync does not "fix" them), and the worked examples.

## Upstream

Adapted from https://github.com/JuliusBrussee/caveman (MIT), tracked against upstream **v2.2.0** (20.08.2026). Upstream's non-skill surface (proxy / engine / CLI / `caveman learn` / `/caveman-compress` / hooks, v2.x) is out of scope: this repo adapts the skill only, which upstream keeps MIT and unchanged by the v2 engine release.

## Honest numbers

The 65% is the upstream-measured average **output**-token reduction on chat prose against an unprompted baseline (range 22–87%). The style itself costs ~1–1.5k input tokens per turn, and on agentic coding runs the independently measured net effect is single-digit (JetBrains: 8.5% across 86 SkillsBench tasks, no detectable quality change). Use it for signal density, not as a token-budget lever.

## Upstream deviations (deliberate)

Kept against upstream v2.2.0 with a stated reason — do not "re-sync" them away on the next update:

- **Causal arrows stay allowed at `ultra`.** Upstream bans `→` because in English it costs its own token and saves nothing. In Russian it replaces a 2–3-token connector (`из-за чего`, `поэтому`), so the saving is real. Ambiguity is still governed by *Auto-clarity*.
- **Wenyan levels (`wenyan-lite` / `-full` / `-ultra`) are not ported** — the answer language is Russian per `AGENTS.md`; a classical-Chinese register has no use here.
- **Scope gating, `CAVEMAN` values and the boundary list are project-specific** — upstream has no `.dev.env`, no task-type classification and no Python / ML artifacts (experiment reports, configs, OpenSpec deltas).
- **The "Never drop" list is project-specific** — it enumerates the evidence lines this ruleset mandates in a delivery report (`Tests:` / `Lint:` / `Types:` / `Experiment:`, `Memory:`, gate-skip risk lines). Upstream has no such obligations; the list exists so that compression cannot be mistaken for permission to omit them.

## Worked examples

### Levels — "Почему своп считается медленно?"

- lite: «Своп считается медленно, потому что калибровочные активации пересчитываются внутри цикла по блокам. Вынести расчёт из цикла.»
- full: «Пересчёт активаций внутри цикла по блокам → G× лишних проходов. Вынести из цикла, кэшировать `X_block`.»
- ultra: «Пересчёт `X_block` в цикле по блокам → G× лишней работы. Кэшировать до цикла.»

### Core pattern — `[вещь] [действие] [причина]. [следующий шаг].`

Bad: «Скорее всего, проблема в том, что в функции декомпозиции остаток обновляется до вычисления масштаба, и это приводит к немонотонному убыванию нормы.»
Good: «Баг в `extrend_lite_decompose`: дефляция до оценки `d` → немонотонная норма. Считать `d` первым, потом вычитать.»

Bad: «Сначала, если вы не возражаете, я бы хотел уточнить, какой именно dtype используется при загрузке модели, чтобы корректно подобрать вариант реализации.»
Good: «Какой dtype при загрузке модели? От него зависит выбор реализации.»

## Removed on purpose

The former "Quick checklist before sending a reply" restated *Core rules* and *Boundaries* item by item and was dropped; the rules themselves are the checklist.
