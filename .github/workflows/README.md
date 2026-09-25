# Workflows — intentionally disabled

GitHub Actions is **disabled for this repository**. `ci.yml.disabled` is kept only as a
reference: GitHub loads workflows exclusively from `*.yml` / `*.yaml` files in this directory,
so renaming the file to `.disabled` stops every run.

Why it is disabled (2026-09-25):

- The workflow pinned **Python 3.11**, while the project floor is **3.12+** (raised by the
  pinned `numpy` `requires-python`), so it would fail on every push.
- Continuous integration adds nothing here: the gate chain (`ruff check`,
  `ruff format --check`, `mypy`, `pytest`) runs locally before every delivery, and the research
  runs need a GPU that hosted runners do not provide.

To re-enable: rename `ci.yml.disabled` back to `ci.yml` and fix the Python version pin first.

To disable the Actions feature for the whole repository (so that no workflow file, present or
future, can run), use the GitHub web UI: **Settings → Actions → General → "Disable actions"**.
That switch also survives a re-added workflow file. It requires repository-admin rights and
cannot be set from this checkout (`gh` is not installed here).
