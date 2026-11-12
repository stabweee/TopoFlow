# AGENTS.md — TopoFlow Operating Guidelines

## 1. Project Overview & Tech Stack

TopoFlow models vertical sag curve anomalies and traffic phase breakdown at I-495 American Legion Bridge (-3.2% descent, +3.8% climb) using empirical transport physics and causal econometrics.

- **Stack**: Python 3.12+ (`uv`), `polars`, `duckdb`, `geopandas`, `shapely`, `pyproj` (UTM 18N), `laspy`, `statsmodels`, `linearmodels`, `scipy`, `pydantic` v2, `pytest`, `ruff`.

## 2. Developer Commands (via `uv`)

- Dependencies: `uv sync` | `uv add <pkg> [--dev]`
- Lint & Format: `uv run ruff check . [--fix]` | `uv run ruff format . [--check]`
- Tests & Typing: `uv run pytest -v` | `uv run --with mypy mypy src tests`

## 3. Boundaries & Constraints

- **Package Management**: Use `uv` only; never invoke global `pip`.
- **CRS Discipline**: All metric calculations, grades, and buffers **must** use **UTM 18N (EPSG:26918)**. `EPSG:4326` is strictly for ingestion/export.
- **Data & Storage**: `data/raw/` is read-only. Store marts as compressed Parquet. Vectorized Polars/DuckDB operations only (no row iteration).
- **Security & Isolation**: No hardcoded credentials. Tests must use synthetic mocks in `tests/fixtures/`, never live network or production data.

## 4. Mandatory Pre-Change Protocol: Plan & Permission

> **CRITICAL**: Never write or modify code without prior plan approval.

1. **Analyze**: Inspect files, dependencies, and schemas.
2. **Plan**: Formulate a concise plan with scope, files, signatures, edge cases, and verification commands. Break complex problems into smaller phases.
3. **Request Permission**: Pause and ask: _"Would you like me to proceed with these changes as planned?"_ Do not proceed until approved.

## 5. Clean-Room Test Author Subagent Protocol

Never write tests in primary thread. To prevent confirmation bias and context bloat:

1. **Spawn Subagent**: Run `invoke_subagent(TypeName="self", Role="Test Author")`.
2. **Specification-Only Prompt**: Pass public signatures, functional expectations, target test file path, and domain invariants (EPSG:26918, mocks in `tests/fixtures/`, zero-division). Deliberately omit internal implementation code or reasoning.
3. **Handoff**: Subagent writes tests, runs `uv run pytest <test_path> -v`, and reports results.

## 6. Post-Change Protocol: Verification

Immediately after approved modifications, run:

```bash
uv run ruff check . && uv run ruff format --check . && uv run pytest -v
```

Fix all failures and re-run until 100% green before proceeding.

## 7. Milestone Visual Verification Gate

At each milestone (Phases 1-6), generate and visually inspect a domain-specific artifact (e.g., centerlines, sag profiles, space-time heatmaps) saved to `outputs/figures/` (e.g., `p1_corridor_centerlines.png`). Visually verify physical properties before sign-off.

## 8. Code Reviewer Subagent Protocol

After verification passes:

1. Spawn subagent (`invoke_subagent`, Role: `"Code Reviewer"`) to evaluate `git diff`.
2. Inspect scope, edge cases (zero-division, nulls, CRS), vectorization, test rigor, and typing. Resolve issues and re-verify until clean.

## 9. Definition of Done (DoD)

A task is Done only when:

- [ ] Plan approved by user.
- [ ] Implementation complete without TODO stubs.
- [ ] `uv run ruff check .` & `uv run ruff format --check .` & `uv run pytest -v` pass 100%.
- [ ] New code covered by clean-room tests (`Test Author` subagent).
- [ ] Visual verification artifact generated, inspected, and saved to `outputs/figures/`.
- [ ] Code Reviewer subagent diff review approved.
- [ ] Documentation & `AGENT_STATE.md` updated.

## 10. Session Continuity (`AGENT_STATE.md`)

Maintain `AGENT_STATE.md` in root across all milestones to record progress, test health, blockers, next actions, and technical context.

## 11. Git Operations & Permission Protocol

Before any git write command, present target files, command, and commit message; ask: _"Would you like me to stage/commit/push these changes?"_ Do not execute until explicitly approved.
