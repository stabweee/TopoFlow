# AGENTS.md — TopoFlow Operating Guidelines

## 1. Project Overview & Tech Stack

**TopoFlow** models vertical sag curve anomalies and traffic phase breakdown at the I-495 American Legion Bridge (ALMB) Potomac River Crossing (-3.2% descent, +3.8% climb) using empirical transport physics and causal econometrics.

- **Runtime & Package Manager**: Python 3.12+, `uv` (`uv_build`).
- **Data & Storage**: `polars`, `duckdb`, `pyarrow` (Parquet data marts).
- **Geospatial & 3D LiDAR**: `geopandas`, `shapely`, `rasterio`, `pyproj` (UTM 18N), `laspy[lazrs]` (USGS 3DEP LiDAR).
- **Econometrics & Modeling**: `statsmodels`, `linearmodels`, `scipy`, `numpy`, `causalimpact`.
- **Validation & Tooling**: `pydantic` (v2), `pyyaml`, `pytest`, `pytest-cov`, `ruff`.

---

## 2. Executable Developer Commands

All commands run via `uv`:

```bash
# Dependencies
uv sync
uv add <pkg> / uv add --dev <pkg>

# Linting & Formatting
uv run ruff check . [--fix]
uv run ruff format . [--check]

# Type Checking
uv run --with mypy mypy src tests

# Testing
uv run pytest -v                                        # Full suite
uv run pytest tests/unit -v                             # Unit tests
uv run pytest tests/integration -v                      # Integration tests
uv run pytest --cov=src/topoflow --cov-report=term-missing # Coverage
```

---

## 3. Boundaries & Constraints

- **Package Management**: Use `uv` only; never invoke global `pip`.
- **CRS Discipline**: All metric calculations, grades, and buffers **must** use **UTM 18N (EPSG:26918)**. WGS84 (`EPSG:4326`) is strictly for data ingestion and export.
- **Data Immutability & Vectorization**: `data/raw/` is read-only. Store marts as compressed Parquet. Never iterate rows (`iterrows`); use vectorized Polars/DuckDB operations.
- **Security & Isolation**: No hardcoded credentials. Tests must run against synthetic mocks in `tests/fixtures/`, never against live network or production data.

---

## 4. Mandatory Pre-Change Protocol: Plan & Permission

> **CRITICAL**: Never write or modify code without prior plan approval.

For any task requiring code alterations:

1. **Analyze**: Inspect impacted files, dependencies, and schemas.
2. **Plan**: Formulate a concise plan with scope, files affected, design/signatures, edge cases, and verification commands. Break down large, complex problems into simpler individual phases that can be implemented one by one with higher confidence and accuracy.
3. **Request Permission**: Pause and ask: _"Would you like me to proceed with these changes as planned?"_ Do not proceed until approved.

---

## 5. Post-Change Protocol: Verification

Immediately after approved code modifications, run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest -v
```

Fix all failures and re-run until 100% green before proceeding.

---

## 6. Code Reviewer Subagent Protocol

After verification passes:

1. Spawn a subagent (`invoke_subagent`, Role: `"Code Reviewer"`) to evaluate `git diff`.
2. Inspect for scope conformance, edge cases (zero-division, null propagation, CRS alignment), vectorization/performance, test rigor, and typing.
3. If issues are found, resolve them, re-run verification, and re-review until approved.

---

## 7. Definition of Done (DoD)

A task is **Done** only when:

- [ ] Plan approved by user.
- [ ] Implementation complete without TODO stubs.
- [ ] `uv run ruff check .` passes (0 errors/warnings).
- [ ] `uv run ruff format --check .` passes.
- [ ] `uv run pytest -v` passes (100% green).
- [ ] New code covered by unit/integration tests.
- [ ] Code Reviewer subagent diff review clean and approved.
- [ ] Relevant documentation updated.
- [ ] `AGENT_STATE.md` updated.

---

## 8. Session Continuity (`AGENT_STATE.md`)

Maintain `AGENT_STATE.md` in the root directory across all milestones and handovers. It must record completed items, test health, failing tests/blockers, exact next actions, and technical context.

---

## 9. Git Operations & Permission Protocol

> **CRITICAL**: Never stage (`git add`), commit (`git commit`), or push (`git push`) without explicit user permission.

Before running any git write command:
1. Present the target files, command, and proposed commit message (if committing).
2. Pause and ask: _"Would you like me to stage/commit/push these changes?"_
3. Do not execute until explicitly approved by the user.

