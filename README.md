# TopoFlow: Spatio-Temporal Topographical Friction & Causal Impact Modeling

> Empirical transport physics, causal econometrics, and counterfactual eco-routing simulation analyzing the I-495 American Legion Bridge Potomac River sag bottleneck.

---

## Documentation Navigation

* **[System Overview](docs/system_overview.md)**: Comprehensive, end-to-end technical overview explaining how TopoFlow works from raw data ingestion to causal econometrics and simulation.
* **[Decision Framework](docs/decision_framework.md)**: Exhaustive catalog justifying all structural, data engineering, statistical, and simulation decisions across the project lifecycle.
* **[Domain Foundations](docs/domain_foundations.md)**: Detailed background on traffic physics, network topology, and transport economics.
* **[Engineering Tickets & Backlog](docs/tickets.md)**: Phase-by-phase implementation issues, deliverables, and dependency sequencing matrix.

---

## Quickstart

```bash
# Clone the repository
git clone https://github.com/stabweee/topoflow.git
cd topoflow

# Install dependencies using uv
uv sync

# Run tests
uv run pytest -v

# Run linter
uv run ruff check .
```
