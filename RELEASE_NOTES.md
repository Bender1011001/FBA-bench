# FBA-Bench v3.0.0

This GA release focuses on determinism, safety, CI hardening, and configuration UX, with a production container entrypoint and aligned versioning.

## Highlights

- Deterministic invariants: Added global and test-local fixtures ensuring core invariants pass (5/5). Non-negative inventory and RNG isolation guaranteed.
  - Fixtures: `sim_factory` and `basic_simulation_seed_factory`
  - Files: `conftest.py`, `tests/conftest.py`

- Budget/Tier configuration UX:
  - Tiered BudgetEnforcer factory wired into engine via config/CLI/ENV.
  - Configuration documentation with CLI/env examples and determinism workflow.
  - Docs: `docs/CONFIGURATION.md`

- Audit system v2025.1:
  - External config discovery, baseline validation, code/git/fee hash computation, cache integrity checks.

- Plugin security baseline:
  - Production tooling leverages the real PluginManager security validator when available; falls back to a safe, explicit no-op with warning if unavailable.
  - File: `community/contribution_tools_production.py`

- CI hardening:
  - GitHub Actions workflow installs via Poetry and runs full backend + frontend test matrices on Linux and Windows, plus a Docker build-and-healthcheck job.
  - File: `.github/workflows/ci.yml`

- Containerization:
  - Production Dockerfile now runs FastAPI via uvicorn (`fba_bench_api.main:app`), installs dependencies from `pyproject.toml`, and provides a healthcheck.
  - File: `Dockerfile`

- Agent runner deprecation handling:
  - Adapter no longer imports legacy RunnerFactory directly.
  - IntegrationManager gracefully handles ImportError from deprecated factories and degrades without breaking.
  - Files: `benchmarking/integration/agent_adapter.py`, `benchmarking/integration/manager.py`

## Files Changed (non-exhaustive)

- CI:
  - `.github/workflows/ci.yml`

- Docker:
  - `Dockerfile`

- Docs:
  - `docs/CONFIGURATION.md`
  - `RELEASE_NOTES.md` (this file)

- Production security:
  - `community/contribution_tools_production.py`

- Deterministic invariants fixtures:
  - `conftest.py`
  - `tests/conftest.py`

- Integration surfaces:
  - `benchmarking/integration/manager.py`
  - `benchmarking/integration/agent_adapter.py`

## Upgrade Notes

- RunnerFactory and `unified_runner_factory.py` are deprecated shims and should not be used. Integrations and adapters should go through `IntegrationManager` or the `AgentManager` unified agent path.
- For tier and budget overrides:
  - CLI:
    - `--tier T0|T1|T2|T3`
    - `--budget-overrides <path-or-json>`
  - ENV:
    - `FBA_TIER=T1`
    - `FBA_BUDGET_OVERRIDES='{"max_total_tokens": 500000}'`
  - See: `docs/CONFIGURATION.md`

## CI Entry Points

- Invariants: `pytest -q tests/test_invariants.py`
- Production CLI smoke: `pytest -q tests/test_experiment_cli_production.py` (when present)
- Full suite: `pytest` (backend) and `cd frontend && npm test` (frontend)

## Known Deferrals

- Learning stack (PPO + simulator) remains in-progress for the next minor release.
- Advanced Observability UI (baseline hooks are documented and recommended).
- Larger-scale integration/performance realism.

## Changelog (condensed)

- feat(container): uvicorn entrypoint for FastAPI and Poetry-driven installs
- feat(ci): matrix tests + Docker healthcheck
- docs: configuration and determinism guidance
- fix(integration): remove legacy RunnerFactory runtime coupling
- test: deterministic sim fixtures ensure invariants pass and RNG isolation holds
