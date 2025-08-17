# FBA-Bench v0.9.0-rc.1

This release candidate focuses on determinism, safety, CI hardening, and configuration UX. It removes reliance on deprecated agent runner factories at runtime surfaces, adds baseline plugin security validation, and documents tier/budget configuration.

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
  - Production tooling now leverages the real PluginManager security validator when available; falls back to a safe, explicit no-op with warning if unavailable.
  - File: `community/contribution_tools_production.py`

- CI hardening:
  - GitHub Actions workflow runs invariants on Linux and Windows and includes a production CLI smoke test when present.
  - File: `.github/workflows/ci.yml`

- Agent runner deprecation handling:
  - Adapter no longer imports legacy RunnerFactory directly.
  - IntegrationManager gracefully handles ImportError from deprecated factories and degrades without breaking.
  - Files: `benchmarking/integration/agent_adapter.py`, `benchmarking/integration/manager.py`

## Files Changed (non-exhaustive)

- CI:
  - `.github/workflows/ci.yml`

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
- Production CLI smoke: `pytest -q tests/test_experiment_cli_production.py` (optional; runs when present)

## Known Deferrals to GA

- Learning stack (PPO + simulator) remains in-progress for the next release.
- Advanced Observability UI (baseline hooks are documented and recommended).
- Larger-scale integration/performance realism.
- Refactor out remaining legacy RunnerFactory references in integration tests (runtime surfaces are already resilient).

## Changelog (condensed)

- feat(ci): add GitHub Actions with invariants + CLI smoke
- feat(security): production contribution manager uses real PluginManager validator when available
- docs: add tier/budget CLI/env configuration guide with determinism and audit overview
- fix(integration): AgentAdapter no longer imports RunnerFactory directly
- fix(integration): IntegrationManager gracefully handles ImportError from deprecated factory paths
- test: deterministic sim fixtures ensure invariants pass and RNG isolation holds
