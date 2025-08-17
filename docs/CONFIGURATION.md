# FBA-Bench Configuration Guide

This document describes how to configure core runtime parameters for deterministic, cost-aware simulations, including constraint tiers and budget overrides. It also documents supported CLI flags and environment variables.

## Constraint Tiers (T0–T3)

Constraint tiers control token budgets and overall metering for agent runs. The engine selects a BudgetEnforcer from either explicit overrides or a tier preset.

- Engine wiring: [`BenchmarkEngine.__init__()`](benchmarking/core/engine.py:1)
- Tier enum and config: [`Tier`, `BenchmarkConfig`](benchmarking/config/pydantic_config.py:1)
- Budget enforcer factory: [`BudgetEnforcer.from_tier_config()`](constraints/budget_enforcer.py:1)

Supported tiers:
- T0: minimal cost ceiling for fast CI/iterative tests
- T1: moderate evaluation ceiling
- T2: extended experiments
- T3: large-scale or stress evaluation

## Budget Overrides

You can override the tier preset with explicit budget configuration. Budget overrides are validated and consumed by the engine. Use this when you need fine-grained budgets differing from preset tiers.

- Engine selection logic (tier vs overrides): [`BenchmarkEngine`](benchmarking/core/engine.py:1)
- Pydantic validation for overrides: [`BenchmarkConfig.model_validator`](benchmarking/config/pydantic_config.py:1)

Example JSON payload (file or inline via CLI):
```json
{
  "max_tokens_per_action": 16000,
  "max_total_tokens": 500000,
  "token_cost_per_1k": 0.002,
  "hard_fail_on_violation": false,
  "inject_budget_status": true,
  "track_token_efficiency": true
}
```

## Environment Variables

These environment variables can be used to override values from configuration files. They are read and validated by the Pydantic config loader.

- `FBA_TIER`: One of `T0`, `T1`, `T2`, `T3`
- `FBA_BUDGET_OVERRIDES`: Either an absolute/relative path to a JSON file or raw JSON content

The config loader reads these in [`BenchmarkConfig`](benchmarking/config/pydantic_config.py:1) during initialization.

## CLI Flags

The Experiment CLI wires tier and budget overrides into the environment before loading configuration. This ensures a single source of truth and consistent behavior across entry points.

- CLI definition: [`experiment_cli` run subcommand](experiment_cli.py:247)

Flags:
- `--tier {T0,T1,T2,T3}`: Select a constraint tier
- `--budget-overrides <path-or-json>`: Provide a JSON file path or inline JSON string for explicit budget overrides

Examples:
```bash
# Run with Tier 1 constraints (preset)
python experiment_cli.py run --tier T1 --scenario "Tier 1 Moderate" --agents GPT4oMiniBot

# Run with explicit overrides from JSON file
python experiment_cli.py run --budget-overrides config/budget_overrides.json --scenario "Tier 0 Baseline" --agents ClaudeSonnetBot

# Run with inline JSON overrides (quotes required)
python experiment_cli.py run --budget-overrides '{"max_tokens_per_action": 20000, "max_total_tokens": 600000}' --scenario "Tier 2 Stress" --agents Grok4Bot
```

Validation behavior:
- If both `--tier` and `--budget-overrides` are provided, overrides take precedence.
- Invalid JSON values or file read errors will cause the CLI to exit with a non-zero status (see [`experiment_cli`](experiment_cli.py:302)).

## Determinism and Audit Artifacts

For deterministic runs and golden snapshots:
- Audit run structure: [`RunAudit`](audit.py:64)
- Golden snapshot helpers and cache: see audit v2025.1 improvements in [`audit.py`](audit.py:1)

The engine computes config/code/git/fee hashes for each run, enabling reproducibility checks and cache hits. See:
- Hash generation and cache: [`run_and_audit`](audit.py:81)

## Minimal Observability Hooks (Baseline)

While the advanced TraceAnalyzer module is pending, the following baseline hooks are recommended:
- Log budget warnings/exceeded events from `BudgetEnforcer` (already integrated)
- Log agent-runner exceptions with context (agent id, tick number)
- Emit a final run summary with token usage and constraint violations

Operational guidelines:
- Use your process supervisor or container logs to scrape WARN/ERROR for budget or exception alerts.
- For production pipelines, ensure the CLI exit code is surfaced to deployment/test runners.

## Release Checklist Tie-In

Before tagging a release:
- Verify invariants pass in CI (see workflow): [`.github/workflows/ci.yml`](.github/workflows/ci.yml:1)
- Confirm tier/budget documentation (this file) is up-to-date
- Ensure plugin safety validation uses the real validator when available:
  - Production tooling: [`ProductionContributionManager`](community/contribution_tools_production.py:45)
  - Plugin security validation: [`PluginManager.validate_plugin_security`](plugins/plugin_framework.py:152)

## FAQ

- Q: What if I pass an invalid `--budget-overrides` JSON?
  - The CLI will log an error and return exit code 2 (see [`experiment_cli`](experiment_cli.py:302)).

- Q: Which source wins if I set both env vars and pass CLI flags?
  - The CLI writes flags into env variables before configuration load; the unified source is effectively the CLI-provided values.

- Q: Can I mix tier presets with partial overrides?
  - Prefer explicit overrides for clarity. If both are provided, the engine treats overrides as authoritative.
