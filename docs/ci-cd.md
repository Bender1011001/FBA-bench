# CI/CD Workflows

This repository includes two GitHub Actions workflows for backend CI and safe database migrations.

- Backend CI: runs tests on pushes and pull requests affecting backend or test code.
- Alembic Migrate: runs database migrations safely, manually or from other workflows.


## Backend CI

Workflow file: `.github/workflows/backend-ci.yml`

Triggers:
- Runs on `push` and `pull_request` when changes touch:
  - `fba_bench_api/**`
  - `tests/**`
  - `pyproject.toml`
  - `poetry.lock`
  - `.github/workflows/backend-ci.yml`

Key behavior:
- Python 3.11, Poetry 1.7.1
- In-project virtualenv (`.venv`) with caching keyed on `poetry.lock`
- Installs deps via `poetry install --no-interaction --no-ansi`
- Executes `pytest -q`
- Always uploads pytest output to `artifacts/test-output.txt` on failure


### Test environment variables (why they are set)

- `DATABASE_URL=sqlite+aiosqlite:///./test.db`
  - Uses SQLite with `aiosqlite` for fast, dependency-free async database tests.
- `DB_AUTO_CREATE="true"`
  - Allows tests to auto-initialize DB schema/tables if the app supports it.
- `AUTH_ENABLED="false"` and `AUTH_TEST_BYPASS="true"`
  - Disables authentication and enables test bypass, so integration tests can focus on API behavior.
- `API_RATE_LIMIT="100/minute"`
  - Reduces flakiness from rate limiting during CI load.
- `FBA_CORS_ALLOW_ORIGINS="*"`
  - Avoids CORS-related issues for HTTP client-based tests.


## Alembic Migrate

Workflow file: `.github/workflows/db-migrate.yml`

Supports:
- Manual execution (`workflow_dispatch`) with an `environment` input (default: `staging`).
- Reusable workflow invocation (`workflow_call`) with required secret `DATABASE_URL`.

Behavior:
1. Sets up Python 3.11 and Poetry, installs project dependencies.
2. Ensures Alembic is available (installs via pip if not part of Poetry dependencies).
3. Runs migrations guarded by config presence:
   - If `alembic.ini` exists: runs `alembic upgrade head`.
   - Else if `alembic/` directory exists but no `alembic.ini`: skips with a clear message.
   - Else: prints “No Alembic configuration found; skipping.”


### Manual dispatch

1. Create a GitHub Environment (e.g., `staging` or `production`) if you use environment-level secrets/protections (optional).
2. Add a secret named `DATABASE_URL` either:
   - As an Environment secret (preferred if using environment protections), or
   - As a Repository secret.
3. Go to “Actions” → “Alembic Migrate” → “Run workflow”.
4. Select the target `environment` (defaults to `staging`).
5. Run.

Example `DATABASE_URL` values:
- PostgreSQL (async): `postgres+asyncpg://user:pass@host:5432/db`
- PostgreSQL (sync): `postgresql://user:pass@host:5432/db`

Note: Use a URL matching how your Alembic `env.py` config initializes the engine.


### Reusable call from another workflow

Example deploy workflow invoking migrations:

```yaml
name: Deploy

on:
  workflow_dispatch:

jobs:
  migrate:
    uses: ./.github/workflows/db-migrate.yml
    with:
      environment: production
    secrets:
      DATABASE_URL: ${{ secrets.PROD_DATABASE_URL }}

  # ... other deploy jobs (build, release, etc.)
```

- `environment` controls which GitHub Environment the job is associated with (for approvals, environment secrets, etc.).
- `DATABASE_URL` must be provided as a secret from the caller workflow.


### Safe no-op behavior

- If no Alembic configuration is present:
  - With `alembic.ini`: migrations run (`upgrade head`).
  - With `alembic/` but no `alembic.ini`: the job logs a skip message.
  - Without both: the job logs “No Alembic configuration found; skipping.”
- This ensures deployments don’t fail when no migrations are defined.


## Notes

- Both workflows use Poetry 1.7.1 and Python 3.11 for consistent environments.
- CI artifacts on test failures include `artifacts/test-output.txt` for quick debugging.