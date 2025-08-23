# Performance Testing

This guide provides a minimal k6 setup to validate API rate limiting on a protected endpoint. Rate limiting is integrated in [python.module main](fba_bench_api/main.py:114). Health endpoints are limiter-exempt: see [python.function health()](fba_bench_api/main.py:226) and [python.function health_v1()](fba_bench_api/main.py:265).

## k6 script location

- Script: [js.module k6-ratelimit](scripts/perf/k6-ratelimit.js)

## Prerequisites

- k6 installed (https://k6.io/docs/get-started/installation/)
- API running locally or in a test environment
- A valid JWT (if auth is enabled) — see [python.module gen_test_jwt](scripts/smoke/jwt/gen_test_jwt.py)

## Run commands

Bash:
- Without token (expect 401/429 depending on config; primarily used to observe limiter behavior if your auth allows):
  k6 run -e BASE_URL=http://localhost:8000 scripts/perf/k6-ratelimit.js

- With token (recommended to hit protected endpoint /api/v1/settings):
  JWT="$(python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub tester)"
  k6 run -e BASE_URL=http://localhost:8000 -e TOKEN="$JWT" scripts/perf/k6-ratelimit.js

PowerShell:
  $env:BASE_URL = "http://localhost:8000"
  $env:TOKEN = (python .\scripts\smoke\jwt\gen_test_jwt.py --private-key .\private.pem --sub tester)
  k6 run .\scripts\perf\k6-ratelimit.js

## Interpretation

- 200/2xx: Requests accepted under the configured limit.
- 429: Requests throttled by the limiter — expected once you exceed configured budget (API_RATE_LIMIT).
- The script prints a summary of 2xx vs 429 to stdout.

Notes:
- Tune API_RATE_LIMIT through env to match your environment’s expectations.
- Avoid running aggressive tests against shared environments; coordinate windows and ramp settings to prevent disruption.
- Health endpoints (/health and /api/v1/health) are limiter-exempt; do not use them to measure rate limiting behavior.