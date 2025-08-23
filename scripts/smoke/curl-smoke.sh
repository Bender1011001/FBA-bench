#!/usr/bin/env bash
# Smoke test script for REST endpoints (bash)
# Requirements: bash, curl
# Usage:
#   API_URL=http://localhost:8000 scripts/smoke/curl-smoke.sh
#   JWT="$(python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub tester)" API_URL=http://localhost:8000 scripts/smoke/curl-smoke.sh
#
# Validations:
#   - health_check: expect HTTP 200 or 503 JSON
#   - protected_check_no_token: expect HTTP 401 when AUTH is enabled
#   - protected_check_with_token: expect HTTP 200 when JWT is provided
#
# Notes:
# - Health endpoints are limiter-exempt per [python.function health()](fba_bench_api/main.py:226) and [python.function health_v1()](fba_bench_api/main.py:265).
# - Rate limiting is configured in [python.module main](fba_bench_api/main.py:114) via API_RATE_LIMIT.

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
JWT="${JWT:-}"

pass() { echo "PASS: $*"; }
fail() { echo "FAIL: $*" 1>&2; exit 1; }

health_check() {
  echo "==> Health check: $API_URL/api/v1/health"
  # Print body and capture status
  local body status
  body="$(curl -sS "$API_URL/api/v1/health" || true)"
  status="$(curl -sS -o /dev/null -w "%{http_code}" "$API_URL/api/v1/health" || true)"
  echo "$body"
  if [[ "$status" == "200" || "$status" == "503" ]]; then
    pass "Health returned $status"
  else
    fail "Health unexpected status: $status"
  fi
}

protected_check_no_token() {
  echo "==> Protected endpoint without token (expect 401): $API_URL/api/v1/settings"
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" "$API_URL/api/v1/settings" || true)"
  if [[ "$code" == "401" ]]; then
    pass "Unauthorized as expected (401)"
  else
    fail "Expected 401 without token, got $code"
  fi
}

protected_check_with_token() {
  echo "==> Protected endpoint with token (expect 200): $API_URL/api/v1/settings"
  if [[ -z "${JWT}" ]]; then
    echo "JWT not provided; skipping token-auth check."
    return 0
  fi
  local code
  code="$(curl -sS -o /dev/null -w "%{http_code}" -H "Authorization: Bearer ${JWT}" "$API_URL/api/v1/settings" || true)"
  if [[ "$code" == "200" ]]; then
    pass "Authorized as expected (200)"
  else
    fail "With JWT expected 200, got $code"
  fi
}

main() {
  echo "API_URL=${API_URL}"
  [[ -n "${JWT}" ]] && echo "JWT provided: yes" || echo "JWT provided: no"

  health_check
  protected_check_no_token
  protected_check_with_token

  echo "All checks completed."
}

main "$@"