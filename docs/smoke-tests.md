# Smoke Tests Guide

Quick steps to validate JWT auth, REST, and WebSocket behavior locally. Auth is enforced by [python.class JWTAuthMiddleware](fba_bench_api/main.py:52) in the app factory [python.function create_app()](fba_bench_api/main.py:128). Health endpoints exist at [python.function health()](fba_bench_api/main.py:226) (/health) and [python.function health_v1()](fba_bench_api/main.py:265) (/api/v1/health). WebSocket auth is handled in [python.function websocket_realtime()](fba_bench_api/api/routes/realtime.py:190).

Prereqs:
- curl
- Python 3.10+ (for helper scripts)
- OpenSSL (for key generation)
- Optional installs for helpers:
  - pip install pyjwt cryptography (JWT generator)
  - pip install websockets (WS smoke client)

## 1) Generate RSA keys (OpenSSL)

Bash:
- Private + public:
  openssl genrsa -out private.pem 2048
  openssl rsa -in private.pem -pubout -out public.pem

PowerShell:
- Private + public:
  openssl genrsa -out private.pem 2048
  openssl rsa -in private.pem -pubout -out public.pem

Any equivalent tool works; keep private.pem safe.

## 2) Export env and run the API

Set mandatory flags and start the API. The middleware expects:
- AUTH_ENABLED=true
- AUTH_TEST_BYPASS=false
- AUTH_JWT_PUBLIC_KEY set to the RSA public key (PEM)
- DATABASE_URL pointing to sqlite+aiosqlite or dev DB

Option A — Bash:
- Multi-line PEM directly:
  export AUTH_ENABLED=true
  export AUTH_TEST_BYPASS=false
  export AUTH_JWT_PUBLIC_KEY="$(cat public.pem)"
  export AUTH_JWT_ALG=RS256
  export DATABASE_URL=sqlite+aiosqlite:///./dev.db
  # Optional claims
  # export AUTH_JWT_ISSUER="https://auth.example.com/"
  # export AUTH_JWT_AUDIENCE="fba-api"
  # Example run (adjust to your runner):
  uvicorn fba_bench_api.main:create_app --factory --host 0.0.0.0 --port 8000

- Or single-line with literal \n:
  export AUTH_JWT_PUBLIC_KEY="$(awk 'BEGIN{RS=""; gsub(/\n/,"\\n"); print}' public.pem)"

Option B — PowerShell:
  $env:AUTH_ENABLED = "true"
  $env:AUTH_TEST_BYPASS = "false"
  $env:AUTH_JWT_ALG = "RS256"
  $env:DATABASE_URL = "sqlite+aiosqlite:///./dev.db"
  $env:AUTH_JWT_PUBLIC_KEY = Get-Content -Raw -Path .\public.pem
  # Optional claims
  # $env:AUTH_JWT_ISSUER = "https://auth.example.com/"
  # $env:AUTH_JWT_AUDIENCE = "fba-api"
  uvicorn fba_bench_api.main:create_app --factory --host 0.0.0.0 --port 8000

CORS allow-list (if testing from a browser) is parsed by [python.function _get_cors_allowed_origins()](fba_bench_api/main.py:103). For local tools and curl this is not required.

## 3) Generate a signed test JWT

Use [python.module gen_test_jwt](scripts/smoke/jwt/gen_test_jwt.py) (install locally: pip install pyjwt cryptography).

Bash:
- Basic:
  python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub "tester"
- With claims to match middleware (if configured):
  python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub "tester" --iss "https://auth.example.com/" --aud "fba-api" --exp-min 10

PowerShell:
  python .\scripts\smoke\jwt\gen_test_jwt.py --private-key .\private.pem --sub "tester" --iss "https://auth.example.com/" --aud "fba-api" --exp-min 10

The script prints the token to stdout.

## 4) REST smoke with curl

Use the provided scripts or raw curl.

- Bash script [bash.module curl-smoke.sh](scripts/smoke/curl-smoke.sh):
  # Without JWT (expects 401 when AUTH_ENABLED=true)
  API_URL=http://localhost:8000 scripts/smoke/curl-smoke.sh
  # With JWT (expects 200)
  JWT="$(python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub tester)" API_URL=http://localhost:8000 scripts/smoke/curl-smoke.sh

- PowerShell script [ps.module curl-smoke.ps1](scripts/smoke/curl-smoke.ps1):
  # Without JWT
  .\scripts\smoke\curl-smoke.ps1 -ApiUrl "http://localhost:8000"
  # With JWT
  $jwt = python .\scripts\smoke\jwt\gen_test_jwt.py --private-key .\private.pem --sub tester
  .\scripts\smoke\curl-smoke.ps1 -ApiUrl "http://localhost:8000" -Jwt $jwt

Manual curl (for reference):
- Protected endpoint without token (expect 401 if AUTH_ENABLED=true):
  curl -i http://localhost:8000/api/v1/settings
- With token (expect 200):
  curl -i -H "Authorization: Bearer <JWT>" http://localhost:8000/api/v1/settings

## 5) WebSocket smoke (optional)

Use [python.module ws_smoke](scripts/smoke/ws_smoke.py) (install: pip install websockets).

Bash:
  python scripts/smoke/ws_smoke.py --url "ws://localhost:8000/ws/realtime?topic=health" --jwt "<JWT>"

PowerShell:
  python .\scripts\smoke\ws_smoke.py --url "ws://localhost:8000/ws/realtime?topic=health" --jwt "<JWT>"

The client sets Sec-WebSocket-Protocol to "auth.bearer.token.<JWT>". Expect connection accepted when JWT is valid; otherwise closed with policy violation.

Notes:
- If your middleware enforces issuer/audience, pass them to the JWT generator via --iss/--aud and set corresponding envs (AUTH_JWT_ISSUER, AUTH_JWT_AUDIENCE).
- Health endpoints (/health, /api/v1/health) are limiter-exempt; rate limiting is configured in [python.module main](fba_bench_api/main.py:114) via API_RATE_LIMIT.