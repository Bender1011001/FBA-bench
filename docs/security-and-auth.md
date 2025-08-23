# Security and Auth

Production deployments must explicitly enable JWT auth and configure CORS. The backend middleware is implemented in [python.class JWTAuthMiddleware](fba_bench_api/main.py:52) and wired in [python.function create_app()](fba_bench_api/main.py:128).

## Auth configuration (env vars)

- AUTH_ENABLED: true to enforce JWT on protected routes. Safe prod default: true.
- AUTH_TEST_BYPASS: when true, bypasses auth for dev/tests. Safe prod default: false.
- AUTH_JWT_PUBLIC_KEY: RSA public key in PEM. Accepts multi-line value or a single line with literal \n characters.
- AUTH_JWT_ALG: JWT verification algorithm; default RS256.
- AUTH_JWT_ISSUER: optional expected iss claim.
- AUTH_JWT_AUDIENCE: optional expected aud claim.
- AUTH_PROTECT_DOCS: true to require auth for API docs.

Notes:
- Provide AUTH_JWT_PUBLIC_KEY exactly as issued by your IdP. Example (single line):

  AUTH_JWT_PUBLIC_KEY=-----BEGIN PUBLIC KEY-----\nMIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...\n-----END PUBLIC KEY-----

  Or multi-line (ensure your process manager preserves newlines):

  -----BEGIN PUBLIC KEY-----
  MIIBIjANBgkqhkiG9w0BAQEFAAOCAQ8A...
  -----END PUBLIC KEY-----

## Protected vs exempt endpoints

- Health checks are not rate limited and are available at [python.function health()](fba_bench_api/main.py:226) (/health) and [python.function health_v1()](fba_bench_api/main.py:265) (/api/v1/health).
- Rate limiting is initialized in [python.module main](fba_bench_api/main.py:114); tune with API_RATE_LIMIT (e.g., 600/minute).

## CORS allow-list

Origins are parsed by [python.function _get_cors_allowed_origins()](fba_bench_api/main.py:103):

- Single origin:
  FBA_CORS_ALLOW_ORIGINS=https://app.example.com

- Multiple origins (comma-separated):
  FBA_CORS_ALLOW_ORIGINS=https://app.example.com,https://admin.example.com

- Wildcard (allow all; not recommended for prod):
  FBA_CORS_ALLOW_ORIGINS=*

## WebSocket auth

- Realtime endpoint: /ws/realtime?topic=<topic>
- Clients must send Sec-WebSocket-Protocol: auth.bearer.token.<JWT>
- The frontend attaches this in [ts.hook useWebSocket](frontend/src/hooks/useWebSocket.ts:56).

## Minimal hardening checklist

- Set AUTH_ENABLED=true, AUTH_TEST_BYPASS=false.
- Provide AUTH_JWT_PUBLIC_KEY and optional AUTH_JWT_ISSUER/AUTH_JWT_AUDIENCE to match your IdP.
- Set AUTH_PROTECT_DOCS=true to gate interactive docs.
- Configure FBA_CORS_ALLOW_ORIGINS with explicit domains (no wildcard).
- Run Alembic migrations before deploy.
- Set API_RATE_LIMIT to a realistic value for your environment.