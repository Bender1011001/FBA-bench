#!/usr/bin/env python3
"""
Generate an RS256-signed JWT for local smoke testing.

Requirements (install locally, not added to project dependencies):
  pip install pyjwt cryptography

Usage examples:
  python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub "tester"
  python scripts/smoke/jwt/gen_test_jwt.py --private-key ./private.pem --sub "tester" --iss "https://auth.example.com/" --aud "fba-api" --exp-min 10

Notes:
- Backend auth middleware is implemented in [python.class JWTAuthMiddleware](fba_bench_api/main.py:52) and wired in [python.function create_app()](fba_bench_api/main.py:128).
- If your backend enforces issuer/audience claims, pass --iss/--aud here and set AUTH_JWT_ISSUER/AUTH_JWT_AUDIENCE accordingly.
"""

import argparse
import sys
import time
from typing import Optional

try:
    import jwt  # PyJWT
except ImportError as e:
    sys.stderr.write("Missing dependency: pyjwt. Install with: pip install pyjwt cryptography\n")
    raise

def _read_key(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        sys.stderr.write(f"Private key file not found: {path}\n")
        sys.exit(2)

def _build_payload(sub: str, exp_minutes: int, iss: Optional[str], aud: Optional[str]) -> dict:
    now = int(time.time())
    exp = now + (exp_minutes * 60)
    payload = {
        "sub": sub,
        "iat": now,
        "exp": exp,
    }
    if iss:
        payload["iss"] = iss
    if aud:
        payload["aud"] = aud
    return payload

def main():
    parser = argparse.ArgumentParser(description="Generate an RS256 JWT for smoke testing.")
    parser.add_argument("--private-key", required=True, help="Path to RSA private key PEM (e.g., ./private.pem)")
    parser.add_argument("--sub", default="tester", help='Subject claim "sub" (default: tester)')
    parser.add_argument("--iss", default=None, help="Issuer claim 'iss' (optional)")
    parser.add_argument("--aud", default=None, help="Audience claim 'aud' (optional)")
    parser.add_argument("--exp-min", type=int, default=10, help="Minutes until token expiry (default: 10)")
    parser.add_argument("--alg", default="RS256", help="JWT signing algorithm (default: RS256)")
    args = parser.parse_args()

    private_key = _read_key(args.private_key)
    payload = _build_payload(args.sub, args.exp_min, args.iss, args.aud)

    try:
        token = jwt.encode(payload, private_key, algorithm=args.alg)
    except Exception as e:
        sys.stderr.write(f"Failed to sign JWT: {e}\n")
        sys.exit(3)

    # Print token only, to stdout
    sys.stdout.write(token if isinstance(token, str) else token.decode("utf-8"))
    sys.stdout.flush()

if __name__ == "__main__":
    main()