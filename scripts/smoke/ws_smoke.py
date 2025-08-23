#!/usr/bin/env python3
"""
WebSocket smoke client.

Requirements (install locally, not added to project dependencies):
  pip install websockets

CLI:
  python scripts/smoke/ws_smoke.py --url "ws://localhost:8000/ws/realtime?topic=health" --jwt "<TOKEN>"

Behavior:
- Sets Sec-WebSocket-Protocol to "auth.bearer.token.<JWT>" as required by the backend
  WebSocket endpoint implemented at [python.function websocket_realtime()](fba_bench_api/api/routes/realtime.py:190).
- On successful connection, waits briefly, prints "WS connected", then exits 0.
- On failure, prints the error and exits non-zero.
"""

import argparse
import asyncio
import sys

try:
    import websockets  # type: ignore
except ImportError:
    sys.stderr.write("Missing dependency: websockets. Install with: pip install websockets\n")
    sys.exit(2)


async def ws_probe(url: str, jwt: str, timeout: float = 10.0) -> int:
    subprotocol = f"auth.bearer.token.{jwt}"
    try:
        async with asyncio.timeout(timeout):
            async with websockets.connect(
                url,
                subprotocols=[subprotocol],
                ping_interval=20,
                close_timeout=5,
                max_queue=None,
            ) as ws:
                # Optionally read a short message if server sends one; not required
                await asyncio.sleep(0.5)
                print("WS connected")
                try:
                    await ws.close()
                except Exception:
                    pass
                return 0
    except Exception as e:
        sys.stderr.write(f"WS connection failed: {e}\n")
        return 1


def main():
    parser = argparse.ArgumentParser(description="WebSocket smoke client")
    parser.add_argument("--url", required=True, help='WebSocket URL, e.g., "ws://localhost:8000/ws/realtime?topic=health"')
    parser.add_argument("--jwt", required=True, help="Bearer token to attach via Sec-WebSocket-Protocol")
    args = parser.parse_args()

    try:
        code = asyncio.run(ws_probe(args.url, args.jwt))
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()