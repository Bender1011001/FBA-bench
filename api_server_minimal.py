# =========================================================
# DEPRECATED: This file has been retired from production.
# It remains only to prevent import errors if referenced inadvertently.
# The canonical backend is fba_bench_api/main.py. For tests, see:
# tests/fixtures/api_server_minimal.py
# =========================================================

raise RuntimeError(
    "api_server_minimal.py is not part of production code. "
    "Use fba_bench_api.main:app to run the API. "
    "For test fixtures, import tests.fixtures.api_server_minimal."
)