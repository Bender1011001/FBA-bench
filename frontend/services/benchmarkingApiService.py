"""
Frontend-facing Benchmarking API service used by integration tests.

Implements an async HTTP client with typed helpers for common endpoints:
- GET /api/scenarios
- GET /api/scenarios/{scenario_name}
- POST /api/benchmarks/run
- GET /api/benchmarks/{benchmark_id}
- GET /api/benchmarks

Tests patch `get`/`post` methods on this class, so these methods must:
- Be async
- Return an object exposing `status_code` and `.json()` when not patched

This implementation uses httpx.AsyncClient under the hood, but will still work
with the test suite's monkeypatching of get/post to return a compatible object.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional

try:
    import httpx
except Exception:  # pragma: no cover - httpx should be available, but keep hardening
    httpx = None  # type: ignore[assignment]


class _HTTPResponse:
    """Lightweight response wrapper to mirror the interface used in tests."""

    def __init__(self, status_code: int, data: Any):
        self.status_code = status_code
        self._data = data

    def json(self) -> Any:
        return self._data


class BenchmarkingApiService:
    """
    Async API client for the backend, mirroring the frontend service surface used in tests.

    Typical usage in tests:
        api = BenchmarkingApiService("http://testserver")
        scenarios = await api.get_scenarios()
        scenario = await api.get_scenario("my_scenario")
        result = await api.run_benchmark("my_scenario", ["agent_a"], ["metric_x"])
        single = await api.get_benchmark_result("bench_1")
        many = await api.get_benchmarks()
    """

    def __init__(self, base_url: str, timeout: float = 15.0, headers: Optional[Dict[str, str]] = None) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = float(timeout)
        self.headers = headers or {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "User-Agent": "FBA-Bench-Frontend/1.0",
        }

    async def get(self, path: str, params: Optional[Dict[str, Any]] = None) -> _HTTPResponse:
        """
        Perform an HTTP GET and return a response-like object with status_code and .json().

        Note: The test suite patches this method to return a MockResponse, so keep signature stable.
        """
        url = f"{self.base_url}{path}"
        if httpx is None:
            raise RuntimeError("httpx is required for real HTTP calls but is not available")

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            resp = await client.get(url, params=params)
            # Attempt JSON parsing; on failure, surface a structured error
            try:
                data = resp.json()
            except Exception:
                data = {"error": "Invalid JSON response", "text": resp.text}
            return _HTTPResponse(resp.status_code, data)

    async def post(self, path: str, json: Optional[Dict[str, Any]] = None) -> _HTTPResponse:
        """
        Perform an HTTP POST and return a response-like object with status_code and .json().

        Note: The test suite patches this method to return a MockResponse, so keep signature stable.
        """
        url = f"{self.base_url}{path}"
        if httpx is None:
            raise RuntimeError("httpx is required for real HTTP calls but is not available")

        async with httpx.AsyncClient(timeout=self.timeout, headers=self.headers) as client:
            resp = await client.post(url, json=json)
            try:
                data = resp.json()
            except Exception:
                data = {"error": "Invalid JSON response", "text": resp.text}
            return _HTTPResponse(resp.status_code, data)

    # -------------------------
    # High-level convenience API
    # -------------------------

    async def get_scenarios(self) -> List[Dict[str, Any]]:
        """Return a list of available scenarios from the backend."""
        resp = await self.get("/api/scenarios")
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch scenarios (status {resp.status_code})")
        data = resp.json()
        scenarios = data.get("scenarios")
        if not isinstance(scenarios, list):
            raise Exception("Invalid scenarios payload")
        return scenarios

    async def get_scenario(self, scenario_name: str) -> Dict[str, Any]:
        """Return details for a specific scenario."""
        if not scenario_name:
            raise ValueError("scenario_name is required")
        resp = await self.get(f"/api/scenarios/{scenario_name}")
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch scenario '{scenario_name}' (status {resp.status_code})")
        data = resp.json()
        if not isinstance(data, dict):
            raise Exception("Invalid scenario payload")
        return data

    async def run_benchmark(
        self,
        scenario_name: str,
        agent_ids: Optional[List[str]] = None,
        metric_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Trigger a benchmark run and return the backend's run descriptor/result stub."""
        if not scenario_name:
            raise ValueError("scenario_name is required")
        payload = {
            "scenario_name": scenario_name,
            "agent_ids": agent_ids or [],
            "metric_names": metric_names or [],
        }
        resp = await self.post("/api/benchmarks/run", json=payload)
        if resp.status_code != 200:
            raise Exception(f"Failed to start benchmark (status {resp.status_code})")
        data = resp.json()
        if not isinstance(data, dict):
            raise Exception("Invalid benchmark run response")
        return data

    async def get_benchmark_result(self, benchmark_id: str) -> Dict[str, Any]:
        """Fetch a single benchmark result by id."""
        if not benchmark_id:
            raise ValueError("benchmark_id is required")
        resp = await self.get(f"/api/benchmarks/{benchmark_id}")
        if resp.status_code != 200:
            raise Exception(f"Failed to fetch benchmark result (status {resp.status_code})")
        data = resp.json()
        if not isinstance(data, dict):
            raise Exception("Invalid benchmark result payload")
        return data

    async def get_benchmarks(self) -> List[Dict[str, Any]]:
        """List recent benchmark results."""
        resp = await self.get("/api/benchmarks")
        if resp.status_code != 200:
            raise Exception(f"Failed to list benchmarks (status {resp.status_code})")
        data = resp.json()
        benchmarks = data.get("benchmarks")
        if not isinstance(benchmarks, list):
            raise Exception("Invalid benchmarks payload")
        return benchmarks