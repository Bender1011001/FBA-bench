import pytest
import httpx
import asyncio
import subprocess
import time
import signal
import os

class TestAPIServerBasic:
    """
    Test suite for basic API server functionality.
    Uses the api_server fixture from conftest.py to manage server lifecycle.
    """

    def test_server_starts_successfully(self, api_server):
        """
        Verify that the API server starts successfully and is running.
        The api_server fixture handles the startup and ensures it's reachable.
        This test primarily verifies the fixture's success.
        """
        assert api_server is not None, "API server fixture did not return a valid server URL."
        print(f"API server is running at {api_server}")
        # A simple request confirms the server is alive
        with httpx.Client() as client:
            try:
                response = client.get(f"{api_server}/health", timeout=5) # Using dummy endpoint to check connection
                assert response.status_code in [200, 404], f"Unexpected status code: {response.status_code}"
                print(f"API server responded to basic request: {response.status_code}")
            except httpx.RequestError as e:
                pytest.fail(f"Could not connect to API server at {api_server}: {e}")

    @pytest.mark.asyncio
    async def test_health_endpoint(self, api_server):
        """
        Test the /api/v1/health endpoint for proper response.
        Ensures the server is healthy and provides expected information.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_server}/api/v1/health", timeout=10)
                assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
                data = response.json()
                assert "status" in data and data["status"] == "healthy"
                assert "service" in data and data["service"] is not None
                assert "version" in data and data["version"] is not None
                print(f"Health check passed: {data}")
            except httpx.RequestError as e:
                pytest.fail(f"Failed to connect to health endpoint at {api_server}/api/v1/health: {e}")
            except Exception as e:
                pytest.fail(f"An unexpected error occurred during health check: {e}")

    @pytest.mark.asyncio
    async def test_root_endpoint(self, api_server):
        """
        Test the root endpoint (/) for proper response.
        Ensures the server serves the basic index page correctly.
        """
        async with httpx.AsyncClient() as client:
            try:
                response = await client.get(f"{api_server}/", timeout=10)
                assert response.status_code == 200, f"Expected status code 200, got {response.status_code}"
                assert "text/html" in response.headers.get("content-type", "")
                print(f"Root endpoint check passed. Content-Type: {response.headers.get('content-type')}")
            except httpx.RequestError as e:
                pytest.fail(f"Failed to connect to root endpoint at {api_server}/: {e}")
            except Exception as e:
                pytest.fail(f"An unexpected error occurred during root endpoint check: {e}")
