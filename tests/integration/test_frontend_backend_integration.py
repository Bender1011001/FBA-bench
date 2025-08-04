"""
Integration tests for frontend-backend communication and data flow.

This module contains comprehensive integration tests that verify the interaction
between the frontend and backend components of the FBA-Bench system, including
API communication, WebSocket connections, data synchronization, and real-time updates.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
import json
import tempfile
import os
import numpy as np
import websockets
import requests
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.testclient import TestClient
import pytest_asyncio

# Import backend components
from benchmarking.core.engine import BenchmarkEngine, BenchmarkConfig, BenchmarkResult
from benchmarking.scenarios.base import ScenarioConfig, BaseScenario
from benchmarking.scenarios.registry import ScenarioRegistry
from benchmarking.metrics.base import BaseMetric, MetricConfig, MetricResult
from benchmarking.metrics.registry import MetricRegistry
from agent_runners.base_runner import BaseAgentRunner, AgentConfig

# Import frontend components
from frontend.services.benchmarkingApiService import BenchmarkingApiService
from frontend.services.webSocketService import WebSocketService


class MockBackendAPI:
    """Mock backend API for integration testing."""
    
    def __init__(self):
        self.app = FastAPI(title="FBA-Bench Test API")
        self.benchmark_engine = None
        self.active_connections: List[WebSocket] = []
        self.setup_routes()
    
    def setup_routes(self):
        """Setup API routes for testing."""
        
        @self.app.get("/api/health")
        async def health_check():
            return {"status": "healthy", "timestamp": datetime.now().isoformat()}
        
        @self.app.get("/api/scenarios")
        async def get_scenarios():
            if not self.benchmark_engine:
                raise HTTPException(status_code=500, detail="Benchmark engine not initialized")
            
            scenarios = []
            for name, scenario in self.benchmark_engine.scenarios.items():
                scenarios.append({
                    "name": name,
                    "description": scenario.config.description,
                    "domain": scenario.config.domain,
                    "duration_ticks": scenario.config.duration_ticks,
                    "enabled": scenario.config.enabled
                })
            
            return {"scenarios": scenarios}
        
        @self.app.get("/api/scenarios/{scenario_name}")
        async def get_scenario(scenario_name: str):
            if not self.benchmark_engine:
                raise HTTPException(status_code=500, detail="Benchmark engine not initialized")
            
            if scenario_name not in self.benchmark_engine.scenarios:
                raise HTTPException(status_code=404, detail="Scenario not found")
            
            scenario = self.benchmark_engine.scenarios[scenario_name]
            return {
                "name": scenario_name,
                "description": scenario.config.description,
                "domain": scenario.config.domain,
                "duration_ticks": scenario.config.duration_ticks,
                "parameters": scenario.config.parameters,
                "enabled": scenario.config.enabled
            }
        
        @self.app.post("/api/benchmarks/run")
        async def run_benchmark(request: Dict[str, Any]):
            if not self.benchmark_engine:
                raise HTTPException(status_code=500, detail="Benchmark engine not initialized")
            
            scenario_name = request.get("scenario_name")
            agent_ids = request.get("agent_ids", [])
            metric_names = request.get("metric_names", [])
            
            if not scenario_name:
                raise HTTPException(status_code=400, detail="Scenario name is required")
            
            if scenario_name not in self.benchmark_engine.scenarios:
                raise HTTPException(status_code=404, detail="Scenario not found")
            
            # Run the benchmark
            result = await self.benchmark_engine.run_benchmark(
                scenario_name=scenario_name,
                agent_ids=agent_ids,
                metric_names=metric_names
            )
            
            # Convert to JSON-serializable format
            result_dict = {
                "scenario_name": result.scenario_name,
                "agent_ids": result.agent_ids,
                "metric_names": result.metric_names,
                "start_time": result.start_time.isoformat() if result.start_time else None,
                "end_time": result.end_time.isoformat() if result.end_time else None,
                "duration_seconds": result.duration_seconds,
                "success": result.success,
                "errors": result.errors,
                "results": result.results
            }
            
            return result_dict
        
        @self.app.get("/api/benchmarks/{benchmark_id}")
        async def get_benchmark_result(benchmark_id: str):
            # In a real implementation, this would fetch from a database
            # For testing, we'll return a mock result
            return {
                "id": benchmark_id,
                "scenario_name": "test_scenario",
                "agent_ids": ["test_agent"],
                "metric_names": ["test_metric"],
                "status": "completed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 10.5,
                "success": True,
                "results": {
                    "test_metric": {
                        "value": 85.0,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        
        @self.app.get("/api/benchmarks")
        async def get_benchmarks():
            # In a real implementation, this would fetch from a database
            # For testing, we'll return mock results
            return {
                "benchmarks": [
                    {
                        "id": "benchmark_1",
                        "scenario_name": "test_scenario",
                        "agent_ids": ["test_agent"],
                        "status": "completed",
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "success": True
                    },
                    {
                        "id": "benchmark_2",
                        "scenario_name": "another_scenario",
                        "agent_ids": ["another_agent"],
                        "status": "running",
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "success": None
                    }
                ]
            }
        
        @self.app.websocket("/ws")
        async def websocket_endpoint(websocket: WebSocket):
            await websocket.accept()
            self.active_connections.append(websocket)
            try:
                while True:
                    # Wait for messages from the client
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Process the message
                    response = await self.process_websocket_message(message)
                    
                    # Send response back to the client
                    await websocket.send_text(json.dumps(response))
                    
            except WebSocketDisconnect:
                self.active_connections.remove(websocket)
    
    async def process_websocket_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a WebSocket message and return a response."""
        message_type = message.get("type")
        
        if message_type == "subscribe":
            # Subscribe to updates for a specific benchmark
            benchmark_id = message.get("benchmark_id")
            return {
                "type": "subscription_confirmed",
                "benchmark_id": benchmark_id,
                "timestamp": datetime.now().isoformat()
            }
        
        elif message_type == "get_status":
            # Get the status of a running benchmark
            benchmark_id = message.get("benchmark_id")
            return {
                "type": "status_update",
                "benchmark_id": benchmark_id,
                "status": "running",
                "progress": 0.5,
                "timestamp": datetime.now().isoformat()
            }
        
        elif message_type == "ping":
            # Respond to a ping
            return {
                "type": "pong",
                "timestamp": datetime.now().isoformat()
            }
        
        else:
            return {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.now().isoformat()
            }
    
    async def broadcast_update(self, message: Dict[str, Any]):
        """Broadcast a message to all connected WebSocket clients."""
        if self.active_connections:
            message_str = json.dumps(message)
            for connection in self.active_connections:
                try:
                    await connection.send_text(message_str)
                except:
                    # Connection might be closed, remove it
                    if connection in self.active_connections:
                        self.active_connections.remove(connection)
    
    def set_benchmark_engine(self, engine: BenchmarkEngine):
        """Set the benchmark engine for the API."""
        self.benchmark_engine = engine


class MockAgentForIntegration(BaseAgentRunner):
    """Mock agent implementation for frontend-backend integration testing."""
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.responses = []
        self.actions_taken = []
    
    async def initialize(self) -> None:
        """Initialize the mock agent."""
        self.is_initialized = True
    
    async def process_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process input and return a response."""
        response = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "response": f"Mock response to: {input_data.get('content', '')}",
            "confidence": np.random.uniform(0.7, 0.95),
            "processing_time": np.random.uniform(0.01, 0.1)
        }
        
        self.responses.append(response)
        return response
    
    async def execute_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Execute an action and return the result."""
        success = np.random.random() > 0.1  # 90% success rate
        
        result = {
            "agent_id": self.config.agent_id,
            "action": action.get("type", "unknown"),
            "timestamp": datetime.now().isoformat(),
            "status": "completed" if success else "failed",
            "result": f"Executed action: {action.get('type', 'unknown')}",
            "execution_time": np.random.uniform(0.01, 0.2)
        }
        
        self.actions_taken.append(result)
        return result
    
    async def collect_metrics(self) -> Dict[str, Any]:
        """Collect metrics from the agent."""
        metrics = {
            "agent_id": self.config.agent_id,
            "timestamp": datetime.now().isoformat(),
            "responses_count": len(self.responses),
            "actions_count": len(self.actions_taken),
            "avg_response_time": np.mean([r["processing_time"] for r in self.responses]) if self.responses else 0,
            "success_rate": len([a for a in self.actions_taken if a["status"] == "completed"]) / len(self.actions_taken) if self.actions_taken else 0
        }
        
        return metrics
    
    async def shutdown(self) -> None:
        """Shutdown the mock agent."""
        self.is_initialized = False


class TestScenarioForIntegration(BaseScenario):
    """Test scenario implementation for frontend-backend integration testing."""
    
    def _validate_domain_parameters(self) -> List[str]:
        """Validate domain-specific parameters."""
        return []
    
    async def initialize(self, parameters: Dict[str, Any]) -> None:
        """Initialize the test scenario."""
        await super().initialize(parameters)
        self.test_data = parameters.get("test_data", {})
    
    async def setup_for_agent(self, agent_id: str) -> None:
        """Setup the scenario for a specific agent."""
        await super().setup_for_agent(agent_id)
        self.agent_states[agent_id]["test_data"] = self.test_data
    
    async def update_tick(self, tick: int, state) -> None:
        """Update the scenario for a specific tick."""
        await super().update_tick(tick, state)
        
        # Simulate some scenario state changes
        for agent_id in self.agent_states:
            agent_state = self.agent_states[agent_id]
            agent_state["progress"] = tick / self.duration_ticks
    
    async def evaluate_agent_performance(self, agent_id: str) -> Dict[str, Any]:
        """Evaluate agent performance in the scenario."""
        base_metrics = await super().evaluate_agent_performance(agent_id)
        
        agent_state = self.agent_states[agent_id]
        progress = agent_state.get("progress", 0.0)
        
        scenario_metrics = {
            "progress": progress,
            "efficiency_score": progress * 0.9,
            "task_completion_rate": progress * 0.85
        }
        
        return {**base_metrics, **scenario_metrics}


class TestMetricForIntegration(BaseMetric):
    """Test metric implementation for frontend-backend integration testing."""
    
    def __init__(self, config: MetricConfig):
        super().__init__(config)
    
    def calculate(self, data: Dict[str, Any]) -> float:
        """Calculate the metric value."""
        # Simple calculation for testing
        return np.random.uniform(70.0, 95.0)


class TestFrontendBackendIntegration:
    """Test cases for frontend-backend communication and data flow."""
    
    @pytest.fixture
    def agent_config(self):
        """Create a test agent configuration."""
        return AgentConfig(
            agent_id="test_agent",
            agent_type="mock_for_integration",
            agent_class="MockAgentForIntegration",
            parameters={"test_param": "test_value"}
        )
    
    @pytest.fixture
    def scenario_config(self):
        """Create a test scenario configuration."""
        return ScenarioConfig(
            name="test_scenario_for_integration",
            description="Test scenario for frontend-backend integration",
            domain="test",
            duration_ticks=20,
            parameters={"test_data": {"key": "value"}}
        )
    
    @pytest.fixture
    def metric_config(self):
        """Create a test metric configuration."""
        return MetricConfig(
            name="test_metric_for_integration",
            description="Test metric for frontend-backend integration",
            unit="score",
            min_value=0.0,
            max_value=100.0,
            target_value=85.0
        )
    
    @pytest.fixture
    def benchmark_config(self):
        """Create a test benchmark configuration."""
        return BenchmarkConfig(
            name="frontend_backend_integration_test",
            description="Test frontend-backend communication and data flow",
            max_duration=300,
            tick_interval=0.1,
            metrics_collection_interval=1.0
        )
    
    @pytest.fixture
    def mock_agent(self, agent_config):
        """Create a mock agent for integration testing."""
        return MockAgentForIntegration(agent_config)
    
    @pytest.fixture
    def test_scenario(self, scenario_config):
        """Create a test scenario for integration testing."""
        return TestScenarioForIntegration(scenario_config)
    
    @pytest.fixture
    def test_metric(self, metric_config):
        """Create a test metric for integration testing."""
        return TestMetricForIntegration(metric_config)
    
    @pytest.fixture
    def benchmark_engine(self, benchmark_config):
        """Create a benchmark engine instance."""
        return BenchmarkEngine(benchmark_config)
    
    @pytest.fixture
    def backend_api(self, benchmark_engine):
        """Create a backend API instance."""
        api = MockBackendAPI()
        api.set_benchmark_engine(benchmark_engine)
        return api
    
    @pytest.fixture
    def test_client(self, backend_api):
        """Create a test client for the backend API."""
        return TestClient(backend_api.app)
    
    @pytest.fixture
    def api_service(self, test_client):
        """Create a frontend API service instance."""
        return BenchmarkingApiService("http://testserver")
    
    @pytest.fixture
    def websocket_service(self):
        """Create a WebSocket service instance."""
        return WebSocketService("ws://testserver/ws")
    
    @pytest.mark.asyncio
    async def test_backend_api_health_check(self, test_client):
        """Test backend API health check endpoint."""
        response = test_client.get("/api/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
    
    @pytest.mark.asyncio
    async def test_backend_api_get_scenarios(self, test_client, benchmark_engine, test_scenario):
        """Test backend API get scenarios endpoint."""
        # Register the scenario
        benchmark_engine.register_scenario(test_scenario)
        
        response = test_client.get("/api/scenarios")
        
        assert response.status_code == 200
        data = response.json()
        assert "scenarios" in data
        assert len(data["scenarios"]) > 0
        
        # Check that our test scenario is in the list
        scenario_names = [s["name"] for s in data["scenarios"]]
        assert "test_scenario_for_integration" in scenario_names
        
        # Check scenario data structure
        for scenario in data["scenarios"]:
            assert "name" in scenario
            assert "description" in scenario
            assert "domain" in scenario
            assert "duration_ticks" in scenario
            assert "enabled" in scenario
    
    @pytest.mark.asyncio
    async def test_backend_api_get_scenario(self, test_client, benchmark_engine, test_scenario):
        """Test backend API get specific scenario endpoint."""
        # Register the scenario
        benchmark_engine.register_scenario(test_scenario)
        
        response = test_client.get("/api/scenarios/test_scenario_for_integration")
        
        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "test_scenario_for_integration"
        assert data["description"] == "Test scenario for frontend-backend integration"
        assert data["domain"] == "test"
        assert data["duration_ticks"] == 20
        assert "parameters" in data
        assert "enabled" in data
        
        # Test with non-existent scenario
        response = test_client.get("/api/scenarios/nonexistent_scenario")
        assert response.status_code == 404
    
    @pytest.mark.asyncio
    async def test_backend_api_run_benchmark(self, test_client, benchmark_engine, mock_agent, test_scenario, test_metric):
        """Test backend API run benchmark endpoint."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        benchmark_engine.register_metric(test_metric)
        
        # Run benchmark via API
        request_data = {
            "scenario_name": "test_scenario_for_integration",
            "agent_ids": ["test_agent"],
            "metric_names": ["test_metric_for_integration"]
        }
        
        response = test_client.post("/api/benchmarks/run", json=request_data)
        
        assert response.status_code == 200
        data = response.json()
        assert data["scenario_name"] == "test_scenario_for_integration"
        assert data["agent_ids"] == ["test_agent"]
        assert data["metric_names"] == ["test_metric_for_integration"]
        assert data["success"] is True
        assert "duration_seconds" in data
        assert "results" in data
        
        # Test with non-existent scenario
        request_data["scenario_name"] = "nonexistent_scenario"
        response = test_client.post("/api/benchmarks/run", json=request_data)
        assert response.status_code == 404
        
        # Test with missing scenario name
        del request_data["scenario_name"]
        response = test_client.post("/api/benchmarks/run", json=request_data)
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.asyncio
    async def test_backend_api_get_benchmark_result(self, test_client):
        """Test backend API get benchmark result endpoint."""
        benchmark_id = "test_benchmark_id"
        
        response = test_client.get(f"/api/benchmarks/{benchmark_id}")
        
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == benchmark_id
        assert data["scenario_name"] == "test_scenario"
        assert data["agent_ids"] == ["test_agent"]
        assert data["status"] == "completed"
        assert data["success"] is True
        assert "results" in data
    
    @pytest.mark.asyncio
    async def test_backend_api_get_benchmarks(self, test_client):
        """Test backend API get benchmarks endpoint."""
        response = test_client.get("/api/benchmarks")
        
        assert response.status_code == 200
        data = response.json()
        assert "benchmarks" in data
        assert len(data["benchmarks"]) > 0
        
        # Check benchmark data structure
        for benchmark in data["benchmarks"]:
            assert "id" in benchmark
            assert "scenario_name" in benchmark
            assert "agent_ids" in benchmark
            assert "status" in benchmark
            assert "start_time" in benchmark
            assert "success" in benchmark
    
    @pytest.mark.asyncio
    async def test_frontend_api_service_get_scenarios(self, api_service, test_client, benchmark_engine, test_scenario):
        """Test frontend API service get scenarios method."""
        # Register the scenario
        benchmark_engine.register_scenario(test_scenario)
        
        # Mock the HTTP request
        with patch.object(api_service, 'get', return_value=MockResponse(
            status_code=200,
            json_data={
                "scenarios": [
                    {
                        "name": "test_scenario_for_integration",
                        "description": "Test scenario for frontend-backend integration",
                        "domain": "test",
                        "duration_ticks": 20,
                        "enabled": True
                    }
                ]
            }
        )):
            scenarios = await api_service.get_scenarios()
            
            assert len(scenarios) == 1
            assert scenarios[0]["name"] == "test_scenario_for_integration"
            assert scenarios[0]["description"] == "Test scenario for frontend-backend integration"
            assert scenarios[0]["domain"] == "test"
            assert scenarios[0]["duration_ticks"] == 20
            assert scenarios[0]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_frontend_api_service_get_scenario(self, api_service, test_client, benchmark_engine, test_scenario):
        """Test frontend API service get specific scenario method."""
        # Register the scenario
        benchmark_engine.register_scenario(test_scenario)
        
        # Mock the HTTP request
        with patch.object(api_service, 'get', return_value=MockResponse(
            status_code=200,
            json_data={
                "name": "test_scenario_for_integration",
                "description": "Test scenario for frontend-backend integration",
                "domain": "test",
                "duration_ticks": 20,
                "parameters": {"test_data": {"key": "value"}},
                "enabled": True
            }
        )):
            scenario = await api_service.get_scenario("test_scenario_for_integration")
            
            assert scenario["name"] == "test_scenario_for_integration"
            assert scenario["description"] == "Test scenario for frontend-backend integration"
            assert scenario["domain"] == "test"
            assert scenario["duration_ticks"] == 20
            assert scenario["parameters"] == {"test_data": {"key": "value"}}
            assert scenario["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_frontend_api_service_run_benchmark(self, api_service, test_client, benchmark_engine, mock_agent, test_scenario, test_metric):
        """Test frontend API service run benchmark method."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        benchmark_engine.register_metric(test_metric)
        
        # Mock the HTTP request
        with patch.object(api_service, 'post', return_value=MockResponse(
            status_code=200,
            json_data={
                "scenario_name": "test_scenario_for_integration",
                "agent_ids": ["test_agent"],
                "metric_names": ["test_metric_for_integration"],
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 10.5,
                "success": True,
                "errors": [],
                "results": {
                    "test_metric_for_integration": {
                        "value": 85.0,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        )):
            result = await api_service.run_benchmark(
                scenario_name="test_scenario_for_integration",
                agent_ids=["test_agent"],
                metric_names=["test_metric_for_integration"]
            )
            
            assert result["scenario_name"] == "test_scenario_for_integration"
            assert result["agent_ids"] == ["test_agent"]
            assert result["metric_names"] == ["test_metric_for_integration"]
            assert result["success"] is True
            assert result["duration_seconds"] == 10.5
            assert "results" in result
            assert "test_metric_for_integration" in result["results"]
    
    @pytest.mark.asyncio
    async def test_frontend_api_service_get_benchmark_result(self, api_service, test_client):
        """Test frontend API service get benchmark result method."""
        benchmark_id = "test_benchmark_id"
        
        # Mock the HTTP request
        with patch.object(api_service, 'get', return_value=MockResponse(
            status_code=200,
            json_data={
                "id": benchmark_id,
                "scenario_name": "test_scenario",
                "agent_ids": ["test_agent"],
                "metric_names": ["test_metric"],
                "status": "completed",
                "start_time": datetime.now().isoformat(),
                "end_time": datetime.now().isoformat(),
                "duration_seconds": 10.5,
                "success": True,
                "results": {
                    "test_metric": {
                        "value": 85.0,
                        "timestamp": datetime.now().isoformat()
                    }
                }
            }
        )):
            result = await api_service.get_benchmark_result(benchmark_id)
            
            assert result["id"] == benchmark_id
            assert result["scenario_name"] == "test_scenario"
            assert result["agent_ids"] == ["test_agent"]
            assert result["status"] == "completed"
            assert result["success"] is True
            assert "results" in result
    
    @pytest.mark.asyncio
    async def test_frontend_api_service_get_benchmarks(self, api_service, test_client):
        """Test frontend API service get benchmarks method."""
        # Mock the HTTP request
        with patch.object(api_service, 'get', return_value=MockResponse(
            status_code=200,
            json_data={
                "benchmarks": [
                    {
                        "id": "benchmark_1",
                        "scenario_name": "test_scenario",
                        "agent_ids": ["test_agent"],
                        "status": "completed",
                        "start_time": datetime.now().isoformat(),
                        "end_time": datetime.now().isoformat(),
                        "success": True
                    },
                    {
                        "id": "benchmark_2",
                        "scenario_name": "another_scenario",
                        "agent_ids": ["another_agent"],
                        "status": "running",
                        "start_time": datetime.now().isoformat(),
                        "end_time": None,
                        "success": None
                    }
                ]
            }
        )):
            benchmarks = await api_service.get_benchmarks()
            
            assert len(benchmarks) == 2
            assert benchmarks[0]["id"] == "benchmark_1"
            assert benchmarks[0]["scenario_name"] == "test_scenario"
            assert benchmarks[0]["status"] == "completed"
            assert benchmarks[0]["success"] is True
            
            assert benchmarks[1]["id"] == "benchmark_2"
            assert benchmarks[1]["scenario_name"] == "another_scenario"
            assert benchmarks[1]["status"] == "running"
            assert benchmarks[1]["success"] is None
    
    @pytest.mark.asyncio
    async def test_websocket_service_connection(self, websocket_service):
        """Test WebSocket service connection."""
        # Mock the WebSocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Test connection
            await websocket_service.connect()
            
            # Verify connection was attempted
            mock_connect.assert_called_once_with("ws://testserver/ws")
            
            # Verify connection state
            assert websocket_service.connected is True
            assert websocket_service.websocket == mock_websocket
    
    @pytest.mark.asyncio
    async def test_websocket_service_subscribe(self, websocket_service):
        """Test WebSocket service subscription."""
        # Mock the WebSocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Connect to WebSocket
            await websocket_service.connect()
            
            # Mock the response to subscription
            mock_websocket.recv.side_effect = [
                json.dumps({
                    "type": "subscription_confirmed",
                    "benchmark_id": "test_benchmark_id",
                    "timestamp": datetime.now().isoformat()
                })
            ]
            
            # Subscribe to benchmark updates
            response = await websocket_service.subscribe("test_benchmark_id")
            
            # Verify subscription message was sent
            expected_message = {
                "type": "subscribe",
                "benchmark_id": "test_benchmark_id"
            }
            mock_websocket.send.assert_called_with(json.dumps(expected_message))
            
            # Verify subscription response
            assert response["type"] == "subscription_confirmed"
            assert response["benchmark_id"] == "test_benchmark_id"
    
    @pytest.mark.asyncio
    async def test_websocket_service_get_status(self, websocket_service):
        """Test WebSocket service get status."""
        # Mock the WebSocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Connect to WebSocket
            await websocket_service.connect()
            
            # Mock the response to status request
            mock_websocket.recv.side_effect = [
                json.dumps({
                    "type": "status_update",
                    "benchmark_id": "test_benchmark_id",
                    "status": "running",
                    "progress": 0.5,
                    "timestamp": datetime.now().isoformat()
                })
            ]
            
            # Request status
            response = await websocket_service.get_status("test_benchmark_id")
            
            # Verify status message was sent
            expected_message = {
                "type": "get_status",
                "benchmark_id": "test_benchmark_id"
            }
            mock_websocket.send.assert_called_with(json.dumps(expected_message))
            
            # Verify status response
            assert response["type"] == "status_update"
            assert response["benchmark_id"] == "test_benchmark_id"
            assert response["status"] == "running"
            assert response["progress"] == 0.5
    
    @pytest.mark.asyncio
    async def test_websocket_service_ping_pong(self, websocket_service):
        """Test WebSocket service ping-pong."""
        # Mock the WebSocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Connect to WebSocket
            await websocket_service.connect()
            
            # Mock the response to ping
            mock_websocket.recv.side_effect = [
                json.dumps({
                    "type": "pong",
                    "timestamp": datetime.now().isoformat()
                })
            ]
            
            # Send ping
            response = await websocket_service.ping()
            
            # Verify ping message was sent
            expected_message = {
                "type": "ping"
            }
            mock_websocket.send.assert_called_with(json.dumps(expected_message))
            
            # Verify pong response
            assert response["type"] == "pong"
    
    @pytest.mark.asyncio
    async def test_websocket_service_listen_for_updates(self, websocket_service):
        """Test WebSocket service listen for updates."""
        # Mock the WebSocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Connect to WebSocket
            await websocket_service.connect()
            
            # Mock multiple updates
            mock_websocket.recv.side_effect = [
                json.dumps({
                    "type": "status_update",
                    "benchmark_id": "test_benchmark_id",
                    "status": "running",
                    "progress": 0.3,
                    "timestamp": datetime.now().isoformat()
                }),
                json.dumps({
                    "type": "status_update",
                    "benchmark_id": "test_benchmark_id",
                    "status": "running",
                    "progress": 0.6,
                    "timestamp": datetime.now().isoformat()
                }),
                json.dumps({
                    "type": "status_update",
                    "benchmark_id": "test_benchmark_id",
                    "status": "completed",
                    "progress": 1.0,
                    "timestamp": datetime.now().isoformat()
                })
            ]
            
            # Listen for updates
            updates = []
            async for update in websocket_service.listen_for_updates("test_benchmark_id"):
                updates.append(update)
                
                # Stop after 3 updates
                if len(updates) >= 3:
                    break
            
            # Verify updates
            assert len(updates) == 3
            assert updates[0]["type"] == "status_update"
            assert updates[0]["progress"] == 0.3
            assert updates[1]["type"] == "status_update"
            assert updates[1]["progress"] == 0.6
            assert updates[2]["type"] == "status_update"
            assert updates[2]["progress"] == 1.0
            assert updates[2]["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_websocket_service_disconnect(self, websocket_service):
        """Test WebSocket service disconnect."""
        # Mock the WebSocket connection
        with patch('websockets.connect', new_callable=AsyncMock) as mock_connect:
            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket
            
            # Connect to WebSocket
            await websocket_service.connect()
            
            # Verify connection state
            assert websocket_service.connected is True
            
            # Disconnect
            await websocket_service.disconnect()
            
            # Verify disconnect was called
            mock_websocket.close.assert_called_once()
            
            # Verify connection state
            assert websocket_service.connected is False
            assert websocket_service.websocket is None
    
    @pytest.mark.asyncio
    async def test_backend_api_websocket_broadcast(self, backend_api):
        """Test backend API WebSocket broadcast functionality."""
        # Create mock WebSocket connections
        mock_connection1 = AsyncMock()
        mock_connection2 = AsyncMock()
        
        # Add connections to the backend API
        backend_api.active_connections = [mock_connection1, mock_connection2]
        
        # Broadcast a message
        message = {
            "type": "benchmark_update",
            "benchmark_id": "test_benchmark_id",
            "status": "completed",
            "timestamp": datetime.now().isoformat()
        }
        
        await backend_api.broadcast_update(message)
        
        # Verify message was sent to all connections
        expected_message = json.dumps(message)
        mock_connection1.send_text.assert_called_once_with(expected_message)
        mock_connection2.send_text.assert_called_once_with(expected_message)
    
    @pytest.mark.asyncio
    async def test_end_to_end_benchmark_execution_with_real_time_updates(self, backend_api, benchmark_engine, mock_agent, test_scenario, test_metric):
        """Test end-to-end benchmark execution with real-time updates via WebSocket."""
        # Initialize and register components
        await mock_agent.initialize()
        benchmark_engine.register_agent(mock_agent)
        benchmark_engine.register_scenario(test_scenario)
        benchmark_engine.register_metric(test_metric)
        
        # Create a test client for the API
        test_client = TestClient(backend_api.app)
        
        # Create mock WebSocket connections
        mock_websocket = AsyncMock()
        
        # Add connection to the backend API
        backend_api.active_connections = [mock_websocket]
        
        # Run benchmark via API
        request_data = {
            "scenario_name": "test_scenario_for_integration",
            "agent_ids": ["test_agent"],
            "metric_names": ["test_metric_for_integration"]
        }
        
        response = test_client.post("/api/benchmarks/run", json=request_data)
        
        # Verify benchmark was started successfully
        assert response.status_code == 200
        result_data = response.json()
        assert result_data["success"] is True
        
        # Verify real-time updates were broadcast
        # The exact number of broadcasts depends on the implementation
        # but we should at least have some
        assert mock_websocket.send_text.call_count > 0
        
        # Verify the content of the broadcasts
        for call in mock_websocket.send_text.call_args_list:
            message = json.loads(call[0][0])
            assert "type" in message
            assert "timestamp" in message
    
    @pytest.mark.asyncio
    async def test_error_handling_in_frontend_backend_communication(self, api_service, websocket_service):
        """Test error handling in frontend-backend communication."""
        # Test API service error handling
        with patch.object(api_service, 'get', side_effect=Exception("Connection error")):
            with pytest.raises(Exception) as excinfo:
                await api_service.get_scenarios()
            assert "Connection error" in str(excinfo.value)
        
        # Test WebSocket service error handling
        with patch('websockets.connect', side_effect=Exception("WebSocket connection error")):
            with pytest.raises(Exception) as excinfo:
                await websocket_service.connect()
            assert "WebSocket connection error" in str(excinfo.value)


class MockResponse:
    """Mock HTTP response for testing."""
    
    def __init__(self, status_code, json_data):
        self.status_code = status_code
        self._json_data = json_data
    
    def json(self):
        return self._json_data