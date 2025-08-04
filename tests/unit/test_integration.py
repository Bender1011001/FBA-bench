"""
Unit tests for the integration components.

This module provides comprehensive tests for the integration framework components,
including the integration manager, agent adapter, and metrics adapter.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from datetime import datetime
from typing import Dict, Any, List

from benchmarking.integration.manager import (
    IntegrationManager,
    IntegrationStatus,
    IntegrationConfig,
    SimpleEventBus
)
from benchmarking.integration.agent_adapter import (
    AgentAdapter,
    AgentAdapterConfig,
    AgentExecutionResult,
    AgentAdapterFactory
)
from benchmarking.integration.metrics_adapter import (
    MetricsAdapter,
    MetricsAdapterConfig,
    MetricsAdapterResult,
    MetricsAdapterFactory
)
from benchmarking.core.engine import BenchmarkEngine, BenchmarkResult
from benchmarking.config.manager import ConfigurationManager
from benchmarking.metrics.registry import metrics_registry
from benchmarking.metrics.base import BaseMetric, MetricResult


class TestIntegrationStatus:
    """Test cases for IntegrationStatus class."""
    
    def test_integration_status_creation(self):
        """Test creating an integration status."""
        status = IntegrationStatus(
            component="test_component",
            available=True,
            version="1.0.0",
            capabilities=["capability1", "capability2"],
            issues=["issue1"]
        )
        
        assert status.component == "test_component"
        assert status.available is True
        assert status.version == "1.0.0"
        assert status.capabilities == ["capability1", "capability2"]
        assert status.issues == ["issue1"]
    
    def test_integration_status_defaults(self):
        """Test integration status with default values."""
        status = IntegrationStatus(
            component="test_component",
            available=True
        )
        
        assert status.version is None
        assert status.capabilities == []
        assert status.issues == []
    
    def test_integration_status_to_dict(self):
        """Test converting integration status to dictionary."""
        status = IntegrationStatus(
            component="test_component",
            available=True,
            version="1.0.0",
            capabilities=["capability1"],
            issues=["issue1"]
        )
        
        result = status.to_dict()
        
        expected = {
            "component": "test_component",
            "available": True,
            "version": "1.0.0",
            "capabilities": ["capability1"],
            "issues": ["issue1"]
        }
        
        assert result == expected


class TestIntegrationConfig:
    """Test cases for IntegrationConfig class."""
    
    def test_integration_config_creation(self):
        """Test creating an integration configuration."""
        config = IntegrationConfig(
            enable_agent_runners=False,
            enable_legacy_metrics=False,
            enable_infrastructure=False,
            enable_event_bus=False,
            enable_memory_systems=False,
            custom_integrations={"custom1": {"type": "test"}}
        )
        
        assert config.enable_agent_runners is False
        assert config.enable_legacy_metrics is False
        assert config.enable_infrastructure is False
        assert config.enable_event_bus is False
        assert config.enable_memory_systems is False
        assert config.custom_integrations == {"custom1": {"type": "test"}}
    
    def test_integration_config_defaults(self):
        """Test integration configuration with default values."""
        config = IntegrationConfig()
        
        assert config.enable_agent_runners is True
        assert config.enable_legacy_metrics is True
        assert config.enable_infrastructure is True
        assert config.enable_event_bus is True
        assert config.enable_memory_systems is True
        assert config.custom_integrations == {}
    
    def test_integration_config_to_dict(self):
        """Test converting integration configuration to dictionary."""
        config = IntegrationConfig(
            enable_agent_runners=False,
            custom_integrations={"custom1": {"type": "test"}}
        )
        
        result = config.to_dict()
        
        expected = {
            "enable_agent_runners": False,
            "enable_legacy_metrics": True,
            "enable_infrastructure": True,
            "enable_event_bus": True,
            "enable_memory_systems": True,
            "custom_integrations": {"custom1": {"type": "test"}}
        }
        
        assert result == expected


class TestSimpleEventBus:
    """Test cases for SimpleEventBus class."""
    
    @pytest.fixture
    def event_bus(self):
        """Create a simple event bus instance."""
        return SimpleEventBus()
    
    @pytest.mark.asyncio
    async def test_event_bus_publish(self, event_bus):
        """Test publishing an event."""
        event_type = "test_event"
        data = {"key": "value"}
        
        await event_bus.publish(event_type, data)
        
        history = event_bus.get_event_history()
        assert len(history) == 1
        assert history[0]["type"] == event_type
        assert history[0]["data"] == data
        assert "timestamp" in history[0]
    
    @pytest.mark.asyncio
    async def test_event_bus_subscribe_and_publish(self, event_bus):
        """Test subscribing to events and receiving notifications."""
        event_type = "test_event"
        data = {"key": "value"}
        received_events = []
        
        async def event_handler(event):
            received_events.append(event)
        
        await event_bus.subscribe(event_type, event_handler)
        await event_bus.publish(event_type, data)
        
        # Allow time for async processing
        await asyncio.sleep(0.01)
        
        assert len(received_events) == 1
        assert received_events[0]["type"] == event_type
        assert received_events[0]["data"] == data
    
    @pytest.mark.asyncio
    async def test_event_bus_multiple_subscribers(self, event_bus):
        """Test multiple subscribers to the same event type."""
        event_type = "test_event"
        data = {"key": "value"}
        received_events_1 = []
        received_events_2 = []
        
        async def handler_1(event):
            received_events_1.append(event)
        
        async def handler_2(event):
            received_events_2.append(event)
        
        await event_bus.subscribe(event_type, handler_1)
        await event_bus.subscribe(event_type, handler_2)
        await event_bus.publish(event_type, data)
        
        # Allow time for async processing
        await asyncio.sleep(0.01)
        
        assert len(received_events_1) == 1
        assert len(received_events_2) == 1
    
    @pytest.mark.asyncio
    async def test_event_bus_handler_exception(self, event_bus):
        """Test handling exceptions in event handlers."""
        event_type = "test_event"
        data = {"key": "value"}
        
        async def failing_handler(event):
            raise ValueError("Handler error")
        
        async def working_handler(event):
            pass
        
        await event_bus.subscribe(event_type, failing_handler)
        await event_bus.subscribe(event_type, working_handler)
        
        # Should not raise exception
        await event_bus.publish(event_type, data)
        
        # Allow time for async processing
        await asyncio.sleep(0.01)
        
        # Event should still be recorded
        history = event_bus.get_event_history()
        assert len(history) == 1
    
    def test_event_bus_get_event_history_filtered(self, event_bus):
        """Test getting filtered event history."""
        # Add events directly for testing
        event_bus._event_history = [
            {"type": "event1", "data": {}, "timestamp": "2023-01-01T00:00:00"},
            {"type": "event2", "data": {}, "timestamp": "2023-01-01T00:00:01"},
            {"type": "event1", "data": {}, "timestamp": "2023-01-01T00:00:02"}
        ]
        
        # Get all events
        all_events = event_bus.get_event_history()
        assert len(all_events) == 3
        
        # Get filtered events
        filtered_events = event_bus.get_event_history("event1")
        assert len(filtered_events) == 2
        assert all(event["type"] == "event1" for event in filtered_events)


class TestIntegrationManager:
    """Test cases for IntegrationManager class."""
    
    @pytest.fixture
    def integration_config(self):
        """Create a test integration configuration."""
        return IntegrationConfig(
            enable_agent_runners=True,
            enable_legacy_metrics=True,
            enable_infrastructure=True,
            enable_event_bus=True,
            enable_memory_systems=True
        )
    
    @pytest.fixture
    def integration_manager(self, integration_config):
        """Create an integration manager instance."""
        return IntegrationManager(integration_config)
    
    def test_integration_manager_initialization(self, integration_config):
        """Test integration manager initialization."""
        manager = IntegrationManager(integration_config)
        
        assert manager.config == integration_config
        assert manager.status == {}
        assert manager._initialized is False
        assert manager.benchmark_engine is None
        assert manager.config_manager is None
        assert manager.legacy_metric_suite is None
        assert manager.deployment_manager is None
        assert manager._event_handlers == {}
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize(self, integration_manager):
        """Test integration manager initialization."""
        await integration_manager.initialize()
        
        assert integration_manager._initialized is True
        assert "agent_runners" in integration_manager.status
        assert "legacy_metrics" in integration_manager.status
        assert "infrastructure" in integration_manager.status
        assert "event_bus" in integration_manager.status
        assert "memory_systems" in integration_manager.status
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_already_initialized(self, integration_manager):
        """Test integration manager initialization when already initialized."""
        integration_manager._initialized = True
        
        # Should not raise exception
        await integration_manager.initialize()
        
        assert integration_manager._initialized is True
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_agent_runners_success(self, integration_manager):
        """Test successful agent runners integration initialization."""
        with patch('benchmarking.integration.manager.AGENT_RUNNERS_AVAILABLE', True):
            with patch('benchmarking.integration.manager.RunnerFactory') as mock_factory:
                mock_factory.list_runners.return_value = ["diy", "crewai"]
                
                await integration_manager._initialize_agent_runners_integration()
                
                status = integration_manager.status["agent_runners"]
                assert status.available is True
                assert status.version == "integrated"
                assert "runner_diy" in status.capabilities
                assert "runner_crewai" in status.capabilities
                assert len(status.issues) == 0
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_agent_runners_unavailable(self, integration_manager):
        """Test agent runners integration initialization when unavailable."""
        with patch('benchmarking.integration.manager.AGENT_RUNNERS_AVAILABLE', False):
            await integration_manager._initialize_agent_runners_integration()
            
            status = integration_manager.status["agent_runners"]
            assert status.available is False
            assert "agent_runners module not available" in status.issues
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_agent_runners_exception(self, integration_manager):
        """Test agent runners integration initialization with exception."""
        with patch('benchmarking.integration.manager.AGENT_RUNNERS_AVAILABLE', True):
            with patch('benchmarking.integration.manager.RunnerFactory') as mock_factory:
                mock_factory.list_runners.side_effect = Exception("Test error")
                
                await integration_manager._initialize_agent_runners_integration()
                
                status = integration_manager.status["agent_runners"]
                assert status.available is True  # Module is available, but initialization failed
                assert "Failed to initialize agent runners: Test error" in status.issues
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_legacy_metrics_success(self, integration_manager):
        """Test successful legacy metrics integration initialization."""
        with patch('benchmarking.integration.manager.METRICS_AVAILABLE', True):
            with patch('benchmarking.integration.manager.MetricSuite') as mock_suite:
                mock_suite.return_value = Mock()
                
                await integration_manager._initialize_legacy_metrics_integration()
                
                status = integration_manager.status["legacy_metrics"]
                assert status.available is True
                assert status.version == "integrated"
                assert "finance_metrics" in status.capabilities
                assert "cognitive_metrics" in status.capabilities
                assert len(status.issues) == 0
                assert integration_manager.legacy_metric_suite is not None
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_legacy_metrics_unavailable(self, integration_manager):
        """Test legacy metrics integration initialization when unavailable."""
        with patch('benchmarking.integration.manager.METRICS_AVAILABLE', False):
            await integration_manager._initialize_legacy_metrics_integration()
            
            status = integration_manager.status["legacy_metrics"]
            assert status.available is False
            assert "metrics module not available" in status.issues
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_infrastructure_success(self, integration_manager):
        """Test successful infrastructure integration initialization."""
        with patch('benchmarking.integration.manager.INFRASTRUCTURE_AVAILABLE', True):
            with patch('benchmarking.integration.manager.DeploymentManager') as mock_manager:
                mock_manager.return_value = Mock()
                
                await integration_manager._initialize_infrastructure_integration()
                
                status = integration_manager.status["infrastructure"]
                assert status.available is True
                assert status.version == "integrated"
                assert "deployment" in status.capabilities
                assert "scaling" in status.capabilities
                assert len(status.issues) == 0
                assert integration_manager.deployment_manager is not None
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_event_bus_success(self, integration_manager):
        """Test successful event bus integration initialization."""
        await integration_manager._initialize_event_bus_integration()
        
        status = integration_manager.status["event_bus"]
        assert status.available is True
        assert status.version == "integrated"
        assert "publish_subscribe" in status.capabilities
        assert "event_routing" in status.capabilities
        assert len(status.issues) == 0
        assert hasattr(integration_manager, '_event_bus')
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_memory_systems_success(self, integration_manager):
        """Test successful memory systems integration initialization."""
        with patch('benchmarking.integration.manager.DualMemoryManager') as mock_manager:
            with patch('benchmarking.integration.manager.MemoryConfig') as mock_config:
                await integration_manager._initialize_memory_systems_integration()
                
                status = integration_manager.status["memory_systems"]
                assert status.available is True
                assert status.version == "integrated"
                assert "dual_memory" in status.capabilities
                assert "memory_config" in status.capabilities
                assert len(status.issues) == 0
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_memory_systems_unavailable(self, integration_manager):
        """Test memory systems integration initialization when unavailable."""
        with patch('benchmarking.integration.manager.DualMemoryManager', side_effect=ImportError):
            await integration_manager._initialize_memory_systems_integration()
            
            status = integration_manager.status["memory_systems"]
            assert status.available is False
            assert "memory_experiments module not available" in status.issues
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_custom_integrations_success(self, integration_manager):
        """Test successful custom integrations initialization."""
        integration_manager.config.custom_integrations = {
            "custom1": {
                "type": "test",
                "module": "test_module",
                "capabilities": ["cap1", "cap2"],
                "version": "1.0"
            }
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.initialize_integration = AsyncMock()
            mock_import.return_value = mock_module
            
            await integration_manager._initialize_custom_integrations()
            
            status = integration_manager.status["custom_custom1"]
            assert status.available is True
            assert status.version == "1.0"
            assert status.capabilities == ["cap1", "cap2"]
            assert len(status.issues) == 0
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_custom_integrations_missing_function(self, integration_manager):
        """Test custom integrations initialization with missing function."""
        integration_manager.config.custom_integrations = {
            "custom1": {
                "type": "test",
                "module": "test_module"
            }
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            del mock_module.initialize_integration  # Remove the function
            mock_import.return_value = mock_module
            
            await integration_manager._initialize_custom_integrations()
            
            status = integration_manager.status["custom_custom1"]
            assert status.available is False
            assert "missing initialize_integration function" in status.issues
    
    def test_integration_manager_get_integration_status(self, integration_manager):
        """Test getting integration status."""
        integration_manager.status = {
            "component1": IntegrationStatus("component1", available=True),
            "component2": IntegrationStatus("component2", available=False)
        }
        
        status = integration_manager.get_integration_status()
        
        assert len(status) == 2
        assert "component1" in status
        assert "component2" in status
        assert status["component1"].available is True
        assert status["component2"].available is False
    
    def test_integration_manager_is_integration_available(self, integration_manager):
        """Test checking if integration is available."""
        integration_manager.status = {
            "component1": IntegrationStatus("component1", available=True),
            "component2": IntegrationStatus("component2", available=False)
        }
        
        assert integration_manager.is_integration_available("component1") is True
        assert integration_manager.is_integration_available("component2") is False
        assert integration_manager.is_integration_available("nonexistent") is False
    
    def test_integration_manager_get_integration_capabilities(self, integration_manager):
        """Test getting integration capabilities."""
        integration_manager.status = {
            "component1": IntegrationStatus(
                "component1", 
                available=True, 
                capabilities=["cap1", "cap2"]
            ),
            "component2": IntegrationStatus(
                "component2", 
                available=False, 
                capabilities=[]
            )
        }
        
        caps1 = integration_manager.get_integration_capabilities("component1")
        caps2 = integration_manager.get_integration_capabilities("component2")
        caps3 = integration_manager.get_integration_capabilities("nonexistent")
        
        assert caps1 == ["cap1", "cap2"]
        assert caps2 == []
        assert caps3 == []
    
    @pytest.mark.asyncio
    async def test_integration_manager_create_agent_runner_success(self, integration_manager):
        """Test successful agent runner creation."""
        with patch('benchmarking.integration.manager.AGENT_RUNNERS_AVAILABLE', True):
            with patch('benchmarking.integration.manager.RunnerFactory') as mock_factory:
                mock_runner = Mock()
                mock_factory.create_and_initialize_runner.return_value = mock_runner
                
                runner = await integration_manager.create_agent_runner(
                    "diy", "test_agent", {"param": "value"}
                )
                
                assert runner == mock_runner
                mock_factory.create_and_initialize_runner.assert_called_once_with(
                    "diy", "test_agent", {"param": "value"}
                )
    
    @pytest.mark.asyncio
    async def test_integration_manager_create_agent_runner_unavailable(self, integration_manager):
        """Test agent runner creation when unavailable."""
        with patch('benchmarking.integration.manager.AGENT_RUNNERS_AVAILABLE', False):
            runner = await integration_manager.create_agent_runner(
                "diy", "test_agent", {"param": "value"}
            )
            
            assert runner is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_create_agent_runner_exception(self, integration_manager):
        """Test agent runner creation with exception."""
        with patch('benchmarking.integration.manager.AGENT_RUNNERS_AVAILABLE', True):
            with patch('benchmarking.integration.manager.RunnerFactory') as mock_factory:
                mock_factory.create_and_initialize_runner.side_effect = Exception("Test error")
                
                runner = await integration_manager.create_agent_runner(
                    "diy", "test_agent", {"param": "value"}
                )
                
                assert runner is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_legacy_metrics_success(self, integration_manager):
        """Test successful legacy metrics run."""
        integration_manager.legacy_metric_suite = Mock()
        integration_manager.legacy_metric_suite.calculate_kpis.return_value = {
            "overall_score": 0.8,
            "breakdown": {"finance": 0.9},
            "timestamp": "2023-01-01T00:00:00",
            "tick_number": 1
        }
        
        events = [
            {"type": "SaleOccurred", "data": {"amount": 100}},
            {"type": "SetPriceCommand", "data": {"price": 50}}
        ]
        
        result = await integration_manager.run_legacy_metrics(1, events)
        
        assert result is not None
        assert result["overall_score"] == 0.8
        assert result["tick_number"] == 1
        
        # Verify events were processed
        assert integration_manager.legacy_metric_suite._handle_general_event.call_count == 2
        integration_manager.legacy_metric_suite.calculate_kpis.assert_called_once_with(1)
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_legacy_metrics_unavailable(self, integration_manager):
        """Test legacy metrics run when unavailable."""
        integration_manager.legacy_metric_suite = None
        
        result = await integration_manager.run_legacy_metrics(1, [])
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_legacy_metrics_exception(self, integration_manager):
        """Test legacy metrics run with exception."""
        integration_manager.legacy_metric_suite = Mock()
        integration_manager.legacy_metric_suite.calculate_kpis.side_effect = Exception("Test error")
        
        result = await integration_manager.run_legacy_metrics(1, [])
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_deploy_benchmark_success(self, integration_manager):
        """Test successful benchmark deployment."""
        integration_manager.deployment_manager = Mock()
        integration_manager.deployment_manager.deploy.return_value = "deployment_123"
        
        config = {
            "benchmark_id": "test_benchmark",
            "environment": {
                "max_workers": 4,
                "parallel_execution": True
            }
        }
        
        deployment_id = await integration_manager.deploy_benchmark(config)
        
        assert deployment_id == "deployment_123"
        
        # Verify deployment configuration
        call_args = integration_manager.deployment_manager.deploy.call_args[0][0]
        assert call_args["name"] == "test_benchmark"
        assert call_args["type"] == "benchmark"
        assert call_args["resources"]["cpu"] == 4
        assert call_args["scaling"]["enabled"] is True
    
    @pytest.mark.asyncio
    async def test_integration_manager_deploy_benchmark_unavailable(self, integration_manager):
        """Test benchmark deployment when unavailable."""
        integration_manager.deployment_manager = None
        
        deployment_id = await integration_manager.deploy_benchmark({})
        
        assert deployment_id is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_deploy_benchmark_exception(self, integration_manager):
        """Test benchmark deployment with exception."""
        integration_manager.deployment_manager = Mock()
        integration_manager.deployment_manager.deploy.side_effect = Exception("Test error")
        
        deployment_id = await integration_manager.deploy_benchmark({})
        
        assert deployment_id is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_publish_event_success(self, integration_manager):
        """Test successful event publishing."""
        integration_manager._event_bus = Mock()
        integration_manager._event_bus.publish = AsyncMock()
        
        await integration_manager.publish_event("test_event", {"key": "value"})
        
        integration_manager._event_bus.publish.assert_called_once_with(
            "test_event", {"key": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_integration_manager_publish_event_unavailable(self, integration_manager):
        """Test event publishing when unavailable."""
        integration_manager._event_bus = None
        
        # Should not raise exception
        await integration_manager.publish_event("test_event", {"key": "value"})
    
    @pytest.mark.asyncio
    async def test_integration_manager_subscribe_to_event_success(self, integration_manager):
        """Test successful event subscription."""
        integration_manager._event_bus = Mock()
        integration_manager._event_bus.subscribe = AsyncMock()
        
        async def handler(event):
            pass
        
        await integration_manager.subscribe_to_event("test_event", handler)
        
        integration_manager._event_bus.subscribe.assert_called_once_with(
            "test_event", handler
        )
    
    @pytest.mark.asyncio
    async def test_integration_manager_subscribe_to_event_unavailable(self, integration_manager):
        """Test event subscription when unavailable."""
        integration_manager._event_bus = None
        
        async def handler(event):
            pass
        
        # Should not raise exception
        await integration_manager.subscribe_to_event("test_event", handler)
    
    def test_integration_manager_set_benchmark_engine(self, integration_manager):
        """Test setting benchmark engine."""
        engine = Mock(spec=BenchmarkEngine)
        
        integration_manager.set_benchmark_engine(engine)
        
        assert integration_manager.benchmark_engine == engine
    
    def test_integration_manager_set_config_manager(self, integration_manager):
        """Test setting configuration manager."""
        manager = Mock(spec=ConfigurationManager)
        
        integration_manager.set_config_manager(manager)
        
        assert integration_manager.config_manager == manager
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_integrated_benchmark_success(self, integration_manager):
        """Test successful integrated benchmark run."""
        # Setup mocks
        integration_manager._initialized = True
        integration_manager.benchmark_engine = Mock(spec=BenchmarkEngine)
        integration_manager.benchmark_engine.run_benchmark = AsyncMock()
        
        benchmark_result = Mock(spec=BenchmarkResult)
        benchmark_result.metadata = {}
        integration_manager.benchmark_engine.run_benchmark.return_value = benchmark_result
        
        integration_manager.status = {
            "infrastructure": IntegrationStatus("infrastructure", available=True)
        }
        integration_manager.deployment_manager = Mock()
        integration_manager.deployment_manager.deploy = AsyncMock(return_value="deployment_123")
        
        config = {"benchmark_id": "test_benchmark"}
        
        result = await integration_manager.run_integrated_benchmark(config)
        
        assert result == benchmark_result
        assert result.metadata["integration_status"]["infrastructure"]["available"] is True
        assert result.metadata["deployment_id"] == "deployment_123"
        
        integration_manager.benchmark_engine.run_benchmark.assert_called_once_with(config)
        integration_manager.deployment_manager.deploy.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_integrated_benchmark_not_initialized(self, integration_manager):
        """Test integrated benchmark run when not initialized."""
        integration_manager._initialized = False
        integration_manager.benchmark_engine = Mock(spec=BenchmarkEngine)
        integration_manager.benchmark_engine.run_benchmark = AsyncMock()
        
        # Should initialize automatically
        with patch.object(integration_manager, 'initialize', AsyncMock()) as mock_init:
            mock_init.return_value = None
            
            await integration_manager.run_integrated_benchmark({})
            
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_integrated_benchmark_no_engine(self, integration_manager):
        """Test integrated benchmark run with no engine."""
        integration_manager._initialized = True
        integration_manager.benchmark_engine = None
        
        result = await integration_manager.run_integrated_benchmark({})
        
        assert result is None
    
    @pytest.mark.asyncio
    async def test_integration_manager_run_integrated_benchmark_exception(self, integration_manager):
        """Test integrated benchmark run with exception."""
        integration_manager._initialized = True
        integration_manager.benchmark_engine = Mock(spec=BenchmarkEngine)
        integration_manager.benchmark_engine.run_benchmark.side_effect = Exception("Test error")
        
        result = await integration_manager.run_integrated_benchmark({})
        
        assert result is None


class TestAgentAdapterConfig:
    """Test cases for AgentAdapterConfig class."""
    
    def test_agent_adapter_config_creation(self):
        """Test creating an agent adapter configuration."""
        config = AgentAdapterConfig(
            framework="diy",
            agent_id="test_agent",
            config={"param": "value"},
            timeout=600,
            retry_attempts=5,
            enable_monitoring=False,
            enable_tracing=False
        )
        
        assert config.framework == "diy"
        assert config.agent_id == "test_agent"
        assert config.config == {"param": "value"}
        assert config.timeout == 600
        assert config.retry_attempts == 5
        assert config.enable_monitoring is False
        assert config.enable_tracing is False
    
    def test_agent_adapter_config_defaults(self):
        """Test agent adapter configuration with default values."""
        config = AgentAdapterConfig(
            framework="diy",
            agent_id="test_agent"
        )
        
        assert config.config == {}
        assert config.timeout == 300
        assert config.retry_attempts == 3
        assert config.enable_monitoring is True
        assert config.enable_tracing is True
    
    def test_agent_adapter_config_to_dict(self):
        """Test converting agent adapter configuration to dictionary."""
        config = AgentAdapterConfig(
            framework="diy",
            agent_id="test_agent",
            config={"param": "value"},
            timeout=600
        )
        
        result = config.to_dict()
        
        expected = {
            "framework": "diy",
            "agent_id": "test_agent",
            "config": {"param": "value"},
            "timeout": 600,
            "retry_attempts": 3,
            "enable_monitoring": True,
            "enable_tracing": True
        }
        
        assert result == expected


class TestAgentExecutionResult:
    """Test cases for AgentExecutionResult class."""
    
    def test_agent_execution_result_creation(self):
        """Test creating an agent execution result."""
        tool_call = Mock()
        tool_call.tool_name = "test_tool"
        tool_call.parameters = {"param": "value"}
        tool_call.confidence = 0.9
        tool_call.reasoning = "test reasoning"
        tool_call.priority = 1
        
        result = AgentExecutionResult(
            agent_id="test_agent",
            framework="diy",
            success=True,
            tool_calls=[tool_call],
            execution_time=1.5,
            error_message=None,
            metrics={"accuracy": 0.8},
            trace_data={"step": "test"}
        )
        
        assert result.agent_id == "test_agent"
        assert result.framework == "diy"
        assert result.success is True
        assert result.tool_calls == [tool_call]
        assert result.execution_time == 1.5
        assert result.error_message is None
        assert result.metrics == {"accuracy": 0.8}
        assert result.trace_data == {"step": "test"}
    
    def test_agent_execution_result_defaults(self):
        """Test agent execution result with default values."""
        result = AgentExecutionResult(
            agent_id="test_agent",
            framework="diy",
            success=True
        )
        
        assert result.tool_calls == []
        assert result.execution_time == 0.0
        assert result.error_message is None
        assert result.metrics == {}
        assert result.trace_data == {}
    
    def test_agent_execution_result_to_dict(self):
        """Test converting agent execution result to dictionary."""
        tool_call = Mock()
        tool_call.tool_name = "test_tool"
        tool_call.parameters = {"param": "value"}
        tool_call.confidence = 0.9
        tool_call.reasoning = "test reasoning"
        tool_call.priority = 1
        
        result = AgentExecutionResult(
            agent_id="test_agent",
            framework="diy",
            success=True,
            tool_calls=[tool_call]
        )
        
        dict_result = result.to_dict()
        
        assert dict_result["agent_id"] == "test_agent"
        assert dict_result["framework"] == "diy"
        assert dict_result["success"] is True
        assert len(dict_result["tool_calls"]) == 1
        assert dict_result["tool_calls"][0]["tool_name"] == "test_tool"
        assert dict_result["tool_calls"][0]["parameters"] == {"param": "value"}
        assert dict_result["tool_calls"][0]["confidence"] == 0.9
        assert dict_result["tool_calls"][0]["reasoning"] == "test reasoning"
        assert dict_result["tool_calls"][0]["priority"] == 1


class MockAgentRunner:
    """Mock agent runner for testing."""
    
    def __init__(self):
        self.initialized = False
        self.health_status = {"status": "healthy"}
    
    async def initialize(self, config):
        """Initialize the mock agent runner."""
        self.initialized = True
    
    async def decide(self, simulation_state):
        """Make a decision."""
        return [
            Mock(
                tool_name="test_tool",
                parameters={"param": "value"},
                confidence=0.9,
                reasoning="test reasoning",
                priority=1
            )
        ]
    
    async def health_check(self):
        """Perform health check."""
        return self.health_status
    
    async def cleanup(self):
        """Cleanup resources."""
        self.initialized = False


class TestAgentAdapter:
    """Test cases for AgentAdapter class."""
    
    @pytest.fixture
    def agent_adapter_config(self):
        """Create a test agent adapter configuration."""
        return AgentAdapterConfig(
            framework="diy",
            agent_id="test_agent",
            config={"param": "value"},
            timeout=10,
            retry_attempts=2
        )
    
    @pytest.fixture
    def integration_manager(self):
        """Create a mock integration manager."""
        manager = Mock(spec=IntegrationManager)
        manager.create_agent_runner = AsyncMock()
        manager.publish_event = AsyncMock()
        return manager
    
    @pytest.fixture
    def agent_adapter(self, agent_adapter_config, integration_manager):
        """Create an agent adapter instance."""
        return AgentAdapter(agent_adapter_config, integration_manager)
    
    def test_agent_adapter_initialization(self, agent_adapter_config, integration_manager):
        """Test agent adapter initialization."""
        adapter = AgentAdapter(agent_adapter_config, integration_manager)
        
        assert adapter.config == agent_adapter_config
        assert adapter.integration_manager == integration_manager
        assert adapter.agent_runner is None
        assert adapter._initialized is False
        assert adapter._execution_history == []
        assert adapter._current_trace == {}
    
    @pytest.mark.asyncio
    async def test_agent_adapter_initialize_success(self, agent_adapter):
        """Test successful agent adapter initialization."""
        mock_runner = MockAgentRunner()
        agent_adapter.integration_manager.create_agent_runner.return_value = mock_runner
        
        result = await agent_adapter.initialize()
        
        assert result is True
        assert agent_adapter._initialized is True
        assert agent_adapter.agent_runner == mock_runner
        assert mock_runner.initialized is True
        
        agent_adapter.integration_manager.create_agent_runner.assert_called_once_with(
            "diy", "test_agent", {"param": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_agent_adapter_initialize_already_initialized(self, agent_adapter):
        """Test agent adapter initialization when already initialized."""
        agent_adapter._initialized = True
        
        result = await agent_adapter.initialize()
        
        assert result is True
        # Should not try to create agent runner again
        agent_adapter.integration_manager.create_agent_runner.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_agent_adapter_initialize_agent_runners_unavailable(self, agent_adapter):
        """Test agent adapter initialization when agent runners unavailable."""
        with patch('benchmarking.integration.agent_adapter.AGENT_RUNNERS_AVAILABLE', False):
            result = await agent_adapter.initialize()
            
            assert result is False
            assert agent_adapter._initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_adapter_initialize_create_runner_failure(self, agent_adapter):
        """Test agent adapter initialization when runner creation fails."""
        agent_adapter.integration_manager.create_agent_runner.return_value = None
        
        result = await agent_adapter.initialize()
        
        assert result is False
        assert agent_adapter._initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_adapter_initialize_runner_init_failure(self, agent_adapter):
        """Test agent adapter initialization when runner initialization fails."""
        mock_runner = MockAgentRunner()
        mock_runner.initialize = AsyncMock(side_effect=Exception("Test error"))
        agent_adapter.integration_manager.create_agent_runner.return_value = mock_runner
        
        result = await agent_adapter.initialize()
        
        assert result is False
        assert agent_adapter._initialized is False
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_decision_success(self, agent_adapter):
        """Test successful decision execution."""
        # Setup
        mock_runner = MockAgentRunner()
        agent_adapter.agent_runner = mock_runner
        agent_adapter._initialized = True
        
        simulation_state = {
            "tick": 1,
            "simulation_time": "2023-01-01T00:00:00",
            "products": [],
            "recent_events": [],
            "financial_position": {},
            "market_conditions": {},
            "agent_state": {}
        }
        
        # Execute
        result = await agent_adapter.execute_decision(simulation_state)
        
        # Verify
        assert result.success is True
        assert result.agent_id == "test_agent"
        assert result.framework == "diy"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].tool_name == "test_tool"
        assert result.execution_time > 0
        assert result.metrics == {"total_executions": 1, "successful_executions": 1, "failed_executions": 0, "average_execution_time": result.execution_time, "success_rate": 1.0}
        assert len(agent_adapter._execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_decision_not_initialized(self, agent_adapter):
        """Test decision execution when not initialized."""
        agent_adapter._initialized = False
        agent_adapter.initialize = AsyncMock(return_value=False)
        
        simulation_state = {}
        
        result = await agent_adapter.execute_decision(simulation_state)
        
        assert result.success is False
        assert result.error_message == "Agent adapter not initialized"
        assert len(agent_adapter._execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_decision_exception(self, agent_adapter):
        """Test decision execution with exception."""
        # Setup
        mock_runner = MockAgentRunner()
        mock_runner.decide = AsyncMock(side_effect=Exception("Test error"))
        agent_adapter.agent_runner = mock_runner
        agent_adapter._initialized = True
        
        simulation_state = {}
        
        # Execute
        result = await agent_adapter.execute_decision(simulation_state)
        
        # Verify
        assert result.success is False
        assert result.error_message == "Test error"
        assert result.execution_time > 0
        assert len(agent_adapter._execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_decision_timeout(self, agent_adapter):
        """Test decision execution with timeout."""
        # Setup
        mock_runner = MockAgentRunner()
        mock_runner.decide = AsyncMock(side_effect=asyncio.TimeoutError())
        agent_adapter.agent_runner = mock_runner
        agent_adapter._initialized = True
        
        simulation_state = {}
        
        # Execute
        result = await agent_adapter.execute_decision(simulation_state)
        
        # Verify
        assert result.success is False
        assert "Agent decision failed after 2 attempts" in result.error_message
        assert result.execution_time > 0
        assert len(agent_adapter._execution_history) == 1
    
    @pytest.mark.asyncio
    async def test_agent_adapter_execute_decision_retry_success(self, agent_adapter):
        """Test decision execution with retry success."""
        # Setup
        mock_runner = MockAgentRunner()
        mock_runner.decide = AsyncMock(side_effect=[Exception("First error"), mock_runner.decide()])
        agent_adapter.agent_runner = mock_runner
        agent_adapter._initialized = True
        
        simulation_state = {}
        
        # Execute
        result = await agent_adapter.execute_decision(simulation_state)
        
        # Verify
        assert result.success is True
        assert len(result.tool_calls) == 1
        assert result.execution_time > 0
        assert len(agent_adapter._execution_history) == 1
    
    def test_agent_adapter_convert_to_simulation_state_without_models(self, agent_adapter):
        """Test converting dictionary to simulation state without models."""
        with patch('benchmarking.integration.agent_adapter.MODELS_AVAILABLE', False):
            state_dict = {
                "tick": 1,
                "simulation_time": "2023-01-01T00:00:00",
                "products": [
                    {"asin": "B001", "name": "Test Product", "price": 10.0}
                ],
                "recent_events": ["event1"],
                "financial_position": {"cash": 1000},
                "market_conditions": {"demand": 0.8},
                "agent_state": {"position": "manager"}
            }
            
            sim_state = agent_adapter._convert_to_simulation_state(state_dict)
            
            assert sim_state.tick == 1
            assert sim_state.simulation_time.isoformat() == "2023-01-01T00:00:00"
            assert sim_state.products == []  # Empty without Product objects
            assert sim_state.recent_events == ["event1"]
            assert sim_state.financial_position == {"cash": 1000}
            assert sim_state.market_conditions == {"demand": 0.8}
            assert sim_state.agent_state == {"position": "manager"}
    
    def test_agent_adapter_convert_to_simulation_state_with_models(self, agent_adapter):
        """Test converting dictionary to simulation state with models."""
        with patch('benchmarking.integration.agent_adapter.MODELS_AVAILABLE', True):
            with patch('benchmarking.integration.agent_adapter.Product') as mock_product:
                mock_product.return_value = Mock()
                
                state_dict = {
                    "tick": 1,
                    "products": [
                        {"asin": "B001", "name": "Test Product", "price": 10.0}
                    ]
                }
                
                sim_state = agent_adapter._convert_to_simulation_state(state_dict)
                
                assert sim_state.tick == 1
                assert len(sim_state.products) == 1
                mock_product.assert_called_once_with(
                    asin="B001", name="Test Product", price=10.0, category="", brand=""
                )
    
    def test_agent_adapter_collect_metrics_disabled(self, agent_adapter):
        """Test metrics collection when disabled."""
        agent_adapter.config.enable_monitoring = False
        
        metrics = agent_adapter._collect_metrics()
        
        assert metrics == {}
    
    def test_agent_adapter_collect_metrics_enabled_no_history(self, agent_adapter):
        """Test metrics collection when enabled with no history."""
        agent_adapter.config.enable_monitoring = True
        agent_adapter._execution_history = []
        
        metrics = agent_adapter._collect_metrics()
        
        expected = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "average_execution_time": 0.0,
            "success_rate": 0.0
        }
        
        assert metrics == expected
    
    def test_agent_adapter_collect_metrics_enabled_with_history(self, agent_adapter):
        """Test metrics collection when enabled with history."""
        agent_adapter.config.enable_monitoring = True
        agent_adapter._execution_history = [
            AgentExecutionResult("agent1", "diy", success=True, execution_time=1.0),
            AgentExecutionResult("agent2", "diy", success=False, execution_time=2.0),
            AgentExecutionResult("agent3", "diy", success=True, execution_time=3.0)
        ]
        
        metrics = agent_adapter._collect_metrics()
        
        expected = {
            "total_executions": 3,
            "successful_executions": 2,
            "failed_executions": 1,
            "average_execution_time": 2.0,  # (1.0 + 3.0) / 2
            "success_rate": 2/3
        }
        
        assert metrics == expected
    
    def test_agent_adapter_start_trace_disabled(self, agent_adapter):
        """Test starting trace when disabled."""
        agent_adapter.config.enable_tracing = False
        
        agent_adapter._start_trace("test_operation")
        
        assert agent_adapter._current_trace == {}
    
    def test_agent_adapter_start_trace_enabled(self, agent_adapter):
        """Test starting trace when enabled."""
        agent_adapter.config.enable_tracing = True
        
        agent_adapter._start_trace("test_operation")
        
        assert agent_adapter._current_trace["operation"] == "test_operation"
        assert "start_time" in agent_adapter._current_trace
        assert agent_adapter._current_trace["agent_id"] == "test_agent"
        assert agent_adapter._current_trace["framework"] == "diy"
        assert agent_adapter._current_trace["steps"] == []
    
    def test_agent_adapter_end_trace_disabled(self, agent_adapter):
        """Test ending trace when disabled."""
        agent_adapter.config.enable_tracing = False
        agent_adapter._current_trace = {"operation": "test"}
        
        agent_adapter._end_trace("success")
        
        # Should not publish event
        agent_adapter.integration_manager.publish_event.assert_not_called()
    
    def test_agent_adapter_end_trace_enabled(self, agent_adapter):
        """Test ending trace when enabled."""
        agent_adapter.config.enable_tracing = True
        agent_adapter._current_trace = {"operation": "test"}
        
        agent_adapter._end_trace("success")
        
        assert "end_time" in agent_adapter._current_trace
        assert agent_adapter._current_trace["status"] == "success"
        
        # Should publish event
        agent_adapter.integration_manager.publish_event.assert_called_once_with(
            "agent_trace", agent_adapter._current_trace
        )
    
    def test_agent_adapter_end_trace_with_error(self, agent_adapter):
        """Test ending trace with error."""
        agent_adapter.config.enable_tracing = True
        agent_adapter._current_trace = {"operation": "test"}
        
        agent_adapter._end_trace("error", "Test error")
        
        assert agent_adapter._current_trace["status"] == "error"
        assert agent_adapter._current_trace["error"] == "Test error"
    
    def test_agent_adapter_get_execution_history(self, agent_adapter):
        """Test getting execution history."""
        result1 = AgentExecutionResult("agent1", "diy", success=True)
        result2 = AgentExecutionResult("agent2", "diy", success=False)
        
        agent_adapter._execution_history = [result1, result2]
        
        history = agent_adapter.get_execution_history()
        
        assert len(history) == 2
        assert history[0] == result1
        assert history[1] == result2
        # Should return a copy
        assert history is not agent_adapter._execution_history
    
    def test_agent_adapter_get_metrics(self, agent_adapter):
        """Test getting agent metrics."""
        agent_adapter.config.enable_monitoring = True
        result = AgentExecutionResult("agent1", "diy", success=True, execution_time=1.0)
        agent_adapter._execution_history = [result]
        
        metrics = agent_adapter.get_metrics()
        
        expected = {
            "total_executions": 1,
            "successful_executions": 1,
            "failed_executions": 0,
            "average_execution_time": 1.0,
            "success_rate": 1.0
        }
        
        assert metrics == expected
    
    @pytest.mark.asyncio
    async def test_agent_adapter_health_check_not_initialized(self, agent_adapter):
        """Test health check when not initialized."""
        agent_adapter._initialized = False
        
        health = await agent_adapter.health_check()
        
        assert health["agent_id"] == "test_agent"
        assert health["framework"] == "diy"
        assert health["initialized"] is False
        assert health["healthy"] is False
        assert "Agent adapter not initialized" in health["issues"]
    
    @pytest.mark.asyncio
    async def test_agent_adapter_health_check_no_runner(self, agent_adapter):
        """Test health check with no runner."""
        agent_adapter._initialized = True
        agent_adapter.agent_runner = None
        
        health = await agent_adapter.health_check()
        
        assert health["initialized"] is True
        assert health["healthy"] is False
        assert "Agent runner not available" in health["issues"]
    
    @pytest.mark.asyncio
    async def test_agent_adapter_health_check_healthy(self, agent_adapter):
        """Test healthy health check."""
        agent_adapter._initialized = True
        agent_adapter.agent_runner = MockAgentRunner()
        
        health = await agent_adapter.health_check()
        
        assert health["initialized"] is True
        assert health["healthy"] is True
        assert health["agent_health"]["status"] == "healthy"
        assert len(health["issues"]) == 0
    
    @pytest.mark.asyncio
    async def test_agent_adapter_health_check_high_failure_rate(self, agent_adapter):
        """Test health check with high failure rate."""
        agent_adapter._initialized = True
        agent_adapter.agent_runner = MockAgentRunner()
        
        # Add many recent failures
        for _ in range(6):
            result = AgentExecutionResult("agent1", "diy", success=False)
            agent_adapter._execution_history.append(result)
        
        health = await agent_adapter.health_check()
        
        assert health["initialized"] is True
        assert health["healthy"] is False
        assert "High failure rate: 6/10 recent executions failed" in health["issues"]
    
    @pytest.mark.asyncio
    async def test_agent_adapter_health_check_exception(self, agent_adapter):
        """Test health check with exception."""
        agent_adapter._initialized = True
        agent_adapter.agent_runner = MockAgentRunner()
        agent_adapter.agent_runner.health_check = AsyncMock(side_effect=Exception("Test error"))
        
        health = await agent_adapter.health_check()
        
        assert health["initialized"] is True
        assert health["healthy"] is False
        assert "Health check failed: Test error" in health["issues"]
    
    @pytest.mark.asyncio
    async def test_agent_adapter_cleanup_success(self, agent_adapter):
        """Test successful cleanup."""
        mock_runner = MockAgentRunner()
        agent_adapter.agent_runner = mock_runner
        agent_adapter._initialized = True
        
        await agent_adapter.cleanup()
        
        assert agent_adapter._initialized is False
        mock_runner.cleanup.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_adapter_cleanup_exception(self, agent_adapter):
        """Test cleanup with exception."""
        mock_runner = MockAgentRunner()
        mock_runner.cleanup = AsyncMock(side_effect=Exception("Test error"))
        agent_adapter.agent_runner = mock_runner
        agent_adapter._initialized = True
        
        # Should not raise exception
        await agent_adapter.cleanup()
        
        assert agent_adapter._initialized is False


class TestAgentAdapterFactory:
    """Test cases for AgentAdapterFactory class."""
    
    @pytest.fixture
    def integration_manager(self):
        """Create a mock integration manager."""
        return Mock(spec=IntegrationManager)
    
    def test_agent_adapter_factory_create_adapter(self, integration_manager):
        """Test creating an adapter through the factory."""
        adapter = AgentAdapterFactory.create_adapter(
            framework="diy",
            agent_id="test_agent",
            config={"param": "value"},
            integration_manager=integration_manager,
            timeout=600
        )
        
        assert isinstance(adapter, AgentAdapter)
        assert adapter.config.framework == "diy"
        assert adapter.config.agent_id == "test_agent"
        assert adapter.config.config == {"param": "value"}
        assert adapter.config.timeout == 600
        assert adapter.integration_manager == integration_manager
    
    @pytest.mark.asyncio
    async def test_agent_adapter_factory_create_and_initialize_adapter_success(self, integration_manager):
        """Test creating and initializing an adapter successfully."""
        with patch.object(AgentAdapter, 'initialize', return_value=True) as mock_init:
            adapter = await AgentAdapterFactory.create_and_initialize_adapter(
                framework="diy",
                agent_id="test_agent",
                config={},
                integration_manager=integration_manager
            )
            
            assert isinstance(adapter, AgentAdapter)
            mock_init.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_agent_adapter_factory_create_and_initialize_adapter_failure(self, integration_manager):
        """Test creating and initializing an adapter with failure."""
        with patch.object(AgentAdapter, 'initialize', return_value=False) as mock_init:
            with patch.object(AgentAdapter, 'cleanup') as mock_cleanup:
                adapter = await AgentAdapterFactory.create_and_initialize_adapter(
                    framework="diy",
                    agent_id="test_agent",
                    config={},
                    integration_manager=integration_manager
                )
                
                assert adapter is None
                mock_init.assert_called_once()
                mock_cleanup.assert_called_once()


class TestMetricsAdapterConfig:
    """Test cases for MetricsAdapterConfig class."""
    
    def test_metrics_adapter_config_creation(self):
        """Test creating a metrics adapter configuration."""
        config = MetricsAdapterConfig(
            enable_legacy_metrics=False,
            enable_new_metrics=False,
            merge_results=False,
            legacy_weights={"finance": 0.5},
            custom_transformers={"metric1": "normalize"}
        )
        
        assert config.enable_legacy_metrics is False
        assert config.enable_new_metrics is False
        assert config.merge_results is False
        assert config.legacy_weights == {"finance": 0.5}
        assert config.custom_transformers == {"metric1": "normalize"}
    
    def test_metrics_adapter_config_defaults(self):
        """Test metrics adapter configuration with default values."""
        config = MetricsAdapterConfig()
        
        assert config.enable_legacy_metrics is True
        assert config.enable_new_metrics is True
        assert config.merge_results is True
        assert config.legacy_weights == {}
        assert config.custom_transformers == {}
    
    def test_metrics_adapter_config_to_dict(self):
        """Test converting metrics adapter configuration to dictionary."""
        config = MetricsAdapterConfig(
            enable_legacy_metrics=False,
            custom_transformers={"metric1": "normalize"}
        )
        
        result = config.to_dict()
        
        expected = {
            "enable_legacy_metrics": False,
            "enable_new_metrics": True,
            "merge_results": True,
            "legacy_weights": {},
            "custom_transformers": {"metric1": "normalize"}
        }
        
        assert result == expected


class TestMetricsAdapterResult:
    """Test cases for MetricsAdapterResult class."""
    
    def test_metrics_adapter_result_creation(self):
        """Test creating a metrics adapter result."""
        result = MetricsAdapterResult(
            success=True,
            legacy_metrics={"score": 0.8},
            new_metrics={"accuracy": 0.9},
            merged_metrics={"combined": 0.85},
            execution_time=1.5,
            error_message=None,
            warnings=["warning1"]
        )
        
        assert result.success is True
        assert result.legacy_metrics == {"score": 0.8}
        assert result.new_metrics == {"accuracy": 0.9}
        assert result.merged_metrics == {"combined": 0.85}
        assert result.execution_time == 1.5
        assert result.error_message is None
        assert result.warnings == ["warning1"]
    
    def test_metrics_adapter_result_defaults(self):
        """Test metrics adapter result with default values."""
        result = MetricsAdapterResult(success=True)
        
        assert result.legacy_metrics == {}
        assert result.new_metrics == {}
        assert result.merged_metrics == {}
        assert result.execution_time == 0.0
        assert result.error_message is None
        assert result.warnings == []
    
    def test_metrics_adapter_result_to_dict(self):
        """Test converting metrics adapter result to dictionary."""
        result = MetricsAdapterResult(
            success=True,
            legacy_metrics={"score": 0.8},
            warnings=["warning1"]
        )
        
        dict_result = result.to_dict()
        
        assert dict_result["success"] is True
        assert dict_result["legacy_metrics"] == {"score": 0.8}
        assert dict_result["new_metrics"] == {}
        assert dict_result["merged_metrics"] == {}
        assert dict_result["execution_time"] == 0.0
        assert dict_result["error_message"] is None
        assert dict_result["warnings"] == ["warning1"]


class MockMetricSuite:
    """Mock metric suite for testing."""
    
    def __init__(self):
        self.events = []
    
    def _handle_general_event(self, event_type, event):
        """Handle a general event."""
        self.events.append((event_type, event))
    
    def calculate_kpis(self, tick_number):
        """Calculate KPIs."""
        return {
            "overall_score": 0.8,
            "breakdown": {"finance": 0.9},
            "timestamp": "2023-01-01T00:00:00",
            "tick_number": tick_number
        }


class MockBaseMetric(BaseMetric):
    """Mock base metric for testing."""
    
    def __init__(self, score=0.8):
        self.score = score
        self.name = "mock_metric"
        self.description = "Mock metric for testing"
        self.category = "test"
    
    async def calculate(self, context):
        """Calculate the metric."""
        return MetricResult(
            name=self.name,
            score=self.score,
            details={"test": "value"}
        )


class TestMetricsAdapter:
    """Test cases for MetricsAdapter class."""
    
    @pytest.fixture
    def metrics_adapter_config(self):
        """Create a test metrics adapter configuration."""
        return MetricsAdapterConfig(
            enable_legacy_metrics=True,
            enable_new_metrics=True,
            merge_results=True,
            legacy_weights={"finance": 0.5},
            custom_transformers={"metric1": "normalize"}
        )
    
    @pytest.fixture
    def integration_manager(self):
        """Create a mock integration manager."""
        return Mock(spec=IntegrationManager)
    
    @pytest.fixture
    def metrics_adapter(self, metrics_adapter_config, integration_manager):
        """Create a metrics adapter instance."""
        return MetricsAdapter(metrics_adapter_config, integration_manager)
    
    def test_metrics_adapter_initialization(self, metrics_adapter_config, integration_manager):
        """Test metrics adapter initialization."""
        adapter = MetricsAdapter(metrics_adapter_config, integration_manager)
        
        assert adapter.config == metrics_adapter_config
        assert adapter.integration_manager == integration_manager
        assert adapter.legacy_metric_suite is None
        assert adapter._initialized is False
        
        # Check default legacy weights
        assert adapter.legacy_weights["finance"] == 0.20
        assert adapter.legacy_weights["ops"] == 0.15
        # Check that provided weights override defaults
        assert adapter.legacy_weights["finance"] == 0.5  # Overridden
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_initialize_success(self, metrics_adapter):
        """Test successful metrics adapter initialization."""
        with patch('benchmarking.integration.metrics_adapter.LEGACY_METRICS_AVAILABLE', True):
            with patch('benchmarking.integration.metrics_adapter.MetricSuite') as mock_suite:
                mock_suite.return_value = MockMetricSuite()
                
                result = await metrics_adapter.initialize()
                
                assert result is True
                assert metrics_adapter._initialized is True
                assert metrics_adapter.legacy_metric_suite is not None
                assert isinstance(metrics_adapter.legacy_metric_suite, MockMetricSuite)
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_initialize_already_initialized(self, metrics_adapter):
        """Test metrics adapter initialization when already initialized."""
        metrics_adapter._initialized = True
        
        result = await metrics_adapter.initialize()
        
        assert result is True
        # Should not try to create metric suite again
        with patch('benchmarking.integration.metrics_adapter.MetricSuite') as mock_suite:
            mock_suite.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_initialize_legacy_metrics_unavailable(self, metrics_adapter):
        """Test metrics adapter initialization when legacy metrics unavailable."""
        with patch('benchmarking.integration.metrics_adapter.LEGACY_METRICS_AVAILABLE', False):
            result = await metrics_adapter.initialize()
            
            assert result is True  # Should still succeed, just without legacy metrics
            assert metrics_adapter._initialized is True
            assert metrics_adapter.legacy_metric_suite is None
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_initialize_exception(self, metrics_adapter):
        """Test metrics adapter initialization with exception."""
        with patch('benchmarking.integration.metrics_adapter.LEGACY_METRICS_AVAILABLE', True):
            with patch('benchmarking.integration.metrics_adapter.MetricSuite') as mock_suite:
                mock_suite.side_effect = Exception("Test error")
                
                result = await metrics_adapter.initialize()
                
                assert result is False
                assert metrics_adapter._initialized is False
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_metrics_success(self, metrics_adapter):
        """Test successful metrics calculation."""
        # Setup
        metrics_adapter._initialized = True
        metrics_adapter.legacy_metric_suite = MockMetricSuite()
        
        with patch.object(metrics_adapter, '_calculate_legacy_metrics') as mock_legacy:
            with patch.object(metrics_adapter, '_calculate_new_metrics') as mock_new:
                mock_legacy.return_value = {"score": 0.8, "warnings": []}
                mock_new.return_value = {"accuracy": 0.9, "warnings": []}
                
                with patch.object(metrics_adapter, '_merge_metrics') as mock_merge:
                    mock_merge.return_value = {"combined": 0.85}
                    
                    result = await metrics_adapter.calculate_metrics(
                        tick_number=1,
                        events=[{"type": "test", "data": {}}],
                        context={"additional": "info"}
                    )
                    
                    # Verify
                    assert result.success is True
                    assert result.legacy_metrics == {"score": 0.8, "warnings": []}
                    assert result.new_metrics == {"accuracy": 0.9, "warnings": []}
                    assert result.merged_metrics == {"combined": 0.85}
                    assert result.execution_time > 0
                    assert result.warnings == []
                    
                    mock_legacy.assert_called_once_with(1, [{"type": "test", "data": {}}], {"additional": "info"})
                    mock_new.assert_called_once_with(1, [{"type": "test", "data": {}}], {"additional": "info"})
                    mock_merge.assert_called_once_with({"score": 0.8, "warnings": []}, {"accuracy": 0.9, "warnings": []})
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_metrics_not_initialized(self, metrics_adapter):
        """Test metrics calculation when not initialized."""
        metrics_adapter._initialized = False
        metrics_adapter.initialize = AsyncMock(return_value=False)
        
        result = await metrics_adapter.calculate_metrics(1, [])
        
        assert result.success is False
        assert result.error_message == "Metrics adapter not initialized"
        assert result.execution_time == 0.0
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_metrics_legacy_disabled(self, metrics_adapter):
        """Test metrics calculation with legacy disabled."""
        metrics_adapter._initialized = True
        metrics_adapter.config.enable_legacy_metrics = False
        
        with patch.object(metrics_adapter, '_calculate_new_metrics') as mock_new:
            mock_new.return_value = {"accuracy": 0.9, "warnings": []}
            
            result = await metrics_adapter.calculate_metrics(1, [])
            
            assert result.success is True
            assert result.legacy_metrics == {}
            assert result.new_metrics == {"accuracy": 0.9, "warnings": []}
            assert result.merged_metrics == {}  # No legacy metrics to merge
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_metrics_new_disabled(self, metrics_adapter):
        """Test metrics calculation with new disabled."""
        metrics_adapter._initialized = True
        metrics_adapter.config.enable_new_metrics = False
        metrics_adapter.legacy_metric_suite = MockMetricSuite()
        
        with patch.object(metrics_adapter, '_calculate_legacy_metrics') as mock_legacy:
            mock_legacy.return_value = {"score": 0.8, "warnings": []}
            
            result = await metrics_adapter.calculate_metrics(1, [])
            
            assert result.success is True
            assert result.legacy_metrics == {"score": 0.8, "warnings": []}
            assert result.new_metrics == {}
            assert result.merged_metrics == {}  # No new metrics to merge
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_metrics_merge_disabled(self, metrics_adapter):
        """Test metrics calculation with merge disabled."""
        metrics_adapter._initialized = True
        metrics_adapter.config.merge_results = False
        metrics_adapter.legacy_metric_suite = MockMetricSuite()
        
        with patch.object(metrics_adapter, '_calculate_legacy_metrics') as mock_legacy:
            with patch.object(metrics_adapter, '_calculate_new_metrics') as mock_new:
                mock_legacy.return_value = {"score": 0.8, "warnings": []}
                mock_new.return_value = {"accuracy": 0.9, "warnings": []}
                
                result = await metrics_adapter.calculate_metrics(1, [])
                
                assert result.success is True
                assert result.legacy_metrics == {"score": 0.8, "warnings": []}
                assert result.new_metrics == {"accuracy": 0.9, "warnings": []}
                assert result.merged_metrics == {}  # Merge disabled
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_metrics_exception(self, metrics_adapter):
        """Test metrics calculation with exception."""
        metrics_adapter._initialized = True
        
        with patch.object(metrics_adapter, '_calculate_legacy_metrics') as mock_legacy:
            mock_legacy.side_effect = Exception("Test error")
            
            result = await metrics_adapter.calculate_metrics(1, [])
            
            assert result.success is False
            assert result.error_message == "Test error"
            assert result.execution_time > 0
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_legacy_metrics_success(self, metrics_adapter):
        """Test successful legacy metrics calculation."""
        metrics_adapter.legacy_metric_suite = MockMetricSuite()
        
        events = [
            {"type": "SaleOccurred", "amount": 100},
            {"type": "SetPriceCommand", "price": 50},
            {"type": "UnknownEvent", "data": {}}
        ]
        
        result = await metrics_adapter._calculate_legacy_metrics(1, events, {"context": "info"})
        
        assert result["overall_score"] == 0.8
        assert result["breakdown"] == {"finance": 0.9}
        assert result["timestamp"] == "2023-01-01T00:00:00"
        assert result["tick_number"] == 1
        
        # Verify events were processed
        assert len(metrics_adapter.legacy_metric_suite.events) == 3
        assert metrics_adapter.legacy_metric_suite.events[0][0] == "SaleOccurred"
        assert metrics_adapter.legacy_metric_suite.events[1][0] == "SetPriceCommand"
        assert metrics_adapter.legacy_metric_suite.events[2][0] == "UnknownEvent"
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_legacy_metrics_no_suite(self, metrics_adapter):
        """Test legacy metrics calculation with no suite."""
        metrics_adapter.legacy_metric_suite = None
        
        result = await metrics_adapter._calculate_legacy_metrics(1, [])
        
        assert result == {}
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_legacy_metrics_exception(self, metrics_adapter):
        """Test legacy metrics calculation with exception."""
        metrics_adapter.legacy_metric_suite = MockMetricSuite()
        metrics_adapter.legacy_metric_suite.calculate_kpis.side_effect = Exception("Test error")
        
        result = await metrics_adapter._calculate_legacy_metrics(1, [])
        
        assert result["error"] == "Test error"
        assert "Legacy metrics calculation failed: Test error" in result["warnings"]
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_new_metrics_success(self, metrics_adapter):
        """Test successful new metrics calculation."""
        with patch.object(metrics_registry, 'get_all_metrics') as mock_get_all:
            mock_metric = MockBaseMetric(score=0.8)
            mock_get_all.return_value = {"metric1": mock_metric}
            
            result = await metrics_adapter._calculate_new_metrics(
                tick_number=1,
                events=[{"type": "test"}],
                context={"additional": "info"}
            )
            
            assert "metric1" in result
            assert result["metric1"]["name"] == "mock_metric"
            assert result["metric1"]["score"] == 0.8
            assert result["metric1"]["details"] == {"test": "value"}
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_new_metrics_with_transformer(self, metrics_adapter):
        """Test new metrics calculation with transformer."""
        metrics_adapter.config.custom_transformers = {"metric1": "normalize"}
        
        with patch.object(metrics_registry, 'get_all_metrics') as mock_get_all:
            mock_metric = MockBaseMetric(score=80.0)  # Score outside 0-1 range
            mock_get_all.return_value = {"metric1": mock_metric}
            
            result = await metrics_adapter._calculate_new_metrics(1, [])
            
            assert "metric1" in result
            assert result["metric1"]["score"] == 80.0
            assert result["metric1"]["normalized_score"] == 0.8  # Normalized to 0-1 range
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_new_metrics_metric_exception(self, metrics_adapter):
        """Test new metrics calculation with metric exception."""
        with patch.object(metrics_registry, 'get_all_metrics') as mock_get_all:
            mock_metric = MockBaseMetric()
            mock_metric.calculate = AsyncMock(side_effect=Exception("Metric error"))
            mock_get_all.return_value = {"metric1": mock_metric}
            
            result = await metrics_adapter._calculate_new_metrics(1, [])
            
            assert "metric1" in result
            assert result["metric1"]["error"] == "Metric error"
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_new_metrics_transformer_exception(self, metrics_adapter):
        """Test new metrics calculation with transformer exception."""
        metrics_adapter.config.custom_transformers = {"metric1": "unknown_transformer"}
        
        with patch.object(metrics_registry, 'get_all_metrics') as mock_get_all:
            mock_metric = MockBaseMetric()
            mock_get_all.return_value = {"metric1": mock_metric}
            
            result = await metrics_adapter._calculate_new_metrics(1, [])
            
            assert "metric1" in result
            assert result["metric1"]["score"] == 0.8
            # Should still have original metric data
    
    @pytest.mark.asyncio
    async def test_metrics_adapter_calculate_new_metrics_exception(self, metrics_adapter):
        """Test new metrics calculation with exception."""
        with patch.object(metrics_registry, 'get_all_metrics') as mock_get_all:
            mock_get_all.side_effect = Exception("Registry error")
            
            result = await metrics_adapter._calculate_new_metrics(1, [])
            
            assert result["error"] == "Registry error"
            assert "New metrics calculation failed: Registry error" in result["warnings"]
    
    def test_metrics_adapter_merge_metrics_both_empty(self, metrics_adapter):
        """Test merging metrics when both are empty."""
        result = metrics_adapter._merge_metrics({}, {})
        
        assert result["legacy_metrics"] == {}
        assert result["new_metrics"] == {}
        assert "merged_at" in result
        assert "overall_score" not in result
    
    def test_metrics_adapter_merge_metrics_legacy_only(self, metrics_adapter):
        """Test merging metrics with only legacy metrics."""
        legacy = {"overall_score": 0.8, "breakdown": {"finance": 0.9}}
        new = {}
        
        result = metrics_adapter._merge_metrics(legacy, new)
        
        assert result["legacy_metrics"] == legacy
        assert result["new_metrics"] == new
        assert "merged_at" in result
        assert "overall_score" not in result
    
    def test_metrics_adapter_merge_metrics_new_only(self, metrics_adapter):
        """Test merging metrics with only new metrics."""
        legacy = {}
        new = {"metric1": {"score": 0.9}, "metric2": {"value": 0.7}}
        
        result = metrics_adapter._merge_metrics(legacy, new)
        
        assert result["legacy_metrics"] == legacy
        assert result["new_metrics"] == new
        assert "merged_at" in result
        assert "overall_score" not in result
    
    def test_metrics_adapter_merge_metrics_both_with_scores(self, metrics_adapter):
        """Test merging metrics with both having scores."""
        legacy = {"overall_score": 0.8}
        new = {"metric1": {"score": 0.9}, "metric2": {"score": 0.7}}
        
        result = metrics_adapter._merge_metrics(legacy, new)
        
        assert result["legacy_metrics"] == legacy
        assert result["new_metrics"] == new
        assert "merged_at" in result
        assert result["overall_score"] == 0.75  # (0.8 + 0.8) / 2
        assert result["score_breakdown"]["legacy_score"] == 0.8
        assert result["score_breakdown"]["new_score"] == 0.8
        assert result["score_breakdown"]["legacy_weight"] == 0.5
        assert result["score_breakdown"]["new_weight"] == 0.5
    
    def test_metrics_adapter_merge_metrics_new_without_scores(self, metrics_adapter):
        """Test merging metrics with new metrics without scores."""
        legacy = {"overall_score": 0.8}
        new = {"metric1": {"value": 0.9}, "metric2": {"data": "test"}}
        
        result = metrics_adapter._merge_metrics(legacy, new)
        
        assert result["legacy_metrics"] == legacy
        assert result["new_metrics"] == new
        assert "merged_at" in result
        assert result["overall_score"] == 0.4  # (0.8 + 0.0) / 2
        assert result["score_breakdown"]["legacy_score"] == 0.8
        assert result["score_breakdown"]["new_score"] == 0.0
    
    def test_metrics_adapter_get_transformer_normalize(self, metrics_adapter):
        """Test getting normalize transformer."""
        transformer = metrics_adapter._get_transformer("normalize")
        
        assert transformer is not None
        
        # Test transformer