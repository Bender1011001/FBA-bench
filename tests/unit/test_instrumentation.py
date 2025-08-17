import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from instrumentation.agent_tracer import AgentTracer
from instrumentation.simulation_tracer import SimulationTracer
from instrumentation.tracer import Tracer
from instrumentation.export_utils import ExportUtils


class TestTracer(unittest.TestCase):
    """Test suite for the Tracer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.tracer = Tracer()
    
    def test_tracer_initialization(self):
        """Test that the tracer initializes correctly."""
        self.assertIsNotNone(self.tracer)
        self.assertEqual(len(self.tracer._spans), 0)
        self.assertEqual(len(self.tracer._traces), 0)
    
    def test_start_span(self):
        """Test starting a span."""
        span_id = self.tracer.start_span("test_span", {"key": "value"})
        
        self.assertIsNotNone(span_id)
        self.assertIn(span_id, self.tracer._spans)
        self.assertEqual(self.tracer._spans[span_id]["name"], "test_span")
        self.assertEqual(self.tracer._spans[span_id]["attributes"]["key"], "value")
    
    def test_end_span(self):
        """Test ending a span."""
        span_id = self.tracer.start_span("test_span")
        
        # End the span
        self.tracer.end_span(span_id)
        
        # Verify the span is marked as ended
        self.assertTrue(self.tracer._spans[span_id]["ended"])
        self.assertIsNotNone(self.tracer._spans[span_id]["end_time"])
    
    def test_get_trace(self):
        """Test getting a trace."""
        # Start and end a span
        span_id = self.tracer.start_span("test_span", {"trace_id": "trace123"})
        self.tracer.end_span(span_id)
        
        # Get the trace
        trace = self.tracer.get_trace("trace123")
        
        self.assertIsNotNone(trace)
        self.assertEqual(len(trace["spans"]), 1)
        self.assertEqual(trace["spans"][0]["name"], "test_span")
    
    def test_export_trace(self):
        """Test exporting a trace."""
        # Start and end a span
        span_id = self.tracer.start_span("test_span", {"trace_id": "trace123"})
        self.tracer.end_span(span_id)
        
        # Mock the export function
        with patch.object(self.tracer, '_export_to_json') as mock_export:
            mock_export.return_value = True
            
            # Export the trace
            result = self.tracer.export_trace("trace123", "json")
            
            self.assertTrue(result)
            mock_export.assert_called_once()


class TestAgentTracer(unittest.TestCase):
    """Test suite for the AgentTracer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.agent_tracer = AgentTracer()
    
    def test_agent_tracer_initialization(self):
        """Test that the agent tracer initializes correctly."""
        self.assertIsNotNone(self.agent_tracer)
        self.assertIsInstance(self.agent_tracer, Tracer)
        self.assertEqual(len(self.agent_tracer._agent_spans), 0)
    
    def test_trace_agent_decision(self):
        """Test tracing agent decisions."""
        trace_id = self.agent_tracer.trace_agent_decision(
            agent_id="agent1",
            decision_type="pricing",
            decision_data={"price": 10.0},
            context={"market_conditions": "stable"}
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.agent_tracer._agent_spans)
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["agent_id"], "agent1")
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["decision_type"], "pricing")
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["decision_data"]["price"], 10.0)
    
    def test_trace_agent_action(self):
        """Test tracing agent actions."""
        trace_id = self.agent_tracer.trace_agent_action(
            agent_id="agent1",
            action_type="purchase",
            action_data={"quantity": 100},
            result={"success": True}
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.agent_tracer._agent_spans)
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["agent_id"], "agent1")
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["action_type"], "purchase")
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["action_data"]["quantity"], 100)
        self.assertTrue(self.agent_tracer._agent_spans[trace_id]["result"]["success"])
    
    def test_trace_agent_learning(self):
        """Test tracing agent learning."""
        trace_id = self.agent_tracer.trace_agent_learning(
            agent_id="agent1",
            learning_type="reinforcement",
            learning_data={"reward": 1.0},
            model_update={"accuracy": 0.85}
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.agent_tracer._agent_spans)
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["agent_id"], "agent1")
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["learning_type"], "reinforcement")
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["learning_data"]["reward"], 1.0)
        self.assertEqual(self.agent_tracer._agent_spans[trace_id]["model_update"]["accuracy"], 0.85)
    
    def test_get_agent_trace(self):
        """Test getting agent traces."""
        # Trace an agent decision
        trace_id = self.agent_tracer.trace_agent_decision(
            agent_id="agent1",
            decision_type="pricing",
            decision_data={"price": 10.0}
        )
        
        # Get the agent trace
        trace = self.agent_tracer.get_agent_trace("agent1", trace_id)
        
        self.assertIsNotNone(trace)
        self.assertEqual(trace["agent_id"], "agent1")
        self.assertEqual(trace["decision_type"], "pricing")
        self.assertEqual(trace["decision_data"]["price"], 10.0)
    
    def test_get_agent_traces(self):
        """Test getting all agent traces."""
        # Trace multiple agent decisions
        self.agent_tracer.trace_agent_decision(
            agent_id="agent1",
            decision_type="pricing",
            decision_data={"price": 10.0}
        )
        
        self.agent_tracer.trace_agent_action(
            agent_id="agent1",
            action_type="purchase",
            action_data={"quantity": 100}
        )
        
        # Get all agent traces
        traces = self.agent_tracer.get_agent_traces("agent1")
        
        self.assertEqual(len(traces), 2)
        self.assertTrue(any(t["decision_type"] == "pricing" for t in traces))
        self.assertTrue(any(t["action_type"] == "purchase" for t in traces))


class TestSimulationTracer(unittest.TestCase):
    """Test suite for the SimulationTracer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.simulation_tracer = SimulationTracer()
    
    def test_simulation_tracer_initialization(self):
        """Test that the simulation tracer initializes correctly."""
        self.assertIsNotNone(self.simulation_tracer)
        self.assertIsInstance(self.simulation_tracer, Tracer)
        self.assertEqual(len(self.simulation_tracer._simulation_spans), 0)
    
    def test_trace_simulation_event(self):
        """Test tracing simulation events."""
        trace_id = self.simulation_tracer.trace_simulation_event(
            event_type="market_update",
            event_data={"price": 10.0, "volume": 1000},
            tick=100,
            scenario_id="scenario1"
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.simulation_tracer._simulation_spans)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["event_type"], "market_update")
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["event_data"]["price"], 10.0)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["tick"], 100)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["scenario_id"], "scenario1")
    
    def test_trace_simulation_state(self):
        """Test tracing simulation state."""
        trace_id = self.simulation_tracer.trace_simulation_state(
            state_data={"agents": 10, "market_price": 10.0},
            tick=100,
            scenario_id="scenario1"
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.simulation_tracer._simulation_spans)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["state_data"]["agents"], 10)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["state_data"]["market_price"], 10.0)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["tick"], 100)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["scenario_id"], "scenario1")
    
    def test_trace_simulation_metrics(self):
        """Test tracing simulation metrics."""
        trace_id = self.simulation_tracer.trace_simulation_metrics(
            metrics_data={"profit": 1000.0, "market_share": 0.25},
            tick=100,
            scenario_id="scenario1"
        )
        
        self.assertIsNotNone(trace_id)
        self.assertIn(trace_id, self.simulation_tracer._simulation_spans)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["metrics_data"]["profit"], 1000.0)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["metrics_data"]["market_share"], 0.25)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["tick"], 100)
        self.assertEqual(self.simulation_tracer._simulation_spans[trace_id]["scenario_id"], "scenario1")
    
    def test_get_simulation_trace(self):
        """Test getting simulation traces."""
        # Trace a simulation event
        trace_id = self.simulation_tracer.trace_simulation_event(
            event_type="market_update",
            event_data={"price": 10.0},
            tick=100,
            scenario_id="scenario1"
        )
        
        # Get the simulation trace
        trace = self.simulation_tracer.get_simulation_trace("scenario1", trace_id)
        
        self.assertIsNotNone(trace)
        self.assertEqual(trace["event_type"], "market_update")
        self.assertEqual(trace["event_data"]["price"], 10.0)
        self.assertEqual(trace["tick"], 100)
        self.assertEqual(trace["scenario_id"], "scenario1")
    
    def test_get_simulation_traces_by_tick(self):
        """Test getting simulation traces by tick."""
        # Trace multiple simulation events
        self.simulation_tracer.trace_simulation_event(
            event_type="market_update",
            event_data={"price": 10.0},
            tick=100,
            scenario_id="scenario1"
        )
        
        self.simulation_tracer.trace_simulation_state(
            state_data={"agents": 10},
            tick=100,
            scenario_id="scenario1"
        )
        
        # Get simulation traces by tick
        traces = self.simulation_tracer.get_simulation_traces_by_tick("scenario1", 100)
        
        self.assertEqual(len(traces), 2)
        self.assertTrue(any(t["event_type"] == "market_update" for t in traces))
        self.assertTrue(any("state_data" in t for t in traces))
    
    def test_get_simulation_traces_by_scenario(self):
        """Test getting simulation traces by scenario."""
        # Trace multiple simulation events
        self.simulation_tracer.trace_simulation_event(
            event_type="market_update",
            event_data={"price": 10.0},
            tick=100,
            scenario_id="scenario1"
        )
        
        self.simulation_tracer.trace_simulation_state(
            state_data={"agents": 10},
            tick=200,
            scenario_id="scenario1"
        )
        
        # Get simulation traces by scenario
        traces = self.simulation_tracer.get_simulation_traces_by_scenario("scenario1")
        
        self.assertEqual(len(traces), 2)
        self.assertTrue(any(t["event_type"] == "market_update" for t in traces))
        self.assertTrue(any("state_data" in t for t in traces))


class TestExportUtils(unittest.TestCase):
    """Test suite for the ExportUtils class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.export_utils = ExportUtils()
    
    def test_export_utils_initialization(self):
        """Test that the export utils initializes correctly."""
        self.assertIsNotNone(self.export_utils)
    
    def test_export_to_json(self):
        """Test exporting to JSON."""
        data = {
            "trace_id": "trace123",
            "spans": [
                {
                    "name": "test_span",
                    "attributes": {"key": "value"}
                }
            ]
        }
        
        # Mock file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            result = self.export_utils.export_to_json(data, "test_trace.json")
            
            self.assertTrue(result)
            mock_file.assert_called_once_with("test_trace.json", "w")
    
    def test_export_to_csv(self):
        """Test exporting to CSV."""
        data = [
            {
                "trace_id": "trace123",
                "span_name": "test_span",
                "duration": 100,
                "attributes": {"key": "value"}
            }
        ]
        
        # Mock file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            result = self.export_utils.export_to_csv(data, "test_trace.csv")
            
            self.assertTrue(result)
            mock_file.assert_called_once_with("test_trace.csv", "w", newline="")
    
    def test_export_to_xml(self):
        """Test exporting to XML."""
        data = {
            "trace_id": "trace123",
            "spans": [
                {
                    "name": "test_span",
                    "attributes": {"key": "value"}
                }
            ]
        }
        
        # Mock file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            result = self.export_utils.export_to_xml(data, "test_trace.xml")
            
            self.assertTrue(result)
            mock_file.assert_called_once_with("test_trace.xml", "w")
    
    def test_export_to_parquet(self):
        """Test exporting to Parquet."""
        data = [
            {
                "trace_id": "trace123",
                "span_name": "test_span",
                "duration": 100,
                "attributes": {"key": "value"}
            }
        ]
        
        # Mock pandas and file operations
        with patch('pandas.DataFrame') as mock_df, \
             patch('builtins.open', unittest.mock.mock_open()) as mock_file:
            
            mock_df_instance = Mock()
            mock_df.return_value = mock_df_instance
            mock_df_instance.to_parquet = Mock()
            
            result = self.export_utils.export_to_parquet(data, "test_trace.parquet")
            
            self.assertTrue(result)
            mock_df.assert_called_once_with(data)
            mock_df_instance.to_parquet.assert_called_once()
    
    def test_compress_export(self):
        """Test compressing exported files."""
        # Mock file operations
        with patch('builtins.open', unittest.mock.mock_open()) as mock_file, \
             patch('gzip.open') as mock_gzip:
            
            mock_gzip.return_value.__enter__ = Mock(return_value=Mock())
            mock_gzip.return_value.__exit__ = Mock(return_value=None)
            
            result = self.export_utils.compress_export("test_trace.json", "test_trace.json.gz")
            
            self.assertTrue(result)
            mock_file.assert_called_once_with("test_trace.json", "rb")
            mock_gzip.assert_called_once_with("test_trace.json.gz", "wb")


if __name__ == '__main__':
    unittest.main()