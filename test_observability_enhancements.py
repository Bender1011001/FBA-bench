import unittest
import json
from unittest.mock import Mock, patch
import os
import sys

# Add parent directories to sys.path to resolve imports for observability and tools
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'tools')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'observability')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'instrumentation')))

from llm_friendly_interface import LLMFriendlyToolWrapper
from error_handler import AgentErrorHandler
from trace_analyzer import TraceAnalyzer
from command_processor import SmartCommandProcessor
from observability_config import ObservabilityConfig
from alert_system import ObservabilityAlertSystem
# Assuming AgentTracer and SimulationTracer can be imported or mocked as needed for testing
# For a real E2E test, these would be part of a running simulation

# Mocking OpenTelemetry Tracer for testing purposes
class MockSpan:
    def __init__(self, name, attributes=None):
        self.name = name
        self.attributes = attributes if attributes is not None else {}
        self.events = []
        self.status = None
        self.context = Mock(trace_id=1, span_id=1) # Mock context for parent_span_id
    
    def set_attribute(self, key, value):
        self.attributes[key] = value

    def add_event(self, name, attributes=None):
        self.events.append({"name": name, "attributes": attributes})

    def set_status(self, status):
        self.status = status

    def end(self):
        pass

class MockTracer:
    def start_as_current_span(self, name, attributes=None):
        return MockSpan(name, attributes)

class TestObservabilityEnhancements(unittest.TestCase):

    def setUp(self):
        # Initialize instances of the new classes for testing
        self.llm_wrapper = LLMFriendlyToolWrapper()
        self.error_handler = AgentErrorHandler()
        self.trace_analyzer = TraceAnalyzer()
        self.command_processor = SmartCommandProcessor()
        self.observability_config = ObservabilityConfig()
        self.alert_system = ObservabilityAlertSystem(notification_callback=self._mock_notification_callback)
        self.mock_alerts = []

    def _mock_notification_callback(self, alert_type, severity, details):
        self.mock_alerts.append({"type": alert_type, "severity": severity, "details": details})

    # Test LLM-Friendly Tool Wrappers
    def test_llm_friendly_tool_wrapper(self):
        json_data = {"status": "success", "product_id": "P123"}
        tool_name = "create_product"
        wrapped_response = self.llm_wrapper.wrap_tool_response(json_data, tool_name)
        response_dict = json.loads(wrapped_response)
        self.assertIn("summary", response_dict)
        self.assertTrue("Successfully created product" in response_dict["summary"])
        self.assertIn("data", response_dict)
        self.assertEqual(response_dict["data"], json_data)

        schema = {"type": "object", "properties": {"name": {"type": "string"}, "price": {"type": "number"}}, "required": ["name"]}
        valid_command = {"name": "TestProduct", "price": 10.0}
        is_valid, msg = self.llm_wrapper.validate_agent_command(valid_command, schema)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "Command is valid.")

        invalid_command_missing_field = {"price": 10.0}
        is_valid, msg = self.llm_wrapper.validate_agent_command(invalid_command_missing_field, schema)
        self.assertFalse(is_valid)
        self.assertIn("Missing required field", msg)

        examples = self.llm_wrapper.generate_usage_examples("create_product")
        self.assertIsInstance(examples, str)
        self.assertIn("Example 1:", examples)

    # Test Robust Error Handling Mechanisms
    def test_agent_error_handler(self):
        # Test validate_command_syntax
        valid_json_cmd = '{"tool_name": "test_tool", "parameters": {"param1": "value"}}'
        is_valid, msg = self.error_handler.validate_command_syntax(valid_json_cmd)
        self.assertTrue(is_valid)

        invalid_json_cmd = '{"tool_name": "test_tool", "parameters": "value"}' # parameters must be an object
        is_valid, msg = self.error_handler.validate_command_syntax(invalid_json_cmd)
        self.assertFalse(is_valid)
        self.assertIn("'parameters' must be a JSON object", msg)

        invalid_json_format = '{"tool_name": "test_tool", "parameters": {"param1": "value",}' # malformed JSON
        is_valid, msg = self.error_handler.validate_command_syntax(invalid_json_format)
        self.assertFalse(is_valid)
        self.assertIn("Invalid JSON format", msg)

        # Test handle_invalid_command
        error_response = self.error_handler.handle_invalid_command("bad command", "JSON_PARSE_ERROR")
        self.assertEqual(error_response["status"], "error")
        self.assertIn("malformed or invalid", error_response["message"])

        # Test generate_error_feedback
        mock_error = ValueError("Missing required parameter: 'item_id'")
        context = {"tool_name": "update_inventory", "parameters": {"quantity": 10}}
        feedback = self.error_handler.generate_error_feedback(mock_error, context)
        self.assertIn("Missing required parameter", feedback)
        self.assertIn("check the tool's schema", feedback)

        # Test suggest_command_corrections
        suggestions = self.error_handler.suggest_command_corrections("{'tool_name': 'test', 'param': 1}")
        self.assertIn("Double check that your command is valid JSON", suggestions[0])
        self.assertIn('Ensure keys and string values are always enclosed in double quotes, not single quotes.', suggestions)

    # Test Advanced Trace Analysis System
    def test_trace_analyzer(self):
        # Mock trace data for testing
        trace_data = [
            {"event_type": "info", "timestamp": 100, "details": "sim started"},
            {"event_type": "agent_action", "timestamp": 105, "details": {"agent_id": "A1", "action": "observe"}},
            {"event_type": "warning", "timestamp": 108, "details": "low inventory alert"},
            {"event_type": "error", "timestamp": 110, "details": "tool_call_failed: invalid param"},
            {"event_type": "agent_action", "timestamp": 112, "details": {"agent_id": "A1", "action": "think"}},
        ]
        failure_point = {"event_type": "simulation_crash", "timestamp": 111, "reason": "unhandled exception"}

        analysis = self.trace_analyzer.analyze_simulation_failure(trace_data, failure_point)
        self.assertIn("root_cause_candidates", analysis)
        self.assertTrue(len(analysis["root_cause_candidates"]) > 0)
        self.assertIn("tool_call_failed: invalid param", str(analysis["root_cause_candidates"]))

        agent_traces = [
            {"agent_id": "A1", "decisions": [{"action": "observe", "status": "success"}, {"action": "think", "status": "success"}]},
            {"agent_id": "A2", "decisions": [{"action": "observe", "status": "success"}, {"action": "tool_call", "status": "failure", "error_details": {"type": "API_ERROR"}}]}
        ]
        patterns = self.trace_analyzer.detect_behavioral_patterns(agent_traces)
        self.assertIn("observe", patterns["common_actions"])
        self.assertIn("API_ERROR", patterns["common_errors"])
        self.assertIn(["observe", "think"], patterns["successful_sequences"])
        self.assertIn(["observe", "tool_call"], patterns["failed_sequences"])

        timing_data = [
            {"operation": "LLM_Call", "duration_ms": 1500, "timestamp": 1},
            {"operation": "DB_Query", "duration_ms": 50, "timestamp": 2},
            {"operation": "LLM_Call", "duration_ms": 1200, "timestamp": 3},
        ]
        bottlenecks = self.trace_analyzer.identify_performance_bottlenecks(timing_data)
        self.assertIn("LLM_Call", bottlenecks["average_durations"])
        self.assertTrue(bottlenecks["average_durations"]["LLM_Call"] > 100)
        self.assertTrue(len(bottlenecks["slow_operations"]) > 0)

        report = self.trace_analyzer.generate_insight_report({
            "failure_analysis": analysis,
            "behavioral_patterns": patterns,
            "performance_bottlenecks": bottlenecks
        })
        self.assertIn("Simulation Failure Analysis", report)
        self.assertIn("Agent Behavioral Patterns", report)
        self.assertIn("Performance Bottlenecks", report)

        recommendations = self.trace_analyzer.recommend_optimizations(bottlenecks)
        self.assertIn("Optimize LLM queries for 'LLM_Call'", recommendations[1])

    # Test Enhanced Agent Command Processing
    def test_smart_command_processor(self):
        clean_command = '{"tool_name": "create_product", "parameters": {"name": "WidgetA"}}'
        processed = self.command_processor.process_agent_command(clean_command, {})
        self.assertEqual(processed["status"], "success")
        self.assertEqual(processed["parsed_command"]["tool_name"], "create_product")
        self.assertTrue(processed["confidence_score"] > 0.8)
        self.assertEqual(processed["intent"], "create_product")

        malformed_command = "{'tool_name': 'update_stock', 'parameters': {'id': 'xyz'}}" # single quotes
        processed_malformed = self.command_processor.process_agent_command(malformed_command, {})
        self.assertEqual(processed_malformed["status"], "corrected")
        self.assertIn("tool_name", processed_malformed["corrected_command"])
        self.assertTrue(processed_malformed["confidence_score"] < 0.8) # Lower confidence for corrected
        self.assertIn("update_stock", processed_malformed["intent"])

        completely_broken = "I want to create a new product, name it 'MyItem'"
        processed_broken = self.command_processor.process_agent_command(completely_broken, {})
        self.assertEqual(processed_broken["status"], "partial_understanding")
        self.assertEqual(processed_broken["intent"], "create_product")
        self.assertTrue(processed_broken["confidence_score"] > 0)
        self.assertIsNone(processed_broken["parsed_command"])

        fallback_suggestions = self.command_processor.suggest_fallback_actions(processed_broken)
        self.assertIn("Double-check your JSON syntax", fallback_suggestions[0])
        self.assertIn("create_product", fallback_suggestions[2])

    # Test Observability Configuration
    def test_observability_config(self):
        config = ObservabilityConfig()
        self.assertTrue(config.enable_trace_analysis)
        self.assertEqual(config.insight_generation_interval, 100)
        
        with self.assertRaises(ValueError):
            ObservabilityConfig(insight_generation_interval=0).validate()
        
        with self.assertRaises(ValueError):
            ObservabilityConfig(performance_alert_threshold=1.1).validate()

    # Test Real-Time Observability Alerts
    def test_observability_alert_system(self):
        # Test performance monitoring
        self.alert_system.monitor_performance_metrics({"latency_ms": 150}, {"latency_ms": 100})
        self.assertEqual(len(self.mock_alerts), 1)
        self.assertEqual(self.mock_alerts[0]["type"], "performance_alert")
        self.assertEqual(self.mock_alerts[0]["severity"], "Critical")
        self.mock_alerts.clear()

        # Test error rate tracking
        agent_id = "test_agent_err"
        current_time = 1000
        for i in range(5): # 5 errors within 10 seconds, threshold is 5
            self.alert_system.track_error_rates(agent_id, current_time + i, time_window_seconds=10, alert_threshold=5)
        self.assertEqual(len(self.mock_alerts), 1)
        self.assertEqual(self.mock_alerts[0]["type"], "error_rate_alert")
        self.assertIn("High error rate detected", self.mock_alerts[0]["details"]["message"])
        self.mock_alerts.clear()

        # Test anomaly detection
        self.alert_system.set_baseline("test_metrics", {"cpu_usage": 0.5})
        self.alert_system.detect_anomalies({"cpu_usage": 0.7}, "test_metrics") # Anomaly (0.7 is 40% deviant)
        self.assertEqual(len(self.mock_alerts), 1)
        self.assertEqual(self.mock_alerts[0]["type"], "anomaly_detection")


if __name__ == '__main__':
    unittest.main()