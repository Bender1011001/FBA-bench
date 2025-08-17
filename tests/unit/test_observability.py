import unittest
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import sys
import os
import numpy as np
from datetime import datetime, timedelta

# Add the project root to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from observability.alert_system import AlertSystem
from observability.observability_config import ObservabilityConfig
from observability.trace_analyzer import TraceAnalyzer


class TestAlertSystem(unittest.TestCase):
    """Test suite for the AlertSystem class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.alert_system = AlertSystem()
    
    def test_alert_system_initialization(self):
        """Test that the alert system initializes correctly."""
        self.assertIsNotNone(self.alert_system)
        self.assertEqual(len(self.alert_system._alerts), 0)
        self.assertEqual(len(self.alert_system._alert_rules), 0)
        self.assertEqual(len(self.alert_system._alert_subscribers), 0)
    
    def test_create_alert_rule(self):
        """Test creating an alert rule."""
        rule_data = {
            "name": "High CPU Usage",
            "description": "Alert when CPU usage exceeds 90%",
            "condition": "cpu_usage > 90",
            "severity": "high",
            "enabled": True
        }
        
        rule_id = self.alert_system.create_alert_rule(rule_data)
        
        self.assertIsNotNone(rule_id)
        self.assertIn(rule_id, self.alert_system._alert_rules)
        self.assertEqual(self.alert_system._alert_rules[rule_id]["name"], "High CPU Usage")
        self.assertEqual(self.alert_system._alert_rules[rule_id]["condition"], "cpu_usage > 90")
        self.assertEqual(self.alert_system._alert_rules[rule_id]["severity"], "high")
        self.assertTrue(self.alert_system._alert_rules[rule_id]["enabled"])
    
    def test_update_alert_rule(self):
        """Test updating an alert rule."""
        rule_data = {
            "name": "High CPU Usage",
            "description": "Alert when CPU usage exceeds 90%",
            "condition": "cpu_usage > 90",
            "severity": "high",
            "enabled": True
        }
        
        rule_id = self.alert_system.create_alert_rule(rule_data)
        
        # Update the rule
        updated_data = {
            "name": "Very High CPU Usage",
            "description": "Alert when CPU usage exceeds 95%",
            "condition": "cpu_usage > 95",
            "severity": "critical",
            "enabled": False
        }
        
        self.alert_system.update_alert_rule(rule_id, updated_data)
        
        self.assertEqual(self.alert_system._alert_rules[rule_id]["name"], "Very High CPU Usage")
        self.assertEqual(self.alert_system._alert_rules[rule_id]["description"], "Alert when CPU usage exceeds 95%")
        self.assertEqual(self.alert_system._alert_rules[rule_id]["condition"], "cpu_usage > 95")
        self.assertEqual(self.alert_system._alert_rules[rule_id]["severity"], "critical")
        self.assertFalse(self.alert_system._alert_rules[rule_id]["enabled"])
    
    def test_delete_alert_rule(self):
        """Test deleting an alert rule."""
        rule_data = {
            "name": "High CPU Usage",
            "description": "Alert when CPU usage exceeds 90%",
            "condition": "cpu_usage > 90",
            "severity": "high",
            "enabled": True
        }
        
        rule_id = self.alert_system.create_alert_rule(rule_data)
        
        # Delete the rule
        self.alert_system.delete_alert_rule(rule_id)
        
        self.assertNotIn(rule_id, self.alert_system._alert_rules)
    
    def test_evaluate_alert_rules(self):
        """Test evaluating alert rules."""
        rule_data = {
            "name": "High CPU Usage",
            "description": "Alert when CPU usage exceeds 90%",
            "condition": "cpu_usage > 90",
            "severity": "high",
            "enabled": True
        }
        
        self.alert_system.create_alert_rule(rule_data)
        
        # Evaluate with high CPU usage
        metrics = {"cpu_usage": 95}
        alerts = self.alert_system.evaluate_alert_rules(metrics)
        
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["rule_name"], "High CPU Usage")
        self.assertEqual(alerts[0]["severity"], "high")
        
        # Evaluate with normal CPU usage
        metrics = {"cpu_usage": 50}
        alerts = self.alert_system.evaluate_alert_rules(metrics)
        
        self.assertEqual(len(alerts), 0)
    
    def test_subscribe_to_alerts(self):
        """Test subscribing to alerts."""
        subscriber = Mock()
        
        subscription_id = self.alert_system.subscribe_to_alerts(subscriber)
        
        self.assertIsNotNone(subscription_id)
        self.assertIn(subscription_id, self.alert_system._alert_subscribers)
        self.assertEqual(self.alert_system._alert_subscribers[subscription_id], subscriber)
    
    def test_unsubscribe_from_alerts(self):
        """Test unsubscribing from alerts."""
        subscriber = Mock()
        
        subscription_id = self.alert_system.subscribe_to_alerts(subscriber)
        
        # Unsubscribe
        self.alert_system.unsubscribe_from_alerts(subscription_id)
        
        self.assertNotIn(subscription_id, self.alert_system._alert_subscribers)
    
    def test_notify_subscribers(self):
        """Test notifying subscribers of alerts."""
        subscriber1 = Mock()
        subscriber2 = Mock()
        
        self.alert_system.subscribe_to_alerts(subscriber1)
        self.alert_system.subscribe_to_alerts(subscriber2)
        
        # Create an alert
        alert = {
            "rule_name": "High CPU Usage",
            "severity": "high",
            "message": "CPU usage is 95%",
            "timestamp": datetime.now()
        }
        
        self.alert_system._notify_subscribers(alert)
        
        subscriber1.handle_alert.assert_called_once_with(alert)
        subscriber2.handle_alert.assert_called_once_with(alert)
    
    def test_get_alert_history(self):
        """Test getting alert history."""
        # Create some alerts
        alert1 = {
            "rule_name": "High CPU Usage",
            "severity": "high",
            "message": "CPU usage is 95%",
            "timestamp": datetime.now() - timedelta(hours=1)
        }
        
        alert2 = {
            "rule_name": "Low Memory",
            "severity": "medium",
            "message": "Memory usage is 85%",
            "timestamp": datetime.now() - timedelta(minutes=30)
        }
        
        self.alert_system._alerts.append(alert1)
        self.alert_system._alerts.append(alert2)
        
        # Get alert history
        history = self.alert_system.get_alert_history()
        
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0]["rule_name"], "High CPU Usage")
        self.assertEqual(history[1]["rule_name"], "Low Memory")
        
        # Get filtered alert history
        history = self.alert_system.get_alert_history(severity="high")
        
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["rule_name"], "High CPU Usage")


class TestObservabilityConfig(unittest.TestCase):
    """Test suite for the ObservabilityConfig class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.observability_config = ObservabilityConfig()
    
    def test_observability_config_initialization(self):
        """Test that the observability config initializes correctly."""
        self.assertIsNotNone(self.observability_config)
        self.assertIsNotNone(self.observability_config._config)
        self.assertIn("tracing", self.observability_config._config)
        self.assertIn("metrics", self.observability_config._config)
        self.assertIn("logging", self.observability_config._config)
        self.assertIn("alerts", self.observability_config._config)
    
    def test_get_config(self):
        """Test getting the configuration."""
        config = self.observability_config.get_config()
        
        self.assertIsNotNone(config)
        self.assertIn("tracing", config)
        self.assertIn("metrics", config)
        self.assertIn("logging", config)
        self.assertIn("alerts", config)
    
    def test_set_config(self):
        """Test setting the configuration."""
        new_config = {
            "tracing": {
                "enabled": True,
                "sampling_rate": 0.5,
                "exporter": "jaeger"
            },
            "metrics": {
                "enabled": True,
                "exporter": "prometheus",
                "interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "file"
            },
            "alerts": {
                "enabled": True,
                "channels": ["email", "slack"]
            }
        }
        
        self.observability_config.set_config(new_config)
        
        config = self.observability_config.get_config()
        
        self.assertEqual(config["tracing"]["enabled"], True)
        self.assertEqual(config["tracing"]["sampling_rate"], 0.5)
        self.assertEqual(config["tracing"]["exporter"], "jaeger")
        self.assertEqual(config["metrics"]["enabled"], True)
        self.assertEqual(config["metrics"]["exporter"], "prometheus")
        self.assertEqual(config["metrics"]["interval"], 30)
        self.assertEqual(config["logging"]["level"], "INFO")
        self.assertEqual(config["logging"]["format"], "json")
        self.assertEqual(config["logging"]["output"], "file")
        self.assertEqual(config["alerts"]["enabled"], True)
        self.assertEqual(config["alerts"]["channels"], ["email", "slack"])
    
    def test_get_tracing_config(self):
        """Test getting the tracing configuration."""
        tracing_config = self.observability_config.get_tracing_config()
        
        self.assertIsNotNone(tracing_config)
        self.assertIn("enabled", tracing_config)
        self.assertIn("sampling_rate", tracing_config)
        self.assertIn("exporter", tracing_config)
    
    def test_set_tracing_config(self):
        """Test setting the tracing configuration."""
        new_tracing_config = {
            "enabled": True,
            "sampling_rate": 0.5,
            "exporter": "jaeger"
        }
        
        self.observability_config.set_tracing_config(new_tracing_config)
        
        tracing_config = self.observability_config.get_tracing_config()
        
        self.assertEqual(tracing_config["enabled"], True)
        self.assertEqual(tracing_config["sampling_rate"], 0.5)
        self.assertEqual(tracing_config["exporter"], "jaeger")
    
    def test_get_metrics_config(self):
        """Test getting the metrics configuration."""
        metrics_config = self.observability_config.get_metrics_config()
        
        self.assertIsNotNone(metrics_config)
        self.assertIn("enabled", metrics_config)
        self.assertIn("exporter", metrics_config)
        self.assertIn("interval", metrics_config)
    
    def test_set_metrics_config(self):
        """Test setting the metrics configuration."""
        new_metrics_config = {
            "enabled": True,
            "exporter": "prometheus",
            "interval": 30
        }
        
        self.observability_config.set_metrics_config(new_metrics_config)
        
        metrics_config = self.observability_config.get_metrics_config()
        
        self.assertEqual(metrics_config["enabled"], True)
        self.assertEqual(metrics_config["exporter"], "prometheus")
        self.assertEqual(metrics_config["interval"], 30)
    
    def test_get_logging_config(self):
        """Test getting the logging configuration."""
        logging_config = self.observability_config.get_logging_config()
        
        self.assertIsNotNone(logging_config)
        self.assertIn("level", logging_config)
        self.assertIn("format", logging_config)
        self.assertIn("output", logging_config)
    
    def test_set_logging_config(self):
        """Test setting the logging configuration."""
        new_logging_config = {
            "level": "INFO",
            "format": "json",
            "output": "file"
        }
        
        self.observability_config.set_logging_config(new_logging_config)
        
        logging_config = self.observability_config.get_logging_config()
        
        self.assertEqual(logging_config["level"], "INFO")
        self.assertEqual(logging_config["format"], "json")
        self.assertEqual(logging_config["output"], "file")
    
    def test_get_alerts_config(self):
        """Test getting the alerts configuration."""
        alerts_config = self.observability_config.get_alerts_config()
        
        self.assertIsNotNone(alerts_config)
        self.assertIn("enabled", alerts_config)
        self.assertIn("channels", alerts_config)
    
    def test_set_alerts_config(self):
        """Test setting the alerts configuration."""
        new_alerts_config = {
            "enabled": True,
            "channels": ["email", "slack"]
        }
        
        self.observability_config.set_alerts_config(new_alerts_config)
        
        alerts_config = self.observability_config.get_alerts_config()
        
        self.assertEqual(alerts_config["enabled"], True)
        self.assertEqual(alerts_config["channels"], ["email", "slack"])
    
    def test_load_config_from_file(self):
        """Test loading configuration from file."""
        # Create a temporary config file
        import tempfile
        import json
        
        config_data = {
            "tracing": {
                "enabled": True,
                "sampling_rate": 0.5,
                "exporter": "jaeger"
            },
            "metrics": {
                "enabled": True,
                "exporter": "prometheus",
                "interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "file"
            },
            "alerts": {
                "enabled": True,
                "channels": ["email", "slack"]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_file_path = f.name
        
        try:
            self.observability_config.load_config_from_file(temp_file_path)
            
            config = self.observability_config.get_config()
            
            self.assertEqual(config["tracing"]["enabled"], True)
            self.assertEqual(config["tracing"]["sampling_rate"], 0.5)
            self.assertEqual(config["tracing"]["exporter"], "jaeger")
            self.assertEqual(config["metrics"]["enabled"], True)
            self.assertEqual(config["metrics"]["exporter"], "prometheus")
            self.assertEqual(config["metrics"]["interval"], 30)
            self.assertEqual(config["logging"]["level"], "INFO")
            self.assertEqual(config["logging"]["format"], "json")
            self.assertEqual(config["logging"]["output"], "file")
            self.assertEqual(config["alerts"]["enabled"], True)
            self.assertEqual(config["alerts"]["channels"], ["email", "slack"])
        finally:
            os.unlink(temp_file_path)
    
    def test_save_config_to_file(self):
        """Test saving configuration to file."""
        # Set a specific configuration
        new_config = {
            "tracing": {
                "enabled": True,
                "sampling_rate": 0.5,
                "exporter": "jaeger"
            },
            "metrics": {
                "enabled": True,
                "exporter": "prometheus",
                "interval": 30
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "file"
            },
            "alerts": {
                "enabled": True,
                "channels": ["email", "slack"]
            }
        }
        
        self.observability_config.set_config(new_config)
        
        # Save to a temporary file
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_file_path = f.name
        
        try:
            self.observability_config.save_config_to_file(temp_file_path)
            
            # Read the file and verify the contents
            with open(temp_file_path, 'r') as f:
                saved_config = json.load(f)
            
            self.assertEqual(saved_config["tracing"]["enabled"], True)
            self.assertEqual(saved_config["tracing"]["sampling_rate"], 0.5)
            self.assertEqual(saved_config["tracing"]["exporter"], "jaeger")
            self.assertEqual(saved_config["metrics"]["enabled"], True)
            self.assertEqual(saved_config["metrics"]["exporter"], "prometheus")
            self.assertEqual(saved_config["metrics"]["interval"], 30)
            self.assertEqual(saved_config["logging"]["level"], "INFO")
            self.assertEqual(saved_config["logging"]["format"], "json")
            self.assertEqual(saved_config["logging"]["output"], "file")
            self.assertEqual(saved_config["alerts"]["enabled"], True)
            self.assertEqual(saved_config["alerts"]["channels"], ["email", "slack"])
        finally:
            os.unlink(temp_file_path)


class TestTraceAnalyzer(unittest.TestCase):
    """Test suite for the TraceAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.trace_analyzer = TraceAnalyzer()
    
    def test_trace_analyzer_initialization(self):
        """Test that the trace analyzer initializes correctly."""
        self.assertIsNotNone(self.trace_analyzer)
        self.assertEqual(len(self.trace_analyzer._traces), 0)
        self.assertEqual(len(self.trace_analyzer._trace_patterns), 0)
        self.assertEqual(len(self.trace_analyzer._anomalies), 0)
    
    def test_add_trace(self):
        """Test adding a trace."""
        trace_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,  # milliseconds
            "tags": {"tag1": "value1", "tag2": "value2"},
            "logs": [{"timestamp": datetime.now(), "message": "log1"}],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace_id = self.trace_analyzer.add_trace(trace_data)
        
        self.assertEqual(trace_id, "trace1")
        self.assertIn("trace1", self.trace_analyzer._traces)
        self.assertEqual(self.trace_analyzer._traces["trace1"]["operation_name"], "operation1")
        self.assertEqual(self.trace_analyzer._traces["trace1"]["duration"], 100)
    
    def test_get_trace(self):
        """Test getting a trace."""
        trace_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,  # milliseconds
            "tags": {"tag1": "value1", "tag2": "value2"},
            "logs": [{"timestamp": datetime.now(), "message": "log1"}],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace_data)
        
        trace = self.trace_analyzer.get_trace("trace1")
        
        self.assertIsNotNone(trace)
        self.assertEqual(trace["trace_id"], "trace1")
        self.assertEqual(trace["operation_name"], "operation1")
        self.assertEqual(trace["duration"], 100)
    
    def test_find_traces_by_operation(self):
        """Test finding traces by operation name."""
        # Add multiple traces
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {"tag1": "value1"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation2",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=200),
            "duration": 200,
            "tags": {"tag2": "value2"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace3_data = {
            "trace_id": "trace3",
            "span_id": "span3",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=150),
            "duration": 150,
            "tags": {"tag3": "value3"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        self.trace_analyzer.add_trace(trace3_data)
        
        # Find traces by operation
        traces = self.trace_analyzer.find_traces_by_operation("operation1")
        
        self.assertEqual(len(traces), 2)
        self.assertTrue(any(t["trace_id"] == "trace1" for t in traces))
        self.assertTrue(any(t["trace_id"] == "trace3" for t in traces))
    
    def test_find_traces_by_tag(self):
        """Test finding traces by tag."""
        # Add multiple traces
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {"tag1": "value1", "common": "shared"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation2",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=200),
            "duration": 200,
            "tags": {"tag2": "value2"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace3_data = {
            "trace_id": "trace3",
            "span_id": "span3",
            "parent_span_id": None,
            "operation_name": "operation3",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=150),
            "duration": 150,
            "tags": {"tag3": "value3", "common": "shared"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        self.trace_analyzer.add_trace(trace3_data)
        
        # Find traces by tag
        traces = self.trace_analyzer.find_traces_by_tag("common", "shared")
        
        self.assertEqual(len(traces), 2)
        self.assertTrue(any(t["trace_id"] == "trace1" for t in traces))
        self.assertTrue(any(t["trace_id"] == "trace3" for t in traces))
    
    def test_find_traces_by_time_range(self):
        """Test finding traces by time range."""
        now = datetime.now()
        
        # Add multiple traces
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": now - timedelta(hours=2),
            "end_time": now - timedelta(hours=2) + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation2",
            "start_time": now - timedelta(hours=1),
            "end_time": now - timedelta(hours=1) + timedelta(milliseconds=200),
            "duration": 200,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace3_data = {
            "trace_id": "trace3",
            "span_id": "span3",
            "parent_span_id": None,
            "operation_name": "operation3",
            "start_time": now - timedelta(minutes=30),
            "end_time": now - timedelta(minutes=30) + timedelta(milliseconds=150),
            "duration": 150,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        self.trace_analyzer.add_trace(trace3_data)
        
        # Find traces by time range
        start_time = now - timedelta(hours=1, minutes=30)
        end_time = now - timedelta(minutes=15)
        
        traces = self.trace_analyzer.find_traces_by_time_range(start_time, end_time)
        
        self.assertEqual(len(traces), 1)
        self.assertEqual(traces[0]["trace_id"], "trace2")
    
    def test_calculate_average_duration(self):
        """Test calculating average duration."""
        # Add multiple traces
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=200),
            "duration": 200,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace3_data = {
            "trace_id": "trace3",
            "span_id": "span3",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=150),
            "duration": 150,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        self.trace_analyzer.add_trace(trace3_data)
        
        # Calculate average duration
        avg_duration = self.trace_analyzer.calculate_average_duration("operation1")
        
        self.assertEqual(avg_duration, 150)  # (100 + 200 + 150) / 3
    
    def test_detect_anomalies(self):
        """Test detecting anomalies in traces."""
        # Add multiple traces
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=120),
            "duration": 120,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace3_data = {
            "trace_id": "trace3",
            "span_id": "span3",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=500),  # Anomaly
            "duration": 500,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        self.trace_analyzer.add_trace(trace3_data)
        
        # Detect anomalies
        anomalies = self.trace_analyzer.detect_anomalies("operation1", threshold=2.0)
        
        self.assertEqual(len(anomalies), 1)
        self.assertEqual(anomalies[0]["trace_id"], "trace3")
        self.assertIn("reason", anomalies[0])
        self.assertIn("severity", anomalies[0])
    
    def test_identify_patterns(self):
        """Test identifying patterns in traces."""
        # Add multiple traces with similar patterns
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {"pattern": "A"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=110),
            "duration": 110,
            "tags": {"pattern": "A"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace3_data = {
            "trace_id": "trace3",
            "span_id": "span3",
            "parent_span_id": None,
            "operation_name": "operation2",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=200),
            "duration": 200,
            "tags": {"pattern": "B"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace4_data = {
            "trace_id": "trace4",
            "span_id": "span4",
            "parent_span_id": None,
            "operation_name": "operation2",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=210),
            "duration": 210,
            "tags": {"pattern": "B"},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        self.trace_analyzer.add_trace(trace3_data)
        self.trace_analyzer.add_trace(trace4_data)
        
        # Identify patterns
        patterns = self.trace_analyzer.identify_patterns()
        
        self.assertEqual(len(patterns), 2)
        self.assertTrue(any(p["operation"] == "operation1" and p["tag"] == "pattern" and p["value"] == "A" for p in patterns))
        self.assertTrue(any(p["operation"] == "operation2" and p["tag"] == "pattern" and p["value"] == "B" for p in patterns))
    
    def test_generate_trace_report(self):
        """Test generating a trace report."""
        # Add multiple traces
        trace1_data = {
            "trace_id": "trace1",
            "span_id": "span1",
            "parent_span_id": None,
            "operation_name": "operation1",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=100),
            "duration": 100,
            "tags": {},
            "logs": [],
            "status_code": 0,
            "status_message": "OK"
        }
        
        trace2_data = {
            "trace_id": "trace2",
            "span_id": "span2",
            "parent_span_id": None,
            "operation_name": "operation2",
            "start_time": datetime.now(),
            "end_time": datetime.now() + timedelta(milliseconds=200),
            "duration": 200,
            "tags": {},
            "logs": [],
            "status_code": 1,
            "status_message": "Error"
        }
        
        self.trace_analyzer.add_trace(trace1_data)
        self.trace_analyzer.add_trace(trace2_data)
        
        # Generate report
        report = self.trace_analyzer.generate_trace_report()
        
        self.assertIsNotNone(report)
        self.assertIn("total_traces", report)
        self.assertIn("operations", report)
        self.assertIn("average_durations", report)
        self.assertIn("error_rates", report)
        self.assertEqual(report["total_traces"], 2)
        self.assertEqual(len(report["operations"]), 2)
        self.assertIn("operation1", report["average_durations"])
        self.assertIn("operation2", report["average_durations"])
        self.assertEqual(report["average_durations"]["operation1"], 100)
        self.assertEqual(report["average_durations"]["operation2"], 200)
        self.assertEqual(report["error_rates"]["operation1"], 0.0)
        self.assertEqual(report["error_rates"]["operation2"], 1.0)


if __name__ == '__main__':
    unittest.main()