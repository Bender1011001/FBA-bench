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

from plugins.plugin_framework import PluginFramework
from plugins.agent_plugins.base_agent_plugin import BaseAgentPlugin
from plugins.scenario_plugins.base_scenario_plugin import BaseScenarioPlugin


class TestPluginFramework(unittest.TestCase):
    """Test suite for the PluginFramework class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.plugin_framework = PluginFramework()
    
    def test_plugin_framework_initialization(self):
        """Test that the plugin framework initializes correctly."""
        self.assertIsNotNone(self.plugin_framework)
        self.assertEqual(len(self.plugin_framework._plugins), 0)
        self.assertEqual(len(self.plugin_framework._plugin_types), 0)
        self.assertEqual(len(self.plugin_framework._plugin_dependencies), 0)
    
    def test_register_plugin_type(self):
        """Test registering a plugin type."""
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        self.assertIsNotNone(type_id)
        self.assertIn(type_id, self.plugin_framework._plugin_types)
        self.assertEqual(self.plugin_framework._plugin_types[type_id]["name"], "agent")
        self.assertEqual(self.plugin_framework._plugin_types[type_id]["description"], "Agent plugins")
        self.assertEqual(self.plugin_framework._plugin_types[type_id]["base_class"], "BaseAgentPlugin")
    
    def test_register_plugin(self):
        """Test registering a plugin."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        self.assertIsNotNone(plugin_id)
        self.assertIn(plugin_id, self.plugin_framework._plugins)
        self.assertEqual(self.plugin_framework._plugins[plugin_id]["name"], "Test Agent Plugin")
        self.assertEqual(self.plugin_framework._plugins[plugin_id]["type"], type_id)
    
    def test_unregister_plugin(self):
        """Test unregistering a plugin."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        # Unregister the plugin
        self.plugin_framework.unregister_plugin(plugin_id)
        
        self.assertNotIn(plugin_id, self.plugin_framework._plugins)
    
    def test_load_plugin(self):
        """Test loading a plugin."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        # Mock the plugin loading
        with patch.object(self.plugin_framework, '_load_plugin_module') as mock_load:
            mock_plugin_instance = Mock()
            mock_load.return_value = mock_plugin_instance
            
            # Load the plugin
            plugin_instance = self.plugin_framework.load_plugin(plugin_id)
            
            self.assertEqual(plugin_instance, mock_plugin_instance)
            mock_load.assert_called_once_with(plugin_id)
    
    def test_unload_plugin(self):
        """Test unloading a plugin."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        # Mock the plugin instance
        plugin_instance = Mock()
        self.plugin_framework._loaded_plugins[plugin_id] = plugin_instance
        
        # Unload the plugin
        self.plugin_framework.unload_plugin(plugin_id)
        
        self.assertNotIn(plugin_id, self.plugin_framework._loaded_plugins)
        plugin_instance.cleanup.assert_called_once()
    
    def test_get_plugin(self):
        """Test getting a plugin."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        # Get the plugin
        retrieved_plugin = self.plugin_framework.get_plugin(plugin_id)
        
        self.assertIsNotNone(retrieved_plugin)
        self.assertEqual(retrieved_plugin["name"], "Test Agent Plugin")
        self.assertEqual(retrieved_plugin["type"], type_id)
    
    def test_get_plugins_by_type(self):
        """Test getting plugins by type."""
        # Register two plugin types
        agent_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        scenario_type = {
            "name": "scenario",
            "description": "Scenario plugins",
            "base_class": "BaseScenarioPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        agent_type_id = self.plugin_framework.register_plugin_type(agent_type)
        scenario_type_id = self.plugin_framework.register_plugin_type(scenario_type)
        
        # Register plugins of different types
        agent_plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": agent_type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        scenario_plugin = {
            "name": "Test Scenario Plugin",
            "description": "A test scenario plugin",
            "version": "1.0.0",
            "type": scenario_type_id,
            "module_path": "plugins.scenario_plugins.test_scenario_plugin",
            "class_name": "TestScenarioPlugin",
            "config": {},
            "dependencies": []
        }
        
        agent_plugin_id = self.plugin_framework.register_plugin(agent_plugin)
        scenario_plugin_id = self.plugin_framework.register_plugin(scenario_plugin)
        
        # Get plugins by type
        agent_plugins = self.plugin_framework.get_plugins_by_type(agent_type_id)
        scenario_plugins = self.plugin_framework.get_plugins_by_type(scenario_type_id)
        
        self.assertEqual(len(agent_plugins), 1)
        self.assertEqual(len(scenario_plugins), 1)
        self.assertEqual(agent_plugins[0]["name"], "Test Agent Plugin")
        self.assertEqual(scenario_plugins[0]["name"], "Test Scenario Plugin")
    
    def test_resolve_plugin_dependencies(self):
        """Test resolving plugin dependencies."""
        # Register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Register plugins with dependencies
        plugin1 = {
            "name": "Plugin 1",
            "description": "A test plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_plugin1",
            "class_name": "TestPlugin1",
            "config": {},
            "dependencies": []
        }
        
        plugin2 = {
            "name": "Plugin 2",
            "description": "A test plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_plugin2",
            "class_name": "TestPlugin2",
            "config": {},
            "dependencies": ["plugin1"]
        }
        
        plugin3 = {
            "name": "Plugin 3",
            "description": "A test plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_plugin3",
            "class_name": "TestPlugin3",
            "config": {},
            "dependencies": ["plugin1", "plugin2"]
        }
        
        plugin1_id = self.plugin_framework.register_plugin(plugin1)
        plugin2_id = self.plugin_framework.register_plugin(plugin2)
        plugin3_id = self.plugin_framework.register_plugin(plugin3)
        
        # Update dependencies with actual plugin IDs
        self.plugin_framework._plugins[plugin2_id]["dependencies"] = [plugin1_id]
        self.plugin_framework._plugins[plugin3_id]["dependencies"] = [plugin1_id, plugin2_id]
        
        # Resolve dependencies
        resolved_order = self.plugin_framework.resolve_plugin_dependencies([plugin3_id, plugin2_id, plugin1_id])
        
        self.assertEqual(len(resolved_order), 3)
        self.assertEqual(resolved_order[0], plugin1_id)  # Plugin 1 has no dependencies
        self.assertEqual(resolved_order[1], plugin2_id)  # Plugin 2 depends on Plugin 1
        self.assertEqual(resolved_order[2], plugin3_id)  # Plugin 3 depends on Plugin 1 and Plugin 2
    
    def test_execute_plugin(self):
        """Test executing a plugin."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        # Mock the plugin instance
        plugin_instance = Mock()
        plugin_instance.execute.return_value = {"result": "success"}
        self.plugin_framework._loaded_plugins[plugin_id] = plugin_instance
        
        # Execute the plugin
        result = self.plugin_framework.execute_plugin(plugin_id, {"input": "test"})
        
        self.assertEqual(result, {"result": "success"})
        plugin_instance.execute.assert_called_once_with({"input": "test"})
    
    def test_get_plugin_status(self):
        """Test getting plugin status."""
        # First register a plugin type
        plugin_type = {
            "name": "agent",
            "description": "Agent plugins",
            "base_class": "BaseAgentPlugin",
            "interface": {
                "initialize": "method",
                "execute": "method",
                "cleanup": "method"
            }
        }
        
        type_id = self.plugin_framework.register_plugin_type(plugin_type)
        
        # Now register a plugin
        plugin = {
            "name": "Test Agent Plugin",
            "description": "A test agent plugin",
            "version": "1.0.0",
            "type": type_id,
            "module_path": "plugins.agent_plugins.test_agent_plugin",
            "class_name": "TestAgentPlugin",
            "config": {},
            "dependencies": []
        }
        
        plugin_id = self.plugin_framework.register_plugin(plugin)
        
        # Get plugin status (not loaded)
        status = self.plugin_framework.get_plugin_status(plugin_id)
        
        self.assertEqual(status, "registered")
        
        # Mock the plugin instance
        plugin_instance = Mock()
        self.plugin_framework._loaded_plugins[plugin_id] = plugin_instance
        
        # Get plugin status (loaded)
        status = self.plugin_framework.get_plugin_status(plugin_id)
        
        self.assertEqual(status, "loaded")


class TestBaseAgentPlugin(unittest.TestCase):
    """Test suite for the BaseAgentPlugin class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a concrete implementation of BaseAgentPlugin for testing
        class TestAgentPlugin(BaseAgentPlugin):
            def __init__(self):
                super().__init__()
                self.initialized = False
            
            def initialize(self, config):
                self.initialized = True
                return {"status": "initialized"}
            
            def execute(self, input_data):
                if not self.initialized:
                    raise RuntimeError("Plugin not initialized")
                return {"result": "executed", "input": input_data}
            
            def cleanup(self):
                self.initialized = False
                return {"status": "cleaned"}
        
        self.agent_plugin = TestAgentPlugin()
    
    def test_base_agent_plugin_initialization(self):
        """Test that the base agent plugin initializes correctly."""
        self.assertIsNotNone(self.agent_plugin)
        self.assertFalse(self.agent_plugin.initialized)
        self.assertEqual(self.agent_plugin.name, "")
        self.assertEqual(self.agent_plugin.version, "")
        self.assertEqual(self.agent_plugin.description, "")
    
    def test_base_agent_plugin_initialize(self):
        """Test initializing the base agent plugin."""
        config = {"param1": "value1", "param2": "value2"}
        
        result = self.agent_plugin.initialize(config)
        
        self.assertTrue(self.agent_plugin.initialized)
        self.assertEqual(result, {"status": "initialized"})
    
    def test_base_agent_plugin_execute(self):
        """Test executing the base agent plugin."""
        # Initialize first
        self.agent_plugin.initialize({})
        
        input_data = {"data": "test"}
        
        result = self.agent_plugin.execute(input_data)
        
        self.assertEqual(result, {"result": "executed", "input": input_data})
    
    def test_base_agent_plugin_execute_without_initialization(self):
        """Test executing the base agent plugin without initialization."""
        input_data = {"data": "test"}
        
        with self.assertRaises(RuntimeError):
            self.agent_plugin.execute(input_data)
    
    def test_base_agent_plugin_cleanup(self):
        """Test cleaning up the base agent plugin."""
        # Initialize first
        self.agent_plugin.initialize({})
        
        result = self.agent_plugin.cleanup()
        
        self.assertFalse(self.agent_plugin.initialized)
        self.assertEqual(result, {"status": "cleaned"})
    
    def test_base_agent_plugin_get_info(self):
        """Test getting info about the base agent plugin."""
        info = self.agent_plugin.get_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("version", info)
        self.assertIn("description", info)
        self.assertEqual(info["name"], "")
        self.assertEqual(info["version"], "")
        self.assertEqual(info["description"], "")


class TestBaseScenarioPlugin(unittest.TestCase):
    """Test suite for the BaseScenarioPlugin class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create a concrete implementation of BaseScenarioPlugin for testing
        class TestScenarioPlugin(BaseScenarioPlugin):
            def __init__(self):
                super().__init__()
                self.initialized = False
            
            def initialize(self, config):
                self.initialized = True
                return {"status": "initialized"}
            
            def execute(self, input_data):
                if not self.initialized:
                    raise RuntimeError("Plugin not initialized")
                return {"result": "executed", "input": input_data}
            
            def cleanup(self):
                self.initialized = False
                return {"status": "cleaned"}
            
            def get_scenario_config(self):
                return {
                    "name": "Test Scenario",
                    "description": "A test scenario",
                    "parameters": {
                        "param1": {"type": "string", "description": "Parameter 1"},
                        "param2": {"type": "integer", "description": "Parameter 2"}
                    }
                }
        
        self.scenario_plugin = TestScenarioPlugin()
    
    def test_base_scenario_plugin_initialization(self):
        """Test that the base scenario plugin initializes correctly."""
        self.assertIsNotNone(self.scenario_plugin)
        self.assertFalse(self.scenario_plugin.initialized)
        self.assertEqual(self.scenario_plugin.name, "")
        self.assertEqual(self.scenario_plugin.version, "")
        self.assertEqual(self.scenario_plugin.description, "")
    
    def test_base_scenario_plugin_initialize(self):
        """Test initializing the base scenario plugin."""
        config = {"param1": "value1", "param2": "value2"}
        
        result = self.scenario_plugin.initialize(config)
        
        self.assertTrue(self.scenario_plugin.initialized)
        self.assertEqual(result, {"status": "initialized"})
    
    def test_base_scenario_plugin_execute(self):
        """Test executing the base scenario plugin."""
        # Initialize first
        self.scenario_plugin.initialize({})
        
        input_data = {"data": "test"}
        
        result = self.scenario_plugin.execute(input_data)
        
        self.assertEqual(result, {"result": "executed", "input": input_data})
    
    def test_base_scenario_plugin_execute_without_initialization(self):
        """Test executing the base scenario plugin without initialization."""
        input_data = {"data": "test"}
        
        with self.assertRaises(RuntimeError):
            self.scenario_plugin.execute(input_data)
    
    def test_base_scenario_plugin_cleanup(self):
        """Test cleaning up the base scenario plugin."""
        # Initialize first
        self.scenario_plugin.initialize({})
        
        result = self.scenario_plugin.cleanup()
        
        self.assertFalse(self.scenario_plugin.initialized)
        self.assertEqual(result, {"status": "cleaned"})
    
    def test_base_scenario_plugin_get_scenario_config(self):
        """Test getting scenario config from the base scenario plugin."""
        config = self.scenario_plugin.get_scenario_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn("name", config)
        self.assertIn("description", config)
        self.assertIn("parameters", config)
        self.assertEqual(config["name"], "Test Scenario")
        self.assertEqual(config["description"], "A test scenario")
        self.assertIn("param1", config["parameters"])
        self.assertIn("param2", config["parameters"])
        self.assertEqual(config["parameters"]["param1"]["type"], "string")
        self.assertEqual(config["parameters"]["param2"]["type"], "integer")
    
    def test_base_scenario_plugin_get_info(self):
        """Test getting info about the base scenario plugin."""
        info = self.scenario_plugin.get_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn("name", info)
        self.assertIn("version", info)
        self.assertIn("description", info)
        self.assertEqual(info["name"], "")
        self.assertEqual(info["version"], "")
        self.assertEqual(info["description"], "")


if __name__ == '__main__':
    unittest.main()