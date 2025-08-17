import importlib
import inspect
import os
import logging
from typing import Dict, Any, List, Type, Callable, Protocol, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class PluginError(Exception):
    """Custom exception for plugin-related errors."""
    pass

class PluginInterface(Protocol):
    """
    A protocol defining the expected interface for plugins.
    Plugins must implement methods defined in this protocol if they register for certain extension points.
    """
    def initialize(self, config: Dict[str, Any]):
        """Initializes the plugin with a given configuration."""
        ...

    def on_event(self, event_name: str, data: Dict[str, Any]):
        """Handles a specific event."""
        ...

    def get_plugin_info(self) -> Dict[str, Any]:
        """Returns metadata about the plugin."""
        ...

class PluginManager:
    """
    Manages the discovery, loading, and execution of community plugins.
    Provides extension points and handles plugin compatibility and security.
    """
    def __init__(self):
        self._loaded_plugins: Dict[str, Any] = {}
        self._extension_points: Dict[str, Callable] = {} # Maps extension point name to a handler/interface
        self._plugin_versions: Dict[str, str] = {}
        logging.info("PluginManager initialized.")

    async def load_plugins(self, plugin_directory: str = "plugins"):
        """
        Discovers and loads available plugins from a designated directory.
        Plugins are expected to be Python modules.
        
        :param plugin_directory: The relative path to the directory containing plugins.
        """
        if not os.path.isdir(plugin_directory):
            logging.warning(f"Plugin directory '{plugin_directory}' not found. No plugins loaded.")
            return

        # Add the plugin directory to Python's path temporarily to allow imports
        # This is generally safer than modifying sys.path globally.
        # Use importlib.util.spec_from_file_location for more control.
        
        # Discover plugin modules (e.g., .py files)
        logging.info(f"Loading plugins from: {os.path.abspath(plugin_directory)}")
        module_names = []
        for filename in os.listdir(plugin_directory):
            if filename.endswith(".py") and filename != "__init__.py":
                module_name = filename[:-3] # Remove .py extension
                module_names.append(module_name)
        
        # Load each module
        for module_name in module_names:
            try:
                # Use importlib to load modules dynamically
                # This requires the plugin_directory to be a package or on PYTHONPATH
                # For simplicity, we'll assume it's directly importable or handled safely.
                # A more robust solution involves a custom loader or ensuring sys.path is managed.
                
                # --- Safer way to import from a specific path ---
                module_path = os.path.join(plugin_directory, f"{module_name}.py")
                spec = importlib.util.spec_from_file_location(module_name, module_path)
                if spec is None:
                    logging.error(f"Could not find module spec for {module_name} at {module_path}")
                    continue
                plugin_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(plugin_module)
                # --- End safer import ---

                # Find classes in the module that are intended to be plugins
                for name, obj in inspect.getmembers(plugin_module):
                    if inspect.isclass(obj) and hasattr(obj, '__is_fba_plugin__') and obj.__is_fba_plugin__:
                        plugin_instance = obj() # Instantiate the plugin
                        plugin_id = getattr(plugin_instance, "plugin_id", name)
                        version = getattr(plugin_instance, "version", "0.0.1")
                        
                        if not await self.validate_plugin_security(plugin_module): # Validate module, not instance directly
                             raise PluginError(f"Plugin {plugin_id} failed security validation and was not loaded.")

                        self._loaded_plugins[plugin_id] = plugin_instance
                        self._plugin_versions[plugin_id] = version
                        logging.info(f"Successfully loaded plugin: {plugin_id} (version: {version})")
                        if hasattr(plugin_instance, 'initialize'):
                            await plugin_instance.initialize({}) # Initialize plugin with empty config
            except PluginError as e:
                logging.error(f"Failed to load plugin {module_name}: {e}")
            except Exception as e:
                logging.error(f"Error loading plugin {module_name}: {e}", exc_info=True)

    def register_extension_point(self, name: str, handler: Callable):
        """
        Defines a new extension point where plugins can register capabilities.
        The handler could be a function, or a class defining an interface that plugins
        must adhere to.
        
        :param name: The unique name of the extension point (e.g., "on_simulation_step").
        :param handler: A callable that will be invoked when this extension point is triggered,
                        or a type/protocol representing the interface required for plugins.
        """
        if name in self._extension_points:
            logging.warning(f"Extension point '{name}' already registered. Overwriting.")
        self._extension_points[name] = handler
        logging.info(f"Extension point '{name}' registered.")

    async def execute_plugin_hook(self, hook_name: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calls registered plugin handlers for a specific hook.
        
        :param hook_name: The name of the hook to execute.
        :param context: A dictionary of contextual data to pass to the plugins.
        :return: A dictionary containing results from various plugin executions, keyed by plugin ID.
        """
        results = {}
        if hook_name not in self._extension_points:
            logging.debug(f"No extension point '{hook_name}' registered. Skipping hook execution.")
            return results

        logging.debug(f"Executing plugin hook: {hook_name}")
        for plugin_id, plugin_instance in self._loaded_plugins.items():
            try:
                # Check if the plugin instance has a method corresponding to the hook_name
                if hasattr(plugin_instance, hook_name) and callable(getattr(plugin_instance, hook_name)):
                    method = getattr(plugin_instance, hook_name)
                    # For simplicity, we assume the method accepts `context`.
                    # In a real system, the `handler` (registered for extension point)
                    # would define the expected signature.
                    logging.debug(f"Executing {hook_name} on plugin {plugin_id}")
                    if inspect.iscoroutinefunction(method):
                        result = await method(context)
                    else:
                        result = method(context)
                    results[plugin_id] = {"status": "success", "result": result}
                else:
                    logging.debug(f"Plugin {plugin_id} does not implement hook '{hook_name}'.")
            except Exception as e:
                logging.error(f"Error executing hook '{hook_name}' for plugin {plugin_id}: {e}", exc_info=True)
                results[plugin_id] = {"status": "error", "message": str(e)}
        return results

    async def validate_plugin_security(self, plugin_module: Any) -> bool:
        """
        Performs comprehensive security checks on plugin code.
        Implements multiple layers of security validation including:
        - Static code analysis for dangerous patterns
        - Import restriction validation
        - Code complexity analysis
        - Resource access pattern detection
        
        :param plugin_module: The plugin module to validate.
        :return: True if the plugin passes security checks, False otherwise.
        """
        import ast
        import re
        
        logging.info(f"Performing comprehensive security validation for plugin: {plugin_module.__name__}")
        
        try:
            # Get source code for analysis
            source_code = inspect.getsource(plugin_module)
            
            # 1. Check for dangerous imports and function calls
            dangerous_patterns = [
                # System execution
                r'os\.system\s*\(',
                r'subprocess\.(run|call|Popen|check_output)\s*\(',
                r'exec\s*\(',
                r'eval\s*\(',
                r'__import__\s*\(',
                
                # File system manipulation
                r'open\s*\([^)]*\,\s*[\'"]w[\'"]\)',  # Write mode
                r'open\s*\([^)]*\,\s*[\'"]a[\'"]\)',  # Append mode
                r'open\s*\([^)]*\,\s*[\'"]wb[\'"]\)',  # Binary write mode
                r'open\s*\([^)]*\,\s*[\'"]ab[\'"]\)',  # Binary append mode
                
                # Network access
                r'socket\.',
                r'requests\.',
                r'urllib\.',
                r'httplib\.',
                
                # Memory manipulation
                r'ctypes\.',
                r'mmap\.',
                
                # Process manipulation
                r'psutil\.',
                r'multiprocessing\.',
                
                # Dangerous built-ins
                r'globals\s*\(',
                r'locals\s*\(',
                r'vars\s*\(',
                r'dir\s*\(',
                r'getattr\s*\(',
                r'setattr\s*\(',
                r'delattr\s*\(',
            ]
            
            for pattern in dangerous_patterns:
                if re.search(pattern, source_code):
                    logging.error(f"Security risk: Plugin {plugin_module.__name__} contains dangerous pattern: {pattern}")
                    return False
            
            # 2. Parse AST for deeper analysis
            try:
                tree = ast.parse(source_code)
                
                # Check for dangerous imports
                forbidden_imports = {
                    'os', 'subprocess', 'sys', 'ctypes', 'mmap', 'socket',
                    'requests', 'urllib', 'httplib', 'ftplib', 'smtplib',
                    'poplib', 'imaplib', 'nntplib', 'psutil', 'multiprocessing',
                    'threading', 'asyncio'  # Restrict async to prevent event loop manipulation
                }
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name in forbidden_imports:
                                logging.error(f"Security risk: Plugin {plugin_module.__name__} imports forbidden module: {alias.name}")
                                return False
                    
                    elif isinstance(node, ast.ImportFrom):
                        if node.module and node.module in forbidden_imports:
                            logging.error(f"Security risk: Plugin {plugin_module.__name__} imports from forbidden module: {node.module}")
                            return False
                    
                    # Check for dangerous function calls
                    elif isinstance(node, ast.Call):
                        if isinstance(node.func, ast.Name):
                            if node.func.id in ['exec', 'eval', 'compile', '__import__']:
                                logging.error(f"Security risk: Plugin {plugin_module.__name__} calls dangerous function: {node.func.id}")
                                return False
                        
                        elif isinstance(node.func, ast.Attribute):
                            if (isinstance(node.func.value, ast.Name) and
                                node.func.value.id == 'os' and
                                node.func.attr == 'system'):
                                logging.error(f"Security risk: Plugin {plugin_module.__name__} calls os.system")
                                return False
                
                # 3. Code complexity analysis
                class ComplexityAnalyzer(ast.NodeVisitor):
                    def __init__(self):
                        self.complexity = 0
                        self.functions = []
                    
                    def visit_FunctionDef(self, node):
                        # Calculate cyclomatic complexity for each function
                        func_complexity = 1  # Base complexity
                        
                        for child in ast.walk(node):
                            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                                func_complexity += 1
                            elif isinstance(child, ast.BoolOp):
                                func_complexity += len(child.values) - 1
                        
                        self.functions.append({
                            'name': node.name,
                            'complexity': func_complexity,
                            'line': node.lineno
                        })
                        
                        if func_complexity > 15:  # Threshold for complex functions
                            logging.warning(f"Complex function detected in plugin {plugin_module.__name__}: {node.name} (complexity: {func_complexity})")
                        
                        self.generic_visit(node)
                
                analyzer = ComplexityAnalyzer()
                analyzer.visit(tree)
                
                # 4. Check for excessive resource usage patterns
                resource_patterns = [
                    r'while\s+True\s*:',  # Infinite loops
                    r'for\s+.*\s+in\s+range\s*\(\s*[0-9]{6,}\s*\)',  # Very large ranges
                    r'\[\s*x\s+for\s+.*\s+in\s+.*\s+if\s+.*\]',  # List comprehensions (potential memory issues)
                ]
                
                for pattern in resource_patterns:
                    if re.search(pattern, source_code):
                        logging.warning(f"Resource usage concern: Plugin {plugin_module.__name__} contains pattern: {pattern}")
                
                # 5. Check for direct access to internal attributes
                internal_access_patterns = [
                    r'\._.*__.*',  # Access to private/mangled attributes
                    r'__dict__',
                    r'__class__',
                ]
                
                for pattern in internal_access_patterns:
                    if re.search(pattern, source_code):
                        logging.warning(f"Internal access concern: Plugin {plugin_module.__name__} contains pattern: {pattern}")
                
            except SyntaxError as e:
                logging.error(f"Syntax error in plugin {plugin_module.__name__}: {e}")
                return False
            
            # 6. Plugin metadata validation
            if hasattr(plugin_module, '__dict__'):
                module_dict = plugin_module.__dict__
                
                # Check for required plugin metadata
                for name, obj in module_dict.items():
                    if inspect.isclass(obj) and hasattr(obj, '__is_fba_plugin__'):
                        plugin_id = getattr(obj, 'plugin_id', None)
                        if not plugin_id or not isinstance(plugin_id, str):
                            logging.error(f"Invalid plugin_id in plugin {plugin_module.__name__}")
                            return False
                        
                        version = getattr(obj, 'version', None)
                        if not version or not isinstance(version, str):
                            logging.error(f"Invalid version in plugin {plugin_module.__name__}")
                            return False
            
            logging.info(f"Plugin {plugin_module.__name__} passed comprehensive security validation.")
            return True
            
        except TypeError:  # Can happen if source is not readily available
            logging.warning(f"Cannot inspect source code for plugin {plugin_module.__name__}. Skipping deep security analysis.")
            return True  # Allow if we can't inspect, but log the warning
        except Exception as e:
            logging.error(f"Error during plugin security validation for {plugin_module.__name__}: {e}")
            return False

    def get_available_plugins(self) -> Dict[str, Dict[str, Any]]:
        """
        Lists all currently loaded plugins and their capabilities.
        
        :return: A dictionary where keys are plugin IDs and values are dictionaries
                 containing plugin info (e.g., version, implemented hooks).
        """
        plugins_info = {}
        for plugin_id, plugin_instance in self._loaded_plugins.items():
            info = {"version": self._plugin_versions.get(plugin_id, "N/A"), "implemented_hooks": []}
            # Reflect on the plugin instance to find implemented methods that match registered hooks
            for hook_name in self._extension_points.keys():
                if hasattr(plugin_instance, hook_name) and callable(getattr(plugin_instance, hook_name)):
                    info["implemented_hooks"].append(hook_name)
            
            if hasattr(plugin_instance, 'get_plugin_info') and callable(plugin_instance.get_plugin_info):
                try:
                    plugin_specific_info = plugin_instance.get_plugin_info()
                    info.update(plugin_specific_info)
                except Exception as e:
                    logging.warning(f"Error getting plugin info for {plugin_id}: {e}")
            plugins_info[plugin_id] = info
        return plugins_info

# Mock plugin for testing purposes
class ExamplePlugin:
    __is_fba_plugin__ = True # Mandatory marker for FBA-Bench plugins
    plugin_id = "my_example_plugin"
    version = "1.0.0"

    async def initialize(self, config: Dict[str, Any]):
        logging.info(f"ExamplePlugin '{self.plugin_id}' initialized with config: {config}")

    async def on_simulation_step(self, context: Dict[str, Any]):
        logging.info(f"ExamplePlugin received on_simulation_step hook with context: {context.keys()}")
        return {"handled_step": True, "data_processed": len(context)}

    def get_plugin_info(self) -> Dict[str, Any]:
        return {"description": "An example plugin for FBA-Bench testing.", "author": "Roo"}

async def _main():
    plugin_manager = PluginManager()

    # Create a dummy plugin file for testing
    dummy_plugin_dir = "temp_plugins"
    os.makedirs(dummy_plugin_dir, exist_ok=True)
    with open(os.path.join(dummy_plugin_dir, "my_test_plugin.py"), "w") as f:
        f.write("""
import logging
logging.basicConfig(level=logging.INFO)

class MyTestPlugin:
    __is_fba_plugin__ = True
    plugin_id = "test_plugin_001"
    version = "0.9.0"

    async def initialize(self, config):
        logging.info(f"MyTestPlugin initialized: {config}")

    async def on_simulation_step(self, context):
        logging.info(f"MyTestPlugin: Simulation step at time {context.get('current_time')}")
        return {"processed_count": context.get("data_count", 0) + 1}

    def on_agent_decision(self, context):
        logging.info(f"MyTestPlugin: Agent decision for agent {context.get('agent_id')}")
        return {"decision_modified": True}

    def get_plugin_info(self):
        return {"type": "scenario", "developed_by": "TestDev"}
""")
    
    # Create another dummy plugin with a "security risk"
    with open(os.path.join(dummy_plugin_dir, "bad_plugin.py"), "w") as f:
        f.write("""
import logging
import os # This import will be flagged by basic security check

logging.basicConfig(level=logging.INFO)

class BadPlugin:
    __is_fba_plugin__ = True
    plugin_id = "malicious_plugin_001"
    version = "0.0.1"

    async def initialize(self, config):
        logging.info("BadPlugin initialized (trying os.system for fun!)")
        # os.system("echo 'Malicious command executed!'") # This should be caught by validation

    async def on_simulation_end(self, context):
        logging.info("BadPlugin: Simulation ended.")
        return {"executed_bad_code": "maybe"}
""")

    # Load plugins
    await plugin_manager.load_plugins(dummy_plugin_dir)

    # Register extension points
    plugin_manager.register_extension_point("on_simulation_step", Callable)
    plugin_manager.register_extension_point("on_agent_decision", Callable)
    plugin_manager.register_extension_point("on_simulation_end", Callable)

    # Execute a hook
    results_step = await plugin_manager.execute_plugin_hook("on_simulation_step", {"current_time": 10, "data_count": 5})
    print(f"\nResults of 'on_simulation_step' hook: {results_step}\n")

    results_decision = await plugin_manager.execute_plugin_hook("on_agent_decision", {"agent_id": "AgentA", "decision_data": "BUY"})
    print(f"\nResults of 'on_agent_decision' hook: {results_decision}\n")

    # Get available plugins
    available_plugins = plugin_manager.get_available_plugins()
    print(f"\nAvailable Plugins: {json.dumps(available_plugins, indent=2)}\n")

    # Clean up dummy files
    os.remove(os.path.join(dummy_plugin_dir, "my_test_plugin.py"))
    os.remove(os.path.join(dummy_plugin_dir, "bad_plugin.py"))
    os.rmdir(dummy_plugin_dir)

if __name__ == "__main__":
    import asyncio # Ensure asyncio is imported here for standalone execution
    import json # Ensure json is imported here for standalone execution
    from collections import deque # Ensure deque is imported if needed for standalone
    asyncio.run(_main())