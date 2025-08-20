import logging
import os
import importlib.util
from typing import Dict, Any, List, Optional
import asyncio
# Added for runtime usage outside __main__ guard
import inspect
import json
import time
from dataclasses import dataclass

# Assuming these base classes and manager exist from previous steps
# from plugins.plugin_framework import PluginManager, PluginError
# from plugins.scenario_plugins.base_scenario_plugin import ScenarioPlugin
# from plugins.agent_plugins.base_agent_plugin import AgentPlugin

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

@dataclass(frozen=True)
class QualityAssessment:
    """
    Summary of contribution quality used by tests and reports.
    Provides a normalized, single object importable from community.contribution_tools.
    """
    code_quality: float          # 0.0 - 1.0
    test_coverage: float         # 0.0 - 1.0
    documentation_score: float   # 0.0 - 1.0
    security_score: float        # 0.0 - 1.0
    performance_score: float     # 0.0 - 1.0
    passed: bool

    @staticmethod
    def from_results(results: Dict[str, Any]) -> "QualityAssessment":
        """
        Build QualityAssessment from heterogeneous validation/benchmark results.
        Missing fields default to 0.0. 'passed' is derived unless explicitly provided.
        """
        cq = float(results.get("code_quality", results.get("lint_score", 0.0)) or 0.0)
        tc = float(results.get("test_coverage", results.get("coverage", 0.0)) or 0.0)
        ds = float(results.get("documentation_score", results.get("docs_score", 0.0)) or 0.0)
        ss = float(results.get("security_score", 1.0 if results.get("security_issues", 0) == 0 else 0.0))
        ps = float(results.get("performance_score", results.get("throughput_score", 0.0)) or 0.0)
        passed = bool(results.get("passed",
                                  (cq >= 0.7 and tc >= 0.8 and ds >= 0.6 and ss >= 0.8 and ps >= 0.6)))
        return QualityAssessment(
            code_quality=cq,
            test_coverage=tc,
            documentation_score=ds,
            security_score=ss,
            performance_score=ps,
            passed=passed,
        )

class ContributionManager:
    """
    Manages automated testing, documentation generation, and performance benchmarking
    for community contributions (plugins). Ensures contributions meet project standards.
    """

    def __init__(self, plugin_manager: Any = None): # Use Any to avoid circular import for now
        self.plugin_manager = plugin_manager
        logging.info("ContributionManager initialized.")

    async def _load_plugin_module(self, plugin_path: str) -> Optional[Any]:
        """
        Loads a plugin module dynamically from a given file path.
        Returns the module object if successful, None otherwise.
        """
        if not os.path.exists(plugin_path):
            logging.error(f"Plugin file not found: {plugin_path}")
            return None
        
        module_name = os.path.basename(plugin_path).replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, plugin_path)
        if spec is None:
            logging.error(f"Could not find module spec for {module_name} at {plugin_path}")
            return None
        
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
            return module
        except Exception as e:
            logging.error(f"Error loading plugin module from '{plugin_path}': {e}", exc_info=True)
            return None

    async def validate_contribution(self, plugin_path: str, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Runs comprehensive validation tests on a community plugin.
        
        :param plugin_path: The file path to the plugin's Python file.
        :param tests: A list of test definitions to run against the plugin.
        :return: A dictionary summarizing the validation results.
        """
        logging.info(f"Starting validation for contribution: {plugin_path}")
        results = {"overall_status": "pending", "tests_run": [], "errors": []}
        
        plugin_module = await self._load_plugin_module(plugin_path)
        if not plugin_module:
            results["overall_status"] = "failed"
            results["errors"].append(f"Failed to load plugin module from {plugin_path}.")
            return results

        # Basic structural checks
        if not hasattr(plugin_module, '__is_fba_plugin__') or not getattr(plugin_module, '__is_fba_plugin__'):
            results["tests_run"].append({"name": "FBA Plugin Marker", "status": "failed", "message": "Missing __is_fba_plugin__ = True marker."})
            results["overall_status"] = "failed"
            logging.error(f"Plugin {plugin_path} is not marked as an FBA plugin.")
            return results
        else:
            results["tests_run"].append({"name": "FBA Plugin Marker", "status": "passed"})

        # Security validation (leveraging PluginManager's security check)
        if self.plugin_manager:
            security_status = await self.plugin_manager.validate_plugin_security(plugin_module)
            if not security_status:
                results["tests_run"].append({"name": "Security Validation", "status": "failed", "message": "Plugin failed security checks."})
                results["overall_status"] = "failed"
                logging.error(f"Plugin {plugin_path} failed security validation.")
                return results
            else:
                results["tests_run"].append({"name": "Security Validation", "status": "passed"})
        else:
            logging.warning("PluginManager not available. Skipping security validation.")
            results["tests_run"].append({"name": "Security Validation", "status": "skipped", "message": "PluginManager not provided."})


        # Instantiate the plugin (find the class marked with __is_fba_plugin__)
        plugin_instance = None
        for name, obj in inspect.getmembers(plugin_module):
            if inspect.isclass(obj) and hasattr(obj, '__is_fba_plugin__') and obj.__is_fba_plugin__:
                try:
                    plugin_instance = obj()
                    results["tests_run"].append({"name": "Plugin Instantiation", "status": "passed"})
                    break
                except Exception as e:
                    results["tests_run"].append({"name": "Plugin Instantiation", "status": "failed", "message": f"Error instantiating plugin: {e}"})
                    results["overall_status"] = "failed"
                    logging.error(f"Failed to instantiate plugin from {plugin_path}: {e}")
                    return results
        
        if not plugin_instance:
            results["overall_status"] = "failed"
            results["errors"].append("No FBA-Bench plugin class found in the module.")
            return results

        # Run specific tests defined in the `tests` list
        for test in tests:
            test_name = test.get("name", "Unnamed Test")
            test_type = test.get("type")
            test_config = test.get("config", {})

            try:
                if test_type == "scenario_validation" and hasattr(plugin_instance, 'validate_scenario_constraints'):
                    is_valid = await plugin_instance.validate_scenario_constraints(test_config.get("scenario_data", {}))
                    results["tests_run"].append({"name": test_name, "status": "passed" if is_valid else "failed"})
                elif test_type == "agent_decision_test" and hasattr(plugin_instance, 'decide_action'):
                    # Mock state and context for decision test
                    mock_state = {"time": 1, "inventory": 50, "price": 10.0, "demand": 30}
                    mock_context = {"episode_id": "test_episode_1"}
                    action = await plugin_instance.decide_action(mock_state, mock_context)
                    if not action or not isinstance(action, dict):
                        raise ValueError("Agent decision method did not return a valid action dictionary.")
                    results["tests_run"].append({"name": test_name, "status": "passed", "action": action})
                # Add more test types here as needed
                else:
                    results["tests_run"].append({"name": test_name, "status": "skipped", "message": f"Unsupported test type '{test_type}' or method not implemented."})
            except Exception as e:
                results["tests_run"].append({"name": test_name, "status": "error", "message": str(e)})
                results["errors"].append(f"Error in test '{test_name}': {str(e)}")
                logging.error(f"Error executing test '{test_name}' for plugin {plugin_path}: {e}", exc_info=True)

        results["overall_status"] = "succeeded" if not results["errors"] and all(t["status"] != "failed" and t["status"] != "error" for t in results["tests_run"]) else "failed"
        logging.info(f"Validation finished for {plugin_path}. Status: {results['overall_status']}")
        return results

    async def generate_plugin_docs(self, plugin_module: Any) -> Dict[str, str]:
        """
        Auto-generates documentation for a community plugin based on its methods and properties.
        
        :param plugin_module: The loaded plugin module object.
        :return: A dictionary containing generated documentation strings (e.g., {"README.md": "..."}).
        """
        logging.info(f"Generating documentation for plugin: {plugin_module.__name__}")
        docs_content = {}
        
        # Try to find the main plugin class within the module
        plugin_class = None
        for name, obj in inspect.getmembers(plugin_module):
            if inspect.isclass(obj) and hasattr(obj, '__is_fba_plugin__') and obj.__is_fba_plugin__:
                plugin_class = obj
                break
        
        if not plugin_class:
            logging.error(f"No FBA-Bench plugin class found in module {plugin_module.__name__}. Cannot generate docs.")
            return docs_content

        # Instantiate to access class properties and potentially methods like get_documentation_template
        plugin_instance = None
        try:
            plugin_instance = plugin_class()
        except Exception as e:
            logging.error(f"Could not instantiate plugin class {plugin_class.__name__} for docs generation: {e}")
            # Still try to extract info from class attributes if instance failed
            pass

        readme_content = f"# {getattr(plugin_class, 'name', 'Community Plugin')} Documentation\n\n"
        readme_content += f"**ID:** `{getattr(plugin_class, 'plugin_id', 'N/A')}`\n"
        readme_content += f"**Version:** `{getattr(plugin_class, 'version', 'N/A')}`\n\n"
        readme_content += f"**Description:** {getattr(plugin_class, 'description', 'No description provided.')}\n\n"

        # If plugin instance has a specific documentation template method, use it
        if plugin_instance and hasattr(plugin_instance, 'get_documentation_template') and callable(plugin_instance.get_documentation_template):
            try:
                template = plugin_instance.get_documentation_template()
                readme_content += "\n## Detailed Documentation (from plugin template)\n" + template
            except Exception as e:
                logging.warning(f"Error getting plugin-specific documentation template: {e}. Falling back to generic doc generation.")

        else:
            readme_content += "\n## Key Methods\n"
            # Inspect methods and add basic docstrings
            for name, method in inspect.getmembers(plugin_class, inspect.isfunction):
                if not name.startswith("_") and inspect.iscoroutinefunction(method) or inspect.isfunction(method):
                    signature = inspect.signature(method)
                    doc = inspect.getdoc(method)
                    readme_content += f"### `{name}{signature}`\n"
                    readme_content += f"{doc if doc else 'No documentation.'}\n\n"

        docs_content["README.md"] = readme_content
        logging.info(f"Generated README.md for {plugin_module.__name__}.")
        return docs_content

    async def benchmark_plugin_performance(self, plugin_path: str, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Tests plugin performance against standard scenarios and provides benchmarks.
        
        :param plugin_path: The file path to the plugin's Python file.
        :param scenarios: A list of scenario configurations to run benchmarks against.
        :return: A dictionary summarizing performance metrics.
        """
        logging.info(f"Starting performance benchmarking for plugin: {plugin_path}")
        results = {"overall_metrics": {}, "scenario_results": []}

        plugin_module = await self._load_plugin_module(plugin_path)
        if not plugin_module:
            results["overall_metrics"]["status"] = "failed_load"
            return results
        
        plugin_instance = None
        for name, obj in inspect.getmembers(plugin_module):
            if inspect.isclass(obj) and hasattr(obj, '__is_fba_plugin__') and obj.__is_fba_plugin__:
                try:
                    plugin_instance = obj()
                    break
                except Exception as e:
                    logging.error(f"Failed to instantiate plugin for benchmarking: {e}")
                    results["overall_metrics"]["status"] = "failed_instantiation"
                    return results

        if not plugin_instance:
            logging.error(f"No FBA-Bench plugin found in {plugin_path} for benchmarking.")
            results["overall_metrics"]["status"] = "no_plugin_found"
            return results
        
        # Assume a mock simulator or actual simulation runner
        # For this example, we'll just simulate running a scenario.
        
        total_profit = 0.0
        total_episodes = 0
        total_time_taken = 0.0

        for scenario_config in scenarios:
            scenario_name = scenario_config.get("name", "Unnamed Scenario")
            logging.info(f"Running benchmark for scenario: {scenario_name}")
            scenario_metrics = {"scenario_name": scenario_name}
            
            start_time = time.monotonic()
            
            # Simulate an episode or multiple episodes
            # If it's an agent, call decide_action multiple times.
            # If it's a scenario, generate_initial_state and inject_events.
            
            # Placeholder simulation loop (very simplified)
            mock_state = {"time": 0, "inventory": 100, "price": 10.0, "demand": 50, "cash": 1000.0}
            episode_profit = 0.0
            num_steps = 10 # Simulate 10 steps
            
            for step in range(num_steps):
                if hasattr(plugin_instance, 'decide_action'): # It's an agent plugin
                    action_decision_time_start = time.monotonic()
                    action = await plugin_instance.decide_action(mock_state, {"step": step, "scenario_config": scenario_config})
                    action_decision_time_end = time.monotonic()
                    
                    # Update mock state based on action
                    if action.get("type") == "set_price":
                        mock_state["price"] = action.get("value", mock_state["price"])
                    elif action.get("type") == "adjust_inventory":
                        mock_state["inventory"] += action.get("value", 0)

                    # Simulate some profit
                    profit_this_step = mock_state["price"] * (mock_state["demand"] * 0.5) # Dummy calculation
                    episode_profit += profit_this_step
                    mock_state["cash"] += profit_this_step
                    
                    total_time_taken += (action_decision_time_end - action_decision_time_start)

                elif hasattr(plugin_instance, 'inject_events'): # It's a scenario plugin
                    events = await plugin_instance.inject_events(step)
                    # Simulate processing of events if needed
                    # For benchmarking a scenario, we might primarily check event generation speed or complexity.
                    
                    total_time_taken += 0.001 # Small time for event injection

                await asyncio.sleep(0.005) # Simulate a small time step

            scenario_metrics["profit"] = episode_profit
            scenario_metrics["duration_seconds"] = total_time_taken
            results["scenario_results"].append(scenario_metrics)
            
            total_profit += episode_profit
            total_episodes += 1 # In this simplified loop, each scenario is one "episode"
        
        if total_episodes > 0:
            results["overall_metrics"]["average_profit_per_episode"] = total_profit / total_episodes
            results["overall_metrics"]["total_benchmark_time_seconds"] = sum(s["duration_seconds"] for s in results["scenario_results"])
            if hasattr(plugin_instance, 'get_performance_benchmarks'):
                plugin_provided_metrics = plugin_instance.get_performance_benchmarks()
                results["overall_metrics"].update(plugin_provided_metrics)
        
        results["overall_metrics"]["status"] = "completed"
        logging.info(f"Finished performance benchmarking for {plugin_path}.")
        return results

    async def package_for_distribution(self, plugin_path: str, metadata: Dict[str, Any]) -> str:
        """
        Prepares a plugin for distribution, e.g., zipping it with metadata and documentation.
        
        :param plugin_path: The file path to the plugin's Python file.
        :param metadata: Additional metadata for packaging (e.g., author, dependencies).
        :return: The path to the created distribution package.
        """
        logging.info(f"Packaging plugin {plugin_path} for distribution.")
        package_dir = "dist"
        os.makedirs(package_dir, exist_ok=True)
        
        plugin_name = os.path.basename(plugin_path).replace(".py", "")
        version = metadata.get("version", "0.0.1")
        package_filename = os.path.join(package_dir, f"{plugin_name}-v{version}.zip")

        import zipfile
        with zipfile.ZipFile(package_filename, 'w') as zipf:
            zipf.write(plugin_path, arcname=os.path.basename(plugin_path)) # Add the plugin file

            # Generate and add docs
            plugin_module = await self._load_plugin_module(plugin_path)
            if plugin_module:
                docs = await self.generate_plugin_docs(plugin_module)
                for doc_name, content in docs.items():
                    doc_path_in_zip = os.path.join(plugin_name, doc_name)
                    zipf.writestr(doc_path_in_zip, content)
                    logging.info(f"Added doc '{doc_name}' to package.")
            
            # Add metadata file
            metadata_filename = os.path.join(plugin_name, "metadata.json")
            zipf.writestr(metadata_filename, json.dumps(metadata, indent=4))
            logging.info("Added metadata.json to package.")

        logging.info(f"Plugin packaged to: {package_filename}")
        return package_filename

    async def create_contribution_report(self, validation_results: Dict[str, Any], benchmark_results: Dict[str, Any]) -> str:
        """
        Summarizes plugin quality based on validation and benchmarking results.
        
        :param validation_results: Results from `validate_contribution`.
        :param benchmark_results: Results from `benchmark_plugin_performance`.
        :return: A markdown formatted string of the report.
        """
        logging.info("Creating contribution report.")
        report = "# Community Plugin Contribution Report\n\n"
        report += "## 1. Validation Results\n"
        report += f"**Overall Status:** `{validation_results.get('overall_status', 'N/A')}`\n\n"
        
        report += "| Test Name | Status | Message/Details |\n"
        report += "|-----------|--------|-----------------|\n"
        for test in validation_results.get("tests_run", []):
            report += f"| {test.get('name')} | `{test.get('status')}` | {test.get('message', '')} |\n"
        
        if validation_results.get("errors"):
            report += "\n### Validation Errors:\n"
            for error in validation_results["errors"]:
                report += f"- {error}\n"
        report += "\n"

        report += "## 2. Performance Benchmarking Results\n"
        report += f"**Overall Status:** `{benchmark_results.get('overall_metrics', {}).get('status', 'N/A')}`\n"
        
        overall_metrics = benchmark_results.get("overall_metrics", {})
        report += "**Average Profit per Episode (Simulated):** " + \
                  f"{overall_metrics.get('average_profit_per_episode', 'N/A'):.2f}\n" + \
                  "**Total Benchmark Time (Simulated):** " + \
                  f"{overall_metrics.get('total_benchmark_time_seconds', 'N/A'):.2f} seconds\n"
        
        if overall_metrics.get("inventory_turnover_rate"):
             report += f"**Inventory Turnover Rate:** {overall_metrics['inventory_turnover_rate']:.2f}\n"

        if overall_metrics.get("decision_latency_ms"):
            report += f"**Decision Latency:** {overall_metrics['decision_latency_ms']:.2f} ms\n"

        if benchmark_results.get("scenario_results"):
            report += "\n### Per-Scenario Results:\n"
            report += "| Scenario | Profit | Duration (s) |\n"
            report += "|----------|--------|--------------|\n"
            for scenario in benchmark_results["scenario_results"]:
                report += f"| {scenario.get('scenario_name')} | {scenario.get('profit', 'N/A'):.2f} | {scenario.get('duration_seconds', 'N/A'):.2f} |\n"
        
        report += "\n\nThis report summarizes the automated checks for the community plugin."
        logging.info("Contribution report created.")
        return report

# Example usage (for testing purposes outside the main framework flow)
import inspect # Already imported

async def _main():
    # Placeholder for `PluginManager` instance
    # In a real setup, you would pass an actual PluginManager
    class MockPluginManager:
        async def validate_plugin_security(self, plugin_module: Any) -> bool:
            logging.info(f"MockPluginManager: Validating security for {plugin_module.__name__}")
            # Simulate security check pass/fail
            if "bad_plugin" in plugin_module.__name__:
                logging.error("Mock security check failed for 'bad_plugin'.")
                return False
            return True

    mock_pm = MockPluginManager()
    manager = ContributionManager(plugin_manager=mock_pm)

    # Create a dummy plugin file for testing
    dummy_plugin_dir = "temp_community_plugins"
    os.makedirs(dummy_plugin_dir, exist_ok=True)
    
    plugin_file_path = os.path.join(dummy_plugin_dir, "my_scenario_plugin_v1.py")
    with open(plugin_file_path, "w") as f:
        f.write("""
import logging
from typing import Dict, Any, List, Protocol

logging.basicConfig(level=logging.INFO)

class ScenarioPlugin:
    __is_fba_plugin__ = True
    plugin_id = "test_scenario_plugin"
    version = "1.0.0"
    name = "Test Scenario Plugin"
    description = "A dummy scenario plugin for testing purposes."
    scenario_type = "test_type"

    def initialize(self, config: Dict[str, Any]):
        logging.info(f"TestScenarioPlugin initialized: {config}")

    async def generate_initial_state(self) -> Dict[str, Any]:
        return {"test_val": 100}

    async def inject_events(self, current_time: int) -> List[Dict[str, Any]]:
        if current_time == 5:
            return [{"type": "test_event", "data": "triggered"}]
        return []

    async def validate_scenario_constraints(self, scenario_data: Dict[str, Any]) -> bool:
        return scenario_data.get("valid", True) # Customizable validation for testing

    def get_documentation_template(self) -> str:
        return "This is a test doc template for my scenario."
    
    def get_plugin_info(self) -> Dict[str, Any]:
        return {"id": self.plugin_id, "name": self.name}
""")

    # Create a dummy agent plugin for testing
    agent_plugin_file_path = os.path.join(dummy_plugin_dir, "my_agent_plugin_v1.py")
    with open(agent_plugin_file_path, "w") as f:
        f.write("""
import logging
from typing import Dict, Any, List, Protocol

logging.basicConfig(level=logging.INFO)

class AgentPlugin:
    __is_fba_plugin__ = True
    plugin_id = "test_agent_plugin"
    version = "1.0.0"
    name = "Test Agent Plugin"
    description = "A dummy agent plugin for testing purposes."
    agent_type = "test_type"

    def initialize(self, config: Dict[str, Any]):
        logging.info(f"TestAgentPlugin initialized: {config}")

    async def decide_action(self, current_state: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        return {"action": "buy", "amount": current_state.get("inventory", 0) + 10}

    async def learn_from_experience(self, episode_experience: Dict[str, Any]):
        logging.info(f"TestAgentPlugin learning from: {episode_experience.keys()}")

    def get_plugin_info(self) -> Dict[str, Any]:
        return {"id": self.plugin_id, "name": self.name}
    
    def get_performance_benchmarks(self) -> Dict[str, Any]:
        return {"test_metric": 0.95}
""")

    # Test validation for scenario plugin
    validation_tests_scenario = [
        {"name": "Basic Scenario Validations", "type": "scenario_validation", "config": {"scenario_data": {"initial_cash": 100, "valid": True}}},
        {"name": "Invalid Scenario Data", "type": "scenario_validation", "config": {"scenario_data": {"initial_cash": -10, "valid": False}}}
    ]
    scenario_validation_results = await manager.validate_contribution(plugin_file_path, validation_tests_scenario)
    print("\n--- Scenario Plugin Validation Results ---")
    print(json.dumps(scenario_validation_results, indent=2))

    # Test validation for agent plugin
    validation_tests_agent = [
        {"name": "Agent Decision Logic", "type": "agent_decision_test", "config": {}}
    ]
    agent_validation_results = await manager.validate_contribution(agent_plugin_file_path, validation_tests_agent)
    print("\n--- Agent Plugin Validation Results ---")
    print(json.dumps(agent_validation_results, indent=2))

    # Test documentation generation
    print("\n--- Documentation Generation (Scenario Plugin) ---")
    scenario_module = await manager._load_plugin_module(plugin_file_path)
    if scenario_module:
        scenario_docs = await manager.generate_plugin_docs(scenario_module)
        print(scenario_docs.get("README.md"))
    
    print("\n--- Documentation Generation (Agent Plugin) ---")
    agent_module = await manager._load_plugin_module(agent_plugin_file_path)
    if agent_module:
        agent_docs = await manager.generate_plugin_docs(agent_module)
        print(agent_docs.get("README.md"))

    # Test benchmarking
    benchmark_scenarios = [
        {"name": "Default", "iterations": 10},
        {"name": "High Volatility", "iterations": 5}
    ]
    benchmark_results = await manager.benchmark_plugin_performance(agent_plugin_file_path, benchmark_scenarios)
    print("\n--- Benchmarking Results (Agent Plugin) ---")
    print(json.dumps(benchmark_results, indent=2))

    # Test packaging
    package_metadata = {"author": "Community Dev", "version": "1.0.0", "description": "My awesome plugin."}
    package_path = await manager.package_for_distribution(plugin_file_path, package_metadata)
    print(f"\nPlugin package created at: {package_path}")

    # Test report generation
    report = await manager.create_contribution_report(scenario_validation_results, benchmark_results)
    print("\n--- Contribution Report ---")
    print(report)

    # Clean up dummy files
    os.remove(plugin_file_path)
    os.remove(agent_plugin_file_path)
    import shutil
    if os.path.exists("./dist"): shutil.rmtree("./dist")
    if os.path.exists(dummy_plugin_dir): shutil.rmtree(dummy_plugin_dir)


if __name__ == "__main__":
    import json # Already imported in main part of file
    import inspect # Already imported in main part of file
    asyncio.run(_main())