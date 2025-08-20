import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple
import asyncio
import time

# Assuming RealWorldAdapter is implemented
# from .real_world_adapter import RealWorldAdapter

@dataclass
class ValidationResult:
    """Standard validation result type expected by tests and examples."""
    passed: bool
    score: float
    issues: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class IntegrationValidator:
    """
    Provides tools for integration testing and validation between simulation and real-world
    systems, ensuring action consistency, safety, and performance parity.
    """

    def __init__(self, real_world_adapter: Any): # Use Any to avoid circular import for now
        self.real_world_adapter = real_world_adapter
        self.validation_results: Dict[str, Any] = {}
        logging.info("IntegrationValidator initialized.")
        

    async def validate_action_consistency(self, sim_action: Dict[str, Any], real_expected_action: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Checks if a simulation action, when translated, matches an expected real-world action.
        This ensures 'Simulation-to-real validation'.
        
        :param sim_action: The action as it would be in the simulation.
        :param real_expected_action: The expected translated real-world action.
        :return: A tuple of (bool, str) indicating success/failure and a message.
        """
        logging.info(f"Validating action consistency for sim_action: {sim_action.get('type')}")
        try:
            translated_action = await self.real_world_adapter.translate_simulation_action(sim_action)
            
            # Simple content comparison. In practice, this might need more sophisticated diffing
            # considering order of keys, floating point precision, etc.
            is_consistent = (translated_action == real_expected_action)
            
            message = f"Translated action: {translated_action}, Expected: {real_expected_action}"
            if is_consistent:
                logging.info(f"Action consistency PASSED: {sim_action.get('type')}")
            else:
                logging.warning(f"Action consistency FAILED for {sim_action.get('type')}. {message}")
            
            return is_consistent, message
        except Exception as e:
            logging.error(f"Error validating action consistency for {sim_action.get('type')}: {e}", exc_info=True)
            return False, f"Error during translation or comparison: {e}"

    async def test_safety_constraints(self, dangerous_actions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validates that safety mechanisms prevent specified dangerous actions in live/sandbox modes.
        This provides 'Safety testing'.
        
        :param dangerous_actions: A list of actions that are expected to be blocked by safety constraints.
        :return: A dictionary summarizing the safety test results for each action.
        """
        logging.info("Starting safety constraint testing.")
        results = {"overall_status": "pending", "tests": []}
        
        # Test in sandbox mode first
        await self.real_world_adapter.set_mode("sandbox")
        logging.info("Testing safety constraints in SANDBOX mode.")
        for i, action in enumerate(dangerous_actions):
            test_name = f"Sandbox Safety Test for Action {i+1} ({action.get('type', 'Unknown A.')})"
            try:
                # Attempt to execute the dangerous action with safety_check=True
                await self.real_world_adapter.execute_action(action, safety_check=True)
                # If we reach here, it means the action *wasn't* blocked, which is a failure
                results["tests"].append({"name": test_name, "action": action, "status": "FAILED", "message": "Action was not blocked by safety constraints."})
                logging.error(f"Safety test FAILED: {test_name}. Action was executed.")
            except ValueError as e: # Assuming RealWorldAdapter raises ValueError for blocked unsafe actions
                results["tests"].append({"name": test_name, "action": action, "status": "PASSED", "message": f"Action successfully blocked: {e}"})
                logging.info(f"Safety test PASSED: {test_name}. Action blocked as expected.")
            except Exception as e:
                results["tests"].append({"name": test_name, "action": action, "status": "ERROR", "message": f"Unexpected error during test: {e}"})
                logging.error(f"Safety test ERROR: {test_name}. Unexpected error: {e}", exc_info=True)

        results["overall_status"] = "PASSED" if all(t["status"] == "PASSED" for t in results["tests"]) else "FAILED"
        logging.info(f"Safety constraint testing finished with overall status: {results['overall_status']}")
        return results

    async def compare_performance_metrics(self, sim_results: Dict[str, Any], real_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compares performance metrics between simulation and real-world results.
        This identifies 'Performance comparison' discrepancies.
        
        :param sim_results: Performance metrics from a simulation run.
        :param real_results: Performance metrics from a corresponding real-world run.
        :return: A dictionary of comparison results, highlighting differences.
        """
        logging.info("Starting performance metrics comparison.")
        comparison = {"discrepancies": {}, "overall_status": "MATCH"}
        
        # Example comparison: check key metrics like revenue, profit, inventory
        metrics_to_compare = ["total_revenue", "net_profit", "average_inventory_level", "order_fulfillment_rate"]

        for metric in metrics_to_compare:
            sim_val = sim_results.get(metric)
            real_val = real_results.get(metric)

            if sim_val is None or real_val is None:
                logging.warning(f"Skipping comparison for metric '{metric}': missing data in one or both result sets.")
                continue

            # Using a simple percentage difference for comparison
            if sim_val != 0:
                diff_percentage = abs((real_val - sim_val) / sim_val) * 100
            else:
                diff_percentage = 0 if real_val == 0 else float('inf') # Handle division by zero

            threshold = 5.0 # Example: allow up to 5% difference
            if diff_percentage > threshold:
                comparison["discrepancies"][metric] = {
                    "simulation": sim_val,
                    "real_world": real_val,
                    "difference_percent": f"{diff_percentage:.2f}%",
                    "status": "SIGNIFICANT_DIFFERENCE"
                }
                comparison["overall_status"] = "DISCREPANCY"
                logging.warning(f"Significant difference for {metric}: {diff_percentage:.2f}%")
            else:
                logging.info(f"Metric {metric} matches closely ({diff_percentage:.2f}% difference).")
        
        logging.info(f"Performance comparison finished. Overall status: {comparison['overall_status']}")
        return comparison

    async def run_integration_test_suite(self) -> Dict[str, Any]:
        """
        Runs a comprehensive suite of integration validation tests.
        This encompasses 'Regression testing' to ensure new integrations don't break existing functionality.
        """
        logging.info("Running comprehensive integration test suite.")
        suite_results = {
            "action_consistency_tests": {},
            "safety_tests": {},
            "performance_comparison_tests": {},
            "overall_suite_status": "PENDING"
        }

        # Define mock data for tests
        # Example for action consistency
        sim_action_price_set = {"type": "set_price", "value": 25.0}
        real_expected_price_update = {"api_call": "update_product_price", "parameters": {"product_sku": "FBA-SKU-123", "new_price": 25.0}}
        
        consistency_status, consistency_message = await self.validate_action_consistency(sim_action_price_set, real_expected_price_update)
        suite_results["action_consistency_tests"] = {
            "status": "PASSED" if consistency_status else "FAILED",
            "message": consistency_message
        }
        
        # Example for safety testing
        dangerous_actions_list = [
            {"type": "set_price", "value": 10000.0}, # Too high price
            {"type": "adjust_inventory", "value": -10000} # Too large negative inventory adjustment
        ]
        safety_test_results = await self.test_safety_constraints(dangerous_actions_list)
        suite_results["safety_tests"] = safety_test_results

        # Example for performance comparison (mock data)
        mock_sim_perf = {"total_revenue": 10000.0, "net_profit": 2000.0, "average_inventory_level": 80, "order_fulfillment_rate": 0.98}
        # Simulate a slight deviation for real-world
        mock_real_perf = {"total_revenue": 10200.0, "net_profit": 1950.0, "average_inventory_level": 82, "order_fulfillment_rate": 0.97}

        await self.real_world_adapter.set_mode("live") # Perform perf comparison in live mode context
        perf_comparison_results = await self.compare_performance_metrics(mock_sim_perf, mock_real_perf)
        suite_results["performance_comparison_tests"] = perf_comparison_results

        # Determine overall suite status
        all_passed = (suite_results["action_consistency_tests"]["status"] == "PASSED" and
                      suite_results["safety_tests"]["overall_status"] == "PASSED" and
                      suite_results["performance_comparison_tests"]["overall_status"] == "MATCH")
        
        suite_results["overall_suite_status"] = "PASSED" if all_passed else "FAILED"
        self.validation_results = suite_results # Store for report generation
        logging.info(f"Integration test suite completed. Overall status: {suite_results['overall_suite_status']}")
        return suite_results

    async def generate_integration_report(self) -> str:
        """
        Summarizes the results of the comprehensive integration validation.
        
        :return: A markdown formatted string of the integration report.
        """
        logging.info("Generating integration report.")
        report = "# FBA-Bench Integration Validation Report\n\n"
        
        results = self.validation_results
        if not results:
            report += "No integration test suite has been run yet. Please run `run_integration_test_suite()` first.\n"
            return report

        report += f"**Overall Integration Suite Status:** `{results.get('overall_suite_status', 'N/A')}`\n\n"

        # Action Consistency
        report += "## 1. Action Consistency Validation\n"
        consistency = results.get("action_consistency_tests", {})
        report += f"Status: `{consistency.get('status', 'N/A')}`\n"
        report += f"Details: {consistency.get('message', 'N/A')}\n\n"

        # Safety Tests
        report += "## 2. Safety Constraint Testing\n"
        safety = results.get("safety_tests", {})
        report += f"Overall Safety Test Status: `{safety.get('overall_status', 'N/A')}`\n\n"
        report += "| Test Name | Action Type | Status | Message |\n"
        report += "|-----------|-------------|--------|---------|\n"
        for test in safety.get("tests", []):
            action_type = test.get("action", {}).get("type", "N/A")
            report += f"| {test.get('name')} | {action_type} | `{test.get('status')}` | {test.get('message', '')} |\n"
        report += "\n"

        # Performance Comparison
        report += "## 3. Performance Comparison (Simulation vs. Real-World)\n"
        performance = results.get("performance_comparison_tests", {})
        report += f"Overall Comparison Status: `{performance.get('overall_status', 'N/A')}`\n"
        if performance.get("discrepancies"):
            report += "**Identified Discrepancies:**\n"
            report += "| Metric | Simulation | Real-World | Difference (%) | Status |\n"
            report += "|--------|------------|------------|----------------|--------|\n"
            for metric, data in performance["discrepancies"].items():
                report += f"| {metric} | {data.get('simulation')} | {data.get('real_world')} | {data.get('difference_percent')} | `{data.get('status')}` |\n"
        else:
            report += "No significant performance discrepancies detected within set thresholds.\n"
        report += "\n"

        report += "---\nGenerated by FBA-Bench Integration Validator."
        logging.info("Integration report generated successfully.")
        return report

# Example usage for testing
async def _main():
    # Mock RealWorldAdapter for testing IntegrationValidator
    class MockRealWorldAdapter:
        def __init__(self):
            self.mode = "simulation"

        async def set_mode(self, mode: str):
            self.mode = mode
            logging.info(f"Mock RealWorldAdapter switched to {self.mode} mode.")

        async def translate_simulation_action(self, sim_action: Dict[str, Any]) -> Dict[str, Any]:
            # Simple mock translation
            if sim_action.get("type") == "set_price":
                return {"api_call": "update_product_price", "parameters": {"product_sku": "FBA-SKU-123", "new_price": sim_action.get("value")}}
            return sim_action

        async def execute_action(self, action: Dict[str, Any], safety_check=True) -> Dict[str, Any]:
            if self.mode == "live" and safety_check and not await self.validate_real_world_safety(action):
                raise ValueError("Action blocked by safety constraint.")
            return {"status": "success", "mode": self.mode, "action_executed": action}

        async def validate_real_world_safety(self, action: Dict[str, Any]) -> bool:
            if action.get("api_call") == "update_product_price" and action.get("parameters", {}).get("new_price") > 1000:
                return False # Example: Price too high
            if action.get("api_call") == "adjust_inventory_level" and action.get("parameters", {}).get("quantity_change") < -100:
                return False # Example: Excessive negative inventory adjustment
            return True

    mock_adapter = MockRealWorldAdapter()
    validator = IntegrationValidator(real_world_adapter=mock_adapter)

    # Test validate_action_consistency
    sim_action = {"type": "set_price", "value": 99.99}
    expected_real_action = {"api_call": "update_product_price", "parameters": {"product_sku": "FBA-SKU-123", "new_price": 99.99}}
    consistent, message = await validator.validate_action_consistency(sim_action, expected_real_action)
    print(f"\nAction Consistency Test: {consistent} - {message}")

    # Test test_safety_constraints
    dangerous_actions = [
        {"type": "set_price", "value": 1500.0},
        {"type": "adjust_inventory", "value": -200}
    ]
    safety_results = await validator.test_safety_constraints(dangerous_actions)
    print("\nSafety Test Results:")
    print(safety_results)

    # Test run_integration_test_suite
    suite_results = await validator.run_integration_test_suite()
    print("\nComprehensive Integration Test Suite Results:")
    print(suite_results)
    
    # Generate report
    report = await validator.generate_integration_report()
    print("\n--- Integration Report ---")
    print(report)

if __name__ == "__main__":
    asyncio.run(_main())