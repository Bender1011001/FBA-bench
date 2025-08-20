"""
Automated Test Orchestration System for FBA-Bench

Coordinates all test categories with dependency management, allowing for
flexible execution modes and integrated reporting.
"""

import asyncio
import logging
import pytest
import time
import json
import tempfile
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os

from enum import Enum
# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Import individual test suites
from tests.integration.test_comprehensive_integration import ComprehensiveIntegrationTests
from tests.benchmarks.performance_benchmarks import PerformanceBenchmarkSuite
from tests.validation.functional_validation import FunctionalValidationSuite
from tests.curriculum.scenario_curriculum_tests import ScenarioAndCurriculumTestSuite
from tests.community.extensibility_tests import CommunityAndExtensibilityTestSuite
from tests.integration.real_world_integration_tests import RealWorldIntegrationTestSuite
from tests.regression.regression_tests import RegressionTestSuite

logger = logging.getLogger(__name__)


class TestCategory(str, Enum):
    """Categorization of test suites."""
    INTEGRATION = "integration"
    PERFORMANCE = "performance"
    FUNCTIONAL = "functional"
    SCENARIO_CURRICULUM = "scenario_curriculum"
    EXTENSIBILITY = "extensibility"
    REAL_WORLD_INTEGRATION = "real_world_integration"
    REGRESSION = "regression"


class ExecutionMode(str, Enum):
    """Modes for running the test orchestration."""
    FULL_SUITE = "full_suite"
    CRITICAL_PATH = "critical_path"
    CATEGORY = "category"
    SPECIFIC_TEST = "specific_test"


@dataclass
class OrchestrationConfig:
    """Configuration for a test orchestration run."""
    mode: ExecutionMode = ExecutionMode.FULL_SUITE
    selected_categories: Optional[List[TestCategory]] = None
    selected_tests: Optional[List[str]] = None
    stop_on_failure: bool = True
    generate_report: bool = True
    output_dir: str = "test_reports"
    clear_output_dir: bool = False


@dataclass
class OrchestrationResult:
    """Overall results from the test orchestration."""
    start_time: str
    end_time: str
    duration_seconds: float
    total_suites_run: int
    suites_passed: int
    suites_failed: int
    overall_success: bool
    detailed_results: Dict[TestCategory, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


class TestOrchestrationSystem:
    """
    Automated system for orchestrating, executing, and reporting on FBA-Bench tests.
    """
    
    def __init__(self):
        self.test_suites: Dict[TestCategory, Any] = {
            TestCategory.INTEGRATION: ComprehensiveIntegrationTests(),
            TestCategory.PERFORMANCE: PerformanceBenchmarkSuite(),
            TestCategory.FUNCTIONAL: FunctionalValidationSuite(),
            TestCategory.SCENARIO_CURRICULUM: ScenarioAndCurriculumTestSuite(),
            TestCategory.EXTENSIBILITY: CommunityAndExtensibilityTestSuite(),
            TestCategory.REAL_WORLD_INTEGRATION: RealWorldIntegrationTestSuite(),
            TestCategory.REGRESSION: RegressionTestSuite()
        }
        self.inter_suite_dependencies: Dict[TestCategory, List[TestCategory]] = {
            # Functional validation should pass before performance benchmarks or scenario tests
            TestCategory.PERFORMANCE: [TestCategory.FUNCTIONAL],
            TestCategory.SCENARIO_CURRICULUM: [TestCategory.FUNCTIONAL],
            TestCategory.REAL_WORLD_INTEGRATION: [TestCategory.FUNCTIONAL],
            # Regression tests might depend on functional/integration stability
            TestCategory.REGRESSION: [TestCategory.FUNCTIONAL, TestCategory.INTEGRATION],
            # Extensibility tests might depend on core functionality
            TestCategory.EXTENSIBILITY: [TestCategory.FUNCTIONAL]
        }
        self.orchestration_config: Optional[OrchestrationConfig] = None
        self.output_dir = Path("test_reports") # Default output directory
        
    async def _resolve_dependencies(self, categories_to_run: List[TestCategory]) -> List[TestCategory]:
        """Resolves the order of test categories based on dependencies."""
        resolved_order = []
        satisfied_categories = set()
        
        # Add a dummy `self` to the beginning of the path to resolve imports correctly
        # This is strictly for the dummy run and can be removed/adjusted based on actual project structure
        initial_path = sys.path[0]
        # sys.path.insert(0, os.getcwd()) 

        # Keep track of remaining categories and their dependencies
        remaining_categories = {
            cat: set(self.inter_suite_dependencies.get(cat, [])) 
            for cat in categories_to_run
        }
        
        while remaining_categories:
            # Find categories with no unsatisfied dependencies
            runnable_now = [
                cat for cat, deps in remaining_categories.items() 
                if not any(d not in satisfied_categories for d in deps)
            ]
            
            if not runnable_now:
                # Circular dependency or unsatisfied dependency
                logger.error("Circular or unsatisfied dependencies detected after trying to resolve all dependencies. Cannot determine a valid test order.")
                unresolved = ", ".join(remaining_categories.keys())
                raise ValueError(f"Could not resolve test dependencies. Unresolved: {unresolved}")
            
            # Add runnable categories to the resolved order and mark dependencies as satisfied
            for cat in runnable_now:
                resolved_order.append(cat)
                satisfied_categories.add(cat)
                del remaining_categories[cat]
        
        # Restore sys.path if it was modified
        # sys.path.pop(0) 

        return resolved_order
            
    async def run_suite(self, config: OrchestrationConfig) -> OrchestrationResult:
        """Runs the complete test suite based on the provided configuration."""
        self.orchestration_config = config
        
        start_time = datetime.now()
        logger.info(f"Starting FBA-Bench Test Orchestration at {start_time.isoformat()}")
        
        if self.orchestration_config.clear_output_dir and self.output_dir.exists():
            shutil.rmtree(self.output_dir)
            logger.info(f"Cleared output directory: {self.output_dir}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        suites_to_run: List[TestCategory] = []
        if config.mode == ExecutionMode.FULL_SUITE:
            suites_to_run = list(TestCategory)
        elif config.mode == ExecutionMode.CRITICAL_PATH:
            # Define critical path manually or dynamically
            suites_to_run = [
                TestCategory.FUNCTIONAL, 
                TestCategory.INTEGRATION, 
                TestCategory.REGRESSION
            ]
        elif config.mode == ExecutionMode.CATEGORY and config.selected_categories:
            suites_to_run = config.selected_categories
        elif config.mode == ExecutionMode.SPECIFIC_TEST and config.selected_tests:
            # This mode would require more granular logic to run specific methods within suites,
            # which is beyond the scope of this high-level orchestrator.
            # For now, we'll treat it as running the entire category if a test name is provided.
            logger.warning("Specific test execution mode not fully implemented. Running relevant categories.")
            detected_categories = set()
            for test_name in config.selected_tests:
                # Basic mapping from test name to category (very simplified)
                if "integration" in test_name: detected_categories.add(TestCategory.INTEGRATION)
                if "performance" in test_name: detected_categories.add(TestCategory.PERFORMANCE)
                if "functional" in test_name: detected_categories.add(TestCategory.FUNCTIONAL)
                if "scenario" in test_name or "curriculum" in test_name: detected_categories.add(TestCategory.SCENARIO_CURRICULUM)
                if "extensibility" in test_name or "community" in test_name: detected_categories.add(TestCategory.EXTENSIBILITY)
                if "real_world" in test_name: detected_categories.add(TestCategory.REAL_WORLD_INTEGRATION)
                if "regression" in test_name: detected_categories.add(TestCategory.REGRESSION)
            suites_to_run = list(detected_categories)
            if not suites_to_run:
                logger.error("No categories derived from specific tests. Aborting.")
                return self._create_early_exit_result(start_time, "No categories derived from specific tests.")
        else:
            logger.error(f"Invalid orchestration mode or missing parameters: {config.mode}")
            return self._create_early_exit_result(start_time, "Invalid orchestration configuration.")

        try:
            resolved_order = await self._resolve_dependencies(suites_to_run)
        except ValueError as e:
            logger.error(f"Dependency resolution failed: {e}")
            return self._create_early_exit_result(start_time, f"Dependency resolution failed: {e}")

        detailed_results: Dict[TestCategory, Any] = {}
        successful_suites = 0
        failed_suites = 0
        errors_during_run: List[str] = []

        for category in resolved_order:
            if category not in self.test_suites:
                error_msg = f"Test suite for category '{category.value}' not found."
                logger.error(error_msg)
                errors_during_run.append(error_msg)
                failed_suites += 1
                if config.stop_on_failure:
                    break
                continue

            test_suite_instance = self.test_suites[category]
            logger.info(f"Running test suite: {category.value}")
            
            try:
                # Dynamically call the primary run method of each suite
                if hasattr(test_suite_instance, f"run_{category.value.replace('_', '')}_suite"):
                    run_method = getattr(test_suite_instance, f"run_{category.value.replace('_', '')}_suite")
                    result = await run_method()
                elif hasattr(test_suite_instance, f"run_suite"):
                    # Fallback for generic 'run_suite' if specific names aren't adopted
                    result = await test_suite_instance.run_suite()
                else:
                    raise AttributeError(f"Could not find a recognized run method for {category.value} suite.")

                detailed_results[category] = result
                
                if result.get("overall_success", True) or result.get("success_rate", 0) > 0.7: # Consider partial success as overall_success for report
                    successful_suites += 1
                    logger.info(f"✅ Suite '{category.value}' completed successfully.")
                else:
                    failed_suites += 1
                    errors_during_run.append(f"Suite '{category.value}' reported failures.")
                    logger.error(f"❌ Suite '{category.value}' reported failures.")
                    if config.stop_on_failure:
                        logger.warning("Stopping on first failure as configured.")
                        break

            except Exception as e:
                error_msg = f"Error running test suite '{category.value}': {e}"
                logger.error(error_msg, exc_info=True)
                errors_during_run.append(error_msg)
                failed_suites += 1
                if config.stop_on_failure:
                    logger.warning("Stopping on first failure as configured.")
                    break
            finally:
                # Ensure cleanup methods are called
                if hasattr(test_suite_instance, 'cleanup_test_environment'):
                    await test_suite_instance.cleanup_test_environment()

        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        overall_success = failed_suites == 0 and not errors_during_run
        
        orchestration_result = OrchestrationResult(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_suites_run=successful_suites + failed_suites,
            suites_passed=successful_suites,
            suites_failed=failed_suites,
            overall_success=overall_success,
            detailed_results=detailed_results,
            errors=errors_during_run
        )
        
        if config.generate_report:
            await self._generate_orchestration_report(orchestration_result)
        
        logger.info(f"FBA-Bench Test Orchestration finished. Overall success: {overall_success}")
        return orchestration_result

    async def _generate_orchestration_report(self, result: OrchestrationResult):
        """Generates a summary report of the orchestration run."""
        report_file = self.output_dir / "orchestration_report.json"
        
        report_data = {
            "summary": {
                "start_time": result.start_time,
                "end_time": result.end_time,
                "duration_seconds": result.duration_seconds,
                "total_suites_run": result.total_suites_run,
                "suites_passed": result.suites_passed,
                "suites_failed": result.suites_failed,
                "overall_success": result.overall_success,
                "errors": result.errors
            },
            "detailed_results_by_category": {cat.value:res for cat,res in result.detailed_results.items()},
            "configuration_used": self.orchestration_config.__dict__ if self.orchestration_config else {}
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_data, f, indent=4)
        logger.info(f"Orchestration report generated: {report_file.resolve()}")

        # Also generate a human-readable markdown report
        markdown_report_file = self.output_dir / "orchestration_report.md"
        with open(markdown_report_file, 'w') as f:
            f.write(f"# FBA-Bench Test Orchestration Report\n\n")
            f.write(f"**Run Start:** {result.start_time}\n")
            f.write(f"**Run End:** {result.end_time}\n")
            f.write(f"**Duration:** {result.duration_seconds:.2f} seconds\n")
            f.write(f"**Overall Success:** {'✅ Yes' if result.overall_success else '❌ No'}\n\n")
            
            f.write("## Summary\n")
            f.write(f"- Total Test Suites Run: {result.total_suites_run}\n")
            f.write(f"- Suites Passed: {result.suites_passed}\n")
            f.write(f"- Suites Failed: {result.suites_failed}\n")
            f.write(f"- Overall Success Rate: {result.suites_passed / max(1, result.total_suites_run):.1%}\n\n")

            if result.errors:
                f.write("## Errors During Orchestration\n")
                for error in result.errors:
                    f.write(f"- ❌ {error}\n")
                f.write("\n")

            f.write("## Detailed Results by Category\n")
            for category, details in result.detailed_results.items():
                category_success = details.get("overall_success", False) or details.get("success_rate", 0.0) > 0.7
                status_icon = '✅' if category_success else '❌'
                f.write(f"### {status_icon} {category.value.replace('_', ' ').title()} Suite\n")
                if "success_rate" in details:
                    f.write(f"- Success Rate: {details.get('success_rate', 0.0):.1%}\n")
                if "total_tests" in details:
                    f.write(f"- Total Tests: {details.get('total_tests', 'N/A')}\n")
                if "passed_tests" in details:
                    f.write(f"- Passed Tests: {details.get('passed_tests', 'N/A')}\n")
                if "failed_tests" in details:
                    f.write(f"- Failed Tests: {details.get('failed_tests', 'N/A')}\n")
                if "regression_free" in details:
                    f.write(f"- Regression Free: {'Yes' if details['regression_free'] else 'No'}\n")
                if "curriculum_validated" in details:
                    f.write(f"- Curriculum Validated: {'Yes' if details['curriculum_validated'] else 'No'}\n")
                if "real_world_ready" in details:
                    f.write(f"- Real-World Ready: {'Yes' if details['real_world_ready'] else 'No'}\n")
                if "extensibility_validated" in details:
                    f.write(f"- Extensibility Validated: {'Yes' if details['extensibility_validated'] else 'No'}\n")
                if "error_details" in details and details["error_details"]:
                    f.write(f"- Specific Error: {details['error_details']}\n")
                f.write("\n")

        logger.info(f"Markdown report generated: {markdown_report_file.resolve()}")

    def _create_early_exit_result(self, start_time: datetime.now(), error_message: str) -> OrchestrationResult:
        """Helper to create a result when orchestration exits early due to configuration errors."""
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        return OrchestrationResult(
            start_time=start_time.isoformat(),
            end_time=end_time.isoformat(),
            duration_seconds=duration,
            total_suites_run=0,
            suites_passed=0,
            suites_failed=0,
            overall_success=False,
            errors=[error_message]
        )


# CLI runner for direct execution
async def main():
    """Run automated test orchestration suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    orchestrator = TestOrchestrationSystem()
    
    # Example usage:
    # config = OrchestrationConfig(mode=ExecutionMode.FULL_SUITE, stop_on_failure=False)
    # config = OrchestrationConfig(mode=ExecutionMode.CRITICAL_PATH)
    # config = OrchestrationConfig(mode=ExecutionMode.CATEGORY, selected_categories=[TestCategory.FUNCTIONAL, TestCategory.PERFORMANCE])
    config = OrchestrationConfig(mode=ExecutionMode.FULL_SUITE, clear_output_dir=True) # Default to full suite with cleanup
    
    try:
        results = await orchestrator.run_suite(config)
        
        print("\n" + "="*80)
        print("FBA-BENCH TEST ORCHESTRATION SUMMARY")
        print("="*80)
        print(f"Overall Success: {'🎉 PASSED' if results.overall_success else '❌ FAILED'}")
        print(f"Total Suites Run: {results.total_suites_run}")
        print(f"Suites Passed: {results.suites_passed}")
        print(f"Suites Failed: {results.suites_failed}")
        print(f"Total Duration: {results.duration_seconds:.2f}s")
        
        if results.errors:
            print("\nErrors and Warnings:")
            for error in results.errors:
                print(f"- {error}")
        
        print("\nDetailed Suite Results:")
        for category, details in results.detailed_results.items():
            status_icon = '✅' if details.get("overall_success", True) or details.get("success_rate", 0.0) > 0.7 else '❌'
            print(f"  {status_icon} {category.value.replace('_', ' ').title()}: Passed: {details.get('passed_tests', 'N/A')}, Failed: {details.get('failed_tests', 'N/A')}, Success Rate: {details.get('success_rate', 0.0):.1%}")
            
        print("="*80)
        
        if not results.overall_success:
            sys.exit(1) # Indicate failure in CLI if running from script
            
    except Exception as e:
        logger.critical(f"Orchestration system crashed unexpectedly: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())