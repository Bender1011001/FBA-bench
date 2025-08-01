"""
Documentation and Examples Testing Framework for FBA-Bench

Validates that all documentation examples work correctly,
are up-to-date, and are consistent with the current codebase.
"""

import asyncio
import logging
import pytest
import time
import json
import tempfile
import shutil
import requests
import importlib.util
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
from enum import Enum
from unittest.mock import patch, MagicMock

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from community.community_tools import ExampleValidator, DocumentationGenerator
from llm_interface.openrouter_client import OpenRouterLLMClient
from llm_interface.deterministic_client import DeterministicLLMClient
from agents.hierarchical_planner import StrategicPlanner
from agents.skill_coordinator import SkillCoordinator
from memory_experiments.dual_memory_manager import DualMemoryManager
from event_bus import EventBus
from events import TickEvent, SaleOccurred, SetPriceCommand

logger = logging.getLogger(__name__)


class DocTestType(Enum):
    """Types of documentation and example tests."""
    CODE_EXAMPLE_EXECUTION = "code_example_execution"
    DOCUMENTATION_CONSISTENCY = "documentation_consistency"
    EXAMPLE_COMPLETENESS = "example_completeness"
    API_DOC_ACCURACY = "api_doc_accuracy"
    TUTORIAL_VALIDATION = "tutorial_validation"


@dataclass
class DocExampleTestResult:
    """Results from documentation and example testing."""
    test_name: str
    test_type: DocTestType
    success: bool
    details: Dict[str, Any]
    duration_seconds: float
    error_details: Optional[str] = None


class MockAgent:
    """A minimal mock agent for example execution."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.actions = []
        self.metrics = {"revenue": 0.0, "profit": 0.0}

    async def initialize(self):
        pass

    async def execute_action(self, action: Dict[str, Any]):
        self.actions.append(action)
        if action.get("type") == "set_price":
            print(f"Agent {self.agent_id} setting price for {action.get('asin')} to {action.get('new_price')}")
        elif action.get("type") == "place_order":
            print(f"Agent {self.agent_id} placing order for {action.get('quantity')} of {action.get('asin')}")
        self.metrics["revenue"] += action.get("revenue_impact", 0)
        self.metrics["profit"] += action.get("profit_impact", 0)


class MockLLMClient:
    """A mock LLM client for testing purposes."""
    def __init__(self, responses: Dict[str, str]):
        self._responses = responses
        self.call_count = 0

    async def get_completion(self, prompt: str, model: str, temperature: float) -> str:
        self.call_count += 1
        key = f"prompt: {prompt}, model: {model}, temp: {temperature}"
        return self._responses.get(key, "Mocked LLM response.")


class DocumentationAndExamplesTestSuite:
    """
    Comprehensive suite for validating FBA-Bench documentation and examples.
    """
    
    def __init__(self):
        self.test_results: List[DocExampleTestResult] = []
        self.temp_dir = None
        
    async def setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment for doc and examples testing."""
        logger.info("Setting up documentation and examples test environment")
        
        self.temp_dir = tempfile.mkdtemp(prefix="fba_bench_docs_examples_")
        
        # Define common paths for examples and documentation
        project_root = Path(__file__).parent.parent.parent
        self.example_dirs = [
            project_root / "examples",
            project_root / "plugins" / "examples",
            project_root / "redteam_scripts" / "examples"
        ]
        self.doc_files = [
            project_root / "README.md",
            project_root / "docs" / "LLM_contract.md",
            project_root / "docs" / "red_team_framework.md",
            project_root / "Benchmark Philosophy.md",
            project_root / "curriculum_system_design.md",
            project_root / "FBA-Bench-Implementation-Plan.md",
            project_root / "Key-Issues-and-Proposed-Changes-Ver.txt",
            project_root / "Foundation_research.txt",
            project_root / "From-Simple-Commerce-to-Complex-Eco.txt"
        ]
        
        environment = {
            "temp_dir": self.temp_dir,
            "example_dirs": self.example_dirs,
            "doc_files": self.doc_files
        }
        
        return environment

    async def _extract_code_blocks_from_markdown(self, markdown_content: str) -> List[str]:
        """Extracts Python code blocks from markdown content."""
        code_blocks = []
        in_code_block = False
        current_block = []
        for line in markdown_content.splitlines():
            if line.strip().startswith("```python"):
                in_code_block = True
                current_block = []
            elif line.strip() == "```" and in_code_block:
                code_blocks.append("\n".join(current_block))
                in_code_block = False
                current_block = []
            elif in_code_block:
                current_block.append(line)
        return code_blocks

    async def _execute_python_code_snippet(self, code_snippet: str, file_name: str) -> Tuple[bool, Optional[str]]:
        """Executes a Python code snippet in a sandboxed environment."""
        temp_file_path = Path(self.temp_dir) / file_name
        
        # Patch external dependencies for safe execution
        mock_requests = MagicMock(spec=requests)
        mock_llm_client = MockLLMClient({
            "prompt: Tell me about X, model: gpt-4, temp: 0.0": "X is a cool concept.",
            "prompt: plan, model: gpt-4, temp: 0.0": "Strategic plan for maximizing profit."
        })
        mock_event_bus = EventBus() # Use a real event bus for internal event flow
        mock_agent = MockAgent("test_example_agent")

        with open(temp_file_path, "w") as f:
            f.write(code_snippet)
        
        # Temporarily add temp_dir to sys.path to allow internal imports in the snippet
        sys.path.insert(0, str(self.temp_dir))
        
        try:
            # Use importlib to load the module from the temporary file path
            spec = importlib.util.spec_from_file_location("temp_module", temp_file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                
                # Patch modules before execution
                with patch('requests', new=mock_requests), \
                     patch('llm_interface.openrouter_client.OpenRouterLLMClient', new=lambda *a, **kw: mock_llm_client), \
                     patch('llm_interface.deterministic_client.DeterministicLLMClient', new=lambda *a, **kw: mock_llm_client), \
                     patch('event_bus.get_event_bus', new=lambda: mock_event_bus), \
                     patch('agents.hierarchical_planner.StrategicPlanner', new=lambda *a, **kw: MagicMock(spec=StrategicPlanner, create_strategic_plan=lambda *a, **kw: asyncio.sleep(0.01) or [{"objective": "improve"}], validate_action_alignment=lambda *a, **kw: (True, 0.9, "reasoning"))), \
                     patch('agents.skill_coordinator.SkillCoordinator', new=lambda *a, **kw: MagicMock(spec=SkillCoordinator, dispatch_event=lambda event: asyncio.sleep(0.01) or [f"action_for_{type(event).__name__}"])), \
                     patch('memory_experiments.dual_memory_manager.DualMemoryManager', new=lambda *a, **kw: MagicMock(spec=DualMemoryManager, store_memory=lambda *a, **kw: None, query_memories=lambda *a, **kw: [])):
                    
                    spec.loader.exec_module(module)

                # If the snippet has a main function, try to run it
                if hasattr(module, 'main') and callable(module.main):
                    if asyncio.iscoroutinefunction(module.main):
                        await module.main()
                    else:
                        module.main()
                
                return True, None
            return False, "Could not load module spec."
        except Exception as e:
            return False, str(e)
        finally:
            if str(self.temp_dir) in sys.path:
                sys.path.remove(str(self.temp_dir))
            
    async def test_code_example_execution(self) -> DocExampleTestResult:
        """Tests that all executable code examples run without errors."""
        test_name = "code_example_execution"
        start_time = time.time()
        
        try:
            logger.info("Testing code example execution")
            
            environment = await self.setup_test_environment()
            
            example_execution_results = {}
            total_examples = 0
            
            # Iterate through example directories
            for example_dir in environment["example_dirs"]:
                if not example_dir.exists():
                    logger.warning(f"Example directory not found: {example_dir}")
                    continue
                    
                for root, _, files in os.walk(example_dir):
                    for file_name in files:
                        if file_name.endswith(".py"):
                            file_path = Path(root) / file_name
                            total_examples += 1
                            logger.info(f"Executing example: {file_path.relative_to(example_dir.parent)}")
                            
                            try:
                                with open(file_path, 'r') as f:
                                    code_content = f.read()

                                # Execute the file
                                success, error_msg = await self._execute_python_code_snippet(
                                    code_content, file_name
                                )
                                example_execution_results[str(file_path)] = {
                                    "success": success, 
                                    "error": error_msg
                                }
                                if not success:
                                    logger.error(f"Example {file_path} failed: {error_msg}")
                            except Exception as e:
                                example_execution_results[str(file_path)] = {
                                    "success": False, 
                                    "error": str(e)
                                }
                                logger.error(f"Failed to read or execute example {file_path}: {e}")
            
            # Extract code blocks from markdown files as well
            for doc_file_path in environment["doc_files"]:
                if doc_file_path.exists() and doc_file_path.suffix == ".md":
                    try:
                        with open(doc_file_path, 'r', encoding='utf-8') as f:
                            markdown_content = f.read()
                        
                        code_blocks = await self._extract_code_blocks_from_markdown(markdown_content)
                        for i, code_block in enumerate(code_blocks):
                            temp_file_name = f"{doc_file_path.stem}_block_{i}.py"
                            total_examples += 1
                            logger.info(f"Executing code block from {doc_file_path.name}: Block {i}")
                            
                            success, error_msg = await self._execute_python_code_snippet(
                                code_block, temp_file_name
                            )
                            example_execution_results[f"{str(doc_file_path)}:Block{i}"] = {
                                "success": success, 
                                "error": error_msg
                            }
                            if not success:
                                logger.error(f"Code block from {doc_file_path.name} (Block {i}) failed: {error_msg}")
                    except Exception as e:
                        logger.error(f"Error processing markdown file {doc_file_path}: {e}")

            # Calculate overall success
            passed_examples = sum(1 for r in example_execution_results.values() if r["success"])
            overall_success = passed_examples == total_examples and total_examples > 0
            
            details = {
                "total_examples_tested": total_examples,
                "passed_examples": passed_examples,
                "failed_examples": total_examples - passed_examples,
                "example_results": example_execution_results
            }
            
            duration = time.time() - start_time
            
            return DocExampleTestResult(
                test_name=test_name,
                test_type=DocTestType.CODE_EXAMPLE_EXECUTION,
                success=overall_success,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Code example execution test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return DocExampleTestResult(
                test_name=test_name,
                test_type=DocTestType.CODE_EXAMPLE_EXECUTION,
                success=False,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_documentation_consistency(self) -> DocExampleTestResult:
        """Tests for consistency between documentation and actual implementation (e.g., API references)."""
        test_name = "documentation_consistency"
        start_time = time.time()
        
        try:
            logger.info("Testing documentation consistency")
            environment = await self.setup_test_environment()
            
            consistency_checks = {}
            overall_consistent = True

            # Example: Check if API version in LLM_contract.md matches a hardcoded or derived version
            llm_contract_file = Path(os.path.join(Path(__file__).parent.parent.parent, "docs", "LLM_contract.md"))
            if llm_contract_file.exists():
                with open(llm_contract_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for "Version 1.0" in the contract
                has_version_1_0 = "Version 1.0" in content
                consistency_checks["LLM_contract_version_1_0"] = has_version_1_0

                # Check for "Core LLM API Endpoints" section
                has_api_endpoints_section = "## Core LLM API Endpoints" in content
                consistency_checks["LLM_contract_api_endpoints_section"] = has_api_endpoints_section

                if not (has_version_1_0 and has_api_endpoints_section):
                    overall_consistent = False
            else:
                logger.warning(f"LLM_contract.md not found at {llm_contract_file}. Skipping consistency check.")
                consistency_checks["LLM_contract_existence"] = False
                overall_consistent = False

            # Example: Check if key issues documented in Key-Issues-and-Proposed-Changes-Ver.txt are consistently referenced
            key_issues_file = Path(os.path.join(Path(__file__).parent.parent.parent, "Key-Issues-and-Proposed-Changes-Ver.txt"))
            if key_issues_file.exists():
                with open(key_issues_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                expected_issues = [
                    "Cognitive Loop Improvements",
                    "Multi-Skill Agent Capabilities",
                    "Reproducibility and Determinism",
                    "Infrastructure Scalability",
                    "Scenario Diversity and Curriculum Validation",
                    "Tool Interfaces and Observability",
                    "Missing Features"
                ]
                
                all_issues_referenced = all(issue in content for issue in expected_issues)
                consistency_checks["Key_Issues_document_completeness"] = all_issues_referenced
                if not all_issues_referenced:
                    overall_consistent = False
            else:
                logger.warning(f"Key-Issues-and-Proposed-Changes-Ver.txt not found at {key_issues_file}. Skipping consistency check.")
                consistency_checks["Key_Issues_document_existence"] = False
                overall_consistent = False
            
            details = {
                "consistency_checks_performed": consistency_checks
            }
            
            duration = time.time() - start_time
            
            return DocExampleTestResult(
                test_name=test_name,
                test_type=DocTestType.DOCUMENTATION_CONSISTENCY,
                success=overall_consistent,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Documentation consistency test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return DocExampleTestResult(
                test_name=test_name,
                test_type=DocTestType.DOCUMENTATION_CONSISTENCY,
                success=False,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def test_example_completeness(self) -> DocExampleTestResult:
        """Checks if all major features/modules have corresponding examples."""
        test_name = "example_completeness"
        start_time = time.time()
        
        try:
            logger.info("Testing example completeness")
            environment = await self.setup_test_environment()
            
            completeness_status = {}
            major_components_with_examples = {
                "agents": False,
                "memory_experiments": False,
                "reproducibility": False,
                "infrastructure": False,
                "observability": False,
                "scenarios": False,
                "plugins": False,
                "learning": False,
                "integration": False,
                "redteam": False
            }

            # Map example files to components
            component_examples_found = {
                "agents": ["advanced_agent.py", "hierarchical_planner.py", "skill_coordinator.py"],
                "memory_experiments": ["dual_memory_manager.py", "reflection_module.py"],
                "reproducibility": ["llm_cache.py", "sim_seed.py"],
                "infrastructure": ["llm_batcher.py", "performance_monitor.py"],
                "observability": ["trace_analyzer.py", "alert_system.py"],
                "scenarios": ["scenario_engine.py", "curriculum_validator.py"],
                "plugins": ["plugin_framework.py", "base_agent_plugin.py"],
                "learning": ["episodic_learning.py"],
                "integration": ["real_world_adapter.py", "marketplace_apis"],
                "redteam": ["adversarial_event_injector.py", "gauntlet_runner.py"]
            }

            all_example_files = []
            for example_dir in environment["example_dirs"]:
                if example_dir.exists():
                    for root, _, files in os.walk(example_dir):
                        for file_name in files:
                            if file_name.endswith(".py") or Path(file_name).suffix in ['.yaml', '.md']:
                                all_example_files.append(str(Path(root) / file_name))
            
            # Check for specific example types that indicate coverage
            for comp, expected_files in component_examples_found.items():
                found_for_comp = False
                for expected_file_part in expected_files:
                    # Check if any example file path contains the component's expected file part
                    if any(expected_file_part in f for f in all_example_files):
                        found_for_comp = True
                        break
                major_components_with_examples[comp] = found_for_comp
            
            completeness_status["major_components_covered"] = major_components_with_examples
            
            # Overall completeness check
            overall_complete = all(major_components_with_examples.values())
            
            details = {
                "example_completeness_status": completeness_status
            }
            
            duration = time.time() - start_time
            
            return DocExampleTestResult(
                test_name=test_name,
                test_type=DocTestType.EXAMPLE_COMPLETENESS,
                success=overall_complete,
                details=details,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Example completeness test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return DocExampleTestResult(
                test_name=test_name,
                test_type=DocTestType.EXAMPLE_COMPLETENESS,
                success=False,
                details={},
                duration_seconds=duration,
                error_details=str(e)
            )

    async def run_documentation_examples_test_suite(self) -> Dict[str, Any]:
        """Runs the complete documentation and examples testing suite."""
        logger.info("Starting comprehensive documentation and examples testing suite")
        suite_start = time.time()
        
        test_methods = [
            self.test_code_example_execution,
            self.test_documentation_consistency,
            self.test_example_completeness,
        ]
        
        results = []
        
        for test_method in test_methods:
            try:
                logger.info(f"Running {test_method.__name__}")
                result = await test_method()
                results.append(result)
                self.test_results.append(result)
                
                if result.success:
                    logger.info(f"✅ {result.test_name} passed")
                else:
                    logger.error(f"❌ {result.test_name} failed: {result.error_details if result.error_details else 'Failure detected'}")
                    
            except Exception as e:
                logger.error(f"Execution of test method {test_method.__name__} crashed: {e}", exc_info=True)
                results.append(DocExampleTestResult(
                    test_name=test_method.__name__,
                    test_type=DocTestType.CODE_EXAMPLE_EXECUTION,
                    success=False,
                    details={},
                    duration_seconds=0,
                    error_details=f"Test runner crashed: {str(e)}"
                ))
        
        suite_duration = time.time() - suite_start
        
        # Compile summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        overall_doc_quality = passed_tests == total_tests
        
        summary = {
            "suite_duration_seconds": suite_duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "overall_documentation_quality_pass": overall_doc_quality,
            "test_results": [result.__dict__ for result in results]
        }
        
        logger.info(f"Documentation and examples testing suite completed: {passed_tests}/{total_tests} tests passed.")
        if overall_doc_quality:
            logger.info("🎉 Documentation and examples validated!")
        else:
            logger.warning("⚠️ Some documentation or example issues detected.")

        return summary
    
    async def cleanup_test_environment(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Documentation and examples test environment cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# CLI runner for direct execution
async def main():
    """Run documentation and examples testing suite."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = DocumentationAndExamplesTestSuite()
    
    try:
        results = await test_suite.run_documentation_examples_test_suite()
        
        print("\n" + "="*80)
        print("DOCUMENTATION AND EXAMPLES TESTING RESULTS")
        print("="*80)
        print(f"Total Tests Run: {results['total_tests']}")
        print(f"Tests Passed: {results['passed_tests']}")
        print(f"Tests Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Suite Duration: {results['suite_duration_seconds']:.2f}s")
        
        if results['overall_documentation_quality_pass']:
            print("\n🎉 DOCUMENTATION AND EXAMPLES VALIDATED!")
            print("All code examples run successfully, and documentation consistent.")
        else:
            print("\n⚠️  DOCUMENTATION OR EXAMPLES ISSUES DETECTED.")
            print("Review test results for details.")
        
        print("="*80)
        
    finally:
        await test_suite.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(main())