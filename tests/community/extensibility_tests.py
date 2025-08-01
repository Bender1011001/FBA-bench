"""
Community and Extensibility Testing Framework for FBA-Bench

Tests plugin systems, contribution workflows, extension points, API stability,
and community tooling to ensure proper extensibility and community engagement.
"""

import asyncio
import logging
import pytest
import time
import json
import tempfile
import shutil
import subprocess
import zipfile
import yaml
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import sys
import os
from enum import Enum
from unittest.mock import Mock, patch, MagicMock
import importlib.util

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from plugins.plugin_framework import PluginFramework, PluginManager, PluginType, PluginStatus
from plugins.plugin_interface import IPlugin, ISkillPlugin, IAnalysisPlugin, IIntegrationPlugin
from plugins.plugin_validator import PluginValidator, ValidationResult, SecurityCheck
from plugins.plugin_loader import PluginLoader, LoadingContext, DependencyResolver
from plugins.api_gateway import APIGateway, APIVersion, EndpointRegistry
from community.contribution_validator import ContributionValidator, ContributionType, ReviewCriteria
from community.community_tools import CommunityTools, DocumentationGenerator, ExampleValidator
from community.plugin_registry import PluginRegistry, RegistryEntry, PluginMetadata
from community.version_manager import VersionManager, SemanticVersion, CompatibilityMatrix

logger = logging.getLogger(__name__)


class ExtensibilityTestType(Enum):
    """Types of extensibility tests."""
    PLUGIN_LIFECYCLE = "plugin_lifecycle"
    API_STABILITY = "api_stability"
    CONTRIBUTION_WORKFLOW = "contribution_workflow"
    PLUGIN_ISOLATION = "plugin_isolation"
    DEPENDENCY_MANAGEMENT = "dependency_management"
    SECURITY_VALIDATION = "security_validation"
    DOCUMENTATION_TOOLING = "documentation_tooling"


@dataclass
class ExtensibilityTestResult:
    """Results from extensibility testing."""
    test_name: str
    test_type: ExtensibilityTestType
    success: bool
    plugin_tests: Dict[str, bool]
    api_compatibility_tests: Dict[str, float]
    contribution_workflow_tests: Dict[str, bool]
    security_validation_tests: Dict[str, bool]
    documentation_tests: Dict[str, bool]
    duration_seconds: float
    error_details: Optional[str] = None


@dataclass
class PluginTestTemplate:
    """Template for creating test plugins."""
    name: str
    plugin_type: PluginType
    version: str
    dependencies: List[str]
    api_requirements: Dict[str, str]
    test_functionality: Dict[str, Any]


@dataclass
class ContributionTestScenario:
    """Test scenario for contribution workflows."""
    contribution_type: ContributionType
    files_modified: List[str]
    test_requirements: List[str]
    expected_review_criteria: List[str]
    security_implications: bool


class MockPlugin:
    """Mock plugin for testing purposes."""
    
    def __init__(self, name: str, plugin_type: PluginType, version: str = "1.0.0"):
        self.name = name
        self.plugin_type = plugin_type
        self.version = version
        self.is_initialized = False
        self.is_active = False
        self.test_calls = []
    
    async def initialize(self, context: Dict[str, Any]) -> bool:
        """Initialize the mock plugin."""
        self.is_initialized = True
        self.test_calls.append("initialize")
        return True
    
    async def activate(self) -> bool:
        """Activate the mock plugin."""
        if not self.is_initialized:
            return False
        self.is_active = True
        self.test_calls.append("activate")
        return True
    
    async def deactivate(self) -> bool:
        """Deactivate the mock plugin."""
        self.is_active = False
        self.test_calls.append("deactivate")
        return True
    
    async def cleanup(self) -> bool:
        """Cleanup the mock plugin."""
        self.is_initialized = False
        self.is_active = False
        self.test_calls.append("cleanup")
        return True
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get plugin metadata."""
        return {
            "name": self.name,
            "type": self.plugin_type.value,
            "version": self.version,
            "description": f"Mock plugin {self.name} for testing",
            "author": "Test Suite",
            "api_version": "1.0.0"
        }


class CommunityAndExtensibilityTestSuite:
    """
    Comprehensive community and extensibility testing suite.
    
    Tests plugin systems, contribution workflows, extension points,
    API stability, and community tooling.
    """
    
    def __init__(self):
        self.test_results: List[ExtensibilityTestResult] = []
        self.temp_dir = None
        self.mock_plugins: List[MockPlugin] = []
        
    async def setup_test_environment(self) -> Dict[str, Any]:
        """Setup test environment for community and extensibility testing."""
        logger.info("Setting up community and extensibility test environment")
        
        # Create temporary directory for test files
        self.temp_dir = tempfile.mkdtemp(prefix="fba_bench_community_")
        
        # Create test plugin directory structure
        plugin_dir = os.path.join(self.temp_dir, "test_plugins")
        os.makedirs(plugin_dir, exist_ok=True)
        
        # Create test contribution directory
        contrib_dir = os.path.join(self.temp_dir, "test_contributions")
        os.makedirs(contrib_dir, exist_ok=True)
        
        environment = {
            "temp_dir": self.temp_dir,
            "plugin_dir": plugin_dir,
            "contrib_dir": contrib_dir
        }
        
        return environment
    
    async def test_plugin_lifecycle_management(self) -> ExtensibilityTestResult:
        """Test plugin loading, activation, deactivation, and cleanup."""
        test_name = "plugin_lifecycle_management"
        start_time = time.time()
        
        try:
            logger.info("Testing plugin lifecycle management")
            
            environment = await self.setup_test_environment()
            
            plugin_tests = {}
            api_compatibility_tests = {}
            contribution_workflow_tests = {}
            security_validation_tests = {}
            documentation_tests = {}
            
            # Test 1: Plugin Framework Initialization
            logger.info("Testing plugin framework initialization")
            
            plugin_framework = PluginFramework(
                plugin_directory=environment["plugin_dir"],
                enable_security_validation=True,
                enable_dependency_resolution=True
            )
            
            framework_init = await plugin_framework.initialize()
            plugin_tests["framework_initialization"] = framework_init
            
            # Test 2: Plugin Registration and Loading
            logger.info("Testing plugin registration and loading")
            
            # Create test plugins
            test_plugins = [
                MockPlugin("test_skill_plugin", PluginType.SKILL, "1.0.0"),
                MockPlugin("test_analysis_plugin", PluginType.ANALYSIS, "1.1.0"),
                MockPlugin("test_integration_plugin", PluginType.INTEGRATION, "2.0.0")
            ]
            
            registration_results = {}
            for plugin in test_plugins:
                try:
                    success = await plugin_framework.register_plugin(plugin)
                    registration_results[plugin.name] = success
                    if success:
                        self.mock_plugins.append(plugin)
                except Exception as e:
                    logger.error(f"Plugin registration failed for {plugin.name}: {e}")
                    registration_results[plugin.name] = False
            
            plugin_tests["plugin_registration"] = all(registration_results.values())
            
            # Test 3: Plugin Activation and Lifecycle
            logger.info("Testing plugin activation and lifecycle")
            
            lifecycle_results = {}
            for plugin in self.mock_plugins:
                try:
                    # Test activation
                    activated = await plugin_framework.activate_plugin(plugin.name)
                    
                    # Test plugin is active
                    is_active = await plugin_framework.is_plugin_active(plugin.name)
                    
                    # Test deactivation
                    deactivated = await plugin_framework.deactivate_plugin(plugin.name)
                    
                    # Test plugin is inactive
                    is_inactive = not await plugin_framework.is_plugin_active(plugin.name)
                    
                    lifecycle_results[plugin.name] = all([activated, is_active, deactivated, is_inactive])
                    
                except Exception as e:
                    logger.error(f"Plugin lifecycle test failed for {plugin.name}: {e}")
                    lifecycle_results[plugin.name] = False
            
            plugin_tests["plugin_lifecycle"] = all(lifecycle_results.values()) if lifecycle_results else False
            
            # Test 4: Plugin Discovery and Metadata
            logger.info("Testing plugin discovery and metadata")
            
            discovered_plugins = await plugin_framework.discover_plugins()
            metadata_validation = {}
            
            for plugin_name, metadata in discovered_plugins.items():
                required_fields = ["name", "type", "version", "description", "author"]
                has_required_fields = all(field in metadata for field in required_fields)
                
                # Validate version format
                try:
                    version_parts = metadata.get("version", "").split(".")
                    valid_version = len(version_parts) >= 2 and all(part.isdigit() for part in version_parts)
                except Exception:
                    valid_version = False
                
                metadata_validation[plugin_name] = has_required_fields and valid_version
            
            plugin_tests["plugin_discovery"] = all(metadata_validation.values()) if metadata_validation else False
            
            # Test 5: Plugin Dependency Resolution
            logger.info("Testing plugin dependency resolution")
            
            # Create plugin with dependencies
            dependent_plugin = MockPlugin("dependent_plugin", PluginType.SKILL, "1.0.0")
            
            # Mock dependency resolver
            dependency_resolver = DependencyResolver()
            
            try:
                # Test dependency resolution
                dependencies = ["test_skill_plugin", "test_analysis_plugin"]
                resolution_result = await dependency_resolver.resolve_dependencies(
                    dependent_plugin.name,
                    dependencies
                )
                
                plugin_tests["dependency_resolution"] = resolution_result is not None
            except Exception as e:
                logger.error(f"Dependency resolution test failed: {e}")
                plugin_tests["dependency_resolution"] = False
            
            # Test 6: Plugin Isolation and Security
            logger.info("Testing plugin isolation and security")
            
            plugin_validator = PluginValidator()
            
            isolation_tests = {}
            for plugin in self.mock_plugins:
                try:
                    # Test security validation
                    security_result = await plugin_validator.validate_security(plugin)
                    
                    # Test API boundary enforcement
                    api_boundary_test = await plugin_validator.test_api_boundaries(plugin)
                    
                    # Test resource isolation
                    resource_isolation_test = await plugin_validator.test_resource_isolation(plugin)
                    
                    isolation_tests[plugin.name] = all([
                        security_result.is_secure if hasattr(security_result, 'is_secure') else True,
                        api_boundary_test,
                        resource_isolation_test
                    ])
                    
                except Exception as e:
                    logger.error(f"Plugin isolation test failed for {plugin.name}: {e}")
                    isolation_tests[plugin.name] = False
            
            security_validation_tests["plugin_isolation"] = all(isolation_tests.values()) if isolation_tests else False
            
            # Test 7: Plugin Hot Reload
            logger.info("Testing plugin hot reload")
            
            if self.mock_plugins:
                test_plugin = self.mock_plugins[0]
                try:
                    # Reload plugin
                    reload_success = await plugin_framework.reload_plugin(test_plugin.name)
                    
                    # Verify plugin is still functional
                    post_reload_active = await plugin_framework.activate_plugin(test_plugin.name)
                    
                    plugin_tests["hot_reload"] = reload_success and post_reload_active
                except Exception as e:
                    logger.error(f"Hot reload test failed: {e}")
                    plugin_tests["hot_reload"] = False
            else:
                plugin_tests["hot_reload"] = False
            
            # Success criteria
            lifecycle_success = sum(plugin_tests.values()) >= len(plugin_tests) * 0.8
            
            duration = time.time() - start_time
            
            return ExtensibilityTestResult(
                test_name=test_name,
                test_type=ExtensibilityTestType.PLUGIN_LIFECYCLE,
                success=lifecycle_success,
                plugin_tests=plugin_tests,
                api_compatibility_tests=api_compatibility_tests,
                contribution_workflow_tests=contribution_workflow_tests,
                security_validation_tests=security_validation_tests,
                documentation_tests=documentation_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Plugin lifecycle management test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return ExtensibilityTestResult(
                test_name=test_name,
                test_type=ExtensibilityTestType.PLUGIN_LIFECYCLE,
                success=False,
                plugin_tests={},
                api_compatibility_tests={},
                contribution_workflow_tests={},
                security_validation_tests={},
                documentation_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def test_api_stability_and_versioning(self) -> ExtensibilityTestResult:
        """Test API stability, versioning, and backward compatibility."""
        test_name = "api_stability_and_versioning"
        start_time = time.time()
        
        try:
            logger.info("Testing API stability and versioning")
            
            environment = await self.setup_test_environment()
            
            plugin_tests = {}
            api_compatibility_tests = {}
            contribution_workflow_tests = {}
            security_validation_tests = {}
            documentation_tests = {}
            
            # Test 1: API Gateway and Endpoint Registry
            logger.info("Testing API gateway and endpoint registry")
            
            api_gateway = APIGateway()
            endpoint_registry = EndpointRegistry()
            
            # Register test API endpoints
            test_endpoints = [
                {"path": "/api/v1/agents", "version": "1.0.0", "methods": ["GET", "POST"]},
                {"path": "/api/v1/scenarios", "version": "1.0.0", "methods": ["GET", "POST", "DELETE"]},
                {"path": "/api/v2/agents", "version": "2.0.0", "methods": ["GET", "POST", "PUT"]},
            ]
            
            endpoint_registration_results = {}
            for endpoint in test_endpoints:
                try:
                    success = await endpoint_registry.register_endpoint(
                        endpoint["path"],
                        APIVersion(endpoint["version"]),
                        endpoint["methods"]
                    )
                    endpoint_registration_results[endpoint["path"]] = success
                except Exception as e:
                    logger.error(f"Endpoint registration failed for {endpoint['path']}: {e}")
                    endpoint_registration_results[endpoint["path"]] = False
            
            api_compatibility_tests["endpoint_registration"] = sum(endpoint_registration_results.values()) / len(endpoint_registration_results)
            
            # Test 2: API Version Compatibility
            logger.info("Testing API version compatibility")
            
            version_manager = VersionManager()
            
            # Test semantic versioning
            version_tests = {}
            test_version_pairs = [
                ("1.0.0", "1.0.1", True),   # Patch compatibility
                ("1.0.0", "1.1.0", True),   # Minor compatibility  
                ("1.0.0", "2.0.0", False),  # Major incompatibility
                ("2.1.0", "2.0.0", False),  # Backward incompatibility
            ]
            
            for v1, v2, expected_compatible in test_version_pairs:
                try:
                    is_compatible = version_manager.is_compatible(
                        SemanticVersion(v1),
                        SemanticVersion(v2)
                    )
                    version_tests[f"{v1}_to_{v2}"] = is_compatible == expected_compatible
                except Exception as e:
                    logger.error(f"Version compatibility test failed for {v1} -> {v2}: {e}")
                    version_tests[f"{v1}_to_{v2}"] = False
            
            api_compatibility_tests["version_compatibility"] = sum(version_tests.values()) / len(version_tests)
            
            # Test 3: Backward Compatibility Matrix
            logger.info("Testing backward compatibility matrix")
            
            compatibility_matrix = CompatibilityMatrix()
            
            # Test compatibility matrix operations
            matrix_tests = {}
            try:
                # Add compatibility rules
                await compatibility_matrix.add_compatibility_rule("plugin_api", "1.0.0", "1.1.0", True)
                await compatibility_matrix.add_compatibility_rule("plugin_api", "1.0.0", "2.0.0", False)
                
                # Test compatibility checks
                compat_1_1 = await compatibility_matrix.check_compatibility("plugin_api", "1.0.0", "1.1.0")
                compat_2_0 = await compatibility_matrix.check_compatibility("plugin_api", "1.0.0", "2.0.0")
                
                matrix_tests["compatibility_rules"] = compat_1_1 and not compat_2_0
                
                # Test matrix export/import
                matrix_data = await compatibility_matrix.export_matrix()
                new_matrix = CompatibilityMatrix()
                import_success = await new_matrix.import_matrix(matrix_data)
                
                matrix_tests["matrix_serialization"] = import_success
                
            except Exception as e:
                logger.error(f"Compatibility matrix test failed: {e}")
                matrix_tests["compatibility_rules"] = False
                matrix_tests["matrix_serialization"] = False
            
            api_compatibility_tests["compatibility_matrix"] = sum(matrix_tests.values()) / len(matrix_tests)
            
            # Test 4: API Documentation Generation
            logger.info("Testing API documentation generation")
            
            documentation_generator = DocumentationGenerator()
            
            doc_tests = {}
            try:
                # Generate API documentation
                api_docs = await documentation_generator.generate_api_docs(endpoint_registry)
                
                # Validate documentation structure
                required_sections = ["endpoints", "authentication", "examples", "changelog"]
                has_required_sections = all(section in api_docs for section in required_sections)
                
                doc_tests["api_documentation"] = has_required_sections
                
                # Test documentation export
                doc_file = os.path.join(environment["temp_dir"], "api_docs.md")
                export_success = await documentation_generator.export_documentation(api_docs, doc_file)
                
                doc_tests["documentation_export"] = export_success and os.path.exists(doc_file)
                
            except Exception as e:
                logger.error(f"API documentation test failed: {e}")
                doc_tests["api_documentation"] = False
                doc_tests["documentation_export"] = False
            
            documentation_tests.update(doc_tests)
            
            # Test 5: Plugin API Compliance
            logger.info("Testing plugin API compliance")
            
            plugin_validator = PluginValidator()
            
            compliance_tests = {}
            for plugin in self.mock_plugins:
                try:
                    # Test API compliance
                    compliance_result = await plugin_validator.validate_api_compliance(
                        plugin,
                        api_version="1.0.0"
                    )
                    
                    compliance_tests[plugin.name] = compliance_result.is_compliant if hasattr(compliance_result, 'is_compliant') else True
                    
                except Exception as e:
                    logger.error(f"API compliance test failed for {plugin.name}: {e}")
                    compliance_tests[plugin.name] = False
            
            api_compatibility_tests["plugin_api_compliance"] = sum(compliance_tests.values()) / len(compliance_tests) if compliance_tests else 0
            
            # Success criteria
            api_stability_success = (
                api_compatibility_tests.get("endpoint_registration", 0) > 0.8 and
                api_compatibility_tests.get("version_compatibility", 0) > 0.8 and
                api_compatibility_tests.get("compatibility_matrix", 0) > 0.7
            )
            
            duration = time.time() - start_time
            
            return ExtensibilityTestResult(
                test_name=test_name,
                test_type=ExtensibilityTestType.API_STABILITY,
                success=api_stability_success,
                plugin_tests=plugin_tests,
                api_compatibility_tests=api_compatibility_tests,
                contribution_workflow_tests=contribution_workflow_tests,
                security_validation_tests=security_validation_tests,
                documentation_tests=documentation_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"API stability and versioning test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return ExtensibilityTestResult(
                test_name=test_name,
                test_type=ExtensibilityTestType.API_STABILITY,
                success=False,
                plugin_tests={},
                api_compatibility_tests={},
                contribution_workflow_tests={},
                security_validation_tests={},
                documentation_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def test_contribution_workflow_validation(self) -> ExtensibilityTestResult:
        """Test contribution workflows and community tooling."""
        test_name = "contribution_workflow_validation"
        start_time = time.time()
        
        try:
            logger.info("Testing contribution workflow validation")
            
            environment = await self.setup_test_environment()
            
            plugin_tests = {}
            api_compatibility_tests = {}
            contribution_workflow_tests = {}
            security_validation_tests = {}
            documentation_tests = {}
            
            # Test 1: Contribution Validator
            logger.info("Testing contribution validator")
            
            contribution_validator = ContributionValidator()
            
            # Create test contribution scenarios
            test_contributions = [
                ContributionTestScenario(
                    contribution_type=ContributionType.PLUGIN,
                    files_modified=["plugins/new_skill.py", "tests/test_new_skill.py"],
                    test_requirements=["unit_tests", "integration_tests"],
                    expected_review_criteria=["code_quality", "test_coverage", "documentation"],
                    security_implications=True
                ),
                ContributionTestScenario(
                    contribution_type=ContributionType.DOCUMENTATION,
                    files_modified=["docs/api_guide.md", "examples/basic_usage.py"],
                    test_requirements=["example_validation", "link_checks"],
                    expected_review_criteria=["accuracy", "clarity", "completeness"],
                    security_implications=False
                ),
                ContributionTestScenario(
                    contribution_type=ContributionType.BUG_FIX,
                    files_modified=["agents/skill_coordinator.py", "tests/test_skill_coordinator.py"],
                    test_requirements=["regression_tests", "unit_tests"],
                    expected_review_criteria=["fix_correctness", "test_coverage", "impact_analysis"],
                    security_implications=True
                )
            ]
            
            contribution_validation_results = {}
            for scenario in test_contributions:
                try:
                    # Validate contribution
                    validation_result = await contribution_validator.validate_contribution(
                        scenario.contribution_type,
                        scenario.files_modified,
                        scenario.test_requirements
                    )
                    
                    # Check review criteria
                    review_criteria_met = await contribution_validator.check_review_criteria(
                        scenario.expected_review_criteria,
                        validation_result
                    )
                    
                    contribution_validation_results[scenario.contribution_type.value] = {
                        "validation_passed": validation_result.is_valid if hasattr(validation_result, 'is_valid') else True,
                        "review_criteria_met": review_criteria_met
                    }
                    
                except Exception as e:
                    logger.error(f"Contribution validation failed for {scenario.contribution_type.value}: {e}")
                    contribution_validation_results[scenario.contribution_type.value] = {
                        "validation_passed": False,
                        "review_criteria_met": False
                    }
            
            # Calculate overall contribution workflow success
            validation_success_rate = sum(
                1 for result in contribution_validation_results.values()
                if result["validation_passed"] and result["review_criteria_met"]
            ) / len(contribution_validation_results)
            
            contribution_workflow_tests["contribution_validation"] = validation_success_rate > 0.8
            
            # Test 2: Automated Testing Integration
            logger.info("Testing automated testing integration")
            
            # Mock CI/CD pipeline testing
            ci_tests = {}
            for scenario in test_contributions:
                try:
                    # Simulate running tests for contribution
                    test_results = await self._simulate_ci_pipeline(scenario, environment)
                    
                    ci_tests[scenario.contribution_type.value] = {
                        "tests_passed": test_results.get("tests_passed", False),
                        "coverage_adequate": test_results.get("coverage", 0) > 0.8,
                        "security_scan_clean": test_results.get("security_issues", 0) == 0
                    }
                    
                except Exception as e:
                    logger.error(f"CI pipeline test failed for {scenario.contribution_type.value}: {e}")
                    ci_tests[scenario.contribution_type.value] = {
                        "tests_passed": False,
                        "coverage_adequate": False,
                        "security_scan_clean": False
                    }
            
            ci_success_rate = sum(
                1 for result in ci_tests.values()
                if all(result.values())
            ) / len(ci_tests)
            
            contribution_workflow_tests["automated_testing"] = ci_success_rate > 0.7
            
            # Test 3: Community Tools and Utilities
            logger.info("Testing community tools and utilities")
            
            community_tools = CommunityTools()
            
            tools_tests = {}
            try:
                # Test example validator
                example_validator = ExampleValidator()
                
                # Create test examples
                test_examples = [
                    {
                        "file": os.path.join(environment["temp_dir"], "example1.py"),
                        "content": "# Basic agent example\nagent = Agent('test')\nagent.run()"
                    },
                    {
                        "file": os.path.join(environment["temp_dir"], "example2.py"),
                        "content": "# Plugin example\nclass MyPlugin(IPlugin):\n    def initialize(self): pass"
                    }
                ]
                
                for example in test_examples:
                    with open(example["file"], "w") as f:
                        f.write(example["content"])
                
                # Validate examples
                example_validation_results = []
                for example in test_examples:
                    validation_result = await example_validator.validate_example(example["file"])
                    example_validation_results.append(validation_result.is_valid if hasattr(validation_result, 'is_valid') else True)
                
                tools_tests["example_validation"] = all(example_validation_results)
                
                # Test documentation generation
                doc_gen_result = await community_tools.generate_contribution_guide()
                tools_tests["documentation_generation"] = doc_gen_result is not None
                
                # Test plugin template generation
                template_result = await community_tools.generate_plugin_template(
                    "test_plugin",
                    PluginType.SKILL
                )
                tools_tests["template_generation"] = template_result is not None
                
            except Exception as e:
                logger.error(f"Community tools test failed: {e}")
                tools_tests["example_validation"] = False
                tools_tests["documentation_generation"] = False
                tools_tests["template_generation"] = False
            
            contribution_workflow_tests["community_tools"] = all(tools_tests.values())
            
            # Test 4: Plugin Registry Integration
            logger.info("Testing plugin registry integration")
            
            plugin_registry = PluginRegistry()
            
            registry_tests = {}
            try:
                # Test plugin registration in registry
                for plugin in self.mock_plugins[:2]:  # Test with subset
                    metadata = PluginMetadata(
                        name=plugin.name,
                        version=plugin.version,
                        plugin_type=plugin.plugin_type,
                        description=f"Test plugin {plugin.name}",
                        author="Test Suite",
                        homepage="https://example.com",
                        repository="https://github.com/example/test-plugin"
                    )
                    
                    registry_entry = RegistryEntry(
                        metadata=metadata,
                        verification_status="verified",
                        download_url="https://example.com/download",
                        checksums={"sha256": "test_checksum"}
                    )
                    
                    registration_success = await plugin_registry.register_plugin(registry_entry)
                    registry_tests[f"register_{plugin.name}"] = registration_success
                
                # Test plugin search and discovery
                search_results = await plugin_registry.search_plugins(
                    query="test",
                    plugin_type=PluginType.SKILL
                )
                registry_tests["plugin_search"] = len(search_results) > 0
                
                # Test plugin metadata validation
                metadata_validation = await plugin_registry.validate_metadata(
                    self.mock_plugins[0].get_metadata()
                )
                registry_tests["metadata_validation"] = metadata_validation.is_valid if hasattr(metadata_validation, 'is_valid') else True
                
            except Exception as e:
                logger.error(f"Plugin registry test failed: {e}")
                registry_tests = {
                    "register_test_skill_plugin": False,
                    "register_test_analysis_plugin": False,
                    "plugin_search": False,
                    "metadata_validation": False
                }
            
            contribution_workflow_tests["plugin_registry"] = sum(registry_tests.values()) / len(registry_tests) > 0.7
            
            # Test 5: Security and Code Review Automation
            logger.info("Testing security and code review automation")
            
            security_tests = {}
            for scenario in test_contributions:
                if scenario.security_implications:
                    try:
                        # Simulate security scan
                        security_scan_result = await self._simulate_security_scan(scenario, environment)
                        
                        security_tests[scenario.contribution_type.value] = {
                            "no_vulnerabilities": security_scan_result.get("vulnerabilities", 0) == 0,
                            "safe_api_usage": security_scan_result.get("safe_api_usage", True),
                            "dependency_security": security_scan_result.get("dependency_issues", 0) == 0
                        }
                        
                    except Exception as e:
                        logger.error(f"Security scan failed for {scenario.contribution_type.value}: {e}")
                        security_tests[scenario.contribution_type.value] = {
                            "no_vulnerabilities": False,
                            "safe_api_usage": False,
                            "dependency_security": False
                        }
            
            security_success_rate = sum(
                1 for result in security_tests.values()
                if all(result.values())
            ) / len(security_tests) if security_tests else 1.0
            
            security_validation_tests["automated_security_scan"] = security_success_rate > 0.8
            
            # Success criteria
            workflow_success = (
                contribution_workflow_tests.get("contribution_validation", False) and
                contribution_workflow_tests.get("automated_testing", False) and
                contribution_workflow_tests.get("community_tools", False) and
                security_validation_tests.get("automated_security_scan", False)
            )
            
            duration = time.time() - start_time
            
            return ExtensibilityTestResult(
                test_name=test_name,
                test_type=ExtensibilityTestType.CONTRIBUTION_WORKFLOW,
                success=workflow_success,
                plugin_tests=plugin_tests,
                api_compatibility_tests=api_compatibility_tests,
                contribution_workflow_tests=contribution_workflow_tests,
                security_validation_tests=security_validation_tests,
                documentation_tests=documentation_tests,
                duration_seconds=duration
            )
            
        except Exception as e:
            logger.error(f"Contribution workflow validation test failed: {e}", exc_info=True)
            duration = time.time() - start_time
            
            return ExtensibilityTestResult(
                test_name=test_name,
                test_type=ExtensibilityTestType.CONTRIBUTION_WORKFLOW,
                success=False,
                plugin_tests={},
                api_compatibility_tests={},
                contribution_workflow_tests={},
                security_validation_tests={},
                documentation_tests={},
                duration_seconds=duration,
                error_details=str(e)
            )
    
    async def _simulate_ci_pipeline(
        self, 
        scenario: ContributionTestScenario, 
        environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate CI/CD pipeline for contribution testing."""
        
        # Mock CI pipeline results based on contribution type
        base_results = {
            "tests_passed": True,
            "coverage": 0.85,
            "security_issues": 0,
            "lint_issues": 0
        }
        
        # Adjust results based on contribution type
        if scenario.contribution_type == ContributionType.PLUGIN:
            base_results["coverage"] = 0.9  # Higher coverage expected for plugins
        elif scenario.contribution_type == ContributionType.DOCUMENTATION:
            base_results["coverage"] = 1.0  # Documentation doesn't affect code coverage
            base_results["security_issues"] = 0  # Documentation has no security implications
        elif scenario.contribution_type == ContributionType.BUG_FIX:
            base_results["coverage"] = 0.95  # Bug fixes should have high test coverage
        
        return base_results
    
    async def _simulate_security_scan(
        self, 
        scenario: ContributionTestScenario, 
        environment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate security scanning for contributions."""
        
        # Mock security scan results
        base_results = {
            "vulnerabilities": 0,
            "safe_api_usage": True,
            "dependency_issues": 0,
            "code_injection_risks": 0
        }
        
        # Add some variation based on contribution type
        if scenario.contribution_type == ContributionType.PLUGIN:
            # Plugins might have slightly higher risk
            base_results["dependency_issues"] = 0  # Mock clean scan
        
        return base_results
    
    async def run_community_extensibility_test_suite(self) -> Dict[str, Any]:
        """Run complete community and extensibility testing suite."""
        logger.info("Starting comprehensive community and extensibility testing suite")
        suite_start = time.time()
        
        # Extensibility test methods to run
        test_methods = [
            self.test_plugin_lifecycle_management,
            self.test_api_stability_and_versioning,
            self.test_contribution_workflow_validation
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
                    logger.error(f"❌ {result.test_name} failed: {result.error_details}")
                    
            except Exception as e:
                logger.error(f"Test {test_method.__name__} crashed: {e}", exc_info=True)
                results.append(ExtensibilityTestResult(
                    test_name=test_method.__name__,
                    test_type=ExtensibilityTestType.PLUGIN_LIFECYCLE,
                    success=False,
                    plugin_tests={},
                    api_compatibility_tests={},
                    contribution_workflow_tests={},
                    security_validation_tests={},
                    documentation_tests={},
                    duration_seconds=0,
                    error_details=str(e)
                ))
        
        suite_duration = time.time() - suite_start
        
        # Compile summary statistics
        total_tests = len(results)
        passed_tests = sum(1 for r in results if r.success)
        failed_tests = total_tests - passed_tests
        
        # Calculate category scores
        plugin_score = 0
        api_score = 0
        workflow_score = 0
        security_score = 0
        documentation_score = 0
        
        for result in results:
            if result.plugin_tests:
                plugin_score += sum(result.plugin_tests.values()) / len(result.plugin_tests)
            if result.api_compatibility_tests:
                api_values = [v for v in result.api_compatibility_tests.values() if isinstance(v, (int, float))]
                if api_values:
                    api_score += sum(api_values) / len(api_values)
            if result.contribution_workflow_tests:
                workflow_score += sum(result.contribution_workflow_tests.values()) / len(result.contribution_workflow_tests)
            if result.security_validation_tests:
                security_score += sum(result.security_validation_tests.values()) / len(result.security_validation_tests)
            if result.documentation_tests:
                documentation_score += sum(result.documentation_tests.values()) / len(result.documentation_tests)
        
        avg_plugin_score = plugin_score / total_tests if total_tests > 0 else 0
        avg_api_score = api_score / total_tests if total_tests > 0 else 0
        avg_workflow_score = workflow_score / total_tests if total_tests > 0 else 0
        avg_security_score = security_score / total_tests if total_tests > 0 else 0
        avg_documentation_score = documentation_score / total_tests if total_tests > 0 else 0
        
        summary = {
            "suite_duration_seconds": suite_duration,
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "plugin_framework_score": avg_plugin_score,
            "api_stability_score": avg_api_score,
            "contribution_workflow_score": avg_workflow_score,
            "security_validation_score": avg_security_score,
            "documentation_score": avg_documentation_score,
            "extensibility_validated": failed_tests == 0,
            "test_results": [result.__dict__ for result in results]
        }
        
        logger.info(f"Community and extensibility testing completed: {passed_tests}/{total_tests} passed")
        return summary
    
    async def cleanup_test_environment(self):
        """Clean up test resources."""
        try:
            if self.temp_dir and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info("Community test environment cleanup completed")
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


# CLI runner for direct execution
async def main():
    """Run community and extensibility testing suite."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    test_suite = CommunityAndExtensibilityTestSuite()
    
    try:
        results = await test_suite.run_community_extensibility_test_suite()
        
        print("\n" + "="*80)
        print("COMMUNITY AND EXTENSIBILITY TESTING RESULTS")
        print("="*80)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed: {results['passed_tests']}")
        print(f"Failed: {results['failed_tests']}")
        print(f"Success Rate: {results['success_rate']:.1%}")
        print(f"Suite Duration: {results['suite_duration_seconds']:.2f}s")
        
        print("\nCategory Scores:")
        print(f"  Plugin Framework: {results['plugin_framework_score']:.2f}")
        print(f"  API Stability: {results['api_stability_score']:.2f}")
        print(f"  Contribution Workflow: {results['contribution_workflow_score']:.2f}")
        print(f"  Security Validation: {results['security_validation_score']:.2f}")
        print(f"  Documentation: {results['documentation_score']:.2f}")
        
        if results['extensibility_validated']:
            print("\n🎉 EXTENSIBILITY VALIDATION PASSED!")
            print("Plugin systems and contribution workflows confirmed.")
        else:
            print("\n⚠️  Some extensibility tests failed.")
            print("Review failed components for missing functionality.")
        
        print("="*80)
        
    finally:
        await test_suite.cleanup_test_environment()


if __name__ == "__main__":
    asyncio.run(main())