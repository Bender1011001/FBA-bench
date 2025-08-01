"""
Requirements Validation Matrix for FBA-Bench

Maps each Key Issues requirement to specific tests, ensuring complete coverage
and traceability from requirements to validation.
"""

import asyncio
import logging
import json
import yaml
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from pathlib import Path
import sys
from enum import Enum

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class RequirementStatus(Enum):
    """Status of requirement validation."""
    VALIDATED = "validated"
    PARTIALLY_VALIDATED = "partially_validated"
    NOT_VALIDATED = "not_validated"
    NOT_TESTABLE = "not_testable"


class RequirementPriority(Enum):
    """Priority level of requirements."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class TestMapping:
    """Maps a requirement to specific tests."""
    test_file: str
    test_class: str
    test_method: str
    validation_type: str  # "direct", "indirect", "integration"
    coverage_percentage: float  # 0.0 to 1.0


@dataclass
class Requirement:
    """Individual requirement from Key Issues document."""
    requirement_id: str
    category: str
    title: str
    description: str
    acceptance_criteria: List[str]
    priority: RequirementPriority
    test_mappings: List[TestMapping] = field(default_factory=list)
    validation_status: RequirementStatus = RequirementStatus.NOT_VALIDATED
    notes: str = ""


@dataclass
class RequirementCategory:
    """Category of related requirements."""
    category_id: str
    name: str
    description: str
    requirements: List[Requirement] = field(default_factory=list)


@dataclass
class ValidationMatrix:
    """Complete requirements validation matrix."""
    matrix_id: str
    creation_date: str
    categories: List[RequirementCategory] = field(default_factory=list)
    overall_coverage: float = 0.0
    validation_summary: Dict[str, Any] = field(default_factory=dict)


class RequirementsValidationMatrix:
    """
    Comprehensive requirements validation matrix for FBA-Bench.
    
    Maps all Key Issues requirements to specific tests and validates coverage.
    """
    
    def __init__(self):
        self.matrix = self._initialize_requirements_matrix()
        
    def _initialize_requirements_matrix(self) -> ValidationMatrix:
        """Initialize the complete requirements validation matrix."""
        
        # Category 1: Cognitive Loop Improvements
        cognitive_requirements = [
            Requirement(
                requirement_id="COG-001",
                category="cognitive_loops",
                title="Structured Reflection Loop Implementation",
                description="Implement structured reflection loops that allow agents to learn from past decisions",
                acceptance_criteria=[
                    "Agents can trigger reflection based on time intervals or major events",
                    "Reflection incorporates memory consolidation and strategic adjustment",
                    "Reflection outcomes influence future decision-making"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_cognitive_loop_completeness",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/integration/test_comprehensive_integration.py",
                        test_class="ComprehensiveIntegrationTests",
                        test_method="test_cognitive_multi_skill_integration",
                        validation_type="integration",
                        coverage_percentage=0.7
                    )
                ]
            ),
            Requirement(
                requirement_id="COG-002",
                category="cognitive_loops",
                title="Hierarchical Planning System",
                description="Implement strategic and tactical planning layers with clear hierarchies",
                acceptance_criteria=[
                    "Strategic planning generates high-level objectives",
                    "Tactical planning translates objectives into actionable steps",
                    "Planning adapts to changing market conditions"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite", 
                        test_method="validate_cognitive_loop_completeness",
                        validation_type="direct",
                        coverage_percentage=0.85
                    )
                ]
            ),
            Requirement(
                requirement_id="COG-003",
                category="cognitive_loops",
                title="Memory Integration and Consolidation",
                description="Integrate dual memory systems with consolidation algorithms",
                acceptance_criteria=[
                    "Short-term and long-term memory systems work together",
                    "Memory consolidation preserves important experiences",
                    "Memory retrieval supports decision-making processes"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_cognitive_loop_completeness",
                        validation_type="direct",
                        coverage_percentage=0.8
                    ),
                    TestMapping(
                        test_file="tests/regression/regression_tests.py",
                        test_class="RegressionTestSuite",
                        test_method="test_data_consistency_regression",
                        validation_type="indirect",
                        coverage_percentage=0.6
                    )
                ]
            )
        ]
        
        # Category 2: Multi-Skill Agent Capabilities
        multi_skill_requirements = [
            Requirement(
                requirement_id="MSK-001",
                category="multi_skill",
                title="Event-Driven Skill Coordination",
                description="Implement event-driven triggers for multi-skill coordination beyond basic pricing",
                acceptance_criteria=[
                    "Multiple skills can respond to the same event",
                    "Skill coordination prevents conflicts and redundancy",
                    "Skills can collaborate on complex scenarios"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_multi_skill_coordination",
                        validation_type="direct",
                        coverage_percentage=0.95
                    ),
                    TestMapping(
                        test_file="tests/integration/test_comprehensive_integration.py",
                        test_class="ComprehensiveIntegrationTests",
                        test_method="test_skill_coordination_under_load",
                        validation_type="integration",
                        coverage_percentage=0.8
                    )
                ]
            ),
            Requirement(
                requirement_id="MSK-002",
                category="multi_skill",
                title="Comprehensive Skill Module Coverage",
                description="Implement skill modules for supply management, marketing, customer service, and financial analysis",
                acceptance_criteria=[
                    "Supply management skill handles inventory and sourcing",
                    "Marketing skill manages campaigns and customer acquisition",
                    "Customer service skill handles support interactions",
                    "Financial analysis skill provides business insights"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_multi_skill_coordination",
                        validation_type="direct",
                        coverage_percentage=0.9
                    )
                ]
            ),
            Requirement(
                requirement_id="MSK-003",
                category="multi_skill",
                title="Priority-Based Coordination Strategy",
                description="Implement priority-based coordination with resource allocation",
                acceptance_criteria=[
                    "Skills have configurable priority levels",
                    "Higher priority skills get resource preference",
                    "Coordination prevents deadlocks and conflicts"
                ],
                priority=RequirementPriority.MEDIUM,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_multi_skill_coordination",
                        validation_type="direct",
                        coverage_percentage=0.7
                    )
                ]
            )
        ]
        
        # Category 3: Reproducibility and Determinism
        reproducibility_requirements = [
            Requirement(
                requirement_id="REP-001",
                category="reproducibility",
                title="LLM Response Caching System",
                description="Implement comprehensive LLM response caching for deterministic behavior",
                acceptance_criteria=[
                    "LLM responses are cached with content-based hashing",
                    "Cache integrity is validated and maintained",
                    "Deterministic mode ensures bit-perfect reproduction"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_reproducibility_guarantees",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/regression/regression_tests.py",
                        test_class="RegressionTestSuite",
                        test_method="test_golden_master_regression",
                        validation_type="integration",
                        coverage_percentage=0.8
                    )
                ]
            ),
            Requirement(
                requirement_id="REP-002",
                category="reproducibility",
                title="Simulation Seed Management",
                description="Implement robust seed management for deterministic simulations",
                acceptance_criteria=[
                    "Seeds control all random number generation",
                    "Same seed produces identical simulation results",
                    "Seed management supports parallel execution"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_reproducibility_guarantees",
                        validation_type="direct",
                        coverage_percentage=0.85
                    )
                ]
            ),
            Requirement(
                requirement_id="REP-003",
                category="reproducibility",
                title="Golden Master Validation",
                description="Implement golden master testing for regression detection",
                acceptance_criteria=[
                    "Golden masters capture simulation state snapshots",
                    "Validation detects changes in simulation behavior",
                    "Golden masters support version control workflows"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_reproducibility_guarantees",
                        validation_type="direct",
                        coverage_percentage=0.8
                    ),
                    TestMapping(
                        test_file="tests/regression/regression_tests.py",
                        test_class="RegressionTestSuite",
                        test_method="test_golden_master_regression",
                        validation_type="direct",
                        coverage_percentage=0.95
                    )
                ]
            )
        ]
        
        # Category 4: Infrastructure Scalability
        infrastructure_requirements = [
            Requirement(
                requirement_id="INF-001",
                category="infrastructure",
                title="LLM Request Batching",
                description="Implement LLM request batching for cost and performance optimization",
                acceptance_criteria=[
                    "Multiple LLM requests are batched together",
                    "Batching reduces overall API costs by 30%+",
                    "Batch processing maintains response quality"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/benchmarks/performance_benchmarks.py",
                        test_class="PerformanceBenchmarkSuite",
                        test_method="benchmark_llm_batching_efficiency",
                        validation_type="direct",
                        coverage_percentage=0.9
                    )
                ]
            ),
            Requirement(
                requirement_id="INF-002",
                category="infrastructure",
                title="Distributed Agent Simulation",
                description="Support 20+ agents running simultaneously with coordination",
                acceptance_criteria=[
                    "System supports 20+ concurrent agents",
                    "Agent coordination prevents conflicts",
                    "Performance scales linearly with agent count"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/benchmarks/performance_benchmarks.py",
                        test_class="PerformanceBenchmarkSuite",
                        test_method="benchmark_distributed_agent_performance",
                        validation_type="direct",
                        coverage_percentage=0.95
                    ),
                    TestMapping(
                        test_file="tests/integration/test_comprehensive_integration.py",
                        test_class="ComprehensiveIntegrationTests",
                        test_method="test_scalability_with_determinism",
                        validation_type="integration",
                        coverage_percentage=0.8
                    )
                ]
            ),
            Requirement(
                requirement_id="INF-003",
                category="infrastructure",
                title="Performance Monitoring and Optimization",
                description="Implement comprehensive performance monitoring with 2000+ ticks/minute",
                acceptance_criteria=[
                    "System achieves 2000+ simulation ticks per minute",
                    "Performance monitoring tracks resource usage",
                    "Optimization maintains quality while improving speed"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/benchmarks/performance_benchmarks.py",
                        test_class="PerformanceBenchmarkSuite",
                        test_method="benchmark_simulation_throughput",
                        validation_type="direct",
                        coverage_percentage=0.9
                    )
                ]
            )
        ]
        
        # Category 5: Scenario Diversity and Curriculum Validation
        scenario_requirements = [
            Requirement(
                requirement_id="SCN-001",
                category="scenarios",
                title="Tier-Based Curriculum System",
                description="Implement T0-T3 tier progression with increasing complexity",
                acceptance_criteria=[
                    "T0 scenarios test basic single-agent capabilities",
                    "T1 scenarios test advanced single-agent scenarios",
                    "T2 scenarios test multi-agent competitive scenarios",
                    "T3 scenarios test complex market dynamics"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/curriculum/scenario_curriculum_tests.py",
                        test_class="ScenarioAndCurriculumTestSuite",
                        test_method="test_tier_0_scenarios",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/curriculum/scenario_curriculum_tests.py",
                        test_class="ScenarioAndCurriculumTestSuite",
                        test_method="test_tier_1_scenarios",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/curriculum/scenario_curriculum_tests.py",
                        test_class="ScenarioAndCurriculumTestSuite",
                        test_method="test_tier_2_scenarios",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/curriculum/scenario_curriculum_tests.py",
                        test_class="ScenarioAndCurriculumTestSuite",
                        test_method="test_tier_3_scenarios",
                        validation_type="direct",
                        coverage_percentage=0.9
                    )
                ]
            ),
            Requirement(
                requirement_id="SCN-002",
                category="scenarios",
                title="Multi-Agent Scenario Orchestration",
                description="Implement complex multi-agent competitive and collaborative scenarios",
                acceptance_criteria=[
                    "Scenarios support 2-4 agents with different roles",
                    "Agent interactions are coordinated and realistic",
                    "Competitive dynamics are properly simulated"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/curriculum/scenario_curriculum_tests.py",
                        test_class="ScenarioAndCurriculumTestSuite",
                        test_method="test_tier_2_scenarios",
                        validation_type="direct",
                        coverage_percentage=0.8
                    )
                ]
            ),
            Requirement(
                requirement_id="SCN-003",
                category="scenarios",
                title="Curriculum Progression Validation",
                description="Validate agent progression through curriculum tiers",
                acceptance_criteria=[
                    "Agents demonstrate skill improvement across tiers",
                    "Progression criteria are measurable and objective",
                    "Curriculum adapts to agent performance"
                ],
                priority=RequirementPriority.MEDIUM,
                test_mappings=[
                    TestMapping(
                        test_file="tests/curriculum/scenario_curriculum_tests.py",
                        test_class="ScenarioAndCurriculumTestSuite",
                        test_method="test_curriculum_progression",
                        validation_type="direct",
                        coverage_percentage=0.85
                    )
                ]
            )
        ]
        
        # Category 6: Tool Interfaces and Observability
        observability_requirements = [
            Requirement(
                requirement_id="OBS-001",
                category="observability",
                title="Comprehensive Trace Analysis",
                description="Implement detailed trace analysis for debugging and optimization",
                acceptance_criteria=[
                    "All agent actions and decisions are traced",
                    "Trace data supports performance analysis",
                    "Debugging tools help identify issues quickly"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_observability_insights",
                        validation_type="direct",
                        coverage_percentage=0.8
                    )
                ]
            ),
            Requirement(
                requirement_id="OBS-002",
                category="observability",
                title="Real-Time Alert System",
                description="Implement alerting for performance degradation and errors",
                acceptance_criteria=[
                    "Alerts trigger on performance thresholds",
                    "Error patterns are automatically detected",
                    "Alert system integrates with monitoring dashboards"
                ],
                priority=RequirementPriority.MEDIUM,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_observability_insights",
                        validation_type="direct",
                        coverage_percentage=0.7
                    )
                ]
            ),
            Requirement(
                requirement_id="OBS-003",
                category="observability",
                title="LLM-Friendly API Design",
                description="Design APIs that are easily consumable by LLM agents",
                acceptance_criteria=[
                    "APIs have clear, consistent interfaces",
                    "Error messages are descriptive and actionable",
                    "API documentation is comprehensive"
                ],
                priority=RequirementPriority.MEDIUM,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_observability_insights",
                        validation_type="indirect",
                        coverage_percentage=0.6
                    ),
                    TestMapping(
                        test_file="tests/regression/regression_tests.py",
                        test_class="RegressionTestSuite",
                        test_method="test_api_contract_validation_regression",
                        validation_type="direct",
                        coverage_percentage=0.8
                    )
                ]
            )
        ]
        
        # Category 7: Missing Features
        missing_features_requirements = [
            Requirement(
                requirement_id="MIS-001",
                category="missing_features",
                title="Agent Learning and Adaptation",
                description="Implement episodic learning with evaluation separation",
                acceptance_criteria=[
                    "Agents learn from simulation experiences",
                    "Learning data is isolated from evaluation data",
                    "Learning improves agent performance over time"
                ],
                priority=RequirementPriority.HIGH,
                test_mappings=[
                    TestMapping(
                        test_file="tests/validation/functional_validation.py",
                        test_class="FunctionalValidationSuite",
                        test_method="validate_learning_system_safety",
                        validation_type="direct",
                        coverage_percentage=0.85
                    )
                ]
            ),
            Requirement(
                requirement_id="MIS-002",
                category="missing_features",
                title="Real-World Integration Capabilities",
                description="Implement safe real-world integration with sandbox testing",
                acceptance_criteria=[
                    "Simulation-to-sandbox consistency is maintained",
                    "Safety constraints prevent harmful actions",
                    "Gradual rollout mechanisms are implemented"
                ],
                priority=RequirementPriority.CRITICAL,
                test_mappings=[
                    TestMapping(
                        test_file="tests/integration/real_world_integration_tests.py",
                        test_class="RealWorldIntegrationTestSuite",
                        test_method="test_simulation_to_sandbox_consistency",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/integration/real_world_integration_tests.py",
                        test_class="RealWorldIntegrationTestSuite",
                        test_method="test_safety_constraints_and_guardrails",
                        validation_type="direct",
                        coverage_percentage=0.9
                    ),
                    TestMapping(
                        test_file="tests/integration/real_world_integration_tests.py",
                        test_class="RealWorldIntegrationTestSuite",
                        test_method="test_gradual_rollout_mechanisms",
                        validation_type="direct",
                        coverage_percentage=0.85
                    )
                ]
            ),
            Requirement(
                requirement_id="MIS-003",
                category="missing_features",
                title="Community Extensibility Framework",
                description="Implement plugin system and contribution workflows",
                acceptance_criteria=[
                    "Plugin framework supports multiple plugin types",
                    "Contribution workflows are validated and secure",
                    "API stability is maintained across versions"
                ],
                priority=RequirementPriority.MEDIUM,
                test_mappings=[
                    TestMapping(
                        test_file="tests/community/extensibility_tests.py",
                        test_class="CommunityAndExtensibilityTestSuite",
                        test_method="test_plugin_lifecycle_management",
                        validation_type="direct",
                        coverage_percentage=0.8
                    ),
                    TestMapping(
                        test_file="tests/community/extensibility_tests.py",
                        test_class="CommunityAndExtensibilityTestSuite",
                        test_method="test_api_stability_and_versioning",
                        validation_type="direct",
                        coverage_percentage=0.8
                    ),
                    TestMapping(
                        test_file="tests/community/extensibility_tests.py",
                        test_class="CommunityAndExtensibilityTestSuite",
                        test_method="test_contribution_workflow_validation",
                        validation_type="direct",
                        coverage_percentage=0.75
                    )
                ]
            )
        ]
        
        # Create categories
        categories = [
            RequirementCategory(
                category_id="cognitive_loops",
                name="Cognitive Loop Improvements",
                description="Structured reflection, hierarchical planning, and memory integration",
                requirements=cognitive_requirements
            ),
            RequirementCategory(
                category_id="multi_skill",
                name="Multi-Skill Agent Capabilities",
                description="Event-driven multi-skill coordination beyond basic pricing",
                requirements=multi_skill_requirements
            ),
            RequirementCategory(
                category_id="reproducibility",
                name="Reproducibility and Determinism",
                description="LLM caching, seed management, and golden master validation",
                requirements=reproducibility_requirements
            ),
            RequirementCategory(
                category_id="infrastructure",
                name="Infrastructure Scalability",
                description="Batching, distributed simulation, and performance optimization",
                requirements=infrastructure_requirements
            ),
            RequirementCategory(
                category_id="scenarios",
                name="Scenario Diversity and Curriculum Validation",
                description="Tier-based curriculum with multi-agent scenarios",
                requirements=scenario_requirements
            ),
            RequirementCategory(
                category_id="observability",
                name="Tool Interfaces and Observability",
                description="Trace analysis, alerting, and LLM-friendly APIs",
                requirements=observability_requirements
            ),
            RequirementCategory(
                category_id="missing_features",
                name="Missing Features Implementation",
                description="Agent learning, real-world integration, and community extensibility",
                requirements=missing_features_requirements
            )
        ]
        
        return ValidationMatrix(
            matrix_id="fba_bench_requirements_v1",
            creation_date=datetime.now().isoformat(),
            categories=categories
        )
    
    def calculate_validation_coverage(self) -> None:
        """Calculate validation coverage for all requirements."""
        total_requirements = 0
        validated_requirements = 0
        total_coverage_score = 0.0
        
        for category in self.matrix.categories:
            for requirement in category.requirements:
                total_requirements += 1
                
                if requirement.test_mappings:
                    # Calculate weighted coverage based on validation types
                    coverage_scores = []
                    for mapping in requirement.test_mappings:
                        weight = {
                            "direct": 1.0,
                            "integration": 0.8,
                            "indirect": 0.6
                        }.get(mapping.validation_type, 0.5)
                        
                        coverage_scores.append(mapping.coverage_percentage * weight)
                    
                    # Take the maximum coverage score
                    max_coverage = max(coverage_scores) if coverage_scores else 0.0
                    total_coverage_score += max_coverage
                    
                    # Determine validation status
                    if max_coverage >= 0.8:
                        requirement.validation_status = RequirementStatus.VALIDATED
                        validated_requirements += 1
                    elif max_coverage >= 0.5:
                        requirement.validation_status = RequirementStatus.PARTIALLY_VALIDATED
                    else:
                        requirement.validation_status = RequirementStatus.NOT_VALIDATED
                else:
                    requirement.validation_status = RequirementStatus.NOT_VALIDATED
        
        # Calculate overall coverage
        self.matrix.overall_coverage = total_coverage_score / total_requirements if total_requirements > 0 else 0.0
        
        # Generate validation summary
        self.matrix.validation_summary = {
            "total_requirements": total_requirements,
            "validated_requirements": validated_requirements,
            "partially_validated": sum(1 for cat in self.matrix.categories 
                                     for req in cat.requirements 
                                     if req.validation_status == RequirementStatus.PARTIALLY_VALIDATED),
            "not_validated": sum(1 for cat in self.matrix.categories 
                               for req in cat.requirements 
                               if req.validation_status == RequirementStatus.NOT_VALIDATED),
            "validation_rate": validated_requirements / total_requirements if total_requirements > 0 else 0,
            "overall_coverage_score": self.matrix.overall_coverage
        }
    
    def generate_validation_report(self) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        self.calculate_validation_coverage()
        
        # Category-wise analysis
        category_analysis = {}
        for category in self.matrix.categories:
            validated = sum(1 for req in category.requirements 
                          if req.validation_status == RequirementStatus.VALIDATED)
            total = len(category.requirements)
            
            category_analysis[category.category_id] = {
                "name": category.name,
                "total_requirements": total,
                "validated_requirements": validated,
                "validation_rate": validated / total if total > 0 else 0,
                "requirements": [
                    {
                        "id": req.requirement_id,
                        "title": req.title,
                        "status": req.validation_status.value,
                        "priority": req.priority.value,
                        "test_count": len(req.test_mappings),
                        "max_coverage": max([tm.coverage_percentage for tm in req.test_mappings], default=0.0)
                    }
                    for req in category.requirements
                ]
            }
        
        # Critical requirements analysis
        critical_requirements = []
        for category in self.matrix.categories:
            for req in category.requirements:
                if req.priority == RequirementPriority.CRITICAL:
                    critical_requirements.append({
                        "id": req.requirement_id,
                        "title": req.title,
                        "category": category.name,
                        "status": req.validation_status.value,
                        "test_mappings": len(req.test_mappings)
                    })
        
        critical_validated = sum(1 for req in critical_requirements 
                               if req["status"] == RequirementStatus.VALIDATED.value)
        
        # Test coverage analysis
        test_file_coverage = {}
        for category in self.matrix.categories:
            for req in category.requirements:
                for mapping in req.test_mappings:
                    test_file = mapping.test_file
                    if test_file not in test_file_coverage:
                        test_file_coverage[test_file] = {
                            "requirements_covered": 0,
                            "total_coverage_score": 0.0,
                            "test_methods": set()
                        }
                    
                    test_file_coverage[test_file]["requirements_covered"] += 1
                    test_file_coverage[test_file]["total_coverage_score"] += mapping.coverage_percentage
                    test_file_coverage[test_file]["test_methods"].add(mapping.test_method)
        
        # Convert sets to lists for JSON serialization
        for test_file in test_file_coverage:
            test_file_coverage[test_file]["test_methods"] = list(test_file_coverage[test_file]["test_methods"])
            test_file_coverage[test_file]["average_coverage"] = (
                test_file_coverage[test_file]["total_coverage_score"] / 
                test_file_coverage[test_file]["requirements_covered"]
            )
        
        return {
            "matrix_summary": self.matrix.validation_summary,
            "category_analysis": category_analysis,
            "critical_requirements": {
                "total": len(critical_requirements),
                "validated": critical_validated,
                "validation_rate": critical_validated / len(critical_requirements) if critical_requirements else 0,
                "details": critical_requirements
            },
            "test_coverage_analysis": test_file_coverage,
            "recommendations": self._generate_coverage_recommendations()
        }
    
    def _generate_coverage_recommendations(self) -> List[str]:
        """Generate recommendations for improving validation coverage."""
        recommendations = []
        
        # Check for unvalidated critical requirements
        unvalidated_critical = []
        for category in self.matrix.categories:
            for req in category.requirements:
                if (req.priority == RequirementPriority.CRITICAL and 
                    req.validation_status != RequirementStatus.VALIDATED):
                    unvalidated_critical.append(f"{req.requirement_id}: {req.title}")
        
        if unvalidated_critical:
            recommendations.append(
                f"Critical requirements need additional validation: {', '.join(unvalidated_critical[:3])}"
            )
        
        # Check for categories with low coverage
        for category in self.matrix.categories:
            validated = sum(1 for req in category.requirements 
                          if req.validation_status == RequirementStatus.VALIDATED)
            total = len(category.requirements)
            validation_rate = validated / total if total > 0 else 0
            
            if validation_rate < 0.8:
                recommendations.append(
                    f"Category '{category.name}' has low validation coverage ({validation_rate:.1%}). "
                    f"Consider adding more tests."
                )
        
        # Check for requirements without test mappings
        unmapped_requirements = []
        for category in self.matrix.categories:
            for req in category.requirements:
                if not req.test_mappings:
                    unmapped_requirements.append(f"{req.requirement_id}: {req.title}")
        
        if unmapped_requirements:
            recommendations.append(
                f"Requirements without test mappings: {', '.join(unmapped_requirements[:3])}"
            )
        
        # Overall coverage recommendation
        if self.matrix.overall_coverage < 0.8:
            recommendations.append(
                f"Overall validation coverage is {self.matrix.overall_coverage:.1%}. "
                f"Target should be 80%+ for production readiness."
            )
        
        return recommendations
    
    def save_matrix(self, output_file: str) -> bool:
        """Save validation matrix to file."""
        try:
            self.calculate_validation_coverage()
            
            # Convert matrix to serializable format
            matrix_dict = {
                "matrix_id": self.matrix.matrix_id,
                "creation_date": self.matrix.creation_date,
                "overall_coverage": self.matrix.overall_coverage,
                "validation_summary": self.matrix.validation_summary,
                "categories": []
            }
            
            for category in self.matrix.categories:
                cat_dict = {
                    "category_id": category.category_id,
                    "name": category.name,
                    "description": category.description,
                    "requirements": []
                }
                
                for req in category.requirements:
                    req_dict = {
                        "requirement_id": req.requirement_id,
                        "category": req.category,
                        "title": req.title,
                        "description": req.description,
                        "acceptance_criteria": req.acceptance_criteria,
                        "priority": req.priority.value,
                        "validation_status": req.validation_status.value,
                        "notes": req.notes,
                        "test_mappings": [
                            {
                                "test_file": tm.test_file,
                                "test_class": tm.test_class,
                                "test_method": tm.test_method,
                                "validation_type": tm.validation_type,
                                "coverage_percentage": tm.coverage_percentage
                            }
                            for tm in req.test_mappings
                        ]
                    }
                    cat_dict["requirements"].append(req_dict)
                
                matrix_dict["categories"].append(cat_dict)
            
            # Save to file
            output_path = Path(output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(matrix_dict, f, indent=2)
            
            logger.info(f"Requirements validation matrix saved to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save validation matrix: {e}")
            return False


# CLI runner for direct execution
async def main():
    """Run requirements validation matrix analysis."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize validation matrix
    validator = RequirementsValidationMatrix()
    
    # Generate validation report
    report = validator.generate_validation_report()
    
    # Save matrix to file
    matrix_file = "test_reports/requirements_validation_matrix.json"
    validator.save_matrix(matrix_file)
    
    print("\n" + "="*80)
    print("REQUIREMENTS VALIDATION MATRIX ANALYSIS")
    print("="*80)
    
    summary = report["matrix_summary"]
    print(f"Total Requirements: {summary['total_requirements']}")
    print(f"Validated Requirements: {summary['validated_requirements']}")
    print(f"Validation Rate: {summary['validation_rate']:.1%}")
    print(f"Overall Coverage Score: {summary['overall_coverage_score']:.1%}")
    
    print(f"\nCritical Requirements:")
    critical = report["critical_requirements"]
    print(f"  Total: {critical['total']}")
    print(f"  Validated: {critical['validated']}")
    print(f"  Validation Rate: {critical['validation_rate']:.1%}")
    
    print(f"\nCategory Analysis:")
    for cat_id, analysis in report["category_analysis"].items():
        print(f"  {analysis['name']}: {analysis['validated_requirements']}/{analysis['total_requirements']} "
              f"({analysis['validation_rate']:.1%})")
    
    if report["recommendations"]:
        print(f"\nRecommendations:")
        for i, rec in enumerate(report["recommendations"], 1):
            print(f"  {i}. {rec}")
    
    print(f"\nMatrix saved to: {matrix_file}")
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())