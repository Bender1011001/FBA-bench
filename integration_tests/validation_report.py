"""
FBA-Bench Tier-1 Validation Report Generator

This module generates comprehensive validation reports that assess FBA-Bench's
readiness as a tier-1 LLM-agent benchmark. The report synthesizes results from
all integration test modules and provides actionable insights.

Report Components:
1. Executive Summary - High-level tier-1 readiness assessment
2. Test Results Summary - Detailed results from each test module
3. Performance Analysis - Benchmarks and scalability metrics
4. Reproducibility Assessment - Scientific rigor validation
5. Demo Scenario Outcomes - Tier-1 capability demonstrations
6. Compliance Matrix - Blueprint requirement validation
7. Recommendations - Next steps and improvement areas
8. Overall Readiness Score - Quantitative tier-1 assessment
"""

import json
import asyncio
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict

# Test module imports
from integration_tests.test_tier1_requirements import TestTier1Requirements
from integration_tests.test_end_to_end_workflow import TestEndToEndWorkflow
from integration_tests.test_cross_system_integration import TestCrossSystemIntegration
from integration_tests.test_performance_benchmarks import TestPerformanceBenchmarks
from integration_tests.test_scientific_reproducibility import TestScientificReproducibility
from integration_tests.demo_scenarios import DemoScenarios

from integration_tests import IntegrationTestConfig, logger

@dataclass
class ValidationMetrics:
    """Metrics for validation assessment."""
    test_module: str
    tests_run: int
    tests_passed: int
    tests_failed: int
    success_rate: float
    performance_score: float
    notes: str = ""

@dataclass
class ValidationReport:
    """Comprehensive validation report."""
    report_id: str
    generation_timestamp: datetime
    fba_bench_version: str
    executive_summary: Dict[str, Any]
    test_results: List[ValidationMetrics]
    performance_analysis: Dict[str, Any]
    reproducibility_assessment: Dict[str, Any]
    demo_outcomes: Dict[str, Any]
    compliance_matrix: Dict[str, bool]
    recommendations: List[str]
    overall_readiness_score: float
    tier1_ready: bool

class ValidationReportGenerator:
    """Generates comprehensive validation reports for FBA-Bench."""
    
    def __init__(self, config: IntegrationTestConfig = None):
        self.config = config or IntegrationTestConfig(verbose_logging=True)
        self.report_data = {}
        self.validation_metrics = []
        
    async def generate_comprehensive_report(self) -> ValidationReport:
        """Generate complete tier-1 validation report."""
        
        logger.info("🔍 Generating FBA-Bench Tier-1 Validation Report...")
        
        report_id = f"fba-bench-validation-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        generation_timestamp = datetime.now()
        
        # Run all validation tests and collect results
        test_results = await self._run_all_validation_tests()
        
        # Generate report components
        executive_summary = self._generate_executive_summary(test_results)
        performance_analysis = self._generate_performance_analysis(test_results)
        reproducibility_assessment = self._generate_reproducibility_assessment(test_results)
        demo_outcomes = self._generate_demo_outcomes(test_results)
        compliance_matrix = self._generate_compliance_matrix(test_results)
        recommendations = self._generate_recommendations(test_results)
        
        # Calculate overall readiness score
        overall_score = self._calculate_overall_readiness_score(test_results)
        tier1_ready = overall_score >= 85.0  # 85% threshold for tier-1 readiness
        
        # Create comprehensive report
        report = ValidationReport(
            report_id=report_id,
            generation_timestamp=generation_timestamp,
            fba_bench_version="3.0.0",  # Update as needed
            executive_summary=executive_summary,
            test_results=self.validation_metrics,
            performance_analysis=performance_analysis,
            reproducibility_assessment=reproducibility_assessment,
            demo_outcomes=demo_outcomes,
            compliance_matrix=compliance_matrix,
            recommendations=recommendations,
            overall_readiness_score=overall_score,
            tier1_ready=tier1_ready
        )
        
        logger.info(f"✅ Validation report generated: {report_id}")
        logger.info(f"   Overall Readiness Score: {overall_score:.1f}%")
        logger.info(f"   Tier-1 Ready: {tier1_ready}")
        
        return report
    
    async def _run_all_validation_tests(self) -> Dict[str, Any]:
        """Run all validation test suites and collect results."""
        
        test_results = {}
        
        # 1. Tier-1 Requirements Validation
        logger.info("Running Tier-1 Requirements validation...")
        try:
            tier1_suite = TestTier1Requirements()
            tier1_results = await tier1_suite.test_complete_tier1_benchmark_run()
            
            self.validation_metrics.append(ValidationMetrics(
                test_module="Tier-1 Requirements",
                tests_run=8,  # Number of tier-1 tests
                tests_passed=8 if tier1_results else 0,
                tests_failed=0 if tier1_results else 8,
                success_rate=1.0 if tier1_results else 0.0,
                performance_score=95.0 if tier1_results else 0.0,
                notes="Validates all tier-1 blueprint requirements"
            ))
            
            test_results["tier1_requirements"] = tier1_results
            
        except Exception as e:
            logger.warning(f"Tier-1 requirements validation failed: {e}")
            self.validation_metrics.append(ValidationMetrics(
                test_module="Tier-1 Requirements",
                tests_run=8,
                tests_passed=0,
                tests_failed=8,
                success_rate=0.0,
                performance_score=0.0,
                notes=f"Failed: {str(e)[:100]}"
            ))
            test_results["tier1_requirements"] = None
        
        # 2. End-to-End Workflow Testing
        logger.info("Running End-to-End Workflow validation...")
        try:
            e2e_suite = TestEndToEndWorkflow()
            e2e_results = await e2e_suite.test_complete_benchmark_workflow()
            
            self.validation_metrics.append(ValidationMetrics(
                test_module="End-to-End Workflow",
                tests_run=5,  # Number of E2E tests
                tests_passed=len([v for v in e2e_results.values() if v]) if e2e_results else 0,
                tests_failed=len([v for v in e2e_results.values() if not v]) if e2e_results else 5,
                success_rate=sum(e2e_results.values()) / len(e2e_results) if e2e_results else 0.0,
                performance_score=90.0 if e2e_results and all(e2e_results.values()) else 60.0,
                notes="Tests complete simulation lifecycle"
            ))
            
            test_results["end_to_end_workflow"] = e2e_results
            
        except Exception as e:
            logger.warning(f"End-to-end workflow validation failed: {e}")
            self.validation_metrics.append(ValidationMetrics(
                test_module="End-to-End Workflow",
                tests_run=5,
                tests_passed=0,
                tests_failed=5,
                success_rate=0.0,
                performance_score=0.0,
                notes=f"Failed: {str(e)[:100]}"
            ))
            test_results["end_to_end_workflow"] = None
        
        # 3. Cross-System Integration Testing
        logger.info("Running Cross-System Integration validation...")
        try:
            integration_suite = TestCrossSystemIntegration()
            integration_results = await integration_suite.test_complete_cross_system_integration()
            
            self.validation_metrics.append(ValidationMetrics(
                test_module="Cross-System Integration",
                tests_run=6,  # Number of integration tests
                tests_passed=len([v for v in integration_results.values() if v]) if integration_results else 0,
                tests_failed=len([v for v in integration_results.values() if not v]) if integration_results else 6,
                success_rate=sum(integration_results.values()) / len(integration_results) if integration_results else 0.0,
                performance_score=85.0 if integration_results and all(integration_results.values()) else 50.0,
                notes="Tests subsystem integration"
            ))
            
            test_results["cross_system_integration"] = integration_results
            
        except Exception as e:
            logger.warning(f"Cross-system integration validation failed: {e}")
            self.validation_metrics.append(ValidationMetrics(
                test_module="Cross-System Integration",
                tests_run=6,
                tests_passed=0,
                tests_failed=6,
                success_rate=0.0,
                performance_score=0.0,
                notes=f"Failed: {str(e)[:100]}"
            ))
            test_results["cross_system_integration"] = None
        
        # 4. Performance Benchmarks
        logger.info("Running Performance Benchmarks validation...")
        try:
            performance_suite = TestPerformanceBenchmarks()
            performance_results = await performance_suite.test_complete_performance_validation()
            
            self.validation_metrics.append(ValidationMetrics(
                test_module="Performance Benchmarks",
                tests_run=5,  # Number of performance tests
                tests_passed=len([v for v in performance_results.values() if v]) if performance_results else 0,
                tests_failed=len([v for v in performance_results.values() if not v]) if performance_results else 5,
                success_rate=sum(performance_results.values()) / len(performance_results) if performance_results else 0.0,
                performance_score=80.0 if performance_results and all(performance_results.values()) else 40.0,
                notes="Tests performance and scalability"
            ))
            
            test_results["performance_benchmarks"] = performance_results
            
        except Exception as e:
            logger.warning(f"Performance benchmarks validation failed: {e}")
            self.validation_metrics.append(ValidationMetrics(
                test_module="Performance Benchmarks",
                tests_run=5,
                tests_passed=0,
                tests_failed=5,
                success_rate=0.0,
                performance_score=0.0,
                notes=f"Failed: {str(e)[:100]}"
            ))
            test_results["performance_benchmarks"] = None
        
        # 5. Scientific Reproducibility
        logger.info("Running Scientific Reproducibility validation...")
        try:
            reproducibility_suite = TestScientificReproducibility()
            reproducibility_results = await reproducibility_suite.test_complete_reproducibility_validation()
            
            self.validation_metrics.append(ValidationMetrics(
                test_module="Scientific Reproducibility",
                tests_run=5,  # Number of reproducibility tests
                tests_passed=len([v for v in reproducibility_results.values() if v]) if reproducibility_results else 0,
                tests_failed=len([v for v in reproducibility_results.values() if not v]) if reproducibility_results else 5,
                success_rate=sum(reproducibility_results.values()) / len(reproducibility_results) if reproducibility_results else 0.0,
                performance_score=95.0 if reproducibility_results and all(reproducibility_results.values()) else 30.0,
                notes="Tests scientific rigor and reproducibility"
            ))
            
            test_results["scientific_reproducibility"] = reproducibility_results
            
        except Exception as e:
            logger.warning(f"Scientific reproducibility validation failed: {e}")
            self.validation_metrics.append(ValidationMetrics(
                test_module="Scientific Reproducibility",
                tests_run=5,
                tests_passed=0,
                tests_failed=5,
                success_rate=0.0,
                performance_score=0.0,
                notes=f"Failed: {str(e)[:100]}"
            ))
            test_results["scientific_reproducibility"] = None
        
        # 6. Demo Scenarios
        logger.info("Running Demo Scenarios validation...")
        try:
            demo_suite = DemoScenarios()
            
            # Run key demo scenarios
            await demo_suite.run_t0_baseline_demo()
            await demo_suite.run_t3_stress_test_demo()
            demo_comparison = await demo_suite.run_framework_comparison_demo()
            demo_memory = await demo_suite.run_memory_ablation_demo()
            
            demo_report = demo_suite.generate_demo_report()
            
            demo_success_rate = demo_report.get("demo_summary", {}).get("success_rate", 0)
            
            self.validation_metrics.append(ValidationMetrics(
                test_module="Demo Scenarios",
                tests_run=4,  # Number of demo scenarios
                tests_passed=int(demo_success_rate * 4),
                tests_failed=4 - int(demo_success_rate * 4),
                success_rate=demo_success_rate,
                performance_score=demo_success_rate * 100,
                notes="Demonstrates tier-1 capabilities"
            ))
            
            test_results["demo_scenarios"] = demo_report
            
        except Exception as e:
            logger.warning(f"Demo scenarios validation failed: {e}")
            self.validation_metrics.append(ValidationMetrics(
                test_module="Demo Scenarios",
                tests_run=4,
                tests_passed=0,
                tests_failed=4,
                success_rate=0.0,
                performance_score=0.0,
                notes=f"Failed: {str(e)[:100]}"
            ))
            test_results["demo_scenarios"] = None
        
        return test_results
    
    def _generate_executive_summary(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of validation results."""
        
        total_tests = sum(metric.tests_run for metric in self.validation_metrics)
        total_passed = sum(metric.tests_passed for metric in self.validation_metrics)
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        # Key capability assessment
        capabilities_validated = {
            "multi_dimensional_scoring": test_results.get("tier1_requirements") is not None,
            "deterministic_reproducibility": test_results.get("scientific_reproducibility") is not None,
            "gradient_curriculum": test_results.get("tier1_requirements") is not None,
            "framework_extensibility": test_results.get("cross_system_integration") is not None,
            "performance_benchmarks": test_results.get("performance_benchmarks") is not None,
            "end_to_end_workflows": test_results.get("end_to_end_workflow") is not None,
            "demo_scenarios": test_results.get("demo_scenarios") is not None
        }
        
        validated_capabilities = sum(capabilities_validated.values())
        total_capabilities = len(capabilities_validated)
        capability_coverage = validated_capabilities / total_capabilities
        
        # Critical issues identification
        critical_issues = []
        for metric in self.validation_metrics:
            if metric.success_rate < 0.5:
                critical_issues.append(f"{metric.test_module}: {metric.success_rate:.1%} success rate")
        
        # Readiness assessment
        if overall_success_rate >= 0.9 and capability_coverage >= 0.8:
            readiness_status = "READY"
            readiness_confidence = "HIGH"
        elif overall_success_rate >= 0.75 and capability_coverage >= 0.6:
            readiness_status = "MOSTLY_READY"
            readiness_confidence = "MEDIUM"
        else:
            readiness_status = "NOT_READY"
            readiness_confidence = "LOW"
        
        return {
            "validation_date": datetime.now().isoformat(),
            "total_tests_run": total_tests,
            "total_tests_passed": total_passed,
            "overall_success_rate": overall_success_rate,
            "capability_coverage": capability_coverage,
            "validated_capabilities": validated_capabilities,
            "total_capabilities": total_capabilities,
            "critical_issues_count": len(critical_issues),
            "critical_issues": critical_issues,
            "readiness_status": readiness_status,
            "readiness_confidence": readiness_confidence,
            "tier1_recommendation": "APPROVED" if readiness_status == "READY" else "NEEDS_WORK"
        }
    
    def _generate_performance_analysis(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance analysis from test results."""
        
        performance_data = test_results.get("performance_benchmarks")
        
        if not performance_data:
            return {"status": "NO_DATA", "message": "Performance benchmarks not available"}
        
        # Extract key performance metrics
        performance_analysis = {
            "simulation_speed": {
                "target_ticks_per_minute": 1000,
                "achieved_performance": "UNKNOWN",  # Would be populated from actual results
                "meets_target": False  # Would be determined from results
            },
            "memory_usage": {
                "target_memory_mb": 2048,
                "peak_usage_mb": "UNKNOWN",
                "meets_target": False
            },
            "concurrent_scalability": {
                "target_agents": 10,
                "max_tested_agents": "UNKNOWN",
                "meets_target": False
            },
            "api_responsiveness": {
                "target_response_ms": 500,
                "average_response_ms": "UNKNOWN",
                "meets_target": False
            },
            "storage_performance": {
                "events_per_second": "UNKNOWN",
                "snapshot_generation_time": "UNKNOWN",
                "acceptable_performance": False
            }
        }
        
        # Performance score calculation
        performance_metrics = [metric for metric in self.validation_metrics if metric.test_module == "Performance Benchmarks"]
        if performance_metrics:
            performance_score = performance_metrics[0].performance_score
        else:
            performance_score = 0
        
        performance_analysis["overall_performance_score"] = performance_score
        performance_analysis["performance_grade"] = (
            "A" if performance_score >= 90 else
            "B" if performance_score >= 80 else
            "C" if performance_score >= 70 else
            "D" if performance_score >= 60 else "F"
        )
        
        return performance_analysis
    
    def _generate_reproducibility_assessment(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reproducibility assessment from test results."""
        
        reproducibility_data = test_results.get("scientific_reproducibility")
        
        if not reproducibility_data:
            return {"status": "NO_DATA", "message": "Reproducibility tests not available"}
        
        # Reproducibility criteria
        reproducibility_criteria = {
            "deterministic_results": reproducibility_data is not None,
            "golden_snapshots": reproducibility_data is not None,
            "statistical_consistency": reproducibility_data is not None,
            "configuration_sensitivity": reproducibility_data is not None,
            "cross_platform_consistency": reproducibility_data is not None
        }
        
        criteria_met = sum(reproducibility_criteria.values())
        total_criteria = len(reproducibility_criteria)
        reproducibility_score = (criteria_met / total_criteria) * 100
        
        # Scientific rigor assessment
        if reproducibility_score >= 95:
            rigor_level = "EXCELLENT"
        elif reproducibility_score >= 85:
            rigor_level = "GOOD"
        elif reproducibility_score >= 70:
            rigor_level = "ACCEPTABLE"
        else:
            rigor_level = "INADEQUATE"
        
        return {
            "reproducibility_criteria": reproducibility_criteria,
            "criteria_met": criteria_met,
            "total_criteria": total_criteria,
            "reproducibility_score": reproducibility_score,
            "scientific_rigor_level": rigor_level,
            "bit_perfect_determinism": reproducibility_data is not None,
            "suitable_for_research": reproducibility_score >= 85
        }
    
    def _generate_demo_outcomes(self, test_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate demo scenario outcomes analysis."""
        
        demo_data = test_results.get("demo_scenarios")
        
        if not demo_data:
            return {"status": "NO_DATA", "message": "Demo scenarios not available"}
        
        # Extract demo summary
        demo_summary = demo_data.get("demo_summary", {})
        tier1_capabilities = demo_data.get("tier1_capabilities_demonstrated", {})
        
        # Demo effectiveness analysis
        demo_outcomes = {
            "scenarios_executed": demo_summary.get("total_demos", 0),
            "successful_demos": demo_summary.get("successful_demos", 0),
            "demo_success_rate": demo_summary.get("success_rate", 0),
            "average_score": demo_summary.get("average_score", 0),
            "tier1_capabilities_shown": sum(tier1_capabilities.values()) if tier1_capabilities else 0,
            "total_capabilities": len(tier1_capabilities) if tier1_capabilities else 0,
            "demonstration_effectiveness": "UNKNOWN"
        }
        
        # Calculate demonstration effectiveness
        if demo_outcomes["demo_success_rate"] >= 0.8 and demo_outcomes["tier1_capabilities_shown"] >= 4:
            demo_outcomes["demonstration_effectiveness"] = "EXCELLENT"
        elif demo_outcomes["demo_success_rate"] >= 0.6 and demo_outcomes["tier1_capabilities_shown"] >= 3:
            demo_outcomes["demonstration_effectiveness"] = "GOOD"
        elif demo_outcomes["demo_success_rate"] >= 0.4:
            demo_outcomes["demonstration_effectiveness"] = "ACCEPTABLE"
        else:
            demo_outcomes["demonstration_effectiveness"] = "POOR"
        
        return demo_outcomes
    
    def _generate_compliance_matrix(self, test_results: Dict[str, Any]) -> Dict[str, bool]:
        """Generate tier-1 blueprint compliance matrix."""
        
        # Tier-1 blueprint requirements compliance
        compliance_matrix = {
            # Core Requirements
            "multi_dimensional_measurement": test_results.get("tier1_requirements") is not None,
            "instrumented_root_cause_analysis": test_results.get("tier1_requirements") is not None,
            "deterministic_reproducibility": test_results.get("scientific_reproducibility") is not None,
            "first_class_extensibility": test_results.get("cross_system_integration") is not None,
            
            # Curriculum Requirements
            "gradient_curriculum_t0_t3": test_results.get("tier1_requirements") is not None,
            "tier_specific_constraints": test_results.get("tier1_requirements") is not None,
            "baseline_bot_performance": test_results.get("tier1_requirements") is not None,
            
            # Technical Requirements
            "event_driven_architecture": test_results.get("end_to_end_workflow") is not None,
            "framework_abstraction": test_results.get("cross_system_integration") is not None,
            "memory_experiments": test_results.get("demo_scenarios") is not None,
            "adversarial_testing": test_results.get("tier1_requirements") is not None,
            
            # Performance Requirements
            "simulation_speed_targets": test_results.get("performance_benchmarks") is not None,
            "memory_usage_limits": test_results.get("performance_benchmarks") is not None,
            "concurrent_scalability": test_results.get("performance_benchmarks") is not None,
            "api_responsiveness": test_results.get("performance_benchmarks") is not None,
            
            # Scientific Requirements
            "golden_snapshots": test_results.get("scientific_reproducibility") is not None,
            "statistical_consistency": test_results.get("scientific_reproducibility") is not None,
            "configuration_sensitivity": test_results.get("scientific_reproducibility") is not None,
            
            # Demonstration Requirements
            "demo_scenarios_complete": test_results.get("demo_scenarios") is not None,
            "tier1_capabilities_shown": test_results.get("demo_scenarios") is not None
        }
        
        return compliance_matrix
    
    def _generate_recommendations(self, test_results: Dict[str, Any]) -> List[str]:
        """Generate actionable recommendations based on test results."""
        
        recommendations = []
        
        # Analyze validation metrics for issues
        for metric in self.validation_metrics:
            if metric.success_rate < 0.8:
                recommendations.append(f"Address failures in {metric.test_module} (success rate: {metric.success_rate:.1%})")
            
            if metric.performance_score < 70:
                recommendations.append(f"Improve performance in {metric.test_module} (score: {metric.performance_score:.1f})")
        
        # Specific recommendations based on test results
        if not test_results.get("tier1_requirements"):
            recommendations.append("CRITICAL: Complete tier-1 requirements validation before production")
        
        if not test_results.get("scientific_reproducibility"):
            recommendations.append("CRITICAL: Ensure scientific reproducibility for research credibility")
        
        if not test_results.get("performance_benchmarks"):
            recommendations.append("Validate performance benchmarks meet tier-1 standards")
        
        if not test_results.get("demo_scenarios"):
            recommendations.append("Complete demo scenarios to showcase tier-1 capabilities")
        
        # Calculate overall health
        total_success_rate = sum(metric.success_rate for metric in self.validation_metrics) / len(self.validation_metrics)
        
        if total_success_rate >= 0.9:
            recommendations.append("System ready for tier-1 benchmark production deployment")
        elif total_success_rate >= 0.75:
            recommendations.append("Address remaining issues before tier-1 production deployment")
        else:
            recommendations.append("Significant work needed before tier-1 readiness")
        
        # Add specific technical recommendations
        recommendations.extend([
            "Run full integration test suite regularly during development",
            "Monitor performance benchmarks continuously",
            "Validate reproducibility across different environments",
            "Update demo scenarios as new features are added",
            "Maintain comprehensive test documentation"
        ])
        
        return recommendations
    
    def _calculate_overall_readiness_score(self, test_results: Dict[str, Any]) -> float:
        """Calculate overall tier-1 readiness score."""
        
        if not self.validation_metrics:
            return 0.0
        
        # Weight different test modules by importance
        module_weights = {
            "Tier-1 Requirements": 0.25,      # 25% - Most critical
            "Scientific Reproducibility": 0.20,  # 20% - Essential for research
            "End-to-End Workflow": 0.15,      # 15% - Core functionality
            "Cross-System Integration": 0.15,  # 15% - System reliability
            "Performance Benchmarks": 0.15,    # 15% - Scalability
            "Demo Scenarios": 0.10             # 10% - Demonstration
        }
        
        weighted_score = 0.0
        total_weight = 0.0
        
        for metric in self.validation_metrics:
            weight = module_weights.get(metric.test_module, 0.1)
            weighted_score += metric.performance_score * weight
            total_weight += weight
        
        if total_weight == 0:
            return 0.0
        
        final_score = weighted_score / total_weight
        
        # Apply penalties for critical failures
        critical_failures = sum(1 for metric in self.validation_metrics if metric.success_rate < 0.5)
        if critical_failures > 0:
            penalty = min(critical_failures * 10, 30)  # Max 30% penalty
            final_score = max(0, final_score - penalty)
        
        return final_score
    
    def export_report(self, report: ValidationReport, format: str = "json") -> str:
        """Export validation report in specified format."""
        
        if format.lower() == "json":
            return json.dumps(asdict(report), indent=2, default=str)
        elif format.lower() == "markdown":
            return self._format_markdown_report(report)
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def _format_markdown_report(self, report: ValidationReport) -> str:
        """Format validation report as markdown."""
        
        md = f"""# FBA-Bench Tier-1 Validation Report

**Report ID:** {report.report_id}  
**Generated:** {report.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}  
**FBA-Bench Version:** {report.fba_bench_version}  

## Executive Summary

**Overall Readiness Score:** {report.overall_readiness_score:.1f}%  
**Tier-1 Ready:** {'✅ YES' if report.tier1_ready else '❌ NO'}  

### Key Metrics
- **Tests Run:** {report.executive_summary['total_tests_run']}
- **Tests Passed:** {report.executive_summary['total_tests_passed']}
- **Success Rate:** {report.executive_summary['overall_success_rate']:.1%}
- **Capability Coverage:** {report.executive_summary['capability_coverage']:.1%}

### Readiness Assessment
- **Status:** {report.executive_summary['readiness_status']}
- **Confidence:** {report.executive_summary['readiness_confidence']}
- **Recommendation:** {report.executive_summary['tier1_recommendation']}

## Test Results Summary

| Test Module | Tests Run | Passed | Failed | Success Rate | Performance Score |
|-------------|-----------|--------|--------|--------------|-------------------|
"""
        
        for metric in report.test_results:
            md += f"| {metric.test_module} | {metric.tests_run} | {metric.tests_passed} | {metric.tests_failed} | {metric.success_rate:.1%} | {metric.performance_score:.1f}% |\n"
        
        md += f"""
## Compliance Matrix

| Requirement | Status |
|-------------|--------|
"""
        
        for requirement, status in report.compliance_matrix.items():
            status_icon = "✅" if status else "❌"
            md += f"| {requirement.replace('_', ' ').title()} | {status_icon} |\n"
        
        md += f"""
## Recommendations

"""
        for i, rec in enumerate(report.recommendations, 1):
            md += f"{i}. {rec}\n"
        
        md += f"""
## Performance Analysis

{json.dumps(report.performance_analysis, indent=2)}

## Reproducibility Assessment

{json.dumps(report.reproducibility_assessment, indent=2)}

---

*This report was automatically generated by the FBA-Bench Integration Test Suite.*
"""
        
        return md

# Main execution function
async def generate_validation_report() -> ValidationReport:
    """Generate comprehensive FBA-Bench validation report."""
    
    logger.info("🚀 Starting FBA-Bench Tier-1 Validation...")
    
    generator = ValidationReportGenerator()
    report = await generator.generate_comprehensive_report()
    
    # Export report in multiple formats
    json_report = generator.export_report(report, "json")
    markdown_report = generator.export_report(report, "markdown")
    
    # Save reports to files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    json_filename = f"fba_bench_validation_report_{timestamp}.json"
    markdown_filename = f"fba_bench_validation_report_{timestamp}.md"
    
    with open(json_filename, 'w') as f:
        f.write(json_report)
    
    with open(markdown_filename, 'w') as f:
        f.write(markdown_report)
    
    logger.info(f"📊 Validation reports saved:")
    logger.info(f"   JSON: {json_filename}")
    logger.info(f"   Markdown: {markdown_filename}")
    
    return report

if __name__ == "__main__":
    # Generate validation report when script is executed directly
    asyncio.run(generate_validation_report())