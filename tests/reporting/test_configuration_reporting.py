"""
Test Configuration and Reporting Framework for FBA-Bench

Provides comprehensive test results analysis, configuration management,
and reporting capabilities across all test categories.
"""

import asyncio
import logging
import json
import yaml
import time
import os
import shutil
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
import sys
from enum import Enum
import jinja2

# Add project root to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logger = logging.getLogger(__name__)


class ReportFormat(Enum):
    """Supported report formats."""
    JSON = "json"
    HTML = "html"
    MARKDOWN = "markdown"
    PDF = "pdf"


class TestStatus(Enum):
    """Test execution status."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"


@dataclass
class TestConfiguration:
    """Configuration for test execution and reporting."""
    # Test execution settings
    enable_integration_tests: bool = True
    enable_performance_tests: bool = True
    enable_functional_tests: bool = True
    enable_scenario_tests: bool = True
    enable_extensibility_tests: bool = True
    enable_real_world_tests: bool = True
    enable_regression_tests: bool = True
    enable_documentation_tests: bool = True
    
    # Execution parameters
    stop_on_first_failure: bool = False
    parallel_execution: bool = False
    max_parallel_workers: int = 4
    test_timeout_seconds: int = 3600
    
    # Reporting settings
    output_directory: str = "test_reports"
    report_formats: List[ReportFormat] = field(default_factory=lambda: [ReportFormat.JSON, ReportFormat.HTML])
    include_detailed_logs: bool = True
    include_performance_charts: bool = True
    generate_executive_summary: bool = True
    
    # Historical comparison
    enable_historical_comparison: bool = True
    historical_data_retention_days: int = 30
    baseline_comparison_enabled: bool = True
    
    # CI/CD integration
    ci_integration_enabled: bool = False
    junit_xml_output: bool = False
    publish_to_dashboard: bool = False
    dashboard_url: Optional[str] = None
    
    # Notification settings
    email_notifications: bool = False
    slack_notifications: bool = False
    notification_on_failure_only: bool = True


@dataclass
class TestSuiteResult:
    """Results from a single test suite."""
    suite_name: str
    category: str
    status: TestStatus
    start_time: str
    end_time: str
    duration_seconds: float
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    success_rate: float
    detailed_results: Dict[str, Any] = field(default_factory=dict)
    error_details: Optional[str] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)


@dataclass
class ComprehensiveTestReport:
    """Comprehensive test report combining all suite results."""
    report_id: str
    generation_time: str
    configuration: TestConfiguration
    overall_status: TestStatus
    total_duration_seconds: float
    
    # Aggregate statistics
    total_suites: int
    passed_suites: int
    failed_suites: int
    total_tests_across_suites: int
    total_passed_tests: int
    total_failed_tests: int
    overall_success_rate: float
    
    # Suite results
    suite_results: List[TestSuiteResult] = field(default_factory=list)
    
    # Analysis
    performance_summary: Dict[str, Any] = field(default_factory=dict)
    regression_analysis: Dict[str, Any] = field(default_factory=dict)
    quality_metrics: Dict[str, float] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


class TestConfigurationManager:
    """Manages test configuration and provides defaults."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or "tests/config/test_config.yaml"
        self.config: TestConfiguration = TestConfiguration()
        
    def load_configuration(self) -> TestConfiguration:
        """Load configuration from file or return defaults."""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    config_data = yaml.safe_load(f)
                
                # Convert to TestConfiguration object
                self.config = TestConfiguration(**config_data)
                logger.info(f"Loaded test configuration from {self.config_file}")
                
            except Exception as e:
                logger.error(f"Failed to load configuration from {self.config_file}: {e}")
                logger.info("Using default configuration")
        else:
            logger.info("No configuration file found, using defaults")
            
        return self.config
    
    def save_configuration(self, config: TestConfiguration) -> bool:
        """Save configuration to file."""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            
            with open(self.config_file, 'w') as f:
                yaml.dump(asdict(config), f, default_flow_style=False)
            
            logger.info(f"Configuration saved to {self.config_file}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate_configuration(self, config: TestConfiguration) -> List[str]:
        """Validate configuration and return any issues."""
        issues = []
        
        if config.max_parallel_workers < 1:
            issues.append("max_parallel_workers must be at least 1")
        
        if config.test_timeout_seconds < 60:
            issues.append("test_timeout_seconds should be at least 60 seconds")
        
        if config.historical_data_retention_days < 1:
            issues.append("historical_data_retention_days must be at least 1")
        
        if config.ci_integration_enabled and not config.junit_xml_output:
            issues.append("CI integration typically requires JUnit XML output")
        
        return issues


class TestReportGenerator:
    """Generates comprehensive test reports in multiple formats."""
    
    def __init__(self, config: TestConfiguration):
        self.config = config
        self.output_dir = Path(config.output_directory)
        self.templates_dir = Path(__file__).parent / "templates"
        
        # Setup Jinja2 for HTML template rendering
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(self.templates_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
    
    async def generate_comprehensive_report(
        self, 
        suite_results: List[TestSuiteResult],
        report_id: Optional[str] = None
    ) -> ComprehensiveTestReport:
        """Generate a comprehensive test report from all suite results."""
        
        if not report_id:
            report_id = f"test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Calculate aggregate statistics
        total_suites = len(suite_results)
        passed_suites = sum(1 for r in suite_results if r.status == TestStatus.PASSED)
        failed_suites = total_suites - passed_suites
        
        total_tests = sum(r.total_tests for r in suite_results)
        total_passed = sum(r.passed_tests for r in suite_results)
        total_failed = sum(r.failed_tests for r in suite_results)
        
        overall_success_rate = total_passed / total_tests if total_tests > 0 else 0
        overall_status = TestStatus.PASSED if failed_suites == 0 else TestStatus.FAILED
        
        total_duration = sum(r.duration_seconds for r in suite_results)
        
        # Generate performance summary
        performance_summary = await self._analyze_performance(suite_results)
        
        # Generate regression analysis
        regression_analysis = await self._analyze_regressions(suite_results)
        
        # Calculate quality metrics
        quality_metrics = await self._calculate_quality_metrics(suite_results)
        
        # Generate recommendations
        recommendations = await self._generate_recommendations(suite_results, quality_metrics)
        
        report = ComprehensiveTestReport(
            report_id=report_id,
            generation_time=datetime.now().isoformat(),
            configuration=self.config,
            overall_status=overall_status,
            total_duration_seconds=total_duration,
            total_suites=total_suites,
            passed_suites=passed_suites,
            failed_suites=failed_suites,
            total_tests_across_suites=total_tests,
            total_passed_tests=total_passed,
            total_failed_tests=total_failed,
            overall_success_rate=overall_success_rate,
            suite_results=suite_results,
            performance_summary=performance_summary,
            regression_analysis=regression_analysis,
            quality_metrics=quality_metrics,
            recommendations=recommendations
        )
        
        return report
    
    async def save_report(self, report: ComprehensiveTestReport) -> Dict[str, str]:
        """Save report in all configured formats."""
        saved_files = {}
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        for format_type in self.config.report_formats:
            try:
                if format_type == ReportFormat.JSON:
                    file_path = await self._save_json_report(report)
                    saved_files["json"] = str(file_path)
                    
                elif format_type == ReportFormat.HTML:
                    file_path = await self._save_html_report(report)
                    saved_files["html"] = str(file_path)
                    
                elif format_type == ReportFormat.MARKDOWN:
                    file_path = await self._save_markdown_report(report)
                    saved_files["markdown"] = str(file_path)
                    
                elif format_type == ReportFormat.PDF:
                    file_path = await self._save_pdf_report(report)
                    saved_files["pdf"] = str(file_path)
                    
            except Exception as e:
                logger.error(f"Failed to save report in {format_type.value} format: {e}")
        
        # Generate JUnit XML if requested
        if self.config.junit_xml_output:
            try:
                file_path = await self._save_junit_xml(report)
                saved_files["junit_xml"] = str(file_path)
            except Exception as e:
                logger.error(f"Failed to save JUnit XML: {e}")
        
        logger.info(f"Test report saved in {len(saved_files)} format(s)")
        return saved_files
    
    async def _save_json_report(self, report: ComprehensiveTestReport) -> Path:
        """Save report as JSON."""
        file_path = self.output_dir / f"{report.report_id}.json"
        
        # Convert dataclass to dict for JSON serialization
        report_dict = asdict(report)
        
        with open(file_path, 'w') as f:
            json.dump(report_dict, f, indent=2, default=str)
        
        return file_path
    
    async def _save_html_report(self, report: ComprehensiveTestReport) -> Path:
        """Save report as HTML."""
        file_path = self.output_dir / f"{report.report_id}.html"
        
        # Create HTML template if it doesn't exist
        await self._ensure_html_template()
        
        try:
            template = self.jinja_env.get_template("test_report.html")
            html_content = template.render(
                report=report,
                generation_time=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            )
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
                
        except Exception as e:
            # Fallback to simple HTML if template fails
            logger.warning(f"Template rendering failed, using simple HTML: {e}")
            html_content = await self._generate_simple_html(report)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
        
        return file_path
    
    async def _save_markdown_report(self, report: ComprehensiveTestReport) -> Path:
        """Save report as Markdown."""
        file_path = self.output_dir / f"{report.report_id}.md"
        
        md_content = f"""# FBA-Bench Test Report

**Report ID:** {report.report_id}  
**Generated:** {report.generation_time}  
**Overall Status:** {'✅ PASSED' if report.overall_status == TestStatus.PASSED else '❌ FAILED'}

## Executive Summary

- **Total Test Suites:** {report.total_suites}
- **Passed Suites:** {report.passed_suites}
- **Failed Suites:** {report.failed_suites}
- **Total Tests:** {report.total_tests_across_suites}
- **Overall Success Rate:** {report.overall_success_rate:.1%}
- **Total Duration:** {report.total_duration_seconds:.2f} seconds

## Quality Metrics

"""
        
        for metric, value in report.quality_metrics.items():
            md_content += f"- **{metric.replace('_', ' ').title()}:** {value:.3f}\n"
        
        md_content += "\n## Test Suite Results\n\n"
        
        for suite in report.suite_results:
            status_icon = '✅' if suite.status == TestStatus.PASSED else '❌'
            md_content += f"### {status_icon} {suite.suite_name}\n"
            md_content += f"- **Category:** {suite.category}\n"
            md_content += f"- **Duration:** {suite.duration_seconds:.2f}s\n"
            md_content += f"- **Success Rate:** {suite.success_rate:.1%}\n"
            md_content += f"- **Tests:** {suite.passed_tests}/{suite.total_tests} passed\n"
            if suite.error_details:
                md_content += f"- **Error:** {suite.error_details}\n"
            md_content += "\n"
        
        if report.recommendations:
            md_content += "## Recommendations\n\n"
            for i, rec in enumerate(report.recommendations, 1):
                md_content += f"{i}. {rec}\n"
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        
        return file_path
    
    async def _save_pdf_report(self, report: ComprehensiveTestReport) -> Path:
        """Save report as PDF (requires additional dependencies)."""
        # This would require a PDF library like reportlab or weasyprint
        # For now, return a placeholder
        file_path = self.output_dir / f"{report.report_id}.pdf"
        
        # Create a simple text-based PDF placeholder
        with open(file_path, 'w') as f:
            f.write(f"PDF Report for {report.report_id}\n")
            f.write("Note: Full PDF generation requires additional dependencies\n")
        
        return file_path
    
    async def _save_junit_xml(self, report: ComprehensiveTestReport) -> Path:
        """Save JUnit XML format for CI/CD integration."""
        file_path = self.output_dir / f"{report.report_id}_junit.xml"
        
        xml_content = '<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_content += f'<testsuites tests="{report.total_tests_across_suites}" failures="{report.total_failed_tests}" time="{report.total_duration_seconds:.3f}">\n'
        
        for suite in report.suite_results:
            xml_content += f'  <testsuite name="{suite.suite_name}" tests="{suite.total_tests}" failures="{suite.failed_tests}" time="{suite.duration_seconds:.3f}">\n'
            
            # Add individual test cases (simplified)
            for i in range(suite.total_tests):
                test_name = f"test_{i}"
                if i < suite.passed_tests:
                    xml_content += f'    <testcase name="{test_name}" time="1.0"/>\n'
                else:
                    xml_content += f'    <testcase name="{test_name}" time="1.0">\n'
                    xml_content += f'      <failure message="Test failed">Test {test_name} failed</failure>\n'
                    xml_content += f'    </testcase>\n'
            
            xml_content += '  </testsuite>\n'
        
        xml_content += '</testsuites>\n'
        
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(xml_content)
        
        return file_path
    
    async def _analyze_performance(self, suite_results: List[TestSuiteResult]) -> Dict[str, Any]:
        """Analyze performance metrics across all suites."""
        performance_suites = [s for s in suite_results if s.performance_metrics]
        
        if not performance_suites:
            return {"analysis": "No performance data available"}
        
        # Aggregate performance metrics
        all_metrics = {}
        for suite in performance_suites:
            for metric, value in suite.performance_metrics.items():
                if metric not in all_metrics:
                    all_metrics[metric] = []
                all_metrics[metric].append(value)
        
        summary = {}
        for metric, values in all_metrics.items():
            summary[metric] = {
                "average": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "count": len(values)
            }
        
        return {
            "metrics_summary": summary,
            "total_suites_with_metrics": len(performance_suites),
            "performance_score": sum(s.success_rate for s in performance_suites) / len(performance_suites)
        }
    
    async def _analyze_regressions(self, suite_results: List[TestSuiteResult]) -> Dict[str, Any]:
        """Analyze for potential regressions."""
        regression_suites = [s for s in suite_results if "regression" in s.category.lower()]
        
        if not regression_suites:
            return {"analysis": "No regression test data available"}
        
        regressions_detected = sum(1 for s in regression_suites if s.status == TestStatus.FAILED)
        
        return {
            "regression_suites_run": len(regression_suites),
            "regressions_detected": regressions_detected,
            "regression_free": regressions_detected == 0,
            "regression_rate": regressions_detected / len(regression_suites) if regression_suites else 0
        }
    
    async def _calculate_quality_metrics(self, suite_results: List[TestSuiteResult]) -> Dict[str, float]:
        """Calculate overall quality metrics."""
        if not suite_results:
            return {}
        
        total_tests = sum(s.total_tests for s in suite_results)
        passed_tests = sum(s.passed_tests for s in suite_results)
        
        return {
            "overall_pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "suite_completion_rate": sum(1 for s in suite_results if s.status == TestStatus.PASSED) / len(suite_results),
            "average_suite_duration": sum(s.duration_seconds for s in suite_results) / len(suite_results),
            "test_coverage_score": min(1.0, len(suite_results) / 8),  # Assuming 8 expected categories
            "reliability_score": passed_tests / total_tests if total_tests > 0 else 0
        }
    
    async def _generate_recommendations(
        self, 
        suite_results: List[TestSuiteResult], 
        quality_metrics: Dict[str, float]
    ) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        # Check overall pass rate
        if quality_metrics.get("overall_pass_rate", 0) < 0.9:
            recommendations.append("Overall test pass rate is below 90%. Review failing tests and address root causes.")
        
        # Check for failed suites
        failed_suites = [s for s in suite_results if s.status == TestStatus.FAILED]
        if failed_suites:
            recommendations.append(f"{len(failed_suites)} test suite(s) failed. Prioritize fixing: {', '.join(s.suite_name for s in failed_suites[:3])}")
        
        # Check performance
        long_running_suites = [s for s in suite_results if s.duration_seconds > 300]  # 5 minutes
        if long_running_suites:
            recommendations.append(f"Consider optimizing performance for slow test suites: {', '.join(s.suite_name for s in long_running_suites[:2])}")
        
        # Check coverage
        if quality_metrics.get("test_coverage_score", 0) < 1.0:
            recommendations.append("Consider adding more test categories to improve coverage completeness.")
        
        # Check for regressions
        regression_suites = [s for s in suite_results if "regression" in s.category.lower() and s.status == TestStatus.FAILED]
        if regression_suites:
            recommendations.append("Regression tests failed. Review recent changes and implement fixes before deployment.")
        
        return recommendations
    
    async def _ensure_html_template(self):
        """Ensure HTML template exists."""
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        template_file = self.templates_dir / "test_report.html"
        
        if not template_file.exists():
            # Create a basic HTML template
            template_content = """
<!DOCTYPE html>
<html>
<head>
    <title>FBA-Bench Test Report</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { background-color: #f0f0f0; padding: 20px; }
        .summary { margin: 20px 0; }
        .suite { margin: 10px 0; padding: 10px; border: 1px solid #ddd; }
        .passed { background-color: #d4edda; }
        .failed { background-color: #f8d7da; }
        .metrics { display: flex; flex-wrap: wrap; }
        .metric { margin: 10px; padding: 10px; background-color: #e9ecef; }
    </style>
</head>
<body>
    <div class="header">
        <h1>FBA-Bench Test Report</h1>
        <p><strong>Report ID:</strong> {{ report.report_id }}</p>
        <p><strong>Generated:</strong> {{ generation_time }}</p>
        <p><strong>Overall Status:</strong> 
            {% if report.overall_status.value == 'passed' %}
                <span style="color: green;">✅ PASSED</span>
            {% else %}
                <span style="color: red;">❌ FAILED</span>
            {% endif %}
        </p>
    </div>
    
    <div class="summary">
        <h2>Executive Summary</h2>
        <div class="metrics">
            <div class="metric">
                <strong>Total Suites:</strong> {{ report.total_suites }}
            </div>
            <div class="metric">
                <strong>Success Rate:</strong> {{ "%.1f"|format(report.overall_success_rate * 100) }}%
            </div>
            <div class="metric">
                <strong>Duration:</strong> {{ "%.2f"|format(report.total_duration_seconds) }}s
            </div>
        </div>
    </div>
    
    <div class="suites">
        <h2>Test Suite Results</h2>
        {% for suite in report.suite_results %}
        <div class="suite {% if suite.status.value == 'passed' %}passed{% else %}failed{% endif %}">
            <h3>{{ suite.suite_name }}</h3>
            <p><strong>Category:</strong> {{ suite.category }}</p>
            <p><strong>Tests:</strong> {{ suite.passed_tests }}/{{ suite.total_tests }} passed</p>
            <p><strong>Duration:</strong> {{ "%.2f"|format(suite.duration_seconds) }}s</p>
            {% if suite.error_details %}
            <p><strong>Error:</strong> {{ suite.error_details }}</p>
            {% endif %}
        </div>
        {% endfor %}
    </div>
    
    {% if report.recommendations %}
    <div class="recommendations">
        <h2>Recommendations</h2>
        <ul>
        {% for rec in report.recommendations %}
            <li>{{ rec }}</li>
        {% endfor %}
        </ul>
    </div>
    {% endif %}
</body>
</html>
            """
            
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template_content)
    
    async def _generate_simple_html(self, report: ComprehensiveTestReport) -> str:
        """Generate simple HTML report as fallback."""
        status_color = "green" if report.overall_status == TestStatus.PASSED else "red"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>FBA-Bench Test Report</title>
    <style>body {{ font-family: Arial, sans-serif; margin: 20px; }}</style>
</head>
<body>
    <h1>FBA-Bench Test Report</h1>
    <p><strong>Report ID:</strong> {report.report_id}</p>
    <p><strong>Status:</strong> <span style="color: {status_color};">{report.overall_status.value.upper()}</span></p>
    <p><strong>Success Rate:</strong> {report.overall_success_rate:.1%}</p>
    <p><strong>Duration:</strong> {report.total_duration_seconds:.2f}s</p>
    
    <h2>Test Suites</h2>
"""
        
        for suite in report.suite_results:
            suite_color = "green" if suite.status == TestStatus.PASSED else "red"
            html += f"""
    <div style="margin: 10px 0; padding: 10px; border: 1px solid #ddd;">
        <h3 style="color: {suite_color};">{suite.suite_name}</h3>
        <p>Tests: {suite.passed_tests}/{suite.total_tests} passed</p>
        <p>Duration: {suite.duration_seconds:.2f}s</p>
    </div>
"""
        
        html += "</body></html>"
        return html


# CLI runner for direct execution
async def main():
    """Run test configuration and reporting framework."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Example usage of the framework
    config_manager = TestConfigurationManager()
    config = config_manager.load_configuration()
    
    # Example test results (normally would come from actual test execution)
    sample_results = [
        TestSuiteResult(
            suite_name="Integration Tests",
            category="integration",
            status=TestStatus.PASSED,
            start_time="2024-01-01T10:00:00",
            end_time="2024-01-01T10:15:00",
            duration_seconds=900,
            total_tests=25,
            passed_tests=25,
            failed_tests=0,
            skipped_tests=0,
            success_rate=1.0,
            performance_metrics={"average_response_time": 120.5}
        ),
        TestSuiteResult(
            suite_name="Performance Tests",
            category="performance",
            status=TestStatus.FAILED,
            start_time="2024-01-01T10:15:00",
            end_time="2024-01-01T10:45:00",
            duration_seconds=1800,
            total_tests=15,
            passed_tests=12,
            failed_tests=3,
            skipped_tests=0,
            success_rate=0.8,
            error_details="3 performance tests exceeded timeout thresholds"
        )
    ]
    
    # Generate comprehensive report
    report_generator = TestReportGenerator(config)
    report = await report_generator.generate_comprehensive_report(sample_results)
    
    # Save report in all configured formats
    saved_files = await report_generator.save_report(report)
    
    print("\n" + "="*80)
    print("TEST CONFIGURATION AND REPORTING FRAMEWORK")
    print("="*80)
    print(f"Report ID: {report.report_id}")
    print(f"Overall Status: {report.overall_status.value.upper()}")
    print(f"Success Rate: {report.overall_success_rate:.1%}")
    print(f"Total Duration: {report.total_duration_seconds:.2f}s")
    print(f"Reports Generated: {len(saved_files)}")
    
    for format_name, file_path in saved_files.items():
        print(f"  - {format_name.upper()}: {file_path}")
    
    if report.recommendations:
        print("\nRecommendations:")
        for i, rec in enumerate(report.recommendations, 1):
            print(f"  {i}. {rec}")
    
    print("="*80)


if __name__ == "__main__":
    asyncio.run(main())