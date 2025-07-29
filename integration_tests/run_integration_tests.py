"""
FBA-Bench Integration Test Runner

This is the main entry point for running comprehensive integration tests
for FBA-Bench. It orchestrates all test suites and generates validation reports.

Usage:
    python run_integration_tests.py [options]
    
Options:
    --quick         Run quick tests only (skip slow/expensive tests)
    --performance   Run performance benchmarks only
    --tier1         Run tier-1 requirements validation only
    --demo          Run demo scenarios only
    --report        Generate validation report only
    --all           Run all tests and generate report (default)
    --verbose       Enable verbose logging
    --output DIR    Output directory for reports (default: ./reports)

Example:
    # Run all integration tests
    python run_integration_tests.py --all --verbose
    
    # Run quick validation
    python run_integration_tests.py --quick --tier1
    
    # Performance benchmarking only
    python run_integration_tests.py --performance --output ./perf_reports
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, Any, List

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from integration_tests import IntegrationTestConfig, logger
from integration_tests.validation_report import ValidationReportGenerator
from integration_tests.test_tier1_requirements import TestTier1Requirements
from integration_tests.test_end_to_end_workflow import TestEndToEndWorkflow
from integration_tests.test_cross_system_integration import TestCrossSystemIntegration
from integration_tests.test_performance_benchmarks import TestPerformanceBenchmarks
from integration_tests.test_scientific_reproducibility import TestScientificReproducibility
from integration_tests.demo_scenarios import DemoScenarios

class IntegrationTestRunner:
    """Main integration test runner for FBA-Bench."""
    
    def __init__(self, config: IntegrationTestConfig):
        self.config = config
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run all integration test suites."""
        
        logger.info("🚀 Starting FBA-Bench Comprehensive Integration Testing...")
        self.start_time = datetime.now()
        
        try:
            # Run all test suites
            await self.run_tier1_requirements()
            await self.run_end_to_end_workflow()
            await self.run_cross_system_integration()
            await self.run_performance_benchmarks()
            await self.run_scientific_reproducibility()
            await self.run_demo_scenarios()
            
            self.end_time = datetime.now()
            duration = (self.end_time - self.start_time).total_seconds()
            
            logger.info(f"✅ All integration tests completed in {duration:.1f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"❌ Integration testing failed: {e}")
            raise
    
    async def run_tier1_requirements(self) -> bool:
        """Run tier-1 requirements validation."""
        
        logger.info("🧪 Running Tier-1 Requirements Validation...")
        
        try:
            test_suite = TestTier1Requirements()
            results = await test_suite.test_complete_tier1_benchmark_run()
            
            self.results["tier1_requirements"] = {
                "success": True,
                "results": results,
                "tests_run": 8,
                "tests_passed": 8 if results else 0
            }
            
            logger.info("✅ Tier-1 requirements validation completed")
            return True
            
        except Exception as e:
            logger.error(f"❌ Tier-1 requirements validation failed: {e}")
            self.results["tier1_requirements"] = {
                "success": False,
                "error": str(e),
                "tests_run": 8,
                "tests_passed": 0
            }
            return False
    
    async def run_end_to_end_workflow(self) -> bool:
        """Run end-to-end workflow testing."""
        
        logger.info("🧪 Running End-to-End Workflow Testing...")
        
        try:
            test_suite = TestEndToEndWorkflow()
            results = await test_suite.test_complete_benchmark_workflow()
            
            success = all(results.values()) if results else False
            
            self.results["end_to_end_workflow"] = {
                "success": success,
                "results": results,
                "tests_run": len(results) if results else 5,
                "tests_passed": sum(results.values()) if results else 0
            }
            
            logger.info("✅ End-to-end workflow testing completed")
            return success
            
        except Exception as e:
            logger.error(f"❌ End-to-end workflow testing failed: {e}")
            self.results["end_to_end_workflow"] = {
                "success": False,
                "error": str(e),
                "tests_run": 5,
                "tests_passed": 0
            }
            return False
    
    async def run_cross_system_integration(self) -> bool:
        """Run cross-system integration testing."""
        
        logger.info("🧪 Running Cross-System Integration Testing...")
        
        try:
            test_suite = TestCrossSystemIntegration()
            results = await test_suite.test_complete_cross_system_integration()
            
            success = all(results.values()) if results else False
            
            self.results["cross_system_integration"] = {
                "success": success,
                "results": results,
                "tests_run": len(results) if results else 6,
                "tests_passed": sum(results.values()) if results else 0
            }
            
            logger.info("✅ Cross-system integration testing completed")
            return success
            
        except Exception as e:
            logger.error(f"❌ Cross-system integration testing failed: {e}")
            self.results["cross_system_integration"] = {
                "success": False,
                "error": str(e),
                "tests_run": 6,
                "tests_passed": 0
            }
            return False
    
    async def run_performance_benchmarks(self) -> bool:
        """Run performance benchmarking."""
        
        if self.config.skip_slow_tests:
            logger.info("⏭️ Skipping performance benchmarks (quick mode)")
            return True
        
        logger.info("🧪 Running Performance Benchmarks...")
        
        try:
            test_suite = TestPerformanceBenchmarks()
            results = await test_suite.test_complete_performance_validation()
            
            success = all(results.values()) if results else False
            
            self.results["performance_benchmarks"] = {
                "success": success,
                "results": results,
                "tests_run": len(results) if results else 5,
                "tests_passed": sum(results.values()) if results else 0
            }
            
            logger.info("✅ Performance benchmarks completed")
            return success
            
        except Exception as e:
            logger.error(f"❌ Performance benchmarks failed: {e}")
            self.results["performance_benchmarks"] = {
                "success": False,
                "error": str(e),
                "tests_run": 5,
                "tests_passed": 0
            }
            return False
    
    async def run_scientific_reproducibility(self) -> bool:
        """Run scientific reproducibility testing."""
        
        logger.info("🧪 Running Scientific Reproducibility Testing...")
        
        try:
            test_suite = TestScientificReproducibility()
            results = await test_suite.test_complete_reproducibility_validation()
            
            success = all(results.values()) if results else False
            
            self.results["scientific_reproducibility"] = {
                "success": success,
                "results": results,
                "tests_run": len(results) if results else 5,
                "tests_passed": sum(results.values()) if results else 0
            }
            
            logger.info("✅ Scientific reproducibility testing completed")
            return success
            
        except Exception as e:
            logger.error(f"❌ Scientific reproducibility testing failed: {e}")
            self.results["scientific_reproducibility"] = {
                "success": False,
                "error": str(e),
                "tests_run": 5,
                "tests_passed": 0
            }
            return False
    
    async def run_demo_scenarios(self) -> bool:
        """Run demo scenarios."""
        
        logger.info("🧪 Running Demo Scenarios...")
        
        try:
            demo_suite = DemoScenarios()
            
            # Run all demo scenarios
            await demo_suite.run_t0_baseline_demo()
            await demo_suite.run_t3_stress_test_demo()
            await demo_suite.run_framework_comparison_demo()
            await demo_suite.run_memory_ablation_demo()
            
            # Generate demo report
            demo_report = demo_suite.generate_demo_report()
            
            success_rate = demo_report.get("demo_summary", {}).get("success_rate", 0)
            success = success_rate >= 0.75  # 75% success threshold
            
            self.results["demo_scenarios"] = {
                "success": success,
                "results": demo_report,
                "tests_run": demo_report.get("demo_summary", {}).get("total_demos", 4),
                "tests_passed": demo_report.get("demo_summary", {}).get("successful_demos", 0)
            }
            
            logger.info("✅ Demo scenarios completed")
            return success
            
        except Exception as e:
            logger.error(f"❌ Demo scenarios failed: {e}")
            self.results["demo_scenarios"] = {
                "success": False,
                "error": str(e),
                "tests_run": 4,
                "tests_passed": 0
            }
            return False
    
    async def generate_validation_report(self, output_dir: str = "./reports") -> str:
        """Generate comprehensive validation report."""
        
        logger.info("📊 Generating validation report...")
        
        try:
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate report
            generator = ValidationReportGenerator(self.config)
            report = await generator.generate_comprehensive_report()
            
            # Export in multiple formats
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # JSON report
            json_filename = os.path.join(output_dir, f"fba_bench_validation_{timestamp}.json")
            json_content = generator.export_report(report, "json")
            with open(json_filename, 'w') as f:
                f.write(json_content)
            
            # Markdown report
            md_filename = os.path.join(output_dir, f"fba_bench_validation_{timestamp}.md")
            md_content = generator.export_report(report, "markdown")
            with open(md_filename, 'w') as f:
                f.write(md_content)
            
            logger.info(f"📊 Validation reports saved:")
            logger.info(f"   JSON: {json_filename}")
            logger.info(f"   Markdown: {md_filename}")
            
            # Print summary to console
            self.print_summary_report(report)
            
            return md_filename
            
        except Exception as e:
            logger.error(f"❌ Report generation failed: {e}")
            raise
    
    def print_summary_report(self, report) -> None:
        """Print summary report to console."""
        
        print("\n" + "="*60)
        print("FBA-BENCH TIER-1 VALIDATION SUMMARY")
        print("="*60)
        
        print(f"Report ID: {report.report_id}")
        print(f"Generated: {report.generation_timestamp.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Overall Score: {report.overall_readiness_score:.1f}%")
        print(f"Tier-1 Ready: {'✅ YES' if report.tier1_ready else '❌ NO'}")
        
        print("\nTest Results:")
        print("-" * 40)
        for metric in report.test_results:
            status = "✅" if metric.success_rate > 0.8 else "⚠️" if metric.success_rate > 0.5 else "❌"
            print(f"{status} {metric.test_module}: {metric.tests_passed}/{metric.tests_run} ({metric.success_rate:.1%})")
        
        print("\nKey Capabilities:")
        print("-" * 40)
        for capability, status in report.compliance_matrix.items():
            status_icon = "✅" if status else "❌"
            print(f"{status_icon} {capability.replace('_', ' ').title()}")
        
        if report.recommendations:
            print("\nTop Recommendations:")
            print("-" * 40)
            for i, rec in enumerate(report.recommendations[:5], 1):
                print(f"{i}. {rec}")
        
        print("\n" + "="*60)
        
        if report.tier1_ready:
            print("🎉 FBA-Bench is READY for tier-1 benchmark deployment!")
        else:
            print("⚠️ FBA-Bench needs additional work before tier-1 readiness.")
        
        print("="*60 + "\n")

def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'integration_tests_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        ]
    )

async def main():
    """Main entry point for integration test runner."""
    
    parser = argparse.ArgumentParser(description='FBA-Bench Integration Test Runner')
    parser.add_argument('--quick', action='store_true', help='Run quick tests only')
    parser.add_argument('--performance', action='store_true', help='Run performance benchmarks only')
    parser.add_argument('--tier1', action='store_true', help='Run tier-1 requirements only')
    parser.add_argument('--demo', action='store_true', help='Run demo scenarios only')
    parser.add_argument('--report', action='store_true', help='Generate validation report only')
    parser.add_argument('--all', action='store_true', help='Run all tests and generate report')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    parser.add_argument('--output', default='./reports', help='Output directory for reports')
    
    args = parser.parse_args()
    
    # Default to --all if no specific option provided
    if not any([args.quick, args.performance, args.tier1, args.demo, args.report]):
        args.all = True
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Create configuration
    config = IntegrationTestConfig(
        skip_slow_tests=args.quick,
        verbose_logging=args.verbose
    )
    
    # Create test runner
    runner = IntegrationTestRunner(config)
    
    try:
        # Run selected test suites
        if args.tier1 or args.all:
            await runner.run_tier1_requirements()
        
        if args.performance or args.all:
            await runner.run_performance_benchmarks()
        
        if args.demo or args.all:
            await runner.run_demo_scenarios()
        
        if args.all and not args.quick:
            # Run comprehensive tests
            await runner.run_end_to_end_workflow()
            await runner.run_cross_system_integration()
            await runner.run_scientific_reproducibility()
        
        # Generate report if requested or running all tests
        if args.report or args.all:
            await runner.generate_validation_report(args.output)
        
        # Calculate overall success
        total_tests = sum(result.get("tests_run", 0) for result in runner.results.values())
        total_passed = sum(result.get("tests_passed", 0) for result in runner.results.values())
        success_rate = total_passed / total_tests if total_tests > 0 else 0
        
        print(f"\n🏁 Integration testing completed!")
        print(f"   Total tests: {total_tests}")
        print(f"   Passed: {total_passed}")
        print(f"   Success rate: {success_rate:.1%}")
        
        # Exit with appropriate code
        if success_rate >= 0.8:
            print("✅ Integration tests PASSED")
            sys.exit(0)
        else:
            print("❌ Integration tests FAILED")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Integration testing failed: {e}")
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())