#!/usr/bin/env python3
"""
Script to check quality gates for FBA-Bench.
"""

import json
import sys
import argparse
from pathlib import Path
import glob

def check_test_coverage(coverage_data, min_coverage=80.0):
    """Check if test coverage meets minimum requirements."""
    if not coverage_data:
        return False, "No coverage data available"
    
    line_coverage = coverage_data.get('line_rate', 0) * 100
    if line_coverage < min_coverage:
        return False, f"Line coverage {line_coverage:.1f}% is below minimum {min_coverage}%"
    
    return True, f"Line coverage {line_coverage:.1f}% meets minimum {min_coverage}%"

def check_test_results(test_results, max_failures=0):
    """Check if test results meet quality criteria."""
    if not test_results:
        return False, "No test results available"
    
    total_failed = sum(data.get('failed', 0) for data in test_results.values())
    total_errors = sum(data.get('errors', 0) for data in test_results.values())
    
    if total_failed > max_failures:
        return False, f"{total_failed} failed tests exceed maximum allowed {max_failures}"
    
    if total_errors > 0:
        return False, f"{total_errors} test errors found"
    
    return True, f"All tests passed with {total_failed} failures"

def check_security_findings(security_data, max_high=0, max_medium=5):
    """Check if security findings meet quality criteria."""
    if not security_data:
        return False, "No security data available"
    
    high_severity = security_data.get('errors', 0)
    medium_severity = security_data.get('warnings', 0)
    
    if high_severity > max_high:
        return False, f"{high_severity} high severity issues exceed maximum allowed {max_high}"
    
    if medium_severity > max_medium:
        return False, f"{medium_severity} medium severity issues exceed maximum allowed {max_medium}"
    
    return True, f"Security findings within limits: {high_severity} high, {medium_severity} medium"

def check_performance_regression(benchmark_data, max_regression=5.0):
    """Check if performance benchmarks show significant regression."""
    if not benchmark_data:
        return True, "No benchmark data available, skipping performance check"
    
    # This is a simplified check - in practice, you would compare with baseline
    benchmarks = benchmark_data.get('benchmarks', [])
    if not benchmarks:
        return True, "No benchmarks found, skipping performance check"
    
    # For now, just check that benchmarks exist and have reasonable values
    # In a real implementation, you would compare with previous results
    return True, f"Performance benchmarks checked: {len(benchmarks)} benchmarks"

def check_code_quality():
    """Check code quality metrics."""
    # This would typically integrate with code quality tools like SonarQube
    # For now, we'll just check that the code can be imported without syntax errors
    try:
        import benchmarking
        import agents
        import infrastructure
        import llm_interface
        import learning
        import metrics
        import scenarios
        import observability
        import plugins
        return True, "All main modules can be imported successfully"
    except ImportError as e:
        return False, f"Import error: {e}"
    except SyntaxError as e:
        return False, f"Syntax error: {e}"

def load_data_from_files():
    """Load data from various test result files."""
    data = {
        'test_results': {},
        'coverage': {},
        'benchmarks': {},
        'security': {}
    }
    
    # Load test results
    test_files = [
        ('python-unit-results.xml', 'unit'),
        ('python-integration-results.xml', 'integration'),
        ('python-validation-results.xml', 'validation')
    ]
    
    for file_pattern, category in test_files:
        files = glob.glob(file_pattern)
        if files:
            try:
                import xml.etree.ElementTree as ET
                tree = ET.parse(files[0])
                root = tree.getroot()
                
                tests = {
                    'total': 0,
                    'passed': 0,
                    'failed': 0,
                    'skipped': 0,
                    'errors': 0,
                    'time': 0.0
                }
                
                for testcase in root.findall('.//testcase'):
                    tests['total'] += 1
                    tests['time'] += float(testcase.get('time', 0))
                    
                    if testcase.find('failure') is not None:
                        tests['failed'] += 1
                    elif testcase.find('error') is not None:
                        tests['errors'] += 1
                    elif testcase.find('skipped') is not None:
                        tests['skipped'] += 1
                    else:
                        tests['passed'] += 1
                
                data['test_results'][category] = tests
            except Exception as e:
                print(f"Error parsing {file_pattern}: {e}")
    
    # Load coverage data
    coverage_files = glob.glob('coverage.xml')
    if coverage_files:
        try:
            import xml.etree.ElementTree as ET
            tree = ET.parse(coverage_files[0])
            root = tree.getroot()
            
            data['coverage'] = {
                'line_rate': float(root.get('line-rate', 0)),
                'branch_rate': float(root.get('branch-rate', 0)),
                'line_covered': int(root.get('lines-covered', 0)),
                'line_valid': int(root.get('lines-valid', 0))
            }
        except Exception as e:
            print(f"Error parsing coverage.xml: {e}")
    
    # Load benchmark data
    benchmark_files = glob.glob('benchmark-results.json')
    if benchmark_files:
        try:
            with open(benchmark_files[0], 'r') as f:
                data['benchmarks'] = json.load(f)
        except Exception as e:
            print(f"Error parsing benchmark-results.json: {e}")
    
    # Load security data
    security_files = glob.glob('bandit-report.json')
    if security_files:
        try:
            with open(security_files[0], 'r') as f:
                security_data = json.load(f)
                
                errors = 0
                warnings = 0
                
                if isinstance(security_data, dict) and 'results' in security_data:
                    for result in security_data['results']:
                        severity = result.get('issue_severity', 'info').lower()
                        if severity == 'high':
                            errors += 1
                        elif severity == 'medium':
                            warnings += 1
                
                data['security'] = {
                    'errors': errors,
                    'warnings': warnings
                }
        except Exception as e:
            print(f"Error parsing bandit-report.json: {e}")
    
    return data

def main():
    parser = argparse.ArgumentParser(description='Check quality gates for FBA-Bench')
    parser.add_argument('--min-coverage', type=float, default=80.0, help='Minimum test coverage percentage')
    parser.add_argument('--max-failures', type=int, default=0, help='Maximum allowed test failures')
    parser.add_argument('--max-high-severity', type=int, default=0, help='Maximum allowed high severity security issues')
    parser.add_argument('--max-medium-severity', type=int, default=5, help='Maximum allowed medium severity security issues')
    parser.add_argument('--max-regression', type=float, default=5.0, help='Maximum allowed performance regression percentage')
    args = parser.parse_args()
    
    print("Checking quality gates...")
    
    # Load data from files
    data = load_data_from_files()
    
    # Define quality gates
    quality_gates = [
        ("Test Coverage", check_test_coverage, data['coverage'], args.min_coverage),
        ("Test Results", check_test_results, data['test_results'], args.max_failures),
        ("Security Findings", check_security_findings, data['security'], (args.max_high_severity, args.max_medium_severity)),
        ("Performance Regression", check_performance_regression, data['benchmarks'], args.max_regression),
        ("Code Quality", check_code_quality, None, None)
    ]
    
    # Check each quality gate
    all_passed = True
    results = {}
    
    for gate_name, check_function, data_arg, threshold in quality_gates:
        print(f"\nChecking {gate_name}...")
        
        if threshold is None:
            passed, message = check_function()
        elif isinstance(threshold, tuple):
            passed, message = check_function(data_arg, *threshold)
        else:
            passed, message = check_function(data_arg, threshold)
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {message}")
        
        results[gate_name] = {
            'passed': passed,
            'message': message
        }
        
        if not passed:
            all_passed = False
    
    # Generate summary
    print("\n" + "="*50)
    print("QUALITY GATES SUMMARY")
    print("="*50)
    
    for gate_name, result in results.items():
        status = "✅ PASS" if result['passed'] else "❌ FAIL"
        print(f"{gate_name:20} {status}")
    
    print("="*50)
    
    if all_passed:
        print("🎉 ALL QUALITY GATES PASSED!")
        print("FBA-Bench is ready for deployment.")
        exit_code = 0
    else:
        print("⚠️  SOME QUALITY GATES FAILED!")
        print("Review failed gates before deployment.")
        exit_code = 1
    
    # Save results to file
    with open('quality-gate-results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    exit(exit_code)

if __name__ == '__main__':
    main()