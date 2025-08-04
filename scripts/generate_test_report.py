#!/usr/bin/env python3
"""
Script to generate comprehensive test reports from various test result files.
"""

import os
import json
import xml.etree.ElementTree as ET
import argparse
from datetime import datetime
from pathlib import Path
import glob

def parse_junit_xml(xml_file):
    """Parse JUnit XML file and extract test results."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        tests = {
            'total': 0,
            'passed': 0,
            'failed': 0,
            'skipped': 0,
            'errors': 0,
            'time': 0.0,
            'test_cases': []
        }
        
        for testcase in root.findall('.//testcase'):
            tests['total'] += 1
            tests['time'] += float(testcase.get('time', 0))
            
            test_case = {
                'name': testcase.get('name'),
                'classname': testcase.get('classname'),
                'time': float(testcase.get('time', 0)),
                'status': 'passed'
            }
            
            # Check for failures
            failure = testcase.find('failure')
            if failure is not None:
                tests['failed'] += 1
                test_case['status'] = 'failed'
                test_case['failure_message'] = failure.get('message', '')
                test_case['failure_text'] = failure.text or ''
            
            # Check for errors
            error = testcase.find('error')
            if error is not None:
                tests['errors'] += 1
                test_case['status'] = 'error'
                test_case['error_message'] = error.get('message', '')
                test_case['error_text'] = error.text or ''
            
            # Check for skipped tests
            skipped = testcase.find('skipped')
            if skipped is not None:
                tests['skipped'] += 1
                test_case['status'] = 'skipped'
            
            tests['test_cases'].append(test_case)
        
        return tests
    except Exception as e:
        print(f"Error parsing {xml_file}: {e}")
        return None

def parse_coverage_xml(xml_file):
    """Parse coverage XML file and extract coverage data."""
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        coverage = {
            'line_rate': 0.0,
            'branch_rate': 0.0,
            'line_covered': 0,
            'line_valid': 0,
            'branch_covered': 0,
            'branch_valid': 0,
            'files': []
        }
        
        # Extract overall coverage
        coverage['line_rate'] = float(root.get('line-rate', 0))
        coverage['branch_rate'] = float(root.get('branch-rate', 0))
        coverage['line_covered'] = int(root.get('lines-covered', 0))
        coverage['line_valid'] = int(root.get('lines-valid', 0))
        coverage['branch_covered'] = int(root.get('branches-covered', 0))
        coverage['branch_valid'] = int(root.get('branches-valid', 0))
        
        # Extract per-file coverage
        for package in root.findall('.//package'):
            for cls in package.findall('.//class'):
                file_info = {
                    'name': cls.get('name', ''),
                    'line_rate': float(cls.get('line-rate', 0)),
                    'branch_rate': float(cls.get('branch-rate', 0)),
                    'line_covered': int(cls.get('lines-covered', 0)),
                    'line_valid': int(cls.get('lines-valid', 0))
                }
                coverage['files'].append(file_info)
        
        return coverage
    except Exception as e:
        print(f"Error parsing coverage {xml_file}: {e}")
        return None

def parse_benchmark_json(json_file):
    """Parse benchmark JSON file and extract benchmark results."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        benchmarks = {
            'benchmarks': [],
            'summary': {}
        }
        
        if 'benchmarks' in data:
            for benchmark in data['benchmarks']:
                benchmark_info = {
                    'name': benchmark.get('name', ''),
                    'group': benchmark.get('group', ''),
                    'stats': benchmark.get('stats', {}),
                    'params': benchmark.get('params', {}),
                    'options': benchmark.get('options', {})
                }
                benchmarks['benchmarks'].append(benchmark_info)
        
        # Calculate summary statistics
        if benchmarks['benchmarks']:
            all_times = [b['stats'].get('mean', 0) for b in benchmarks['benchmarks']]
            benchmarks['summary'] = {
                'min_time': min(all_times),
                'max_time': max(all_times),
                'avg_time': sum(all_times) / len(all_times),
                'total_benchmarks': len(benchmarks['benchmarks'])
            }
        
        return benchmarks
    except Exception as e:
        print(f"Error parsing benchmark {json_file}: {e}")
        return None

def parse_security_json(json_file):
    """Parse security scan JSON file and extract security findings."""
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        security = {
            'errors': 0,
            'warnings': 0,
            'info': 0,
            'findings': []
        }
        
        if isinstance(data, dict) and 'results' in data:
            for result in data['results']:
                severity = result.get('issue_severity', 'info').lower()
                if severity == 'high':
                    security['errors'] += 1
                elif severity == 'medium':
                    security['warnings'] += 1
                else:
                    security['info'] += 1
                
                finding = {
                    'code': result.get('code', ''),
                    'filename': result.get('filename', ''),
                    'line_number': result.get('line_number', 0),
                    'severity': severity,
                    'test_id': result.get('test_id', ''),
                    'issue_text': result.get('issue_text', '')
                }
                security['findings'].append(finding)
        
        return security
    except Exception as e:
        print(f"Error parsing security {json_file}: {e}")
        return None

def generate_html_report(report_data, output_file):
    """Generate HTML test report."""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FBA-Bench Test Report</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        h1 {
            color: #333;
            text-align: center;
            margin-bottom: 30px;
        }
        .summary {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .summary-card {
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            text-align: center;
        }
        .summary-card h3 {
            margin: 0 0 10px 0;
            color: #555;
        }
        .summary-card .value {
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        .summary-card.passed .value {
            color: #28a745;
        }
        .summary-card.failed .value {
            color: #dc3545;
        }
        .summary-card.warning .value {
            color: #ffc107;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .test-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        .test-table th, .test-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .test-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .test-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .status-passed {
            color: #28a745;
            font-weight: bold;
        }
        .status-failed {
            color: #dc3545;
            font-weight: bold;
        }
        .status-error {
            color: #fd7e14;
            font-weight: bold;
        }
        .status-skipped {
            color: #6c757d;
            font-weight: bold;
        }
        .coverage-bar {
            width: 100%;
            height: 20px;
            background-color: #e9ecef;
            border-radius: 10px;
            overflow: hidden;
        }
        .coverage-fill {
            height: 100%;
            background-color: #28a745;
            transition: width 0.3s ease;
        }
        .benchmark-table th, .benchmark-table td {
            padding: 8px;
            text-align: right;
        }
        .benchmark-table th:first-child,
        .benchmark-table td:first-child {
            text-align: left;
        }
        .security-finding {
            margin-bottom: 10px;
            padding: 10px;
            border-radius: 5px;
        }
        .security-finding.high {
            background-color: #f8d7da;
            border-left: 4px solid #dc3545;
        }
        .security-finding.medium {
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
        }
        .security-finding.low {
            background-color: #d1ecf1;
            border-left: 4px solid #17a2b8;
        }
        .timestamp {
            text-align: center;
            color: #666;
            font-size: 14px;
            margin-top: 30px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>FBA-Bench Test Report</h1>
        
        <div class="summary">
            <div class="summary-card">
                <h3>Total Tests</h3>
                <div class="value">{total_tests}</div>
            </div>
            <div class="summary-card passed">
                <h3>Passed</h3>
                <div class="value">{passed_tests}</div>
            </div>
            <div class="summary-card failed">
                <h3>Failed</h3>
                <div class="value">{failed_tests}</div>
            </div>
            <div class="summary-card">
                <h3>Pass Rate</h3>
                <div class="value">{pass_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>Coverage</h3>
                <div class="value">{coverage:.1f}%</div>
            </div>
            <div class="summary-card warning">
                <h3>Security Issues</h3>
                <div class="value">{security_issues}</div>
            </div>
        </div>
        
        {test_sections}
        
        <div class="timestamp">Generated on {timestamp}</div>
    </div>
</body>
</html>
    """
    
    # Calculate overall statistics
    total_tests = sum(data.get('total', 0) for data in report_data.get('test_results', {}).values())
    passed_tests = sum(data.get('passed', 0) for data in report_data.get('test_results', {}).values())
    failed_tests = sum(data.get('failed', 0) for data in report_data.get('test_results', {}).values())
    pass_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    # Get coverage data
    coverage_data = report_data.get('coverage', {})
    coverage = coverage_data.get('line_rate', 0) * 100
    
    # Get security data
    security_data = report_data.get('security', {})
    security_issues = security_data.get('errors', 0) + security_data.get('warnings', 0)
    
    # Generate test sections
    test_sections = ""
    
    # Unit Tests Section
    if 'unit' in report_data.get('test_results', {}):
        unit_data = report_data['test_results']['unit']
        test_sections += f"""
        <div class="section">
            <h2>Unit Tests</h2>
            <table class="test-table">
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for test_case in unit_data.get('test_cases', []):
            status_class = f"status-{test_case['status']}"
            test_sections += f"""
                    <tr>
                        <td>{test_case['name']}</td>
                        <td class="{status_class}">{test_case['status'].upper()}</td>
                        <td>{test_case['time']:.3f}</td>
                    </tr>
            """
        
        test_sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Integration Tests Section
    if 'integration' in report_data.get('test_results', {}):
        integration_data = report_data['test_results']['integration']
        test_sections += f"""
        <div class="section">
            <h2>Integration Tests</h2>
            <table class="test-table">
                <thead>
                    <tr>
                        <th>Test Name</th>
                        <th>Status</th>
                        <th>Time (s)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for test_case in integration_data.get('test_cases', []):
            status_class = f"status-{test_case['status']}"
            test_sections += f"""
                    <tr>
                        <td>{test_case['name']}</td>
                        <td class="{status_class}">{test_case['status'].upper()}</td>
                        <td>{test_case['time']:.3f}</td>
                    </tr>
            """
        
        test_sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Coverage Section
    if coverage_data:
        test_sections += f"""
        <div class="section">
            <h2>Code Coverage</h2>
            <p>Line Coverage: {coverage:.1f}%</p>
            <div class="coverage-bar">
                <div class="coverage-fill" style="width: {coverage:.1f}%"></div>
            </div>
            <table class="test-table">
                <thead>
                    <tr>
                        <th>File</th>
                        <th>Line Coverage</th>
                        <th>Lines Covered</th>
                        <th>Total Lines</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for file_info in coverage_data.get('files', []):
            file_coverage = file_info['line_rate'] * 100
            test_sections += f"""
                    <tr>
                        <td>{file_info['name']}</td>
                        <td>{file_coverage:.1f}%</td>
                        <td>{file_info['line_covered']}</td>
                        <td>{file_info['line_valid']}</td>
                    </tr>
            """
        
        test_sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Security Section
    if security_data and security_data.get('findings'):
        test_sections += """
        <div class="section">
            <h2>Security Findings</h2>
        """
        
        for finding in security_data['findings']:
            severity_class = finding['severity']
            test_sections += f"""
            <div class="security-finding {severity_class}">
                <strong>{finding['filename']}:{finding['line_number']}</strong> - {finding['test_id']}<br>
                {finding['issue_text']}
            </div>
            """
        
        test_sections += """
        </div>
        """
    
    # Benchmark Section
    if 'benchmarks' in report_data:
        benchmark_data = report_data['benchmarks']
        test_sections += """
        <div class="section">
            <h2>Performance Benchmarks</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Mean Time (s)</th>
                        <th>Min Time (s)</th>
                        <th>Max Time (s)</th>
                        <th>Std Dev</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for benchmark in benchmark_data.get('benchmarks', []):
            stats = benchmark.get('stats', {})
            test_sections += f"""
                    <tr>
                        <td>{benchmark['name']}</td>
                        <td>{stats.get('mean', 0):.6f}</td>
                        <td>{stats.get('min', 0):.6f}</td>
                        <td>{stats.get('max', 0):.6f}</td>
                        <td>{stats.get('stddev', 0):.6f}</td>
                    </tr>
            """
        
        test_sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Generate the final HTML
    html_content = html_template.format(
        total_tests=total_tests,
        passed_tests=passed_tests,
        failed_tests=failed_tests,
        pass_rate=pass_rate,
        coverage=coverage,
        security_issues=security_issues,
        test_sections=test_sections,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Generate comprehensive test report')
    parser.add_argument('--output', '-o', default='test-report.html', help='Output HTML file')
    args = parser.parse_args()
    
    # Initialize report data
    report_data = {
        'test_results': {},
        'coverage': {},
        'benchmarks': {},
        'security': {}
    }
    
    # Parse test results
    print("Parsing test results...")
    
    # Parse unit test results
    unit_files = glob.glob('python-unit-results.xml')
    if unit_files:
        for file in unit_files:
            result = parse_junit_xml(file)
            if result:
                report_data['test_results']['unit'] = result
    
    # Parse integration test results
    integration_files = glob.glob('python-integration-results.xml')
    if integration_files:
        for file in integration_files:
            result = parse_junit_xml(file)
            if result:
                report_data['test_results']['integration'] = result
    
    # Parse validation test results
    validation_files = glob.glob('python-validation-results.xml')
    if validation_files:
        for file in validation_files:
            result = parse_junit_xml(file)
            if result:
                report_data['test_results']['validation'] = result
    
    # Parse coverage data
    coverage_files = glob.glob('coverage.xml')
    if coverage_files:
        for file in coverage_files:
            result = parse_coverage_xml(file)
            if result:
                report_data['coverage'] = result
    
    # Parse benchmark data
    benchmark_files = glob.glob('benchmark-results.json')
    if benchmark_files:
        for file in benchmark_files:
            result = parse_benchmark_json(file)
            if result:
                report_data['benchmarks'] = result
    
    # Parse security data
    security_files = glob.glob('bandit-report.json')
    if security_files:
        for file in security_files:
            result = parse_security_json(file)
            if result:
                report_data['security'] = result
    
    # Generate HTML report
    print(f"Generating HTML report: {args.output}")
    generate_html_report(report_data, args.output)
    
    # Generate JSON summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'test_results': report_data['test_results'],
        'coverage': report_data['coverage'],
        'benchmarks': report_data['benchmarks'],
        'security': report_data['security']
    }
    
    with open('test-summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("Test report generation completed!")

if __name__ == '__main__':
    main()