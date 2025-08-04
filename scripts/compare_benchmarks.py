#!/usr/bin/env python3
"""
Script to compare benchmark results and detect performance regressions.
"""

import json
import argparse
import sys
from pathlib import Path
import statistics
from datetime import datetime

def load_benchmark_results(file_path):
    """Load benchmark results from JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading benchmark results from {file_path}: {e}")
        return None

def extract_benchmark_stats(benchmark_data):
    """Extract statistics from benchmark data."""
    stats = {}
    
    if 'benchmarks' in benchmark_data:
        for benchmark in benchmark_data['benchmarks']:
            name = benchmark.get('name', '')
            benchmark_stats = benchmark.get('stats', {})
            
            if name:
                stats[name] = {
                    'mean': benchmark_stats.get('mean', 0),
                    'min': benchmark_stats.get('min', 0),
                    'max': benchmark_stats.get('max', 0),
                    'stddev': benchmark_stats.get('stddev', 0),
                    'rounds': benchmark_stats.get('rounds', 0),
                    'iterations': benchmark_stats.get('iterations', 0)
                }
    
    return stats

def compare_benchmarks(current_stats, baseline_stats, threshold=5.0):
    """Compare current benchmark stats with baseline and detect regressions."""
    results = {
        'regressions': [],
        'improvements': [],
        'unchanged': [],
        'missing': [],
        'new': []
    }
    
    # Check for regressions and improvements
    for name, current in current_stats.items():
        if name in baseline_stats:
            baseline = baseline_stats[name]
            
            # Calculate percentage change
            if baseline['mean'] > 0:
                change_pct = ((current['mean'] - baseline['mean']) / baseline['mean']) * 100
            else:
                change_pct = 0
            
            comparison = {
                'name': name,
                'current_mean': current['mean'],
                'baseline_mean': baseline['mean'],
                'change_pct': change_pct,
                'threshold': threshold
            }
            
            if change_pct > threshold:
                results['regressions'].append(comparison)
            elif change_pct < -threshold:
                results['improvements'].append(comparison)
            else:
                results['unchanged'].append(comparison)
        else:
            results['new'].append({
                'name': name,
                'current_mean': current['mean']
            })
    
    # Check for missing benchmarks
    for name in baseline_stats:
        if name not in current_stats:
            results['missing'].append({
                'name': name,
                'baseline_mean': baseline_stats[name]['mean']
            })
    
    return results

def generate_html_report(comparison_results, output_file, threshold=5.0):
    """Generate HTML report for benchmark comparison."""
    html_template = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>FBA-Bench Performance Regression Report</title>
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
        }
        .summary-card.regression .value {
            color: #dc3545;
        }
        .summary-card.improvement .value {
            color: #28a745;
        }
        .summary-card.unchanged .value {
            color: #6c757d;
        }
        .summary-card.missing .value {
            color: #ffc107;
        }
        .summary-card.new .value {
            color: #17a2b8;
        }
        .section {
            margin-bottom: 30px;
        }
        .section h2 {
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }
        .benchmark-table {
            width: 100%;
            border-collapse: collapse;
            margin-bottom: 20px;
        }
        .benchmark-table th, .benchmark-table td {
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }
        .benchmark-table th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .benchmark-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .change-positive {
            color: #dc3545;
            font-weight: bold;
        }
        .change-negative {
            color: #28a745;
            font-weight: bold;
        }
        .change-neutral {
            color: #6c757d;
        }
        .threshold-info {
            background-color: #e9ecef;
            padding: 10px;
            border-radius: 5px;
            margin-bottom: 20px;
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
        <h1>FBA-Bench Performance Regression Report</h1>
        
        <div class="threshold-info">
            <strong>Regression Threshold:</strong> {threshold}% (any change greater than this value is flagged)
        </div>
        
        <div class="summary">
            <div class="summary-card regression">
                <h3>Regressions</h3>
                <div class="value">{regressions_count}</div>
            </div>
            <div class="summary-card improvement">
                <h3>Improvements</h3>
                <div class="value">{improvements_count}</div>
            </div>
            <div class="summary-card unchanged">
                <h3>Unchanged</h3>
                <div class="value">{unchanged_count}</div>
            </div>
            <div class="summary-card missing">
                <h3>Missing</h3>
                <div class="value">{missing_count}</div>
            </div>
            <div class="summary-card new">
                <h3>New</h3>
                <div class="value">{new_count}</div>
            </div>
        </div>
        
        {sections}
        
        <div class="timestamp">Generated on {timestamp}</div>
    </div>
</body>
</html>
    """
    
    # Count benchmarks in each category
    regressions_count = len(comparison_results['regressions'])
    improvements_count = len(comparison_results['improvements'])
    unchanged_count = len(comparison_results['unchanged'])
    missing_count = len(comparison_results['missing'])
    new_count = len(comparison_results['new'])
    
    # Generate sections
    sections = ""
    
    # Regressions section
    if comparison_results['regressions']:
        sections += """
        <div class="section">
            <h2>Performance Regressions</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Current Mean (s)</th>
                        <th>Baseline Mean (s)</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for regression in comparison_results['regressions']:
            change_class = "change-positive"
            sections += f"""
                    <tr>
                        <td>{regression['name']}</td>
                        <td>{regression['current_mean']:.6f}</td>
                        <td>{regression['baseline_mean']:.6f}</td>
                        <td class="{change_class}">+{regression['change_pct']:.2f}%</td>
                    </tr>
            """
        
        sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Improvements section
    if comparison_results['improvements']:
        sections += """
        <div class="section">
            <h2>Performance Improvements</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Current Mean (s)</th>
                        <th>Baseline Mean (s)</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for improvement in comparison_results['improvements']:
            change_class = "change-negative"
            sections += f"""
                    <tr>
                        <td>{improvement['name']}</td>
                        <td>{improvement['current_mean']:.6f}</td>
                        <td>{improvement['baseline_mean']:.6f}</td>
                        <td class="{change_class}">{improvement['change_pct']:.2f}%</td>
                    </tr>
            """
        
        sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Unchanged section
    if comparison_results['unchanged']:
        sections += """
        <div class="section">
            <h2>Unchanged Benchmarks</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Current Mean (s)</th>
                        <th>Baseline Mean (s)</th>
                        <th>Change</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for unchanged in comparison_results['unchanged']:
            change_class = "change-neutral"
            sections += f"""
                    <tr>
                        <td>{unchanged['name']}</td>
                        <td>{unchanged['current_mean']:.6f}</td>
                        <td>{unchanged['baseline_mean']:.6f}</td>
                        <td class="{change_class}">{unchanged['change_pct']:.2f}%</td>
                    </tr>
            """
        
        sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Missing benchmarks section
    if comparison_results['missing']:
        sections += """
        <div class="section">
            <h2>Missing Benchmarks</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Baseline Mean (s)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for missing in comparison_results['missing']:
            sections += f"""
                    <tr>
                        <td>{missing['name']}</td>
                        <td>{missing['baseline_mean']:.6f}</td>
                    </tr>
            """
        
        sections += """
                </tbody>
            </table>
        </div>
        """
    
    # New benchmarks section
    if comparison_results['new']:
        sections += """
        <div class="section">
            <h2>New Benchmarks</h2>
            <table class="benchmark-table">
                <thead>
                    <tr>
                        <th>Benchmark</th>
                        <th>Current Mean (s)</th>
                    </tr>
                </thead>
                <tbody>
        """
        
        for new in comparison_results['new']:
            sections += f"""
                    <tr>
                        <td>{new['name']}</td>
                        <td>{new['current_mean']:.6f}</td>
                    </tr>
            """
        
        sections += """
                </tbody>
            </table>
        </div>
        """
    
    # Generate the final HTML
    html_content = html_template.format(
        threshold=threshold,
        regressions_count=regressions_count,
        improvements_count=improvements_count,
        unchanged_count=unchanged_count,
        missing_count=missing_count,
        new_count=new_count,
        sections=sections,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    with open(output_file, 'w') as f:
        f.write(html_content)

def main():
    parser = argparse.ArgumentParser(description='Compare benchmark results and detect performance regressions')
    parser.add_argument('current_file', help='Current benchmark results file')
    parser.add_argument('--baseline', '-b', default='baseline-benchmark-results.json', help='Baseline benchmark results file')
    parser.add_argument('--output', '-o', default='benchmark-comparison-report.html', help='Output HTML report file')
    parser.add_argument('--threshold', '-t', type=float, default=5.0, help='Regression threshold percentage')
    parser.add_argument('--update-baseline', '-u', action='store_true', help='Update baseline with current results')
    args = parser.parse_args()
    
    print(f"Loading current benchmark results from {args.current_file}...")
    current_data = load_benchmark_results(args.current_file)
    
    if not current_data:
        print("Failed to load current benchmark results")
        sys.exit(1)
    
    current_stats = extract_benchmark_stats(current_data)
    
    # Try to load baseline
    baseline_stats = {}
    if Path(args.baseline).exists():
        print(f"Loading baseline benchmark results from {args.baseline}...")
        baseline_data = load_benchmark_results(args.baseline)
        if baseline_data:
            baseline_stats = extract_benchmark_stats(baseline_data)
    else:
        print(f"Baseline file {args.baseline} not found, creating new baseline...")
    
    # Compare benchmarks
    print("Comparing benchmarks...")
    comparison_results = compare_benchmarks(current_stats, baseline_stats, args.threshold)
    
    # Generate report
    print(f"Generating HTML report: {args.output}")
    generate_html_report(comparison_results, args.output, args.threshold)
    
    # Print summary
    print("\nBenchmark Comparison Summary:")
    print(f"  Regressions: {len(comparison_results['regressions'])}")
    print(f"  Improvements: {len(comparison_results['improvements'])}")
    print(f"  Unchanged: {len(comparison_results['unchanged'])}")
    print(f"  Missing: {len(comparison_results['missing'])}")
    print(f"  New: {len(comparison_results['new'])}")
    
    # Exit with error code if there are regressions
    if comparison_results['regressions']:
        print("\n⚠️  Performance regressions detected!")
        print("Review the benchmark comparison report for details.")
        sys.exit(1)
    else:
        print("\n✅ No performance regressions detected.")
    
    # Update baseline if requested
    if args.update_baseline:
        print(f"\nUpdating baseline file: {args.baseline}")
        with open(args.baseline, 'w') as f:
            json.dump(current_data, f, indent=2)
        print("Baseline updated successfully.")

if __name__ == '__main__':
    main()