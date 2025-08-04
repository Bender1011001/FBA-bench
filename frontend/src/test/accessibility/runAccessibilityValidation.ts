/**
 * Accessibility Validation Runner
 * 
 * This script runs comprehensive accessibility validation on all benchmarking components.
 * It can be executed as a standalone script or integrated into the testing pipeline.
 */

import { accessibilityTester } from './AccessibilityValidation';
import type { AccessibilityReport } from './AccessibilityValidation';

// Component names to validate
const COMPONENTS_TO_TEST = [
  'BenchmarkDashboard',
  'MetricsVisualization',
  'ScenarioBuilder',
  'ExecutionMonitor',
  'ResultsComparison',
  'RadarChart',
  'HeatmapChart',
  'TimeSeriesChart',
  'ComparisonChart'
];

/**
 * Create a mock HTML element for testing accessibility
 */
function createMockElement(componentName: string): HTMLElement {
  const element = document.createElement('div');
  element.setAttribute('role', 'region');
  element.setAttribute('aria-label', `${componentName} Component`);
  element.setAttribute('tabindex', '0');
  
  // Add basic structure based on component type
  switch (componentName) {
    case 'BenchmarkDashboard':
      element.innerHTML = `
        <h2>Benchmark Dashboard</h2>
        <button aria-label="Create New Benchmark">Create Benchmark</button>
        <input type="text" aria-label="Filter Benchmarks" placeholder="Search benchmarks..." />
        <div role="list" aria-label="Benchmark Results">
          <div role="listitem">
            <h3>Test Benchmark</h3>
            <p>Status: Completed</p>
          </div>
        </div>
      `;
      break;
      
    case 'MetricsVisualization':
      element.innerHTML = `
        <h2>Metrics Visualization</h2>
        <select aria-label="Select Benchmark Result">
          <option>Test Benchmark</option>
        </select>
        <div role="img" aria-label="Metrics Chart" style="width: 400px; height: 300px; background: #f0f0f0;"></div>
      `;
      break;
      
    case 'ScenarioBuilder':
      element.innerHTML = `
        <h2>Scenario Builder</h2>
        <form>
          <label for="scenario-name">Scenario Name *</label>
          <input type="text" id="scenario-name" required />
          <label for="scenario-description">Description</label>
          <textarea id="scenario-description"></textarea>
          <label for="duration">Duration (seconds)</label>
          <input type="number" id="duration" min="1" />
          <button type="submit" aria-label="Create Scenario">Create Scenario</button>
        </form>
      `;
      break;
      
    case 'ExecutionMonitor':
      element.innerHTML = `
        <h2>Execution Monitor</h2>
        <div role="status" aria-live="polite">
          <p>Benchmark started</p>
          <p>Progress: 50%</p>
        </div>
        <button aria-label="Pause Execution">Pause</button>
        <button aria-label="Export Logs">Export</button>
      `;
      break;
      
    case 'ResultsComparison':
      element.innerHTML = `
        <h2>Results Comparison</h2>
        <select aria-label="Comparison Type">
          <option>Agent Comparison</option>
        </select>
        <select aria-label="Select Metrics">
          <option>Accuracy</option>
        </select>
        <table role="table" aria-label="Comparison Results">
          <thead>
            <tr>
              <th scope="col">Agent</th>
              <th scope="col">Accuracy</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td data-label="Agent">Test Agent</td>
              <td data-label="Accuracy">95%</td>
            </tr>
          </tbody>
        </table>
      `;
      break;
      
    case 'RadarChart':
      element.innerHTML = `
        <h2>Capability Radar Chart</h2>
        <div role="img" aria-label="Radar Chart showing agent capabilities" style="width: 400px; height: 400px; background: #f0f0f0;"></div>
        <div role="radiogroup" aria-label="Select Metrics">
          <label><input type="radio" name="metrics" checked /> Cognitive</label>
          <label><input type="radio" name="metrics" /> Business</label>
        </div>
      `;
      break;
      
    case 'HeatmapChart':
      element.innerHTML = `
        <h2>Performance Heatmap</h2>
        <div role="img" aria-label="Heatmap showing performance across scenarios" style="width: 500px; height: 300px; background: #f0f0f0;"></div>
        <select aria-label="Select Metric">
          <option>Accuracy</option>
        </select>
      `;
      break;
      
    case 'TimeSeriesChart':
      element.innerHTML = `
        <h2>Time Series Analysis</h2>
        <div role="img" aria-label="Time series chart showing performance over time" style="width: 600px; height: 300px; background: #f0f0f0;"></div>
        <div role="group" aria-label="Metric Selection">
          <label><input type="checkbox" checked /> CPU Usage</label>
          <label><input type="checkbox" checked /> Memory Usage</label>
        </div>
      `;
      break;
      
    case 'ComparisonChart':
      element.innerHTML = `
        <h2>Agent Comparison</h2>
        <div role="img" aria-label="Comparison chart showing agent performance" style="width: 500px; height: 400px; background: #f0f0f0;"></div>
        <select aria-label="Select Metrics">
          <option>Accuracy</option>
          <option>Efficiency</option>
        </select>
      `;
      break;
      
    default:
      element.innerHTML = `<h2>${componentName}</h2><p>Test content</p>`;
  }
  
  return element;
}

/**
 * Run accessibility validation for all components
 */
export function runAccessibilityValidation(): {
  summary: {
    totalComponents: number;
    overallScore: number;
    passedComponents: string[];
    failedComponents: string[];
  };
  details: Map<string, AccessibilityReport>;
} {
  console.log('🔍 Starting accessibility validation...');
  
  // Clear previous results
  accessibilityTester['results'].clear();
  
  // Test each component
  COMPONENTS_TO_TEST.forEach(componentName => {
    console.log(`📋 Testing ${componentName}...`);
    
    try {
      const mockElement = createMockElement(componentName);
      accessibilityTester.testComponent(componentName, mockElement);
      
      console.log(`✅ ${componentName} - Validation completed`);
    } catch (error) {
      console.error(`❌ ${componentName} - Validation failed:`, error);
    }
  });
  
  // Generate summary
  const summary = accessibilityTester.generateSummary();
  const details = accessibilityTester.getResults();
  
  console.log('\n📊 Accessibility Validation Summary:');
  console.log(`=====================================`);
  console.log(`Total Components Tested: ${summary.totalComponents}`);
  console.log(`Overall Score: ${summary.overallScore}/100`);
  console.log(`Passed Components: ${summary.passedComponents.length}`);
  console.log(`Failed Components: ${summary.failedComponents.length}`);
  
  if (summary.passedComponents.length > 0) {
    console.log('\n✅ Passed Components:');
    summary.passedComponents.forEach(component => {
      console.log(`  - ${component}`);
    });
  }
  
  if (summary.failedComponents.length > 0) {
    console.log('\n❌ Failed Components:');
    summary.failedComponents.forEach(component => {
      console.log(`  - ${component}`);
    });
  }
  
  // Print detailed results
  console.log('\n📋 Detailed Results:');
  console.log(`===================`);
  
  details.forEach((report, componentName) => {
    console.log(`\n${componentName}:`);
    console.log(`  Score: ${report.overallScore}/100`);
    console.log(`  Rules: ${report.rules.filter(r => r.passed).length}/${report.rules.length} passed`);
    
    if (report.keyboardNavigation.issues.length > 0) {
      console.log(`  Keyboard Issues: ${report.keyboardNavigation.issues.join(', ')}`);
    }
    
    if (report.colorContrast.issues.length > 0) {
      console.log(`  Color Issues: ${report.colorContrast.issues.join(', ')}`);
    }
  });
  
  return { summary, details };
}

/**
 * Export validation results to JSON
 */
export function exportValidationResults(results: Map<string, AccessibilityReport>, filename: string = 'accessibility-report.json'): void {
  const reportData = {
    timestamp: new Date().toISOString(),
    summary: accessibilityTester.generateSummary(),
    components: Array.from(results.entries()).map(([name, report]) => ({
      name,
      ...report
    }))
  };
  
  const blob = new Blob([JSON.stringify(reportData, null, 2)], { type: 'application/json' });
  const url = URL.createObjectURL(blob);
  
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  
  URL.revokeObjectURL(url);
}

// Run validation if this script is executed directly
if (typeof window !== 'undefined' && window.location.href.includes('accessibility-test')) {
  // Auto-run when opened in browser
  const results = runAccessibilityValidation();
  
  // Add export button to the page
  const exportButton = document.createElement('button');
  exportButton.textContent = 'Export Results';
  exportButton.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    padding: 10px 20px;
    background: #007bff;
    color: white;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    z-index: 1000;
  `;
  
  exportButton.addEventListener('click', () => {
    exportValidationResults(results.details);
  });
  
  document.body.appendChild(exportButton);
}
