import React, { useState } from 'react';
import type { ResultsData } from '../types';

interface AdvancedAnalyticsProps {
  resultsData: ResultsData | null; // Full results data for analysis
}

const AdvancedAnalytics: React.FC<AdvancedAnalyticsProps> = ({ resultsData }) => {
  const [analysisType, setAnalysisType] = useState<string>('correlation');
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>([]);
  const [analysisOutput, setAnalysisOutput] = useState<string>('Run an analysis to see results.');

  // Example available metrics for analysis
  const availableMetrics = ['revenue', 'profit', 'costs', 'decisionsMade', 'accuracy', 'priceTrend', 'inventoryLevels'];

  const handleRunAnalysis = () => {
    if (!resultsData) {
      setAnalysisOutput('No data available to run analysis.');
      return;
    }

    let output = '';
    switch (analysisType) {
      case 'correlation':
        output = runCorrelationAnalysis();
        break;
      case 'trend-forecasting':
        output = runTrendForecasting();
        break;
      case 'anomaly-detection':
        output = runAnomalyDetection();
        break;
      case 'regression-analysis':
        output = runPerformanceRegressionAnalysis();
        break;
      case 'cost-benefit':
        output = runCostBenefitAnalysis();
        break;
      default:
        output = 'Please select a valid analysis type.';
    }
    setAnalysisOutput(output);
  };

  const runCorrelationAnalysis = (): string => {
    if (selectedMetrics.length < 2) {
      return 'Please select at least two metrics for correlation analysis.';
    }
    // Placeholder for actual correlation calculation
    // In a real app, this would involve a complex algorithm or backend call
    const metric1 = selectedMetrics[0];
    const metric2 = selectedMetrics[1];
    const correlation = Math.random() * 2 - 1; // Between -1 and 1
    return `Correlation between ${metric1} and ${metric2}: ${correlation.toFixed(2)}. (Placeholder)`;
  };

  const runTrendForecasting = (): string => {
    // Placeholder for trend analysis and forecasting
    // Would analyze timeSeriesData
    return 'Trend analysis and forecasting results: (Placeholder)';
  };

  const runAnomalyDetection = (): string => {
    // Placeholder for anomaly detection
    return 'Anomaly detection results: No anomalies detected. (Placeholder)';
  };

  const runPerformanceRegressionAnalysis = (): string => {
    // Placeholder for performance regression analysis
    return 'Performance Regression Analysis: No significant regressions found. (Placeholder)';
  };

  const runCostBenefitAnalysis = (): string => {
    // Placeholder for cost-benefit analysis
    const totalBenefit = resultsData?.aggregatedMetrics.totalRevenue || 0;
    const totalCost = resultsData?.aggregatedMetrics.totalCosts || 0;
    const netBenefit = totalBenefit - totalCost;
    return `Cost-Benefit Analysis: Total Benefit: $${totalBenefit.toFixed(2)}, Total Cost: $${totalCost.toFixed(2)}, Net Benefit: $${netBenefit.toFixed(2)}. (Placeholder)`;
  };

  const handleMetricSelection = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const options = Array.from(e.target.selectedOptions).map(option => option.value);
    setSelectedMetrics(options);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Advanced Analytics</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Analysis Type Selection */}
        <div>
          <label htmlFor="analysisType" className="block text-sm font-medium text-gray-700 mb-1">Analysis Type</label>
          <select
            id="analysisType"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={analysisType}
            onChange={(e) => setAnalysisType(e.target.value)}
          >
            <option value="correlation">Statistical Correlation Analysis</option>
            <option value="trend-forecasting">Trend Analysis & Forecasting</option>
            <option value="anomaly-detection">Anomaly Detection</option>
            <option value="regression-analysis">Performance Regression Analysis</option>
            <option value="cost-benefit">Cost-Benefit Analysis</option>
          </select>
        </div>

        {/* Metric Selection for Correlation */}
        {analysisType === 'correlation' && (
          <div>
            <label htmlFor="selectedMetrics" className="block text-sm font-medium text-gray-700 mb-1">Select Metrics (for Correlation)</label>
            <select
              id="selectedMetrics"
              multiple
              className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md h-24"
              value={selectedMetrics}
              onChange={handleMetricSelection}
            >
              {availableMetrics.map((metric) => (
                <option key={metric} value={metric}>
                  {metric.charAt(0).toUpperCase() + metric.slice(1).replace(/([A-Z])/g, ' $1').trim()}
                </option>
              ))}
            </select>
            <p className="mt-2 text-sm text-gray-500">Hold Ctrl/Cmd to select multiple.</p>
          </div>
        )}
      </div>

      <button
        onClick={handleRunAnalysis}
        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
      >
        Run Analysis
      </button>

      {/* Analysis Output */}
      <div className="mt-8 p-4 bg-gray-50 border border-gray-200 rounded-md">
        <h3 className="text-lg font-semibold text-gray-800 mb-2">Analysis Output</h3>
        <pre className="whitespace-pre-wrap text-gray-700 text-sm">{analysisOutput}</pre>
      </div>
    </div>
  );
};

export default AdvancedAnalytics;