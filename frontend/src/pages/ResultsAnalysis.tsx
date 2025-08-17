import React, { useState, useEffect } from 'react';
import type { ResultsData, ExperimentExecution } from '../types';
import { fetchResultsData, fetchExperimentDetails, fetchExperimentList } from '../services/apiService';

// Import the actual components
import FinancialChart from '../components/charts/FinancialChart';
import AgentComparisonChart from '../components/charts/AgentComparisonChart';
import DataExporter from '../components/DataExporter';
import ResultsFilter from '../components/ResultsFilter';
import ComparisonTool from '../components/ComparisonTool';
import MetricWidget from '../components/widgets/MetricWidget';
import SummaryWidget from '../components/widgets/SummaryWidget';
import AdvancedAnalytics from '../components/AdvancedAnalytics';
import TimeSeriesChart from '../components/charts/TimeSeriesChart';

const ResultsAnalysis: React.FC = () => {
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [experiment, setExperiment] = useState<ExperimentExecution | null>(null);
  const [experimentId, setExperimentId] = useState<string>('example-experiment-123');
  const [experiments, setExperiments] = useState<ExperimentExecution[]>([]);
  const [selectedMetrics, setSelectedMetrics] = useState<string[]>(['revenue', 'profit', 'costs']);
  const [timeRange, setTimeRange] = useState<{ start: string; end: string } | null>(null);
  const [chartType, setChartType] = useState<'line' | 'bar' | 'area' | 'scatter'>('line');

  useEffect(() => {
    const loadData = async () => {
      try {
        setLoading(true);
        
        // Fetch experiment list
        const expList = await fetchExperimentList();
        setExperiments(expList);
        
        // If we have experiments, use the first one or the one from URL
        if (expList.length > 0) {
          const selectedExp = expList.find(exp => exp.id === experimentId) || expList[0];
          setExperimentId(selectedExp.id);
          
          // Fetch detailed results data
          const data = await fetchResultsData(selectedExp.id);
          setResultsData(data);

          // Fetch experiment details for context
          const expDetails = await fetchExperimentDetails(selectedExp.id);
          setExperiment(expDetails);
        }
      } catch (err) {
        setError('Failed to load results data.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, [experimentId]);

  if (loading) return <div className="p-6 text-center">Loading results...</div>;
  if (error) return <div className="p-6 text-center text-red-500">{error}</div>;
  if (!resultsData) return <div className="p-6 text-center">No results data available.</div>;

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-3xl font-bold text-gray-800">Results Analysis</h1>
          <p className="text-gray-600">{experiment?.description || 'Detailed analysis of simulation outcomes.'}</p>
        </div>
        
        {/* Experiment Selector */}
        <div className="flex items-center space-x-4">
          <label htmlFor="experiment-select" className="text-sm font-medium text-gray-700">Select Experiment:</label>
          <select
            id="experiment-select"
            value={experimentId}
            onChange={(e) => setExperimentId(e.target.value)}
            className="block w-64 pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
          >
            {experiments.map((exp) => (
              <option key={exp.id} value={exp.id}>
                {exp.experimentName} ({exp.status})
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Experiment Info Header */}
      {experiment && (
        <div className="bg-white rounded-lg shadow p-4 mb-6">
          <div className="flex justify-between">
            <div>
              <h2 className="text-xl font-semibold text-gray-800">{experiment.experimentName}</h2>
              <p className="text-gray-600">Status: <span className={`font-medium capitalize ${
                experiment.status === 'completed' ? 'text-green-600' :
                experiment.status === 'running' ? 'text-blue-600' :
                experiment.status === 'failed' ? 'text-red-600' : 'text-gray-600'
              }`}>{experiment.status}</span></p>
              <p className="text-gray-600">Started: {new Date(experiment.startTime || '').toLocaleString()}</p>
              {experiment.endTime && (
                <p className="text-gray-600">Ended: {new Date(experiment.endTime).toLocaleString()}</p>
              )}
            </div>
            <div className="text-right">
              <p className="text-gray-600">Progress: {experiment.progress || 0}%</p>
              <div className="w-32 bg-gray-200 rounded-full h-2.5 mt-1">
                <div
                  className={`h-2.5 rounded-full ${
                    experiment.status === 'completed' ? 'bg-green-600' :
                    experiment.status === 'running' ? 'bg-blue-600' :
                    experiment.status === 'failed' ? 'bg-red-600' : 'bg-gray-600'
                  }`}
                  style={{ width: `${experiment.progress || 0}%` }}
                ></div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Visualization Controls */}
      <div className="bg-white rounded-lg shadow p-4 mb-6">
        <h2 className="text-xl font-semibold text-gray-800 mb-4">Visualization Controls</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Chart Type</label>
            <select
              value={chartType}
              onChange={(e) => setChartType(e.target.value as 'line' | 'bar' | 'area' | 'scatter')}
              className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
            >
              <option value="line">Line Chart</option>
              <option value="bar">Bar Chart</option>
              <option value="area">Area Chart</option>
              <option value="scatter">Scatter Plot</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Metrics</label>
            <div className="space-y-1">
              {['revenue', 'profit', 'costs', 'decisionsMade', 'accuracy'].map((metric) => (
                <label key={metric} className="inline-flex items-center mr-4">
                  <input
                    type="checkbox"
                    checked={selectedMetrics.includes(metric)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        setSelectedMetrics([...selectedMetrics, metric]);
                      } else {
                        setSelectedMetrics(selectedMetrics.filter(m => m !== metric));
                      }
                    }}
                    className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
                  />
                  <span className="ml-2 text-sm text-gray-700 capitalize">{metric}</span>
                </label>
              ))}
            </div>
          </div>
          
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Time Range</label>
            <div className="flex space-x-2">
              <input
                type="date"
                value={timeRange?.start || ''}
                onChange={(e) => setTimeRange(timeRange ? { ...timeRange, start: e.target.value } : { start: e.target.value, end: '' })}
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              />
              <input
                type="date"
                value={timeRange?.end || ''}
                onChange={(e) => setTimeRange(timeRange ? { ...timeRange, end: e.target.value } : { start: '', end: e.target.value })}
                className="block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-indigo-500 focus:border-indigo-500 sm:text-sm rounded-md"
              />
            </div>
          </div>
        </div>
      </div>

      {/* Results Filtering and Search */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Filter Results</h2>
        <ResultsFilter onFilterChange={(filters) => console.log('Filters applied:', filters)} />
      </section>

      {/* Statistical Summary Tables */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8 overflow-x-auto"> {/* Adjusted grid for more widgets */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 min-w-max">
          <h2 className="text-2xl font-semibold mb-4 text-gray-700 col-span-full">Statistical Summary</h2>
          <MetricWidget metric={{ label: "Total Revenue", value: resultsData.aggregatedMetrics.totalRevenue || 0, formatType: "currency" }} />
          <MetricWidget metric={{ label: "Total Profit", value: resultsData.aggregatedMetrics.totalProfit || 0, formatType: "currency" }} />
          <MetricWidget metric={{ label: "Experiment Duration", value: resultsData.aggregatedMetrics.experimentDuration || 'N/A', unit: 'ticks' }} />
          <MetricWidget metric={{ label: "Top Agent", value: resultsData.aggregatedMetrics.topPerformingAgent || 'N/A' }} />
          {/* Add more summary metrics here */}
          <div className="lg:col-span-2"> {/* Make SummaryWidget span 2 columns on large screens */}
            <SummaryWidget title="Experiment Highlights" content="Best run achieved max profit with Agent X. Overall profit increased by 15%." />
          </div>
        </div>
      </section>

      {/* Financial Performance Charts */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Financial Performance</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 overflow-x-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 min-w-max">
            {selectedMetrics.includes('revenue') && selectedMetrics.includes('profit') && (
              <FinancialChart
                data={resultsData.timeSeriesData}
                config={{
                  chartType: chartType,
                  dataSource: 'timeSeriesData',
                  metrics: ['revenue', 'profit'],
                  title: 'Revenue & Profit Over Time'
                }}
              />
            )}
            {selectedMetrics.includes('costs') && (
              <FinancialChart
                data={resultsData.timeSeriesData}
                config={{
                  chartType: chartType === 'line' ? 'area' : chartType,
                  dataSource: 'timeSeriesData',
                  metrics: ['costs'],
                  title: 'Cumulative Costs'
                }}
              />
            )}
            {/* Time Series Chart for additional metrics */}
            <TimeSeriesChart
              data={resultsData.timeSeriesData}
              title="Time Series Analysis"
              metricsToShow={selectedMetrics}
              timeRange={timeRange || undefined}
              height={300}
            />
          </div>
        </div>
      </section>

      {/* Agent Performance Comparison */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Agent Performance Comparison</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 overflow-x-auto">
          <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6 min-w-max">
            {selectedMetrics.includes('profit') && (
              <AgentComparisonChart
                data={resultsData.agentPerformance}
                config={{
                  chartType: 'bar',
                  dataSource: 'agentPerformance',
                  metrics: ['profit'],
                  groupBy: 'agentId',
                  title: 'Agent Profit Ranking'
                }}
              />
            )}
            {selectedMetrics.includes('decisionsMade') && selectedMetrics.includes('accuracy') && (
              <AgentComparisonChart
                data={resultsData.agentPerformance}
                config={{
                  chartType: 'scatter',
                  dataSource: 'agentPerformance',
                  metrics: ['decisionsMade', 'accuracy'],
                  title: 'Agent Decision Accuracy vs. Decisions Made'
                }}
              />
            )}
            {/* Radar Chart for multi-dimensional comparison */}
            <div className="bg-white p-4 rounded shadow-md">
              <h3 className="text-lg font-medium text-gray-700 mb-2">Agent Performance Radar</h3>
              <div className="h-64 flex items-center justify-center text-gray-500">
                Radar Chart Visualization
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Market Dynamics Visualization */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Market Dynamics</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white p-4 rounded shadow-md">
            <h3 className="text-lg font-medium text-gray-700 mb-2">Performance Heatmap</h3>
            <div className="h-64 flex items-center justify-center text-gray-500">
              Heatmap Visualization
            </div>
          </div>
          <div className="bg-white p-4 rounded shadow-md">
            <h3 className="text-lg font-medium text-gray-700 mb-2">Market Insights</h3>
            <div className="space-y-2">
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Price Volatility:</span>
                <span className="text-sm font-medium">Medium</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Market Efficiency:</span>
                <span className="text-sm font-medium">High</span>
              </div>
              <div className="flex justify-between">
                <span className="text-sm text-gray-600">Competition Level:</span>
                <span className="text-sm font-medium">Intense</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Experiment Comparison Interface (requires selecting multiple experiments) */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Experiment Comparison</h2>
        <ComparisonTool /> {/* This component will handle fetching/displaying multiple datasets */}
      </section>

      {/* Advanced Analytics */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Advanced Analytics</h2>
        <AdvancedAnalytics resultsData={resultsData} />
      </section>

      {/* Data Export and Reporting */}
      <section>
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Export & Reporting</h2>
        <DataExporter />
      </section>
    </div>
  );
};

export default ResultsAnalysis;