import React, { useState, useEffect } from 'react';
import type { ResultsData, ExperimentExecution } from '../types';
import { fetchResultsData, fetchExperimentDetails } from '../services/apiService';

// Import the actual components
import FinancialChart from '../components/charts/FinancialChart';
import AgentComparisonChart from '../components/charts/AgentComparisonChart';
import DataExporter from '../components/DataExporter';
import ResultsFilter from '../components/ResultsFilter';
import ComparisonTool from '../components/ComparisonTool';
import MetricWidget from '../components/widgets/MetricWidget';
import SummaryWidget from '../components/widgets/SummaryWidget';
import AdvancedAnalytics from '../components/AdvancedAnalytics';

const ResultsAnalysis: React.FC = () => {
  const [resultsData, setResultsData] = useState<ResultsData | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);
  const [experiment, setExperiment] = useState<ExperimentExecution | null>(null);

  // Example experiment ID - this would ideally come from a selected experiment in ExperimentManagement or URL param
  const experimentId = 'example-experiment-123';

  useEffect(() => {
    const loadResults = async () => {
      try {
        setLoading(true);
        // Fetch detailed results data
        const data = await fetchResultsData(experimentId);
        setResultsData(data);

        // Fetch experiment details for context
        const expDetails = await fetchExperimentDetails(experimentId);
        setExperiment(expDetails);
      } catch (err) {
        setError('Failed to load results data.');
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    loadResults();
  }, [experimentId]);

  if (loading) return <div className="p-6 text-center">Loading results...</div>;
  if (error) return <div className="p-6 text-center text-red-500">{error}</div>;
  if (!resultsData) return <div className="p-6 text-center">No results data available.</div>;

  return (
    <div className="p-6 bg-gray-100 min-h-screen">
      <h1 className="text-3xl font-bold mb-6 text-gray-800">Results Analysis: {experiment?.experimentName || experimentId}</h1>
      <p className="mb-8 text-gray-600">{experiment?.description || 'Detailed analysis of simulation outcomes.'}</p>

      {/* Results Filtering and Search */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Filter Results</h2>
        <ResultsFilter onFilterChange={(filters) => console.log('Filters applied:', filters)} />
      </section>

      {/* Statistical Summary Tables */}
      <section className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"> {/* Adjusted grid for more widgets */}
        <h2 className="text-2xl font-semibold mb-4 text-gray-700 col-span-full">Statistical Summary</h2>
        <MetricWidget metric={{ label: "Total Revenue", value: resultsData.aggregatedMetrics.totalRevenue || 0, formatType: "currency" }} />
        <MetricWidget metric={{ label: "Total Profit", value: resultsData.aggregatedMetrics.totalProfit || 0, formatType: "currency" }} />
        <MetricWidget metric={{ label: "Experiment Duration", value: resultsData.aggregatedMetrics.experimentDuration || 'N/A', unit: 'ticks' }} />
        <MetricWidget metric={{ label: "Top Agent", value: resultsData.aggregatedMetrics.topPerformingAgent || 'N/A' }} />
        {/* Add more summary metrics here */}
        <div className="lg:col-span-2"> {/* Make SummaryWidget span 2 columns on large screens */}
          <SummaryWidget title="Experiment Highlights" content="Best run achieved max profit with Agent X. Overall profit increased by 15%." />
        </div>
      </section>

      {/* Financial Performance Charts */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Financial Performance</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <FinancialChart
            data={resultsData.timeSeriesData}
            config={{ chartType: 'line', dataSource: 'timeSeriesData', metrics: ['revenue', 'profit'], title: 'Revenue & Profit Over Time' }}
          />
          <FinancialChart
            data={resultsData.timeSeriesData}
            config={{ chartType: 'area', dataSource: 'timeSeriesData', metrics: ['costs'], title: 'Cumulative Costs' }}
          />
        </div>
      </section>

      {/* Agent Performance Comparison */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Agent Performance Comparison</h2>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <AgentComparisonChart
            data={resultsData.agentPerformance}
            config={{ chartType: 'bar', dataSource: 'agentPerformance', metrics: ['profit'], groupBy: 'agentId', title: 'Agent Profit Ranking' }}
          />
          <AgentComparisonChart
            data={resultsData.agentPerformance}
            config={{ chartType: 'scatter', dataSource: 'agentPerformance', metrics: ['decisionsMade', 'accuracy'], title: 'Agent Decision Accuracy vs. Decisions Made' }}
          />
        </div>
      </section>

      {/* Market Dynamics Visualization (requires MarketMetrics in SimulationResult) */}
      <section className="mb-8">
        <h2 className="text-2xl font-semibold mb-4 text-gray-700">Market Dynamics</h2>
        {/* You'd use a generic chart component here, passing marketMetrics data */}
        <div className="bg-white p-4 rounded shadow-md h-64 flex items-center justify-center">
          <p>Market Dynamics Chart Placeholder (e.g., Price Trends, Inventory Levels)</p>
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