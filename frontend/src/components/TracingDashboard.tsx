import React from 'react';
import AgentReasoningViewer from './observability/AgentInsightsDashboard';
import ErrorPatternChart from './observability/ToolUsageAnalyzer';
import PerformanceHeatmap from './observability/SimulationHealthMonitor';
import InteractiveTraceExplorer from './TraceViewer';

const TracingDashboard: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">OpenTelemetry Tracing Dashboard</h2>
      <p className="text-gray-600">
        This dashboard visualizes OpenTelemetry traces for agent actions and simulation events,
        now enhanced with advanced observability features:
      </p>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
        <div className="col-span-1">
          <h3 className="text-lg font-semibold mb-2">Agent Reasoning Visualization</h3>
          <AgentReasoningViewer />
        </div>
        <div className="col-span-1">
          <h3 className="text-lg font-semibold mb-2">Error Pattern Analysis</h3>
          <ErrorPatternChart />
        </div>
        <div className="col-span-1">
          <h3 className="text-lg font-semibold mb-2">Performance Heatmap</h3>
          <PerformanceHeatmap />
        </div>
        <div className="col-span-1">
          <h3 className="text-lg font-semibold mb-2">Interactive Trace Exploration</h3>
          <InteractiveTraceExplorer />
        </div>
      </div>

      <div className="mt-6 p-4 border rounded-md bg-purple-50 text-purple-800">
        <p>
          <strong>Note:</strong> Data for these new visualizations will be populated as advanced
          trace analysis and real-time alert systems are integrated.
        </p>
      </div>
    </div>
  );
};

export default TracingDashboard;