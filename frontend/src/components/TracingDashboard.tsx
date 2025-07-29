import React from 'react';

const TracingDashboard: React.FC = () => {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">OpenTelemetry Tracing Dashboard</h2>
      <p className="text-gray-600">
        This dashboard will visualize OpenTelemetry traces for agent actions and simulation events.
        Features will include:
      </p>
      <ul className="list-disc list-inside text-gray-700 mt-2">
        <li>Timeline view of agent decisions</li>
        <li>Span hierarchy visualization</li>
        <li>Performance metrics display</li>
        <li>Trace export controls (e.g., to Chrome DevTools format)</li>
      </ul>
      <div className="mt-4 p-4 border rounded-md bg-gray-50 text-gray-700">
        <p>
          <strong>Coming Soon:</strong> Real-time trace visualization will be integrated here.
        </p>
      </div>
    </div>
  );
};

export default TracingDashboard;