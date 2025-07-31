import React from 'react';

interface SummaryWidgetProps {
  title: string;
  content: React.ReactNode;
  // Potentially add props for specific data summaries, e.g., best agents, recent experiments
  // bestPerformingAgents?: { id: string; profit: number }[];
  // recentExperimentHighlights?: { id: string; name: string; status: string }[];
}

const SummaryWidget: React.FC<SummaryWidgetProps> = ({ title, content }) => {
  return (
    <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200 h-full flex flex-col">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">{title}</h3>
      <div className="flex-grow text-gray-700">
        {content}
        {/*
          Example usage with dynamic content:
          {bestPerformingAgents && (
            <div className="mt-4">
              <h4 className="font-medium text-gray-600">Top Performing Agents:</h4>
              <ul className="list-disc list-inside mt-2">
                {bestPerformingAgents.map((agent) => (
                  <li key={agent.id}>{agent.id}: ${agent.profit.toFixed(2)} Profit</li>
                ))}
              </ul>
            </div>
          )}
          {recentExperimentHighlights && (
            <div className="mt-4">
              <h4 className="font-medium text-gray-600">Recent Highlights:</h4>
              <ul className="list-disc list-inside mt-2">
                {recentExperimentHighlights.map((exp) => (
                  <li key={exp.id}>{exp.name} ({exp.status})</li>
                ))}
              </ul>
            </div>
          )}
        */}
      </div>
    </div>
  );
};

export default SummaryWidget;