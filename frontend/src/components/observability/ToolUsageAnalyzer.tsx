import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell } from 'recharts';

interface ToolUsageData {
  toolName: string;
  successCount: number;
  failureCount: number;
  avgResponseTime: number;
}

interface CommonErrorData {
  errorType: string;
  count: number;
}

interface ToolUsageAnalyzerProps {
  toolUsageMetrics?: ToolUsageData[];
  commonErrorPatterns?: CommonErrorData[];
  recommendedToolUsage?: { toolName: string; recommendation: string; }[];
}

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#AF19FF', '#FF19A0', '#19FFD4'];

const ToolUsageAnalyzer: React.FC<ToolUsageAnalyzerProps> = ({
  toolUsageMetrics = [],
  commonErrorPatterns = [],
  recommendedToolUsage = [],
}) => {
  const successFailureData = toolUsageMetrics.map(tool => ({
    toolName: tool.toolName,
    success: tool.successCount,
    failure: tool.failureCount,
  }));

  const responseTimeData = toolUsageMetrics.map(tool => ({
    toolName: tool.toolName,
    "Average Response Time (ms)": tool.avgResponseTime,
  }));

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Tool Usage Analyzer</h3>

      {/* Tool Usage Frequency and Success Rates */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Tool Call Success/Failure Rates</h4>
        {successFailureData.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={successFailureData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="toolName" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="success" stackId="a" fill="#00C49F" name="Success" />
              <Bar dataKey="failure" stackId="a" fill="#FF8042" name="Failure" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No tool usage frequency data available.</p>
        )}
      </div>

      {/* Common Error Patterns by Tool (Pie Chart) */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Common Tool Error Types</h4>
        {commonErrorPatterns.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <PieChart>
              <Pie
                data={commonErrorPatterns}
                dataKey="count"
                nameKey="errorType"
                cx="50%"
                cy="50%"
                outerRadius={80}
                fill="#8884d8"
                label
              >
                {commonErrorPatterns.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
              <Legend />
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No common error pattern data available.</p>
        )}
      </div>

      {/* Performance Metrics per Tool (Average Response Time) */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Average Tool Response Time</h4>
        {responseTimeData.length > 0 ? (
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={responseTimeData} margin={{ top: 20, right: 30, left: 20, bottom: 5 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="toolName" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="Average Response Time (ms)" fill="#8884d8" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No tool response time data available.</p>
        )}
      </div>

      {/* Usage Recommendations */}
      <div>
        <h4 className="text-md font-medium text-gray-700 mb-2">Tool Usage Recommendations</h4>
        {recommendedToolUsage.length > 0 ? (
          <ul className="list-disc list-inside text-gray-700">
            {recommendedToolUsage.map((rec, index) => (
              <li key={index}><strong>{rec.toolName}:</strong> {rec.recommendation}</li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500">No specific tool usage recommendations at this time.</p>
        )}
      </div>
    </div>
  );
};

export default ToolUsageAnalyzer;