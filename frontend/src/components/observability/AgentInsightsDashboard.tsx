import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar } from 'recharts';

interface AgentInsightProps {
  agentPerformanceMetrics?: { timestamp: number; score: number; }[];
  decisionQualityScores?: { timestamp: number; quality: number; }[];
  learningProgressData?: { training_step: number; loss: number; accuracy: number; }[];
  errorPatternTrends?: { errorType: string; count: number; }[];
}

const AgentInsightsDashboard: React.FC<AgentInsightProps> = ({
  agentPerformanceMetrics = [],
  decisionQualityScores = [],
  learningProgressData = [],
  errorPatternTrends = [],
}) => {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Agent Insights Dashboard</h3>

      {/* Real-time agent performance metrics */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Agent Performance Score Over Time</h4>
        {agentPerformanceMetrics.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={agentPerformanceMetrics}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()} />
              <YAxis />
              <Tooltip labelFormatter={(label) => new Date(label * 1000).toLocaleString()} />
              <Legend />
              <Line type="monotone" dataKey="score" stroke="#8884d8" name="Performance Score" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No agent performance data available.</p>
        )}
      </div>

      {/* Decision quality scoring */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Agent Decision Quality</h4>
        {decisionQualityScores.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={decisionQualityScores}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()} />
              <YAxis />
              <Tooltip labelFormatter={(label) => new Date(label * 1000).toLocaleString()} />
              <Legend />
              <Line type="monotone" dataKey="quality" stroke="#82ca9d" name="Decision Quality" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No decision quality data available.</p>
        )}
      </div>

      {/* Learning progress visualization */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Agent Learning Progress</h4>
        {learningProgressData.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={learningProgressData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="training_step" name="Training Step" />
              <YAxis yAxisId="left" orientation="left" stroke="#8884d8" />
              <YAxis yAxisId="right" orientation="right" stroke="#82ca9d" />
              <Tooltip />
              <Legend />
              <Line yAxisId="left" type="monotone" dataKey="loss" stroke="#8884d8" name="Loss" />
              <Line yAxisId="right" type="monotone" dataKey="accuracy" stroke="#82ca9d" name="Accuracy" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No learning progress data available.</p>
        )}
      </div>

      {/* Error pattern trends */}
      <div>
        <h4 className="text-md font-medium text-gray-700 mb-2">Agent Error Pattern Trends</h4>
        {errorPatternTrends.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <BarChart data={errorPatternTrends}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="errorType" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#ffc658" name="Error Count" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p>No error pattern trend data available.</p>
        )}
      </div>
    </div>
  );
};

export default AgentInsightsDashboard;