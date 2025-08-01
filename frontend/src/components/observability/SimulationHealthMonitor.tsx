import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, AreaChart, Area } from 'recharts';

interface HealthMetric {
  timestamp: number;
  value: number;
}

interface SimulationHealthMonitorProps {
  overallHealthIndicators?: HealthMetric[];
  resourceUtilization?: {timestamp: number; cpu: number; memory: number;}[];
  performanceTrendAnalysis?: HealthMetric[];
  proactiveIssueDetection?: { timestamp: number; issue: string; severity: string; }[];
}

const SimulationHealthMonitor: React.FC<SimulationHealthMonitorProps> = ({
  overallHealthIndicators = [],
  resourceUtilization = [],
  performanceTrendAnalysis = [],
  proactiveIssueDetection = [],
}) => {
  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-4 mb-6">
      <h3 className="text-lg font-semibold text-gray-800 mb-4">Simulation Health Monitor</h3>

      {/* Overall simulation health indicators */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Overall Simulation Health Score</h4>
        {overallHealthIndicators.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={overallHealthIndicators}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()} />
              <YAxis domain={[0, 100]} label={{ value: 'Health Score', angle: -90, position: 'insideLeft' }} />
              <Tooltip labelFormatter={(label) => `Time: ${new Date(label * 1000).toLocaleString()}`} />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#8884d8" name="Health Score" activeDot={{ r: 8 }} />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No overall health data available.</p>
        )}
      </div>

      {/* Resource utilization tracking (CPU and Memory) */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Resource Utilization (CPU & Memory)</h4>
        {resourceUtilization.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <AreaChart data={resourceUtilization} margin={{ top: 10, right: 30, left: 0, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()} />
              <YAxis />
              <Tooltip labelFormatter={(label) => `Time: ${new Date(label * 1000).toLocaleString()}`} />
              <Legend />
              <Area type="monotone" dataKey="cpu" stroke="#82ca9d" fillOpacity={1} fill="url(#colorCpu)" name="CPU Usage (%)" />
              <Area type="monotone" dataKey="memory" stroke="#8884d8" fillOpacity={1} fill="url(#colorMemory)" name="Memory Usage (%)" />
              <defs>
                  <linearGradient id="colorCpu" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#82ca9d" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#82ca9d" stopOpacity={0}/>
                  </linearGradient>
                  <linearGradient id="colorMemory" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#8884d8" stopOpacity={0.8}/>
                    <stop offset="95%" stopColor="#8884d8" stopOpacity={0}/>
                  </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No resource utilization data available.</p>
        )}
      </div>

      {/* Performance trend analysis */}
      <div className="mb-6">
        <h4 className="text-md font-medium text-gray-700 mb-2">Simulation Performance Trend</h4>
        {performanceTrendAnalysis.length > 0 ? (
          <ResponsiveContainer width="100%" height={200}>
            <LineChart data={performanceTrendAnalysis}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="timestamp" type="number" domain={['dataMin', 'dataMax']} tickFormatter={(tick) => new Date(tick * 1000).toLocaleTimeString()} />
              <YAxis label={{ value: 'Latency (ms)', angle: -90, position: 'insideLeft' }} />
              <Tooltip labelFormatter={(label) => `Time: ${new Date(label * 1000).toLocaleString()}`} />
              <Legend />
              <Line type="monotone" dataKey="value" stroke="#ffc658" name="Average Latency" />
            </LineChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-gray-500">No performance trend data available.</p>
        )}
      </div>
      
      {/* Proactive issue detection */}
      <div>
        <h4 className="text-md font-medium text-gray-700 mb-2">Proactive Issue Detections</h4>
        {proactiveIssueDetection.length > 0 ? (
          <ul className="list-disc list-inside text-gray-700">
            {proactiveIssueDetection.map((issue, index) => (
              <li key={index} className={issue.severity === 'Critical' ? 'text-red-600' : issue.severity === 'Warning' ? 'text-yellow-600' : ''}>
                <strong>{new Date(issue.timestamp * 1000).toLocaleTimeString()}:</strong> [{issue.severity}] {issue.issue}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-gray-500">No proactive issues detected.</p>
        )}
      </div>
    </div>
  );
};

export default SimulationHealthMonitor;