import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { MetricCard } from './MetricCard';
import type { SystemHealth, DashboardMetric } from '../types';

interface SystemHealthMonitorProps {
  className?: string;
}

export const SystemHealthMonitor: React.FC<SystemHealthMonitorProps> = ({
  className = '',
}) => {
  const systemHealth = useSimulationStore((state) => state.simulation.systemHealth);
  const simulationError = useSimulationStore((state) => state.simulation.error);

  // Determine status color
  const getStatusColor = (status: SystemHealth['wsConnectionStatus'] | 'connected' | 'disconnected') => {
    switch (status) {
      case 'connected':
        return 'text-green-500';
      case 'disconnected':
      case 'error':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const healthMetrics: DashboardMetric[] = [
    {
      label: 'API Response Time',
      value: systemHealth?.apiResponseTime || 'N/A',
      formatType: 'number',
      unit: 'ms',
      description: 'Average response time for backend API calls.',
      trend: (systemHealth?.apiResponseTime && systemHealth.apiResponseTime < 200) ? 'up' : 'neutral',
      color: (systemHealth?.apiResponseTime && systemHealth.apiResponseTime > 500) ? 'text-red-500' : ''
    },
    {
      label: 'WebSocket Connection',
      value: systemHealth?.wsConnectionStatus || 'N/A',
      formatType: 'string', // Display as string directly
      color: getStatusColor(systemHealth?.wsConnectionStatus || 'disconnected'),
      description: 'Status of the real-time WebSocket connection to the backend.'
    },
    {
      label: 'Memory Usage',
      value: systemHealth?.memoryUsage || 'N/A',
      formatType: 'number',
      unit: 'MB',
      description: 'Backend memory consumption.',
      trend: (systemHealth?.memoryUsage && systemHealth.memoryUsage < 800) ? 'neutral' : 'down',
      color: (systemHealth?.memoryUsage && systemHealth.memoryUsage > 1000) ? 'text-red-500' : ''
    },
    {
      label: 'CPU Usage',
      value: systemHealth?.cpuUsage || 'N/A',
      formatType: 'number',
      unit: '%',
      description: 'Backend CPU utilization.',
      trend: (systemHealth?.cpuUsage && systemHealth.cpuUsage < 70) ? 'neutral' : 'down',
      color: (systemHealth?.cpuUsage && systemHealth.cpuUsage > 90) ? 'text-red-500' : ''
    },
    {
      label: 'Database Connection',
      value: systemHealth?.dbConnectionStatus || 'N/A',
      formatType: 'string',
      color: getStatusColor(systemHealth?.dbConnectionStatus || 'disconnected'),
      description: 'Status of the database connection.'
    },
    {
      label: 'Queue Length',
      value: systemHealth?.queueLength || 'N/A',
      formatType: 'number',
      description: 'Number of pending tasks in the backend queue.',
      trend: (systemHealth?.queueLength && systemHealth.queueLength < 50) ? 'neutral' : 'down',
      color: (systemHealth?.queueLength && systemHealth.queueLength > 100) ? 'text-red-500' : ''
    },
  ];

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4">System Health Monitor</h3>
      
      {simulationError && (
        <div className="bg-red-50 border border-red-200 rounded-md p-3 mb-4">
          <p className="text-red-700 text-sm font-medium">Error: {simulationError}</p>
        </div>
      )}

      {!systemHealth && !simulationError && (
        <div className="text-center py-8">
          <svg className="mx-auto h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 12l2 2 4-4m5.618-4.272a11.971 11.971 0 00-17.653 0M12 21.08V15m-.001-3.627A4 4 0 0012 9a4 4 0 00-4 4.373M12 21.08V15m-.001-3.627A4 4 0 0012 9a4 4 0 00-4 4.373" />
          </svg>
          <p className="mt-2 text-sm font-medium text-gray-900">No system health data available</p>
          <p className="mt-1 text-sm text-gray-500">System health metrics will appear here when a simulation is running.</p>
        </div>
      )}

      {systemHealth && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {healthMetrics.map((metric, index) => (
            <MetricCard key={index} metric={metric} />
          ))}
        </div>
      )}
    </div>
  );
};

export default SystemHealthMonitor;