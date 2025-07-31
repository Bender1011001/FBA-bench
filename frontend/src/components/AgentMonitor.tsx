import React, { useState, useMemo } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { useWebSocket } from '../hooks/useWebSocket';
import type { AgentStatus } from '../types';

interface AgentMonitorProps {
  className?: string;
  maxAgentsDisplay?: number;
}

interface AgentMetrics {
  id: string;
  name: string;
  status: AgentStatus['status'];
  performance: number;
  lastActivity: string;
  totalActions: number;
  successRate: number;
  errorCount: number;
  currentTask: string;
  responseTime: number;
}

export const AgentMonitor: React.FC<AgentMonitorProps> = ({
  className = '',
  maxAgentsDisplay = 20
}) => {
  const { simulation } = useSimulationStore();
  const { isConnected } = useWebSocket();
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<'name' | 'performance' | 'activity' | 'status'>('status');
  const [filterStatus, setFilterStatus] = useState<AgentStatus['status'] | 'all'>('all');

  // Transform simulation data into agent metrics
  const agentMetrics = useMemo((): AgentMetrics[] => {
    if (!simulation.snapshot?.agents) {
      return [];
    }

    return Object.entries(simulation.snapshot.agents).map(([agentId, agentData]) => {
      // Mock some additional metrics that would come from real agent monitoring
      const mockMetrics = {
        performance: Math.random() * 100,
        totalActions: Math.floor(Math.random() * 1000),
        successRate: Math.random() * 100,
        errorCount: Math.floor(Math.random() * 10),
        currentTask: ['Processing Orders', 'Market Analysis', 'Price Optimization', 'Customer Service'][Math.floor(Math.random() * 4)],
        responseTime: Math.random() * 2000
      };

      return {
        id: agentId,
        name: agentData.name || `Agent-${agentId}`,
        status: agentData.status || 'idle',
        performance: mockMetrics.performance,
        lastActivity: simulation.snapshot?.timestamp || new Date().toISOString(),
        totalActions: mockMetrics.totalActions,
        successRate: mockMetrics.successRate,
        errorCount: mockMetrics.errorCount,
        currentTask: mockMetrics.currentTask,
        responseTime: mockMetrics.responseTime
      };
    });
  }, [simulation.snapshot?.agents]);

  // Filter and sort agents
  const filteredAndSortedAgents = useMemo(() => {
    let filtered = agentMetrics;

    // Apply status filter
    if (filterStatus !== 'all') {
      filtered = filtered.filter(agent => agent.status === filterStatus);
    }

    // Apply sorting
    filtered.sort((a, b) => {
      switch (sortBy) {
        case 'name':
          return a.name.localeCompare(b.name);
        case 'performance':
          return b.performance - a.performance;
        case 'activity':
          return new Date(b.lastActivity).getTime() - new Date(a.lastActivity).getTime();
        case 'status':
          return a.status.localeCompare(b.status);
        default:
          return 0;
      }
    });

    return filtered.slice(0, maxAgentsDisplay);
  }, [agentMetrics, sortBy, filterStatus, maxAgentsDisplay]);

  // Get status color
  const getStatusColor = (status: AgentStatus['status']) => {
    switch (status) {
      case 'active':
        return 'text-green-600 bg-green-100';
      case 'idle':
        return 'text-yellow-600 bg-yellow-100';
      case 'error':
        return 'text-red-600 bg-red-100';
      case 'paused':
        return 'text-gray-600 bg-gray-100';
      default:
        return 'text-gray-600 bg-gray-100';
    }
  };

  // Get performance color
  const getPerformanceColor = (performance: number) => {
    if (performance >= 80) return 'text-green-600';
    if (performance >= 60) return 'text-yellow-600';
    return 'text-red-600';
  };

  // Get agent status summary
  const statusSummary = useMemo(() => {
    const summary = agentMetrics.reduce((acc, agent) => {
      acc[agent.status] = (acc[agent.status] || 0) + 1;
      return acc;
    }, {} as Record<string, number>);

    return {
      total: agentMetrics.length,
      active: summary.active || 0,
      idle: summary.idle || 0,
      error: summary.error || 0,
      paused: summary.paused || 0
    };
  }, [agentMetrics]);

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 ${className}`}>
      {/* Header */}
      <div className="p-6 border-b border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Agent Monitoring</h3>
          <div className="flex items-center space-x-2">
            <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-500' : 'bg-red-500'}`} />
            <span className="text-sm text-gray-600">
              {isConnected ? 'Live' : 'Offline'}
            </span>
          </div>
        </div>

        {/* Status Summary */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-900">{statusSummary.total}</div>
            <div className="text-sm text-gray-500">Total Agents</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{statusSummary.active}</div>
            <div className="text-sm text-gray-500">Active</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-600">{statusSummary.idle}</div>
            <div className="text-sm text-gray-500">Idle</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{statusSummary.error}</div>
            <div className="text-sm text-gray-500">Error</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-600">{statusSummary.paused}</div>
            <div className="text-sm text-gray-500">Paused</div>
          </div>
        </div>

        {/* Controls */}
        <div className="flex flex-wrap items-center gap-4">
          {/* Sort By */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Sort by:</label>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as typeof sortBy)}
              className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="status">Status</option>
              <option value="name">Name</option>
              <option value="performance">Performance</option>
              <option value="activity">Last Activity</option>
            </select>
          </div>

          {/* Filter By Status */}
          <div className="flex items-center space-x-2">
            <label className="text-sm font-medium text-gray-700">Filter:</label>
            <select
              value={filterStatus}
              onChange={(e) => setFilterStatus(e.target.value as typeof filterStatus)}
              className="text-sm border border-gray-300 rounded-md px-2 py-1 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="all">All Status</option>
              <option value="active">Active</option>
              <option value="idle">Idle</option>
              <option value="error">Error</option>
              <option value="paused">Paused</option>
            </select>
          </div>

          {/* Refresh Button */}
          <button
            onClick={() => window.location.reload()}
            className="text-sm px-3 py-1 bg-blue-100 text-blue-700 rounded-md hover:bg-blue-200 transition-colors"
          >
            Refresh
          </button>
        </div>
      </div>

      {/* Agent List */}
      <div className="p-6">
        {filteredAndSortedAgents.length === 0 ? (
          <div className="text-center py-8">
            <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2 2v-5m16 0h-2M4 13h2m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2 2v-5" />
            </svg>
            <h3 className="mt-2 text-sm font-medium text-gray-900">No agents found</h3>
            <p className="mt-1 text-sm text-gray-500">
              {agentMetrics.length === 0 
                ? 'No agents are currently running in the simulation.'
                : 'No agents match the current filter criteria.'}
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {filteredAndSortedAgents.map((agent) => (
              <div
                key={agent.id}
                className={`p-4 border rounded-lg cursor-pointer transition-all ${
                  selectedAgent === agent.id
                    ? 'border-blue-500 bg-blue-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedAgent(selectedAgent === agent.id ? null : agent.id)}
              >
                {/* Agent Header */}
                <div className="flex items-center justify-between mb-3">
                  <h4 className="font-medium text-gray-900 truncate">{agent.name}</h4>
                  <span className={`px-2 py-1 text-xs font-medium rounded-full ${getStatusColor(agent.status)}`}>
                    {agent.status.toUpperCase()}
                  </span>
                </div>

                {/* Agent Metrics */}
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Performance:</span>
                    <span className={`font-medium ${getPerformanceColor(agent.performance)}`}>
                      {agent.performance.toFixed(1)}%
                    </span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Success Rate:</span>
                    <span className="font-medium text-gray-900">{agent.successRate.toFixed(1)}%</span>
                  </div>
                  
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-500">Actions:</span>
                    <span className="font-medium text-gray-900">{agent.totalActions}</span>
                  </div>

                  {agent.errorCount > 0 && (
                    <div className="flex justify-between text-sm">
                      <span className="text-gray-500">Errors:</span>
                      <span className="font-medium text-red-600">{agent.errorCount}</span>
                    </div>
                  )}

                  <div className="text-sm">
                    <span className="text-gray-500">Current Task:</span>
                    <div className="font-medium text-gray-900 truncate mt-1">{agent.currentTask}</div>
                  </div>

                  <div className="text-sm">
                    <span className="text-gray-500">Last Activity:</span>
                    <div className="font-medium text-gray-900 mt-1">
                      {new Date(agent.lastActivity).toLocaleTimeString()}
                    </div>
                  </div>
                </div>

                {/* Expanded Details */}
                {selectedAgent === agent.id && (
                  <div className="mt-4 pt-3 border-t border-gray-200">
                    <div className="space-y-2 text-sm">
                      <div className="flex justify-between">
                        <span className="text-gray-500">Agent ID:</span>
                        <span className="font-mono text-xs">{agent.id}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-500">Response Time:</span>
                        <span className="font-medium">{agent.responseTime.toFixed(0)}ms</span>
                      </div>
                      
                      {/* Action Buttons */}
                      <div className="flex space-x-2 mt-3">
                        <button className="px-3 py-1 text-xs bg-blue-100 text-blue-700 rounded hover:bg-blue-200 transition-colors">
                          View Logs
                        </button>
                        <button className="px-3 py-1 text-xs bg-yellow-100 text-yellow-700 rounded hover:bg-yellow-200 transition-colors">
                          Restart
                        </button>
                        {agent.status === 'error' && (
                          <button className="px-3 py-1 text-xs bg-red-100 text-red-700 rounded hover:bg-red-200 transition-colors">
                            Debug
                          </button>
                        )}
                      </div>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}

        {/* Show more button if there are more agents */}
        {agentMetrics.length > maxAgentsDisplay && (
          <div className="text-center mt-6">
            <p className="text-sm text-gray-500 mb-2">
              Showing {maxAgentsDisplay} of {agentMetrics.length} agents
            </p>
            <button className="px-4 py-2 text-sm bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors">
              Load More Agents
            </button>
          </div>
        )}
      </div>

      {/* No simulation data */}
      {!simulation.snapshot && (
        <div className="p-6 text-center">
          <div className="text-gray-400 mb-2">
            <svg className="mx-auto h-12 w-12" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
          </div>
          <p className="text-gray-600 font-medium">No simulation data available</p>
          <p className="text-gray-500 text-sm mt-1">Start a simulation to see agent monitoring data</p>
        </div>
      )}
    </div>
  );
};

export default AgentMonitor;