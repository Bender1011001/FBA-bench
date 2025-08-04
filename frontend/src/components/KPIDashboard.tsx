import React, { useEffect, useMemo, useState, useCallback } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { apiService } from '../services/apiService';
import { MetricCard, MetricCardSkeleton, MetricCardError } from './MetricCard';
import { ConnectionStatusCompact } from './ConnectionStatus';
import type { DashboardMetric, SimulationStatus, SystemHealth, SimulationSnapshot } from '../types';
import { Progress } from './ui/progress';

interface KPIDashboardProps {
  className?: string;
  refreshInterval?: number;
}

/**
 * Safe calculation function that handles zero denominators and invalid values
 */
const safeCalculate = (value: number, denominator: number = 1): number => {
  // Check if denominator is zero, NaN, or not finite
  if (denominator === 0 || !isFinite(denominator)) {
    return 0;
  }
  
  // Check if value is NaN or not finite
  if (!isFinite(value)) {
    return 0;
  }
  
  const result = value / denominator;
  
  // Ensure the result is finite
  return isFinite(result) ? result : 0;
};

/**
 * Validates that a value is safe for calculations
 */
const isValidNumber = (value: unknown): value is number => {
  return typeof value === 'number' && isFinite(value) && !isNaN(value);
};

const formatCurrency = (amount: number | string): number => {
  if (typeof amount === 'string') {
    const cleaned = amount.replace(/[$,]/g, '');
    const parsed = parseFloat(cleaned);
    return isValidNumber(parsed) ? parsed : 0;
  }
  return isValidNumber(amount) ? amount : 0;
};

const formatTrustScore = (score: number): number => {
  return isValidNumber(score) ? safeCalculate(score * 100, 1) : 0;
};

export const KPIDashboard: React.FC<KPIDashboardProps> = React.memo(({
  className = '',
  refreshInterval = 1000 // Update frequency for dashboard metrics
}) => {
  const {
    simulation,
    setSnapshot,
    setLoading,
    setError,
    getCurrentMetrics,
    setSimulationStatus,
    setSystemHealth // Add this
  } = useSimulationStore();

  const [progress, setProgress] = useState(0);

  // Initialize WebSocket connection and handle incoming messages
  const { isConnected, lastMessage, sendJsonMessage } = useWebSocket({
    onConnect: () => {
      console.log('Dashboard: WebSocket connected');
      setError(null);
      // Optionally send a message to subscribe to specific topics
      sendJsonMessage({ type: 'subscribe', topics: ['simulation_status', 'agent_status', 'financial_metrics', 'system_health'] });
    },
    onDisconnect: () => {
      console.log('Dashboard: WebSocket disconnected');
      setError('WebSocket disconnected. Attempting to reconnect...');
    },
    onError: (event) => {
      console.error('Dashboard: WebSocket error:', event);
      setError('WebSocket connection failed or encountered an error.');
    }
  });

  // Process incoming WebSocket messages
  useEffect(() => {
    if (lastMessage?.data) {
      try {
        const message = JSON.parse(lastMessage.data);
        switch (message.type) {
          case 'simulation_status':
            setSimulationStatus(message.payload as SimulationStatus);
            break;
          case 'system_health':
            setSystemHealth(message.payload as SystemHealth);
            break;
          case 'simulation_snapshot': // For initial load or manual refresh
            setSnapshot(message.payload);
            break;
          // Add cases for agent status, financial metrics, etc. if they come as separate messages
          default:
            console.log('Unhandled WebSocket message type:', message.type);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    }
  }, [lastMessage, setSimulationStatus, setSystemHealth, setSnapshot]);

  // Fetch initial simulation data (fallback if WebSocket is not used or for initial state)
  const fetchInitialData = async () => {
    try {
      setLoading(true);
      setError(null);
      // Fetch simulation snapshot using generic get method
      const response = await apiService.get<SimulationSnapshot>('/api/v1/simulation/snapshot');
      const snapshot = response.data; // Extract the actual data from the API response
      setSnapshot(snapshot);
      // Assuming API also provides initial simulation status and system health
      // For now, mock these if not available from snapshot or a dedicated API call is needed
      setSimulationStatus({
        id: snapshot.metadata?.simulation_id || 'N/A',
        status: snapshot.metadata?.status || 'idle', // Assuming status is in metadata
        currentTick: snapshot.current_tick,
        totalTicks: snapshot.metadata?.total_ticks || 1000, // Assuming total_ticks in metadata
        simulationTime: snapshot.simulation_time,
        realTime: new Date(snapshot.last_update).toLocaleTimeString(),
        ticksPerSecond: snapshot.event_stats?.ticks_per_second || 0,
        revenue: formatCurrency(snapshot.financial_summary?.total_revenue?.amount || 0),
        costs: formatCurrency(snapshot.financial_summary?.total_costs?.amount || 0),
        profit: formatCurrency(snapshot.financial_summary?.total_profit?.amount || 0),
        activeAgentCount: snapshot.agents ? Object.keys(snapshot.agents).length : 0,
      });
      setSystemHealth({
        apiResponseTime: 120, // Mock value
        wsConnectionStatus: isConnected ? 'connected' : 'disconnected',
        memoryUsage: 500, // Mock value
        cpuUsage: 30, // Mock value
        dbConnectionStatus: 'connected', // Mock value
        queueLength: 5, // Mock value
      });

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch simulation data';
      console.error('Failed to fetch simulation data:', error);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Memoize fetchInitialData to prevent unnecessary recreations
  const memoizedFetchInitialData = useCallback(fetchInitialData, [setSnapshot, setLoading, setError, setSimulationStatus, setSystemHealth, apiService]);

  // Set up periodic data refresh primarily for non-realtime fallback
  useEffect(() => {
    memoizedFetchInitialData(); // Initial fetch

    let intervalId: number | undefined;
    if (refreshInterval > 0) {
      intervalId = window.setInterval(memoizedFetchInitialData, refreshInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [refreshInterval, memoizedFetchInitialData]); // Depend only on refreshInterval, as WS handles real-time

  // Memoize current metrics to prevent unnecessary recalculations
  const currentMetrics = useMemo(() => getCurrentMetrics(), [getCurrentMetrics]);
  
  // Update progress bar
  useEffect(() => {
    if (simulation.status === 'running' && currentMetrics.totalTicks > 0) {
      // Safe calculation for progress percentage
      const calculatedProgress = safeCalculate(currentMetrics.currentTick * 100, currentMetrics.totalTicks);
      setProgress(Math.min(100, Math.max(0, calculatedProgress)));
    } else {
      setProgress(0); // Reset or keep current progress if not running
    }
  }, [simulation.status, currentMetrics]);

  // Memoize status color function to prevent unnecessary recalculations
  const getStatusColor = useMemo(() => {
    return (status: SimulationStatus['status'] | SystemHealth['wsConnectionStatus']) => {
      switch (status) {
        case 'running':
        case 'connected':
          return 'text-green-500';
        case 'paused':
          return 'text-yellow-500';
        case 'stopped':
        case 'disconnected':
        case 'idle':
          return 'text-gray-500';
        case 'error':
          return 'text-red-500';
        case 'starting':
          return 'text-blue-500';
        default:
          return 'text-gray-500';
      }
    };
  }, []);

  // Memoize snapshot-based calculations to prevent unnecessary recalculations
  const competitorCount = useMemo(() => {
    return simulation.snapshot?.competitor_states?.length || 0;
  }, [simulation.snapshot?.competitor_states]);

  const averageCompetitorPrice = useMemo(() => {
    if (competitorCount === 0) return 0;
    const total = simulation.snapshot?.competitor_states?.reduce((sum: number, comp: { current_price?: { amount?: number | string } }) =>
      sum + formatCurrency(comp.current_price?.amount || 0), 0) || 0;
    return safeCalculate(total, competitorCount);
  }, [simulation.snapshot?.competitor_states, competitorCount]);

  const recentSalesValue = useMemo(() => {
    return simulation.snapshot?.recent_sales?.reduce((sum: number, sale: { sale_price?: { amount?: number | string }; quantity?: number }) => {
      const price = formatCurrency(sale.sale_price?.amount || 0);
      const quantity = isValidNumber(sale.quantity) ? sale.quantity : 0;
      return sum + (price * quantity);
    }, 0) || 0;
  }, [simulation.snapshot?.recent_sales]);

  // Calculate metrics from current state
  const metrics = useMemo((): DashboardMetric[] => {
    const { snapshot, status, systemHealth } = simulation;

    // Memoize the base metrics array to prevent unnecessary recreations
    const baseMetrics: DashboardMetric[] = [
      // Simulation Status
      {
        label: 'Simulation Status',
        value: status || 'N/A',
        formatType: 'string', // Display as string directly
        color: getStatusColor(status as SimulationStatus['status']),
        description: 'Current operational state of the simulation.'
      },
      {
        label: 'Current Tick',
        value: currentMetrics.currentTick,
        formatType: 'number',
        trend: currentMetrics.currentTick > 0 ? 'up' : 'neutral',
        description: 'The current simulation step.'
      },
      {
        label: 'Simulation Time',
        value: currentMetrics.simulationTime,
        formatType: 'time',
        description: 'Elapsed simulated time.'
      },
      {
        label: 'Real Time',
        value: currentMetrics.realTime,
        formatType: 'time',
        description: 'Current real-world time of the simulation update.'
      },
      {
        label: 'Estimated Total Ticks',
        value: currentMetrics.totalTicks,
        formatType: 'number',
        description: 'The total number of ticks expected for the simulation run.'
      },
      // Financial metrics
      {
        label: 'Total Revenue',
        value: currentMetrics.revenue,
        formatType: 'currency',
        trend: currentMetrics.revenue > 0 ? 'up' : 'neutral',
        description: 'Accumulated revenue across all agents.'
      },
      {
        label: 'Total Costs',
        value: currentMetrics.costs,
        formatType: 'currency',
        trend: currentMetrics.costs > 0 ? 'up' : 'neutral',
        description: 'Accumulated costs across all agents.'
      },
      {
        label: 'Total Profit',
        value: currentMetrics.profit,
        formatType: 'currency',
        trend: currentMetrics.profit > 0 ? 'up' : 'neutral',
        description: 'Net profit (Revenue - Costs).'
      },
      // Agent metrics
      {
        label: 'Active Agents',
        value: currentMetrics.activeAgentCount,
        formatType: 'number',
        description: 'Number of agents currently active in the simulation.'
      },
      // System performance metrics
      {
        label: 'Ticks Per Second',
        value: simulation.ticksPerSecond,
        formatType: 'number',
        unit: 'TPS',
        description: 'Rate of simulation ticks processed per second (performance indicator).'
      },
      // Connection Quality
      {
        label: 'WebSocket Status',
        value: isConnected ? 'Connected' : 'Disconnected',
        formatType: 'string',
        color: getStatusColor(systemHealth?.wsConnectionStatus || (isConnected ? 'connected' : 'disconnected')),
        description: 'Status of the real-time WebSocket connection to the backend.'
      },
      {
        label: 'API Response Time',
        value: systemHealth?.apiResponseTime || 'N/A',
        formatType: 'number',
        unit: 'ms',
        description: 'Average response time for backend API calls.'
      },
      {
        label: 'Memory Usage',
        value: systemHealth?.memoryUsage || 'N/A',
        formatType: 'number',
        unit: 'MB',
        description: 'Backend memory consumption.'
      },
      {
        label: 'CPU Usage',
        value: systemHealth?.cpuUsage || 'N/A',
        formatType: 'number',
        unit: '%',
        description: 'Backend CPU utilization.'
      }
    ];

    // Only calculate snapshot-based metrics if snapshot exists
    if (snapshot) {
      // Add snapshot-based metrics
      baseMetrics.push(
        {
          label: 'Total Sales (Snapshot)',
          value: formatCurrency(currentMetrics.totalSales),
          formatType: 'currency' as const,
          trend: formatCurrency(currentMetrics.totalSales) > 0 ? 'up' : 'neutral'
        },
        {
          label: 'Our Product Price (Snapshot)',
          value: formatCurrency(currentMetrics.ourProductPrice),
          formatType: 'currency' as const
        },
        {
          label: 'Trust Score (Snapshot)',
          value: formatTrustScore(currentMetrics.trustScore),
          formatType: 'percentage' as const,
          trend: currentMetrics.trustScore > 0.7 ? 'up' :
                 currentMetrics.trustScore > 0.3 ? 'neutral' : 'down'
        },
        {
          label: 'Active Competitors (Snapshot)',
          value: currentMetrics.competitorCount,
          formatType: 'number' as const
        },
        {
          label: 'Recent Sales (Snapshot)',
          value: currentMetrics.recentSalesCount,
          formatType: 'number' as const,
          trend: currentMetrics.recentSalesCount > 0 ? 'up' : 'neutral'
        },
        {
          label: 'Avg Competitor Price (Snapshot)',
          value: averageCompetitorPrice,
          formatType: 'currency' as const
        },
        {
          label: 'Recent Sales Value (Snapshot)',
          value: recentSalesValue,
          formatType: 'currency' as const,
          trend: recentSalesValue > 0 ? 'up' : 'neutral'
        }
      );
    }

    return baseMetrics;
  }, [simulation, isConnected, currentMetrics, getStatusColor, averageCompetitorPrice, recentSalesValue]);

  if (simulation.isLoading && !simulation.snapshot && simulation.status === 'idle') {
    return (
      <div className={`space-y-6 ${className}`}>
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-gray-900">FBA Simulation Dashboard</h2>
          <ConnectionStatusCompact />
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {Array.from({ length: 6 }).map((_, index) => (
            <MetricCardSkeleton key={index} />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className={`space-y-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">FBA Simulation Dashboard</h2>
          <p className="text-gray-600 mt-1">
            Real-time performance metrics and analytics
          </p>
        </div>
        <ConnectionStatusCompact />
      </div>

      {/* Simulation Overview and Progress */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Progress</h3>
        <div className="flex items-center mb-4">
          <span className={`text-xl font-bold ${getStatusColor(simulation.status as SimulationStatus['status'])}`}>
            Status: {simulation.status ? simulation.status.toUpperCase() : 'N/A'}
          </span>
          <span className="ml-4 text-gray-600 flex items-center">
            <svg className="w-5 h-5 mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg"><path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            Tick: {currentMetrics.currentTick} / {currentMetrics.totalTicks}
          </span>
        </div>
        <Progress value={progress} className="w-full h-3 mb-2" />
        <div className="flex justify-between text-sm text-gray-600">
          <span>0%</span>
          <span>{progress.toFixed(1)}% Complete</span>
          <span>100%</span>
        </div>
      </div>

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6 overflow-x-auto">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-6 min-w-max">
          {metrics.map((metric, index) => (
            simulation.error ? (
              <MetricCardError
                key={index}
                label={metric.label}
                error="Data unavailable"
                className="min-w-[250px]"
              />
            ) : (
              <MetricCard
                key={index}
                metric={metric}
                className="min-w-[250px]"
              />
            )
          ))}
        </div>
      </div>

      {/* Information section (updated to include system health and connection) */}
      <div className="bg-gray-50 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-gray-900 mb-3">System Information</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 2xl:grid-cols-5 gap-4 text-sm">
          <div>
            <span className="text-gray-500">Connection:</span>
            <span className={`ml-2 font-medium ${getStatusColor(isConnected ? 'connected' : 'disconnected')}`}>
              {isConnected ? 'LIVE' : 'DISCONNECTED'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Last Data Update:</span>
            <span className="ml-2 font-medium">
              {simulation.snapshot ? new Date(simulation.snapshot.timestamp).toLocaleTimeString() : 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Backend API (mock):</span>
            <span className="ml-2 font-medium">{simulation.systemHealth?.apiResponseTime || 'N/A'} ms</span>
          </div>
          <div>
            <span className="text-gray-500">Backend Memory (mock):</span>
            <span className="ml-2 font-medium">{simulation.systemHealth?.memoryUsage || 'N/A'} MB</span>
          </div>
          <div>
            <span className="text-gray-500">Backend CPU (mock):</span>
            <span className="ml-2 font-medium">{simulation.systemHealth?.cpuUsage || 'N/A'}%</span>
          </div>
          <div>
            <span className="text-gray-500">Database Status (mock):</span>
            <span className={`ml-2 font-medium ${getStatusColor(simulation.systemHealth?.dbConnectionStatus || 'disconnected')}`}>
              {simulation.systemHealth?.dbConnectionStatus ? simulation.systemHealth.dbConnectionStatus.toUpperCase() : 'N/A'}
            </span>
          </div>
          <div>
            <span className="text-gray-500">Queue Length (mock):</span>
            <span className="ml-2 font-medium">{simulation.systemHealth?.queueLength || 'N/A'}</span>
          </div>
        </div>
      </div>

      {/* Error state */}
      {simulation.error && (
        <div className="bg-red-50 border border-red-200 rounded-md p-4">
          <div className="flex items-center space-x-2">
            <svg className="w-5 h-5 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-red-700 font-medium">Error loading dashboard data</span>
          </div>
          <p className="text-red-600 mt-1 text-sm">{simulation.error}</p>
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function to prevent unnecessary re-renders
  return (
    prevProps.className === nextProps.className &&
    prevProps.refreshInterval === nextProps.refreshInterval
  );
});

export default KPIDashboard;