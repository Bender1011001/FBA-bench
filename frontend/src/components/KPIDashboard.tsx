import React, { useEffect, useMemo, useState, useCallback, useRef } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { apiService, getAuditStats, getLedgerSnapshot, getBsrSnapshot, getFeeSummary, type AuditStatsResponse, type LedgerSnapshot, type BsrSnapshot, type FeeSummaryByType } from '../services/apiService';
import { webSocketService } from '../services/webSocketService'; // Import the singleton
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
    setSystemHealth
  } = useSimulationStore();

  const [progress, setProgress] = useState(0);
  // Use a local state to track connection status from the singleton service
  const [localConnectionStatus, setLocalConnectionStatus] =
    useState<'connecting' | 'connected' | 'disconnected' | 'reconnecting'>(webSocketService.getConnectionStatus());

  // KPI datasets: independent loading/error/data states
  const [auditStats, setAuditStats] = useState<AuditStatsResponse | null>(null);
  const [auditLoading, setAuditLoading] = useState<boolean>(false);
  const [auditError, setAuditError] = useState<string | null>(null);

  const [ledger, setLedger] = useState<LedgerSnapshot | null>(null);
  const [ledgerLoading, setLedgerLoading] = useState<boolean>(false);
  const [ledgerError, setLedgerError] = useState<string | null>(null);

  const [bsr, setBsr] = useState<BsrSnapshot | null>(null);
  const [bsrLoading, setBsrLoading] = useState<boolean>(false);
  const [bsrError, setBsrError] = useState<string | null>(null);

  const [fees, setFees] = useState<FeeSummaryByType | null>(null);
  const [feesLoading, setFeesLoading] = useState<boolean>(false);
  const [feesError, setFeesError] = useState<string | null>(null);

  // Mounted guard to avoid state updates after unmount
  const mountedRef = useRef<boolean>(false);

  useEffect(() => {
    // Define callback functions for WebSocketService events
    const handleWsEvent = (eventType: string, data: unknown) => {
      // console.log(`KPIDashboard: Received event type: ${eventType}`, data);
      switch (eventType) {
        case 'simulation_status':
          setSimulationStatus(data as SimulationStatus);
          break;
        case 'financial_metrics':
          // The API server sends total_revenue, total_profit, etc., as part of financial_metrics
          // Update the store's financial summary directly or trigger a fetch for a full snapshot
          console.log('Received financial_metrics, might need to update store:', data);
          break;
        // Add more specific event handlers here as needed
        default:
          console.log(`KPIDashboard: Unhandled general event: ${eventType}`, data);
      }
    };

    const handleSnapshot = (data: unknown) => {
      setSnapshot(data as SimulationSnapshot);
    };

    const handleWsError = (error: Error) => {
      console.error('KPIDashboard: WebSocket error:', error);
      setError(`WebSocket error: ${error.message}`);
    };

    const handleConnectionStatusChange = (status: 'connecting' | 'connected' | 'disconnected' | 'reconnecting') => {
      // console.log('KPIDashboard: Connection status changed:', status);
      setLocalConnectionStatus(status);
      setError(status === 'disconnected' ? 'WebSocket disconnected. Attempting to reconnect...' : null);
    };

    const handleSystemHealth = (data: SystemHealth) => {
      setSystemHealth(data);
    };

    // Subscribe to the WebSocketService
    webSocketService.subscribe('kpi-dashboard', {
      onEvent: handleWsEvent,
      onSnapshot: handleSnapshot,
      onError: handleWsError,
      onConnectionStatusChange: handleConnectionStatusChange,
      onConnectionEstablished: () => console.log('KPIDashboard: WebSocket connection established.'),
      onSystemHealth: handleSystemHealth, // Use the new handler here
      onMetricsUpdate: (data) => console.log('KPIDashboard: Metrics update (benchmark):', data), // Keep for debugging if needed
      onExecutionUpdate: (data) => console.log('KPIDashboard: Execution update (benchmark):', data), // Keep for debugging if needed
    });

    // Cleanup subscription on component unmount
    return () => {
      webSocketService.unsubscribe('kpi-dashboard');
    };
  }, [setSnapshot, setSimulationStatus, setSystemHealth, setError]); // Dependencies for useEffect

  // ---- KPI data fetchers (independent) ----
  const fetchAudit = useCallback(async () => {
    setAuditLoading(true);
    setAuditError(null);
    try {
      const data = await getAuditStats();
      if (!mountedRef.current) return;
      setAuditStats(data);
    } catch (e) {
      if (!mountedRef.current) return;
      setAuditError(e instanceof Error ? e.message : 'Failed to fetch audit metrics');
    } finally {
      if (mountedRef.current) setAuditLoading(false);
    }
  }, []);

  const fetchLedger = useCallback(async () => {
    setLedgerLoading(true);
    setLedgerError(null);
    try {
      const data = await getLedgerSnapshot();
      if (!mountedRef.current) return;
      setLedger(data);
    } catch (e) {
      if (!mountedRef.current) return;
      setLedgerError(e instanceof Error ? e.message : 'Failed to fetch ledger snapshot');
    } finally {
      if (mountedRef.current) setLedgerLoading(false);
    }
  }, []);

  const fetchBsr = useCallback(async () => {
    setBsrLoading(true);
    setBsrError(null);
    try {
      const data = await getBsrSnapshot();
      if (!mountedRef.current) return;
      setBsr(data);
    } catch (e) {
      if (!mountedRef.current) return;
      setBsrError(e instanceof Error ? e.message : 'Failed to fetch BSR indices');
    } finally {
      if (mountedRef.current) setBsrLoading(false);
    }
  }, []);

  const fetchFees = useCallback(async () => {
    setFeesLoading(true);
    setFeesError(null);
    try {
      const data = await getFeeSummary();
      if (!mountedRef.current) return;
      setFees(data);
    } catch (e) {
      if (!mountedRef.current) return;
      setFeesError(e instanceof Error ? e.message : 'Failed to fetch fee summary');
    } finally {
      if (mountedRef.current) setFeesLoading(false);
    }
  }, []);

  const fetchAllKpi = useCallback(async () => {
    await Promise.allSettled([fetchAudit(), fetchLedger(), fetchBsr(), fetchFees()]);
  }, [fetchAudit, fetchLedger, fetchBsr, fetchFees]);

  // Mount/unmount guard + initial fetch
  useEffect(() => {
    mountedRef.current = true;
    fetchAllKpi();
    const intervalId = window.setInterval(fetchAllKpi, 15000); // 15s refresh
    return () => {
      mountedRef.current = false;
      clearInterval(intervalId);
    };
  }, [fetchAllKpi]);

  // Fetch initial simulation data (fallback if WebSocket is not used or for initial state)
  const fetchInitialData = async () => {
    try {
      setLoading(true);
      setError(null);
      const response = await apiService.get<SimulationSnapshot>('/api/v1/simulation/snapshot');
      const snapshot = response.data || {}; // Ensure snapshot is an object even if response.data is null/undefined
      setSnapshot(snapshot as SimulationSnapshot);
      
      const currentTick = snapshot.current_tick || 0;
      const totalTicks = snapshot.metadata?.total_ticks || 1000;
      const simulationTime = snapshot.simulation_time || 'N/A';
      const lastUpdate = snapshot.last_update || new Date().toISOString();
      const realTime = new Date(lastUpdate).toLocaleTimeString();
      const ticksPerSecond = snapshot.event_stats?.ticks_per_second || 0;
      
      const financialSummary = snapshot.financial_summary || {};
      const totalRevenue = formatCurrency(financialSummary.total_revenue?.amount || 0);
      const totalCosts = formatCurrency(financialSummary.total_costs?.amount || 0);
      const totalProfit = formatCurrency(financialSummary.total_profit?.amount || 0);
      
      const activeAgentCount = snapshot.agents ? Object.keys(snapshot.agents).length : 0;

      setSimulationStatus({
        id: (snapshot.metadata as { simulation_id?: string })?.simulation_id || 'N/A',
        status: (snapshot.metadata as { status?: string })?.status || 'idle',
        currentTick: currentTick,
        totalTicks: totalTicks,
        simulationTime: simulationTime,
        realTime: realTime,
        ticksPerSecond: ticksPerSecond,
        revenue: totalRevenue,
        costs: totalCosts,
        profit: totalProfit,
        activeAgentCount: activeAgentCount,
      });
      // SystemHealth primarily comes from WebSocket (real-time). If the snapshot contains health info, use it.
      // Use a typed, defensive extraction to avoid `any` and ensure values align with SystemHealth.
      const snapshotSystemHealth = (snapshot as Partial<{ system_health?: Partial<SystemHealth> }>)?.system_health;
      if (snapshotSystemHealth && typeof snapshotSystemHealth === 'object') {
        setSystemHealth({
          apiResponseTime: typeof snapshotSystemHealth.apiResponseTime === 'number' ? snapshotSystemHealth.apiResponseTime : 0,
          wsConnectionStatus: (snapshotSystemHealth.wsConnectionStatus as SystemHealth['wsConnectionStatus']) ?? (localConnectionStatus === 'connected' ? 'connected' : 'disconnected'),
          memoryUsage: typeof snapshotSystemHealth.memoryUsage === 'number' ? snapshotSystemHealth.memoryUsage : 0,
          cpuUsage: typeof snapshotSystemHealth.cpuUsage === 'number' ? snapshotSystemHealth.cpuUsage : 0,
          dbConnectionStatus: snapshotSystemHealth.dbConnectionStatus === 'connected' ? 'connected' : 'disconnected',
          queueLength: typeof snapshotSystemHealth.queueLength === 'number' ? snapshotSystemHealth.queueLength : 0,
        });
      } else {
        // Do not inject mocked numbers in production; use conservative defaults so the UI shows "N/A" or zero instead of fake values.
        setSystemHealth({
          apiResponseTime: 0,
          wsConnectionStatus: localConnectionStatus === 'connected' ? 'connected' : 'disconnected',
          memoryUsage: 0,
          cpuUsage: 0,
          dbConnectionStatus: localConnectionStatus === 'connected' ? 'connected' : 'disconnected',
          queueLength: 0,
        });
      }

    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch simulation data';
      console.error('Failed to fetch simulation data:', error);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Memoize fetchInitialData to prevent unnecessary recreations
  const memoizedFetchInitialData = useCallback(fetchInitialData, [setSnapshot, setLoading, setError, setSimulationStatus, setSystemHealth, apiService, localConnectionStatus]); // Add localConnectionStatus dependency

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
        value: localConnectionStatus === 'connected' ? 'Connected' : 'Disconnected', // Use localConnectionStatus
        formatType: 'string',
        color: getStatusColor(systemHealth?.wsConnectionStatus || (localConnectionStatus === 'connected' ? 'connected' : 'disconnected')), // Use localConnectionStatus
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
  }, [simulation, localConnectionStatus, currentMetrics, getStatusColor, averageCompetitorPrice, recentSalesValue]); // Depend on localConnectionStatus

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
            Real-time simulation metrics and analytics
          </p>
        </div>
        <div className="flex items-center gap-3">
          <button
            onClick={fetchAllKpi}
            className="inline-flex items-center px-3 py-2 rounded-md bg-blue-600 text-white hover:bg-blue-700 transition-colors text-sm"
            aria-label="Refresh all KPI data"
          >
            <svg className="w-4 h-4 mr-2" viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
              <path fillRule="evenodd" d="M3 10a7 7 0 0112.452-4.391l1.31-1.312a1 1 0 011.415 1.415l-3 3A1 1 0 0113 8V5a1 1 0 112 0v1.528A9 9 0 102 10a1 1 0 112 0z" clipRule="evenodd" />
            </svg>
            Refresh All
          </button>
          <ConnectionStatusCompact />
        </div>
      </div>

      {/* Quick Actions */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Quick Actions</h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <button className="flex flex-col items-center justify-center p-4 bg-blue-50 rounded-lg hover:bg-blue-100 transition-colors">
            <svg className="w-8 h-8 text-blue-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M12 6v6m0 0v6m0-6h6m-6 0H6"></path>
            </svg>
            <span className="text-sm font-medium text-blue-800">New Benchmark</span>
          </button>
          <button className="flex flex-col items-center justify-center p-4 bg-green-50 rounded-lg hover:bg-green-100 transition-colors">
            <svg className="w-8 h-8 text-green-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path>
            </svg>
            <span className="text-sm font-medium text-green-800">Run Benchmark</span>
          </button>
          <button className="flex flex-col items-center justify-center p-4 bg-purple-50 rounded-lg hover:bg-purple-100 transition-colors">
            <svg className="w-8 h-8 text-purple-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z"></path>
            </svg>
            <span className="text-sm font-medium text-purple-800">View Results</span>
          </button>
          <button className="flex flex-col items-center justify-center p-4 bg-yellow-50 rounded-lg hover:bg-yellow-100 transition-colors">
            <svg className="w-8 h-8 text-yellow-600 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path>
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path>
            </svg>
            <span className="text-sm font-medium text-yellow-800">Settings</span>
          </button>
        </div>
      </div>

      {/* Recent Benchmark Runs */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <h3 className="text-lg font-semibold text-gray-900">Real-time Metrics</h3>
        {/* Placeholder for real-time charts/data coming soon */}
        <p className="text-gray-600">Charts will appear here as simulation data streams in.</p>
      </div>

      {/* Financial Overview (cards row) */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900">Financial Overview</h3>
          <div className="flex gap-2">
            <button onClick={fetchLedger} className="text-sm px-3 py-1 rounded-md bg-gray-100 hover:bg-gray-200">Refresh Ledger</button>
            <button onClick={fetchAudit} className="text-sm px-3 py-1 rounded-md bg-gray-100 hover:bg-gray-200">Refresh Audit</button>
          </div>
        </div>
        {(ledgerError || auditError) && (
          <div className="mb-4 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">
            {ledgerError && <div>Ledger: {ledgerError}</div>}
            {auditError && <div>Audit: {auditError}</div>}
          </div>
        )}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Net Worth (derived label) */}
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Net Worth</div>
            {ledgerLoading ? (
              <div className="mt-2 h-5 w-24 animate-pulse bg-gray-200 rounded" aria-label="Loading net worth" />
            ) : (
              <div className="mt-1 text-gray-900">
                Assets: <span className="font-medium">{ledger?.total_assets ?? '—'}</span>
                <span className="mx-1 text-gray-400">|</span>
                Liabilities: <span className="font-medium">{ledger?.total_liabilities ?? '—'}</span>
              </div>
            )}
          </div>

          {/* Cash */}
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Cash</div>
            {ledgerLoading ? (
              <div className="mt-2 h-5 w-20 animate-pulse bg-gray-200 rounded" aria-label="Loading cash" />
            ) : (
              <div className="mt-1 font-semibold text-gray-900">{ledger?.cash ?? '—'}</div>
            )}
          </div>

          {/* Inventory */}
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Inventory</div>
            {ledgerLoading ? (
              <div className="mt-2 h-5 w-24 animate-pulse bg-gray-200 rounded" aria-label="Loading inventory" />
            ) : (
              <div className="mt-1 font-semibold text-gray-900">{ledger?.inventory_value ?? '—'}</div>
            )}
          </div>

          {/* Processed Transactions */}
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Processed Transactions</div>
            {auditLoading ? (
              <div className="mt-2 h-5 w-24 animate-pulse bg-gray-200 rounded" aria-label="Loading processed transactions" />
            ) : (
              <div className="mt-1 font-semibold text-gray-900">{auditStats?.processed_transactions ?? '—'}</div>
            )}
          </div>
        </div>
      </div>

      {/* Accounting Integrity badge */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900">Accounting Integrity</h3>
          <button onClick={fetchAudit} className="text-sm px-3 py-1 rounded-md bg-gray-100 hover:bg-gray-200">Retry</button>
        </div>
        {auditError && <div className="mb-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{auditError}</div>}
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${auditStats?.current_position?.accounting_identity_valid ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'}`}>
            Accounting Identity: {auditStats?.current_position?.accounting_identity_valid ? 'Valid' : 'Invalid'}
          </span>
          <div className="text-sm text-gray-600">
            Assets: <span className="font-medium">{auditStats?.current_position?.total_assets ?? '—'}</span>
            <span className="mx-1 text-gray-400">|</span>
            Liabilities: <span className="font-medium">{auditStats?.current_position?.total_liabilities ?? '—'}</span>
            <span className="mx-1 text-gray-400">|</span>
            Equity: <span className="font-medium">{auditStats?.current_position?.total_equity ?? '—'}</span>
            <span className="mx-1 text-gray-400">|</span>
            Assets-(L+E): <span className="font-medium">{auditStats?.current_position?.identity_difference ?? '—'}</span>
          </div>
        </div>
      </div>

      {/* Revenue/Fee/Profit audited */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900">Audited Totals</h3>
          <button onClick={fetchAudit} className="text-sm px-3 py-1 rounded-md bg-gray-100 hover:bg-gray-200">Refresh</button>
        </div>
        {auditError && <div className="mb-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{auditError}</div>}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Total Revenue Audited</div>
            {auditLoading ? <div className="mt-2 h-5 w-28 animate-pulse bg-gray-200 rounded" /> : <div className="mt-1 font-semibold text-gray-900">{auditStats?.total_revenue_audited ?? '—'}</div>}
          </div>
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Total Fees Audited</div>
            {auditLoading ? <div className="mt-2 h-5 w-28 animate-pulse bg-gray-200 rounded" /> : <div className="mt-1 font-semibold text-gray-900">{auditStats?.total_fees_audited ?? '—'}</div>}
          </div>
          <div className="p-4 rounded-md border border-gray-200">
            <div className="text-sm text-gray-500">Total Profit Audited</div>
            {auditLoading ? <div className="mt-2 h-5 w-28 animate-pulse bg-gray-200 rounded" /> : <div className="mt-1 font-semibold text-gray-900">{auditStats?.total_profit_audited ?? '—'}</div>}
          </div>
        </div>
      </div>

      {/* BSR Indices table */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900">BSR Indices (v3)</h3>
          <button onClick={fetchBsr} className="text-sm px-3 py-1 rounded-md bg-gray-100 hover:bg-gray-200">Refresh</button>
        </div>
        {bsrError && <div className="mb-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{bsrError}</div>}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">ASIN</th>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Velocity Index</th>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Conversion Index</th>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Composite Index</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {(bsr?.products ?? []).map((p, idx) => (
                <tr key={`${p.asin}-${idx}`}>
                  <td className="px-4 py-2 font-mono">{p.asin || '—'}</td>
                  <td className="px-4 py-2">{p.velocity_index === null || p.velocity_index === undefined ? '—' : p.velocity_index.toFixed(3)}</td>
                  <td className="px-4 py-2">{p.conversion_index === null || p.conversion_index === undefined ? '—' : p.conversion_index.toFixed(3)}</td>
                  <td className="px-4 py-2">{p.composite_index === null || p.composite_index === undefined ? '—' : p.composite_index.toFixed(3)}</td>
                </tr>
              ))}
              {bsrLoading && (
                <tr>
                  <td className="px-4 py-3 text-gray-500" colSpan={4}>Loading BSR indices…</td>
                </tr>
              )}
              {!bsrLoading && (bsr?.products?.length ?? 0) === 0 && (
                <tr>
                  <td className="px-4 py-3 text-gray-500" colSpan={4}>No BSR data</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        {(bsr?.market_ema_velocity || bsr?.market_ema_conversion || typeof bsr?.competitor_count === 'number') && (
          <div className="mt-3 text-xs text-gray-600">
            {bsr?.market_ema_velocity && <span className="mr-3">Market EMA Velocity: <span className="font-medium">{bsr.market_ema_velocity}</span></span>}
            {bsr?.market_ema_conversion && <span className="mr-3">Market EMA Conversion: <span className="font-medium">{bsr.market_ema_conversion}</span></span>}
            {typeof bsr?.competitor_count === 'number' && <span>Competitor Count: <span className="font-medium">{bsr.competitor_count}</span></span>}
          </div>
        )}
      </div>

      {/* Fee Summary by Type */}
      <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-lg font-semibold text-gray-900">Fee Summary by Type</h3>
          <button onClick={fetchFees} className="text-sm px-3 py-1 rounded-md bg-gray-100 hover:bg-gray-200">Refresh</button>
        </div>
        {feesError && <div className="mb-3 rounded-md border border-red-200 bg-red-50 p-3 text-sm text-red-700">{feesError}</div>}
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-gray-200 text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Fee Type</th>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Total Amount</th>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Count</th>
                <th className="px-4 py-2 text-left font-semibold text-gray-700">Average Amount</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {Object.entries(fees ?? {}).map(([feeType, summary]) => (
                <tr key={feeType}>
                  <td className="px-4 py-2 font-mono">{feeType}</td>
                  <td className="px-4 py-2">{summary.total_amount}</td>
                  <td className="px-4 py-2">{summary.count}</td>
                  <td className="px-4 py-2">{summary.average_amount}</td>
                </tr>
              ))}
              {feesLoading && (
                <tr>
                  <td className="px-4 py-3 text-gray-500" colSpan={4}>Loading fee summary…</td>
                </tr>
              )}
              {!feesLoading && Object.keys(fees ?? {}).length === 0 && (
                <tr>
                  <td className="px-4 py-3 text-gray-500" colSpan={4}>No fee data</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Simulation Overview and Progress */}
      {simulation.status !== 'idle' && (
        <div className="bg-white p-6 rounded-lg shadow-sm border border-gray-200">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Current Simulation Progress</h3>
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
      )}

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
            <span className={`ml-2 font-medium ${getStatusColor(localConnectionStatus === 'connected' ? 'connected' : 'disconnected')}`}>
              {localConnectionStatus === 'connected' ? 'LIVE' : 'DISCONNECTED'}
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