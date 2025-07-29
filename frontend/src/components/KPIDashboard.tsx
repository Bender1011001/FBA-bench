import React, { useEffect, useMemo } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { apiService } from '../services/apiService';
import { MetricCard, MetricCardSkeleton, MetricCardError } from './MetricCard';
import { ConnectionStatusCompact } from './ConnectionStatus';
import type { DashboardMetric } from '../types';

interface KPIDashboardProps {
  className?: string;
  refreshInterval?: number;
}

const formatCurrency = (amount: string): number => {
  // Parse currency string from backend (e.g., "$123.45" -> 123.45)
  const cleaned = amount.replace(/[$,]/g, '');
  return parseFloat(cleaned) || 0;
};

const formatTrustScore = (score: number): number => {
  // Trust score is typically 0-1, convert to percentage for display
  return score * 100;
};

export const KPIDashboard: React.FC<KPIDashboardProps> = ({ 
  className = '',
  refreshInterval = 5000 
}) => {
  const {
    simulation,
    setSnapshot,
    setLoading,
    setError,
    getCurrentMetrics
  } = useSimulationStore();

  // Initialize WebSocket connection
  const { isConnected } = useWebSocket({
    onConnect: () => {
      console.log('Dashboard: WebSocket connected');
      setError(null);
    },
    onDisconnect: () => {
      console.log('Dashboard: WebSocket disconnected');
    },
    onError: (error) => {
      console.error('Dashboard: WebSocket error:', error);
      setError('WebSocket connection failed');
    }
  });

  // Fetch initial simulation data
  const fetchSimulationData = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const snapshot = await apiService.getSimulationSnapshot();
      setSnapshot(snapshot);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch simulation data';
      console.error('Failed to fetch simulation data:', error);
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  // Set up periodic data refresh when WebSocket is not connected
  useEffect(() => {
    // Initial fetch
    fetchSimulationData();

    // Set up interval only if WebSocket is not connected
    let intervalId: number | undefined;
    
    if (!isConnected && refreshInterval > 0) {
      intervalId = window.setInterval(fetchSimulationData, refreshInterval);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isConnected, refreshInterval]);

  // Calculate metrics from current state
  const metrics = useMemo((): DashboardMetric[] => {
    const currentMetrics = getCurrentMetrics();
    const { snapshot } = simulation;

    // Basic metrics always available
    const baseMetrics: DashboardMetric[] = [
      {
        label: 'Current Tick',
        value: currentMetrics.currentTick,
        formatType: 'number',
        trend: currentMetrics.currentTick > 0 ? 'up' : 'neutral'
      },
      {
        label: 'Total Sales',
        value: formatCurrency(currentMetrics.totalSales),
        formatType: 'currency',
        trend: formatCurrency(currentMetrics.totalSales) > 0 ? 'up' : 'neutral'
      },
      {
        label: 'Our Product Price',
        value: formatCurrency(currentMetrics.ourProductPrice),
        formatType: 'currency'
      },
      {
        label: 'Trust Score',
        value: formatTrustScore(currentMetrics.trustScore),
        formatType: 'percentage',
        trend: currentMetrics.trustScore > 0.7 ? 'up' : 
               currentMetrics.trustScore > 0.3 ? 'neutral' : 'down'
      },
      {
        label: 'Active Competitors',
        value: currentMetrics.competitorCount,
        formatType: 'number'
      },
      {
        label: 'Recent Sales',
        value: currentMetrics.recentSalesCount,
        formatType: 'number',
        trend: currentMetrics.recentSalesCount > 0 ? 'up' : 'neutral'
      }
    ];

    // Add detailed metrics if snapshot is available
    if (snapshot) {
      const averageCompetitorPrice = snapshot.competitor_states.length > 0
        ? snapshot.competitor_states.reduce((sum, comp) => 
            sum + formatCurrency(comp.current_price.amount), 0
          ) / snapshot.competitor_states.length
        : 0;

      const recentSalesValue = snapshot.recent_sales.reduce((sum, sale) => 
        sum + formatCurrency(sale.sale_price.amount) * sale.quantity, 0
      );

      baseMetrics.push(
        {
          label: 'Avg Competitor Price',
          value: averageCompetitorPrice,
          formatType: 'currency'
        },
        {
          label: 'Recent Sales Value',
          value: recentSalesValue,
          formatType: 'currency',
          trend: recentSalesValue > 0 ? 'up' : 'neutral'
        }
      );
    }

    return baseMetrics;
  }, [simulation, getCurrentMetrics]);

  // Loading state
  if (simulation.isLoading && !simulation.snapshot) {
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

      {/* Refresh button for manual updates */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2 text-sm text-gray-500">
          {simulation.snapshot && (
            <span>
              Last updated: {new Date(simulation.snapshot.timestamp).toLocaleTimeString()}
            </span>
          )}
          {simulation.isLoading && (
            <span className="flex items-center space-x-1">
              <svg className="w-4 h-4 animate-spin" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
              </svg>
              <span>Updating...</span>
            </span>
          )}
        </div>
        
        <button
          onClick={fetchSimulationData}
          disabled={simulation.isLoading}
          className="px-4 py-2 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          {simulation.isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
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

      {/* Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {metrics.map((metric, index) => (
          simulation.error ? (
            <MetricCardError
              key={index}
              label={metric.label}
              error="Data unavailable"
            />
          ) : (
            <MetricCard
              key={index}
              metric={metric}
            />
          )
        ))}
      </div>

      {/* Additional Info */}
      {simulation.snapshot && !simulation.error && (
        <div className="bg-gray-50 rounded-lg p-4">
          <h3 className="text-lg font-semibold text-gray-900 mb-3">Simulation Details</h3>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Simulation Time:</span>
              <span className="ml-2 font-medium">Tick {simulation.snapshot.current_tick}</span>
            </div>
            <div>
              <span className="text-gray-500">Active Competitors:</span>
              <span className="ml-2 font-medium">{simulation.snapshot.competitor_states.length}</span>
            </div>
            <div>
              <span className="text-gray-500">Data Source:</span>
              <span className="ml-2 font-medium">{isConnected ? 'Real-time' : 'Polling'}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default KPIDashboard;