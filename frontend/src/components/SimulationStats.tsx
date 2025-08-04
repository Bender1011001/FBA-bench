import React, { useMemo } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { MetricCard } from './MetricCard';
import type { DashboardMetric } from '../types';

interface SimulationStatsProps {
  className?: string;
}

export const SimulationStats: React.FC<SimulationStatsProps> = ({
  className = '',
}) => {
  const { simulation } = useSimulationStore();
  const snapshot = simulation.snapshot;
  
  // Example statistics that could be derived from the snapshot or fetched separately
  const summaryMetrics: DashboardMetric[] = useMemo(() => {
    if (!snapshot) {
      return [];
    }
    
    // Financial Metrics
    const totalRevenue = parseFloat(snapshot.financial_summary?.total_revenue?.amount || '0');
    const totalCosts = parseFloat(snapshot.financial_summary?.total_costs?.amount || '0');
    const totalProfit = parseFloat(snapshot.financial_summary?.total_profit?.amount || '0');
    const totalSales = parseFloat(snapshot.financial_summary?.total_sales?.amount || '0');

    // Agent Metrics
    const totalAgents = Object.keys(snapshot.agents || {}).length;

    // Market Metrics
    const trustScore = snapshot.market_summary?.trust_score || 0;
    const competitorCount = snapshot.competitor_states?.length || 0;

    return [
      {
        label: 'Total Revenue',
        value: totalRevenue,
        formatType: 'currency',
        trend: totalRevenue > 0 ? 'up' : 'neutral',
        description: 'Total revenue generated during the simulation.'
      },
      {
        label: 'Total Costs',
        value: totalCosts,
        formatType: 'currency',
        trend: totalCosts > 0 ? 'up' : 'neutral',
        description: 'Total costs incurred during the simulation.'
      },
      {
        label: 'Net Profit',
        value: totalProfit,
        formatType: 'currency',
        trend: totalProfit > 0 ? 'up' : 'down',
        color: totalProfit > 0 ? 'text-green-500' : 'text-red-500',
        description: 'Overall profit/loss of the simulation.'
      },
      {
        label: 'Total Sales Volume',
        value: totalSales,
        formatType: 'currency',
        trend: totalSales > 0 ? 'up' : 'neutral',
        description: 'Total value of all sales transactions.'
      },
      {
        label: 'Total Agents',
        value: totalAgents,
        formatType: 'number',
        description: 'Total number of agents participating in the simulation.'
      },
      {
        label: 'Average Trust Score',
        value: trustScore * 100, // Display as percentage
        formatType: 'percentage',
        trend: trustScore > 0.7 ? 'up' : 'neutral',
        color: trustScore < 0.5 ? 'text-red-500' : ''
      },
      {
        label: 'Active Competitors',
        value: competitorCount,
        formatType: 'number',
        description: 'Number of active competitors in the market.'
      },
      // Add more aggregate statistics as needed, e.g.,
      // { label: 'Average Product Price', value: calculateAvgPrice(snapshot.products), formatType: 'currency' },
      // { label: 'Total Events Logged', value: simulation.eventLog.length, formatType: 'number' },
    ];
  }, [snapshot]);


  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-6 ${className}`}>
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Simulation Statistics Overview</h3>
      
      {!snapshot && (
        <div className="text-center py-8">
          <svg className="mx-auto h-12 w-12 text-gray-300" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
          </svg>
          <p className="mt-2 text-sm font-medium text-gray-900">No simulation statistics available</p>
          <p className="mt-1 text-sm text-gray-500">Run a simulation to generate and view statistics.</p>
        </div>
      )}

      {snapshot && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 overflow-x-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 min-w-max">
            {summaryMetrics.map((metric, index) => (
              <MetricCard key={index} metric={metric} className="min-w-[250px]" />
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

export default SimulationStats;