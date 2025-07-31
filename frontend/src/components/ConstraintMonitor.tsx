import React from 'react';
import type { ConstraintUsage } from '../types';
import HelpTooltip from './HelpTooltip';

interface ConstraintMonitorProps {
  usage: ConstraintUsage;
}

const ConstraintMonitor: React.FC<ConstraintMonitorProps> = ({ usage }) => {
  const getProgressColor = (health: 'safe' | 'warning' | 'critical') => {
    switch (health) {
      case 'safe':
        return 'bg-green-500';
      case 'warning':
        return 'bg-yellow-500';
      case 'critical':
        return 'bg-red-500';
      default:
        return 'bg-gray-400';
    }
  };

  const MetricProgressBar: React.FC<{
    label: string;
    used: number;
    limit: number;
    health: 'safe' | 'warning' | 'critical';
    unit: string;
    estimatedTime?: string;
    helpText: string;
  }> = ({ label, used, limit, health, unit, estimatedTime, helpText }) => {
    const percentage = limit > 0 ? (used / limit) * 100 : 0;
    const displayUsed = unit === '$' ? `$${used.toFixed(2)}` : used.toLocaleString();
    const displayLimit = unit === '$' ? `$${limit.toFixed(2)}` : limit.toLocaleString();

    return (
      <div className="mb-4">
        <div className="flex justify-between items-center text-sm font-medium text-gray-700">
          <span>
            {label}: {displayUsed} / {displayLimit} {unit !== '$' ? unit : ''} ({percentage.toFixed(1)}% used)
            <HelpTooltip content={helpText} />
          </span>
          <span className={`text-${health}-600 font-semibold uppercase`}>
            {health}
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2.5 mt-1">
          <div
            className={`${getProgressColor(health)} h-2.5 rounded-full`}
            style={{ width: `${Math.min(100, percentage)}%` }}
          ></div>
        </div>
        {estimatedTime && (
          <p className="mt-1 text-xs text-gray-500">
            Estimated time until limit reached: {estimatedTime}
          </p>
        )}
      </div>
    );
  };

  return (
    <div className="p-4 bg-white shadow rounded-lg">
      <h2 className="text-xl font-semibold mb-4">Constraint Monitoring</h2>
      <p className="mb-4 text-gray-700">
        Monitor your current simulation's resource usage against configured limits.
        <HelpTooltip content="Real-time tracking of budget, token, rate, and memory usage helps you understand resource consumption and avoid unexpected interruptions." />
      </p>

      <MetricProgressBar
        label="Budget Usage"
        used={usage.budgetUsedUSD}
        limit={usage.budgetLimitUSD}
        health={usage.budgetHealth}
        unit="$"
        estimatedTime={usage.estimatedTimeUntilBudgetReached}
        helpText="Tracks the estimated cost incurred by the simulation against your set budget limit."
      />
      <MetricProgressBar
        label="Token Usage"
        used={usage.tokenUsed}
        limit={usage.tokenLimit}
        health={usage.tokenHealth}
        unit="tokens"
        estimatedTime={usage.estimatedTimeUntilTokensReached}
        helpText="Monitors the number of tokens consumed by LLM interactions compared to the maximum allowed."
      />
      <MetricProgressBar
        label="Rate Usage"
        used={usage.rateUsed}
        limit={usage.rateLimit}
        health={usage.rateHealth}
        unit="calls/min"
        helpText="Shows the current rate of API calls against the per-minute rate limit. High usage may lead to throttling."
      />
      <MetricProgressBar
        label="Memory Usage"
        used={usage.memoryUsedMB}
        limit={usage.memoryLimitMB}
        health={usage.memoryHealth}
        unit="MB"
        helpText="Displays the memory currently utilized by agents in the simulation against their allocated memory limit."
      />
    </div>
  );
};

export default ConstraintMonitor;