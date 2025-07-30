import React from 'react';
import type { ExperimentConfig } from '../types';

interface ExperimentConfigFormProps {
  config: ExperimentConfig;
  onConfigChange: (newConfig: ExperimentConfig) => void;
}

export function ExperimentConfigForm({ config, onConfigChange }: ExperimentConfigFormProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    onConfigChange({
      ...config,
      [name]: type === 'number' ? parseFloat(value) : value,
    });
  };

  const handleParameterSweepChange = (paramName: string, event: React.ChangeEvent<HTMLTextAreaElement>) => {
    try {
      // Split by comma and trim whitespace, then parse each value
      const values = event.target.value.split(',').map(s => {
        const trimmed = s.trim();
        // Attempt to parse as number, otherwise keep as string
        return isNaN(Number(trimmed)) ? trimmed : Number(trimmed);
      });
      onConfigChange({
        ...config,
        parameter_sweep: {
          ...config.parameter_sweep,
          [paramName]: values,
        },
      });
    } catch (e) {
      console.error("Error parsing parameter sweep values:", e);
      // Optionally provide user feedback here
    }
  };

  // Helper to convert array to comma-separated string for display
  const formatParameterSweepValue = (paramName: string) => {
    return (config.parameter_sweep?.[paramName] || []).join(', ');
  };


  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Experiment Configuration (Parameter Sweep)
      </h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="experimentName" className="block text-sm font-medium text-gray-700">
            Experiment Name
          </label>
          <input
            type="text"
            name="experiment_name"
            id="experimentName"
            value={config.experiment_name || ''}
            onChange={handleChange}
            placeholder="e.g., PricingStrategyComparison"
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
          />
        </div>
        <div>
          <label htmlFor="experimentDescription" className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <textarea
            name="description"
            id="experimentDescription"
            value={config.description || ''}
            onChange={handleChange}
            rows={3}
            placeholder="A description of what this experiment aims to achieve"
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
          ></textarea>
        </div>

        <h3 className="text-xl font-medium text-gray-800 mt-6 mb-3">Parameter Sweep</h3>
        <p className="text-gray-600 text-sm mb-4">
          Enter comma-separated values for each parameter you want to sweep.
          (e.g., initial_price: 29.99, 39.99, 49.99)
        </p>

        {/* Dynamic Parameter Sweep Inputs */}
        <div className="border border-gray-200 rounded-md p-4 space-y-3">
          {Object.keys(config.parameter_sweep || {}).map((paramName) => (
            <div key={paramName}>
              <label htmlFor={`sweep-${paramName}`} className="block text-sm font-medium text-gray-700">
                {paramName.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
              </label>
              <textarea
                name={paramName}
                id={`sweep-${paramName}`}
                value={formatParameterSweepValue(paramName)}
                onChange={(e) => handleParameterSweepChange(paramName, e)}
                rows={1}
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
                placeholder="e.g., val1, val2, val3"
              />
            </div>
          ))}
          <div>
            <label htmlFor="newParam" className="block text-sm font-medium text-gray-700">
              Add New Parameter (e.g., `new_parameter_name`)
            </label>
            <input
              type="text"
              id="newParam"
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
              placeholder="Enter new parameter name and press Enter"
              onKeyDown={(e) => {
                if (e.key === 'Enter' && e.currentTarget.value.trim()) {
                  const newParamName = e.currentTarget.value.trim();
                  if (!config.parameter_sweep || !config.parameter_sweep[newParamName]) {
                    onConfigChange({
                      ...config,
                      parameter_sweep: {
                        ...(config.parameter_sweep || {}),
                        [newParamName]: [], // Initialize with empty array
                      },
                    });
                  }
                  e.currentTarget.value = ''; // Clear input
                  e.preventDefault(); // Prevent form submission
                }
              }}
            />
          </div>
        </div>

        <h3 className="text-xl font-medium text-gray-800 mt-6 mb-3">Output Settings</h3>
        <div className="space-y-2">
          <div>
            <label htmlFor="parallelWorkers" className="block text-sm font-medium text-gray-700">
              Parallel Workers
            </label>
            <input
              type="number"
              name="parallel_workers"
              id="parallelWorkers"
              value={config.parallel_workers || 1}
              onChange={handleChange}
              min="1"
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
            />
          </div>
          <div>
            <label htmlFor="maxRuns" className="block text-sm font-medium text-gray-700">
              Max Runs (Optional, for testing)
            </label>
            <input
              type="number"
              name="max_runs"
              id="maxRuns"
              value={config.max_runs || ''}
              onChange={handleChange}
              min="1"
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
            />
          </div>
          {/* Add more output configuration fields as needed based on sweep.yaml */}
          <div className="flex items-center">
            <input
              type="checkbox"
              name="save_events"
              id="saveEvents"
              checked={config.output_config?.save_events || false}
              onChange={(e) => onConfigChange({
                ...config,
                output_config: {
                  ...config.output_config,
                  save_events: e.target.checked
                }
              })}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded"
            />
            <label htmlFor="saveEvents" className="ml-2 block text-sm text-gray-700">
              Save Full Event Stream
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              name="save_snapshots"
              id="saveSnapshots"
              checked={config.output_config?.save_snapshots || false}
              onChange={(e) => onConfigChange({
                ...config,
                output_config: {
                  ...config.output_config,
                  save_snapshots: e.target.checked
                }
              })}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded"
            />
            <label htmlFor="saveSnapshots" className="ml-2 block text-sm text-gray-700">
              Save Periodic Snapshots
            </label>
          </div>
          {config.output_config?.save_snapshots && (
            <div>
              <label htmlFor="snapshotInterval" className="block text-sm font-medium text-gray-700">
                Snapshot Interval (hours)
              </label>
              <input
                type="number"
                name="snapshot_interval_hours"
                id="snapshotInterval"
                value={config.output_config?.snapshot_interval_hours || ''}
                onChange={(e) => onConfigChange({
                  ...config,
                  output_config: {
                    ...config.output_config,
                    snapshot_interval_hours: parseFloat(e.target.value)
                  }
                })}
                min="1"
                step="0.1"
                className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
