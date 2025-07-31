import React, { useState, useCallback, useMemo } from 'react';
import type { ExperimentConfig, ExperimentParameter } from '../types';
import HelpTooltip from './HelpTooltip';
import { getHelpText } from '../data/helpContent';

interface ExperimentConfigFormProps {
  config: ExperimentConfig;
  onConfigChange: (newConfig: ExperimentConfig) => void;
}

export function ExperimentConfigForm({ config, onConfigChange }: ExperimentConfigFormProps) {
  const [newParamName, setNewParamName] = useState<string>('');

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    onConfigChange({
      ...config,
      [name]: type === 'number' ? parseFloat(value) : value,
    });
  };

  const handleParameterChange = useCallback((index: number, field: string, value: string | number) => {
    const updatedParameters = [...(config.parameters || [])];
    const currentParam = updatedParameters[index];
    if (field === 'type') {
      // When type changes, reset other type-specific fields
      updatedParameters[index] = {
        ...currentParam,
        type: value as 'discrete' | 'range',
        values: undefined,
        min: undefined,
        max: undefined,
        step: undefined
      };
    } else if (field === 'values') {
      // Handle discrete values as an array of strings/numbers
      try {
        const parsedValues = String(value).split(',').map(s => {
          const trimmed = s.trim();
          return isNaN(Number(trimmed)) ? trimmed : Number(trimmed);
        });
        updatedParameters[index] = { ...currentParam, values: parsedValues };
      } catch (e) {
        console.error("Error parsing parameter sweep values:", e);
      }
    } else if (field === 'min' || field === 'max' || field === 'step') {
      // Handle numerical range values
      updatedParameters[index] = { ...currentParam, [field]: parseFloat(String(value)) };
    } else {
      updatedParameters[index] = { ...currentParam, [field]: value };
    }

    onConfigChange({ ...config, parameters: updatedParameters });
  }, [config, onConfigChange]);

  const addParameter = useCallback((paramName: string) => {
    if (paramName && !(config.parameters || []).some(p => p.name === paramName)) {
      onConfigChange({
        ...config,
        parameters: [
          ...(config.parameters || []),
          { name: paramName, type: 'discrete', values: [] }
        ],
      });
      setNewParamName('');
    }
  }, [config, onConfigChange]);

  const removeParameter = useCallback((index: number) => {
    const updatedParameters = (config.parameters || []).filter((_, i) => i !== index);
    onConfigChange({ ...config, parameters: updatedParameters });
  }, [config, onConfigChange]);

  const calculateCombinations = useMemo(() => {
    return (config.parameters || []).reduce((acc, param) => {
      let numValues = 0;
      if (param.type === 'discrete' && param.values) {
        numValues = param.values.length;
      } else if (param.type === 'range' && param.min !== undefined && param.max !== undefined && param.step !== undefined) {
        if (param.step === 0) return 0; // Avoid division by zero
        numValues = Math.floor((param.max - param.min) / param.step) + 1;
        if (numValues < 0) numValues = 0; // Ensure non-negative combinations
      }
      return acc * (numValues || 1); // If a parameter has no values, treat as 1 combination for calculation
    }, 1);
  }, [config.parameters]);

  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Experiment Configuration (Parameter Sweep)
      </h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="experimentName" className="block text-sm font-medium text-gray-700">
            Experiment Name
            <HelpTooltip content={getHelpText('experiment', 'experiment_name')} />
          </label>
          <input
            type="text"
            name="experimentName"
            id="experimentName"
            value={config.experimentName || ''}
            onChange={handleChange}
            placeholder="e.g., PricingStrategyComparison"
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
          />
        </div>
        <div>
          <label htmlFor="experimentDescription" className="block text-sm font-medium text-gray-700">
            Description
            <HelpTooltip content={getHelpText('experiment', 'description')} />
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

        <h3 className="text-xl font-medium text-gray-800 mt-6 mb-3">
          Parameter Sweep
          <HelpTooltip content={getHelpText('experiment', 'parameter_sweep')} />
        </h3>
        <p className="text-gray-600 text-sm mb-4">
          Define parameters for sweeping. Total combinations: <span className="font-semibold">{calculateCombinations}</span>
        </p>

        {/* Dynamic Parameter Sweep Inputs */}
        <div className="border border-gray-200 rounded-md p-4 space-y-3">
          {(config.parameters || []).map((param, index) => (
            <div key={param.name} className="flex flex-col border p-3 rounded-md">
              <div className="flex justify-between items-center mb-2">
                <label className="block text-sm font-medium text-gray-700">
                  {param.name.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())}
                </label>
                <button
                  onClick={() => removeParameter(index)}
                  className="text-red-600 hover:text-red-800 text-sm"
                >
                  Remove
                </button>
              </div>
              <div className="mb-2">
                <label htmlFor={`param-type-${param.name}`} className="block text-xs font-medium text-gray-600">
                  Parameter Type
                </label>
                <select
                  id={`param-type-${param.name}`}
                  value={param.type}
                  onChange={(e) => handleParameterChange(index, 'type', e.target.value as 'discrete' | 'range')}
                  className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
                >
                  <option value="discrete">Discrete Values (comma-separated)</option>
                  <option value="range">Numeric Range (min:step:max)</option>
                </select>
              </div>
              {param.type === 'discrete' && (
                <div>
                  <label htmlFor={`sweep-${param.name}`} className="block text-xs font-medium text-gray-600">
                    Values (comma-separated)
                  </label>
                  <textarea
                    name={`sweep-${param.name}`}
                    id={`sweep-${param.name}`}
                    value={param.values?.join(', ') || ''}
                    onChange={(e) => handleParameterChange(index, 'values', e.target.value)}
                    rows={1}
                    className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
                    placeholder="e.g., val1, val2, val3"
                  />
                </div>
              )}
              {param.type === 'range' && (
                <div className="grid grid-cols-3 gap-3">
                  <div>
                    <label htmlFor={`min-${param.name}`} className="block text-xs font-medium text-gray-600">Min</label>
                    <input
                      type="number"
                      id={`min-${param.name}`}
                      value={param.min ?? ''}
                      onChange={(e) => handleParameterChange(index, 'min', parseFloat(e.target.value))}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
                      step="any"
                    />
                  </div>
                  <div>
                    <label htmlFor={`step-${param.name}`} className="block text-xs font-medium text-gray-600">Step</label>
                    <input
                      type="number"
                      id={`step-${param.name}`}
                      value={param.step ?? ''}
                      onChange={(e) => handleParameterChange(index, 'step', parseFloat(e.target.value))}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
                      step="any"
                    />
                  </div>
                  <div>
                    <label htmlFor={`max-${param.name}`} className="block text-xs font-medium text-gray-600">Max</label>
                    <input
                      type="number"
                      id={`max-${param.name}`}
                      value={param.max ?? ''}
                      onChange={(e) => handleParameterChange(index, 'max', parseFloat(e.target.value))}
                      className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
                      step="any"
                    />
                  </div>
                </div>
              )}
            </div>
          ))}
          <div>
            <label htmlFor="newParam" className="block text-sm font-medium text-gray-700">
              Add New Parameter (e.g., `new_parameter_name`)
            </label>
            <input
              type="text"
              id="newParam"
              value={newParamName}
              onChange={(e) => setNewParamName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && newParamName.trim()) {
                  addParameter(newParamName.trim());
                }
              }}
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
              placeholder="Enter new parameter name and press Enter"
            />
          </div>
        </div>

        <h3 className="text-xl font-medium text-gray-800 mt-6 mb-3">Output Settings</h3>
        <div className="space-y-2">
          <div>
            <label htmlFor="batchSize" className="block text-sm font-medium text-gray-700">
              Batch Size
              <HelpTooltip content={getHelpText('experiment', 'parallel_workers')} />
            </label>
            <input
              type="number"
              name="batchSize"
              id="batchSize"
              value={config.batchSize || ''}
              onChange={handleChange}
              min="1"
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
            />
          </div>
          <div>
            <label htmlFor="maxRuns" className="block text-sm font-medium text-gray-700">
              Max Runs (Optional, for testing)
              <HelpTooltip content={getHelpText('experiment', 'max_runs')} />
            </label>
            <input
              type="number"
              name="iterations"
              id="maxRuns"
              value={config.iterations || ''}
              onChange={handleChange}
              min="1"
              className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3"
            />
          </div>
          {/* Add more output configuration fields as needed based on sweep.yaml */}
          <div className="flex items-center">
            <input
              type="checkbox"
              name="saveEvents"
              id="saveEvents"
              checked={config.outputConfig?.saveEvents || false}
              onChange={(e) => onConfigChange({
                ...config,
                outputConfig: {
                  ...(config.outputConfig || {}),
                  saveEvents: e.target.checked
                }
              })}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded"
            />
            <label htmlFor="saveEvents" className="ml-2 block text-sm text-gray-700">
              Save Full Event Stream
              <HelpTooltip content={getHelpText('experiment', 'save_events')} />
            </label>
          </div>
          <div className="flex items-center">
            <input
              type="checkbox"
              name="saveSnapshots"
              id="saveSnapshots"
              checked={config.outputConfig?.saveSnapshots || false}
              onChange={(e) => onConfigChange({
                ...config,
                outputConfig: {
                  ...(config.outputConfig || {}),
                  saveSnapshots: e.target.checked
                }
              })}
              className="h-4 w-4 text-blue-600 border-gray-300 rounded"
            />
            <label htmlFor="saveSnapshots" className="ml-2 block text-sm text-gray-700">
              Save Periodic Snapshots
              <HelpTooltip content={getHelpText('experiment', 'save_snapshots')} />
            </label>
          </div>
          {config.outputConfig?.saveSnapshots && (
            <div>
              <label htmlFor="snapshotInterval" className="block text-sm font-medium text-gray-700">
                Snapshot Interval (hours)
                <HelpTooltip content={getHelpText('experiment', 'snapshot_interval_hours')} />
              </label>
              <input
                type="number"
                name="snapshotIntervalHours"
                id="snapshotInterval"
                value={config.outputConfig?.snapshotIntervalHours || ''}
                onChange={(e) => onConfigChange({
                  ...config,
                  outputConfig: {
                    ...(config.outputConfig || {}),
                    snapshotIntervalHours: parseFloat(e.target.value)
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
