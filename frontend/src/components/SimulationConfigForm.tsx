// frontend/src/components/SimulationConfigForm.tsx

import React, { useState, useEffect } from 'react';
import type { SimulationSettings } from '../types';
import HelpTooltip from './HelpTooltip';
import { getHelpText } from '../data/helpContent';
import { validateField, validateForm, simulationValidationSchema } from '../utils/validators';

interface SimulationConfigFormProps {
  config: SimulationSettings;
  onConfigChange: (newConfig: SimulationSettings) => void;
  onValidationChange?: (isValid: boolean) => void; // Callback to notify parent of validation status
}

export function SimulationConfigForm({ config, onConfigChange, onValidationChange }: SimulationConfigFormProps) {
  const [validationErrors, setValidationErrors] = useState<Record<keyof SimulationSettings, string | undefined>>(() => {
    // Initialize validationErrors with undefined for all keys in the schema
    const initialErrors: Record<keyof SimulationSettings, string | undefined> = {};
    for (const key in simulationValidationSchema) {
      if (Object.prototype.hasOwnProperty.call(simulationValidationSchema, key)) {
        // Cast key to keyof SimulationSettings to correctly type the initialization
        initialErrors[key as keyof SimulationSettings] = undefined;
      }
    }
    return initialErrors;
  });

  // Effect to re-validate form whenever config changes
  useEffect(() => {
    // Explicitly cast config to the expected type for validateForm
    const errors = validateForm(config, simulationValidationSchema);
    setValidationErrors(errors);
    const isValid = Object.values(errors).every(error => !error);
    if (onValidationChange) {
      onValidationChange(isValid); // Notify parent component
    }
  }, [config, onValidationChange]);

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    let newValue: string | number | boolean | undefined = value; // Default to string or undefined for empty values

    // Handle number inputs: convert to float, or undefined if empty, or string if invalid number
    if (type === 'number') {
      const parsedValue = parseFloat(value);
      if (value === '') {
        newValue = undefined; // Treat empty number input as undefined for validation
      } else if (isNaN(parsedValue)) {
        newValue = value; // Keep as string if not a valid float string, to trigger type/number validation error
      } else {
        newValue = parsedValue;
      }
    } else if (type === 'checkbox') {
      newValue = (e.target as HTMLInputElement).checked;
    }

    const updatedConfig: SimulationSettings = {
      ...config,
      // Use a type assertion for the dynamic key and its value
      [name]: newValue as SimulationSettings[keyof SimulationSettings],
    };

    onConfigChange(updatedConfig);

    // Real-time field validation update
    // The schema keys match SimulationSettings keys, no remapping needed except for 'simulationName' if rules used 'name'.
    // However, simulationValidationSchema uses 'simulationName' directly, so 'name' from event.target matches.
    const rule = simulationValidationSchema[name as keyof typeof simulationValidationSchema];
    if (rule) {
      const fieldError = validateField(newValue, rule);
      setValidationErrors(prevErrors => ({
        ...prevErrors,
        [name as keyof SimulationSettings]: fieldError, // Ensure name is correctly cast
      }));
    }
  };

  const getError = (fieldName: keyof SimulationSettings) => validationErrors[fieldName];
  const hasError = (fieldName: keyof SimulationSettings) => !!getError(fieldName);

  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Basic Simulation Parameters
      </h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="simulationName" className={`block text-sm font-medium ${hasError('simulationName') ? 'text-red-600' : 'text-gray-700'}`}>
            Simulation Name *
            <HelpTooltip content={getHelpText('simulation', 'simulationName')} />
          </label>
          <input
            type="text"
            name="simulationName"
            id="simulationName"
            value={config.simulationName || ''}
            onChange={handleChange}
            onBlur={handleChange} // Validate on blur
            placeholder="e.g., MyCustomSimulation"
            className={`mt-1 block w-full border ${hasError('simulationName') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500`}
          />
          {hasError('simulationName') && (
            <p className="mt-1 text-sm text-red-600">{getError('simulationName')}</p>
          )}
        </div>
        <div>
          <label htmlFor="description" className="block text-sm font-medium text-gray-700">
            Description
            <HelpTooltip content={getHelpText('simulation', 'description')} />
          </label>
          <textarea
            name="description"
            id="description"
            value={config.description || ''}
            onChange={handleChange}
            onBlur={handleChange} // Validate on blur
            rows={3}
            placeholder="A brief description of this simulation"
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          ></textarea>
        </div>
        <div>
          <label htmlFor="duration" className={`block text-sm font-medium ${hasError('duration') ? 'text-red-600' : 'text-gray-700'}`}>
            Duration (ticks) *
            <HelpTooltip content={getHelpText('simulation', 'duration')} />
          </label>
          <input
            type="number"
            name="duration"
            id="duration"
            value={config.duration === undefined ? '' : config.duration} // Render empty string for undefined
            onChange={handleChange}
            onBlur={handleChange} // Validate on blur too
            className={`mt-1 block w-full border ${hasError('duration') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500`}
          />
          {hasError('duration') && (
            <p className="mt-1 text-sm text-red-600">{getError('duration')}</p>
          )}
        </div>
        <div>
            <label htmlFor="tickInterval" className={`block text-sm font-medium ${hasError('tickInterval') ? 'text-red-600' : 'text-gray-700'}`}>
                Tick Interval (seconds) *
                <HelpTooltip content={getHelpText('simulation', 'tickInterval')} />
            </label>
            <input
                type="number"
                name="tickInterval"
                id="tickInterval"
                value={config.tickInterval === undefined ? '' : config.tickInterval}
                onChange={handleChange}
                onBlur={handleChange}
                className={`mt-1 block w-full border ${hasError('tickInterval') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500`}
            />
            {hasError('tickInterval') && (
                <p className="mt-1 text-sm text-red-600">{getError('tickInterval')}</p>
            )}
        </div>
        <div>
          <label htmlFor="initialPrice" className={`block text-sm font-medium ${hasError('initialPrice') ? 'text-red-600' : 'text-gray-700'}`}>
            Initial Price *
            <HelpTooltip content={getHelpText('simulation', 'initialPrice')} />
          </label>
          <input
            type="number"
            name="initialPrice"
            id="initialPrice"
            value={config.initialPrice === undefined ? '' : config.initialPrice}
            onChange={handleChange}
            onBlur={handleChange}
            step="0.01"
            className={`mt-1 block w-full border ${hasError('initialPrice') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500`}
          />
          {hasError('initialPrice') && (
            <p className="mt-1 text-sm text-red-600">{getError('initialPrice')}</p>
          )}
        </div>
        <div>
          <label htmlFor="inventory" className={`block text-sm font-medium ${hasError('inventory') ? 'text-red-600' : 'text-gray-700'}`}>
            Initial Inventory *
            <HelpTooltip content={getHelpText('simulation', 'inventory')} />
          </label>
          <input
            type="number"
            name="inventory"
            id="inventory"
            value={config.inventory === undefined ? '' : config.inventory}
            onChange={handleChange}
            onBlur={handleChange}
            step="1"
            className={`mt-1 block w-full border ${hasError('inventory') ? 'border-red-500' : 'border-gray-300'} rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500`}
          />
          {hasError('inventory') && (
            <p className="mt-1 text-sm text-red-600">{getError('inventory')}</p>
          )}
        </div>
        <div>
          <label htmlFor="randomSeed" className="block text-sm font-medium text-gray-700">
            Random Seed
            <HelpTooltip content={getHelpText('simulation', 'randomSeed')} />
          </label>
          <input
            type="number"
            name="randomSeed"
            id="randomSeed"
            value={config.randomSeed === undefined ? '' : config.randomSeed}
            onChange={handleChange}
            onBlur={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="metricsInterval" className="block text-sm font-medium text-gray-700">
            Metrics Interval (ticks)
            <HelpTooltip content={getHelpText('simulation', 'metricsInterval')} />
          </label>
          <input
            type="number"
            name="metricsInterval"
            id="metricsInterval"
            value={config.metricsInterval === undefined ? '' : config.metricsInterval}
            onChange={handleChange}
            onBlur={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="snapshotInterval" className="block text-sm font-medium text-gray-700">
            Snapshot Interval (ticks)
            <HelpTooltip content={getHelpText('simulation', 'snapshotInterval')} />
          </label>
          <input
            type="number"
            name="snapshotInterval"
            id="snapshotInterval"
            value={config.snapshotInterval === undefined ? '' : config.snapshotInterval}
            onChange={handleChange}
            onBlur={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>
    </div>
  );
}
