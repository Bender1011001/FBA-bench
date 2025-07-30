import React from 'react';
import type { SimulationSettings } from '../types';

interface SimulationConfigFormProps {
  config: SimulationSettings;
  onConfigChange: (newConfig: SimulationSettings) => void;
}

export function SimulationConfigForm({ config, onConfigChange }: SimulationConfigFormProps) {
  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    const { name, value, type } = e.target;
    onConfigChange({
      ...config,
      [name]: type === 'number' ? parseFloat(value) : (type === 'checkbox' ? (e.target as HTMLInputElement).checked : value),
    });
  };

  return (
    <div>
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">
        Basic Simulation Parameters
      </h2>
      <div className="space-y-4">
        <div>
          <label htmlFor="simulationName" className="block text-sm font-medium text-gray-700">
            Simulation Name
          </label>
          <input
            type="text"
            name="simulationName"
            id="simulationName"
            value={config.simulationName || ''}
            onChange={handleChange}
            placeholder="e.g., MyCustomSimulation"
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="description" className="block text-sm font-medium text-gray-700">
            Description
          </label>
          <textarea
            name="description"
            id="description"
            value={config.description || ''}
            onChange={handleChange}
            rows={3}
            placeholder="A brief description of this simulation"
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          ></textarea>
        </div>
        <div>
          <label htmlFor="duration" className="block text-sm font-medium text-gray-700">
            Duration (ticks)
          </label>
          <input
            type="number"
            name="duration"
            id="duration"
            value={config.duration || ''}
            onChange={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="randomSeed" className="block text-sm font-medium text-gray-700">
            Random Seed
          </label>
          <input
            type="number"
            name="randomSeed"
            id="randomSeed"
            value={config.randomSeed || (config.randomSeed === 0 ? 0 : '')}
            onChange={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="metricsInterval" className="block text-sm font-medium text-gray-700">
            Metrics Interval (ticks)
          </label>
          <input
            type="number"
            name="metricsInterval"
            id="metricsInterval"
            value={config.metricsInterval || (config.metricsInterval === 0 ? 0 : '')}
            onChange={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
        <div>
          <label htmlFor="snapshotInterval" className="block text-sm font-medium text-gray-700">
            Snapshot Interval (ticks)
          </label>
          <input
            type="number"
            name="snapshotInterval"
            id="snapshotInterval"
            value={config.snapshotInterval || (config.snapshotInterval === 0 ? 0 : '')}
            onChange={handleChange}
            className="mt-1 block w-full border border-gray-300 rounded-md shadow-sm py-2 px-3 focus:outline-none focus:ring-blue-500 focus:border-blue-500"
          />
        </div>
      </div>
    </div>
  );
}
