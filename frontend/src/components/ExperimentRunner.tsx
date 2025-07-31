import React, { useState, useCallback, Fragment } from 'react';
import type { ExperimentConfig, ExperimentExecution, ExperimentRunResult } from '../types';
import { apiService } from '../services/apiService';
import { Dialog, Transition } from '@headlessui/react'; // For modal or simple dialog

interface ExperimentRunnerProps {
  config: ExperimentConfig;
  onExperimentRun: (experiment: ExperimentExecution) => void;
  onCancel: () => void;
}

export function ExperimentRunner({ config, onExperimentRun, onCancel }: ExperimentRunnerProps) {
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [showConfigModal, setShowConfigModal] = useState(false);

  const calculateCombinations = (expConfig: ExperimentConfig) => {
    return (expConfig.parameters || []).reduce((acc, param) => {
      let numValues = 0;
      if (param.type === 'discrete' && param.values) {
        numValues = param.values.length;
      } else if (param.type === 'range' && param.min !== undefined && param.max !== undefined && param.step !== undefined) {
        if (param.step === 0) return 0; // Avoid division by zero
        numValues = Math.floor((param.max - param.min) / param.step) + 1;
        if (numValues < 0) numValues = 0;
      }
      return acc * (numValues || 1);
    }, 1);
  };

  const handleRunExperiment = useCallback(async () => {
    setIsRunning(true);
    setError(null);
    try {
        const payload: ExperimentConfig = {
            ...config,
            // Transform parameters array to baseParameters object for backend
            baseParameters: (config.parameters || []).reduce((acc, param) => {
                if (param.type === 'discrete' && param.values) {
                    acc[param.name] = param.values;
                } else if (param.type === 'range' && param.min !== undefined && param.max !== undefined && param.step !== undefined) {
                    const rangeValues: number[] = [];
                    for (let i = param.min; i <= param.max; i += param.step) {
                        rangeValues.push(i);
                    }
                    acc[param.name] = rangeValues;
                }
                return acc;
            }, {} as Record<string, (string | number)[]>),
            // Ensure batchSize is correctly mapped from batchSize to parallelRuns if backend expects parallelRuns
            parallelRuns: config.batchSize, 
        };

        // Remove the frontend-only 'parameters' field if it exists
        delete payload.parameters;
        // Also ensure baseParameters is not undefined if no parameters were set to sweep
        if (Object.keys(payload.baseParameters || {}).length === 0) {
            payload.baseParameters = undefined;
        }

        const result = await apiService.post<ExperimentRunResult>('/api/experiment/run', payload);
        
        // Assuming the backend returns the experiment ID and a message
        const newExperiment: ExperimentExecution = {
            id: result.experimentId,
            experimentName: config.experimentName || 'Untitled Experiment',
            description: config.description,
            config: config, // Store the original config for display
            status: 'queued', // Initial status from frontend perspective
            startTime: new Date().toISOString(),
            progress: 0,
        };
        onExperimentRun(newExperiment);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setIsRunning(false);
    }
  }, [config, onExperimentRun]);

  const totalCombinations = calculateCombinations(config);

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-lg p-6">
      <h2 className="text-2xl font-semibold text-gray-800 mb-4">Run Experiment</h2>
      <div className="border-b border-gray-200 pb-4 mb-4">
        <h3 className="text-xl font-medium text-gray-700">Experiment Summary</h3>
        <dl className="mt-2 text-sm text-gray-900">
          <div className="bg-gray-50 px-4 py-2 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="font-medium text-gray-500">Name</dt>
            <dd className="mt-1 text-gray-900 sm:mt-0 sm:col-span-2">{config.experimentName || 'N/A'}</dd>
          </div>
          <div className="px-4 py-2 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="font-medium text-gray-500">Description</dt>
            <dd className="mt-1 text-gray-900 sm:mt-0 sm:col-span-2">{config.description || 'N/A'}</dd>
          </div>
          <div className="bg-gray-50 px-4 py-2 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="font-medium text-gray-500">Total Combinations</dt>
            <dd className="mt-1 text-gray-900 sm:mt-0 sm:col-span-2">{totalCombinations}</dd>
          </div>
          <div className="px-4 py-2 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="font-medium text-gray-500">Batch Size</dt>
            <dd className="mt-1 text-gray-900 sm:mt-0 sm:col-span-2">{config.batchSize || 'N/A'}</dd>
          </div>
          {/* Add more key summary items as needed */}
        </dl>
        <button
          onClick={() => setShowConfigModal(true)}
          className="mt-4 px-4 py-2 text-sm font-medium text-blue-700 bg-blue-100 rounded-md hover:bg-blue-200"
        >
          View Full Configuration
        </button>
      </div>

      <div className="flex justify-end space-x-3">
        <button
          onClick={onCancel}
          className="px-4 py-2 border border-gray-300 rounded-md text-sm font-medium text-gray-700 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500"
        >
          Cancel
        </button>
        <button
          onClick={handleRunExperiment}
          disabled={isRunning}
          className={`inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white ${
            isRunning ? 'bg-indigo-400' : 'bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500'
          }`}
        >
          {isRunning ? 'Running...' : 'Run Experiment'}
        </button>
      </div>

      {error && (
        <div className="mt-4 text-red-600 text-sm">
          Error: {error}
        </div>
      )}

      {/* Full Configuration Modal */}
      <Transition appear show={showConfigModal} as={Fragment}>
        <Dialog as="div" className="relative z-10" onClose={() => setShowConfigModal(false)}>
          <Transition.Child
            as={Fragment}
            enter="ease-out duration-300"
            enterFrom="opacity-0"
            enterTo="opacity-100"
            leave="ease-in duration-200"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <div className="fixed inset-0 bg-black bg-opacity-25" />
          </Transition.Child>

          <div className="fixed inset-0 overflow-y-auto">
            <div className="flex min-h-full items-center justify-center p-4 text-center">
              <Transition.Child
                as={Fragment}
                enter="ease-out duration-300"
                enterFrom="opacity-0 scale-95"
                enterTo="opacity-100 scale-100"
                leave="ease-in duration-200"
                leaveFrom="opacity-100 scale-100"
                leaveTo="opacity-0 scale-95"
              >
                <Dialog.Panel className="w-full max-w-2xl transform overflow-hidden rounded-2xl bg-white p-6 text-left align-middle shadow-xl transition-all">
                  <Dialog.Title as="h3" className="text-lg font-medium leading-6 text-gray-900">
                    Full Experiment Configuration
                  </Dialog.Title>
                  <div className="mt-2">
                    <pre className="text-sm bg-gray-100 p-4 rounded-md overflow-auto max-h-96">
                      {JSON.stringify(config, null, 2)}
                    </pre>
                  </div>

                  <div className="mt-4">
                    <button
                      type="button"
                      className="inline-flex justify-center rounded-md border border-transparent bg-blue-100 px-4 py-2 text-sm font-medium text-blue-900 hover:bg-blue-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2"
                      onClick={() => setShowConfigModal(false)}
                    >
                      Got it, thanks!
                    </button>
                  </div>
                </Dialog.Panel>
              </Transition.Child>
            </div>
          </div>
        </Dialog>
      </Transition>
    </div>
  );
}
