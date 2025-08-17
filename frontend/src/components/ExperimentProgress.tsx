import React, { useState, useEffect, useCallback } from 'react';
import type { ExperimentExecution, ExperimentStatus, ExperimentProgressUpdate } from '../types'; // Import ExperimentProgressUpdate
import useWebSocket from '../hooks/useWebSocket';
import { apiService } from '../services/apiService';

interface ExperimentProgressProps {
  experiment: ExperimentExecution;
  onExperimentUpdate: (updatedExperiment: ExperimentExecution) => void;
  onExperimentEnded: (experimentId: string, status: ExperimentStatus) => void;
}

export function ExperimentProgress({ experiment, onExperimentUpdate, onExperimentEnded }: ExperimentProgressProps) {
  const [currentProgress, setCurrentProgress] = useState<ExperimentExecution>(experiment); // Explicitly type useState
  const [actionLoading, setActionLoading] = useState(false);
  const [actionError, setActionError] = useState<string | null>(null);

  const wsUrl = `ws://localhost:8000/ws/experiment/${experiment.id}/progress`; // Assuming this endpoint

  const handleWsMessage = useCallback((message: ExperimentProgressUpdate) => { // Use ExperimentProgressUpdate type
    // Update currentProgress state with new data from WebSocket
    const updatedExp: ExperimentExecution = {
        ...currentProgress,
        ...message, // Apply partial updates from WebSocket
        lastUpdated: new Date().toISOString()
    };
    setCurrentProgress(updatedExp);
    onExperimentUpdate(updatedExp);

    if (updatedExp.status === 'completed' || updatedExp.status === 'failed' || updatedExp.status === 'cancelled') {
        onExperimentEnded(updatedExp.id, updatedExp.status);
    }
  }, [currentProgress, experiment.id, onExperimentUpdate, onExperimentEnded]);


  const { sendMessage, isConnected, disconnect } = useWebSocket({
    url: wsUrl,
    autoConnect: true,
    onMessage: handleWsMessage,
    onDisconnect: () => console.log(`WebSocket for experiment ${experiment.id} disconnected.`),
    onError: (e) => console.error(`WebSocket error for experiment ${experiment.id}:`, e),
  });

  useEffect(() => {
    // Keep internal state in sync with prop changes (e.g., from initial run to queued)
    setCurrentProgress(experiment);
  }, [experiment]);

  const sendExperimentCommand = useCallback(async (command: 'pause' | 'resume' | 'cancel') => {
    setActionLoading(true);
    setActionError(null);
    try {
      await apiService.post(`/api/experiment/${experiment.id}/${command}`);
      // Optimistically update status if successful based on the command
      let newStatus: ExperimentStatus;
      if (command === 'pause') newStatus = 'paused';
      else if (command === 'resume') newStatus = 'running';
      else if (command === 'cancel') newStatus = 'cancelled';
      else newStatus = currentProgress.status; // Should not happen with defined commands

      onExperimentUpdate({ ...currentProgress, status: newStatus });
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'An unknown error occurred.');
    } finally {
      setActionLoading(false);
    }
  }, [experiment.id, currentProgress, onExperimentUpdate]);

  const renderStatusBadge = (status: ExperimentStatus) => {
    let colorClass = '';
    switch (status) {
      case 'queued':
        colorClass = 'bg-gray-200 text-gray-800';
        break;
      case 'running':
        colorClass = 'bg-blue-200 text-blue-800';
        break;
      case 'completed':
        colorClass = 'bg-green-200 text-green-800';
        break;
      case 'failed':
        colorClass = 'bg-red-200 text-red-800';
        break;
      case 'cancelled':
        colorClass = 'bg-yellow-200 text-yellow-800';
        break;
      default:
        colorClass = 'bg-gray-200 text-gray-800';
    }
    return <span className={`inline-flex items-center px-3 py-0.5 rounded-full text-sm font-medium ${colorClass}`}>{status}</span>;
  };

  const calculateProgressBarWidth = (progress: number | undefined) => {
    if (progress === undefined) return '0%';
    return `${Math.min(100, Math.max(0, progress))}%`;
  };

  return (
    <div className="bg-white shadow overflow-hidden sm:rounded-lg mb-6">
      <div className="px-4 py-5 sm:px-6">
        <h3 className="text-lg leading-6 font-medium text-gray-900">
          Experiment Progress: {currentProgress.experimentName}
        </h3>
        <p className="mt-1 max-w-2xl text-sm text-gray-500">
          ID: {currentProgress.id}
        </p>
      </div>
      <div className="border-t border-gray-200 px-4 py-5 sm:p-0">
        <dl className="sm:divide-y sm:divide-gray-200">
          <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="text-sm font-medium text-gray-500">Status</dt>
            <dd className="mt-1 flex text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              {renderStatusBadge(currentProgress.status)}
              {!isConnected && <span className="ml-2 text-red-500"> (Disconnected)</span>}
            </dd>
          </div>
          <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="text-sm font-medium text-gray-500">Progress</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              <div className="w-full bg-gray-200 rounded-full h-2.5 dark:bg-gray-700">
                <div 
                  className="bg-blue-600 h-2.5 rounded-full" 
                  style={{ width: calculateProgressBarWidth(currentProgress.progress) }}
                ></div>
              </div>
              <span className="mt-1 block text-right">{currentProgress.progress !== undefined ? `${currentProgress.progress.toFixed(2)}%` : 'N/A'}</span>
            </dd>
          </div>
          <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="text-sm font-medium text-gray-500">Start Time</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              {currentProgress.startTime ? new Date(currentProgress.startTime).toLocaleString() : 'N/A'}
            </dd>
          </div>
          <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="text-sm font-medium text-gray-500">Estimated Completion</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              {currentProgress.estimatedCompletionTime ? new Date(currentProgress.estimatedCompletionTime).toLocaleString() : 'N/A'}
            </dd>
          </div>
          <div className="py-4 sm:py-5 sm:grid sm:grid-cols-3 sm:gap-4 sm:px-6">
            <dt className="text-sm font-medium text-gray-500">Last Updated</dt>
            <dd className="mt-1 text-sm text-gray-900 sm:mt-0 sm:col-span-2">
              {currentProgress.lastUpdated ? new Date(currentProgress.lastUpdated).toLocaleString() : 'N/A'}
            </dd>
          </div>
        </dl>
      </div>
      <div className="px-4 py-3 bg-gray-50 text-right sm:px-6 flex justify-end space-x-3">
        {currentProgress.status === 'running' && (
          <button
            onClick={() => sendExperimentCommand('pause')}
            disabled={actionLoading}
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-yellow-600 hover:bg-yellow-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-yellow-500"
          >
            {actionLoading ? 'Pausing...' : 'Pause'}
          </button>
        )}
        {(currentProgress.status === 'queued' || currentProgress.status === 'paused') && (
          <button
            onClick={() => sendExperimentCommand('resume')}
            disabled={actionLoading}
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500"
          >
            {actionLoading ? 'Resuming...' : 'Resume'}
          </button>
        )}
        {(currentProgress.status === 'queued' || currentProgress.status === 'running' || currentProgress.status === 'paused') && (
          <button
            onClick={() => sendExperimentCommand('cancel')}
            disabled={actionLoading}
            className="inline-flex justify-center py-2 px-4 border border-transparent shadow-sm text-sm font-medium rounded-md text-white bg-red-600 hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-500"
          >
            {actionLoading ? 'Cancelling...' : 'Cancel'}
          </button>
        )}
      </div>
      {actionError && (
        <div className="mt-4 text-red-600 text-sm px-4 py-2">
          Error: {actionError}
        </div>
      )}
    </div>
  );
}
