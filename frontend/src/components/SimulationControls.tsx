import React, { useState } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { useWebSocket } from '../hooks/useWebSocket';
import { apiService } from '../services/apiService';
import { sanitizePositiveNumber, sanitizeExternalData } from '../utils/sanitization';

interface SimulationControlsProps {
  className?: string;
  onSimulationStart?: () => void;
  onSimulationStop?: () => void;
  onSimulationPause?: () => void;
  onSimulationReset?: () => void;
}

export const SimulationControls: React.FC<SimulationControlsProps> = React.memo(({
  className = '',
  onSimulationStart,
  onSimulationStop,
  onSimulationPause,
  onSimulationReset,
}) => {
  const {
    simulation,
    setSimulationStatus,
    getCurrentMetrics,
    
    setError,
    resetSimulation,
  } = useSimulationStore();

  const { sendJsonMessage, isConnected, connectionStatus, lastError } = useWebSocket();
  const [isProcessing, setIsProcessing] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(1.0);

  // Helper function to handle API calls with error handling
  const handleApiCall = async (
    apiCall: () => Promise<unknown>,
    successMessage: string,
    onSuccess?: () => void
  ) => {
    try {
      setIsProcessing(true);
      setError(null);
      
      const result = await apiCall();
      console.log(successMessage, result);
      
      onSuccess?.();
      return result;
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Operation failed';
      console.error(errorMessage, error);
      setError(errorMessage);
      throw error;
    } finally {
      setIsProcessing(false);
    }
  };

  // Start simulation
  const handleStart = async () => {
    await handleApiCall(
      () => apiService.post('/api/v1/simulation/start'),
      'Simulation started successfully',
      () => {
        // Update local state immediately for better UX
        const metrics = getCurrentMetrics();
        setSimulationStatus({
          id: simulation.id,
          status: 'starting',
          currentTick: metrics.currentTick,
          totalTicks: metrics.totalTicks,
          simulationTime: metrics.simulationTime,
          realTime: new Date().toLocaleTimeString(),
          ticksPerSecond: 0,
          revenue: metrics.revenue,
          costs: metrics.costs,
          profit: metrics.profit,
          activeAgentCount: metrics.activeAgentCount,
        });
        
        // Send WebSocket message if connected
        if (isConnected) {
          sendJsonMessage({
            type: 'simulation_control',
            action: 'start',
            speed: simulationSpeed,
          });
        }
        
        onSimulationStart?.();
      }
    );
  };

  // Stop simulation
  const handleStop = async () => {
    await handleApiCall(
      () => apiService.post('/api/v1/simulation/stop'),
      'Simulation stopped successfully',
      () => {
        const metrics = getCurrentMetrics();
        setSimulationStatus({
          id: simulation.id,
          status: 'stopped',
          currentTick: metrics.currentTick,
          totalTicks: metrics.totalTicks,
          simulationTime: metrics.simulationTime,
          realTime: new Date().toLocaleTimeString(),
          ticksPerSecond: 0,
          revenue: metrics.revenue,
          costs: metrics.costs,
          profit: metrics.profit,
          activeAgentCount: metrics.activeAgentCount,
        });
        
        if (isConnected) {
          sendJsonMessage({
            type: 'simulation_control',
            action: 'stop',
          });
        }
        
        onSimulationStop?.();
      }
    );
  };

  // Pause simulation
  const handlePause = async () => {
    await handleApiCall(
      () => apiService.post('/api/v1/simulation/pause'),
      'Simulation paused successfully',
      () => {
        const metrics = getCurrentMetrics();
        setSimulationStatus({
          id: simulation.id,
          status: 'paused',
          currentTick: metrics.currentTick,
          totalTicks: metrics.totalTicks,
          simulationTime: metrics.simulationTime,
          realTime: new Date().toLocaleTimeString(),
          ticksPerSecond: 0,
          revenue: metrics.revenue,
          costs: metrics.costs,
          profit: metrics.profit,
          activeAgentCount: metrics.activeAgentCount,
        });
        
        if (isConnected) {
          sendJsonMessage({
            type: 'simulation_control',
            action: 'pause',
          });
        }
        
        onSimulationPause?.();
      }
    );
  };

  // Resume simulation
  const handleResume = async () => {
    await handleApiCall(
      () => apiService.post('/api/v1/simulation/resume'),
      'Simulation resumed successfully',
      () => {
        const metrics = getCurrentMetrics();
        setSimulationStatus({
          id: simulation.id,
          status: 'running',
          currentTick: metrics.currentTick,
          totalTicks: metrics.totalTicks,
          simulationTime: metrics.simulationTime,
          realTime: new Date().toLocaleTimeString(),
          ticksPerSecond: simulation.ticksPerSecond,
          revenue: metrics.revenue,
          costs: metrics.costs,
          profit: metrics.profit,
          activeAgentCount: metrics.activeAgentCount,
        });
        
        if (isConnected) {
          sendJsonMessage({
            type: 'simulation_control',
            action: 'resume',
            speed: simulationSpeed,
          });
        }
      }
    );
  };

  // Reset simulation
  const handleReset = async () => {
    if (!confirm('Are you sure you want to reset the simulation? All progress will be lost.')) {
      return;
    }

    await handleApiCall(
      () => apiService.post('/api/v1/simulation/reset'),
      'Simulation reset successfully',
      () => {
        resetSimulation();
        
        if (isConnected) {
          sendJsonMessage({
            type: 'simulation_control',
            action: 'reset',
          });
        }
        
        onSimulationReset?.();
      }
    );
  };

  // Emergency stop
  const handleEmergencyStop = async () => {
    await handleApiCall(
      () => apiService.post('/api/v1/simulation/emergency-stop'),
      'Emergency stop executed',
      () => {
        const metrics = getCurrentMetrics();
        setSimulationStatus({
          id: simulation.id,
          status: 'stopped',
          currentTick: metrics.currentTick,
          totalTicks: metrics.totalTicks,
          simulationTime: metrics.simulationTime,
          realTime: new Date().toLocaleTimeString(),
          ticksPerSecond: 0,
          revenue: metrics.revenue,
          costs: metrics.costs,
          profit: metrics.profit,
          activeAgentCount: metrics.activeAgentCount,
        });
        
        if (isConnected) {
          sendJsonMessage({
            type: 'simulation_control',
            action: 'emergency_stop',
          });
        }
      }
    );
  };

  // Update simulation speed
  const handleSpeedChange = async (newSpeed: number) => {
    // Sanitize the speed value to ensure it's a positive number
    const sanitizedSpeed = sanitizePositiveNumber(newSpeed, 1.0);
    setSimulationSpeed(sanitizedSpeed);
    
    if (simulation.status === 'running' && isConnected) {
      sendJsonMessage({
        type: 'simulation_control',
        action: 'set_speed',
        speed: sanitizedSpeed,
      });
    }
    
    // Also update via API if needed
    try {
      await apiService.post('/api/v1/simulation/speed', { speed: sanitizedSpeed });
    } catch (error) {
      console.warn('Failed to update simulation speed via API:', error);
    }
  };

  // Determine if controls should be disabled
  const isDisabled = isProcessing || simulation.isLoading;
  const canStart = simulation.status === 'idle' || simulation.status === 'stopped';
  const canPause = simulation.status === 'running';
  const canResume = simulation.status === 'paused';
  const canStop = simulation.status === 'running' || simulation.status === 'paused';

  return (
    <div className={`bg-white p-6 rounded-lg shadow-sm border border-gray-200 ${className}`}>
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Simulation Controls</h3>
        <div className="flex items-center space-x-2 text-sm">
          <span className="text-gray-500">Status:</span>
          <span className={`font-medium ${
            simulation.status === 'running' ? 'text-green-600' :
            simulation.status === 'paused' ? 'text-yellow-600' :
            simulation.status === 'error' ? 'text-red-600' :
            'text-gray-600'
          }`}>
            {simulation.status?.toUpperCase() || 'UNKNOWN'}
          </span>
        </div>
      </div>

      {/* Primary Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Start/Resume Button */}
        {canStart && (
          <button
            onClick={handleStart}
            disabled={isDisabled}
            className="flex items-center justify-center px-4 py-3 bg-green-600 text-white rounded-lg hover:bg-green-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
            Start Simulation
          </button>
        )}

        {canResume && (
          <button
            onClick={handleResume}
            disabled={isDisabled}
            className="flex items-center justify-center px-4 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
            </svg>
            Resume
          </button>
        )}

        {/* Pause Button */}
        {canPause && (
          <button
            onClick={handlePause}
            disabled={isDisabled}
            className="flex items-center justify-center px-4 py-3 bg-yellow-600 text-white rounded-lg hover:bg-yellow-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            Pause
          </button>
        )}

        {/* Stop Button */}
        {canStop && (
          <button
            onClick={handleStop}
            disabled={isDisabled}
            className="flex items-center justify-center px-4 py-3 bg-red-600 text-white rounded-lg hover:bg-red-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <svg className="w-5 h-5 mr-2" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM8 7a1 1 0 00-1 1v4a1 1 0 001 1h4a1 1 0 001-1V8a1 1 0 00-1-1H8z" clipRule="evenodd" />
            </svg>
            Stop
          </button>
        )}
      </div>

      {/* Speed Control */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 mb-2">
          Simulation Speed: {simulationSpeed}x
        </label>
        <div className="flex items-center space-x-4">
          <span className="text-xs text-gray-500">0.1x</span>
          <input
            type="range"
            min="0.1"
            max="5.0"
            step="0.1"
            value={simulationSpeed}
            onChange={(e) => handleSpeedChange(parseFloat(e.target.value || '1.0'))}
            disabled={isDisabled}
            className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer disabled:cursor-not-allowed"
          />
          <span className="text-xs text-gray-500">5.0x</span>
        </div>
        <div className="flex justify-between mt-2">
          {[0.5, 1.0, 2.0, 3.0].map((speed) => (
            <button
              key={speed}
              onClick={() => handleSpeedChange(speed)}
              disabled={isDisabled}
              className={`px-3 py-1 text-xs rounded transition-colors ${
                simulationSpeed === speed
                  ? 'bg-blue-100 text-blue-700 border border-blue-300'
                  : 'bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:opacity-50'
              }`}
            >
              {speed}x
            </button>
          ))}
        </div>
      </div>

      {/* Secondary Controls */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 border-t pt-4">
        <button
          onClick={handleReset}
          disabled={isDisabled}
          className="flex items-center justify-center px-4 py-2 bg-gray-600 text-white rounded-lg hover:bg-gray-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
          </svg>
          Reset
        </button>

        <button
          onClick={handleEmergencyStop}
          disabled={isDisabled || simulation.status === 'idle' || simulation.status === 'stopped'}
          className="flex items-center justify-center px-4 py-2 bg-red-700 text-white rounded-lg hover:bg-red-800 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
        >
          <svg className="w-4 h-4 mr-2" fill="currentColor" viewBox="0 0 20 20">
            <path fillRule="evenodd" d="M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 018.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z" clipRule="evenodd" />
          </svg>
          Emergency Stop
        </button>
      </div>

      {/* Connection Status */}
      <div className="mt-4 flex items-center justify-between text-sm">
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${
            connectionStatus === 'connected' ? 'bg-green-500' :
            connectionStatus === 'connecting' ? 'bg-yellow-500' :
            connectionStatus === 'error' ? 'bg-red-500' : 'bg-gray-500'
          }`} />
          <span className="text-gray-600">
            WebSocket: {connectionStatus === 'connected' ? 'Connected' :
                      connectionStatus === 'connecting' ? 'Connecting...' :
                      connectionStatus === 'error' ? 'Error' : 'Disconnected'}
          </span>
        </div>
        
        {isProcessing && (
          <div className="flex items-center space-x-2 text-blue-600">
            <svg className="animate-spin w-4 h-4" fill="none" viewBox="0 0 24 24">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
            </svg>
            <span>Processing...</span>
          </div>
        )}
      </div>

      {/* Error Display */}
      {(simulation.error || lastError) && (
        <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
          <div className="flex items-center space-x-2">
            <svg className="w-4 h-4 text-red-500" fill="currentColor" viewBox="0 0 20 20">
              <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
            </svg>
            <span className="text-red-700 text-sm font-medium">Control Error</span>
          </div>
          <p className="text-red-600 text-sm mt-1">{sanitizeExternalData(simulation.error || lastError)}</p>
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function to prevent unnecessary re-renders
  return (
    prevProps.className === nextProps.className &&
    prevProps.onSimulationStart === nextProps.onSimulationStart &&
    prevProps.onSimulationStop === nextProps.onSimulationStop &&
    prevProps.onSimulationPause === nextProps.onSimulationPause &&
    prevProps.onSimulationReset === nextProps.onSimulationReset
  );
});

export default SimulationControls;