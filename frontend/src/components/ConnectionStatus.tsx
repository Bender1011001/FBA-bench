import React from 'react';
import { useSimulationStore } from '../store/simulationStore';
import { useWebSocket } from '../hooks/useWebSocket';

interface ConnectionStatusProps {
  className?: string;
  showDetails?: boolean;
}

const formatTimestamp = (timestamp?: string): string => {
  if (!timestamp) return 'Never';
  
  try {
    const date = new Date(timestamp);
    const now = new Date();
    const diffMs = now.getTime() - date.getTime();
    const diffSeconds = Math.floor(diffMs / 1000);
    
    if (diffSeconds < 60) {
      return `${diffSeconds}s ago`;
    } else if (diffSeconds < 3600) {
      return `${Math.floor(diffSeconds / 60)}m ago`;
    } else {
      return date.toLocaleTimeString();
    }
  } catch {
    return 'Invalid time';
  }
};

const getConnectionStatusColor = (connected: boolean, reconnectAttempts: number) => {
  if (connected) {
    return 'text-green-600 bg-green-50 border-green-200';
  } else if (reconnectAttempts > 0) {
    return 'text-yellow-600 bg-yellow-50 border-yellow-200';
  } else {
    return 'text-red-600 bg-red-50 border-red-200';
  }
};

const getConnectionStatusIcon = (connected: boolean, reconnectAttempts: number) => {
  if (connected) {
    return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M16.707 5.293a1 1 0 010 1.414l-8 8a1 1 0 01-1.414 0l-4-4a1 1 0 011.414-1.414L8 12.586l7.293-7.293a1 1 0 011.414 0z" clipRule="evenodd" />
      </svg>
    );
  } else if (reconnectAttempts > 0) {
    return (
      <svg className="w-4 h-4 animate-spin" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M4 2a1 1 0 011 1v2.101a7.002 7.002 0 0111.601 2.566 1 1 0 11-1.885.666A5.002 5.002 0 005.999 7H9a1 1 0 010 2H4a1 1 0 01-1-1V3a1 1 0 011-1zm.008 9.057a1 1 0 011.276.61A5.002 5.002 0 0014.001 13H11a1 1 0 110-2h5a1 1 0 011 1v5a1 1 0 11-2 0v-2.101a7.002 7.002 0 01-11.601-2.566 1 1 0 01.61-1.276z" clipRule="evenodd" />
      </svg>
    );
  } else {
    return (
      <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
        <path fillRule="evenodd" d="M13.477 14.89A6 6 0 015.11 6.524l8.367 8.368zm1.414-1.414L6.524 5.11a6 6 0 018.367 8.367zM18 10a8 8 0 11-16 0 8 8 0 0116 0z" clipRule="evenodd" />
      </svg>
    );
  }
};

const getConnectionStatusText = (connected: boolean, reconnectAttempts: number) => {
  if (connected) {
    return 'Connected';
  } else if (reconnectAttempts > 0) {
    return `Reconnecting... (${reconnectAttempts})`;
  } else {
    return 'Disconnected';
  }
};

export const ConnectionStatus: React.FC<ConnectionStatusProps> = React.memo(({
  className = '',
  showDetails = true
}) => {
  const connectionStatus = useSimulationStore((state) => state.simulation.connectionStatus);
  const error = useSimulationStore((state) => state.simulation.error);
  const { reconnect } = useWebSocket({ autoConnect: false });

  const { connected, reconnectAttempts, lastHeartbeat } = connectionStatus;
  const statusColor = getConnectionStatusColor(connected, reconnectAttempts);
  const statusIcon = getConnectionStatusIcon(connected, reconnectAttempts);
  const statusText = getConnectionStatusText(connected, reconnectAttempts);

  return (
    <div className={`bg-white rounded-lg shadow-sm border border-gray-200 p-4 ${className}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <div className={`
            flex items-center space-x-2 px-3 py-1 rounded-full border text-sm font-medium
            ${statusColor}
          `}>
            {statusIcon}
            <span>{statusText}</span>
          </div>
          
          {!connected && reconnectAttempts === 0 && (
            <button
              onClick={reconnect}
              className="px-3 py-1 text-sm font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded-md hover:bg-blue-100 transition-colors"
            >
              Reconnect
            </button>
          )}
        </div>

        {showDetails && (
          <div className="text-sm text-gray-500">
            Last update: {formatTimestamp(lastHeartbeat)}
          </div>
        )}
      </div>

      {showDetails && (
        <div className="mt-3 space-y-2">
          {error && (
            <div className="flex items-center space-x-2 text-sm text-red-600 bg-red-50 px-3 py-2 rounded-md">
              <svg className="w-4 h-4 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20">
                <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
              </svg>
              <span className="flex-1">{error}</span>
            </div>
          )}

          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Status:</span>
              <span className={`ml-2 font-medium ${
                connected ? 'text-green-600' : 'text-red-600'
              }`}>
                {connected ? 'Online' : 'Offline'}
              </span>
            </div>
            
            <div>
              <span className="text-gray-500">Retry attempts:</span>
              <span className="ml-2 font-medium text-gray-900">
                {reconnectAttempts}
              </span>
            </div>
          </div>

          {lastHeartbeat && (
            <div className="text-sm">
              <span className="text-gray-500">Last heartbeat:</span>
              <span className="ml-2 font-medium text-gray-900">
                {new Date(lastHeartbeat).toLocaleString()}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function to prevent unnecessary re-renders
  return (
    prevProps.className === nextProps.className &&
    prevProps.showDetails === nextProps.showDetails
  );
});

// Compact version for header/navbar
export const ConnectionStatusCompact: React.FC<{ className?: string }> = React.memo(({
  className = ''
}) => {
  const connectionStatus = useSimulationStore((state) => state.simulation.connectionStatus);
  const { reconnect } = useWebSocket({ autoConnect: false });

  const { connected, reconnectAttempts } = connectionStatus;
  const statusColor = getConnectionStatusColor(connected, reconnectAttempts);
  const statusIcon = getConnectionStatusIcon(connected, reconnectAttempts);

  return (
    <div className={`flex items-center space-x-2 ${className}`}>
      <div className={`
        flex items-center space-x-1 px-2 py-1 rounded-full border text-xs font-medium
        ${statusColor}
      `}>
        {statusIcon}
        <span className="hidden sm:inline">
          {getConnectionStatusText(connected, reconnectAttempts)}
        </span>
      </div>
      
      {!connected && reconnectAttempts === 0 && (
        <button
          onClick={reconnect}
          className="px-2 py-1 text-xs font-medium text-blue-600 bg-blue-50 border border-blue-200 rounded hover:bg-blue-100 transition-colors whitespace-nowrap"
          title="Reconnect to server"
        >
          Reconnect
        </button>
      )}
    </div>
  );
}, (prevProps, nextProps) => {
  // Custom comparison function to prevent unnecessary re-renders
  return prevProps.className === nextProps.className;
});

export default ConnectionStatus;