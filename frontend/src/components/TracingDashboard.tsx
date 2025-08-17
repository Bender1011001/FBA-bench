import React, { useState, useEffect } from 'react';
import AgentReasoningViewer from './observability/AgentInsightsDashboard';
import ErrorPatternChart from './observability/ToolUsageAnalyzer';
import PerformanceHeatmap from './observability/SimulationHealthMonitor';
import InteractiveTraceViewer from './TraceViewer';

interface TraceStatus {
  enabled: boolean;
  collectorConnected: boolean;
  fallbackMode: boolean;
  traceCount: number;
}

const TracingDashboard: React.FC = () => {
  const [traceStatus, setTraceStatus] = useState<TraceStatus>({
    enabled: false,
    collectorConnected: false,
    fallbackMode: false,
    traceCount: 0
  });
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check tracing status
    const checkTracingStatus = async () => {
      try {
        const response = await fetch('/api/v1/tracing/status');
        const data = await response.json();
        setTraceStatus(data);
      } catch (error) {
        console.error('Failed to fetch tracing status:', error);
        setTraceStatus({
          enabled: false,
          collectorConnected: false,
          fallbackMode: false,
          traceCount: 0
        });
      } finally {
        setLoading(false);
      }
    };

    checkTracingStatus();
    
    // Set up periodic status checks
    const interval = setInterval(checkTracingStatus, 30000); // Check every 30 seconds
    
    return () => clearInterval(interval);
  }, []);

  const getStatusMessage = () => {
    if (!traceStatus.enabled) {
      return {
        type: 'warning',
        message: 'OpenTelemetry tracing is currently disabled. You can enable it by setting OTEL_ENABLED=true in your environment.',
        action: 'Enable Tracing'
      };
    }
    
    if (!traceStatus.collectorConnected && !traceStatus.fallbackMode) {
      return {
        type: 'error',
        message: 'OpenTelemetry tracing is enabled but cannot connect to the collector. Traces will not be collected.',
        action: 'Configure Collector'
      };
    }
    
    if (traceStatus.fallbackMode) {
      return {
        type: 'info',
        message: 'OpenTelemetry tracing is running in fallback mode. Traces are being logged to console instead of the collector.',
        action: 'Configure Collector'
      };
    }
    
    if (traceStatus.traceCount === 0) {
      return {
        type: 'info',
        message: 'OpenTelemetry tracing is active and connected to the collector, but no traces have been collected yet.',
        action: 'Start Simulation'
      };
    }
    
    return {
      type: 'success',
      message: `OpenTelemetry tracing is active with ${traceStatus.traceCount} traces collected.`,
      action: null
    };
  };

  const status = getStatusMessage();

  const getStatusColor = () => {
    switch (status.type) {
      case 'error': return 'bg-red-50 border-red-200 text-red-800';
      case 'warning': return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      case 'success': return 'bg-green-50 border-green-200 text-green-800';
      default: return 'bg-blue-50 border-blue-200 text-blue-800';
    }
  };

  if (loading) {
    return (
      <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
        <div className="flex items-center justify-center h-64">
          <div className="text-center">
            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
            <p className="text-gray-600">Loading tracing dashboard...</p>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h2 className="text-xl font-semibold text-gray-900 mb-4">OpenTelemetry Tracing Dashboard</h2>
      
      <div className={`p-4 border rounded-md mb-6 ${getStatusColor()}`}>
        <p>
          <strong>{status.type.charAt(0).toUpperCase() + status.type.slice(1)}:</strong> {status.message}
        </p>
        {status.action && (
          <div className="mt-2">
            <button
              className="px-3 py-1 text-sm font-medium rounded-md bg-white border border-current opacity-75 hover:opacity-100"
              onClick={() => {
                // Handle action based on type
                if (status.action === 'Enable Tracing') {
                  // Instructions to enable tracing
                  alert('To enable tracing, set OTEL_ENABLED=true in your environment variables and restart the application.');
                } else if (status.action === 'Configure Collector') {
                  // Instructions to configure collector
                  alert('To configure the OTLP collector, see the documentation at docs/configuration/opentelemetry.md');
                }
              }}
            >
              {status.action}
            </button>
          </div>
        )}
      </div>

      {traceStatus.enabled && (
        <>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mt-4">
            <div className="col-span-1">
              <h3 className="text-lg font-semibold mb-2">Agent Reasoning Visualization</h3>
              <AgentReasoningViewer />
            </div>
            <div className="col-span-1">
              <h3 className="text-lg font-semibold mb-2">Error Pattern Analysis</h3>
              <ErrorPatternChart />
            </div>
            <div className="col-span-1">
              <h3 className="text-lg font-semibold mb-2">Performance Heatmap</h3>
              <PerformanceHeatmap />
            </div>
            <div className="col-span-1">
              <h3 className="text-lg font-semibold mb-2">Interactive Trace Exploration</h3>
              <InteractiveTraceViewer traceData={null} />
            </div>
          </div>

          <div className="mt-6 p-4 border rounded-md bg-gray-50 text-gray-700">
            <h4 className="font-semibold mb-2">Tracing Configuration</h4>
            <div className="grid grid-cols-2 gap-4 text-sm">
              <div>
                <strong>Status:</strong> {traceStatus.enabled ? 'Enabled' : 'Disabled'}
              </div>
              <div>
                <strong>Collector:</strong> {traceStatus.collectorConnected ? 'Connected' : 'Disconnected'}
              </div>
              <div>
                <strong>Mode:</strong> {traceStatus.fallbackMode ? 'Console Fallback' : 'OTLP Export'}
              </div>
              <div>
                <strong>Traces Collected:</strong> {traceStatus.traceCount}
              </div>
            </div>
          </div>
        </>
      )}

      {!traceStatus.enabled && (
        <div className="mt-6 p-4 border rounded-md bg-gray-50 text-gray-700">
          <h4 className="font-semibold mb-2">Getting Started with Tracing</h4>
          <ul className="list-disc list-inside space-y-1 text-sm">
            <li>Enable tracing by setting OTEL_ENABLED=true in your environment</li>
            <li>Configure an OTLP collector endpoint with OTEL_EXPORTER_OTLP_ENDPOINT</li>
            <li>Or use the console fallback for development</li>
            <li>See documentation for detailed setup instructions</li>
          </ul>
        </div>
      )}
    </div>
  );
};

export default TracingDashboard;