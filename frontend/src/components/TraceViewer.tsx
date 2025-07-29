import React from 'react';

interface ChromeTraceEvent {
  name: string;
  cat?: string;
  ph: string; // Phase type, e.g., "X" for complete event
  ts: number; // Timestamp in microseconds
  dur?: number; // Duration in microseconds for complete events
  pid: number; // Process ID
  tid: number; // Thread ID
  args?: { [key: string]: unknown }; // Event arguments/attributes
}

interface ChromeTraceFormat {
  traceEvents: ChromeTraceEvent[];
  displayTimeUnit?: string;
}

interface TraceViewerProps {
  traceData: ChromeTraceFormat | null;
}

const TraceViewer: React.FC<TraceViewerProps> = ({ traceData }) => {
  if (!traceData || !traceData.traceEvents || traceData.traceEvents.length === 0) {
    return <div className="p-4 text-gray-600">No trace data available.</div>;
  }

  return (
    <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
      <h3 className="text-lg font-semibold text-gray-900 mb-4">Trace Details</h3>
      <div className="space-y-3">
        {traceData.traceEvents.map((event: ChromeTraceEvent, index: number) => (
          <div key={index} className="border-b border-gray-100 pb-3 last:border-b-0 last:pb-0">
            <div className="font-medium text-gray-800">{event.name}</div>
            {event.cat && <div className="text-sm text-gray-600">Category: {event.cat}</div>}
            <div className="text-sm text-gray-600">Duration: {event.dur} µs</div>
            <div className="text-sm text-gray-600">Time: {event.ts} µs</div>
            <div className="text-sm text-gray-600">Process ID: {event.pid}, Thread ID: {event.tid}</div>
            {event.args && Object.keys(event.args).length > 0 && (
              <div className="bg-gray-50 p-2 rounded-md mt-1 text-xs">
                <h4 className="font-semibold text-gray-700 mb-1">Attributes:</h4>
                <pre className="whitespace-pre-wrap break-all text-wrap">{JSON.stringify(event.args, null, 2)}</pre>
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default TraceViewer;