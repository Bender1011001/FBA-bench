import React, { useState } from 'react';
import type { ResultsData } from '../types';

interface DataExporterProps {
  resultsData?: ResultsData; // Optional, as some reports might be aggregated or not tied to a single ResultsData
  // onExport: (format: 'csv' | 'json' | 'pdf', data: ResultsData | 'all') => void; // Example callback
}

const DataExporter: React.FC<DataExporterProps> = ({ resultsData }) => {
  const [exportFormat, setExportFormat] = useState<'csv' | 'json' | 'pdf'>('csv');
  const [reportType, setReportType] = useState<'summary' | 'detailed' | 'executive'>('summary');
  const [isScheduled, setIsScheduled] = useState<boolean>(false);
  const [emailRecipient, setEmailRecipient] = useState<string>('');

  const handleExport = () => {
    // Placeholder for actual export logic
    console.log(`Exporting data for experiment ${resultsData?.experimentId || 'N/A'} in ${exportFormat} format, report type: ${reportType}`);
    // In a real application, this would trigger a backend API call to generate and download the file.
    alert(`Initiating export to ${exportFormat} for a ${reportType} report.`);
  };

  const handleScheduleReport = () => {
    // Placeholder for scheduling logic
    console.log(`Scheduling a ${reportType} report to be sent to ${emailRecipient || 'N/A'}`);
    alert(`Report scheduled to be generated and sent to ${emailRecipient || 'N/A'}.`);
  };

  return (
    <div className="bg-white p-6 rounded-lg shadow-md mb-6">
      <h2 className="text-xl font-semibold text-gray-800 mb-4">Data Export & Reporting</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        {/* Export Format Selection */}
        <div>
          <label htmlFor="exportFormat" className="block text-sm font-medium text-gray-700 mb-1">Export Format</label>
          <select
            id="exportFormat"
            name="exportFormat"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={exportFormat}
            onChange={(e) => setExportFormat(e.target.value as 'csv' | 'json' | 'pdf')}
          >
            <option value="csv">CSV</option>
            <option value="json">JSON</option>
            <option value="pdf">PDF</option>
          </select>
        </div>

        {/* Report Type Selection */}
        <div>
          <label htmlFor="reportType" className="block text-sm font-medium text-gray-700 mb-1">Report Type</label>
          <select
            id="reportType"
            name="reportType"
            className="mt-1 block w-full pl-3 pr-10 py-2 text-base border-gray-300 focus:outline-none focus:ring-blue-500 focus:border-blue-500 sm:text-sm rounded-md"
            value={reportType}
            onChange={(e) => setReportType(e.target.value as 'summary' | 'detailed' | 'executive')}
          >
            <option value="summary">Automated Report (Summary)</option>
            <option value="detailed">Automated Report (Detailed)</option>
            <option value="executive">Executive Summary</option>
          </select>
        </div>
      </div>

      <button
        onClick={handleExport}
        className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-indigo-600 hover:bg-indigo-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-indigo-500 mr-4"
      >
        Export / Generate Report
      </button>

      <div className="mt-6 border-t border-gray-200 pt-6">
        <h3 className="text-lg font-semibold text-gray-800 mb-3">Automated Reporting & Distribution</h3>
        <div className="flex items-center mb-4">
          <input
            id="scheduleReport"
            type="checkbox"
            className="h-4 w-4 text-indigo-600 focus:ring-indigo-500 border-gray-300 rounded"
            checked={isScheduled}
            onChange={(e) => setIsScheduled(e.target.checked)}
          />
          <label htmlFor="scheduleReport" className="ml-2 block text-sm text-gray-900">
            Schedule automated report generation
          </label>
        </div>

        {isScheduled && (
          <div className="mb-4">
            <label htmlFor="emailRecipient" className="block text-sm font-medium text-gray-700 mb-1">Email Recipient (comma-separated)</label>
            <input
              type="email"
              id="emailRecipient"
              name="emailRecipient"
              className="mt-1 block w-full shadow-sm sm:text-sm border-gray-300 rounded-md p-2"
              placeholder="e.g., analytics@example.com"
              value={emailRecipient}
              onChange={(e) => setEmailRecipient(e.target.value)}
            />
            <p className="mt-2 text-sm text-gray-500">Reports will be generated and emailed periodically.</p>
          </div>
        )}
        
        {isScheduled && (
          <button
            onClick={handleScheduleReport}
            className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-emerald-600 hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-500"
          >
            Confirm Schedule
          </button>
        )}
      </div>
    </div>
  );
};

export default DataExporter;