import React, { useState, useEffect } from 'react';
import { apiService } from '../services/apiService';
import type { 
  BenchmarkResult, 
  BenchmarkReport, 
  ReportSection,
  ExportOptions 
} from '../types';
import { notificationService } from '../utils/notificationService';
import LoadingSpinner from './LoadingSpinner';
import ErrorBoundary from './ErrorBoundary';

interface ReportGeneratorProps {
  benchmarkResults: BenchmarkResult[];
  className?: string;
}

const ReportGenerator: React.FC<ReportGeneratorProps> = ({ 
  benchmarkResults, 
  className = '' 
}) => {
  const [selectedResult, setSelectedResult] = useState<string>('');
  const [reportTemplates, setReportTemplates] = useState<any[]>([]);
  const [reportConfig, setReportConfig] = useState({
    title: '',
    description: '',
    templateId: '',
    format: 'pdf' as 'pdf' | 'html' | 'json' | 'csv',
    sections: {
      summary: true,
      charts: true,
      tables: true,
      raw_data: false,
      analysis: true
    }
  });
  const [generatedReports, setGeneratedReports] = useState<BenchmarkReport[]>([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isDownloading, setIsDownloading] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'create' | 'history'>('create');

  useEffect(() => {
    fetchReportTemplates();
    fetchGeneratedReports();
  }, []);

  useEffect(() => {
    if (selectedResult) {
      const result = benchmarkResults.find(r => r.benchmark_name === selectedResult);
      if (result) {
        setReportConfig(prev => ({
          ...prev,
          title: `Benchmark Report - ${result.benchmark_name}`,
          description: `Comprehensive report for benchmark execution on ${new Date(result.start_time).toLocaleDateString()}`
        }));
      }
    }
  }, [selectedResult, benchmarkResults]);

  const fetchReportTemplates = async () => {
    setIsLoading(true);
    try {
      const response = await apiService.get<any[]>('/benchmarking/report-templates');
      setReportTemplates(response.data);
      
      // Set default template if available
      if (response.data.length > 0 && !reportConfig.templateId) {
        setReportConfig(prev => ({
          ...prev,
          templateId: response.data[0].id
        }));
      }
    } catch (err) {
      console.error('Error fetching report templates:', err);
      setError('Failed to fetch report templates');
    } finally {
      setIsLoading(false);
    }
  };

  const fetchGeneratedReports = async () => {
    try {
      const response = await apiService.get<BenchmarkReport[]>('/benchmarking/reports');
      setGeneratedReports(response.data);
    } catch (err) {
      console.error('Error fetching generated reports:', err);
      setError('Failed to fetch generated reports');
    }
  };

  const handleConfigChange = (field: string, value: any) => {
    setReportConfig(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleSectionToggle = (section: string) => {
    setReportConfig(prev => ({
      ...prev,
      sections: {
        ...prev.sections,
        [section]: !prev.sections[section as keyof typeof prev.sections]
      }
    }));
  };

  const validateConfig = (): boolean => {
    if (!selectedResult) {
      setError('Please select a benchmark result');
      return false;
    }
    if (!reportConfig.title.trim()) {
      setError('Report title is required');
      return false;
    }
    if (!reportConfig.templateId) {
      setError('Please select a report template');
      return false;
    }
    return true;
  };

  const generateReport = async () => {
    if (!validateConfig()) return;

    setIsGenerating(true);
    setError(null);

    try {
      const sections: ReportSection[] = [];
      
      if (reportConfig.sections.summary) {
        sections.push({
          id: 'summary',
          title: 'Executive Summary',
          type: 'summary',
          content: {},
          order: 1,
          visible: true
        });
      }
      
      if (reportConfig.sections.charts) {
        sections.push({
          id: 'charts',
          title: 'Performance Charts',
          type: 'charts',
          content: {},
          order: 2,
          visible: true
        });
      }
      
      if (reportConfig.sections.tables) {
        sections.push({
          id: 'tables',
          title: 'Detailed Results',
          type: 'tables',
          content: {},
          order: 3,
          visible: true
        });
      }
      
      if (reportConfig.sections.raw_data) {
        sections.push({
          id: 'raw_data',
          title: 'Raw Data',
          type: 'raw_data',
          content: {},
          order: 4,
          visible: true
        });
      }
      
      if (reportConfig.sections.analysis) {
        sections.push({
          id: 'analysis',
          title: 'Analysis & Insights',
          type: 'analysis',
          content: {},
          order: 5,
          visible: true
        });
      }

      const reportData: Partial<BenchmarkReport> = {
        benchmark_id: selectedResult,
        title: reportConfig.title,
        description: reportConfig.description,
        template_id: reportConfig.templateId,
        format: reportConfig.format,
        sections,
        metadata: {
          generated_by: 'user',
          generated_at: new Date().toISOString(),
          benchmark_result: selectedResult
        }
      };

      const response = await apiService.post<BenchmarkReport>('/benchmarking/reports', reportData);
      
      notificationService.success('Report generated successfully', 3000);
      setGeneratedReports(prev => [response.data, ...prev]);
      
      // Reset form
      setSelectedResult('');
      setReportConfig({
        title: '',
        description: '',
        templateId: reportTemplates[0]?.id || '',
        format: 'pdf',
        sections: {
          summary: true,
          charts: true,
          tables: true,
          raw_data: false,
          analysis: true
        }
      });
    } catch (err) {
      console.error('Error generating report:', err);
      setError('Failed to generate report');
    } finally {
      setIsGenerating(false);
    }
  };

  const downloadReport = async (reportId: string, format: string) => {
    setIsDownloading(true);
    try {
      const response = await apiService.get(`/benchmarking/reports/${reportId}/download`, {
        headers: { 
          'Accept': format === 'pdf' ? 'application/pdf' : 
                   format === 'html' ? 'text/html' :
                   format === 'csv' ? 'text/csv' : 'application/json'
        }
      });
      
      // Create download link
      const blob = new Blob([JSON.stringify(response.data, null, 2)], { 
        type: format === 'pdf' ? 'application/pdf' : 
             format === 'html' ? 'text/html' :
             format === 'csv' ? 'text/csv' : 'application/json'
      });
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = `benchmark_report_${reportId}.${format}`;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      
      notificationService.success(`Report downloaded as ${format.toUpperCase()}`, 3000);
    } catch (err) {
      console.error('Error downloading report:', err);
      setError('Failed to download report');
    } finally {
      setIsDownloading(false);
    }
  };

  const deleteReport = async (reportId: string) => {
    if (!confirm('Are you sure you want to delete this report?')) return;
    
    try {
      await apiService.delete(`/benchmarking/reports/${reportId}`);
      notificationService.success('Report deleted successfully', 3000);
      setGeneratedReports(prev => prev.filter(r => r.id !== reportId));
    } catch (err) {
      console.error('Error deleting report:', err);
      setError('Failed to delete report');
    }
  };

  const formatDate = (dateString: string): string => {
    return new Date(dateString).toLocaleString();
  };

  if (isLoading) {
    return (
      <div className={`flex items-center justify-center h-96 ${className}`}>
        <LoadingSpinner size="large" />
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className={`space-y-6 ${className}`}>
        <h2 className="text-2xl font-bold text-gray-900">Report Generator</h2>
        
        {/* Error Display */}
        {error && (
          <div className="p-4 bg-red-50 border border-red-200 rounded-md">
            <h3 className="text-sm font-medium text-red-800">Error</h3>
            <p className="text-sm text-red-700 mt-1">{error}</p>
            <button
              onClick={() => setError(null)}
              className="mt-2 text-sm text-red-600 hover:text-red-800"
            >
              Dismiss
            </button>
          </div>
        )}

        {/* Tab Navigation */}
        <div className="flex border-b border-gray-200 mb-6">
          {[
            { id: 'create', label: 'Create Report' },
            { id: 'history', label: 'Report History' }
          ].map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id as any)}
              className={`px-4 py-2 font-medium transition-colors ${
                activeTab === tab.id
                  ? 'text-blue-600 border-b-2 border-blue-600'
                  : 'text-gray-700 hover:text-blue-600 border-b-2 border-transparent hover:border-blue-300'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </div>

        {/* Create Report Tab */}
        {activeTab === 'create' && (
          <div className="space-y-6">
            {/* Benchmark Selection */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Select Benchmark Result</h3>
              <select
                value={selectedResult}
                onChange={(e) => setSelectedResult(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
              >
                <option value="">Select a benchmark result</option>
                {benchmarkResults.map(result => (
                  <option key={result.benchmark_name} value={result.benchmark_name}>
                    {result.benchmark_name} - {new Date(result.start_time).toLocaleDateString()}
                  </option>
                ))}
              </select>
            </div>

            {/* Report Configuration */}
            <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">Report Configuration</h3>
              
              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Report Title *
                  </label>
                  <input
                    type="text"
                    value={reportConfig.title}
                    onChange={(e) => handleConfigChange('title', e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter report title"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Description
                  </label>
                  <textarea
                    value={reportConfig.description}
                    onChange={(e) => handleConfigChange('description', e.target.value)}
                    rows={3}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    placeholder="Enter report description"
                  />
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Template *
                    </label>
                    <select
                      value={reportConfig.templateId}
                      onChange={(e) => handleConfigChange('templateId', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="">Select a template</option>
                      {reportTemplates.map(template => (
                        <option key={template.id} value={template.id}>
                          {template.name} - {template.category}
                        </option>
                      ))}
                    </select>
                  </div>

                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">
                      Format
                    </label>
                    <select
                      value={reportConfig.format}
                      onChange={(e) => handleConfigChange('format', e.target.value)}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="pdf">PDF</option>
                      <option value="html">HTML</option>
                      <option value="json">JSON</option>
                      <option value="csv">CSV</option>
                    </select>
                  </div>
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">
                    Include Sections
                  </label>
                  <div className="grid grid-cols-2 md:grid-cols-3 gap-3">
                    {Object.entries(reportConfig.sections).map(([key, value]) => (
                      <label key={key} className="flex items-center space-x-2">
                        <input
                          type="checkbox"
                          checked={value}
                          onChange={() => handleSectionToggle(key)}
                          className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                        />
                        <span className="text-sm text-gray-700 capitalize">
                          {key.replace('_', ' ')}
                        </span>
                      </label>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Generate Button */}
            <div className="flex justify-end">
              <button
                onClick={generateReport}
                disabled={isGenerating}
                className="px-6 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors disabled:opacity-50"
              >
                {isGenerating ? 'Generating...' : 'Generate Report'}
              </button>
            </div>
          </div>
        )}

        {/* Report History Tab */}
        {activeTab === 'history' && (
          <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
            <h3 className="text-lg font-semibold text-gray-900 mb-4">Generated Reports</h3>
            
            {generatedReports.length === 0 ? (
              <p className="text-gray-500 text-center py-8">No reports generated yet</p>
            ) : (
              <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Title
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Benchmark
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Format
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Generated
                      </th>
                      <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                        Actions
                      </th>
                    </tr>
                  </thead>
                  <tbody className="bg-white divide-y divide-gray-200">
                    {generatedReports.map(report => (
                      <tr key={report.id} className="hover:bg-gray-50">
                        <td className="px-6 py-4 whitespace-nowrap">
                          <div className="text-sm font-medium text-gray-900">{report.title}</div>
                          {report.description && (
                            <div className="text-sm text-gray-500">{report.description}</div>
                          )}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {report.benchmark_id}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900 uppercase">
                          {report.format}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-900">
                          {formatDate(report.generated_at)}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap text-sm font-medium">
                          <button
                            onClick={() => downloadReport(report.id, report.format)}
                            disabled={isDownloading}
                            className="text-blue-600 hover:text-blue-900 mr-3 disabled:opacity-50"
                          >
                            Download
                          </button>
                          <button
                            onClick={() => deleteReport(report.id)}
                            className="text-red-600 hover:text-red-900"
                          >
                            Delete
                          </button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </div>
    </ErrorBoundary>
  );
};

export default ReportGenerator;