import React, { useState, useEffect } from 'react';
import { KPIDashboard } from './components/KPIDashboard';
import { EventLog } from './components/EventLog';
import { ConnectionStatus } from './components/ConnectionStatus';
import { SimulationControls } from './components/SimulationControls';
import { AgentMonitor } from './components/AgentMonitor';
import { SystemHealthMonitor } from './components/SystemHealthMonitor';
import { SimulationStats } from './components/SimulationStats';
import { ConfigurationWizard } from './pages/ConfigurationWizard';
import { ExperimentManagement } from './pages/ExperimentManagement';
import ResultsAnalysis from './pages/ResultsAnalysis';
import ErrorBoundary from './components/ErrorBoundary';
import NotificationSystem from './components/NotificationSystem';
import { notificationService } from './utils/notificationService';
import { ThemeProvider, useTheme } from './contexts/ThemeContext';
import {
  DashboardErrorBoundary,
  SimulationErrorBoundary,
  AgentMonitorErrorBoundary,
  SystemHealthErrorBoundary,
  EventLogErrorBoundary,
  ConfigurationErrorBoundary,
  ExperimentErrorBoundary,
  ResultsAnalysisErrorBoundary,
  StatsErrorBoundary,
  WebSocketErrorBoundary
} from './components/SpecializedErrorBoundaries';

// Define types for settings
interface AppSettings {
  theme: 'light' | 'dark' | 'auto';
  autoRefresh: boolean;
  refreshInterval: number;
  notificationsEnabled: boolean;
  autoScroll: boolean;
}

// Default settings
const DEFAULT_SETTINGS: AppSettings = {
  theme: 'light',
  autoRefresh: true,
  refreshInterval: 5000,
  notificationsEnabled: true,
  autoScroll: true,
};

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'controls' | 'agents' | 'system-health' | 'events' | 'configure' | 'experiments' | 'results-analysis' | 'settings' | 'stats'>('dashboard');
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const { setTheme } = useTheme();

  // Load settings from localStorage on mount
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('app-settings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        setSettings({ ...DEFAULT_SETTINGS, ...parsedSettings });
      }
    } catch (error) {
      console.error('Failed to load settings from localStorage:', error);
    }
  }, []);

  // Save settings to localStorage when they change
  useEffect(() => {
    try {
      localStorage.setItem('app-settings', JSON.stringify(settings));
    } catch (error) {
      console.error('Failed to save settings to localStorage:', error);
    }
  }, [settings]);

  // Handle settings changes
  const handleSettingsChange = (newSettings: Partial<AppSettings>) => {
    const updatedSettings = { ...settings, ...newSettings };
    setSettings(updatedSettings);
    
    // Update theme if it changed
    if (newSettings.theme) {
      setTheme(newSettings.theme);
    }
    
    // Show notification for settings changes
    if (newSettings.notificationsEnabled && settings.notificationsEnabled !== newSettings.notificationsEnabled) {
      notificationService.success('Browser notifications enabled', 3000);
    }
  };

  // Show welcome notification on first load
  useEffect(() => {
    const hasVisitedBefore = localStorage.getItem('fba-bench-visited');
    if (!hasVisitedBefore) {
      notificationService.info('Welcome to FBA-Bench! Your simulation dashboard is ready.', 5000);
      localStorage.setItem('fba-bench-visited', 'true');
    }
  }, []);

  return (
    <div className="min-h-screen bg-gray-50">
      <ErrorBoundary>
        <NotificationSystem />
        {/* Header */}
        <header className="bg-white shadow-sm border-b border-gray-200">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
            <div className="flex items-center justify-between h-16">
            {/* Logo and Title */}
            <div className="flex items-center">
              <div className="flex-shrink-0">
                <h1 className="text-2xl font-bold text-gray-900">FBA-Bench</h1>
              </div>
              <div className="hidden md:block ml-4">
                <p className="text-sm text-gray-600">Real-time Amazon FBA Simulation Dashboard</p>
              </div>
            </div>

            {/* Navigation Tabs */}
            <nav className="flex space-x-4 lg:space-x-8 overflow-x-auto pb-2">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'dashboard'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('controls')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'controls'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Controls
              </button>
              <button
                onClick={() => setActiveTab('agents')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'agents'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Agents
              </button>
              <button
                onClick={() => setActiveTab('system-health')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'system-health'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                System Health
              </button>
              <button
                onClick={() => setActiveTab('events')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'events'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Event Log
              </button>
              <button
                onClick={() => setActiveTab('stats')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'stats'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Statistics
              </button>
              <button
                onClick={() => setActiveTab('configure')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'configure'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Configure
              </button>
              <button
                onClick={() => setActiveTab('experiments')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'experiments'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Experiments
              </button>
              <button
                onClick={() => setActiveTab('results-analysis')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'results-analysis'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Results Analysis
              </button>
              <button
                onClick={() => setActiveTab('settings')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors whitespace-nowrap
                  ${activeTab === 'settings'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Settings
              </button>
            </nav>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {activeTab === 'dashboard' && (
          <div className="space-y-8">
            <DashboardErrorBoundary>
              <KPIDashboard />
            </DashboardErrorBoundary>
          </div>
        )}

        {activeTab === 'controls' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Simulation Controls</h2>
            <SimulationErrorBoundary>
              <SimulationControls />
            </SimulationErrorBoundary>
          </div>
        )}

        {activeTab === 'agents' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Live Agent Monitoring</h2>
            <AgentMonitorErrorBoundary>
              <AgentMonitor />
            </AgentMonitorErrorBoundary>
          </div>
        )}
        
        {activeTab === 'system-health' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">System Health</h2>
            <SystemHealthErrorBoundary>
              <SystemHealthMonitor />
            </SystemHealthErrorBoundary>
          </div>
        )}

        {activeTab === 'events' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Event Stream</h2>
                <p className="text-gray-600 mt-1">
                  Real-time simulation events and system activity
                </p>
              </div>
              <WebSocketErrorBoundary>
                <ConnectionStatus showDetails={false} className="w-96" />
              </WebSocketErrorBoundary>
            </div>
            <EventLogErrorBoundary>
              <EventLog className="h-[calc(100vh-12rem)]" />
            </EventLogErrorBoundary>
          </div>
        )}

        {activeTab === 'stats' && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-900 mb-4">Simulation Statistics</h2>
            <StatsErrorBoundary>
              <SimulationStats />
            </StatsErrorBoundary>
          </div>
        )}

        {activeTab === 'configure' && (
          <div className="space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-2xl font-bold text-gray-900">Simulation Configuration</h2>
                <p className="text-gray-600 mt-1">
                  Configure and launch new FBA simulation experiments
                </p>
              </div>
            </div>
            <ConfigurationErrorBoundary>
              <ConfigurationWizard />
            </ConfigurationErrorBoundary>
          </div>
        )}

        {activeTab === 'experiments' && (
          <div className="space-y-6">
            <ExperimentErrorBoundary>
              <ExperimentManagement />
            </ExperimentErrorBoundary>
          </div>
        )}

        {activeTab === 'results-analysis' && (
          <div className="space-y-6">
            <ResultsAnalysisErrorBoundary>
              <ResultsAnalysis />
            </ResultsAnalysisErrorBoundary>
          </div>
        )}

        {activeTab === 'settings' && (
          <div className="space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-gray-900">Settings</h2>
              <p className="text-gray-600 mt-1">
                Configure dashboard preferences and connection settings
              </p>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Connection Settings */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Connection</h3>
                <WebSocketErrorBoundary>
                  <ConnectionStatus showDetails={true} />
                </WebSocketErrorBoundary>
              </div>

              {/* Dashboard Settings */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Dashboard Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Refresh Interval
                    </label>
                    <select
                      value={settings.refreshInterval}
                      onChange={(e) => handleSettingsChange({ refreshInterval: parseInt(e.target.value) })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="1000">1 second</option>
                      <option value="5000">5 seconds</option>
                      <option value="10000">10 seconds</option>
                      <option value="30000">30 seconds</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Theme
                    </label>
                    <select
                      value={settings.theme}
                      onChange={(e) => handleSettingsChange({ theme: e.target.value as 'light' | 'dark' | 'auto' })}
                      className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                    >
                      <option value="light">Light</option>
                      <option value="dark">Dark</option>
                      <option value="auto">Auto</option>
                    </select>
                  </div>

                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="autoScroll"
                      checked={settings.autoScroll}
                      onChange={(e) => handleSettingsChange({ autoScroll: e.target.checked })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label htmlFor="autoScroll" className="ml-2 block text-sm text-gray-700">
                      Auto-scroll event log
                    </label>
                  </div>

                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="notifications"
                      checked={settings.notificationsEnabled}
                      onChange={(e) => handleSettingsChange({ notificationsEnabled: e.target.checked })}
                      className="h-4 w-4 text-blue-600 focus:ring-blue-500 border-gray-300 rounded"
                    />
                    <label htmlFor="notifications" className="ml-2 block text-sm text-gray-700">
                      Enable browser notifications
                    </label>
                  </div>
                </div>
              </div>

              {/* System Information */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">System Information</h3>
                <div className="space-y-3 text-sm">
                  <div className="flex justify-between">
                    <span className="text-gray-500">Frontend Version:</span>
                    <span className="font-medium">1.0.0</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">Build:</span>
                    <span className="font-medium">Development</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">API Endpoint:</span>
                    <span className="font-medium">http://localhost:8000</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-500">WebSocket Endpoint:</span>
                    <span className="font-medium">ws://localhost:8000/ws/events</span>
                  </div>
                </div>
              </div>

              {/* About */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">About FBA-Bench</h3>
                <div className="text-sm text-gray-600 space-y-2">
                  <p>
                    FBA-Bench is a comprehensive simulation platform for Amazon FBA (Fulfillment by Amazon)
                    business scenarios, featuring multi-agent competition, real-time pricing dynamics,
                    and advanced analytics.
                  </p>
                  <p>
                    This dashboard provides real-time monitoring and control capabilities for simulation
                    experiments, including KPI tracking, event logging, and system health monitoring.
                  </p>
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <p className="text-xs text-gray-500">
                      Built with React, TypeScript, Tailwind CSS, and Zustand
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500">
              © 2025 FBA-Bench Dashboard. Built for simulation research and analysis.
            </p>
            <div className="flex items-center space-x-4 text-sm text-gray-500">
              <span>Real-time Dashboard</span>
              <span>•</span>
              <span>WebSocket Enabled</span>
              <span>•</span>
              <span>TypeScript</span>
            </div>
          </div>
        </div>
        </footer>
      </ErrorBoundary>
      
      {/* Notification System */}
      <NotificationSystem />
    </div>
  );
}

const AppWithTheme: React.FC = () => (
  <ThemeProvider>
    <App />
  </ThemeProvider>
);

export default AppWithTheme;
