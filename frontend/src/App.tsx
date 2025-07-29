import React, { useState } from 'react';
import { KPIDashboard } from './components/KPIDashboard';
import { EventLog } from './components/EventLog';
import { ConnectionStatus } from './components/ConnectionStatus';

function App() {
  const [activeTab, setActiveTab] = useState<'dashboard' | 'events' | 'settings'>('dashboard');

  return (
    <div className="min-h-screen bg-gray-50">
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
            <nav className="flex space-x-8">
              <button
                onClick={() => setActiveTab('dashboard')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors
                  ${activeTab === 'dashboard'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Dashboard
              </button>
              <button
                onClick={() => setActiveTab('events')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors
                  ${activeTab === 'events'
                    ? 'bg-blue-100 text-blue-700'
                    : 'text-gray-500 hover:text-gray-700'
                  }
                `}
              >
                Event Log
              </button>
              <button
                onClick={() => setActiveTab('settings')}
                className={`
                  px-3 py-2 rounded-md text-sm font-medium transition-colors
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
            <KPIDashboard />
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
              <ConnectionStatus showDetails={false} className="w-96" />
            </div>
            <EventLog className="h-[calc(100vh-12rem)]" />
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
                <ConnectionStatus showDetails={true} />
              </div>

              {/* Dashboard Settings */}
              <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Dashboard Settings</h3>
                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Refresh Interval
                    </label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                      <option value="1000">1 second</option>
                      <option value="5000" defaultChecked>5 seconds</option>
                      <option value="10000">10 seconds</option>
                      <option value="30000">30 seconds</option>
                    </select>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-2">
                      Theme
                    </label>
                    <select className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent">
                      <option value="light" defaultChecked>Light</option>
                      <option value="dark">Dark</option>
                      <option value="auto">Auto</option>
                    </select>
                  </div>

                  <div className="flex items-center">
                    <input
                      type="checkbox"
                      id="autoScroll"
                      defaultChecked
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
    </div>
  );
}

export default App;
