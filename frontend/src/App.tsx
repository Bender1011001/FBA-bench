/**
 * Main App component for FBA-Bench Dashboard
 */

import React, { useEffect, useState } from 'react';
import { useDashboardStore, useTabState, useTabActions, useWebSocketStatus } from '@/store/dashboardStore';
import { webSocketService, dashboardAPI } from '@/services/api';
import { ExecutiveSummaryTab } from '@/components/tabs/ExecutiveSummaryTab';
import { FinancialDeepDiveTab } from '@/components/tabs/FinancialDeepDiveTab';
import { ProductMarketTab } from '@/components/tabs/ProductMarketTab';
import { SupplyChainTab } from '@/components/tabs/SupplyChainTab';
import { AgentCognitionTab } from '@/components/tabs/AgentCognitionTab';
import { Header } from '@/components/layout/Header';
import { TabNavigation } from '@/components/layout/TabNavigation';
import { ConnectionStatus } from '@/components/common/ConnectionStatus';
import { LoadingSpinner } from '@/components/common/LoadingSpinner';
import { ErrorBoundary } from '@/components/common/ErrorBoundary';

const tabs = [
  {
    id: 'executive-summary',
    name: 'Executive Summary',
    icon: '📊',
    component: ExecutiveSummaryTab,
  },
  {
    id: 'financial',
    name: 'Financial Deep Dive',
    icon: '💰',
    component: FinancialDeepDiveTab,
  },
  {
    id: 'product-market',
    name: 'Product & Market',
    icon: '📈',
    component: ProductMarketTab,
  },
  {
    id: 'supply-chain',
    name: 'Supply Chain',
    icon: '🚚',
    component: SupplyChainTab,
  },
  {
    id: 'agent-cognition',
    name: 'Agent Cognition',
    icon: '🧠',
    component: AgentCognitionTab,
  },
];

function App() {
  const tabState = useTabState();
  const { setActiveTab, setError } = useTabActions();
  const wsConnected = useWebSocketStatus();
  const handleWebSocketEvent = useDashboardStore((state) => state.handleWebSocketEvent);
  const setWsConnected = useDashboardStore((state) => state.setWsConnected);
  
  const [apiConnected, setApiConnected] = useState(false);
  const [initializing, setInitializing] = useState(true);

  // Initialize connections and services
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Check API connection
        const health = await dashboardAPI.getHealth();
        setApiConnected(health.simulation_connected);

        // Initialize WebSocket connection
        await webSocketService.connect();
        
        // Set up WebSocket event handlers
        webSocketService.on('connection', (data) => {
          setWsConnected(data.status === 'connected');
        });

        webSocketService.on('kpi_update', handleWebSocketEvent);
        webSocketService.on('simulation_update', handleWebSocketEvent);
        webSocketService.on('financial_update', handleWebSocketEvent);
        webSocketService.on('market_update', handleWebSocketEvent);
        webSocketService.on('agent_action', handleWebSocketEvent);

        webSocketService.on('error', (error) => {
          console.error('WebSocket error:', error);
          setError('WebSocket connection error');
        });

      } catch (error) {
        console.error('Failed to initialize app:', error);
        setError('Failed to connect to dashboard API');
        setApiConnected(false);
      } finally {
        setInitializing(false);
      }
    };

    initializeApp();

    // Cleanup on unmount
    return () => {
      webSocketService.disconnect();
    };
  }, [handleWebSocketEvent, setWsConnected, setError]);

  // Get current tab component
  const currentTab = tabs.find(tab => tab.id === tabState.activeTab) || tabs[0];
  const TabComponent = currentTab.component;

  if (initializing) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <LoadingSpinner size="lg" />
          <p className="mt-4 text-gray-600">Initializing FBA-Bench Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <ErrorBoundary>
      <div className="min-h-screen bg-gray-50">
        {/* Header */}
        <Header />
        
        {/* Connection Status */}
        <ConnectionStatus 
          apiConnected={apiConnected}
          wsConnected={wsConnected}
        />
        
        {/* Main Content */}
        <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          {/* Tab Navigation */}
          <TabNavigation 
            tabs={tabs}
            activeTab={tabState.activeTab}
            onTabChange={setActiveTab}
          />
          
          {/* Tab Content */}
          <div className="mt-6">
            {tabState.error ? (
              <div className="card">
                <div className="text-center py-12">
                  <div className="text-danger-500 text-4xl mb-4">⚠️</div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">
                    Dashboard Error
                  </h3>
                  <p className="text-gray-600 mb-4">{tabState.error}</p>
                  <button
                    onClick={() => setError(null)}
                    className="btn-primary"
                  >
                    Retry
                  </button>
                </div>
              </div>
            ) : (
              <ErrorBoundary>
                <TabComponent />
              </ErrorBoundary>
            )}
          </div>
        </main>
        
        {/* Footer */}
        <footer className="bg-white border-t border-gray-200 mt-12">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-4">
            <div className="flex items-center justify-between text-sm text-gray-500">
              <div>
                FBA-Bench Dashboard v1.0.0
              </div>
              <div className="flex items-center space-x-4">
                <span className={`status-indicator ${apiConnected ? 'status-ok' : 'status-danger'}`}>
                  <span className="status-dot"></span>
                  API {apiConnected ? 'Connected' : 'Disconnected'}
                </span>
                <span className={`status-indicator ${wsConnected ? 'status-ok' : 'status-warning'}`}>
                  <span className="status-dot"></span>
                  Real-time {wsConnected ? 'Active' : 'Inactive'}
                </span>
              </div>
            </div>
          </div>
        </footer>
      </div>
    </ErrorBoundary>
  );
}

export default App;