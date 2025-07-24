/**
 * Zustand store for FBA-Bench Dashboard state management
 */

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import {
  DashboardState,
  ExecutiveSummary,
  FinancialDeepDive,
  ProductMarketAnalysis,
  SupplyChainOperations,
  AgentCognition,
  KPIMetrics,
  AsyncState,
  FilterState,
  TabState,
  WebSocketEvent,
  EventType
} from '@/types/dashboard';

interface DashboardStore {
  // Tab state
  tabState: TabState;
  setActiveTab: (tab: string) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;

  // Dashboard data
  executiveSummary: AsyncState<ExecutiveSummary>;
  financialDeepDive: AsyncState<FinancialDeepDive>;
  productMarketAnalysis: AsyncState<ProductMarketAnalysis>;
  supplyChainOperations: AsyncState<SupplyChainOperations>;
  agentCognition: AsyncState<AgentCognition>;
  kpiMetrics: AsyncState<KPIMetrics>;

  // Filters
  filters: FilterState;
  setTimeRange: (start: string, end: string) => void;
  setEventTypeFilter: (eventTypes: EventType[]) => void;
  setShowOnlyAgent: (showOnly: boolean) => void;

  // WebSocket connection
  wsConnected: boolean;
  setWsConnected: (connected: boolean) => void;

  // Actions
  updateExecutiveSummary: (data: ExecutiveSummary) => void;
  updateFinancialDeepDive: (data: FinancialDeepDive) => void;
  updateProductMarketAnalysis: (data: ProductMarketAnalysis) => void;
  updateSupplyChainOperations: (data: SupplyChainOperations) => void;
  updateAgentCognition: (data: AgentCognition) => void;
  updateKPIMetrics: (data: KPIMetrics) => void;

  // Real-time updates
  handleWebSocketEvent: (event: WebSocketEvent) => void;

  // Utility actions
  clearAllData: () => void;
  refreshAllData: () => Promise<void>;
}

const createAsyncState = <T>(): AsyncState<T> => ({
  data: null,
  loading: false,
  error: null,
  lastUpdated: null,
});

const updateAsyncState = <T>(
  state: AsyncState<T>,
  data: T
): AsyncState<T> => ({
  ...state,
  data,
  loading: false,
  error: null,
  lastUpdated: new Date().toISOString(),
});

const setAsyncLoading = <T>(state: AsyncState<T>): AsyncState<T> => ({
  ...state,
  loading: true,
  error: null,
});

const setAsyncError = <T>(
  state: AsyncState<T>,
  error: string
): AsyncState<T> => ({
  ...state,
  loading: false,
  error,
});

export const useDashboardStore = create<DashboardStore>()(
  subscribeWithSelector((set, get) => ({
    // Initial tab state
    tabState: {
      activeTab: 'executive-summary',
      loading: false,
      error: null,
    },

    // Initial data states
    executiveSummary: createAsyncState<ExecutiveSummary>(),
    financialDeepDive: createAsyncState<FinancialDeepDive>(),
    productMarketAnalysis: createAsyncState<ProductMarketAnalysis>(),
    supplyChainOperations: createAsyncState<SupplyChainOperations>(),
    agentCognition: createAsyncState<AgentCognition>(),
    kpiMetrics: createAsyncState<KPIMetrics>(),

    // Initial filters
    filters: {
      timeRange: {
        start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(), // 7 days ago
        end: new Date().toISOString(),
      },
      eventTypes: ['adversarial', 'agent_action', 'market_event', 'customer_event'],
      showOnlyAgent: false,
    },

    // WebSocket state
    wsConnected: false,

    // Tab actions
    setActiveTab: (tab: string) =>
      set((state) => ({
        tabState: { ...state.tabState, activeTab: tab },
      })),

    setLoading: (loading: boolean) =>
      set((state) => ({
        tabState: { ...state.tabState, loading },
      })),

    setError: (error: string | null) =>
      set((state) => ({
        tabState: { ...state.tabState, error },
      })),

    // Filter actions
    setTimeRange: (start: string, end: string) =>
      set((state) => ({
        filters: {
          ...state.filters,
          timeRange: { start, end },
        },
      })),

    setEventTypeFilter: (eventTypes: EventType[]) =>
      set((state) => ({
        filters: {
          ...state.filters,
          eventTypes,
        },
      })),

    setShowOnlyAgent: (showOnly: boolean) =>
      set((state) => ({
        filters: {
          ...state.filters,
          showOnlyAgent: showOnly,
        },
      })),

    // WebSocket actions
    setWsConnected: (connected: boolean) =>
      set({ wsConnected: connected }),

    // Data update actions
    updateExecutiveSummary: (data: ExecutiveSummary) =>
      set((state) => ({
        executiveSummary: updateAsyncState(state.executiveSummary, data),
      })),

    updateFinancialDeepDive: (data: FinancialDeepDive) =>
      set((state) => ({
        financialDeepDive: updateAsyncState(state.financialDeepDive, data),
      })),

    updateProductMarketAnalysis: (data: ProductMarketAnalysis) =>
      set((state) => ({
        productMarketAnalysis: updateAsyncState(state.productMarketAnalysis, data),
      })),

    updateSupplyChainOperations: (data: SupplyChainOperations) =>
      set((state) => ({
        supplyChainOperations: updateAsyncState(state.supplyChainOperations, data),
      })),

    updateAgentCognition: (data: AgentCognition) =>
      set((state) => ({
        agentCognition: updateAsyncState(state.agentCognition, data),
      })),

    updateKPIMetrics: (data: KPIMetrics) =>
      set((state) => ({
        kpiMetrics: updateAsyncState(state.kpiMetrics, data),
      })),

    // WebSocket event handler
    handleWebSocketEvent: (event: WebSocketEvent) => {
      const { event_type, data } = event;

      switch (event_type) {
        case 'kpi_update':
          get().updateKPIMetrics(data as KPIMetrics);
          break;

        case 'simulation_update':
          // Trigger refresh of all data
          get().refreshAllData();
          break;

        case 'financial_update':
          // Update financial data if provided
          if (data.financial_deep_dive) {
            get().updateFinancialDeepDive(data.financial_deep_dive);
          }
          break;

        case 'market_update':
          // Update market data if provided
          if (data.product_market_analysis) {
            get().updateProductMarketAnalysis(data.product_market_analysis);
          }
          break;

        case 'agent_action':
          // Update agent cognition data if provided
          if (data.agent_cognition) {
            get().updateAgentCognition(data.agent_cognition);
          }
          break;

        default:
          console.log('Unknown WebSocket event type:', event_type);
      }
    },

    // Utility actions
    clearAllData: () =>
      set({
        executiveSummary: createAsyncState<ExecutiveSummary>(),
        financialDeepDive: createAsyncState<FinancialDeepDive>(),
        productMarketAnalysis: createAsyncState<ProductMarketAnalysis>(),
        supplyChainOperations: createAsyncState<SupplyChainOperations>(),
        agentCognition: createAsyncState<AgentCognition>(),
        kpiMetrics: createAsyncState<KPIMetrics>(),
      }),

    refreshAllData: async () => {
      // This will be implemented by the API service
      // For now, just mark as loading
      set((state) => ({
        executiveSummary: setAsyncLoading(state.executiveSummary),
        financialDeepDive: setAsyncLoading(state.financialDeepDive),
        productMarketAnalysis: setAsyncLoading(state.productMarketAnalysis),
        supplyChainOperations: setAsyncLoading(state.supplyChainOperations),
        agentCognition: setAsyncLoading(state.agentCognition),
        kpiMetrics: setAsyncLoading(state.kpiMetrics),
      }));
    },
  }))
);

// Selectors for common data access patterns
export const useExecutiveSummary = () =>
  useDashboardStore((state) => state.executiveSummary);

export const useFinancialDeepDive = () =>
  useDashboardStore((state) => state.financialDeepDive);

export const useProductMarketAnalysis = () =>
  useDashboardStore((state) => state.productMarketAnalysis);

export const useSupplyChainOperations = () =>
  useDashboardStore((state) => state.supplyChainOperations);

export const useAgentCognition = () =>
  useDashboardStore((state) => state.agentCognition);

export const useKPIMetrics = () =>
  useDashboardStore((state) => state.kpiMetrics);

export const useTabState = () =>
  useDashboardStore((state) => state.tabState);

export const useFilters = () =>
  useDashboardStore((state) => state.filters);

export const useWebSocketStatus = () =>
  useDashboardStore((state) => state.wsConnected);

// Action selectors
export const useTabActions = () =>
  useDashboardStore((state) => ({
    setActiveTab: state.setActiveTab,
    setLoading: state.setLoading,
    setError: state.setError,
  }));

export const useFilterActions = () =>
  useDashboardStore((state) => ({
    setTimeRange: state.setTimeRange,
    setEventTypeFilter: state.setEventTypeFilter,
    setShowOnlyAgent: state.setShowOnlyAgent,
  }));

export const useDataActions = () =>
  useDashboardStore((state) => ({
    updateExecutiveSummary: state.updateExecutiveSummary,
    updateFinancialDeepDive: state.updateFinancialDeepDive,
    updateProductMarketAnalysis: state.updateProductMarketAnalysis,
    updateSupplyChainOperations: state.updateSupplyChainOperations,
    updateAgentCognition: state.updateAgentCognition,
    updateKPIMetrics: state.updateKPIMetrics,
    clearAllData: state.clearAllData,
    refreshAllData: state.refreshAllData,
  }));