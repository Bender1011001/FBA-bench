import { create } from 'zustand';
import type {
  SimulationSnapshot,
  SimulationEvent,
  ConnectionStatus,
  SimulationStatus,
  SystemHealth
} from '../types';

interface SimulationState {
  id: string;
  snapshot: SimulationSnapshot | null;
  
  status: SimulationStatus['status'];
  currentTick: number;
  totalTicks: number;
  simulationTime: string;
  realTime: string;
  ticksPerSecond: number;
  
  revenue: number;
  costs: number;
  profit: number;
  activeAgentCount: number;

  systemHealth: SystemHealth | null;

  connectionStatus: ConnectionStatus;
  
  eventLog: SimulationEvent[];
  
  lastWebSocketMessage: MessageEvent | null; // Added

  isLoading: boolean;
  error: string | null;
}

interface SimulationStore {
  simulation: SimulationState;
  
  setSnapshot: (snapshot: SimulationSnapshot) => void;
  setConnectionStatus: (status: Partial<ConnectionStatus>) => void;
  addEvent: (event: SimulationEvent) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  resetSimulation: () => void;
  setSimulationStatus: (status: SimulationStatus) => void;
  setSystemHealth: (health: SystemHealth) => void;
  setLastWebSocketMessage: (message: MessageEvent | null) => void; // Added action
  
  getCurrentMetrics: () => {
    currentTick: number;
    totalSales: string;
    ourProductPrice: string;
    trustScore: number;
    competitorCount: number;
    recentSalesCount: number;
  };
}

const initialSimulationState: SimulationState = {
  id: 'N/A', // Initial simulation ID
  snapshot: null,
  status: 'idle',
  currentTick: 0,
  totalTicks: 0,
  simulationTime: '00:00:00',
  realTime: new Date().toLocaleTimeString(),
  ticksPerSecond: 0,
  revenue: 0,
  costs: 0,
  profit: 0,
  activeAgentCount: 0,
  systemHealth: null,
  connectionStatus: {
    connected: false,
    reconnectAttempts: 0,
  },
  eventLog: [],
  lastWebSocketMessage: null, // Initial state
  isLoading: false,
  error: null,
};

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  simulation: initialSimulationState,
  
  setSnapshot: (snapshot) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        snapshot,
        error: null,
        currentTick: snapshot.current_tick || state.simulation.currentTick,
        simulationTime: snapshot.simulation_time || state.simulation.simulationTime,
        realTime: new Date(snapshot.last_update).toLocaleTimeString() || state.simulation.realTime,
        activeAgentCount: snapshot.agents ? Object.keys(snapshot.agents).length : state.simulation.activeAgentCount,
        revenue: parseFloat(snapshot.financial_summary?.total_revenue?.amount || '0') || state.simulation.revenue,
        costs: parseFloat(snapshot.financial_summary?.total_costs?.amount || '0') || state.simulation.costs,
        profit: parseFloat(snapshot.financial_summary?.total_profit?.amount || '0') || state.simulation.profit,
      }
    })),
    
  setConnectionStatus: (status) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        connectionStatus: {
          ...state.simulation.connectionStatus,
          ...status
        }
      }
    })),
    
  addEvent: (event) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        eventLog: [event, ...state.simulation.eventLog].slice(0, 1000), // Keep last 1000 events
      },
    })),
    
  setLoading: (loading) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        isLoading: loading
      }
    })),
    
  setError: (error) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        error
      }
    })),
    
  resetSimulation: () => set({ simulation: initialSimulationState }),

  setSimulationStatus: (newStatus) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        id: newStatus.id || state.simulation.id,
        status: newStatus.status,
        currentTick: newStatus.currentTick,
        totalTicks: newStatus.totalTicks,
        simulationTime: newStatus.simulationTime,
        realTime: newStatus.realTime,
        ticksPerSecond: newStatus.ticksPerSecond,
        revenue: newStatus.revenue,
        costs: newStatus.costs,
        profit: newStatus.profit,
        activeAgentCount: newStatus.activeAgentCount,
      },
    })),

  setSystemHealth: (health) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        systemHealth: health,
      },
    })),
  
  setLastWebSocketMessage: (message) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        lastWebSocketMessage: message,
      },
    })),

  getCurrentMetrics: () => {
    const { snapshot } = get().simulation;
    if (!snapshot) {
      return {
        currentTick: 0,
        totalSales: '$0.00',
        ourProductPrice: '$0.00',
        trustScore: 0,
        competitorCount: 0,
        recentSalesCount: 0,
      };
    }
    
    return {
      currentTick: snapshot.current_tick,
      totalSales: snapshot.financial_summary?.total_sales?.amount || '$0.00',
      ourProductPrice: snapshot.products?.[Object.keys(snapshot.products)[0]]?.current_price?.amount || '$0.00',
      trustScore: snapshot.market_summary?.trust_score || 0,
      competitorCount: snapshot.competitor_states?.length || 0,
      recentSalesCount: snapshot.recent_sales?.length || 0,
    };
  },
}));