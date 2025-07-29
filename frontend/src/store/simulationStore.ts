import { create } from 'zustand';
import type {
  SimulationSnapshot,
  SimulationEvent,
  ConnectionStatus
} from '../types';

// Enhanced simulation state using proper backend types
interface SimulationState {
  // Core simulation data from backend
  snapshot: SimulationSnapshot | null;
  
  // Connection management
  connectionStatus: ConnectionStatus;
  
  // Event log for debugging and monitoring
  eventLog: SimulationEvent[];
  
  // UI state
  isLoading: boolean;
  error: string | null;
}

interface SimulationStore {
  simulation: SimulationState;
  
  // Actions
  setSnapshot: (snapshot: SimulationSnapshot) => void;
  setConnectionStatus: (status: Partial<ConnectionStatus>) => void;
  addEvent: (event: SimulationEvent) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  resetSimulation: () => void;
  
  // Computed values
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
  snapshot: null,
  connectionStatus: {
    connected: false,
    reconnectAttempts: 0,
  },
  eventLog: [],
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
        error: null
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
      totalSales: snapshot.total_sales.amount,
      ourProductPrice: snapshot.our_product_price.amount,
      trustScore: snapshot.trust_score,
      competitorCount: snapshot.competitor_states.length,
      recentSalesCount: snapshot.recent_sales.length,
    };
  },
}));