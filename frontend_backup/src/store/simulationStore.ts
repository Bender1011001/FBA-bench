import { create } from 'zustand';

// Define initial type for the simulation state (will be refined later with generated types)
interface SimulationState {
  currentTick: number;
  totalRevenue: number;
  totalProfit: number;
  unitsSold: number;
  activeAgents: number;
  connectionStatus: 'connected' | 'disconnected' | 'connecting';
  eventLog: any[]; // Will refine this type later
}

interface SimulationStore {
  simulation: SimulationState;
  setSimulationState: (newState: Partial<SimulationState>) => void;
  addEventToLog: (event: any) => void;
  resetSimulationState: () => void;
}

const initialSimulationState: SimulationState = {
  currentTick: 0,
  totalRevenue: 0,
  totalProfit: 0,
  unitsSold: 0,
  activeAgents: 0,
  connectionStatus: 'connecting',
  eventLog: [],
};

export const useSimulationStore = create<SimulationStore>((set) => ({
  simulation: initialSimulationState,
  setSimulationState: (newState) =>
    set((state) => ({ simulation: { ...state.simulation, ...newState } })),
  addEventToLog: (event) =>
    set((state) => ({
      simulation: {
        ...state.simulation,
        eventLog: [event, ...state.simulation.eventLog].slice(0, 100), // Keep last 100 events
      },
    })),
  resetSimulationState: () => set({ simulation: initialSimulationState }),
}));