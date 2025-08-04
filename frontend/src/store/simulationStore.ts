import { create } from 'zustand';
import type {
  SimulationSnapshot,
  SimulationEvent,
  ConnectionStatus,
  SimulationStatus,
  SystemHealth
} from '../types';

// Message queue for WebSocket updates to prevent race conditions
interface QueuedMessage {
  type: 'snapshot' | 'status' | 'event' | 'health' | 'connection';
  data: unknown;
  timestamp: number;
}

// State update queue to ensure atomic updates
let stateUpdateQueue: Promise<void> = Promise.resolve();

interface SimulationState {
  id: string;
  snapshot: SimulationSnapshot | null;
  
  // Derived state from snapshot - single source of truth
  status: SimulationStatus['status'];
  totalTicks: number;
  ticksPerSecond: number;

  systemHealth: SystemHealth | null;

  connectionStatus: ConnectionStatus;
  
  eventLog: SimulationEvent[];
  
  lastWebSocketMessage: MessageEvent | null; // Added

  isLoading: boolean;
  error: string | null;
}

interface SimulationStore {
  simulation: SimulationState;
  
  // Atomic state update methods
  setSnapshot: (snapshot: SimulationSnapshot) => void;
  setConnectionStatus: (status: Partial<ConnectionStatus>) => void;
  addEvent: (event: SimulationEvent) => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  resetSimulation: () => void;
  setSimulationStatus: (status: SimulationStatus) => void;
  setSystemHealth: (health: SystemHealth) => void;
  setLastWebSocketMessage: (message: MessageEvent | null) => void; // Added action
  
  // Message queue management
  processMessageQueue: () => void;
  clearMessageQueue: () => void;
  
  getCurrentMetrics: () => {
    currentTick: number;
    totalTicks: number;
    simulationTime: string;
    realTime: string;
    revenue: number;
    costs: number;
    profit: number;
    activeAgentCount: number;
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
  totalTicks: 0,
  ticksPerSecond: 0,
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

// Message queue for WebSocket updates
let messageQueue: QueuedMessage[] = [];
let isProcessingQueue = false;

export const useSimulationStore = create<SimulationStore>((set, get) => ({
  simulation: initialSimulationState,
  
  // Atomic state update with queue management
  setSnapshot: (snapshot) => {
    // Add to message queue for processing
    messageQueue.push({
      type: 'snapshot',
      data: snapshot,
      timestamp: Date.now(),
    });
    
    // Process the queue asynchronously
    get().processMessageQueue();
  },
    
  setConnectionStatus: (status) => {
    // Add to message queue for processing
    messageQueue.push({
      type: 'connection',
      data: status,
      timestamp: Date.now(),
    });
    
    // Process the queue asynchronously
    get().processMessageQueue();
  },
    
  addEvent: (event) => {
    // Add to message queue for processing
    messageQueue.push({
      type: 'event',
      data: event,
      timestamp: Date.now(),
    });
    
    // Process the queue asynchronously
    get().processMessageQueue();
  },
    
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

  setSimulationStatus: (newStatus) => {
    // Add to message queue for processing
    messageQueue.push({
      type: 'status',
      data: newStatus,
      timestamp: Date.now(),
    });
    
    // Process the queue asynchronously
    get().processMessageQueue();
  },

  setSystemHealth: (health) => {
    // Add to message queue for processing
    messageQueue.push({
      type: 'health',
      data: health,
      timestamp: Date.now(),
    });
    
    // Process the queue asynchronously
    get().processMessageQueue();
  },
  
  setLastWebSocketMessage: (message) => {
    // This is a direct update as it's just for reference and doesn't affect critical state
    set((state) => ({
      simulation: {
        ...state.simulation,
        lastWebSocketMessage: message,
      },
    }));
  },

  // Process message queue to ensure atomic state updates
  processMessageQueue: () => {
    if (isProcessingQueue || messageQueue.length === 0) {
      return;
    }
    
    isProcessingQueue = true;
    
    // Chain state updates to ensure they happen in order
    stateUpdateQueue = stateUpdateQueue
      .then(async () => {
        // Process all messages in the current queue
        const messagesToProcess = [...messageQueue];
        messageQueue = [];
        
        for (const message of messagesToProcess) {
          try {
            switch (message.type) {
              case 'snapshot':
                await new Promise<void>((resolve) => {
                  set((state) => {
                    const snapshot = message.data as SimulationSnapshot;
                    resolve();
                    return {
                      simulation: {
                        ...state.simulation,
                        snapshot,
                        error: null,
                      }
                    };
                  });
                });
                break;
                
              case 'status':
                await new Promise<void>((resolve) => {
                  set((state) => {
                    const newStatus = message.data as SimulationStatus;
                    resolve();
                    return {
                      simulation: {
                        ...state.simulation,
                        id: newStatus.id || state.simulation.id,
                        status: newStatus.status,
                        totalTicks: newStatus.totalTicks,
                        ticksPerSecond: newStatus.ticksPerSecond,
                      },
                    };
                  });
                });
                break;
                
              case 'event':
                await new Promise<void>((resolve) => {
                  set((state) => {
                    const event = message.data as SimulationEvent;
                    resolve();
                    return {
                      simulation: {
                        ...state.simulation,
                        eventLog: [event, ...state.simulation.eventLog].slice(0, 1000), // Keep last 1000 events
                      },
                    };
                  });
                });
                break;
                
              case 'health':
                await new Promise<void>((resolve) => {
                  set((state) => {
                    const health = message.data as SystemHealth;
                    resolve();
                    return {
                      simulation: {
                        ...state.simulation,
                        systemHealth: health,
                      },
                    };
                  });
                });
                break;
                
              case 'connection':
                await new Promise<void>((resolve) => {
                  set((state) => {
                    const status = message.data as Partial<ConnectionStatus>;
                    resolve();
                    return {
                      simulation: {
                        ...state.simulation,
                        connectionStatus: {
                          ...state.simulation.connectionStatus,
                          ...status
                        }
                      }
                    };
                  });
                });
                break;
            }
          } catch (error) {
            console.error(`Error processing message of type ${message.type}:`, error);
            // Continue processing other messages even if one fails
          }
        }
      })
      .finally(() => {
        isProcessingQueue = false;
        
        // If new messages were added while processing, process them too
        if (messageQueue.length > 0) {
          get().processMessageQueue();
        }
      });
  },
  
  // Clear the message queue (useful for cleanup)
  clearMessageQueue: () => {
    messageQueue = [];
    isProcessingQueue = false;
    stateUpdateQueue = Promise.resolve();
  },
  
  getCurrentMetrics: () => {
    const { snapshot } = get().simulation;
    if (!snapshot) {
      return {
        currentTick: 0,
        totalTicks: 0,
        simulationTime: '00:00:00',
        realTime: new Date().toLocaleTimeString(),
        revenue: 0,
        costs: 0,
        profit: 0,
        activeAgentCount: 0,
        totalSales: '$0.00',
        ourProductPrice: '$0.00',
        trustScore: 0,
        competitorCount: 0,
        recentSalesCount: 0,
      };
    }
    
    // Safe calculation functions
    const safeParseFloat = (value: string | number | undefined): number => {
      if (typeof value === 'number') return isFinite(value) ? value : 0;
      if (typeof value === 'string') {
        const parsed = parseFloat(value.replace(/[$,]/g, ''));
        return isFinite(parsed) ? parsed : 0;
      }
      return 0;
    };
    
    return {
      currentTick: snapshot.current_tick || 0,
      totalTicks: snapshot.metadata?.total_ticks || 1000,
      simulationTime: snapshot.simulation_time || '00:00:00',
      realTime: new Date(snapshot.last_update).toLocaleTimeString(),
      revenue: safeParseFloat(snapshot.financial_summary?.total_revenue?.amount),
      costs: safeParseFloat(snapshot.financial_summary?.total_costs?.amount),
      profit: safeParseFloat(snapshot.financial_summary?.total_profit?.amount),
      activeAgentCount: snapshot.agents ? Object.keys(snapshot.agents).length : 0,
      totalSales: snapshot.financial_summary?.total_sales?.amount || '$0.00',
      ourProductPrice: snapshot.products?.[Object.keys(snapshot.products)[0]]?.current_price?.amount || '$0.00',
      trustScore: snapshot.market_summary?.trust_score || 0,
      competitorCount: snapshot.competitor_states?.length || 0,
      recentSalesCount: snapshot.recent_sales?.length || 0,
    };
  },
}));