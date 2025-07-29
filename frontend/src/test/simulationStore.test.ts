import { describe, it, expect, beforeEach } from 'vitest';
import { useSimulationStore } from '../store/simulationStore';
import type { SimulationSnapshot, SimulationEvent, ConnectionStatus } from '../types';

// Mock data
const mockSnapshot: SimulationSnapshot = {
  current_tick: 42,
  total_sales: { amount: "$1,234.56" },
  our_product_price: { amount: "$29.99" },
  competitor_states: [
    {
      competitor_id: "comp-1",
      current_price: { amount: "$28.99" },
      last_updated: "2025-01-01T12:00:00Z"
    },
    {
      competitor_id: "comp-2", 
      current_price: { amount: "$31.50" },
      last_updated: "2025-01-01T12:00:00Z"
    }
  ],
  recent_sales: [
    {
      timestamp: "2025-01-01T12:00:00Z",
      product_asin: "B123456789",
      sale_price: { amount: "$29.99" },
      quantity: 2,
      buyer_id: "buyer-1"
    }
  ],
  trust_score: 0.85,
  timestamp: "2025-01-01T12:00:00Z"
};

const mockEvent: SimulationEvent = {
  type: 'sale_occurred',
  sale: {
    timestamp: "2025-01-01T12:00:00Z",
    product_asin: "B123456789",
    sale_price: { amount: "$29.99" },
    quantity: 1,
    buyer_id: "buyer-1"
  },
  timestamp: "2025-01-01T12:00:00Z"
};

describe('SimulationStore', () => {
  beforeEach(() => {
    // Reset store state before each test
    useSimulationStore.getState().resetSimulation();
  });

  describe('Initial State', () => {
    it('should have correct initial state', () => {
      const state = useSimulationStore.getState().simulation;
      
      expect(state.snapshot).toBeNull();
      expect(state.connectionStatus.connected).toBe(false);
      expect(state.connectionStatus.reconnectAttempts).toBe(0);
      expect(state.eventLog).toEqual([]);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('Snapshot Management', () => {
    it('should set snapshot correctly', () => {
      const { setSnapshot } = useSimulationStore.getState();
      
      setSnapshot(mockSnapshot);
      
      const state = useSimulationStore.getState().simulation;
      expect(state.snapshot).toEqual(mockSnapshot);
      expect(state.error).toBeNull();
    });

    it('should clear error when setting snapshot', () => {
      const { setError, setSnapshot } = useSimulationStore.getState();
      
      // First set an error
      setError('Test error');
      expect(useSimulationStore.getState().simulation.error).toBe('Test error');
      
      // Then set snapshot - error should be cleared
      setSnapshot(mockSnapshot);
      expect(useSimulationStore.getState().simulation.error).toBeNull();
    });
  });

  describe('Connection Status Management', () => {
    it('should update connection status', () => {
      const { setConnectionStatus } = useSimulationStore.getState();
      
      const newStatus: Partial<ConnectionStatus> = {
        connected: true,
        lastHeartbeat: "2025-01-01T12:00:00Z"
      };
      
      setConnectionStatus(newStatus);
      
      const state = useSimulationStore.getState().simulation;
      expect(state.connectionStatus.connected).toBe(true);
      expect(state.connectionStatus.lastHeartbeat).toBe("2025-01-01T12:00:00Z");
      expect(state.connectionStatus.reconnectAttempts).toBe(0); // Should preserve existing values
    });

    it('should merge connection status updates', () => {
      const { setConnectionStatus } = useSimulationStore.getState();
      
      // First update
      setConnectionStatus({ connected: true });
      
      // Second update should merge, not replace
      setConnectionStatus({ reconnectAttempts: 3 });
      
      const state = useSimulationStore.getState().simulation;
      expect(state.connectionStatus.connected).toBe(true);
      expect(state.connectionStatus.reconnectAttempts).toBe(3);
    });
  });

  describe('Event Log Management', () => {
    it('should add events to log', () => {
      const { addEvent } = useSimulationStore.getState();
      
      addEvent(mockEvent);
      
      const state = useSimulationStore.getState().simulation;
      expect(state.eventLog).toHaveLength(1);
      expect(state.eventLog[0]).toEqual(mockEvent);
    });

    it('should maintain event order (newest first)', () => {
      const { addEvent } = useSimulationStore.getState();
      
      const event1: SimulationEvent = { ...mockEvent, timestamp: "2025-01-01T12:00:00Z" };
      const event2: SimulationEvent = { ...mockEvent, timestamp: "2025-01-01T12:01:00Z" };
      
      addEvent(event1);
      addEvent(event2);
      
      const state = useSimulationStore.getState().simulation;
      expect(state.eventLog[0]).toEqual(event2); // Newest first
      expect(state.eventLog[1]).toEqual(event1);
    });

    it('should limit event log to 1000 events', () => {
      const { addEvent } = useSimulationStore.getState();
      
      // Add 1001 events
      for (let i = 0; i < 1001; i++) {
        const event: SimulationEvent = {
          ...mockEvent,
          timestamp: new Date(Date.now() + i * 1000).toISOString()
        };
        addEvent(event);
      }
      
      const state = useSimulationStore.getState().simulation;
      expect(state.eventLog).toHaveLength(1000);
    });
  });

  describe('Loading and Error States', () => {
    it('should set loading state', () => {
      const { setLoading } = useSimulationStore.getState();
      
      setLoading(true);
      expect(useSimulationStore.getState().simulation.isLoading).toBe(true);
      
      setLoading(false);
      expect(useSimulationStore.getState().simulation.isLoading).toBe(false);
    });

    it('should set error state', () => {
      const { setError } = useSimulationStore.getState();
      
      setError('Test error message');
      expect(useSimulationStore.getState().simulation.error).toBe('Test error message');
      
      setError(null);
      expect(useSimulationStore.getState().simulation.error).toBeNull();
    });
  });

  describe('Reset Functionality', () => {
    it('should reset simulation to initial state', () => {
      const store = useSimulationStore.getState();
      
      // Modify state
      store.setSnapshot(mockSnapshot);
      store.setConnectionStatus({ connected: true, reconnectAttempts: 5 });
      store.addEvent(mockEvent);
      store.setLoading(true);
      store.setError('Test error');
      
      // Reset
      store.resetSimulation();
      
      const state = useSimulationStore.getState().simulation;
      expect(state.snapshot).toBeNull();
      expect(state.connectionStatus.connected).toBe(false);
      expect(state.connectionStatus.reconnectAttempts).toBe(0);
      expect(state.eventLog).toEqual([]);
      expect(state.isLoading).toBe(false);
      expect(state.error).toBeNull();
    });
  });

  describe('Computed Metrics', () => {
    it('should return default metrics when no snapshot', () => {
      const { getCurrentMetrics } = useSimulationStore.getState();
      
      const metrics = getCurrentMetrics();
      
      expect(metrics).toEqual({
        currentTick: 0,
        totalSales: '$0.00',
        ourProductPrice: '$0.00',
        trustScore: 0,
        competitorCount: 0,
        recentSalesCount: 0
      });
    });

    it('should compute metrics from snapshot', () => {
      const { setSnapshot, getCurrentMetrics } = useSimulationStore.getState();
      
      setSnapshot(mockSnapshot);
      
      const metrics = getCurrentMetrics();
      
      expect(metrics.currentTick).toBe(42);
      expect(metrics.totalSales).toBe("$1,234.56");
      expect(metrics.ourProductPrice).toBe("$29.99");
      expect(metrics.trustScore).toBe(0.85);
      expect(metrics.competitorCount).toBe(2);
      expect(metrics.recentSalesCount).toBe(1);
    });
  });
});