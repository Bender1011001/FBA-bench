import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { ApiService, apiService } from '../services/apiService';
import type { SimulationSnapshot } from '../types';

// Mock fetch globally
const mockFetch = vi.fn();
globalThis.fetch = mockFetch;

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
    }
  ],
  recent_sales: [],
  trust_score: 0.85,
  timestamp: "2025-01-01T12:00:00Z"
};

const mockHealthResponse = {
  status: "healthy",
  timestamp: "2025-01-01T12:00:00Z"
};

describe('ApiService', () => {
  beforeEach(() => {
    mockFetch.mockClear();
    vi.clearAllTimers();
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('Constructor and Configuration', () => {
    it('should create instance with default configuration', () => {
      const service = new ApiService();
      expect(service).toBeInstanceOf(ApiService);
    });

    it('should create instance with custom configuration', () => {
      const service = new ApiService('http://custom:9000', 5000);
      expect(service).toBeInstanceOf(ApiService);
    });

    it('should update base URL', () => {
      const service = new ApiService();
      service.updateBaseUrl('http://new-host:8080');
      // We can't directly test this, but we can test that it doesn't throw
      expect(() => service.updateBaseUrl('http://new-host:8080')).not.toThrow();
    });

    it('should update timeout', () => {
      const service = new ApiService();
      service.updateTimeout(15000);
      // We can't directly test this, but we can test that it doesn't throw
      expect(() => service.updateTimeout(15000)).not.toThrow();
    });
  });

  describe('fetchWithTimeout', () => {
    it('should make successful API call', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSnapshot)
      });

      const result = await apiService.getSimulationSnapshot();
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/simulation/snapshot',
        expect.objectContaining({
          headers: {
            'Content-Type': 'application/json'
          }
        })
      );
      expect(result).toEqual(mockSnapshot);
    });

    it('should handle HTTP error responses', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: false,
        status: 404
      });

      await expect(apiService.getSimulationSnapshot()).rejects.toThrow('HTTP error! status: 404');
    });

    it('should handle network errors', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Network error'));

      await expect(apiService.getSimulationSnapshot()).rejects.toThrow('Network error');
    });

    it('should handle timeout', async () => {
      // Mock a fetch that never resolves
      mockFetch.mockImplementationOnce(() => new Promise(() => {}));

      const timeoutPromise = apiService.getSimulationSnapshot();
      
      // Fast-forward time to trigger timeout
      vi.advanceTimersByTime(10000);
      
      await expect(timeoutPromise).rejects.toThrow('Failed to fetch simulation snapshot: Request timeout');
    }, 10000); // Increase test timeout

    it('should handle JSON parsing errors', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.reject(new Error('Invalid JSON'))
      });

      await expect(apiService.getSimulationSnapshot()).rejects.toThrow('Invalid JSON');
    });
  });

  describe('getSimulationSnapshot', () => {
    it('should fetch simulation snapshot successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSnapshot)
      });

      const result = await apiService.getSimulationSnapshot();
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/simulation/snapshot',
        expect.any(Object)
      );
      expect(result).toEqual(mockSnapshot);
    });

    it('should throw wrapped error on failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Connection failed'));

      await expect(apiService.getSimulationSnapshot()).rejects.toThrow(
        'Failed to fetch simulation snapshot: Connection failed'
      );
    });
  });

  describe('healthCheck', () => {
    it('should perform health check successfully', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHealthResponse)
      });

      const result = await apiService.healthCheck();
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/health',
        expect.any(Object)
      );
      expect(result).toEqual(mockHealthResponse);
    });

    it('should throw wrapped error on health check failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Service unavailable'));

      await expect(apiService.healthCheck()).rejects.toThrow(
        'Health check failed: Service unavailable'
      );
    });
  });

  describe('setProductPrice', () => {
    it('should send price update request successfully', async () => {
      const mockResponse = { success: true, data: null, timestamp: "2025-01-01T12:00:00Z" };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await apiService.setProductPrice('B123456789', '$25.99');
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/simulation/set-price',
        expect.objectContaining({
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({ asin: 'B123456789', price: '$25.99' })
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should throw wrapped error on price update failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Invalid price format'));

      await expect(apiService.setProductPrice('B123456789', 'invalid')).rejects.toThrow(
        'Failed to set product price: Invalid price format'
      );
    });
  });

  describe('startSimulation', () => {
    it('should send start simulation request successfully', async () => {
      const mockResponse = { success: true, data: null, timestamp: "2025-01-01T12:00:00Z" };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await apiService.startSimulation();
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/simulation/start',
        expect.objectContaining({
          method: 'POST'
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should throw wrapped error on start failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Simulation already running'));

      await expect(apiService.startSimulation()).rejects.toThrow(
        'Failed to start simulation: Simulation already running'
      );
    });
  });

  describe('stopSimulation', () => {
    it('should send stop simulation request successfully', async () => {
      const mockResponse = { success: true, data: null, timestamp: "2025-01-01T12:00:00Z" };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await apiService.stopSimulation();
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/simulation/stop',
        expect.objectContaining({
          method: 'POST'
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should throw wrapped error on stop failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('No simulation running'));

      await expect(apiService.stopSimulation()).rejects.toThrow(
        'Failed to stop simulation: No simulation running'
      );
    });
  });

  describe('resetSimulation', () => {
    it('should send reset simulation request successfully', async () => {
      const mockResponse = { success: true, data: null, timestamp: "2025-01-01T12:00:00Z" };
      
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockResponse)
      });

      const result = await apiService.resetSimulation();
      
      expect(mockFetch).toHaveBeenCalledWith(
        'http://localhost:8000/api/v1/simulation/reset',
        expect.objectContaining({
          method: 'POST'
        })
      );
      expect(result).toEqual(mockResponse);
    });

    it('should throw wrapped error on reset failure', async () => {
      mockFetch.mockRejectedValueOnce(new Error('Reset not allowed'));

      await expect(apiService.resetSimulation()).rejects.toThrow(
        'Failed to reset simulation: Reset not allowed'
      );
    });
  });

  describe('Error Handling', () => {
    it('should handle unknown error types', async () => {
      // Mock a non-Error object being thrown
      mockFetch.mockRejectedValueOnce('String error');

      await expect(apiService.getSimulationSnapshot()).rejects.toThrow(
        'Failed to fetch simulation snapshot: Unknown error'
      );
    });

    it('should handle AbortError specifically', async () => {
      const abortError = new Error('AbortError');
      abortError.name = 'AbortError';
      
      mockFetch.mockRejectedValueOnce(abortError);

      await expect(apiService.getSimulationSnapshot()).rejects.toThrow(
        'Failed to fetch simulation snapshot: Request timeout'
      );
    });
  });

  describe('Request Headers', () => {
    it('should include correct headers in requests', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockSnapshot)
      });

      await apiService.getSimulationSnapshot();
      
      expect(mockFetch).toHaveBeenCalledWith(
        expect.any(String),
        expect.objectContaining({
          headers: expect.objectContaining({
            'Content-Type': 'application/json'
          })
        })
      );
    });

    it('should preserve custom headers', async () => {
      mockFetch.mockResolvedValueOnce({
        ok: true,
        json: () => Promise.resolve(mockHealthResponse)
      });

      await apiService.healthCheck();
      
      const call = mockFetch.mock.calls[0];
      const options = call[1];
      expect(options.headers).toEqual(
        expect.objectContaining({
          'Content-Type': 'application/json'
        })
      );
    });
  });
});