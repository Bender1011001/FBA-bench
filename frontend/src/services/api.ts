/**
 * API service for FBA-Bench Dashboard
 * Handles all HTTP requests to the FastAPI backend
 */

import {
  DashboardState,
  ExecutiveSummary,
  FinancialDeepDive,
  ProductMarketAnalysis,
  SupplyChainOperations,
  AgentCognition,
  KPIMetrics,
  APIResponse,
} from '@/types/dashboard';

const API_BASE_URL = (window as any).REACT_APP_API_URL || 'http://localhost:8000';

class DashboardAPIError extends Error {
  constructor(
    message: string,
    public status: number,
    public response?: any
  ) {
    super(message);
    this.name = 'APIError';
  }
}

class DashboardAPI {
  private baseURL: string;

  constructor(baseURL: string = API_BASE_URL) {
    this.baseURL = baseURL;
  }

  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${this.baseURL}${endpoint}`;
    
    const config: RequestInit = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        const errorData = await response.json().catch(() => ({}));
        throw new DashboardAPIError(
          errorData.error || `HTTP ${response.status}: ${response.statusText}`,
          response.status,
          errorData
        );
      }

      return await response.json();
    } catch (error) {
      if (error instanceof DashboardAPIError) {
        throw error;
      }
      
      // Network or other errors
      throw new DashboardAPIError(
        `Network error: ${error instanceof Error ? error.message : 'Unknown error'}`,
        0
      );
    }
  }

  // Health and status endpoints
  async getHealth(): Promise<{ status: string; timestamp: string; simulation_connected: boolean }> {
    return this.request('/api/health');
  }

  async getStatus(): Promise<{
    name: string;
    version: string;
    status: string;
    simulation_connected: boolean;
    active_websockets: number;
    cache_size: number;
  }> {
    return this.request('/');
  }

  // Dashboard data endpoints
  async getCompleteDashboardState(): Promise<DashboardState> {
    return this.request('/api/dashboard/complete');
  }

  async getExecutiveSummary(): Promise<ExecutiveSummary> {
    return this.request('/api/dashboard/executive-summary');
  }

  async getFinancialDeepDive(): Promise<FinancialDeepDive> {
    return this.request('/api/dashboard/financial');
  }

  async getProductMarketAnalysis(): Promise<ProductMarketAnalysis> {
    return this.request('/api/dashboard/product-market');
  }

  async getSupplyChainOperations(): Promise<SupplyChainOperations> {
    return this.request('/api/dashboard/supply-chain');
  }

  async getAgentCognition(): Promise<AgentCognition> {
    return this.request('/api/dashboard/agent-cognition');
  }

  async getKPIMetrics(): Promise<KPIMetrics> {
    return this.request('/api/kpis');
  }

  // Cache management
  async clearCache(): Promise<{ message: string }> {
    return this.request('/api/cache/clear', { method: 'POST' });
  }

  async getCacheStats(): Promise<{ size: number; default_ttl: number }> {
    return this.request('/api/cache/stats');
  }

  // Simulation integration
  async connectSimulation(simulationData: Record<string, any>): Promise<{ message: string; timestamp: string }> {
    return this.request('/api/simulation/connect', {
      method: 'POST',
      body: JSON.stringify(simulationData),
    });
  }

  async sendSimulationUpdate(updateData: Record<string, any>): Promise<{ message: string }> {
    return this.request('/api/simulation/update', {
      method: 'POST',
      body: JSON.stringify(updateData),
    });
  }
}

// WebSocket service for real-time updates
export class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000; // Start with 1 second
  private listeners: Map<string, Set<(data: any) => void>> = new Map();

  constructor(private url: string = `ws://localhost:8000/ws`) {}

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url);

        this.ws.onopen = () => {
          console.log('WebSocket connected');
          this.reconnectAttempts = 0;
          this.reconnectDelay = 1000;
          this.emit('connection', { status: 'connected' });
          resolve();
        };

        this.ws.onmessage = (event) => {
          try {
            const data = JSON.parse(event.data);
            this.emit(data.event_type || 'message', data);
          } catch (error) {
            console.error('Failed to parse WebSocket message:', error);
          }
        };

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason);
          this.emit('connection', { status: 'disconnected' });
          
          if (!event.wasClean && this.reconnectAttempts < this.maxReconnectAttempts) {
            this.scheduleReconnect();
          }
        };

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error);
          this.emit('error', { error });
          reject(error);
        };

      } catch (error) {
        reject(error);
      }
    });
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }
  }

  send(data: any): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(data));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  on(event: string, callback: (data: any) => void): void {
    if (!this.listeners.has(event)) {
      this.listeners.set(event, new Set());
    }
    this.listeners.get(event)!.add(callback);
  }

  off(event: string, callback: (data: any) => void): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.delete(callback);
    }
  }

  private emit(event: string, data: any): void {
    const eventListeners = this.listeners.get(event);
    if (eventListeners) {
      eventListeners.forEach(callback => callback(data));
    }
  }

  private scheduleReconnect(): void {
    this.reconnectAttempts++;
    console.log(`Scheduling WebSocket reconnect attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts} in ${this.reconnectDelay}ms`);
    
    setTimeout(() => {
      this.connect().catch(error => {
        console.error('WebSocket reconnect failed:', error);
        this.reconnectDelay = Math.min(this.reconnectDelay * 2, 30000); // Max 30 seconds
      });
    }, this.reconnectDelay);
  }

  get isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}

// Export singleton instances
export const dashboardAPI = new DashboardAPI();
export const webSocketService = new WebSocketService();

// Utility functions for common API patterns
export const withErrorHandling = async <T>(
  apiCall: () => Promise<T>,
  errorCallback?: (error: DashboardAPIError) => void
): Promise<T | null> => {
  try {
    return await apiCall();
  } catch (error) {
    if (error instanceof DashboardAPIError) {
      console.error('API Error:', error.message, error.status);
      errorCallback?.(error);
    } else {
      console.error('Unexpected error:', error);
    }
    return null;
  }
};

export const retryWithBackoff = async <T>(
  apiCall: () => Promise<T>,
  maxRetries: number = 3,
  baseDelay: number = 1000
): Promise<T> => {
  let lastError: Error;

  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await apiCall();
    } catch (error) {
      lastError = error as Error;
      
      if (attempt === maxRetries) {
        throw lastError;
      }

      const delay = baseDelay * Math.pow(2, attempt);
      console.log(`API call failed, retrying in ${delay}ms (attempt ${attempt + 1}/${maxRetries + 1})`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
  }

  throw lastError!;
};

// Export types for external use
export type { DashboardAPIError };