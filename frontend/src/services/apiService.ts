import type { SimulationSnapshot, ApiResponse } from '../types';

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const API_VERSION = 'v1';

class ApiService {
  private baseUrl: string;
  private timeout: number;

  constructor(baseUrl: string = API_BASE_URL, timeout: number = 10000) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
  }

  /**
   * Generic fetch wrapper with error handling and timeout
   */
  private async fetchWithTimeout<T>(
    url: string,
    options: RequestInit = {}
  ): Promise<T> {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), this.timeout);

    try {
      const response = await fetch(url, {
        ...options,
        signal: controller.signal,
        headers: {
          'Content-Type': 'application/json',
          ...options.headers,
        },
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      return data;
    } catch (error) {
      clearTimeout(timeoutId);
      
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          throw new Error('Request timeout');
        }
        throw error;
      }
      
      throw new Error('Unknown error occurred');
    }
  }

  /**
   * Build full API URL
   */
  private getApiUrl(endpoint: string): string {
    return `${this.baseUrl}/api/${API_VERSION}${endpoint}`;
  }

  /**
   * Get current simulation snapshot
   */
  async getSimulationSnapshot(): Promise<SimulationSnapshot> {
    try {
      const data = await this.fetchWithTimeout<SimulationSnapshot>(
        this.getApiUrl('/simulation/snapshot')
      );
      return data;
    } catch (error) {
      console.error('Failed to fetch simulation snapshot:', error);
      throw new Error(`Failed to fetch simulation snapshot: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Health check endpoint
   */
  async healthCheck(): Promise<{ status: string; timestamp: string }> {
    try {
      const data = await this.fetchWithTimeout<{ status: string; timestamp: string }>(
        `${this.baseUrl}/health`
      );
      return data;
    } catch (error) {
      console.error('Health check failed:', error);
      throw new Error(`Health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Set product price (if this endpoint becomes available)
   */
  async setProductPrice(asin: string, price: string): Promise<ApiResponse<void>> {
    try {
      const data = await this.fetchWithTimeout<ApiResponse<void>>(
        this.getApiUrl('/simulation/set-price'),
        {
          method: 'POST',
          body: JSON.stringify({ asin, price }),
        }
      );
      return data;
    } catch (error) {
      console.error('Failed to set product price:', error);
      throw new Error(`Failed to set product price: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Start simulation (if this endpoint becomes available)
   */
  async startSimulation(): Promise<ApiResponse<void>> {
    try {
      const data = await this.fetchWithTimeout<ApiResponse<void>>(
        this.getApiUrl('/simulation/start'),
        { method: 'POST' }
      );
      return data;
    } catch (error) {
      console.error('Failed to start simulation:', error);
      throw new Error(`Failed to start simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Stop simulation (if this endpoint becomes available)
   */
  async stopSimulation(): Promise<ApiResponse<void>> {
    try {
      const data = await this.fetchWithTimeout<ApiResponse<void>>(
        this.getApiUrl('/simulation/stop'),
        { method: 'POST' }
      );
      return data;
    } catch (error) {
      console.error('Failed to stop simulation:', error);
      throw new Error(`Failed to stop simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Reset simulation (if this endpoint becomes available)
   */
  async resetSimulation(): Promise<ApiResponse<void>> {
    try {
      const data = await this.fetchWithTimeout<ApiResponse<void>>(
        this.getApiUrl('/simulation/reset'),
        { method: 'POST' }
      );
      return data;
    } catch (error) {
      console.error('Failed to reset simulation:', error);
      throw new Error(`Failed to reset simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Update configuration (base URL)
   */
  updateBaseUrl(newBaseUrl: string): void {
    this.baseUrl = newBaseUrl;
  }

  /**
   * Update timeout
   */
  updateTimeout(newTimeout: number): void {
    this.timeout = newTimeout;
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export class for testing or multiple instances
export { ApiService };