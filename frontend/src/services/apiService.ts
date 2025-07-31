// frontend/src/services/apiService.ts

import { handleError, ErrorCategory } from '../utils/errorHandler';
import type { AppError } from '../utils/errorHandler';
import type { ApiResponse } from '../types';

// Configuration
const API_BASE_URL = 'http://localhost:8000';
const DEFAULT_TIMEOUT = 15000; // 15 seconds
const RETRY_ATTEMPTS = 3;
const RETRY_DELAY_MS = 1000; // Initial retry delay

class ApiService {
  private baseUrl: string;
  private timeout: number;
  private isOnline: boolean = navigator.onLine;

  constructor(baseUrl: string = API_BASE_URL, timeout: number = DEFAULT_TIMEOUT) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
    window.addEventListener('online', this.updateOnlineStatus.bind(this));
    window.addEventListener('offline', this.updateOnlineStatus.bind(this));
  }

  private updateOnlineStatus() {
    this.isOnline = navigator.onLine;
    console.log(`Connection status changed: ${this.isOnline ? 'Online' : 'Offline'}`);
    // Potentially trigger a global notification system here
    if (this.isOnline) {
      // Logic to resend queued requests could go here if implemented
      console.log('Back online. Attempting to resend any pending requests...');
    } else {
      console.log('Offline. Requests will be queued or fail fast.');
    }
  }

  /**
   * Generic fetch wrapper with timeout and retry logic.
   */
  private async fetchWithRetry<T>(
    url: string,
    options: RequestInit = {},
    retries: number = RETRY_ATTEMPTS,
    delay: number = RETRY_DELAY_MS,
    attempt: number = 1
  ): Promise<ApiResponse<T>> {
    if (!this.isOnline) {
      const offlineError = new Error('Network is offline. Please check your internet connection.');
      // Queueing logic would go here: Add request to a queue and return a promise that resolves later
      // For now, we'll throw an error
      handleError({
        name: 'OfflineError',
        message: offlineError.message,
        category: ErrorCategory.Network,
        isHandled: true,
        details: { url, options },
        userMessage: 'You are currently offline. Actions requiring a network connection are unavailable.',
      });
      throw offlineError;
    }

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
        const errorBody = await response.json().catch(() => ({ message: 'No additional error details.' }));
        const errorMessage = `HTTP error! Status: ${response.status} ${response.statusText || ''}`;
        
        const appError: AppError = { // Changed to const
          name: 'HttpError',
          message: errorMessage,
          category: ErrorCategory.Network,
          statusCode: response.status,
          isHandled: false, // Mark as unhandled by default, errorHandler will log
          details: errorBody,
        };

        // Specific HTTP status code handling
        if (response.status >= 500) {
          appError.category = ErrorCategory.System;
          appError.userMessage = 'A server error occurred. Please try again later.';
        } else if (response.status === 401 || response.status === 403) {
          appError.category = ErrorCategory.User; // Could be auth error, user action needed
          appError.userMessage = 'Authentication failed. Please log in again.';
          // Potentially redirect to login
        } else if (response.status === 404) {
          appError.category = ErrorCategory.User;
          appError.userMessage = 'The requested resource was not found.';
        } else if (response.status >= 400 && response.status < 500) {
            appError.category = ErrorCategory.Validation; // Client-side input or request error
            appError.userMessage = (appError.details as { message?: string })?.message || 'There was an issue with your request. Please check your input.'; // Safely access message
        }

        handleError(appError); // Use centralized error handler
        throw appError; // Re-throw the AppError
      }

      const data: ApiResponse<T> = await response.json();
      return data;

    } catch (error: unknown) {
      clearTimeout(timeoutId);
      
      const caughtError = error instanceof Error ? error : new Error(String(error));
      
      if (caughtError.name === 'AbortError') {
        const timeoutAppError: AppError = {
          name: 'TimeoutError',
          message: 'Request timed out.',
          category: ErrorCategory.Network,
          isHandled: true,
          details: { url, options, attempt },
          userMessage: 'The request took too long to respond. Please try again.',
        };
        handleError(timeoutAppError);
        throw timeoutAppError;
      }
      
      // Retry logic for transient network errors only
      const isNetworkError = caughtError instanceof TypeError || // e.g. network down
                             caughtError.message.includes('Failed to fetch'); // Common fetch network error message

      if (isNetworkError && retries > 0) {
        console.warn(`Retrying request to ${url} (Attempt ${attempt}/${RETRY_ATTEMPTS})...`);
        await new Promise(res => setTimeout(res, delay));
        return this.fetchWithRetry(url, options, retries - 1, delay * 2, attempt + 1); // Exponential backoff
      }

      // If it's an AppError already, just re-throw after logging
      if ((error as AppError)?.isHandled) {
        throw error;
      }

      // Handle other unknown errors
      const finalError = handleError(error); // Let centralized handler categorize and log
      throw finalError; // Re-throw the categorized error
    }
  }

  /**
   * Generic request wrapper for all HTTP methods.
   */
  private async request<T>(
    method: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH',
    endpoint: string,
    data?: unknown,
    options: RequestInit = {}
  ): Promise<ApiResponse<T>> {
    const url = this.getApiUrl(endpoint);
    const config: RequestInit = {
      method,
      ...options,
    };

    if (data) {
      config.body = JSON.stringify(data);
    }

    return await this.fetchWithRetry<T>(url, config);
  }

  /**
   * Build full API URL, assuming endpoint already includes versioning if needed.
   */
  private getApiUrl(endpoint: string): string {
    return `${this.baseUrl}${endpoint}`;
  }

  /**
   * Generic GET request.
   */
  async get<T>(endpoint: string, options?: RequestInit): Promise<ApiResponse<T>> {
    return this.request<T>('GET', endpoint, undefined, options);
  }

  /**
   * Generic POST request.
   */
  async post<T>(endpoint: string, data?: unknown, options?: RequestInit): Promise<ApiResponse<T>> {
    return this.request<T>('POST', endpoint, data, options);
  }

  /**
   * Generic PUT request.
   */
  async put<T>(endpoint: string, data?: unknown, options?: RequestInit): Promise<ApiResponse<T>> {
    return this.request<T>('PUT', endpoint, data, options);
  }

  /**
   * Generic DELETE request.
   */
  async delete<T>(endpoint: string, options?: RequestInit): Promise<ApiResponse<T>> {
    return this.request<T>('DELETE', endpoint, undefined, options);
  }

  /**
   * Generic PATCH request.
   */
  async patch<T>(endpoint: string, data?: unknown, options?: RequestInit): Promise<ApiResponse<T>> {
    return this.request<T>('PATCH', endpoint, data, options);
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

// Remaining mock data from original file. Keep as is.
import type { ResultsData, ExperimentExecution } from '../types';

/**
 * Mock data for demonstration purposes.
 * In a real application, these would come from the backend.
 */
const mockResultsData: ResultsData = {
  experimentId: 'example-experiment-123',
  simulationResults: Array.from({ length: 100 }, (_, i) => ({
    timestamp: new Date(Date.now() - (100 - i) * 1000).toISOString(),
    tick: i + 1,
    revenue: 1000 + i * 10 + Math.random() * 500,
    costs: 500 + i * 5 + Math.random() * 200,
    profit: 500 + i * 5 + Math.random() * 300,
    agentMetrics: {
      'agent-1': Math.random() * 100,
      'agent-2': Math.random() * 100,
    },
    marketMetrics: {
      priceTrend: 100 + i * 0.5,
      inventoryLevels: 500 - i * 2,
    },
  })),
  aggregatedMetrics: {
    totalRevenue: 150000,
    totalCosts: 70000,
    totalProfit: 80000,
    averageTicksPerSecond: 12.5,
    topPerformingAgent: 'Agent Alpha',
    experimentDuration: 100,
  },
  agentPerformance: [
    { agentId: 'Agent Alpha', profit: 50000, decisionsMade: 1200, accuracy: 0.95 },
    { agentId: 'Agent Beta', profit: 25000, decisionsMade: 900, accuracy: 0.88 },
    { agentId: 'Agent Gamma', profit: 5000, decisionsMade: 300, accuracy: 0.75 },
    { agentId: 'Agent Delta', profit: 8000, decisionsMade: 450, accuracy: 0.82 },
  ],
  financialMetrics: {
    totalRevenue: 150000,
    totalCosts: 70000,
    totalProfit: 80000,
  },
  timeSeriesData: Array.from({ length: 100 }, (_, i) => ({
    timestamp: new Date(Date.now() - (100 - i) * 1000).toISOString(),
    tick: i + 1,
    revenue: 1000 + i * 10 + Math.random() * 500,
    costs: 500 + i * 5 + Math.random() * 200,
    profit: 500 + i * 5 + Math.random() * 300,
  })),
};

const mockExperimentDetails: ExperimentExecution = {
  id: 'example-experiment-123',
  experimentName: 'Initial Pricing Strategy Validation',
  description: 'A simulation to test the impact of dynamic pricing on overall revenue and profit.',
  config: {}, // Placeholder
  status: 'completed',
  progress: 100,
  startTime: new Date(Date.now() - 3600000).toISOString(),
  endTime: new Date().toISOString(),
  estimatedCompletionTime: new Date().toISOString(),
  lastUpdated: new Date().toISOString(),
  resultsSummary: {
    maxProfit: 80000,
    avgRevenue: 1500,
  },
};

/**
 * Fetches comprehensive results data for a given experiment ID.
 * @param experimentId The ID of the experiment to fetch results for.
 * @returns A Promise that resolves to ResultsData.
 * TODO: Implement actual API call, data pagination, and caching.
 */
export async function fetchResultsData(experimentId: string): Promise<ResultsData> {
  console.log(`[API Service] Fetching results for experiment: ${experimentId}`);
  // In a real application, make an API call like:
  // return apiService.get<ResultsData>(`/experiments/${experimentId}/results`);
  
  // Simulate API call delay
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve(mockResultsData);
    }, 500); // Simulate network delay
  });
}

/**
 * Fetches details for a specific experiment.
 * @param experimentId The ID of the experiment to fetch details for.
 * @returns A Promise that resolves to ExperimentExecution.
 * TODO: Implement actual API call and caching.
 */
export async function fetchExperimentDetails(experimentId: string): Promise<ExperimentExecution> {
    console.log(`[API Service] Fetching details for experiment: ${experimentId}`);
    // In a real application, make an API call like:
    // return apiService.get<ExperimentExecution>(`/experiments/${experimentId}`);

    // Simulate API call delay
    return new Promise((resolve) => {
        setTimeout(() => {
            resolve(mockExperimentDetails);
        });
    });
}