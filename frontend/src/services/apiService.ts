// frontend/src/services/apiService.ts

import { handleError, ErrorCategory } from '../utils/errorHandler';
import type { AppError } from '../utils/errorHandler';
import type { ApiResponse } from '../types';
import type { ResultsData, ExperimentExecution } from '../types';

// Configuration using environment variables
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const API_KEY = import.meta.env.VITE_API_KEY || '';
const DEFAULT_TIMEOUT = parseInt(import.meta.env.VITE_API_TIMEOUT || '15000', 10); // 15 seconds
const RETRY_ATTEMPTS = parseInt(import.meta.env.VITE_API_RETRY_ATTEMPTS || '3', 10);
const RETRY_DELAY_MS = parseInt(import.meta.env.VITE_API_RETRY_DELAY_MS || '1000', 10); // Initial retry delay
const ALLOW_API_KEY_AUTH = import.meta.env.VITE_ALLOW_API_KEY_AUTH === 'true';

// Type definition for environment configuration
interface EnvironmentConfig {
  apiBaseUrl: string;
  wsUrl: string;
  apiKey: string;
  defaultTimeout: number;
  retryAttempts: number;
  retryDelayMs: number;
  allowApiKeyAuth: boolean;
  isProduction: boolean;
}

// Validate and parse environment variables
const validateEnvironment = (): EnvironmentConfig => {
  const config: EnvironmentConfig = {
    apiBaseUrl: API_BASE_URL,
    wsUrl: WS_URL,
    apiKey: API_KEY,
    defaultTimeout: DEFAULT_TIMEOUT,
    retryAttempts: RETRY_ATTEMPTS,
    retryDelayMs: RETRY_DELAY_MS,
    allowApiKeyAuth: ALLOW_API_KEY_AUTH,
    isProduction: import.meta.env.PROD,
  };

  // Validate required environment variables
  if (!config.apiBaseUrl) {
    throw new Error('VITE_API_URL environment variable is required');
  }
  
  if (!config.wsUrl) {
    throw new Error('VITE_WS_URL environment variable is required');
  }
  
  // Validate numeric values
  if (isNaN(config.defaultTimeout) || config.defaultTimeout <= 0) {
    throw new Error('VITE_API_TIMEOUT must be a positive number');
  }
  
  if (isNaN(config.retryAttempts) || config.retryAttempts < 0) {
    throw new Error('VITE_API_RETRY_ATTEMPTS must be a non-negative number');
  }
  
  if (isNaN(config.retryDelayMs) || config.retryDelayMs < 0) {
    throw new Error('VITE_API_RETRY_DELAY_MS must be a non-negative number');
  }
  
  // In production, API key should be required
  if (config.isProduction && !config.apiKey) {
    throw new Error('VITE_API_KEY environment variable is required in production');
  }
  
  return config;
};

// Validate environment on module load and export configuration
const ENV_CONFIG = validateEnvironment();
export const API_MODE_LABEL = ENV_CONFIG.isProduction ? "Backend API (production)" : "Backend API (mock)";

class ApiService {
  private baseUrl: string;
  private timeout: number;
  private isOnline: boolean = navigator.onLine;
  private apiKey: string;
  private authToken: string | null = null;

  constructor(baseUrl: string = ENV_CONFIG.apiBaseUrl, timeout: number = ENV_CONFIG.defaultTimeout) {
    this.baseUrl = baseUrl;
    this.timeout = timeout;
    this.apiKey = ENV_CONFIG.apiKey;
    this.authToken = this.getStoredAuthToken();
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
   * Get stored authentication token from secure storage
   */
  private getStoredAuthToken(): string | null {
    try {
      return localStorage.getItem('auth_token');
    } catch (error) {
      console.error('Failed to access localStorage for auth token:', error);
      return null;
    }
  }

  /**
   * Store authentication token securely
   */
  private storeAuthToken(token: string): void {
    try {
      localStorage.setItem('auth_token', token);
    } catch (error) {
      console.error('Failed to store auth token:', error);
    }
  }

  /**
   * Clear stored authentication token
   */
  private clearStoredAuthToken(): void {
    try {
      localStorage.removeItem('auth_token');
    } catch (error) {
      console.error('Failed to clear auth token:', error);
    }
  }

  /**
   * Get headers with authentication and security headers
   */
  private getHeaders(additionalHeaders: Record<string, string> = {}): Record<string, string> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
      'X-Content-Type-Options': 'nosniff',
      'X-Frame-Options': 'DENY',
      'X-XSS-Protection': '1; mode=block',
      ...additionalHeaders,
    };

    // Prefer auth token over API key for authentication
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    } else if (this.apiKey) {
      // Only use API key in development or when explicitly configured
      if (!ENV_CONFIG.isProduction || ENV_CONFIG.allowApiKeyAuth) {
        headers['Authorization'] = `Bearer ${this.apiKey}`;
      }
    }

    return headers;
  }

  /**
   * Generic fetch wrapper with timeout and retry logic.
   */
  private async fetchWithRetry<T>(
    url: string,
    options: RequestInit = {},
    retries: number = ENV_CONFIG.retryAttempts,
    delay: number = ENV_CONFIG.retryDelayMs,
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
        headers: this.getHeaders(options.headers as Record<string, string>),
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

    console.log(`ApiService: Making ${method} request to:`, url);
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

  /**
   * Fetch simulation snapshot
   */
  async getSimulationSnapshot(): Promise<unknown> {
    console.log(`[API Service] Fetching simulation snapshot`);
    
    try {
      const response = await this.get<unknown>('/api/v1/simulation/snapshot');
      return response.data;
    } catch (error) {
      console.error(`[API Service] Failed to fetch simulation snapshot:`, error);
      throw new Error(`Failed to fetch simulation snapshot: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Perform health check
   */
  async healthCheck(): Promise<unknown> {
    console.log(`[API Service] Performing health check to /api/v1/health endpoint`);
    
    try {
      const response = await this.get<unknown>('/api/v1/health');
      console.log(`[API Service] Health check response:`, response);
      return response.data;
    } catch (error) {
      console.error(`[API Service] Health check failed:`, error);
      throw new Error(`Health check failed: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Set product price
   */
  async setProductPrice(asin: string, price: string): Promise<unknown> {
    console.log(`[API Service] Setting product price for ${asin} to ${price}`);
    
    try {
      const response = await this.post<unknown>('/api/v1/simulation/set-price', { asin, price });
      return response.data;
    } catch (error) {
      console.error(`[API Service] Failed to set product price:`, error);
      throw new Error(`Failed to set product price: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Start simulation
   */
  async startSimulation(): Promise<unknown> {
    console.log(`[API Service] Starting simulation`);
    
    try {
      const response = await this.post<unknown>('/api/v1/simulation/start');
      return response.data;
    } catch (error) {
      console.error(`[API Service] Failed to start simulation:`, error);
      throw new Error(`Failed to start simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Stop simulation
   */
  async stopSimulation(): Promise<unknown> {
    console.log(`[API Service] Stopping simulation`);
    
    try {
      const response = await this.post<unknown>('/api/v1/simulation/stop');
      return response.data;
    } catch (error) {
      console.error(`[API Service] Failed to stop simulation:`, error);
      throw new Error(`Failed to stop simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Reset simulation
   */
  async resetSimulation(): Promise<unknown> {
    console.log(`[API Service] Resetting simulation`);
    
    try {
      const response = await this.post<unknown>('/api/v1/simulation/reset');
      return response.data;
    } catch (error) {
      console.error(`[API Service] Failed to reset simulation:`, error);
      throw new Error(`Failed to reset simulation: ${error instanceof Error ? error.message : 'Unknown error'}`);
    }
  }

  /**
   * Update API key
   */
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  updateApiKey(_newApiKey: string): void {
    // Update the API key for future requests
    // Note: This doesn't update the environment variable, just the in-memory value
    // In a real implementation, you might want to store this in secure storage
    // Parameter prefixed with underscore to indicate it's intentionally unused in this implementation
  }

  /**
   * Submits benchmark configuration to the backend
   * @param config The benchmark configuration to submit
   * @returns A Promise that resolves to the benchmark configuration response
   */
  async submitBenchmarkConfig(config: BenchmarkConfigRequest): Promise<BenchmarkConfigResponse> {
    console.log(`[API Service] Submitting benchmark configuration`);
    
    try {
      const response = await this.post<BenchmarkConfigResponse>('/api/v1/benchmark/config', config);
      return response.data;
    } catch (error) {
      console.error(`[API Service] Failed to submit benchmark configuration:`, error);
      
      // Re-throw the error after logging
      throw error;
    }
  }
}

// Export singleton instance
export const apiService = new ApiService();

// Export class for testing or multiple instances
export { ApiService };

// Export environment configuration for WebSocket and other services
export const API_CONFIG = {
  baseURL: ENV_CONFIG.apiBaseUrl,
  wsURL: ENV_CONFIG.wsUrl,
  apiKey: ENV_CONFIG.apiKey,
  timeout: ENV_CONFIG.defaultTimeout,
  retryAttempts: ENV_CONFIG.retryAttempts,
  retryDelayMs: ENV_CONFIG.retryDelayMs,
  allowApiKeyAuth: ENV_CONFIG.allowApiKeyAuth,
  isProduction: ENV_CONFIG.isProduction,
};

// Import types for API functions

/**
 * Fetches comprehensive results data for a given experiment ID.
 * @param experimentId The ID of the experiment to fetch results for.
 * @returns A Promise that resolves to ResultsData.
 */
export async function fetchResultsData(experimentId: string): Promise<ResultsData> {
  console.log(`[API Service] Fetching results for experiment: ${experimentId}`);
  
  try {
    const response = await apiService.get<ResultsData>(`/experiments/${experimentId}/results`);
    return response.data;
  } catch (error) {
    console.error(`[API Service] Failed to fetch results for experiment ${experimentId}:`, error);
    
    // Re-throw the error after logging
    throw error;
  }
}

/**
 * Fetches details for a specific experiment.
 * @param experimentId The ID of the experiment to fetch details for.
 * @returns A Promise that resolves to ExperimentExecution.
 */
export async function fetchExperimentDetails(experimentId: string): Promise<ExperimentExecution> {
    console.log(`[API Service] Fetching details for experiment: ${experimentId}`);
    
    try {
        const response = await apiService.get<ExperimentExecution>(`/experiments/${experimentId}`);
        return response.data;
    } catch (error) {
        console.error(`[API Service] Failed to fetch details for experiment ${experimentId}:`, error);
        
        // Re-throw the error after logging
        throw error;
    }
}

/**
 * Starts a new experiment with the given configuration.
 * @param experimentConfig The configuration for the experiment.
 * @returns A Promise that resolves to the created experiment.
 */
export async function startExperiment(experimentConfig: unknown): Promise<ExperimentExecution> {
  console.log(`[API Service] Starting new experiment`);
  
  try {
    const response = await apiService.post<ExperimentExecution>('/experiments', experimentConfig);
    return response.data;
  } catch (error) {
    console.error(`[API Service] Failed to start experiment:`, error);
    
    // Re-throw the error after logging
    throw error;
  }
}

/**
 * Stops a running experiment.
 * @param experimentId The ID of the experiment to stop.
 * @returns A Promise that resolves when the experiment is stopped.
 */
export async function stopExperiment(experimentId: string): Promise<void> {
  console.log(`[API Service] Stopping experiment: ${experimentId}`);
  
  try {
    await apiService.post<void>(`/experiments/${experimentId}/stop`);
  } catch (error) {
    console.error(`[API Service] Failed to stop experiment ${experimentId}:`, error);
    
    // Re-throw the error after logging
    throw error;
  }
}

/**
 * Fetches the list of all experiments.
 * @returns A Promise that resolves to an array of experiment summaries.
 */
export async function fetchExperimentList(): Promise<ExperimentExecution[]> {
  console.log(`[API Service] Fetching experiment list`);
  
  try {
    const response = await apiService.get<ExperimentExecution[]>('/experiments');
    return response.data;
  } catch (error) {
    console.error(`[API Service] Failed to fetch experiment list:`, error);
    
    // Re-throw the error after logging
    throw error;
  }
}

/**
 * Deletes an experiment.
 * @param experimentId The ID of the experiment to delete.
 * @returns A Promise that resolves when the experiment is deleted.
 */
export async function deleteExperiment(experimentId: string): Promise<void> {
  console.log(`[API Service] Deleting experiment: ${experimentId}`);
  
  try {
    await apiService.delete<void>(`/experiments/${experimentId}`);
  } catch (error) {
    console.error(`[API Service] Failed to delete experiment ${experimentId}:`, error);
    
    // Re-throw the error after logging
    throw error;
  }
}

/**
 * Interface for application settings
 */
export interface AppSettings {
  apiKeys: {
    openai: string;
    anthropic: string;
    google: string;
    cohere: string;
    openrouter: string;
  };
  defaults: {
    defaultLLM: string;
    defaultScenario: string;
    defaultAgent: string;
    defaultMetrics: string[];
    autoSave: boolean;
    notifications: boolean;
  };
  ui: {
    theme: 'light' | 'dark' | 'system';
    language: string;
    timezone: string;
  };
}

/**
 * Interface for settings response from API
 */
interface SettingsResponse {
  settings: AppSettings;
  message?: string;
}

/**
 * Fetches application settings
 * @returns A Promise that resolves to the application settings
 */
export async function fetchSettings(): Promise<AppSettings> {
  console.log(`[API Service] Fetching application settings`);
  
  try {
    const response = await apiService.get<SettingsResponse>('/api/v1/settings');
    return response.data.settings;
  } catch (error) {
    console.error(`[API Service] Failed to fetch settings:`, error);
    // Return default settings if fetch fails
    return {
      apiKeys: {
        openai: '',
        anthropic: '',
        google: '',
        cohere: '',
        openrouter: '',
      },
      defaults: {
        defaultLLM: 'gpt-4',
        defaultScenario: 'standard',
        defaultAgent: 'basic',
        defaultMetrics: ['revenue', 'profit', 'costs'],
        autoSave: true,
        notifications: true,
      },
      ui: {
        theme: 'system',
        language: 'en',
        timezone: Intl.DateTimeFormat().resolvedOptions().timeZone,
      },
    };
  }
}

/**
 * Saves application settings
 * @param settings The settings to save
 * @returns A Promise that resolves when settings are saved
 */
export async function saveSettings(settings: AppSettings): Promise<void> {
  console.log(`[API Service] Saving application settings`);
  
  try {
    await apiService.post<void>('/api/v1/settings', settings);
  } catch (error) {
    console.error(`[API Service] Failed to save settings:`, error);
    
    // Re-throw the error after logging
    throw error;
  }
}

/**
 * Interface for benchmark configuration request
 */
export interface BenchmarkConfigRequest {
  simulationSettings: Record<string, unknown>;
  agentConfigs: Record<string, unknown>[];
  llmSettings: Record<string, unknown>;
  constraints: Record<string, unknown>;
  experimentSettings: Record<string, unknown>;
}

/**
 * Interface for benchmark configuration response
 */
export interface BenchmarkConfigResponse {
  success: boolean;
  message: string;
  config_id: string;
  created_at: string;
}

/**
 * KPI Dashboard Metrics: Types and Clients
 * Strongly-typed interfaces and robust client functions for:
 * - Audit stats
 * - Ledger snapshot
 * - BSR indices
 * - Fee summary by type
 *
 * Notes:
 * - Uses existing ApiService fetch wrapper for retries/timeouts/headers.
 * - Accept both enveloped ApiResponse<T> and raw JSON payloads from backend.
 * - Coerce indices that may arrive as strings to numbers; keep money values as strings.
 * - Coarse shape validation with safe defaults to avoid tight coupling.
 */

// 1) Audit metrics
export interface AuditCurrentPosition {
  total_assets: string; // Money string like "$123.45"
  total_liabilities: string;
  total_equity: string;
  accounting_identity_valid: boolean;
  identity_difference: string; // Money string
}

export interface AuditStatsResponse {
  processed_transactions: number;
  total_violations: number;
  critical_violations: number;
  total_revenue_audited: string; // Money string
  total_fees_audited: string;    // Money string
  total_profit_audited: string;  // Money string
  current_position: AuditCurrentPosition;
  audit_enabled: boolean;
  halt_on_violation: boolean;
  tolerance_cents: number;
}

// 2) Ledger snapshot
export interface LedgerSnapshot {
  cash: string;
  inventory_value: string;
  accounts_receivable: string;
  accounts_payable: string;
  accrued_liabilities: string;
  total_assets: string;
  total_liabilities: string;
  total_equity: string;
  accounting_identity_valid: boolean;
  identity_difference: string;
  timestamp: string; // ISO
}

// 3) BSR indices
export interface BsrProductIndices {
  asin: string;
  velocity_index: number | null;
  conversion_index: number | null;
  composite_index: number | null;
}
export interface BsrSnapshot {
  products: BsrProductIndices[];
  market_ema_velocity?: string;     // number as string or numeric; kept flexible
  market_ema_conversion?: string;   // same
  competitor_count?: number;
  timestamp?: string;
}

// 4) Fee summary by type
export interface FeeTypeSummary {
  total_amount: string; // Money string
  count: number;
  average_amount: string; // Money string
}
export type FeeSummaryByType = Record<string, FeeTypeSummary>;

// ---- Helpers (internal to this module) ----
const toNumberOrNull = (v: unknown): number | null => {
  if (v === null || v === undefined) return null;
  if (typeof v === 'number' && isFinite(v)) return v;
  if (typeof v === 'string') {
    const n = Number(v);
    return Number.isFinite(n) ? n : null;
  }
  return null;
};
const toNumber = (v: unknown, fallback = 0): number => {
  const n = toNumberOrNull(v);
  return n === null ? fallback : n;
};
const toBoolean = (v: unknown, fallback = false): boolean => {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'string') {
    const s = v.toLowerCase().trim();
    if (s === 'true') return true;
    if (s === 'false') return false;
  }
  if (typeof v === 'number') return v !== 0;
  return fallback;
};
const toMoneyString = (v: unknown, fallback = '$0.00'): string => {
  if (typeof v === 'string' && v.length > 0) return v;
  return fallback;
};
const ensureObject = (v: unknown): Record<string, unknown> => (v && typeof v === 'object' ? (v as Record<string, unknown>) : {});

// Extract payload whether backend returns ApiResponse<T> or raw object
const unwrapData = (resp: unknown): unknown => {
  if (resp && typeof resp === 'object' && 'data' in (resp as Record<string, unknown>)) {
    return (resp as Record<string, unknown>).data;
  }
  return resp;
};

// ---- Clients ----

/**
 * GET /api/metrics/audit
 */
export async function getAuditStats(): Promise<AuditStatsResponse> {
  // Use existing ApiService for headers, retry, timeout. Explicit Accept header.
  const resp = await apiService.get<unknown>('/api/metrics/audit', { headers: { Accept: 'application/json' } });
  const raw = unwrapData(resp);
  const obj = ensureObject(raw);

  const currentPosObj = ensureObject(obj.current_position);
  const current_position: AuditCurrentPosition = {
    total_assets: toMoneyString(currentPosObj.total_assets),
    total_liabilities: toMoneyString(currentPosObj.total_liabilities),
    total_equity: toMoneyString(currentPosObj.total_equity),
    accounting_identity_valid: toBoolean(currentPosObj.accounting_identity_valid, false),
    identity_difference: toMoneyString(currentPosObj.identity_difference),
  };

  const result: AuditStatsResponse = {
    processed_transactions: toNumber(obj.processed_transactions, 0),
    total_violations: toNumber(obj.total_violations, 0),
    critical_violations: toNumber(obj.critical_violations, 0),
    total_revenue_audited: toMoneyString(obj.total_revenue_audited),
    total_fees_audited: toMoneyString(obj.total_fees_audited),
    total_profit_audited: toMoneyString(obj.total_profit_audited),
    current_position,
    audit_enabled: toBoolean(obj.audit_enabled, false),
    halt_on_violation: toBoolean(obj.halt_on_violation, false),
    tolerance_cents: toNumber(obj.tolerance_cents, 0),
  };

  return result;
}

/**
 * GET /api/metrics/ledger
 */
export async function getLedgerSnapshot(): Promise<LedgerSnapshot> {
  const resp = await apiService.get<unknown>('/api/metrics/ledger', { headers: { Accept: 'application/json' } });
  const raw = unwrapData(resp);
  const obj = ensureObject(raw);

  const result: LedgerSnapshot = {
    cash: toMoneyString(obj.cash),
    inventory_value: toMoneyString(obj.inventory_value),
    accounts_receivable: toMoneyString(obj.accounts_receivable),
    accounts_payable: toMoneyString(obj.accounts_payable),
    accrued_liabilities: toMoneyString(obj.accrued_liabilities),
    total_assets: toMoneyString(obj.total_assets),
    total_liabilities: toMoneyString(obj.total_liabilities),
    total_equity: toMoneyString(obj.total_equity),
    accounting_identity_valid: toBoolean(obj.accounting_identity_valid, false),
    identity_difference: toMoneyString(obj.identity_difference),
    timestamp: typeof obj.timestamp === 'string' && obj.timestamp ? obj.timestamp : new Date().toISOString(),
  };

  return result;
}

/**
 * GET /api/metrics/bsr
 */
export async function getBsrSnapshot(): Promise<BsrSnapshot> {
  const resp = await apiService.get<unknown>('/api/metrics/bsr', { headers: { Accept: 'application/json' } });
  const raw = unwrapData(resp);
  const obj = ensureObject(raw);

  const productsRaw = Array.isArray((obj as Record<string, unknown>).products) ? (obj as Record<string, unknown>).products as unknown[] : [];
  const products: BsrProductIndices[] = productsRaw.map((p) => {
    const po = ensureObject(p);
    return {
      asin: typeof po.asin === 'string' ? po.asin : '',
      velocity_index: toNumberOrNull(po.velocity_index),
      conversion_index: toNumberOrNull(po.conversion_index),
      composite_index: toNumberOrNull(po.composite_index),
    };
  });

  const toOptionalString = (v: unknown): string | undefined => {
    if (v === null || v === undefined) return undefined;
    if (typeof v === 'string') return v;
    if (typeof v === 'number' && Number.isFinite(v)) return String(v);
    return undefined;
  };

  const cc = toNumberOrNull((obj as Record<string, unknown>).competitor_count);
  const result: BsrSnapshot = {
    products,
    market_ema_velocity: toOptionalString((obj as Record<string, unknown>).market_ema_velocity),
    market_ema_conversion: toOptionalString((obj as Record<string, unknown>).market_ema_conversion),
    competitor_count: cc === null ? undefined : cc,
    timestamp: typeof (obj as Record<string, unknown>).timestamp === 'string'
      ? (obj as Record<string, unknown>).timestamp as string
      : undefined,
  };

  return result;
}

/**
 * GET /api/metrics/fees
 */
export async function getFeeSummary(): Promise<FeeSummaryByType> {
  const resp = await apiService.get<unknown>('/api/metrics/fees', { headers: { Accept: 'application/json' } });
  const raw = unwrapData(resp);

  const summary: FeeSummaryByType = {};

  const obj = ensureObject(raw);
  for (const [k, v] of Object.entries(obj)) {
    const vo = ensureObject(v);
    const count = toNumber(vo.count, 0);
    const entry: FeeTypeSummary = {
      total_amount: toMoneyString(vo.total_amount),
      count,
      average_amount: toMoneyString(vo.average_amount),
    };
    summary[k] = entry;
  }

  return summary;
}