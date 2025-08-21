// Centralized frontend configuration for API/Realtime endpoints and app defaults.
// Sources environment variables (Vite) and provides sane fallbacks for local dev.
// This decouples service/feature code from direct env access.

export type AppConfig = {
  apiBaseUrl: string;
  wsUrl: string;
  realtimeUrl: string;
  apiKey: string;
  defaultTimeout: number;
  retryAttempts: number;
  retryDelayMs: number;
  allowApiKeyAuth: boolean;
  isProduction: boolean;
};

function toNumber(envVal: string | undefined, fallback: number): number {
  if (!envVal) return fallback;
  const n = Number(envVal);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

function toBoolean(envVal: string | undefined, fallback: boolean): boolean {
  if (!envVal) return fallback;
  const v = envVal.trim().toLowerCase();
  return v === 'true' || v === '1' || v === 'yes' || v === 'on';
}

const apiBaseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const wsUrl = import.meta.env.VITE_WS_URL || 'ws://localhost:8000';
const realtimeUrl =
  import.meta.env.VITE_REALTIME_URL || `${wsUrl.replace(/\/+$/, '')}/ws/realtime`;

export const appConfig: AppConfig = {
  apiBaseUrl,
  wsUrl,
  realtimeUrl,
  apiKey: import.meta.env.VITE_API_KEY || '',
  defaultTimeout: toNumber(import.meta.env.VITE_API_TIMEOUT, 15000),
  retryAttempts: toNumber(import.meta.env.VITE_API_RETRY_ATTEMPTS, 3),
  retryDelayMs: toNumber(import.meta.env.VITE_API_RETRY_DELAY_MS, 1000),
  allowApiKeyAuth: toBoolean(import.meta.env.VITE_ALLOW_API_KEY_AUTH, false),
  isProduction: import.meta.env.PROD === true,
};

// Optional: runtime override for testing or admin consoles.
export function overrideAppConfig(partial: Partial<AppConfig>): void {
  Object.assign(appConfig, partial);
}