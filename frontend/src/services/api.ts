/**
 * Minimal, production-ready API client with:
 * - Base URL from env
 * - AbortController support
 * - JSON parsing with error normalization
 * - Timeouts and robust error typing
 */

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';

export type ApiErrorCategory =
  | 'network'
  | 'timeout'
  | 'abort'
  | 'http'
  | 'parse'
  | 'unknown';

export interface NormalizedApiError extends Error {
  category: ApiErrorCategory;
  statusCode?: number;
  details?: unknown;
}

export interface RequestOptions extends Omit<RequestInit, 'method' | 'signal' | 'body'> {
  method?: HttpMethod;
  /**
   * Plain JS object will be serialized as JSON. If FormData, it is sent as-is.
   */
  body?: Record<string, unknown> | FormData | string;
  /**
   * Milliseconds before request is aborted. Default: 15000
   */
  timeoutMs?: number;
  /**
   * Optional signal to piggyback cancellation
   */
  signal?: AbortSignal;
  /**
   * Additional headers to merge (authorization, etc.)
   */
  headers?: HeadersInit;
}

export interface ApiResponse<T = unknown> {
  ok: boolean;
  status: number;
  data: T;
  headers: Headers;
}

/**
 * Base URLs (Vite-style)
 */
type ViteImportMeta = ImportMeta & { env: Record<string, string | undefined> };
const viteMeta = import.meta as ViteImportMeta;
const BASE_URL: string = viteMeta.env?.VITE_API_BASE_URL ?? '/api';
const DEFAULT_TIMEOUT_MS = 15000;

/**
 * Optional Authorization token provider set by AuthContext (mock scaffold).
 * If provided and returns a non-empty string, apiFetch will include:
 *   Authorization: Bearer <token>
 */
let authTokenProvider: (() => string | null) | null = null;

/**
 * Allow external code (AuthContext) to register a token getter.
 * Pass undefined or call with no args to reset to null.
 */
export function setAuthTokenProvider(getter?: () => string | null): void {
  authTokenProvider = getter ?? null;
}

/**
 * Serialize a body intelligently:
 * - If FormData: send as-is
 * - If string: send as-is (assumed pre-serialized)
 * - If object: JSON.stringify with appropriate headers
 */
function buildBodyAndHeaders(input: RequestOptions) {
  const baseHeaders: Record<string, string> = {
    Accept: 'application/json',
  };

  if (input.body instanceof FormData) {
    return { body: input.body, headers: { ...baseHeaders, ...(input.headers ?? {}) } };
  }

  if (typeof input.body === 'string') {
    return { body: input.body, headers: { ...baseHeaders, ...(input.headers ?? {}) } };
  }

  if (typeof input.body === 'object' && input.body !== null) {
    return {
      body: JSON.stringify(input.body),
      headers: { 'Content-Type': 'application/json', ...baseHeaders, ...(input.headers ?? {}) },
    };
  }

  return { body: undefined, headers: { ...baseHeaders, ...(input.headers ?? {}) } };
}

function normalizeError(e: unknown, statusCode?: number): NormalizedApiError {
  const err: NormalizedApiError = {
    name: 'RequestError',
    message: 'Request failed',
    category: 'unknown',
  };

  if (e instanceof DOMException && e.name === 'AbortError') {
    err.name = 'AbortError';
    err.category = 'abort';
    err.message = 'The request was aborted';
    return err;
  }

  if (e instanceof TypeError) {
    // Fetch network error typically appears as TypeError
    err.name = 'NetworkError';
    err.category = 'network';
    err.message = e.message || 'Network error';
    return err;
  }

  // If statusCode provided, classify as HTTP error
  if (typeof statusCode === 'number' && (statusCode < 200 || statusCode >= 300)) {
    err.name = 'HttpError';
    err.category = 'http';
    err.statusCode = statusCode;
    err.message = `HTTP ${statusCode}`;
    err.details = e;
    return err;
  }

  // Parsing/unknown
  err.name = 'UnknownError';
  err.category = 'unknown';
  err.message = e instanceof Error ? e.message : 'Unknown error';
  err.details = e;
  return err;
}

/**
 * Core fetch wrapper
 */
export async function apiFetch<T = unknown>(
  path: string,
  options: RequestOptions = {},
): Promise<ApiResponse<T>> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), options.timeoutMs ?? DEFAULT_TIMEOUT_MS);

  // Normalize headers and attach Authorization if provided
  const mergedHeaders: Record<string, string> = {};
  const src = options.headers;
  if (src instanceof Headers) {
    src.forEach((v, k) => {
      mergedHeaders[k] = v;
    });
  } else if (Array.isArray(src)) {
    for (const [k, v] of src) mergedHeaders[k] = v;
  } else if (src) {
    Object.assign(mergedHeaders, src as Record<string, string>);
  }
  const token = authTokenProvider?.() ?? null;
  if (token) {
    mergedHeaders['Authorization'] = `Bearer ${token}`;
  }

  const { body, headers } = buildBodyAndHeaders({ ...options, headers: mergedHeaders });

  // Merge external signal + our controller
  const signals = [controller.signal, options.signal].filter((s): s is AbortSignal => Boolean(s));
  const signal = signals.length === 1 ? signals[0] : anySignal(signals);

  const url = joinUrl(BASE_URL, path);
  try {
    const res = await fetch(url, {
      ...options,
      method: options.method ?? 'GET',
      headers,
      signal,
      body,
    });

    const status = res.status;
    const text = await res.text();

    // Try parse JSON
    let data: unknown = null;
    if (text) {
      try {
        data = JSON.parse(text) as unknown;
      } catch (parseError: unknown) {
        const err = normalizeError(parseError, status);
        err.name = 'ParseError';
        err.category = 'parse';
        err.message = 'Failed to parse JSON response';
        err.statusCode = status;
        err.details = text;
        throw err;
      }
    }

    if (!res.ok) {
      const err = normalizeError(data ?? text, status);
      throw err;
    }

    return { ok: true, status, data: data as T, headers: res.headers };
  } catch (e: unknown) {
    throw normalizeError(e);
  } finally {
    clearTimeout(timeout);
  }
}

/**
 * Helper to join base and relative URLs safely.
 */
function joinUrl(base: string, path: string): string {
  if (!base.endsWith('/') && !path.startsWith('/')) return `${base}/${path}`;
  if (base.endsWith('/') && path.startsWith('/')) return `${base}${path.slice(1)}`;
  return `${base}${path}`;
}

/**
 * Polyfill-ish multi-signal combinator: aborts when any sub-signal aborts.
 */
function anySignal(signals: AbortSignal[]): AbortSignal {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  for (const s of signals) {
    if (s.aborted) {
      controller.abort();
      break;
    }
    s.addEventListener('abort', onAbort, { once: true });
  }
  return controller.signal;
}

/**
 * Convenience JSON helpers
 */
export const http = {
  get: <T = unknown>(path: string, opts?: RequestOptions) => apiFetch<T>(path, { ...opts, method: 'GET' }),
  post: <T = unknown>(path: string, body?: RequestOptions['body'], opts?: RequestOptions) =>
    apiFetch<T>(path, { ...(opts ?? {}), method: 'POST', body }),
  put: <T = unknown>(path: string, body?: RequestOptions['body'], opts?: RequestOptions) =>
    apiFetch<T>(path, { ...(opts ?? {}), method: 'PUT', body }),
  patch: <T = unknown>(path: string, body?: RequestOptions['body'], opts?: RequestOptions) =>
    apiFetch<T>(path, { ...(opts ?? {}), method: 'PATCH', body }),
  delete: <T = unknown>(path: string, opts?: RequestOptions) => apiFetch<T>(path, { ...opts, method: 'DELETE' }),
};

export default apiFetch;
// Typed endpoint helpers and 404 normalization

import type {
  Experiment,
  Agent,
  Simulation,
  EngineReport,
  OkResponse,
} from '../types/api';

// Typed error to signal missing backend route while keeping UI resilient
export class RouteNotAvailableError extends Error {
  readonly code = 'route_not_available';
  readonly statusCode = 404;
  route: string;
  constructor(route: string, message?: string) {
    const msg = message ?? `API route not available: ${route}`;
    super(msg);
    this.name = 'RouteNotAvailableError';
    this.route = route;
  }
}

function normalize404(route: string, e: unknown): never {
  // Convert our normalized API error into a user-friendly typed error
  try {
    const err = e as NormalizedApiError;
    if (err?.category === 'http' && err?.statusCode === 404) {
      throw new RouteNotAvailableError(route);
    }
  } catch {
    // fallthrough
  }
  throw e as Error;
}

// Experiments
export async function getExperiments(signal?: AbortSignal): Promise<Experiment[]> {
  try {
    const res = await http.get<Experiment[]>('/experiments', { signal });
    return res.data;
  } catch (e) {
    return normalize404('/experiments', e);
  }
}

export interface CreateExperimentInput {
  name: string;
  description?: string;
}
export async function createExperiment(input: CreateExperimentInput, signal?: AbortSignal): Promise<Experiment> {
  try {
    const payload = input as unknown as Record<string, unknown>;
    const res = await http.post<Experiment>('/experiments', payload, { signal });
    return res.data;
  } catch (e) {
    return normalize404('/experiments', e);
  }
}

export async function getExperiment(id: string, signal?: AbortSignal): Promise<Experiment> {
  const route = `/experiments/${encodeURIComponent(id)}`;
  try {
    const res = await http.get<Experiment>(route, { signal });
    return res.data;
  } catch (e) {
    return normalize404(route, e);
  }
}

export async function startSimulation(experimentId: string, signal?: AbortSignal): Promise<Simulation> {
  const route = `/experiments/${encodeURIComponent(experimentId)}/simulations`;
  try {
    const res = await http.post<Simulation>(route, {}, { signal });
    return res.data;
  } catch (e) {
    return normalize404(route, e);
  }
}

// Agents
export async function getAgents(signal?: AbortSignal): Promise<Agent[]> {
  try {
    const res = await http.get<Agent[]>('/agents', { signal });
    return res.data;
  } catch (e) {
    return normalize404('/agents', e);
  }
}

export interface CreateAgentInput {
  name: string;
  runner: string;
  description?: string;
  config?: Record<string, unknown>;
}
export async function createAgent(input: CreateAgentInput, signal?: AbortSignal): Promise<Agent> {
  try {
    const payload = input as unknown as Record<string, unknown>;
    const res = await http.post<Agent>('/agents', payload, { signal });
    return res.data;
  } catch (e) {
    return normalize404('/agents', e);
  }
}

export interface UpdateAgentInput {
  name?: string;
  runner?: string;
  description?: string;
  config?: Record<string, unknown>;
}
export async function updateAgent(id: string, input: UpdateAgentInput, signal?: AbortSignal): Promise<Agent> {
  const route = `/agents/${encodeURIComponent(id)}`;
  try {
    const payload = input as unknown as Record<string, unknown>;
    const res = await http.put<Agent>(route, payload, { signal });
    return res.data;
  } catch (e) {
    return normalize404(route, e);
  }
}

export async function deleteAgent(id: string, signal?: AbortSignal): Promise<OkResponse> {
  const route = `/agents/${encodeURIComponent(id)}`;
  try {
    const res = await http.delete<OkResponse>(route, { signal });
    return res.data;
  } catch (e) {
    return normalize404(route, e);
  }
}

// Results
export async function getEngineReport(experimentId: string, signal?: AbortSignal): Promise<EngineReport> {
  const route = `/experiments/${encodeURIComponent(experimentId)}/report`;
  try {
    const res = await http.get<EngineReport>(route, { signal });
    return res.data;
  } catch (e) {
    return normalize404(route, e);
  }
}