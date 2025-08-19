/**
 * Thin realtime WebSocket client with:
 * - subscribe/unsubscribe/publish protocol
 * - heartbeat ping every 25s
 * - auto-reconnect with exponential backoff
 * - topic-based handler registry
 * - no-op when WebSocket is unavailable
 *
 * Protocol frames:
 * { action: "subscribe" | "unsubscribe" | "publish" | "ping"; topic?: string; payload?: unknown }
 *
 * Environment:
 * - VITE_REALTIME_URL (preferred) e.g. ws://localhost:8000/ws/realtime
 * - Fallback to VITE_WS_URL + '/ws/realtime' when provided
 */

export type RealtimeFrame =
  | { action: 'subscribe'; topic: string }
  | { action: 'unsubscribe'; topic: string }
  | { action: 'publish'; topic: string; payload: unknown }
  | { action: 'ping' };

export interface RealtimeClient {
  subscribe(topic: string, handler: (msg: unknown) => void): void;
  unsubscribe(topic: string): void;
  publish(topic: string, payload: unknown): void;
  close(): void;
}

export const SIMULATION_TOPIC_PREFIX = 'simulation:';
export const EXPERIMENT_TOPIC_PREFIX = 'experiment:';

type Handler = (msg: unknown) => void;

interface InternalState {
  url: string;
  ws: WebSocket | null;
  open: boolean;
  closing: boolean;
  heartbeatTimer: number | null;
  reconnectTimer: number | null;
  backoffAttempt: number;
  topics: Set<string>;
  handlers: Map<string, Handler>;
}

const HEARTBEAT_INTERVAL_MS = 25_000; // 25s
const MAX_BACKOFF_MS = 30_000;

function getRealtimeUrl(): string | null {
  const env = (import.meta as unknown as { env: Record<string, string | undefined> }).env;
  // Preferred explicit URL including /ws/realtime path
  const explicit = env?.VITE_REALTIME_URL;
  if (explicit) return explicit;
  // Fallback: build from base WS if provided
  const base = env?.VITE_WS_URL;
  if (base) {
    const trimmed = base.endsWith('/') ? base.slice(0, -1) : base;
    return `${trimmed}/ws/realtime`;
  }
  return null;
}

export function connectRealtime(url?: string): RealtimeClient {
  // Graceful no-op if WebSocket not available
  if (typeof window === 'undefined' || typeof window.WebSocket === 'undefined') {
    return {
      subscribe: () => {},
      unsubscribe: () => {},
      publish: () => {},
      close: () => {},
    };
  }

  const resolvedUrl = url ?? getRealtimeUrl();
  if (!resolvedUrl) {
    // No configured realtime endpoint; behave as no-op client
    return {
      subscribe: () => {},
      unsubscribe: () => {},
      publish: () => {},
      close: () => {},
    };
  }

  const state: InternalState = {
    url: resolvedUrl,
    ws: null,
    open: false,
    closing: false,
    heartbeatTimer: null,
    reconnectTimer: null,
    backoffAttempt: 0,
    topics: new Set<string>(),
    handlers: new Map<string, Handler>(),
  };

  const startHeartbeat = () => {
    stopHeartbeat();
    state.heartbeatTimer = window.setInterval(() => {
      send({ action: 'ping' });
    }, HEARTBEAT_INTERVAL_MS);
  };

  const stopHeartbeat = () => {
    if (state.heartbeatTimer !== null) {
      clearInterval(state.heartbeatTimer);
      state.heartbeatTimer = null;
    }
  };

  const scheduleReconnect = () => {
    if (state.closing) return;
    if (state.reconnectTimer !== null) return;
    const delay = Math.min(MAX_BACKOFF_MS, Math.floor(1000 * Math.pow(2, state.backoffAttempt)));
    state.reconnectTimer = window.setTimeout(() => {
      state.reconnectTimer = null;
      state.backoffAttempt = Math.min(state.backoffAttempt + 1, 10);
      openSocket();
    }, delay);
  };

  const send = (frame: RealtimeFrame) => {
    if (state.open && state.ws && state.ws.readyState === WebSocket.OPEN) {
      try {
        state.ws.send(JSON.stringify(frame));
      } catch {
        // drop frame if send fails
      }
    }
  };

  const resubscribeAll = () => {
    state.topics.forEach((t) => send({ action: 'subscribe', topic: t }));
  };

  const handleIncoming = (ev: MessageEvent) => {
    try {
      const data = JSON.parse(ev.data as string);
      // Expect messages in the shape { topic, payload } or { event, ... }
      const topic: string | undefined = data?.topic;
      if (topic && state.handlers.has(topic)) {
        const handler = state.handlers.get(topic)!;
        handler(data.payload ?? data);
      } else if (data?.event && typeof data.event === 'string') {
        // Some servers emit event-based messages; forward to potential handlers keyed by event
        const evtTopic = String(data.event);
        const handler = state.handlers.get(evtTopic);
        if (handler) handler(data);
      }
    } catch {
      // ignore malformed frames
    }
  };

  const openSocket = () => {
    try {
      const ws = new WebSocket(state.url);
      state.ws = ws;

      ws.onopen = () => {
        state.open = true;
        state.backoffAttempt = 0;
        startHeartbeat();
        resubscribeAll();
      };

      ws.onmessage = handleIncoming;

      ws.onclose = () => {
        state.open = false;
        stopHeartbeat();
        if (!state.closing) {
          scheduleReconnect();
        }
      };

      ws.onerror = () => {
        // will be followed by onclose in most implementations
        try {
          ws.close();
        } catch {
          /* no-op */
        }
      };
    } catch {
      scheduleReconnect();
    }
  };

  openSocket();

  const client: RealtimeClient = {
    subscribe(topic: string, handler: Handler) {
      if (!topic || typeof handler !== 'function') return;
      state.topics.add(topic);
      state.handlers.set(topic, handler);
      if (state.open) {
        send({ action: 'subscribe', topic });
      }
    },
    unsubscribe(topic: string) {
      state.topics.delete(topic);
      state.handlers.delete(topic);
      if (state.open) {
        send({ action: 'unsubscribe', topic });
      }
    },
    publish(topic: string, payload: unknown) {
      if (!topic) return;
      send({ action: 'publish', topic, payload });
    },
    close() {
      state.closing = true;
      stopHeartbeat();
      if (state.reconnectTimer !== null) {
        clearTimeout(state.reconnectTimer);
        state.reconnectTimer = null;
      }
      if (state.ws && (state.ws.readyState === WebSocket.OPEN || state.ws.readyState === WebSocket.CONNECTING)) {
        try {
          state.ws.close();
        } catch {
          /* no-op */
        }
      }
      state.ws = null;
      state.open = false;
      state.handlers.clear();
      state.topics.clear();
    },
  };

  return client;
}

export default connectRealtime;