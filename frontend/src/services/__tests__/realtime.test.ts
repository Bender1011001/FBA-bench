import { describe, it, expect, beforeEach, vi } from 'vitest';
import { connectRealtime, SIMULATION_TOPIC_PREFIX, EXPERIMENT_TOPIC_PREFIX } from '../realtime';

class MockWebSocket {
  static instances: MockWebSocket[] = [];
  static OPEN = 1;
  static CONNECTING = 0;
  readyState = MockWebSocket.OPEN;
  url: string;
  onopen: (() => void) | null = null;
  onmessage: ((ev: MessageEvent) => void) | null = null;
  onclose: (() => void) | null = null;
  onerror: (() => void) | null = null;
  sent: string[] = [];
  constructor(url: string) {
    this.url = url;
    MockWebSocket.instances.push(this);
    // simulate async open
    setTimeout(() => this.onopen && this.onopen(), 0);
  }
  send(data: string) {
    this.sent.push(data);
  }
  close() {
    this.readyState = 3;
    if (this.onclose) {
      this.onclose();
    }
  }
}

declare global {
  interface Window {
    WebSocket: typeof MockWebSocket;
  }
}

describe('realtime client', () => {
  const originalWS = (globalThis.window?.WebSocket as unknown) as typeof MockWebSocket | undefined;

  beforeEach(() => {
    MockWebSocket.instances = [];
    // @ts-expect-error override for test
    window.WebSocket = MockWebSocket;
    vi.useFakeTimers();
  });

  it('subscribes and publishes frames', () => {
    const client = connectRealtime('ws://localhost:8000/ws/realtime');
    const handler = vi.fn();

    client.subscribe(`${SIMULATION_TOPIC_PREFIX}abc`, handler);

    // simulate open + resubscribe flush
    vi.runOnlyPendingTimers();

    const ws = MockWebSocket.instances[0];
    expect(ws).toBeDefined();

    // check subscribe frame was sent eventually
    const subscribeFrame = ws.sent.find((s) => s.includes('"action":"subscribe"'));
    expect(subscribeFrame).toBeTruthy();

    // publish
    client.publish(`${EXPERIMENT_TOPIC_PREFIX}abc`, { hello: 'world' });
    const publishFrame = ws.sent.find((s) => s.includes('"action":"publish"'));
    expect(publishFrame).toBeTruthy();

    // incoming message dispatch
    ws.onmessage?.({ data: JSON.stringify({ topic: `${SIMULATION_TOPIC_PREFIX}abc`, payload: { status: 'running' } }) } as MessageEvent);
    expect(handler).toHaveBeenCalled();
    client.close();
  });

  it('sends heartbeat pings and reconnects after close', () => {
    const client = connectRealtime('ws://localhost:8000/ws/realtime');
    vi.runOnlyPendingTimers();
    const ws = MockWebSocket.instances[0];

    // heartbeat interval 25s
    vi.advanceTimersByTime(25_000);
    const pingSent = ws.sent.find((s) => s.includes('"action":"ping"'));
    expect(pingSent).toBeTruthy();

    // force close -> schedule reconnect
    ws.close();
    // backoff 1000ms first
    vi.advanceTimersByTime(1000);
    expect(MockWebSocket.instances.length).toBeGreaterThan(1);

    client.close();
  });

  afterEach(() => {
    vi.useRealTimers();
    if (originalWS) {
      (window as unknown as { WebSocket: typeof MockWebSocket }).WebSocket = originalWS as unknown as typeof MockWebSocket;
    }
  });
});