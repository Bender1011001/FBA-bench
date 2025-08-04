import { API_CONFIG } from './apiService';

interface WebSocketMessage {
  type: 'execution_update' | 'metrics_update' | 'benchmark_complete' | 'error' | 'heartbeat';
  data: unknown;
  timestamp: string;
  execution_id?: string;
}

interface WebSocketCallbacks {
  onExecutionUpdate?: (data: ExecutionProgress) => void;
  onMetricsUpdate?: (data: MultiDimensionalMetric[]) => void;
  onBenchmarkComplete?: (data: BenchmarkResult) => void;
  onError?: (error: Error) => void;
  onHeartbeat?: () => void;
}

interface Subscription {
  executionId: string;
  callbacks: WebSocketCallbacks;
}

class WebSocketService {
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private subscriptions: Map<string, Subscription> = new Map();
  private isConnecting = false;
  private heartbeatInterval: number | null = null;
  private connectionTimeout: number | null = null;

  constructor() {
    this.connect();
  }

  private connect(): void {
    if (this.isConnecting || this.ws?.readyState === WebSocket.OPEN) {
      return;
    }

    this.isConnecting = true;
    
    try {
      const wsUrl = `${API_CONFIG.wsURL.replace('http', 'ws')}/ws/benchmarking`;
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);

      // Set connection timeout
      this.connectionTimeout = setTimeout(() => {
        if (this.ws?.readyState !== WebSocket.OPEN) {
          this.ws?.close();
          this.handleConnectionError(new Error('Connection timeout'));
        }
      }, 10000);

    } catch (error) {
      this.handleConnectionError(error as Error);
    }
  }

  private handleOpen(): void {
    console.log('WebSocket connection established');
    this.isConnecting = false;
    this.reconnectAttempts = 0;
    
    // Clear connection timeout
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }

    // Start heartbeat
    this.startHeartbeat();

    // Resubscribe to all active subscriptions
    this.resubscribeToAll();
  }

  private handleMessage(event: MessageEvent): void {
    try {
      const message: WebSocketMessage = JSON.parse(event.data);
      this.processMessage(message);
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      this.notifyError(new Error('Invalid WebSocket message format'));
    }
  }

  private processMessage(message: WebSocketMessage): void {
    switch (message.type) {
      case 'execution_update':
        this.handleExecutionUpdate(message);
        break;
      case 'metrics_update':
        this.handleMetricsUpdate(message);
        break;
      case 'benchmark_complete':
        this.handleBenchmarkComplete(message);
        break;
      case 'heartbeat':
        this.handleHeartbeat();
        break;
      case 'error':
        this.notifyError(new Error((message.data as { message?: string })?.message || 'Unknown error'));
        break;
      default:
        console.warn('Unknown WebSocket message type:', message.type);
    }
  }

  private handleExecutionUpdate(message: WebSocketMessage): void {
    const executionId = message.execution_id;
    if (!executionId) return;

    const subscription = this.subscriptions.get(executionId);
    if (subscription?.callbacks.onExecutionUpdate) {
      subscription.callbacks.onExecutionUpdate(message.data as ExecutionProgress);
    }
  }

  private handleMetricsUpdate(message: WebSocketMessage): void {
    const executionId = message.execution_id;
    if (!executionId) return;

    const subscription = this.subscriptions.get(executionId);
    if (subscription?.callbacks.onMetricsUpdate) {
      subscription.callbacks.onMetricsUpdate(message.data as MultiDimensionalMetric[]);
    }
  }

  private handleBenchmarkComplete(message: WebSocketMessage): void {
    const executionId = message.execution_id;
    if (!executionId) return;

    const subscription = this.subscriptions.get(executionId);
    if (subscription?.callbacks.onBenchmarkComplete) {
      subscription.callbacks.onBenchmarkComplete(message.data as BenchmarkResult);
    }

    // Keep subscription for a while after completion for potential final updates
    setTimeout(() => {
      this.unsubscribe(executionId);
    }, 30000);
  }

  private handleHeartbeat(): void {
    // Reset reconnect attempts on successful heartbeat
    this.reconnectAttempts = 0;
    
    const allSubscriptions = Array.from(this.subscriptions.values());
    allSubscriptions.forEach(subscription => {
      if (subscription.callbacks.onHeartbeat) {
        subscription.callbacks.onHeartbeat();
      }
    });
  }

  private handleError(event: Event): void {
    console.error('WebSocket error:', event);
    this.notifyError(new Error('WebSocket connection error'));
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocket connection closed:', event.code, event.reason);
    this.isConnecting = false;
    
    // Clear heartbeat
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    // Clear connection timeout
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }

    // Attempt to reconnect if not closed intentionally
    if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    }
  }

  private handleConnectionError(error: Error): void {
    console.error('WebSocket connection error:', error);
    this.isConnecting = false;
    this.notifyError(error);
    
    // Attempt to reconnect
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++;
      const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          type: 'heartbeat',
          timestamp: new Date().toISOString()
        }));
      }
    }, 30000); // Send heartbeat every 30 seconds
  }

  private resubscribeToAll(): void {
    const allSubscriptions = Array.from(this.subscriptions.entries());
    allSubscriptions.forEach(([executionId, subscription]) => {
      this.sendSubscriptionRequest(executionId, subscription.callbacks);
    });
  }

  private sendSubscriptionRequest(executionId: string, callbacks: WebSocketCallbacks): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        execution_id: executionId,
        callback_types: Object.keys(callbacks).filter(key => callbacks[key as keyof WebSocketCallbacks] !== undefined)
      }));
    }
  }

  private notifyError(error: Error): void {
    const allSubscriptions = Array.from(this.subscriptions.values());
    allSubscriptions.forEach(subscription => {
      if (subscription.callbacks.onError) {
        subscription.callbacks.onError(error);
      }
    });
  }

  // Public API
  subscribe(
    executionId: string,
    callbacks: WebSocketCallbacks
  ): void {
    // Remove existing subscription for this execution ID
    this.unsubscribe(executionId);

    // Add new subscription
    this.subscriptions.set(executionId, {
      executionId,
      callbacks
    });

    // Send subscription request if connected
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.sendSubscriptionRequest(executionId, callbacks);
    }
  }

  unsubscribe(executionId: string): void {
    this.subscriptions.delete(executionId);
    
    // Send unsubscribe request if connected
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'unsubscribe',
        execution_id: executionId
      }));
    }
  }

  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  getConnectionStatus(): 'connecting' | 'connected' | 'disconnected' | 'reconnecting' {
    if (this.isConnecting) return 'connecting';
    if (this.ws?.readyState === WebSocket.OPEN) return 'connected';
    if (this.reconnectAttempts > 0) return 'reconnecting';
    return 'disconnected';
  }

  // Send custom message to WebSocket server
  send(message: Omit<WebSocketMessage, 'timestamp'>): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        ...message,
        timestamp: new Date().toISOString()
      }));
    } else {
      console.warn('WebSocket not connected, cannot send message');
    }
  }

  // Force reconnect
  reconnect(): void {
    if (this.ws) {
      this.ws.close();
    }
    this.connect();
  }

  // Cleanup
  disconnect(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }

    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }

    if (this.ws) {
      this.ws.close(1000, 'Client disconnect');
      this.ws = null;
    }

    this.subscriptions.clear();
    this.reconnectAttempts = 0;
    this.isConnecting = false;
  }
}

// Export singleton instance
export const webSocketService = new WebSocketService();

// Export types for convenience
export type { WebSocketMessage, WebSocketCallbacks, Subscription };

// Import required types
import type { ExecutionProgress, MultiDimensionalMetric, BenchmarkResult } from '../types';