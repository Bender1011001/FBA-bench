import { API_CONFIG } from './apiService';
import type { ExecutionProgress, MultiDimensionalMetric, BenchmarkResult, SystemHealth } from '../types';

// Type guard for SystemHealth
const isSystemHealth = (data: unknown): data is SystemHealth => {
  return typeof data === 'object' && data !== null && 
         'apiResponseTime' in data && typeof (data as SystemHealth).apiResponseTime === 'number' &&
         'wsConnectionStatus' in data && typeof (data as SystemHealth).wsConnectionStatus === 'string'; // Add more robust checks if needed
};

interface WebSocketMessage {
  type: 'execution_update' | 'metrics_update' | 'benchmark_complete' | 'error' | 'heartbeat' | 'connection_established' | 'snapshot' | 'event' | 'subscription_confirmed' | 'unsubscription_confirmed' | 'connection_stats' | 'pong' | 'system_health';
  data?: unknown; // General payload can be anything
  timestamp: string;
  execution_id?: string; // Used for specific execution updates
  event_type?: string; // Used when type is 'event'
  message?: string; // For connection_established, error, etc.
  client_id?: string;
  origin?: string;
  payload?: unknown; // General payload from backend, used by KPIDashboard.tsx
  topics?: string[]; // For subscription confirmations
}

interface WebSocketCallbacks {
  onExecutionUpdate?: (data: ExecutionProgress) => void;
  onMetricsUpdate?: (data: MultiDimensionalMetric[]) => void;
  onBenchmarkComplete?: (data: BenchmarkResult) => void;
  onError?: (error: Error) => void;
  onHeartbeat?: () => void;
  onSnapshot?: (data: unknown) => void;
  onEvent?: (eventType: string, data: unknown) => void;
  onConnectionStatusChange?: (status: 'connecting' | 'connected' | 'disconnected' | 'reconnecting') => void;
  onConnectionEstablished?: (clientId: string, origin: string) => void;
  onSubscriptionConfirmed?: (topics: string[]) => void;
  onUnsubscriptionConfirmed?: (topics: string[]) => void;
  onConnectionStats?: (data: unknown) => void;
  onPingPong?: () => void;
  onSystemHealth?: (data: SystemHealth) => void;
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
      const wsUrl = `${API_CONFIG.wsURL.replace('http', 'ws')}/ws/events`; // Connect to /ws/events
      console.log('WebSocketService: API_CONFIG.wsURL:', API_CONFIG.wsURL);
      console.log('WebSocketService: Attempting to connect to:', wsUrl);
      this.ws = new WebSocket(wsUrl);

      this.ws.onopen = this.handleOpen.bind(this);
      this.ws.onmessage = this.handleMessage.bind(this);
      this.ws.onerror = this.handleError.bind(this);
      this.ws.onclose = this.handleClose.bind(this);

      // Set connection timeout
      this.connectionTimeout = setTimeout(() => {
        console.log('WebSocketService: Connection timeout reached');
        if (this.ws?.readyState !== WebSocket.OPEN) {
          this.ws?.close();
          this.handleConnectionError(new Error('Connection timeout'));
        }
      }, 10000);

    } catch (error) {
      console.error('WebSocketService: Error creating WebSocket connection:', error);
      this.handleConnectionError(error as Error);
    }
  }

  private handleOpen(): void {
    console.log('WebSocketService: Connection established successfully');
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
    this.notifyConnectionStatusChange('connected'); // Notify status change
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
    // Unified event handling
    switch (message.type) {
      case 'connection_established':
        console.log('WebSocketService: Connection established message:', message.message);
        this.notifyConnectionEstablished(message.client_id || '', message.origin || '');
        break;
      case 'snapshot':
        this.notifySnapshot(message.data || message.payload); // Backend can send data or payload
        break;
      case 'event': // General events from /ws/events endpoint
        if (message.event_type) {
          this.notifyEvent(message.event_type, message.data || message.payload);
        } else {
          console.warn('WebSocketService: Received event message without event_type:', message);
        }
        break;
      case 'heartbeat':
        this.handleHeartbeat();
        this.notifyPingPong();
        break;
      case 'pong':
        this.notifyPingPong();
        break;
      case 'subscription_confirmed':
        this.notifySubscriptionConfirmed(message.topics || []);
        break;
      case 'unsubscription_confirmed':
        this.notifyUnsubscriptionConfirmed(message.topics || []);
        break;
      case 'connection_stats':
        this.notifyConnectionStats(message.data);
        break;
      case 'error':
        this.notifyError(new Error((message.message as string) || 'Unknown error from server'));
        break;
      case 'system_health': // Handle system_health explicitly
        if (isSystemHealth(message.data)) {
          this.notifySystemHealth(message.data);
        } else if (isSystemHealth(message.payload)) {
          this.notifySystemHealth(message.payload);
        } else {
          console.warn('WebSocketService: Received system_health message with invalid data format:', message);
          this.notifyError(new Error('Invalid system health data format.'));
        }
        break;
      // Benchmark-specific types - keep for compatibility if other components use them
      case 'execution_update':
        this.handleExecutionUpdate(message);
        break;
      case 'metrics_update':
        this.handleMetricsUpdate(message);
        break;
      case 'benchmark_complete':
        this.handleBenchmarkComplete(message);
        break;
      default:
        console.warn('WebSocketService: Unhandled message type:', message.type, message);
        // Don't throw errors for unknown message types, just log them
        break;
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
    
    // Notify all subscriptions about heartbeat for their own logic if needed
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onHeartbeat) {
        subscription.callbacks.onHeartbeat();
      }
    });
  }

  private handleError(event: Event): void {
    console.error('WebSocketService: WebSocket error:', event);
    console.error('WebSocketService: WebSocket readyState:', this.ws?.readyState);
    this.notifyError(new Error('WebSocket connection error'));
    this.notifyConnectionStatusChange('disconnected'); // Notify disconnected on error
  }

  private handleClose(event: CloseEvent): void {
    console.log('WebSocketService: WebSocket connection closed:', event.code, 'reason:', event.reason);
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
      console.log(`WebSocketService: Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
      
      setTimeout(() => {
        this.connect();
      }, delay);
      this.notifyConnectionStatusChange('reconnecting'); // Notify reconnecting
    } else {
      console.log('WebSocketService: Not attempting to reconnect - either closed intentionally or max attempts reached');
      this.notifyConnectionStatusChange('disconnected'); // Notify disconnected
    }
  }

  private handleConnectionError(error: Error): void {
    console.error('WebSocket connection error:', error);
    this.isConnecting = false;
    this.notifyError(error);
    this.notifyConnectionStatusChange('disconnected'); // Notify disconnected on error
    
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
          type: 'ping',
          timestamp: new Date().toISOString()
        }));
      }
    }, 30000); // Send ping every 30 seconds (use ping instead of heartbeat)
  }

  private resubscribeToAll(): void {
    const allSubscriptions = Array.from(this.subscriptions.entries());
    allSubscriptions.forEach(([executionId, subscription]) => {
      this.sendSubscriptionRequest(executionId, subscription.callbacks);
    });
  }

  private sendSubscriptionRequest(executionId: string, callbacks: WebSocketCallbacks): void {
    // Determine which topics to subscribe to based on provided callbacks
    const topics: string[] = [];
    if (callbacks.onSnapshot) topics.push('snapshot');
    if (callbacks.onEvent) topics.push('event'); // General events
    if (callbacks.onExecutionUpdate) topics.push('execution_update');
    if (callbacks.onMetricsUpdate) topics.push('metrics_update');
    if (callbacks.onBenchmarkComplete) topics.push('benchmark_complete');
    if (callbacks.onSystemHealth) topics.push('system_health'); // Added check for onSystemHealth

    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        type: 'subscribe',
        execution_id: executionId, // This might be a generic "dashboard" ID or specific experiment ID
        topics: topics // Use the derived topics
      }));
    }
  }

  // New notification methods for general event types
  private notifyError(error: Error): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onError) subscription.callbacks.onError(error);
    });
  }

  private notifySnapshot(data: unknown): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onSnapshot) subscription.callbacks.onSnapshot(data);
    });
  }

  private notifyEvent(eventType: string, data: unknown): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onEvent) subscription.callbacks.onEvent(eventType, data);
    });
  }

  private notifyConnectionEstablished(clientId: string, origin: string): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onConnectionEstablished) subscription.callbacks.onConnectionEstablished(clientId, origin);
    });
  }

  private notifySubscriptionConfirmed(topics: string[]): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onSubscriptionConfirmed) subscription.callbacks.onSubscriptionConfirmed(topics);
    });
  }

  private notifyUnsubscriptionConfirmed(topics: string[]): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onUnsubscriptionConfirmed) subscription.callbacks.onUnsubscriptionConfirmed(topics);
    });
  }

  private notifyConnectionStats(data: unknown): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onConnectionStats) subscription.callbacks.onConnectionStats(data);
    });
  }

  private notifyPingPong(): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onPingPong) subscription.callbacks.onPingPong();
    });
  }

  private notifyConnectionStatusChange(status: 'connecting' | 'connected' | 'disconnected' | 'reconnecting'): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onConnectionStatusChange) {
        subscription.callbacks.onConnectionStatusChange(status);
      }
    });
  }

  private notifySystemHealth(data: SystemHealth): void {
    this.subscriptions.forEach(subscription => {
      if (subscription.callbacks.onSystemHealth) {
        subscription.callbacks.onSystemHealth(data);
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