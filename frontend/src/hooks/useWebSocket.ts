import { useEffect, useRef, useCallback } from 'react';
import { useSimulationStore } from '../store/simulationStore';
import type { SimulationEvent } from '../types';

// WebSocket configuration
const WS_BASE_URL = 'ws://localhost:8000';
const WS_ENDPOINT = '/ws/events';
const RECONNECT_INTERVAL = 3000; // 3 seconds
const MAX_RECONNECT_ATTEMPTS = 10;
const HEARTBEAT_INTERVAL = 30000; // 30 seconds

interface UseWebSocketOptions {
  url?: string;
  autoConnect?: boolean;
  maxReconnectAttempts?: number;
  reconnectInterval?: number;
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Event) => void;
  onMessage?: (event: SimulationEvent) => void;
}

interface UseWebSocketReturn {
  connect: () => void;
  disconnect: () => void;
  reconnect: () => void;
  sendMessage: (message: Record<string, unknown>) => void;
  isConnecting: boolean;
  isConnected: boolean;
  lastError: string | null;
}

export const useWebSocket = (options: UseWebSocketOptions = {}): UseWebSocketReturn => {
  const {
    url = `${WS_BASE_URL}${WS_ENDPOINT}`,
    autoConnect = true,
    maxReconnectAttempts = MAX_RECONNECT_ATTEMPTS,
    reconnectInterval = RECONNECT_INTERVAL,
    onConnect,
    onDisconnect,
    onError,
    onMessage,
  } = options;

  // Store actions
  const { 
    setConnectionStatus, 
    addEvent, 
    setError: setStoreError 
  } = useSimulationStore();

  // WebSocket ref and state
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<number | undefined>(undefined);
  const heartbeatIntervalRef = useRef<number | undefined>(undefined);
  const reconnectAttempts = useRef(0);
  const isConnecting = useRef(false);
  const isManuallyDisconnected = useRef(false);

  // Get current connection status from store
  const connectionStatus = useSimulationStore((state) => state.simulation.connectionStatus);

  /**
   * Clear all timers
   */
  const clearTimers = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = undefined;
    }
    if (heartbeatIntervalRef.current) {
      clearInterval(heartbeatIntervalRef.current);
      heartbeatIntervalRef.current = undefined;
    }
  }, []);

  /**
   * Send heartbeat to keep connection alive
   */
  const sendHeartbeat = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify({ type: 'heartbeat' }));
      } catch (error) {
        console.warn('Failed to send heartbeat:', error);
      }
    }
  }, []);

  /**
   * Start heartbeat interval
   */
  const startHeartbeat = useCallback(() => {
    clearInterval(heartbeatIntervalRef.current);
    heartbeatIntervalRef.current = setInterval(sendHeartbeat, HEARTBEAT_INTERVAL);
  }, [sendHeartbeat]);

  /**
   * Handle incoming WebSocket messages
   */
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const data = JSON.parse(event.data);
      
      // Handle heartbeat responses
      if (data.type === 'heartbeat') {
        setConnectionStatus({ lastHeartbeat: new Date().toISOString() });
        return;
      }

      // Handle simulation events
      if (data.type && data.timestamp) {
        const simulationEvent: SimulationEvent = data;
        
        // Add to store
        addEvent(simulationEvent);
        
        // Call custom handler if provided
        onMessage?.(simulationEvent);
        
        // Update last heartbeat on any message
        setConnectionStatus({ lastHeartbeat: new Date().toISOString() });
      }
    } catch (error) {
      console.error('Failed to parse WebSocket message:', error);
      setStoreError('Failed to parse WebSocket message');
    }
  }, [addEvent, onMessage, setConnectionStatus, setStoreError]);

  /**
   * Connect to WebSocket
   */
  const connect = useCallback(() => {
    if (isConnecting.current || (wsRef.current?.readyState === WebSocket.OPEN)) {
      return;
    }

    isConnecting.current = true;
    isManuallyDisconnected.current = false;
    
    setConnectionStatus({ 
      connected: false,
      reconnectAttempts: reconnectAttempts.current 
    });

    try {
      console.log(`Connecting to WebSocket: ${url}`);
      const ws = new WebSocket(url);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('WebSocket connected');
        isConnecting.current = false;
        reconnectAttempts.current = 0;
        
        setConnectionStatus({ 
          connected: true,
          reconnectAttempts: 0,
          lastHeartbeat: new Date().toISOString()
        });
        setStoreError(null);
        
        startHeartbeat();
        onConnect?.();
      };

      ws.onmessage = handleMessage;

      ws.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason);
        isConnecting.current = false;
        
        setConnectionStatus({ 
          connected: false,
          reconnectAttempts: reconnectAttempts.current 
        });
        
        clearTimers();
        onDisconnect?.();

        // Auto-reconnect if not manually disconnected and within retry limit
        if (!isManuallyDisconnected.current && reconnectAttempts.current < maxReconnectAttempts) {
          reconnectAttempts.current++;
          console.log(`Attempting to reconnect... (${reconnectAttempts.current}/${maxReconnectAttempts})`);
          
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        } else if (reconnectAttempts.current >= maxReconnectAttempts) {
          setStoreError(`WebSocket failed to reconnect after ${maxReconnectAttempts} attempts`);
        }
      };

      ws.onerror = (error) => {
        console.error('WebSocket error:', error);
        isConnecting.current = false;
        
        setStoreError('WebSocket connection error');
        onError?.(error);
      };

    } catch (error) {
      console.error('Failed to create WebSocket connection:', error);
      isConnecting.current = false;
      setStoreError('Failed to create WebSocket connection');
    }
  }, [url, maxReconnectAttempts, reconnectInterval, handleMessage, setConnectionStatus, setStoreError, startHeartbeat, onConnect, onDisconnect, onError, clearTimers]);

  /**
   * Disconnect from WebSocket
   */
  const disconnect = useCallback(() => {
    isManuallyDisconnected.current = true;
    clearTimers();
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect');
      wsRef.current = null;
    }
    
    setConnectionStatus({ 
      connected: false,
      reconnectAttempts: 0 
    });
  }, [clearTimers, setConnectionStatus]);

  /**
   * Manually trigger reconnection
   */
  const reconnect = useCallback(() => {
    disconnect();
    reconnectAttempts.current = 0;
    setTimeout(connect, 100);
  }, [disconnect, connect]);

  /**
   * Send message through WebSocket
   */
  const sendMessage = useCallback((message: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      try {
        wsRef.current.send(JSON.stringify(message));
      } catch (error) {
        console.error('Failed to send WebSocket message:', error);
        setStoreError('Failed to send WebSocket message');
      }
    } else {
      console.warn('WebSocket is not connected. Cannot send message.');
    }
  }, [setStoreError]);

  // Auto-connect on mount
  useEffect(() => {
    if (autoConnect) {
      connect();
    }

    // Cleanup on unmount
    return () => {
      isManuallyDisconnected.current = true;
      clearTimers();
      if (wsRef.current) {
        wsRef.current.close(1000, 'Component unmounted');
      }
    };
  }, [autoConnect, connect, clearTimers]);

  return {
    connect,
    disconnect,
    reconnect,
    sendMessage,
    isConnecting: isConnecting.current,
    isConnected: connectionStatus.connected,
    lastError: useSimulationStore((state) => state.simulation.error),
  };
};