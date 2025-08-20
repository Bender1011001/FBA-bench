import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketOptions {
  url: string;
  protocols?: string | string[];
  autoConnect?: boolean;
  onOpen?: (event: Event) => void;
  onClose?: (event: CloseEvent) => void;
  onError?: (event: Event) => void;
  onMessage?: (event: MessageEvent) => void;
  reconnectAttempts?: number;
  reconnectInterval?: number;
}

interface WebSocketHookResult {
  lastMessage: MessageEvent | null;
  connectionStatus: 'connecting' | 'connected' | 'disconnected';
  sendMessage: (data: string | ArrayBufferLike | Blob | ArrayBufferView) => void;
  connect: () => void;
  disconnect: () => void;
  readyState: number;
}

const useWebSocket = (options: WebSocketOptions): WebSocketHookResult => {
  const {
    url,
    protocols,
    autoConnect = true,
    onOpen,
    onClose,
    onError,
    onMessage,
    reconnectAttempts = 5,
    reconnectInterval = 3000
  } = options;

  const [lastMessage, setLastMessage] = useState<MessageEvent | null>(null);
  const [connectionStatus, setConnectionStatus] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');
  const [readyState, setReadyState] = useState<number>(WebSocket.CLOSED);
  
  const websocketRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const reconnectCountRef = useRef<number>(0);
  const shouldConnectRef = useRef<boolean>(autoConnect);

  const connect = useCallback(() => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      return;
    }

    try {
      console.log('useWebSocket: Attempting to connect to:', url);
      setConnectionStatus('connecting');
      setReadyState(WebSocket.CONNECTING);
      
      // Attach Authorization via Sec-WebSocket-Protocol when token is available in localStorage or via a getter
      let subprotocols: string | string[] | undefined = protocols;
      try {
        const token = localStorage.getItem('auth:jwt'); // optional: app should set this on login
        if (token) {
          const proto = `auth.bearer.token.${token}`;
          if (Array.isArray(protocols)) {
            subprotocols = [...protocols, proto];
          } else if (typeof protocols === 'string' && protocols.length > 0) {
            subprotocols = [protocols, proto];
          } else {
            subprotocols = [proto];
          }
        }
      } catch {
        // ignore storage access issues
      }
      const ws = new WebSocket(url, subprotocols);
      
      ws.onopen = (event: Event) => {
       if ((import.meta as unknown as { env: Record<string, string | undefined> }).env.VITE_LOG_LEVEL === 'debug') {
         // eslint-disable-next-line no-console
         console.log('useWebSocket: Connection established successfully');
       }
        setConnectionStatus('connected');
        setReadyState(WebSocket.OPEN);
        reconnectCountRef.current = 0;
        onOpen?.(event);
      };
      
      ws.onclose = (event: CloseEvent) => {
       if ((import.meta as unknown as { env: Record<string, string | undefined> }).env.VITE_LOG_LEVEL === 'debug') {
         // eslint-disable-next-line no-console
         console.log('useWebSocket: Connection closed, code:', event.code, 'reason:', event.reason);
       }
        setConnectionStatus('disconnected');
        setReadyState(WebSocket.CLOSED);
        websocketRef.current = null;
        onClose?.(event);
        
        // Attempt to reconnect if the connection was not closed intentionally
        if (shouldConnectRef.current && event.code !== 1000 && reconnectCountRef.current < reconnectAttempts) {
          console.log('useWebSocket: Attempting to reconnect, attempt:', reconnectCountRef.current + 1);
          reconnectCountRef.current++;
          reconnectTimeoutRef.current = setTimeout(() => {
            connect();
          }, reconnectInterval);
        }
      };
      
      ws.onerror = (event: Event) => {
       if ((import.meta as unknown as { env: Record<string, string | undefined> }).env.VITE_LOG_LEVEL === 'debug') {
         // eslint-disable-next-line no-console
         console.error('useWebSocket: Error event:', event);
       }
        onError?.(event);
      };
      
      ws.onmessage = (event: MessageEvent) => {
        setLastMessage(event);
        onMessage?.(event);
      };
      
      websocketRef.current = ws;
    } catch (error) {
      console.error('Error creating WebSocket:', error);
      setConnectionStatus('disconnected');
      setReadyState(WebSocket.CLOSED);
    }
  }, [url, protocols, onOpen, onClose, onError, onMessage, reconnectAttempts, reconnectInterval]);

  const disconnect = useCallback(() => {
    shouldConnectRef.current = false;
    
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (websocketRef.current) {
      websocketRef.current.close(1000, 'Disconnect requested');
      websocketRef.current = null;
    }
  }, []);

  const sendMessage = useCallback((data: string | ArrayBufferLike | Blob | ArrayBufferView) => {
    if (websocketRef.current && websocketRef.current.readyState === WebSocket.OPEN) {
      websocketRef.current.send(data);
    } else {
      console.error('WebSocket is not connected. Cannot send message.');
    }
  }, []);

  useEffect(() => {
    shouldConnectRef.current = autoConnect;
    
    if (autoConnect) {
      connect();
    }
    
    return () => {
      disconnect();
    };
  }, [autoConnect, connect, disconnect]);

  return {
    lastMessage,
    connectionStatus,
    sendMessage,
    connect,
    disconnect,
    readyState
  };
};

export default useWebSocket;