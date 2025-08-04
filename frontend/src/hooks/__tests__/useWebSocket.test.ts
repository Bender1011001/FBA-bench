import { renderHook, act } from '@testing-library/react';
import useWebSocket from '../useWebSocket';

// Mock WebSocket
class MockWebSocket {
  static CONNECTING = 0;
  static OPEN = 1;
  static CLOSING = 2;
  static CLOSED = 3;
  
  url: string;
  protocols?: string | string[];
  readyState: number;
  onopen: ((event: Event) => void) | null = null;
  onclose: ((event: CloseEvent) => void) | null = null;
  onerror: ((event: Event) => void) | null = null;
  onmessage: ((event: MessageEvent) => void) | null = null;
  
  constructor(url: string, protocols?: string | string[]) {
    this.url = url;
    this.protocols = protocols;
    this.readyState = MockWebSocket.CONNECTING;
    
    // Simulate connection opening after a short delay
    setTimeout(() => {
      this.readyState = MockWebSocket.OPEN;
      if (this.onopen) {
        this.onopen(new Event('open'));
      }
    }, 10);
  }
  
  send(data: string | ArrayBufferLike | Blob | ArrayBufferView): void {
    // Mock implementation
  }
  
  close(code?: number, reason?: string): void {
    this.readyState = MockWebSocket.CLOSED;
    if (this.onclose) {
      this.onclose(new CloseEvent('close', { code, reason }));
    }
  }
}

// @ts-ignore
global.WebSocket = MockWebSocket;

describe('useWebSocket Hook', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  test('initializes with disconnected status', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: false
    }));

    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
    expect(result.current.lastMessage).toBeNull();
  });

  test('connects automatically when autoConnect is true', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true
    }));

    expect(result.current.connectionStatus).toBe('connecting');
    expect(result.current.readyState).toBe(WebSocket.CONNECTING);
  });

  test('does not connect automatically when autoConnect is false', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: false
    }));

    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
  });

  test('can connect manually', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: false
    }));

    act(() => {
      result.current.connect();
    });

    expect(result.current.connectionStatus).toBe('connecting');
    expect(result.current.readyState).toBe(WebSocket.CONNECTING);
  });

  test('can disconnect', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true
    }));

    act(() => {
      result.current.disconnect();
    });

    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
  });

  test('updates status when connection opens', async () => {
    const onOpen = jest.fn();
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      onOpen
    }));

    // Fast-forward until all timers have been executed
    act(() => {
      jest.advanceTimersByTime(20);
    });

    expect(result.current.connectionStatus).toBe('connected');
    expect(result.current.readyState).toBe(WebSocket.OPEN);
    expect(onOpen).toHaveBeenCalled();
  });

  test('updates status when connection closes', () => {
    const onClose = jest.fn();
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      onClose
    }));

    act(() => {
      result.current.disconnect();
    });

    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
    expect(onClose).toHaveBeenCalled();
  });

  test('handles connection errors', () => {
    const onError = jest.fn();
    
    // Mock WebSocket constructor to throw an error
    const originalWebSocket = global.WebSocket;
    global.WebSocket = jest.fn().mockImplementation(() => {
      throw new Error('Connection error');
    });

    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      onError
    }));

    expect(result.current.connectionStatus).toBe('disconnected');
    expect(result.current.readyState).toBe(WebSocket.CLOSED);
    expect(onError).toHaveBeenCalled();

    // Restore original WebSocket
    global.WebSocket = originalWebSocket;
  });

  test('receives messages', () => {
    const onMessage = jest.fn();
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      onMessage
    }));

    // Fast-forward until connection is open
    act(() => {
      jest.advanceTimersByTime(20);
    });

    // Simulate receiving a message
    const mockMessage = new MessageEvent('message', {
      data: JSON.stringify({ type: 'test', payload: 'data' })
    });

    act(() => {
      if (result.current.readyState === WebSocket.OPEN) {
        // @ts-ignore
        const ws = result.current._ws;
        if (ws && ws.onmessage) {
          ws.onmessage(mockMessage);
        }
      }
    });

    expect(result.current.lastMessage).toBe(mockMessage);
    expect(onMessage).toHaveBeenCalledWith(mockMessage);
  });

  test('sends messages when connected', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true
    }));

    // Fast-forward until connection is open
    act(() => {
      jest.advanceTimersByTime(20);
    });

    const mockSend = jest.fn();
    
    // Mock the WebSocket instance's send method
    act(() => {
      // @ts-ignore
      const ws = result.current._ws;
      if (ws) {
        ws.send = mockSend;
      }
    });

    act(() => {
      result.current.sendMessage('test message');
    });

    expect(mockSend).toHaveBeenCalledWith('test message');
  });

  test('does not send messages when not connected', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: false
    }));

    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {});

    act(() => {
      result.current.sendMessage('test message');
    });

    expect(consoleSpy).toHaveBeenCalledWith('WebSocket is not connected. Cannot send message.');
    
    consoleSpy.mockRestore();
  });

  test('attempts to reconnect on unexpected close', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      reconnectAttempts: 2,
      reconnectInterval: 100
    }));

    // Fast-forward until connection is open
    act(() => {
      jest.advanceTimersByTime(20);
    });

    expect(result.current.connectionStatus).toBe('connected');

    // Simulate unexpected close (not initiated by user)
    act(() => {
      // @ts-ignore
      const ws = result.current._ws;
      if (ws && ws.onclose) {
        ws.onclose(new CloseEvent('close', { code: 1006, reason: 'Abnormal closure' }));
      }
    });

    expect(result.current.connectionStatus).toBe('disconnected');

    // Fast-forward to trigger reconnect
    act(() => {
      jest.advanceTimersByTime(150);
    });

    expect(result.current.connectionStatus).toBe('connecting');
  });

  test('stops reconnecting after max attempts', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      reconnectAttempts: 1,
      reconnectInterval: 100
    }));

    // Fast-forward until connection is open
    act(() => {
      jest.advanceTimersByTime(20);
    });

    // Simulate unexpected close
    act(() => {
      // @ts-ignore
      const ws = result.current._ws;
      if (ws && ws.onclose) {
        ws.onclose(new CloseEvent('close', { code: 1006, reason: 'Abnormal closure' }));
      }
    });

    // Fast-forward to trigger first reconnect attempt
    act(() => {
      jest.advanceTimersByTime(150);
    });

    // Simulate another close
    act(() => {
      // @ts-ignore
      const ws = result.current._ws;
      if (ws && ws.onclose) {
        ws.onclose(new CloseEvent('close', { code: 1006, reason: 'Abnormal closure' }));
      }
    });

    // Fast-forward past second reconnect attempt
    act(() => {
      jest.advanceTimersByTime(150);
    });

    // Should not attempt to reconnect again
    expect(result.current.connectionStatus).toBe('disconnected');
  });

  test('does not reconnect on normal close', () => {
    const { result } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true,
      reconnectAttempts: 2,
      reconnectInterval: 100
    }));

    // Fast-forward until connection is open
    act(() => {
      jest.advanceTimersByTime(20);
    });

    // Simulate normal close (initiated by user)
    act(() => {
      // @ts-ignore
      const ws = result.current._ws;
      if (ws && ws.onclose) {
        ws.onclose(new CloseEvent('close', { code: 1000, reason: 'Normal closure' }));
      }
    });

    // Fast-forward past potential reconnect interval
    act(() => {
      jest.advanceTimersByTime(150);
    });

    // Should not attempt to reconnect
    expect(result.current.connectionStatus).toBe('disconnected');
  });

  test('cleans up on unmount', () => {
    const { result, unmount } = renderHook(() => useWebSocket({
      url: 'ws://example.com',
      autoConnect: true
    }));

    // Fast-forward until connection is open
    act(() => {
      jest.advanceTimersByTime(20);
    });

    const mockClose = jest.fn();
    
    // Mock the WebSocket instance's close method
    act(() => {
      // @ts-ignore
      const ws = result.current._ws;
      if (ws) {
        ws.close = mockClose;
      }
    });

    unmount();

    expect(mockClose).toHaveBeenCalledWith(1000, 'Disconnect requested');
  });
});