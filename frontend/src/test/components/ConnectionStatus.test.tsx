import React from 'react';
import { describe, it, expect, beforeEach, vi, type Mock } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { useSimulationStore } from '../../store/simulationStore';
import { ConnectionStatus, ConnectionStatusCompact } from '../../components/ConnectionStatus';

// Mock the store
vi.mock('../../store/simulationStore');

// Mock the useWebSocket hook
const mockReconnect = vi.fn();
vi.mock('../../hooks/useWebSocket', () => ({
  useWebSocket: () => ({
    reconnect: mockReconnect
  })
}));

const mockUseSimulationStore = useSimulationStore as unknown as Mock;

describe('ConnectionStatus', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    
    // Default mock store state
    mockUseSimulationStore.mockReturnValue({
      connected: false,
      reconnectAttempts: 0,
      lastHeartbeat: undefined
    });
  });

  describe('Connection States', () => {
    it('should show disconnected state when not connected', () => {
      mockUseSimulationStore.mockImplementation((selector) => {
        const state = {
          simulation: {
            connectionStatus: {
              connected: false,
              reconnectAttempts: 0
            },
            error: null
          }
        };
        return selector(state);
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Disconnected')).toBeInTheDocument();
      expect(screen.getByText('Offline')).toBeInTheDocument();
      expect(screen.getByText('Reconnect')).toBeInTheDocument();
    });

    it('should show connected state when connected', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0,
        lastHeartbeat: '2025-01-01T12:00:00Z'
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Connected')).toBeInTheDocument();
      expect(screen.getByText('Online')).toBeInTheDocument();
      expect(screen.queryByText('Reconnect')).not.toBeInTheDocument();
    });

    it('should show reconnecting state when attempting to reconnect', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 3
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Reconnecting... (3)')).toBeInTheDocument();
      expect(screen.queryByText('Reconnect')).not.toBeInTheDocument();
    });
  });

  describe('Reconnect Button', () => {
    it('should show reconnect button when disconnected and not reconnecting', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 0
      });

      render(<ConnectionStatus />);
      
      const reconnectButton = screen.getByText('Reconnect');
      expect(reconnectButton).toBeInTheDocument();
    });

    it('should call reconnect function when button clicked', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 0
      });

      render(<ConnectionStatus />);
      
      const reconnectButton = screen.getByText('Reconnect');
      fireEvent.click(reconnectButton);
      
      expect(mockReconnect).toHaveBeenCalledTimes(1);
    });

    it('should not show reconnect button when already reconnecting', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 2
      });

      render(<ConnectionStatus />);
      
      expect(screen.queryByText('Reconnect')).not.toBeInTheDocument();
    });

    it('should not show reconnect button when connected', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0
      });

      render(<ConnectionStatus />);
      
      expect(screen.queryByText('Reconnect')).not.toBeInTheDocument();
    });
  });

  describe('Error Display', () => {
    it('should show error message when error exists', () => {
      mockUseSimulationStore
        .mockReturnValueOnce({
          connected: false,
          reconnectAttempts: 0
        })
        .mockReturnValueOnce('Connection failed');

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Connection failed')).toBeInTheDocument();
    });

    it('should not show error section when no error', () => {
      mockUseSimulationStore
        .mockReturnValueOnce({
          connected: true,
          reconnectAttempts: 0
        })
        .mockReturnValueOnce(null);

      render(<ConnectionStatus />);
      
      expect(screen.queryByText(/Error:/)).not.toBeInTheDocument();
    });
  });

  describe('Timestamp Formatting', () => {
    it('should show "Never" when no lastHeartbeat', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 0,
        lastHeartbeat: undefined
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Last update: Never')).toBeInTheDocument();
    });

    it('should format recent timestamp as seconds ago', () => {
      const now = new Date();
      const thirtySecondsAgo = new Date(now.getTime() - 30000);
      
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0,
        lastHeartbeat: thirtySecondsAgo.toISOString()
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Last update: 30s ago')).toBeInTheDocument();
    });

    it('should format older timestamp as minutes ago', () => {
      const now = new Date();
      const twoMinutesAgo = new Date(now.getTime() - 120000);
      
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0,
        lastHeartbeat: twoMinutesAgo.toISOString()
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Last update: 2m ago')).toBeInTheDocument();
    });

    it('should handle invalid timestamp', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0,
        lastHeartbeat: 'invalid-date'
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Last update: Invalid time')).toBeInTheDocument();
    });
  });

  describe('Details Display', () => {
    it('should show details by default', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0,
        lastHeartbeat: '2025-01-01T12:00:00Z'
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Status:')).toBeInTheDocument();
      expect(screen.getByText('Retry attempts:')).toBeInTheDocument();
    });

    it('should hide details when showDetails is false', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0
      });

      render(<ConnectionStatus showDetails={false} />);
      
      expect(screen.queryByText('Status:')).not.toBeInTheDocument();
      expect(screen.queryByText('Retry attempts:')).not.toBeInTheDocument();
    });

    it('should show last heartbeat details when available', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0,
        lastHeartbeat: '2025-01-01T12:00:00.000Z'
      });

      render(<ConnectionStatus />);
      
      expect(screen.getByText('Last heartbeat:')).toBeInTheDocument();
      expect(screen.getByText('1/1/2025, 12:00:00 PM')).toBeInTheDocument();
    });
  });

  describe('CSS Classes', () => {
    it('should apply custom className', () => {
      const { container } = render(
        <ConnectionStatus className="custom-connection-status" />
      );
      
      expect(container.firstChild).toHaveClass('custom-connection-status');
    });

    it('should apply correct color classes for connected state', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: true,
        reconnectAttempts: 0
      });

      const { container } = render(<ConnectionStatus />);
      
      expect(container.querySelector('.text-green-600')).toBeInTheDocument();
      expect(container.querySelector('.bg-green-50')).toBeInTheDocument();
    });

    it('should apply correct color classes for disconnected state', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 0
      });

      const { container } = render(<ConnectionStatus />);
      
      expect(container.querySelector('.text-red-600')).toBeInTheDocument();
      expect(container.querySelector('.bg-red-50')).toBeInTheDocument();
    });

    it('should apply correct color classes for reconnecting state', () => {
      mockUseSimulationStore.mockReturnValue({
        connected: false,
        reconnectAttempts: 2
      });

      const { container } = render(<ConnectionStatus />);
      
      expect(container.querySelector('.text-yellow-600')).toBeInTheDocument();
      expect(container.querySelector('.bg-yellow-50')).toBeInTheDocument();
    });
  });
});

describe('ConnectionStatusCompact', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockUseSimulationStore.mockReturnValue({
      connected: false,
      reconnectAttempts: 0
    });
  });

  it('should render compact version', () => {
    mockUseSimulationStore.mockReturnValue({
      connected: true,
      reconnectAttempts: 0
    });

    render(<ConnectionStatusCompact />);
    
    expect(screen.getByText('Connected')).toBeInTheDocument();
    // Should not show detailed information
    expect(screen.queryByText('Status:')).not.toBeInTheDocument();
  });

  it('should show reconnect button in compact version when disconnected', () => {
    mockUseSimulationStore.mockReturnValue({
      connected: false,
      reconnectAttempts: 0
    });

    render(<ConnectionStatusCompact />);
    
    expect(screen.getByText('Reconnect')).toBeInTheDocument();
  });

  it('should hide text on small screens', () => {
    mockUseSimulationStore.mockReturnValue({
      connected: true,
      reconnectAttempts: 0
    });

    const { container } = render(<ConnectionStatusCompact />);
    
    // Text should have hidden class for small screens
    expect(container.querySelector('.hidden.sm\\:inline')).toBeInTheDocument();
  });

  it('should apply custom className', () => {
    const { container } = render(
      <ConnectionStatusCompact className="custom-compact" />
    );
    
    expect(container.firstChild).toHaveClass('custom-compact');
  });

  it('should have tooltip on reconnect button', () => {
    mockUseSimulationStore.mockReturnValue({
      connected: false,
      reconnectAttempts: 0
    });

    render(<ConnectionStatusCompact />);
    
    const reconnectButton = screen.getByTitle('Reconnect to server');
    expect(reconnectButton).toBeInTheDocument();
  });
});

describe('ConnectionStatus Edge Cases', () => {
  it('should handle missing store data gracefully', () => {
    mockUseSimulationStore.mockReturnValue({});

    expect(() => render(<ConnectionStatus />)).not.toThrow();
  });

  it('should handle high reconnect attempt numbers', () => {
    mockUseSimulationStore.mockReturnValue({
      connected: false,
      reconnectAttempts: 999
    });

    render(<ConnectionStatus />);
    
    expect(screen.getByText('Reconnecting... (999)')).toBeInTheDocument();
    expect(screen.getByText('999')).toBeInTheDocument();
  });
});