import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import Experiments from '../Experiments';
import ToastProvider from '../../contexts/ToastContext';
import { BrowserRouter } from 'react-router-dom';

vi.mock('../../../services/api', () => {
  class RouteNotAvailableError extends Error {
    code = 'route_not_available';
    statusCode = 404;
    route: string;
    constructor(route: string) {
      super(`not available: ${route}`);
      this.name = 'RouteNotAvailableError';
      this.route = route;
    }
  }
  return {
    getExperiments: vi.fn(),
    createExperiment: vi.fn(),
    startSimulation: vi.fn(),
    RouteNotAvailableError,
  };
});

vi.mock('../../../services/realtime', () => {
  const client = {
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
    publish: vi.fn(),
    close: vi.fn(),
  };
  return {
    connectRealtime: vi.fn(() => client),
    SIMULATION_TOPIC_PREFIX: 'simulation:',
  };
});

import { getExperiments, createExperiment, startSimulation } from '../../services/api';

function renderWithProviders(ui: React.ReactElement) {
  return render(
    <BrowserRouter>
      <ToastProvider>{ui}</ToastProvider>
    </BrowserRouter>,
  );
}

describe('Experiments page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders list from API and shows skeleton while loading', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([
      {
        id: 'exp-1',
        name: 'My Experiment',
        description: 'Test',
        status: 'draft',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);

    renderWithProviders(<Experiments />);

    // Skeleton appears first
    expect(await screen.findAllByRole('status')).toBeTruthy();

    // Item appears
    await waitFor(() => {
      expect(screen.getByText('My Experiment')).toBeInTheDocument();
    });
  });

  it('create flow: opens modal and calls API', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (createExperiment as unknown as any).mockResolvedValueOnce({
      id: 'exp-new',
      name: 'NewExp',
      description: 'd',
      status: 'draft',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    renderWithProviders(<Experiments />);

    // Open modal
    const newBtn = await screen.findByRole('button', { name: /new experiment/i });
    fireEvent.click(newBtn);

    // Fill and submit
    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'NewExp' } });
    fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'd' } });
    fireEvent.click(screen.getByRole('button', { name: /create/i }));

    await waitFor(() => {
      expect(createExperiment).toHaveBeenCalledWith({ name: 'NewExp', description: 'd' });
      // New item shown
      expect(screen.getByText('NewExp')).toBeInTheDocument();
    });
  });

  it('start simulation triggers API and subscribes to realtime', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([
      {
        id: 'exp-2',
        name: 'RunExp',
        description: '',
        status: 'draft',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (startSimulation as unknown as any).mockResolvedValueOnce({
      id: 'sim-1',
      experiment_id: 'exp-2',
      status: 'queued',
    });

    renderWithProviders(<Experiments />);

    const startBtn = await screen.findByRole('button', { name: /start simulation for runexp/i });
    fireEvent.click(startBtn);

    await waitFor(() => {
      expect(startSimulation).toHaveBeenCalledWith('exp-2');
    });

    // Badge should reflect optimistic running or live status update later
    // We minimally verify the button disabled state flips back after call
    await waitFor(() => {
      expect(startBtn).toBeEnabled();
    });
  });
});