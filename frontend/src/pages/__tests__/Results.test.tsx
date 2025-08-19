import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import Results from '../Results';
import ToastProvider from '../../contexts/ToastContext';
import { BrowserRouter } from 'react-router-dom';

vi.mock('../../services/api', () => {
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
    getEngineReport: vi.fn(),
    RouteNotAvailableError,
  };
});

vi.mock('../../services/realtime', () => {
  const client = {
    subscribe: vi.fn(),
    unsubscribe: vi.fn(),
    publish: vi.fn(),
    close: vi.fn(),
  };
  return {
    connectRealtime: vi.fn(() => client),
    EXPERIMENT_TOPIC_PREFIX: 'experiment:',
  };
});

import { getExperiments, getEngineReport } from '../../services/api';

function renderWithProviders(ui: React.ReactElement) {
  return render(
    <BrowserRouter>
      <ToastProvider>{ui}</ToastProvider>
    </BrowserRouter>,
  );
}

describe('Results page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('loads experiments and selects first by default', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([
      {
        id: 'e-1',
        name: 'Exp One',
        description: '',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
      {
        id: 'e-2',
        name: 'Exp Two',
        description: '',
        status: 'draft',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getEngineReport as unknown as any).mockResolvedValueOnce({
      experiment_id: 'e-1',
      scenario_reports: [],
      totals: { Accuracy: 0.92, 'Cost Efficiency': 0.81 },
      created_at: new Date().toISOString(),
    });

    renderWithProviders(<Results />);

    // Experiment select should show items
    const select = await screen.findByLabelText(/experiment/i);
    expect(select).toBeInTheDocument();
    expect((select as HTMLSelectElement).value).toBe('e-1');

    // Key metrics rendered
    await waitFor(() => {
      expect(screen.getByText(/Accuracy/i)).toBeInTheDocument();
      expect(screen.getByText(/0.920/)).toBeInTheDocument();
    });
  });

  it('handles missing metrics gracefully', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([
      {
        id: 'e-3',
        name: 'No Metrics',
        description: '',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getEngineReport as unknown as any).mockResolvedValueOnce({
      experiment_id: 'e-3',
      scenario_reports: [],
      totals: {},
      created_at: new Date().toISOString(),
    });

    renderWithProviders(<Results />);

    // Show no summary metrics message
    await waitFor(() => {
      expect(screen.getByText(/No summary metrics available yet/i)).toBeInTheDocument();
    });
  });

  it('renders scenario validators with indicators', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([
      {
        id: 'e-4',
        name: 'With Scenarios',
        description: '',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getEngineReport as unknown as any).mockResolvedValueOnce({
      experiment_id: 'e-4',
      scenario_reports: [
        {
          scenario_key: 'sc-1',
          metrics: { score: 0.5 },
          validators: { ok_rule: true, bad_rule: false, info_rule: 'n/a' },
          summary: 'test',
        },
      ],
      totals: { Accuracy: 0.5 },
      created_at: new Date().toISOString(),
    });

    renderWithProviders(<Results />);

    await waitFor(() => {
      expect(screen.getByText('sc-1')).toBeInTheDocument();
      expect(screen.getByText(/ok_rule/)).toBeInTheDocument();
      expect(screen.getByText(/bad_rule/)).toBeInTheDocument();
      expect(screen.getByText(/info_rule/)).toBeInTheDocument();
    });
  });

  it('refresh button triggers report fetch', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getExperiments as unknown as any).mockResolvedValueOnce([
      {
        id: 'e-5',
        name: 'Refresh Me',
        description: '',
        status: 'completed',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);

    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getEngineReport as unknown as any).mockResolvedValue({
      experiment_id: 'e-5',
      scenario_reports: [],
      totals: { Accuracy: 0.7 },
      created_at: new Date().toISOString(),
    });

    renderWithProviders(<Results />);

    const refreshBtn = await screen.findByRole('button', { name: /refresh results/i });
    fireEvent.click(refreshBtn);

    await waitFor(() => {
      expect(getEngineReport).toHaveBeenCalled();
    });
  });
});