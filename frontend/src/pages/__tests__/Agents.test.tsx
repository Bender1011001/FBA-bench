import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import React from 'react';
import Agents from '../Agents';
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
    getAgents: vi.fn(),
    createAgent: vi.fn(),
    updateAgent: vi.fn(),
    deleteAgent: vi.fn(),
    RouteNotAvailableError,
  };
});

import { getAgents, createAgent, updateAgent, deleteAgent } from '../../services/api';

function renderWithProviders(ui: React.ReactElement) {
  return render(
    <BrowserRouter>
      <ToastProvider>{ui}</ToastProvider>
    </BrowserRouter>,
  );
}

describe('Agents page', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders list, shows skeleton, then items', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getAgents as unknown as any).mockResolvedValueOnce([
      {
        id: 'a-1',
        name: 'Baseline Agent',
        description: 'desc',
        runner: 'python',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);

    renderWithProviders(<Agents />);

    // skeleton at first
    expect(await screen.findAllByRole('status')).toBeTruthy();

    await waitFor(() => {
      expect(screen.getByText('Baseline Agent')).toBeInTheDocument();
    });
  });

  it('creates an agent via modal form', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getAgents as unknown as any).mockResolvedValueOnce([]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (createAgent as unknown as any).mockResolvedValueOnce({
      id: 'a-new',
      name: 'NewAgent',
      runner: 'node',
      description: 'd',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });

    renderWithProviders(<Agents />);

    const createBtn = await screen.findByRole('button', { name: /register agent/i });
    fireEvent.click(createBtn);

    fireEvent.change(screen.getByLabelText(/name/i), { target: { value: 'NewAgent' } });
    fireEvent.change(screen.getByLabelText(/runner/i), { target: { value: 'node' } });
    fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'd' } });
    fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() => {
      expect(createAgent).toHaveBeenCalledWith({
        name: 'NewAgent',
        runner: 'node',
        description: 'd',
        config: undefined,
      });
      expect(screen.getByText('NewAgent')).toBeInTheDocument();
    });
  });

  it('updates and deletes with optimistic UI', async () => {
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (getAgents as unknown as any).mockResolvedValueOnce([
      {
        id: 'a-2',
        name: 'EditMe',
        runner: 'python',
        description: '',
        created_at: new Date().toISOString(),
        updated_at: new Date().toISOString(),
      },
    ]);
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (updateAgent as unknown as any).mockResolvedValueOnce({
      id: 'a-2',
      name: 'Edited',
      runner: 'python',
      description: 'x',
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString(),
    });
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    (deleteAgent as unknown as any).mockResolvedValueOnce({ ok: true });

    renderWithProviders(<Agents />);

    // Edit
    const editBtn = await screen.findByRole('button', { name: /edit/i });
    fireEvent.click(editBtn);

    const nameInput = await screen.findByLabelText(/name/i);
    fireEvent.change(nameInput, { target: { value: 'Edited' } });
    fireEvent.change(screen.getByLabelText(/description/i), { target: { value: 'x' } });
    fireEvent.click(screen.getByRole('button', { name: /save/i }));

    await waitFor(() => {
      expect(updateAgent).toHaveBeenCalled();
      expect(screen.getByText('Edited')).toBeInTheDocument();
    });

    // Delete
    // Mock confirm
    const confirmSpy = vi.spyOn(window, 'confirm').mockReturnValue(true);
    const deleteBtn = screen.getByRole('button', { name: /delete/i });
    fireEvent.click(deleteBtn);

    await waitFor(() => {
      expect(deleteAgent).toHaveBeenCalledWith('a-2');
    });
    confirmSpy.mockRestore();
  });
});