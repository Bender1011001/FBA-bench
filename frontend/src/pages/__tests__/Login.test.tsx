import React from 'react';
import { describe, it, expect, beforeEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AuthProvider } from '../../contexts/AuthContext';
import Login from '../Login';
import PrivateRoute from '../../routes/PrivateRoute';

function setEnv(next: Record<string, string | undefined>) {
  const meta = (import.meta as unknown as { env: Record<string, string | undefined> });
  meta.env = { ...(meta.env ?? {}), ...next };
}

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
});

describe('Login page', () => {
  it('renders form controls with labels and aria helpers', async () => {
    setEnv({ VITE_AUTH_ENABLED: 'true', VITE_AUTH_DEFAULT_USER: 'prefill@example.com' });

    render(
      <MemoryRouter initialEntries={['/login']}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    expect(await screen.findByRole('heading', { name: /sign in/i })).toBeInTheDocument();
    const email = screen.getByLabelText(/email address/i);
    const pwd = screen.getByLabelText(/password/i);
    expect(email).toBeInTheDocument();
    expect(pwd).toBeInTheDocument();
    // aria hints exist
    expect(screen.getByText(/any non-empty email is accepted/i)).toBeInTheDocument();
    expect(screen.getByText(/ignored in mock mode/i)).toBeInTheDocument();
  });

  it('submitting with empty email surfaces accessible error via aria-live', async () => {
    setEnv({ VITE_AUTH_ENABLED: 'true' });

    render(
      <MemoryRouter initialEntries={['/login?redirect=%2Fexperiments']}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/email is required/i);
  });

  it('successful login redirects to redirect target', async () => {
    setEnv({ VITE_AUTH_ENABLED: 'true' });

    render(
      <MemoryRouter initialEntries={['/login?redirect=%2Fexperiments']}>
        <AuthProvider>
          <Routes>
            <Route path="/login" element={<Login />} />
            <Route
              path="/experiments"
              element={
                <PrivateRoute>
                  <div>Experiments Protected</div>
                </PrivateRoute>
              }
            />
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    const email = await screen.findByLabelText(/email address/i);
    fireEvent.change(email, { target: { value: 'user@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(async () => {
      expect(await screen.findByText('Experiments Protected')).toBeInTheDocument();
    });
  });

  it('when auth disabled, page indicates disabled and link navigates home', async () => {
    setEnv({ VITE_AUTH_ENABLED: 'false' });

    render(
      <MemoryRouter initialEntries={['/login']}>
        <AuthProvider>
          <Routes>
            <Route path="/" element={<div>Home Page</div>} />
            <Route path="/login" element={<Login />} />
          </Routes>
        </AuthProvider>
      </MemoryRouter>,
    );

    // It auto-redirects quickly; accept either state
    // If still on screen, should indicate disabled auth
    const maybeHeading = await screen.findByRole('heading', { name: /authentication disabled/i }).catch(() => null);
    if (maybeHeading) {
      // Click continue to Home
      const link = screen.getByRole('link', { name: /continue to home/i });
      fireEvent.click(link);
      expect(await screen.findByText('Home Page')).toBeInTheDocument();
    } else {
      // Already redirected
      expect(await screen.findByText('Home Page')).toBeInTheDocument();
    }
  });
});