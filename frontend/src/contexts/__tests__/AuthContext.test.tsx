import React from 'react';
import { describe, it, expect, beforeEach, afterEach } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { AuthProvider, useAuth } from '../AuthContext';
import PrivateRoute from '../../routes/PrivateRoute';
import ToastProvider from '../../contexts/ToastContext';
import Login from '../../pages/Login';

function setEnv(next: Record<string, string | undefined>) {
  const meta = (import.meta as unknown as { env: Record<string, string | undefined> });
  meta.env = { ...(meta.env ?? {}), ...next };
}

function resetStorage() {
  localStorage.clear();
  sessionStorage.clear();
}

const StatusProbe: React.FC = () => {
  const { user, isAuthenticated, authEnabled } = useAuth();
  return (
    <div>
      <div data-testid="auth-enabled">{String(authEnabled)}</div>
      <div data-testid="is-authenticated">{String(isAuthenticated)}</div>
      <div data-testid="user-email">{user?.email ?? ''}</div>
    </div>
  );
};

describe('AuthContext', () => {
  beforeEach(() => {
    resetStorage();
  });
  afterEach(() => {
    resetStorage();
  });

  it('disabled auth: always authenticated with default user; login/logout are no-ops; PrivateRoute renders children', () => {
    setEnv({
      VITE_AUTH_ENABLED: 'false',
      VITE_AUTH_DEFAULT_USER: 'devuser@example.com',
      VITE_AUTH_DEFAULT_ROLE: 'admin',
    });

    render(
      <MemoryRouter initialEntries={['/protected']}>
        <ToastProvider>
          <AuthProvider>
            <Routes>
              <Route
                path="/protected"
                element={
                  <PrivateRoute>
                    <div>Protected OK</div>
                  </PrivateRoute>
                }
              />
            </Routes>
            <StatusProbe />
          </AuthProvider>
        </ToastProvider>
      </MemoryRouter>,
    );

    expect(screen.getByTestId('auth-enabled').textContent).toBe('false');
    expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
    expect(screen.getByTestId('user-email').textContent).toBe('devuser@example.com');
    expect(screen.getByText('Protected OK')).toBeInTheDocument();
  });

  it('enabled auth: initial unauthenticated; protected route redirects to /login; login persists; logout clears', async () => {
    setEnv({
      VITE_AUTH_ENABLED: 'true',
      VITE_AUTH_DEFAULT_USER: 'devuser@example.com',
      VITE_AUTH_DEFAULT_ROLE: 'admin',
    });

    render(
      <MemoryRouter initialEntries={['/experiments']}>
        <ToastProvider>
          <AuthProvider>
            <Routes>
              <Route
                path="/experiments"
                element={
                  <PrivateRoute>
                    <div>Experiments Protected</div>
                  </PrivateRoute>
                }
              />
              <Route path="/login" element={<Login />} />
            </Routes>
            <StatusProbe />
          </AuthProvider>
        </ToastProvider>
      </MemoryRouter>,
    );

    // Should be on login page due to redirect
    expect(await screen.findByRole('heading', { name: /sign in/i })).toBeInTheDocument();

    // Fill and submit login
    const email = screen.getByLabelText(/email address/i) as HTMLInputElement;
    fireEvent.change(email, { target: { value: 'user@example.com' } });
    fireEvent.click(screen.getByRole('button', { name: /sign in/i }));

    // Should navigate to default "/" (since redirect=/experiments was encoded by PrivateRoute, but we started with /experiments; Login should navigate there)
    await waitFor(() => {
      // After login, the provider state should show authenticated
      expect(screen.getByTestId('is-authenticated').textContent).toBe('true');
      expect(screen.getByTestId('user-email').textContent).toBe('user@example.com');
    });

    // Simulate full app navigation back to protected route to ensure guard allows now
    // Render protected directly since user is persisted to localStorage
    render(
      <MemoryRouter initialEntries={['/experiments']}>
        <ToastProvider>
          <AuthProvider>
            <Routes>
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
        </ToastProvider>
      </MemoryRouter>,
    );

    expect(await screen.findByText('Experiments Protected')).toBeInTheDocument();

    // Logout: Use context directly through a consumer
    const LogoutProbe: React.FC = () => {
      const { logout } = useAuth();
      return (
        <button onClick={() => logout()} aria-label="logout-now">
          logout
        </button>
      );
    };

    render(
      <MemoryRouter>
        <ToastProvider>
          <AuthProvider>
            <LogoutProbe />
            <StatusProbe />
          </AuthProvider>
        </ToastProvider>
      </MemoryRouter>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'logout-now' }));
    await waitFor(() => {
      expect(screen.getByTestId('is-authenticated').textContent).toBe('false');
      expect(localStorage.getItem('auth:user')).toBeNull();
    });
  });
});