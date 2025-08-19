import React from 'react';
import { describe, it, expect } from 'vitest';
import { MemoryRouter, Route, Routes } from 'react-router-dom';
import { render, screen } from '@testing-library/react';
import PrivateRoute from '../PrivateRoute';
import { AuthProvider } from '../../contexts/AuthContext';

function withEnv(env: Record<string, string | undefined>, renderEl: () => React.ReactElement) {
  const prev = (import.meta as unknown as { env: Record<string, string | undefined> }).env;
  (import.meta as unknown as { env: Record<string, string | undefined> }).env = { ...(prev ?? {}), ...env };
  return renderEl();
}

const Dummy = () => <div>Protected Content</div>;
const LoginPage = () => <div>Login Page</div>;

describe('PrivateRoute', () => {
  it('renders children when auth is disabled (dev mode)', async () => {
    const ui = withEnv(
      {
        VITE_AUTH_ENABLED: 'false',
        VITE_AUTH_DEFAULT_USER: 'devuser@example.com',
      },
      () => (
        <MemoryRouter initialEntries={['/protected']}>
          <AuthProvider>
            <Routes>
              <Route
                path="/protected"
                element={
                  <PrivateRoute>
                    <Dummy />
                  </PrivateRoute>
                }
              />
              <Route path="/login" element={<LoginPage />} />
            </Routes>
          </AuthProvider>
        </MemoryRouter>
      ),
    );

    render(ui);
    expect(await screen.findByText('Protected Content')).toBeInTheDocument();
  });

  it('redirects to /login when auth enabled and not authenticated', async () => {
    const ui = withEnv(
      {
        VITE_AUTH_ENABLED: 'true',
        VITE_AUTH_DEFAULT_USER: 'devuser@example.com',
      },
      () => (
        <MemoryRouter initialEntries={['/protected']}>
          <AuthProvider>
            <Routes>
              <Route
                path="/protected"
                element={
                  <PrivateRoute>
                    <Dummy />
                  </PrivateRoute>
                }
              />
              <Route path="/login" element={<LoginPage />} />
            </Routes>
          </AuthProvider>
        </MemoryRouter>
      ),
    );

    render(ui);
    expect(await screen.findByText('Login Page')).toBeInTheDocument();
  });
});