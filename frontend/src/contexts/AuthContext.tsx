/* eslint-disable react-refresh/only-export-components */
import React, { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState } from 'react';
import { setAuthTokenProvider } from '../services/api';

export type AuthUser = { id: string; email: string; role?: string };

type AuthContextValue = {
  user: AuthUser | null;
  isAuthenticated: boolean;
  authEnabled: boolean;
  login: (email: string, password?: string) => Promise<void> | void;
  logout: () => void;
};

const AuthContext = createContext<AuthContextValue | undefined>(undefined);

const STORAGE_KEY = 'auth:user';

function readEnvBool(value: string | undefined, fallback = false): boolean {
  if (typeof value !== 'string') return fallback;
  const v = value.trim().toLowerCase();
  return v === '1' || v === 'true' || v === 'yes' || v === 'on';
}

function getInitialDefaults(): { enabled: boolean; defaultEmail: string; defaultRole?: string } {
  const env = (import.meta as unknown as { env: Record<string, string | undefined> }).env;
  return {
    enabled: readEnvBool(env.VITE_AUTH_ENABLED, false),
    defaultEmail: env.VITE_AUTH_DEFAULT_USER || 'devuser@example.com',
    defaultRole: env.VITE_AUTH_DEFAULT_ROLE,
  };
}

function isValidUserShape(value: unknown): value is AuthUser {
  if (!value || typeof value !== 'object') return false;
  const v = value as Record<string, unknown>;
  return typeof v.id === 'string' && typeof v.email === 'string' && (v.role === undefined || typeof v.role === 'string');
}

function loadUserFromStorage(): AuthUser | null {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as unknown;
    return isValidUserShape(parsed) ? parsed : null;
  } catch {
    return null;
  }
}

function saveUserToStorage(user: AuthUser | null): void {
  try {
    if (user) localStorage.setItem(STORAGE_KEY, JSON.stringify(user));
    else localStorage.removeItem(STORAGE_KEY);
  } catch {
    // ignore storage errors in mock layer
  }
}

/**
 * AuthProvider
 * - If auth is disabled (env), exposes a fixed dev user and no-op login/logout.
 * - If enabled, manages user state in memory + localStorage and a mock token via setAuthTokenProvider.
 * - Provides simple focus management helpers for consuming UIs.
 */
export const AuthProvider: React.FC<React.PropsWithChildren> = ({ children }) => {
  const { enabled: authEnabled, defaultEmail, defaultRole } = useMemo(getInitialDefaults, []);
  const [user, setUser] = useState<AuthUser | null>(() => {
    if (!authEnabled) {
      return { id: 'dev-user', email: defaultEmail, role: defaultRole };
    }
    return loadUserFromStorage();
  });

  const isAuthenticated = !!user;

  // Live token passthrough
  useEffect(() => {
    if (!authEnabled) {
      // Auth disabled: do not attach Authorization header
      setAuthTokenProvider(undefined);
      return;
    }
    // Enabled: attach token when logged in
    if (user) {
      setAuthTokenProvider(() => 'dev-token');
    } else {
      setAuthTokenProvider(undefined);
    }
    return () => {
      // On unmount, clear provider
      setAuthTokenProvider(undefined);
    };
  }, [authEnabled, user]);

  // Persist on changes (enabled only)
  useEffect(() => {
    if (!authEnabled) return;
    saveUserToStorage(user);
  }, [authEnabled, user]);

  // Restore on load if enabled
  useEffect(() => {
    if (!authEnabled) return;
    if (!user) {
      const restored = loadUserFromStorage();
      if (restored) setUser(restored);
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [authEnabled]);

  const login = useCallback(async (email: string) => {
    if (!authEnabled) return; // no-op when disabled
    // In mock mode, accept any non-empty email
    const trimmed = (email || '').trim();
    if (!trimmed) {
      const err = new Error('Email is required');
      (err as Error & { code?: string }).code = 'invalid_email';
      throw err;
    }
    const mockUser: AuthUser = { id: crypto.randomUUID(), email: trimmed, role: defaultRole };
    setUser(mockUser);
    // token provider set via effect
  }, [authEnabled, defaultRole]);

  const logout = useCallback(() => {
    if (!authEnabled) return; // no-op when disabled
    setUser(null);
    saveUserToStorage(null);
    setAuthTokenProvider(undefined);
  }, [authEnabled]);

  // Focus mgmt: expose a ref that consumers could focus on auth state changes (optional, lightweight)
  const announcerRef = useRef<HTMLSpanElement | null>(null);
  useEffect(() => {
    if (announcerRef.current) {
      announcerRef.current.textContent = authEnabled
        ? isAuthenticated
          ? 'Signed in'
          : 'Signed out'
        : 'Development mode (authentication disabled)';
    }
  }, [authEnabled, isAuthenticated]);

  const value = useMemo<AuthContextValue>(
    () => ({
      user,
      isAuthenticated,
      authEnabled,
      login,
      logout,
    }),
    [user, isAuthenticated, authEnabled, login, logout],
  );

  return (
    <AuthContext.Provider value={value}>
      {children}
      {/* Non-visual polite region for screen readers to announce auth state */}
      <span ref={announcerRef} aria-live="polite" className="sr-only" />
    </AuthContext.Provider>
  );
};

export function useAuth(): AuthContextValue {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}

export default AuthProvider;