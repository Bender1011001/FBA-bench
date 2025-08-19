import React, { useEffect, useMemo, useRef, useState } from 'react';
import { useLocation, useNavigate, useSearchParams, Link } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import useToast from '../contexts/useToast';

function parseRedirect(searchParams: URLSearchParams): string {
  const raw = searchParams.get('redirect');
  if (!raw) return '/';
  try {
    const url = new URL(raw, window.location.origin);
    // Allow only same-origin relative paths for safety
    if (url.origin !== window.location.origin) return '/';
    return url.pathname + url.search + url.hash;
  } catch {
    return '/';
  }
}

const Login: React.FC = () => {
  const { authEnabled, isAuthenticated, login } = useAuth();
  const toast = useToast();
  const navigate = useNavigate();
  const location = useLocation();
  const [params] = useSearchParams();

  const [email, setEmail] = useState<string>('');
  const [password, setPassword] = useState<string>(''); // ignored in mock
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const emailRef = useRef<HTMLInputElement>(null);
  const announcerRef = useRef<HTMLSpanElement | null>(null);

  const redirectTo = useMemo(() => parseRedirect(params), [params]);

  // Prefill from env/default when present
  useEffect(() => {
    try {
      const env = (import.meta as unknown as { env: Record<string, string | undefined> }).env;
      const saved = localStorage.getItem('auth:last-email') || env.VITE_AUTH_DEFAULT_USER || '';
      setEmail(saved);
      // Focus email for keyboard users
      emailRef.current?.focus();
    } catch {
      // ignore
    }
  }, []);

  // If already authenticated while auth enabled, bounce away
  useEffect(() => {
    if (authEnabled && isAuthenticated) {
      navigate(redirectTo, { replace: true });
    }
  }, [authEnabled, isAuthenticated, navigate, redirectTo]);

  // When auth is disabled, there's no need for a login page; guide user home
  useEffect(() => {
    if (!authEnabled) {
      // Soft auto-redirect after a short announcement for SR users
      const t = window.setTimeout(() => navigate('/', { replace: true }), 100);
      return () => clearTimeout(t);
    }
  }, [authEnabled, navigate]);

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    if (!authEnabled) {
      // No-op in disabled mode
      navigate(redirectTo || '/', { replace: true });
      return;
    }
    const trimmed = email.trim();
    if (!trimmed) {
      setError('Email is required');
      if (announcerRef.current) {
        announcerRef.current.textContent = 'Email is required';
      }
      return;
    }
    try {
      setSaving(true);
      await login(trimmed);
      try {
        localStorage.setItem('auth:last-email', trimmed);
      } catch {
        // ignore
      }
      toast.success('Signed in');
      navigate(redirectTo || '/', { replace: true });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Sign-in failed';
      setError(msg);
      if (announcerRef.current) {
        announcerRef.current.textContent = msg;
      }
      toast.error(msg);
    } finally {
      setSaving(false);
    }
  };

  return (
    <section aria-labelledby="login-title" className="w-full h-full flex items-center justify-center py-10">
      <div className="w-full max-w-md rounded-lg border bg-white p-6 shadow-sm">
        <h1 id="login-title" className="text-xl font-semibold text-gray-900">
          {authEnabled ? 'Sign in' : 'Authentication Disabled'}
        </h1>
        <p className="mt-1 text-sm text-gray-600">
          {authEnabled
            ? 'Use your email to sign in (dev mock accepts any email).'
            : 'Auth is disabled via VITE_AUTH_ENABLED=false. You are effectively signed in for development.'}
        </p>

        {authEnabled ? (
          <form className="mt-4 space-y-4" onSubmit={onSubmit} noValidate>
            <div>
              <label htmlFor="email" className="block text-sm text-gray-700">
                Email address
              </label>
              <input
                ref={emailRef}
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-describedby="email-help"
              />
              <p id="email-help" className="mt-1 text-xs text-gray-500">
                Any non-empty email is accepted in mock mode.
              </p>
            </div>

            <div>
              <label htmlFor="password" className="block text-sm text-gray-700">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="mt-1 block w-full rounded-md border px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                aria-describedby="password-help"
              />
              <p id="password-help" className="mt-1 text-xs text-gray-500">
                Ignored in mock mode.
              </p>
            </div>

            {error ? (
              <div role="alert" className="rounded border border-red-200 bg-red-50 p-2 text-sm text-red-800">
                {error}
              </div>
            ) : null}
            <span ref={announcerRef} aria-live="polite" className="sr-only" />

            <button
              type="submit"
              className="inline-flex w-full items-center justify-center rounded-md bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 disabled:opacity-60"
              aria-label="Sign in"
              disabled={saving}
            >
              {saving ? 'Signing in…' : 'Sign in'}
            </button>

            <div className="text-xs text-gray-600">
              Redirect target:{' '}
              <code className="rounded bg-gray-100 px-1 py-0.5">{redirectTo || '/'}</code>
            </div>
          </form>
        ) : (
          <div className="mt-4">
            <Link
              to="/"
              className="inline-flex items-center rounded-md border px-3 py-2 text-sm hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-gray-400"
              aria-label="Continue to home"
              replace
              state={{ from: location }}
            >
              Continue to Home
            </Link>
          </div>
        )}
      </div>
    </section>
  );
};

export default Login;