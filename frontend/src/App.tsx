import React, { Suspense } from 'react';
import { ThemeProvider } from './contexts/ThemeContext';
import { ErrorBoundary } from './components/error/ErrorBoundary';
import Spinner from './components/ui/Spinner';
import Layout from './components/layout/Layout';
import AppRoutes from './routes';
import ToastProvider from './contexts/ToastContext';
import Toast from './components/ui/Toast';
import { AuthProvider } from './contexts/AuthContext';

const App: React.FC = () => {
  return (
    <ThemeProvider>
      <ToastProvider>
        <AuthProvider>
          <a href="#main-content" className="sr-only focus:not-sr-only focus:absolute focus:top-2 focus:left-2 bg-blue-600 text-white px-3 py-2 rounded">
            Skip to content
          </a>
          <ErrorBoundary>
            <Layout>
              <Suspense fallback={<div className="flex items-center justify-center h-64"><Spinner size="xl" label="Loading page…" /></div>}>
                <div id="main-content" role="main" className="h-full">
                  <AppRoutes />
                </div>
              </Suspense>
            </Layout>
          </ErrorBoundary>
        </AuthProvider>
        <Toast />
      </ToastProvider>
    </ThemeProvider>
  );
};

export default App;
