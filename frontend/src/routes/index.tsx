import React, { lazy } from 'react';
import { Route, Routes, Navigate } from 'react-router-dom';
import PrivateRoute from './PrivateRoute';
 
// Lazy-loaded pages for route-based code splitting
const Home = lazy(() => import('../pages/Home'));
const Experiments = lazy(() => import('../pages/Experiments'));
const ExperimentDetail = lazy(() => import('../pages/ExperimentDetail'));
const Agents = lazy(() => import('../pages/Agents'));
const Results = lazy(() => import('../pages/Results'));
const Settings = lazy(() => import('../pages/Settings')); // Reuse existing if default export exists; otherwise wrapper file provides default
const Login = lazy(() => import('../pages/Login'));
 
/**
 * Centralized application routes.
 * Add new feature routes here to keep routing cohesive.
 */
const AppRoutes: React.FC = () => {
  return (
    <Routes>
      <Route path="/" element={<Home />} />
      <Route
        path="/experiments"
        element={
          <PrivateRoute>
            <Experiments />
          </PrivateRoute>
        }
      />
      <Route
        path="/experiments/:id"
        element={
          <PrivateRoute>
            <ExperimentDetail />
          </PrivateRoute>
        }
      />
      <Route
        path="/agents"
        element={
          <PrivateRoute>
            <Agents />
          </PrivateRoute>
        }
      />
      <Route
        path="/results"
        element={
          <PrivateRoute>
            <Results />
          </PrivateRoute>
        }
      />
      <Route path="/settings" element={<Settings />} />
      <Route path="/login" element={<Login />} />
      {/* Legacy or unknown paths redirect to Home */}
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  );
};
 
export default AppRoutes;