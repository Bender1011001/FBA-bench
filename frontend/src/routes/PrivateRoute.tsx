import React from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

export interface PrivateRouteProps {
  children: React.ReactElement;
  requiredRole?: string;
}

/**
 * PrivateRoute
 * - If authEnabled and not authenticated, redirect to /login with `redirect` query.
 * - If authenticated, renders children. Role requirement currently permissive (demo).
 */
const PrivateRoute: React.FC<PrivateRouteProps> = ({ children }) => {
  const { authEnabled, isAuthenticated } = useAuth();
  const location = useLocation();

  if (authEnabled && !isAuthenticated) {
    const redirect = encodeURIComponent(location.pathname + location.search);
    return <Navigate to={`/login?redirect=${redirect}`} replace />;
  }

  return children;
};

export default PrivateRoute;