import React, { useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
 
export interface HeaderProps {
  onToggleSidebar: () => void;
  sidebarOpen: boolean;
  onFocusSearch?: () => void;
}
 
/**
 * Responsive, accessible Header with:
 * - App title
 * - Sidebar toggle button (aria-controls, aria-expanded)
 * - Search input placeholder (focusable via '/' keyboard shortcut)
 * - Theme toggle placeholder button
 * - Landmarks: role="banner", nav has role="navigation"
 */
const Header: React.FC<HeaderProps> = ({ onToggleSidebar, sidebarOpen, onFocusSearch }) => {
  const searchRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();
  const { authEnabled, isAuthenticated, user, logout } = useAuth();
 
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      // Press 's' to toggle sidebar
      if (e.key.toLowerCase() === 's' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        onToggleSidebar();
      }
      // Press '/' to focus search
      if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        if (searchRef.current) {
          searchRef.current.focus();
          searchRef.current.select();
        }
        if (onFocusSearch) onFocusSearch();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [onToggleSidebar, onFocusSearch]);
 
  return (
    <header className="header" role="banner">
      <div className="container h-[var(--header-height)] flex items-center justify-between gap-3">
        <div className="flex items-center gap-2">
          <button
            type="button"
            aria-label="Toggle navigation sidebar"
            aria-controls="sidebar"
            aria-expanded={sidebarOpen}
            className="icon-btn"
            onClick={onToggleSidebar}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
 
          <div className="flex items-center gap-2">
            <span className="font-semibold text-base md:text-lg">FBA-Bench</span>
            {!authEnabled && (
              <span
                className="ml-1 inline-flex items-center rounded bg-gray-200 px-2 py-0.5 text-xs text-gray-700"
                aria-label="Development mode badge"
              >
                Dev Mode
              </span>
            )}
          </div>
        </div>
 
        <nav aria-label="Primary" role="navigation" className="hidden md:flex items-center gap-3">
          <input
            ref={searchRef}
            id="site-search"
            type="search"
            placeholder="Search ( / )"
            className="header-search"
            aria-label="Site search"
          />
        </nav>
 
        <div className="flex items-center gap-2">
          {authEnabled ? (
            isAuthenticated ? (
              <>
                <span className="text-sm text-gray-800" aria-label="Signed in user email">{user?.email}</span>
                <button
                  type="button"
                  aria-label="Sign out"
                  className="icon-btn"
                  onClick={() => {
                    logout();
                    navigate('/login');
                  }}
                >
                  <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                    <path d="M16 13v-2H7V8l-5 4 5 4v-3h9z" />
                    <path d="M20 3h-8a2 2 0 0 0-2 2v4h2V5h8v14h-8v-4h-2v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z" />
                  </svg>
                </button>
              </>
            ) : (
              <button
                type="button"
                aria-label="Go to login"
                className="icon-btn"
                onClick={() => navigate('/login')}
              >
                <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor" aria-hidden="true">
                  <path d="M10 17v-3H3v-4h7V7l5 4-5 4z" />
                  <path d="M20 3h-8a2 2 0 0 0-2 2v4h2V5h8v14h-8v-4h-2v4a2 2 0 0 0 2 2h8a2 2 0 0 0 2-2V5a2 2 0 0 0-2-2z" />
                </svg>
              </button>
            )
          ) : null}
 
          <button
            type="button"
            aria-label="Toggle theme"
            className="icon-btn"
            onClick={() => {
              // Placeholder hook-in for future theme toggling
              // Can integrate with ThemeContext if desired
            }}
          >
            <svg xmlns="http://www.w3.org/2000/svg" className="h-5 w-5" viewBox="0 0 24 24" fill="currentColor">
              <path d="M21.64 13a1 1 0 0 0-1.05-.14 8 8 0 1 1-9.45-9.45 1 1 0 0 0-.14-1.05A1 1 0 0 0 9.5 2a10 10 0 1 0 12.5 12.5 1 1 0 0 0-.36-1.5z" />
            </svg>
          </button>
        </div>
      </div>
    </header>
  );
};
 
export default Header;