import React, { useCallback, useEffect, useState } from 'react';
import Header from './Header';
import Sidebar from './Sidebar';

/**
 * Layout component
 * - Responsive app shell with header, collapsible sidebar, and scrollable content
 * - Mobile overlay on small screens
 * - Keyboard shortcuts are wired in Header:
 *   - 's' toggles sidebar
 *   - '/' focuses search in header
 * - Accessible landmarks: header has role="banner", aside has role="navigation", main has role="main"
 */
const Layout: React.FC<React.PropsWithChildren> = ({ children }) => {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  // Persist collapsed state for md+ screens
  useEffect(() => {
    try {
      const raw = localStorage.getItem('ui.sidebar-collapsed');
      if (raw) setSidebarCollapsed(raw === 'true');
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem('ui.sidebar-collapsed', String(sidebarCollapsed));
    } catch {
      // ignore
    }
  }, [sidebarCollapsed]);

  const toggleSidebar = useCallback(() => {
    setSidebarOpen((s) => !s);
  }, []);

  const closeSidebar = useCallback(() => setSidebarOpen(false), []);

  // Close sidebar on route change if using client-side navigation: listen to hashchange/popstate as a simple heuristic
  useEffect(() => {
    const handler = () => setSidebarOpen(false);
    window.addEventListener('hashchange', handler);
    window.addEventListener('popstate', handler);
    return () => {
      window.removeEventListener('hashchange', handler);
      window.removeEventListener('popstate', handler);
    };
  }, []);

  return (
    <div className={`app-shell with-sidebar`}>
      <Header onToggleSidebar={toggleSidebar} sidebarOpen={sidebarOpen} />

      {/* Sidebar (overlays on mobile, collapses on md+) */}
      <Sidebar open={sidebarOpen} collapsed={sidebarCollapsed} onClose={closeSidebar} />

      {/* Collapse toggle for md+ screens */}
      <div className="hidden md:block absolute left-[var(--sidebar-width)] top-[var(--header-height)] z-10">
        <button
          type="button"
          aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
          className="icon-btn translate-x-[-18px]"
          onClick={() => setSidebarCollapsed((c) => !c)}
        >
          {sidebarCollapsed ? (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M14 7l-5 5 5 5V7z" />
            </svg>
          ) : (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24" fill="currentColor">
              <path d="M10 17l5-5-5-5v10z" />
            </svg>
          )}
        </button>
      </div>

      <main id="main-content" role="main" className="main">
        <div className="container py-6">{children}</div>
      </main>
    </div>
  );
};

export default Layout;