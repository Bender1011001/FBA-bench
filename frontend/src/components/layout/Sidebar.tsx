import React from 'react';
import { NavLink } from 'react-router-dom';

export interface SidebarProps {
  open: boolean;
  collapsed?: boolean;
  onClose: () => void;
}

/**
 * Accessible, responsive Sidebar
 * - Overlays on mobile; collapsible on md+
 * - Uses role="navigation", aria-label
 * - Keyboard accessible: ESC closes when overlayed
 */
const Sidebar: React.FC<SidebarProps> = ({ open, collapsed = false, onClose }) => {
  React.useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && open) onClose();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open, onClose]);

  return (
    <>
      {/* Mobile overlay */}
      <div
        className={`sidebar-overlay ${open ? 'visible' : ''} md:hidden`}
        aria-hidden={!open}
        onClick={onClose}
      />

      <aside
        id="sidebar"
        className={`sidebar ${open ? 'open' : ''} ${collapsed ? 'collapsed' : ''}`}
        role="navigation"
        aria-label="Main"
      >
        <div className="h-full flex flex-col gap-2 p-3">
          <nav className="flex-1 space-y-1">
            <SidebarLink to="/" label="Home" onNavigate={onClose} />
            <SidebarLink to="/experiments" label="Experiments" onNavigate={onClose} />
            <SidebarLink to="/agents" label="Agents" onNavigate={onClose} />
            <SidebarLink to="/results" label="Results" onNavigate={onClose} />
            <SidebarLink to="/settings" label="Settings" onNavigate={onClose} />
          </nav>
          <div className="text-xs text-blue-100/70 px-2 py-3 border-t border-white/10">
            v1.0.0
          </div>
        </div>
      </aside>
    </>
  );
};

type NavLinkClassArgs = { isActive: boolean; isPending: boolean; isTransitioning: boolean };

const SidebarLink: React.FC<{ to: string; label: string; onNavigate: () => void }> = ({ to, label, onNavigate }) => {
  return (
    <NavLink
      to={to}
      onClick={onNavigate}
      className={({ isActive }: NavLinkClassArgs) =>
        `block rounded px-3 py-2 text-sm transition-colors ${
          isActive ? 'bg-white/10 text-white' : 'text-blue-100 hover:bg-white/10 hover:text-white'
        }`
      }
      end
    >
      {label}
    </NavLink>
  );
};

export default Sidebar;