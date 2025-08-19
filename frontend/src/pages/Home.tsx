import React, { useEffect } from 'react';
import { Link } from 'react-router-dom';

/**
 * Home page placeholder.
 * - Responsive, accessible structure
 * - Demonstrates keyboard shortcuts notice
 * - Uses landmark roles and skip link target via main id
 */
const Home: React.FC = () => {
  useEffect(() => {
    // Announce page load to screen readers (simple example)
    const el = document.getElementById('home-page-announcer');
    if (el) {
      el.textContent = 'Home page loaded';
    }
  }, []);

  return (
    <section aria-labelledby="home-title" className="w-full h-full">
      <h1 id="home-title" className="text-2xl font-semibold text-gray-900 mb-4">Home</h1>

      <p className="text-gray-600 mb-6">
        Welcome to the FBA-Bench UI. This is a minimal, production-ready placeholder to validate routing,
        responsiveness, accessibility, and global error handling.
      </p>

      <div className="rounded-md border border-blue-200 bg-blue-50 text-blue-900 p-4 mb-6">
        <p className="text-sm">
          Keyboard tips: Press <kbd className="px-1 py-0.5 border rounded bg-white">s</kbd> to toggle the sidebar,
          and <kbd className="px-1 py-0.5 border rounded bg-white">/</kbd> to focus the header search.
        </p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
        <Link to="/experiments" className="block rounded-lg border p-4 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
          <h2 className="font-medium text-gray-900">Experiments</h2>
          <p className="text-gray-600 text-sm mt-1">Manage and run experiments.</p>
        </Link>
        <Link to="/agents" className="block rounded-lg border p-4 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
          <h2 className="font-medium text-gray-900">Agents</h2>
          <p className="text-gray-600 text-sm mt-1">View and configure agents.</p>
        </Link>
        <Link to="/results" className="block rounded-lg border p-4 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
          <h2 className="font-medium text-gray-900">Results</h2>
          <p className="text-gray-600 text-sm mt-1">Analyze outcomes and KPIs.</p>
        </Link>
        <Link to="/settings" className="block rounded-lg border p-4 hover:shadow-sm focus:outline-none focus:ring-2 focus:ring-blue-500">
          <h2 className="font-medium text-gray-900">Settings</h2>
          <p className="text-gray-600 text-sm mt-1">Adjust UI preferences.</p>
        </Link>
      </div>

      <span id="home-page-announcer" className="sr-only" aria-live="polite" />
    </section>
  );
};

export default Home;