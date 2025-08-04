import React, { createContext, useContext, useEffect, useState } from 'react';
import type { ReactNode } from 'react';

interface ThemeContextType {
  theme: 'light' | 'dark' | 'auto';
  setTheme: (theme: 'light' | 'dark' | 'auto') => void;
  isDarkMode: boolean;
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

interface ThemeProviderProps {
  children: ReactNode;
}

export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const [theme, setTheme] = useState<'light' | 'dark' | 'auto'>('light');
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Load theme from localStorage on mount
  useEffect(() => {
    try {
      const savedSettings = localStorage.getItem('app-settings');
      if (savedSettings) {
        const parsedSettings = JSON.parse(savedSettings);
        if (parsedSettings.theme) {
          setTheme(parsedSettings.theme);
        }
      }
    } catch (error) {
      console.error('Failed to load theme from localStorage:', error);
    }
  }, []);

  // Apply theme to document and update isDarkMode state
  useEffect(() => {
    const root = document.documentElement;
    
    let darkMode = false;
    
    if (theme === 'dark') {
      darkMode = true;
      root.classList.add('dark');
    } else if (theme === 'light') {
      darkMode = false;
      root.classList.remove('dark');
    } else {
      // Auto theme based on system preference
      const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
      darkMode = prefersDark;
      if (prefersDark) {
        root.classList.add('dark');
      } else {
        root.classList.remove('dark');
      }
    }
    
    setIsDarkMode(darkMode);
  }, [theme]);

  // Listen for system theme changes when in auto mode
  useEffect(() => {
    if (theme === 'auto') {
      const mediaQuery = window.matchMedia('(prefers-color-scheme: dark)');
      
      const handleChange = (e: MediaQueryListEvent) => {
        const root = document.documentElement;
        if (e.matches) {
          root.classList.add('dark');
          setIsDarkMode(true);
        } else {
          root.classList.remove('dark');
          setIsDarkMode(false);
        }
      };
      
      mediaQuery.addEventListener('change', handleChange);
      
      return () => {
        mediaQuery.removeEventListener('change', handleChange);
      };
    }
  }, [theme]);

  const value = {
    theme,
    setTheme,
    isDarkMode
  };

  return (
    <ThemeContext.Provider value={value}>
      {children}
    </ThemeContext.Provider>
  );
};

export const useTheme = (): ThemeContextType => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};