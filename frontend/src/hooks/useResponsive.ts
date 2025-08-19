import { useEffect, useMemo, useState } from 'react';

export type Breakpoint = 'sm' | 'md' | 'lg' | 'xl';

/**
 * useResponsive
 * - Returns booleans for current breakpoints based on viewport width
 * - Mobile-first defaults
 * - Updates on resize with passive listener
 */
export default function useResponsive() {
  const getFlags = () => {
    const w = typeof window !== 'undefined' ? window.innerWidth : 0;
    return {
      isSm: w >= 640,
      isMd: w >= 768,
      isLg: w >= 1024,
      isXl: w >= 1280,
      width: w,
    };
  };

  const [state, setState] = useState(getFlags);

  useEffect(() => {
    const onResize = () => setState(getFlags());
    window.addEventListener('resize', onResize, { passive: true });
    return () => window.removeEventListener('resize', onResize);
  }, []);

  return useMemo(() => state, [state]);
}