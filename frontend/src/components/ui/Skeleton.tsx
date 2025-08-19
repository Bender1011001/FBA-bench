import React from 'react';

type SkeletonVariant = 'text' | 'rect' | 'circle' | 'badge' | 'card';

export interface SkeletonProps {
  variant?: SkeletonVariant;
  width?: number | string;
  height?: number | string;
  className?: string;
  radius?: number | string;
  lines?: number; // for text variant
  'aria-label'?: string;
}

/**
 * Lightweight, accessible Skeleton component with multiple variants.
 * Uses prefers-reduced-motion to respect user settings.
 */
const Skeleton: React.FC<SkeletonProps> = ({
  variant = 'rect',
  width,
  height,
  className = '',
  radius,
  lines = 3,
  'aria-label': ariaLabel = 'Loading…',
}) => {
  const style: React.CSSProperties = {
    width: width ?? undefined,
    height: height ?? undefined,
    borderRadius: radius !== undefined ? (typeof radius === 'number' ? `${radius}px` : radius) : undefined,
  };

  if (variant === 'text') {
    const arr = Array.from({ length: Math.max(1, lines) });
    return (
      <div role="status" aria-label={ariaLabel} className={`space-y-2 ${className}`}>
        {arr.map((_, idx) => (
          <div
            key={idx}
            className="skeleton-line"
            style={{
              height: '0.875rem',
              borderRadius: 6,
              width: idx === arr.length - 1 ? '70%' : '100%',
            }}
          />
        ))}
        <span className="visually-hidden">{ariaLabel}</span>
        <style>{css}</style>
      </div>
    );
  }

  if (variant === 'circle') {
    const size = typeof width === 'number' ? width : typeof height === 'number' ? height : 40;
    const sizePx = typeof size === 'number' ? `${size}px` : (size ?? '40px');
    return (
      <div role="status" aria-label={ariaLabel} className={className}>
        <div className="skeleton-block" style={{ width: sizePx, height: sizePx, borderRadius: '9999px' }} />
        <span className="visually-hidden">{ariaLabel}</span>
        <style>{css}</style>
      </div>
    );
  }

  if (variant === 'badge') {
    return (
      <div role="status" aria-label={ariaLabel} className={className}>
        <div className="skeleton-block" style={{ width: width ?? 80, height: height ?? 24, borderRadius: 9999 }} />
        <span className="visually-hidden">{ariaLabel}</span>
        <style>{css}</style>
      </div>
    );
  }

  if (variant === 'card') {
    return (
      <div role="status" aria-label={ariaLabel} className={`space-y-3 ${className}`} style={style}>
        <div className="skeleton-block" style={{ width: '100%', height: 140, borderRadius: 12 }} />
        <div className="skeleton-line" style={{ height: 14, width: '90%', borderRadius: 6 }} />
        <div className="skeleton-line" style={{ height: 14, width: '70%', borderRadius: 6 }} />
        <span className="visually-hidden">{ariaLabel}</span>
        <style>{css}</style>
      </div>
    );
  }

  // rect (default)
  return (
    <div role="status" aria-label={ariaLabel} className={className}>
      <div className="skeleton-block" style={style} />
      <span className="visually-hidden">{ariaLabel}</span>
      <style>{css}</style>
    </div>
  );
};

const css = `
.skeleton-block,
.skeleton-line {
  position: relative;
  overflow: hidden;
  background: color-mix(in oklab, Canvas 8%, transparent);
}

@media (prefers-color-scheme: dark) {
  .skeleton-block,
  .skeleton-line {
    background: color-mix(in oklab, Canvas 20%, transparent);
  }
}

.skeleton-block::after,
.skeleton-line::after {
  content: '';
  position: absolute;
  inset: 0;
  transform: translateX(-100%);
  background: linear-gradient(
    90deg,
    transparent,
    color-mix(in oklab, CanvasText 10%, transparent),
    transparent
  );
  animation: skeleton-shimmer 1.2s ease-in-out infinite;
}

@media (prefers-reduced-motion: reduce) {
  .skeleton-block::after,
  .skeleton-line::after {
    animation: none;
  }
}

@keyframes skeleton-shimmer {
  100% {
    transform: translateX(100%);
  }
}
`;

export default Skeleton;