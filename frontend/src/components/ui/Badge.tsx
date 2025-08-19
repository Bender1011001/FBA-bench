import React from 'react';

type Tone =
  | 'neutral'
  | 'info'
  | 'success'
  | 'warning'
  | 'danger'
  | 'blue'
  | 'yellow'
  | 'green'
  | 'red';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  tone?: Tone;
  children: React.ReactNode;
  className?: string;
}

/**
 * Accessible status badge
 * - Color-coded by tone
 * - Uses semantic role and aria-label passthrough when needed
 */
const Badge: React.FC<BadgeProps> = ({ tone = 'neutral', children, className = '', ...rest }) => {
  const toneClass =
    tone === 'success'
      ? 'badge-success'
      : tone === 'warning'
      ? 'badge-warning'
      : tone === 'danger' || tone === 'red'
      ? 'badge-danger'
      : tone === 'info' || tone === 'blue'
      ? 'badge-info'
      : tone === 'green'
      ? 'badge-success'
      : tone === 'yellow'
      ? 'badge-warning'
      : 'badge-neutral';

  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${toneClass} ${className}`}
      {...rest}
    >
      {children}
    </span>
  );
};

export default Badge;