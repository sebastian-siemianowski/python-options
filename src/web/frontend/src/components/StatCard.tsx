import type { ReactNode } from 'react';
import { useCountUp } from '../hooks/useCountUp';

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  color?: 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'cyan';
}

/* ── Accent map: CSS-variable-only, no hardcoded rgb ── */
const ACCENT_VARS: Record<string, { text: string; glow: string; glowStrong: string }> = {
  green:  { text: 'var(--accent-emerald)', glow: 'var(--emerald-6)',  glowStrong: 'var(--emerald-8)' },
  red:    { text: 'var(--accent-rose)',    glow: 'var(--rose-6)',     glowStrong: 'var(--rose-8)' },
  blue:   { text: 'var(--accent-violet)',  glow: 'var(--violet-6)',   glowStrong: 'var(--violet-8)' },
  amber:  { text: 'var(--accent-amber)',   glow: 'var(--amber-12)',   glowStrong: 'var(--amber-15)' },
  purple: { text: 'var(--accent-violet)',  glow: 'var(--violet-6)',   glowStrong: 'var(--violet-8)' },
  cyan:   { text: 'var(--accent-cyan)',    glow: 'var(--violet-6)',   glowStrong: 'var(--violet-8)' },
};

export default function StatCard({ title, value, subtitle, icon, color = 'blue' }: Props) {
  const a = ACCENT_VARS[color] || ACCENT_VARS.blue;
  const displayValue = useCountUp(typeof value === 'number' ? value : value);
  return (
    <div className="glass-card hover-lift stat-shine group relative p-6">
      {/* Ambient color glow -- visible in top-left corner */}
      <div
        className="absolute top-0 left-0 w-32 h-32 pointer-events-none opacity-60 group-hover:opacity-100 transition-opacity duration-700"
        style={{ background: `radial-gradient(circle at 0% 0%, ${a.glowStrong} 0%, transparent 70%)` }}
      />
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-label">{title}</p>
          <p className="text-stat-value mt-2.5 leading-none" style={{ color: a.text }}>
            {displayValue}
          </p>
          {subtitle && (
            <p className="text-caption mt-2.5">{subtitle}</p>
          )}
        </div>
        {icon && (
          <div
            className="opacity-20 group-hover:opacity-40 transition-opacity duration-500"
            style={{ color: a.text }}
          >
            {icon}
          </div>
        )}
      </div>
    </div>
  );
}
