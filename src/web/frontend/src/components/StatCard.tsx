import type { ReactNode } from 'react';

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  color?: 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'cyan';
}

const ACCENT: Record<string, { text: string; rgb: string }> = {
  green:  { text: 'var(--accent-emerald)', rgb: '62,232,165' },
  red:    { text: 'var(--accent-rose)', rgb: '255,107,138' },
  blue:   { text: '#b49aff', rgb: '167,139,250' },
  amber:  { text: 'var(--accent-amber)', rgb: '245,197,66' },
  purple: { text: '#c084fc', rgb: '192,132,252' },
  cyan:   { text: 'var(--accent-cyan)', rgb: '56,217,245' },
};

export default function StatCard({ title, value, subtitle, icon, color = 'blue' }: Props) {
  const a = ACCENT[color] || ACCENT.blue;
  return (
    <div
      className="glass-card hover-lift stat-shine group relative"
      style={{
        padding: '24px 24px 20px',
        boxShadow: `0 0 0 1px rgba(${a.rgb},0.06), 0 4px 24px rgba(0,0,0,0.2), 0 0 60px rgba(${a.rgb},0.03), inset 0 1px 0 rgba(255,255,255,0.04)`,
      }}
    >
      {/* Ambient color glow -- visible in top-left corner */}
      <div
        className="absolute top-0 left-0 w-32 h-32 pointer-events-none opacity-60 group-hover:opacity-100 transition-opacity duration-700"
        style={{
          background: `radial-gradient(circle at 0% 0%, rgba(${a.rgb},0.08) 0%, transparent 70%)`,
        }}
      />
      <div className="relative flex items-start justify-between">
        <div>
          <p
            className="text-[10px] uppercase font-semibold"
            style={{ color: 'var(--text-muted)', letterSpacing: '0.14em' }}
          >
            {title}
          </p>
          <p
            className="font-bold mt-2.5 tabular-nums tracking-tight leading-none"
            style={{ fontSize: '28px', color: a.text }}
          >
            {value}
          </p>
          {subtitle && (
            <p className="text-[11px] mt-2.5" style={{ color: 'var(--text-secondary)' }}>
              {subtitle}
            </p>
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
