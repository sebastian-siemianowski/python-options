import type { ReactNode } from 'react';

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  color?: 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'cyan';
}

const COLOR_MAP = {
  green: 'text-[#3ee8a5]',
  red: 'text-[#ff6b8a]',
  blue: 'text-[#b49aff]',
  amber: 'text-[#f5c542]',
  purple: 'text-[#c084fc]',
  cyan: 'text-[#38d9f5]',
};

const BG_MAP = {
  green: 'from-[#3ee8a5]/[0.08] to-transparent',
  red: 'from-[#ff6b8a]/[0.08] to-transparent',
  blue: 'from-[#b49aff]/[0.08] to-transparent',
  amber: 'from-[#f5c542]/[0.08] to-transparent',
  purple: 'from-[#c084fc]/[0.08] to-transparent',
  cyan: 'from-[#38d9f5]/[0.08] to-transparent',
};

const GLOW_MAP: Record<string, string> = {
  green: '0 0 30px rgba(62,232,165,0.06)',
  red: '0 0 30px rgba(255,107,138,0.06)',
  blue: '0 0 30px rgba(167,139,250,0.06)',
  amber: '0 0 30px rgba(245,197,66,0.06)',
  purple: '0 0 30px rgba(192,132,252,0.06)',
  cyan: '0 0 30px rgba(56,217,245,0.06)',
};

export default function StatCard({ title, value, subtitle, icon, color = 'blue' }: Props) {
  return (
    <div
      className="glass-card p-5 hover-lift stat-shine group"
      style={{ boxShadow: GLOW_MAP[color] }}
    >
      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${BG_MAP[color]} opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none`} />
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-[10px] uppercase tracking-[0.14em] font-semibold" style={{ color: 'var(--text-muted)' }}>{title}</p>
          <p className={`text-[26px] font-bold mt-2 tabular-nums tracking-tight leading-none ${COLOR_MAP[color]}`}>{value}</p>
          {subtitle && <p className="text-[11px] mt-2" style={{ color: 'var(--text-muted)' }}>{subtitle}</p>}
        </div>
        {icon && <div className={`${COLOR_MAP[color]} opacity-30 group-hover:opacity-50 transition-opacity duration-400`}>{icon}</div>}
      </div>
    </div>
  );
}
