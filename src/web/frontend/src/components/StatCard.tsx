import type { ReactNode } from 'react';

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  color?: 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'cyan';
}

const COLOR_MAP = {
  green: 'text-[#34d399]',
  red: 'text-[#fb7185]',
  blue: 'text-[#a78bfa]',
  amber: 'text-[#f59e0b]',
  purple: 'text-[#c084fc]',
  cyan: 'text-[#22d3ee]',
};

const BG_MAP = {
  green: 'from-[#34d399]/[0.06] to-transparent',
  red: 'from-[#fb7185]/[0.06] to-transparent',
  blue: 'from-[#a78bfa]/[0.06] to-transparent',
  amber: 'from-[#f59e0b]/[0.06] to-transparent',
  purple: 'from-[#c084fc]/[0.06] to-transparent',
  cyan: 'from-[#22d3ee]/[0.06] to-transparent',
};

export default function StatCard({ title, value, subtitle, icon, color = 'blue' }: Props) {
  return (
    <div className="glass-card p-5 hover-lift stat-shine group">
      <div className={`absolute inset-0 rounded-2xl bg-gradient-to-br ${BG_MAP[color]} opacity-0 group-hover:opacity-100 transition-opacity duration-500 pointer-events-none`} />
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-[11px] text-[#64748b] uppercase tracking-widest font-medium">{title}</p>
          <p className={`text-2xl font-bold mt-2 tabular-nums tracking-tight ${COLOR_MAP[color]}`}>{value}</p>
          {subtitle && <p className="text-[11px] text-[#64748b] mt-1.5">{subtitle}</p>}
        </div>
        {icon && <div className={`${COLOR_MAP[color]} opacity-40 group-hover:opacity-60 transition-opacity duration-300`}>{icon}</div>}
      </div>
    </div>
  );
}
