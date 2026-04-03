import type { ReactNode } from 'react';

interface Props {
  title: string;
  value: string | number;
  subtitle?: string;
  icon?: ReactNode;
  color?: 'green' | 'red' | 'blue' | 'amber' | 'purple' | 'cyan';
}

const COLOR_MAP = {
  green: 'text-[#00E676]',
  red: 'text-[#FF1744]',
  blue: 'text-[#42A5F5]',
  amber: 'text-[#FFB300]',
  purple: 'text-[#AB47BC]',
  cyan: 'text-[#00BCD4]',
};

const BG_MAP = {
  green: 'from-[#00E676]/[0.06] to-transparent',
  red: 'from-[#FF1744]/[0.06] to-transparent',
  blue: 'from-[#42A5F5]/[0.06] to-transparent',
  amber: 'from-[#FFB300]/[0.06] to-transparent',
  purple: 'from-[#AB47BC]/[0.06] to-transparent',
  cyan: 'from-[#00BCD4]/[0.06] to-transparent',
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
