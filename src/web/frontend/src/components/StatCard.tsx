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

export default function StatCard({ title, value, subtitle, icon, color = 'blue' }: Props) {
  return (
    <div className="glass-card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs text-[#64748b] uppercase tracking-wider">{title}</p>
          <p className={`text-2xl font-bold mt-1 ${COLOR_MAP[color]}`}>{value}</p>
          {subtitle && <p className="text-xs text-[#94a3b8] mt-1">{subtitle}</p>}
        </div>
        {icon && <div className="text-[#64748b]">{icon}</div>}
      </div>
    </div>
  );
}
