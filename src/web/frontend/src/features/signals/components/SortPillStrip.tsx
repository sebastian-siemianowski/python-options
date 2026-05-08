import { useMemo, type ReactNode } from 'react';
import { Activity, ArrowDown, ArrowUp, BarChart3, RotateCcw, Shield, TrendingUp } from 'lucide-react';
import { formatHorizon } from '../../../utils/horizons';
import type { SortColumn, SortDir } from '../utils';

type SortLevel = { col: SortColumn; dir: SortDir };

type SortPill = {
  col: SortColumn;
  label: string;
  shortLabel: string;
  accent: string;
  icon: ReactNode;
};

const DEFAULT_PROJECTION_HORIZONS = [1, 3, 7, 21, 63, 126, 252];

const BASE_SORT_PILLS: SortPill[] = [
  {
    col: 'quality',
    label: 'Business quality',
    shortLabel: 'Quality',
    accent: '#c4b5fd',
    icon: <Shield className="w-3.5 h-3.5" />,
  },
  {
    col: 'momentum',
    label: 'Momentum score',
    shortLabel: 'Momentum',
    accent: '#22d3ee',
    icon: <TrendingUp className="w-3.5 h-3.5" />,
  },
  {
    col: 'crash_risk',
    label: 'Risk',
    shortLabel: 'Risk',
    accent: '#fb7185',
    icon: <Activity className="w-3.5 h-3.5" />,
  },
  {
    col: 'pct30d',
    label: '30D price change',
    shortLabel: '30D',
    accent: '#38bdf8',
    icon: <BarChart3 className="w-3.5 h-3.5" />,
  },
];

export const buildSignalSortPills = (horizons: number[] = DEFAULT_PROJECTION_HORIZONS): SortPill[] => [
  ...BASE_SORT_PILLS,
  ...horizons.map((h): SortPill => ({
    col: `horizon_${h}` as SortColumn,
    label: `Projected return ${formatHorizon(h)}`,
    shortLabel: formatHorizon(h),
    accent: h <= 7 ? '#6ee7b7' : h <= 63 ? '#38bdf8' : '#c4b5fd',
    icon: <BarChart3 className="w-3.5 h-3.5" />,
  })),
];

export default function SortPillStrip({
  sortLevels,
  onSort,
  onClear,
  title = 'Sort',
  subtitle,
  horizons,
  className = '',
}: {
  sortLevels: SortLevel[];
  onSort: (col: SortColumn) => void;
  onClear?: () => void;
  title?: string;
  subtitle?: string;
  horizons?: number[];
  className?: string;
}) {
  const pills = useMemo(() => buildSignalSortPills(horizons), [horizons]);
  return (
    <div
      className={`w-full flex flex-wrap items-center gap-x-2 gap-y-1.5 ${className}`}
      aria-label={subtitle ? `${title}: ${subtitle}` : title}
    >
      <div className="flex items-center gap-2 min-w-[86px]">
        <span className="inline-flex items-center gap-1.5 text-[9.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
          <ArrowDown className="w-3 h-3" />
          {title}
        </span>
        {subtitle && (
          <span className="hidden sm:inline text-[10px] text-[var(--text-muted)]">
            {subtitle}
          </span>
        )}
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        {pills.map((pill) => {
          const activeIndex = sortLevels.findIndex((s) => s.col === pill.col);
          const active = activeIndex >= 0 ? sortLevels[activeIndex] : null;
          const isActive = !!active;
          return (
            <button
              key={pill.col}
              type="button"
              onClick={() => onSort(pill.col)}
              className="premium-sort-pill group relative inline-flex items-center gap-1.5 overflow-hidden rounded-[11px] px-2.5 py-[5px] text-[11px] font-semibold tabular-nums transition-all duration-[180ms] active:scale-[0.97] hover:-translate-y-[1px]"
              data-active={isActive}
              style={isActive
                ? {
                    color: pill.accent,
                    background: `linear-gradient(135deg, ${pill.accent}24, rgba(255,255,255,0.04) 52%, rgba(15,23,42,0.42))`,
                    border: `1px solid ${pill.accent}55`,
                    boxShadow: `0 0 0 1px ${pill.accent}20, inset 0 1px 0 rgba(255,255,255,0.10), 0 14px 30px -22px ${pill.accent}`,
                  }
                : {
                    color: 'var(--text-secondary)',
                    background: 'linear-gradient(180deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012))',
                    border: '1px solid rgba(255,255,255,0.06)',
                    boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.035)',
                  }}
              title={`Sort by ${pill.label}${active ? ` (${active.dir === 'asc' ? 'ascending' : 'descending'})` : ''}`}
              aria-pressed={isActive}
            >
              {isActive && (
                <span
                  className="pointer-events-none absolute inset-x-2 bottom-0 h-px rounded-full"
                  style={{ background: `linear-gradient(90deg, transparent, ${pill.accent}, transparent)` }}
                />
              )}
              <span
                className="inline-flex items-center justify-center rounded-full transition-colors"
                style={{
                  width: 18,
                  height: 18,
                  color: isActive ? pill.accent : 'rgba(148,163,184,0.78)',
                  background: isActive ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.035)',
                }}
              >
                {pill.icon}
              </span>
              <span>{pill.shortLabel}</span>
              {active && (
                <span
                  className="inline-flex items-center gap-0.5 rounded-full px-1.5 py-[1px] text-[9px] font-bold"
                  style={{ background: 'rgba(0,0,0,0.22)', color: pill.accent }}
                >
                  {sortLevels.length > 1 && activeIndex + 1}
                  {active.dir === 'asc' ? <ArrowUp className="w-2.5 h-2.5" /> : <ArrowDown className="w-2.5 h-2.5" />}
                </span>
              )}
            </button>
          );
        })}
      </div>

      {onClear && (
        <button
          type="button"
          onClick={onClear}
          className="inline-flex items-center gap-1 rounded-[9px] px-2 py-[5px] text-[10px] font-medium transition-all active:scale-[0.97] hover:-translate-y-[1px]"
          style={{
            color: 'var(--text-secondary)',
            background: 'rgba(255,255,255,0.018)',
            border: '1px solid rgba(255,255,255,0.06)',
          }}
          title="Reset sorting"
        >
          <RotateCcw className="w-3 h-3" />
          Reset
        </button>
      )}
    </div>
  );
}
