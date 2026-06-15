import LoadingSpinner from '../../../components/LoadingSpinner';
import type { PoliticiansSummaryResponse } from '../../../api';
import { BadgeDollarSign, Clock3, Landmark, ShieldAlert, Star } from 'lucide-react';

export type PoliticianInsightFilter = 'all' | 'tracked' | 'watchlist' | 'late';

interface Props {
  summary?: PoliticiansSummaryResponse;
  isLoading: boolean;
  activeFilter: PoliticianInsightFilter;
  onFilterChange: (filter: PoliticianInsightFilter) => void;
}

export function insightFilterCount(summary: PoliticiansSummaryResponse | undefined, filter: PoliticianInsightFilter): number {
  if (!summary || summary.status !== 'ok') return 0;
  if (filter === 'tracked') return summary.tracked_asset_trades ?? 0;
  if (filter === 'watchlist') return summary.watchlist_trades ?? 0;
  if (filter === 'late') return summary.late_filings ?? 0;
  return summary.new_disclosures_7d ?? 0;
}

export default function PoliticianInsightBar({ summary, isLoading, activeFilter, onFilterChange }: Props) {
  if (isLoading) return <LoadingSpinner text="Loading politician disclosure summary..." variant="stats" />;
  if (!summary || summary.status !== 'ok') return null;

  const metrics = [
    {
      filter: 'all' as const,
      label: 'Recent disclosures',
      value: summary.new_disclosures_7d ?? 0,
      icon: Clock3,
      color: 'var(--accent-cyan)',
      bg: 'rgba(56,217,245,0.10)',
    },
    {
      filter: 'tracked' as const,
      label: 'Tracked assets',
      value: summary.tracked_asset_trades ?? 0,
      icon: BadgeDollarSign,
      color: 'var(--accent-emerald)',
      bg: 'var(--emerald-12)',
    },
    {
      filter: 'watchlist' as const,
      label: 'Watchlist',
      value: summary.watchlist_trades ?? 0,
      icon: Star,
      color: 'var(--accent-amber)',
      bg: 'var(--amber-12)',
    },
    {
      filter: 'late' as const,
      label: 'Late filings',
      value: summary.late_filings ?? 0,
      icon: ShieldAlert,
      color: 'var(--accent-rose)',
      bg: 'var(--rose-12)',
    },
  ];

  return (
    <section className="glass-card overflow-hidden">
      <div className="grid gap-4 p-4 xl:grid-cols-[minmax(220px,0.85fr)_minmax(0,2.2fr)] xl:items-center">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="flex h-8 w-8 items-center justify-center rounded-[8px]" style={{ color: 'var(--accent-cyan)', background: 'rgba(56,217,245,0.08)' }}>
              <Landmark className="h-4 w-4" />
            </span>
            <div className="min-w-0">
              <h2 className="text-[14px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
                Delayed Public Disclosures
              </h2>
              <p className="mt-1 text-[11px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                {summary.newest_disclosure_date || 'no disclosure date'} · {sourceHealthLabel(summary)}
              </p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 lg:grid-cols-4">
          {metrics.map((metric) => {
            const Icon = metric.icon;
            const active = activeFilter === metric.filter;
            return (
              <button
                key={metric.filter}
                type="button"
                onClick={() => onFilterChange(metric.filter)}
                className="min-h-[76px] rounded-[8px] px-3 py-3 text-left transition-all hover:brightness-110 focus:outline-none focus:ring-2 focus:ring-[var(--accent-cyan)]"
                style={{
                  background: active ? metric.bg : 'linear-gradient(135deg, rgba(255,255,255,0.035), rgba(255,255,255,0.015))',
                  border: active ? `1px solid ${metric.color}` : '1px solid var(--violet-8)',
                  boxShadow: active ? `0 0 24px ${metric.bg}` : undefined,
                }}
              >
                <div className="flex items-center justify-between gap-2">
                  <span className="text-[10px] font-semibold uppercase leading-tight" style={{ color: active ? metric.color : 'var(--text-muted)', letterSpacing: '0.08em' }}>
                    {metric.label}
                  </span>
                  <Icon className="h-3.5 w-3.5 shrink-0" style={{ color: metric.color }} />
                </div>
                <div className="mt-3 text-[24px] font-semibold tabular-nums leading-none" style={{ color: 'var(--text-luminous)' }}>
                  {metric.value.toLocaleString()}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </section>
  );
}

function sourceHealthLabel(summary: PoliticiansSummaryResponse): string {
  const entries = Object.values(summary.source_health?.sources || {});
  const statuses = entries
    .map((entry) => {
      if (entry && typeof entry === 'object' && 'status' in entry) {
        return String((entry as { status?: unknown }).status || 'unknown');
      }
      return 'unknown';
    })
    .filter(Boolean);
  if (statuses.length === 0) return 'pending';
  const nonOk = statuses.filter((status) => status !== 'ok');
  if (nonOk.length === 0) return 'ok';
  return `${nonOk.join(', ')} source health`;
}
