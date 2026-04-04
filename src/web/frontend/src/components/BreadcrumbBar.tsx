import { useLocation, Link, useParams } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import { ChevronRight } from 'lucide-react';
import { useMemo } from 'react';

/* ─── Breadcrumb segment type ───────────────────────────────────── */
interface Crumb {
  label: string;
  to?: string;
}

/* ─── Route → breadcrumb mapping ────────────────────────────────── */
const ROUTE_LABELS: Record<string, string> = {
  '': 'Overview',
  signals: 'Signals',
  risk: 'Risk Dashboard',
  charts: 'Charts',
  tuning: 'Tuning',
  data: 'Data',
  arena: 'Arena',
  diagnostics: 'Diagnostics',
  services: 'Services',
  profitability: 'Profitability',
};

/* ─── Freshness helper ──────────────────────────────────────────── */
function freshnessBadge(ageSeconds: number | null | undefined): {
  label: string;
  color: string;
  glow: string;
  pulseSpeed: string;
} {
  if (ageSeconds == null || ageSeconds < 0) {
    return { label: 'Unknown', color: 'var(--text-muted)', glow: 'transparent', pulseSpeed: '0s' };
  }
  if (ageSeconds < 60) {
    return { label: 'Live', color: 'var(--accent-emerald)', glow: 'var(--emerald-30)', pulseSpeed: '2s' };
  }
  if (ageSeconds < 300) {
    const m = Math.round(ageSeconds / 60);
    return { label: `${m}m ago`, color: 'var(--accent-amber)', glow: 'rgba(245,197,66,0.25)', pulseSpeed: '3s' };
  }
  if (ageSeconds < 3600) {
    const m = Math.round(ageSeconds / 60);
    return { label: `${m}m ago`, color: 'var(--accent-amber)', glow: 'rgba(245,197,66,0.2)', pulseSpeed: '3s' };
  }
  const h = Math.round(ageSeconds / 3600);
  return { label: `Stale: ${h}h`, color: 'var(--accent-rose)', glow: 'var(--rose-30)', pulseSpeed: '1.5s' };
}

/* ─── Component ─────────────────────────────────────────────────── */
export default function BreadcrumbBar() {
  const location = useLocation();
  const params = useParams<{ symbol?: string }>();

  /* Data freshness - use signal cache age from signalStats */
  const signalStatsQ = useQuery({
    queryKey: ['signalStats'],
    queryFn: api.signalStats,
    staleTime: 60_000,
    retry: false,
  });

  /* Build crumbs from pathname */
  const crumbs = useMemo<Crumb[]>(() => {
    const parts = location.pathname.split('/').filter(Boolean);
    if (parts.length === 0) return []; // top-level, no breadcrumb

    const result: Crumb[] = [];
    let pathAcc = '';

    for (let i = 0; i < parts.length; i++) {
      const seg = parts[i];
      pathAcc += `/${seg}`;

      // If it's a symbol parameter in /charts/:symbol
      if (i === 1 && parts[0] === 'charts' && params.symbol) {
        result.push({ label: params.symbol.toUpperCase() });
        continue;
      }

      const label = ROUTE_LABELS[seg] || seg;
      const isLast = i === parts.length - 1;

      result.push({
        label,
        to: isLast ? undefined : pathAcc,
      });
    }

    return result;
  }, [location.pathname, params.symbol]);

  /* Don't render on top-level pages (single-segment) */
  if (crumbs.length < 2) return null;

  const cacheAge = signalStatsQ.data?.cache_age_seconds;
  const fresh = freshnessBadge(cacheAge);

  return (
    <div
      className="flex items-center justify-between h-10 px-6 mb-4 rounded-xl"
      style={{
        background: 'var(--glass-surface)',
        backdropFilter: 'blur(16px) saturate(1.4)',
        WebkitBackdropFilter: 'blur(16px) saturate(1.4)',
        borderBottom: '1px solid var(--border-void)',
      }}
    >
      {/* Crumb segments */}
      <div className="flex items-center gap-1">
        {crumbs.map((crumb, i) => {
          const isLast = i === crumbs.length - 1;

          return (
            <div key={i} className="flex items-center gap-1" style={{ animation: `crumb-in 200ms ${i * 40}ms both cubic-bezier(0.2,0,0,1)` }}>
              {i > 0 && (
                <ChevronRight className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--text-muted)' }} />
              )}
              {crumb.to ? (
                <Link
                  to={crumb.to}
                  className="text-[12px] hover-underline-violet transition-colors"
                  style={{ color: 'var(--text-secondary)' }}
                >
                  {crumb.label}
                </Link>
              ) : (
                <span
                  className="text-[12px] font-semibold"
                  style={{
                    color: isLast ? 'var(--text-luminous)' : 'var(--text-secondary)',
                    textShadow: isLast ? '0 0 12px var(--violet-30)' : 'none',
                  }}
                >
                  {crumb.label}
                </span>
              )}
            </div>
          );
        })}
      </div>

      {/* Freshness badge */}
      <div
        className="flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10px] font-semibold"
        style={{
          background: `${fresh.glow.replace(/[\d.]+\)$/, '0.08)')}`,
          color: fresh.color,
          animation: fresh.pulseSpeed !== '0s' ? `badge-pulse ${fresh.pulseSpeed} ease-in-out infinite` : 'none',
        }}
      >
        <span
          className="w-1.5 h-1.5 rounded-full"
          style={{ background: fresh.color, boxShadow: `0 0 6px ${fresh.glow}` }}
        />
        {fresh.label}
      </div>
    </div>
  );
}
