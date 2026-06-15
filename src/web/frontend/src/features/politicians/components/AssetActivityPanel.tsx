import { useQuery } from '@tanstack/react-query';
import { Link } from 'react-router-dom';
import LoadingSpinner from '../../../components/LoadingSpinner';
import { api, type PoliticiansAssetResponse, type PoliticiansTradeRow } from '../../../api';
import { BarChart3, ExternalLink, TrendingDown, TrendingUp, Users } from 'lucide-react';

interface Props {
  symbol: string | null;
}

export default function AssetActivityPanel({ symbol }: Props) {
  const assetQ = useQuery({
    queryKey: ['politiciansAsset', symbol],
    queryFn: () => api.politiciansAsset(symbol || ''),
    enabled: Boolean(symbol),
    staleTime: 60_000,
    retry: false,
  });

  if (!symbol) {
    return (
      <section className="glass-card p-5">
        <h2 className="text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
          Asset Activity
        </h2>
        <p className="mt-2 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
          Select a ticker from the disclosure feed to inspect asset-level context.
        </p>
      </section>
    );
  }

  if (assetQ.isLoading) return <LoadingSpinner text={`Loading ${symbol} politician activity...`} variant="cards" />;
  if (!assetQ.data || assetQ.data.status !== 'ok') return null;

  const data = assetQ.data as PoliticiansAssetResponse;
  const tracked = (data.recent_trades || []).some((row) => row.is_tracked_asset);
  const imbalance = data.buy_sell_imbalance || {};
  const purchases = Number(imbalance.buy_amount_mid_usd || 0);
  const sales = Number(imbalance.sell_amount_mid_usd || 0);
  const score = Number(data.activity?.politician_activity_score || 0);

  return (
    <section className="glass-card overflow-hidden">
      <div className="flex flex-col gap-3 border-b border-[var(--violet-8)] p-5 md:flex-row md:items-start md:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4" style={{ color: 'var(--accent-cyan)' }} />
            <h2 className="truncate text-[14px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
              {data.symbol} Activity
            </h2>
          </div>
          <p className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            Disclosure-date timeline · {data.total || 0} recent records · {data.unique_filer_count || 0} filers
          </p>
        </div>
        {tracked && (
          <Link
            to={`/charts/${encodeURIComponent(data.symbol || symbol)}`}
            className="inline-flex h-8 items-center justify-center gap-1 rounded-[8px] border px-3 text-[11px] hover:brightness-110"
            style={{ borderColor: 'var(--violet-8)', color: 'var(--accent-cyan)' }}
          >
            Open Chart <ExternalLink className="h-3 w-3" />
          </Link>
        )}
      </div>

      <div className="grid gap-4 p-5 lg:grid-cols-[1fr_1.2fr]">
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-2">
            <AmountCard label="Purchases" value={purchases} tone="buy" />
            <AmountCard label="Sales" value={sales} tone="sell" />
          </div>
          <div className="rounded-[8px] border border-[var(--violet-8)] p-3">
            <div className="flex items-center gap-2 text-[11px] font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              <Users className="h-3.5 w-3.5" /> Unique filers
            </div>
            <div className="mt-2 flex flex-wrap gap-1.5">
              {(data.unique_filers || []).slice(0, 8).map((filer) => (
                <span key={filer} className="rounded-[6px] border border-[var(--violet-8)] px-2 py-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                  {filer}
                </span>
              ))}
              {(data.unique_filers || []).length === 0 && <span className="text-[12px]" style={{ color: 'var(--text-muted)' }}>Unknown</span>}
            </div>
          </div>
          <div className="rounded-[8px] border border-[var(--violet-8)] p-3">
            <div className="text-[11px] font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
              Activity score
            </div>
            <div className="mt-2 text-[28px] font-semibold tabular-nums" style={{ color: score >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
              {score.toFixed(2)}
            </div>
            <div className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              Confidence: {Number(data.activity?.confidence || 0).toFixed(2)}
            </div>
          </div>
          <div className="rounded-[8px] border border-[var(--violet-8)] p-3 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
            Amount estimate: {formatUsd(Number(data.amount_estimates?.amount_mid_usd || 0))}
          </div>
        </div>

        <div className="space-y-4">
          <div>
            <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
              Disclosure-Date Timeline
            </h3>
            <div className="mt-3 space-y-2">
              {(data.disclosure_timeline || []).map((point) => (
                <div key={String(point.date)} className="rounded-[8px] border border-[var(--violet-8)] p-3">
                  <div className="flex items-center justify-between gap-3">
                    <span className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>{String(point.date)}</span>
                    <span className="text-[11px]" style={{ color: Number(point.net_amount_mid_usd || 0) >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                      {formatUsd(Number(point.net_amount_mid_usd || 0))}
                    </span>
                  </div>
                  <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]">
                    <div
                      className="h-full rounded-full"
                      style={{
                        width: `${Math.min(100, Math.max(6, Math.abs(Number(point.net_amount_mid_usd || 0)) / Math.max(1, purchases + sales) * 100))}%`,
                        background: Number(point.net_amount_mid_usd || 0) >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                      }}
                    />
                  </div>
                </div>
              ))}
              {(data.disclosure_timeline || []).length === 0 && <p className="text-[12px]" style={{ color: 'var(--text-muted)' }}>No disclosure dates available.</p>}
            </div>
          </div>

          <div className="border-t border-[var(--violet-8)] pt-4">
            <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
              Retrospective Transaction Dates
            </h3>
            <div className="mt-2 grid gap-2">
              {(data.recent_trades || []).slice(0, 5).map((row) => (
                <TradeMini key={String(row.trade_id || `${row.transaction_date}-${row.ticker}`)} row={row} />
              ))}
            </div>
          </div>
        </div>
      </div>
    </section>
  );
}

function AmountCard({ label, value, tone }: { label: string; value: number; tone: 'buy' | 'sell' }) {
  const positive = tone === 'buy';
  return (
    <div className="rounded-[8px] border p-3" style={{ borderColor: positive ? 'rgba(62,232,165,0.25)' : 'rgba(255,107,138,0.25)', background: positive ? 'var(--emerald-12)' : 'var(--rose-12)' }}>
      <div className="flex items-center gap-1 text-[11px] font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
        {positive ? <TrendingUp className="h-3.5 w-3.5" /> : <TrendingDown className="h-3.5 w-3.5" />} {label}
      </div>
      <div className="mt-2 text-[20px] font-semibold tabular-nums" style={{ color: positive ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
        {formatUsd(value)}
      </div>
    </div>
  );
}

function TradeMini({ row }: { row: PoliticiansTradeRow }) {
  const isBuy = ['purchase', 'received'].includes(String(row.transaction_type || '').toLowerCase());
  return (
    <div className="flex items-center justify-between gap-3 rounded-[8px] border border-[var(--violet-8)] px-3 py-2 text-[11px]">
      <span className="min-w-0 truncate" style={{ color: 'var(--text-secondary)' }}>
        {String(row.transaction_date || 'unknown')} · {String(row.filer_name || 'Unknown')}
      </span>
      <span className="shrink-0 font-semibold" style={{ color: isBuy ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
        {String(row.transaction_type || 'unknown')}
      </span>
    </div>
  );
}

function formatUsd(value: number): string {
  return Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', maximumFractionDigits: 1 }).format(value);
}
