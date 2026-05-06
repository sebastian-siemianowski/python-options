import { useMemo } from 'react';
import type { SignalStats, SummaryRow } from '../../../api';
import type { WSStatus } from '../../../hooks/useWebSocket';

interface SignalsHeroProps {
  stats: SignalStats | undefined;
  rows: SummaryRow[];
  horizons: number[];
  filteredCount: number;
  wsStatus: WSStatus;
}

export default function SignalsHero({
  stats,
  rows,
  horizons,
  filteredCount,
  wsStatus,
}: SignalsHeroProps) {
  const rowCounts = useMemo(() => {
    const counts = { strongBuy: 0, buy: 0, hold: 0, sell: 0, strongSell: 0 };
    for (const row of rows) {
      const label = (row.nearest_label || 'HOLD').toUpperCase().replace(/\s+/g, '_');
      if (label === 'STRONG_BUY') counts.strongBuy += 1;
      else if (label === 'BUY') counts.buy += 1;
      else if (label === 'SELL') counts.sell += 1;
      else if (label === 'STRONG_SELL') counts.strongSell += 1;
      else counts.hold += 1;
    }
    return counts;
  }, [rows]);

  const total = stats?.total_assets ?? rows.length;
  const strongBuy = stats?.strong_buy_signals ?? rowCounts.strongBuy;
  const buy = stats?.buy_signals ?? (rowCounts.buy + rowCounts.strongBuy);
  const hold = stats?.hold_signals ?? rowCounts.hold;
  const sell = stats?.sell_signals ?? (rowCounts.sell + rowCounts.strongSell);
  const strongSell = stats?.strong_sell_signals ?? rowCounts.strongSell;
  const conviction = strongBuy + strongSell;
  const bullishPct = total > 0 ? (buy / total) * 100 : 0;
  const bearishPct = total > 0 ? (sell / total) * 100 : 0;
  const neutralPct = Math.max(0, 100 - bullishPct - bearishPct);

  const wsColor =
    wsStatus === 'connected' ? '#10b981' :
    wsStatus === 'connecting' ? '#f59e0b' :
    '#64748b';

  return (
    <div
      className="hero-surface fade-up relative mb-6 overflow-hidden"
      style={{ borderRadius: 28, padding: '28px 32px' }}
    >
      <div className="flex flex-wrap items-start justify-between gap-8">
        <div className="min-w-[240px] flex-1">
          <div className="label-micro mb-2 flex items-center gap-2">
            <span
              className="h-1.5 w-1.5 rounded-full"
              style={{ background: wsColor, boxShadow: wsStatus === 'connected' ? `0 0 8px ${wsColor}` : 'none' }}
            />
            <span>Signals Engine · {wsStatus === 'connected' ? 'LIVE' : wsStatus.toUpperCase()}</span>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="num-hero text-white">{conviction}</span>
            <span className="text-[13px] tabular-nums text-[var(--text-muted)]">
              of {total} · high conviction
            </span>
          </div>
          <div className="mt-1 text-[12px] tabular-nums text-[var(--text-secondary)]">
            {filteredCount === rows.length
              ? <>{horizons.length} horizons active</>
              : <><span className="font-medium text-white">{filteredCount}</span> shown / {rows.length} total</>
            }
          </div>
        </div>

        <div className="min-w-[320px] max-w-[560px] flex-1">
          <div
            className="mb-3 flex overflow-hidden rounded-full"
            style={{ height: 8, background: 'rgba(255,255,255,0.04)' }}
            title={`${bullishPct.toFixed(1)}% bullish · ${neutralPct.toFixed(1)}% neutral · ${bearishPct.toFixed(1)}% bearish`}
          >
            <div style={{ width: `${bullishPct}%`, background: 'linear-gradient(90deg,#10b981,#6ee7b7)', transition: 'width 600ms ease-out' }} />
            <div style={{ width: `${neutralPct}%`, background: 'rgba(255,255,255,0.06)', transition: 'width 600ms ease-out' }} />
            <div style={{ width: `${bearishPct}%`, background: 'linear-gradient(90deg,#fca5a5,#f43f5e)', transition: 'width 600ms ease-out' }} />
          </div>
          <div className="grid grid-cols-5 gap-0">
            <HeroStat label="Strong Buy" value={strongBuy} color="#10b981" />
            <HeroStat label="Buy" value={Math.max(0, buy - strongBuy)} color="#6ee7b7" divider />
            <HeroStat label="Hold" value={hold} color="#94a3b8" divider />
            <HeroStat label="Sell" value={Math.max(0, sell - strongSell)} color="#fca5a5" divider />
            <HeroStat label="Strong Sell" value={strongSell} color="#f43f5e" divider />
          </div>
        </div>
      </div>
    </div>
  );
}

function HeroStat({ label, value, color, divider }: { label: string; value: number; color: string; divider?: boolean }) {
  return (
    <div className={`flex flex-col items-start px-3 ${divider ? 'stat-col-divider' : ''}`}>
      <span className="num-display text-[22px]" style={{ color }}>{value}</span>
      <span className="label-micro mt-1">{label}</span>
    </div>
  );
}
