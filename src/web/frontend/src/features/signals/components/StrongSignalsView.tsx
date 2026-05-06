import React, { useState } from 'react';
import { ChevronRight, Shield, TrendingDown, TrendingUp } from 'lucide-react';
import type { StrongSignalEntry } from '../../../api';
import SignalDetailPanel from '../../../components/SignalDetailPanel';
import { MomentumBadge } from '../../../components/SignalTableVisuals';
import type { SignalFilter } from '../utils';

/* ── Strong Signals View — Premium Cards ──────────────────────────── */
function StrongSignalPanel({ entries, accent, label, icon, onNavigateChart }: {
  entries: StrongSignalEntry[]; accent: string; label: string; icon: React.ReactNode;
  onNavigateChart: (sym: string) => void;
}) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const signalLabel = accent === '#10b981' ? 'STRONG BUY' : 'STRONG SELL';
  const avgRet = entries.length > 0 ? entries.reduce((s, e) => s + (e.exp_ret ?? 0) * 100, 0) / entries.length : 0;
  const avgPUp = entries.length > 0 ? entries.reduce((s, e) => s + (e.p_up ?? 0), 0) / entries.length : 0;

  return (
    <div className="glass-card overflow-hidden" style={{ borderTop: `2px solid ${accent}40` }}>
      <div className="px-5 py-3.5 flex items-center gap-3"
        style={{ background: `linear-gradient(135deg, ${accent}08 0%, transparent 60%)`, borderBottom: `1px solid ${accent}15` }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${accent}15` }}>
          {icon}
        </div>
        <div>
          <h3 className="text-sm font-semibold" style={{ color: accent }}>{label}</h3>
          <p className="text-[10px] text-[var(--text-muted)]">{entries.length} signals</p>
        </div>
        <div className="ml-auto flex items-center gap-4">
          <div className="text-right">
            <span className="text-[9px] text-[var(--text-muted)] block">Avg Return</span>
            <span className="text-[12px] font-bold tabular-nums" style={{ color: accent }}>
              {avgRet >= 0 ? '+' : ''}{avgRet.toFixed(1)}%
            </span>
          </div>
          <div className="text-right">
            <span className="text-[9px] text-[var(--text-muted)] block">Avg P(up)</span>
            <span className="text-[12px] font-bold tabular-nums" style={{ color: accent }}>
              {(avgPUp * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>
      {entries.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <Shield className="w-6 h-6 mx-auto mb-2" style={{ color: `${accent}30` }} />
          <p className="text-xs text-[var(--text-muted)]">No {label.toLowerCase()}</p>
        </div>
      ) : (
        <div className="divide-y divide-white/[0.03]">
          {entries.map((s, i) => {
            const retPct = s.exp_ret != null ? s.exp_ret * 100 : null;
            const isStandout = retPct != null && Math.abs(retPct) > 5;
            const ticker = s.asset_label?.includes('(') ? s.asset_label.split('(').pop()!.replace(')', '').trim() : (s.symbol || s.asset_label || '--');
            const company = s.asset_label?.includes('(') ? s.asset_label.split('(')[0].trim() : '';
            const isExpanded = expandedIdx === i;
            const horizonKey = s.horizon || '30';
            return (
              <React.Fragment key={i}>
                <button
                  type="button"
                  onClick={() => setExpandedIdx(p => (p === i ? null : i))}
                  aria-expanded={isExpanded}
                  className="w-full flex items-center gap-3 px-5 py-2.5 text-left transition-colors"
                  style={{
                    background: isExpanded ? `${accent}08` : 'transparent',
                    borderLeft: isExpanded ? `2px solid ${accent}` : '2px solid transparent',
                  }}
                >
                  {/* Rank */}
                  <span className="text-[10px] font-bold w-5 text-center tabular-nums" style={{ color: `${accent}60` }}>
                    {i + 1}
                  </span>
                  {/* Color bar */}
                  <div className="w-1 h-8 rounded-full flex-shrink-0" style={{ background: `${accent}50` }} />
                  {/* Asset info */}
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[12px] font-bold text-[#e2e8f0]">{ticker}</span>
                      <span className="text-[9px] px-1.5 py-0.5 rounded" style={{ background: 'var(--void-active)', color: 'var(--text-secondary)' }}>
                        {s.sector || 'Other'}
                      </span>
                    </div>
                    {company && (
                      <span className="text-[9px] text-[var(--text-muted)] truncate max-w-[180px] block leading-tight mt-0.5">{company}</span>
                    )}
                  </div>
                  {/* Horizon */}
                  <span className="text-[10px] px-2 py-0.5 rounded font-medium" style={{ background: 'var(--void-active)', color: 'var(--text-secondary)' }}>
                    {s.horizon || '--'}
                  </span>
                  {/* Return */}
                  <span className={`text-right min-w-[55px] tabular-nums font-bold ${isStandout ? 'text-[13px]' : 'text-[11px]'}`} style={{ color: accent }}>
                    {retPct != null ? `${retPct >= 0 ? '+' : ''}${retPct.toFixed(1)}%` : '--'}
                  </span>
                  {/* Probability bar */}
                  <div className="flex items-center gap-1.5 min-w-[65px]">
                    <div className="w-10 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${(s.p_up ?? 0) * 100}%`, background: accent }} />
                    </div>
                    <span className="text-[10px] tabular-nums text-[var(--text-secondary)]">
                      {s.p_up != null ? `${(s.p_up * 100).toFixed(0)}%` : '--'}
                    </span>
                  </div>
                  {/* Momentum */}
                  <MomentumBadge value={s.momentum} />
                  {/* Chevron */}
                  <ChevronRight
                    className="w-3.5 h-3.5 ml-1 transition-all duration-200 flex-shrink-0"
                    style={{
                      color: isExpanded ? accent : 'var(--text-muted)',
                      transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
                    }}
                  />
                </button>
                {isExpanded && (
                  <SignalDetailPanel
                    ticker={ticker}
                    signal={signalLabel}
                    momentum={s.momentum}
                    crashRisk={undefined}
                    horizonSignals={{ [horizonKey]: { exp_ret: s.exp_ret, p_up: s.p_up, label: signalLabel } } as any}
                    onNavigateChart={() => onNavigateChart(ticker)}
                  />
                )}
              </React.Fragment>
            );
          })}
        </div>
      )}
    </div>
  );
}

export default function StrongSignalsView({ strongBuy, strongSell, filter, onNavigateChart }: {
  strongBuy: StrongSignalEntry[];
  strongSell: StrongSignalEntry[];
  filter: SignalFilter;
  onNavigateChart: (sym: string) => void;
}) {
  const onlyBuy = filter === 'bullish' || filter === 'strong_buy' || filter === 'buy';
  const onlySell = filter === 'bearish' || filter === 'strong_sell' || filter === 'sell';
  const gridCls = !onlyBuy && !onlySell
    ? 'grid grid-cols-1 lg:grid-cols-2 gap-5'
    : 'grid grid-cols-1 gap-5';
  return (
    <div className={gridCls}>
      {!onlySell && (
        <StrongSignalPanel
          entries={strongBuy}
          accent="#10b981"
          label="Strong Buy Signals"
          icon={<TrendingUp className="w-4 h-4" style={{ color: '#10b981' }} />}
          onNavigateChart={onNavigateChart}
        />
      )}
      {!onlyBuy && (
        <StrongSignalPanel
          entries={strongSell}
          accent="#f43f5e"
          label="Strong Sell Signals"
          icon={<TrendingDown className="w-4 h-4" style={{ color: '#f43f5e' }} />}
          onNavigateChart={onNavigateChart}
        />
      )}
    </div>
  );
}
