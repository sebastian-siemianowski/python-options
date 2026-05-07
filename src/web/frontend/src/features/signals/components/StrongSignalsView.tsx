import React, { useEffect, useMemo, useState } from 'react';
import { ChevronRight, Shield, TrendingDown, TrendingUp } from 'lucide-react';
import type { StrongSignalEntry, SummaryRow } from '../../../api';
import SignalDetailPanel from '../../../components/SignalDetailPanel';
import { MomentumBadge } from '../../../components/SignalTableVisuals';
import { smaQualityTone } from '../theme';
import { extractTicker, type SignalFilter } from '../utils';

type SignalTabKey = 'strong_buy' | 'buy' | 'strong_sell' | 'sell';
type SignalLabel = 'STRONG BUY' | 'BUY' | 'STRONG SELL' | 'SELL';

const normalizeLabel = (label: string | undefined | null) =>
  (label || '').toUpperCase().replace(/\s+/g, '_');

const horizonLabel = (raw: string): string => {
  const n = Number.parseInt(raw, 10);
  if (!Number.isFinite(n)) return raw;
  if (n === 1) return '1D';
  if (n === 3) return '3D';
  if (n === 7) return '1W';
  if (n === 30 || n === 21) return '1M';
  if (n === 90 || n === 63) return '3M';
  if (n === 180 || n === 126) return '6M';
  if (n === 365 || n === 252) return '12M';
  return `${n}D`;
};

const toSignalEntry = (row: SummaryRow, signalLabel: SignalLabel): StrongSignalEntry | null => {
  const entries = Object.entries(row.horizon_signals || {});
  if (entries.length === 0) return null;
  const matching = entries.filter(([, sig]) => normalizeLabel(sig?.label) === normalizeLabel(signalLabel));
  const pool = matching.length > 0 ? matching : entries;
  const isSell = signalLabel.includes('SELL');
  const [horizon, signal] = [...pool].sort((a, b) => {
    const ar = a[1]?.exp_ret ?? 0;
    const br = b[1]?.exp_ret ?? 0;
    return isSell ? ar - br : br - ar;
  })[0];
  return {
    symbol: extractTicker(row.asset_label),
    asset_label: row.asset_label,
    sector: row.sector,
    horizon: horizonLabel(horizon),
    p_up: signal?.p_up ?? 0,
    exp_ret: signal?.exp_ret ?? 0,
    momentum: row.momentum_score ?? 0,
  };
};

const entriesForLabel = (rows: SummaryRow[], signalLabel: SignalLabel): StrongSignalEntry[] => (
  rows
    .filter((row) => normalizeLabel(row.nearest_label) === normalizeLabel(signalLabel))
    .map((row) => toSignalEntry(row, signalLabel))
    .filter((entry): entry is StrongSignalEntry => entry != null)
);

const lookupQualityScore = (
  scores: Record<string, number>,
  ticker: string | undefined | null,
  assetLabel?: string,
): number | null => {
  const rawTicker = String(ticker || '').trim();
  const rawLabel = String(assetLabel || '').trim();
  const fromLabel = rawLabel.match(/\(([^)]+)\)\s*$/)?.[1]?.trim() || '';
  const variants = new Set<string>();
  for (const value of [rawTicker, fromLabel, rawLabel].filter(Boolean)) {
    const upper = value.toUpperCase();
    variants.add(value);
    variants.add(upper);
    variants.add(upper.replace(/-/g, '.'));
    variants.add(upper.replace(/\./g, '-'));
    variants.add(upper.replace(/=/g, '_'));
    variants.add(upper.replace(/_/g, '='));
  }
  for (const key of variants) {
    const score = scores[key];
    if (typeof score === 'number' && Number.isFinite(score)) return score;
  }
  return null;
};

function BusinessQualityRing({ score, accent }: { score: number | null | undefined; accent: string }) {
  const hasScore = score != null && Number.isFinite(score);
  const pct = hasScore ? Math.max(0, Math.min(100, Math.round(score))) : 0;
  const tone = smaQualityTone(hasScore ? pct : null);
  const size = 40;
  const stroke = 4;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - pct / 100);

  return (
    <div
      className="strong-quality-ring relative mx-auto flex h-[44px] w-[108px] items-center justify-center"
      title={hasScore ? `Business quality: ${pct} (${tone.label})` : 'Business quality unavailable'}
      style={{ ['--quality-accent' as string]: tone.color, ['--signal-accent' as string]: accent }}
    >
      <div
        className="relative flex h-10 w-10 shrink-0 items-center justify-center overflow-hidden rounded-full"
        style={{
          background:
            `radial-gradient(circle at 50% 38%, ${tone.color}20 0%, rgba(255,255,255,0.030) 46%, rgba(0,0,0,0.26) 100%)`,
          border: `1px solid ${tone.border}`,
          boxShadow: [
            `0 0 0 1px ${tone.color}10`,
            `0 0 16px -10px ${tone.color}`,
            `0 0 18px -16px ${accent}`,
            'inset 0 1px 0 rgba(255,255,255,0.10)',
            '0 10px 18px -16px rgba(0,0,0,0.95)',
          ].join(', '),
        }}
      >
        <svg className="absolute inset-0 h-full w-full -rotate-90" viewBox={`0 0 ${size} ${size}`} aria-hidden>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.070)"
            strokeWidth={stroke}
          />
          {hasScore && (
            <circle
              cx={size / 2}
              cy={size / 2}
              r={radius}
              fill="none"
              stroke={tone.color}
              strokeWidth={stroke}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={dashOffset}
              style={{
                transition: 'stroke-dashoffset 520ms cubic-bezier(0.16, 1, 0.3, 1), stroke 220ms ease',
              }}
            />
          )}
        </svg>
        <span
          className="relative z-[1] text-[12px] font-extrabold leading-none tabular-nums"
          style={{ color: hasScore ? tone.color : 'var(--text-muted)' }}
        >
          {hasScore ? pct : '—'}
        </span>
      </div>
    </div>
  );
}

/* ── Strong Signals View — Premium Cards ──────────────────────────── */
function StrongSignalPanel({ entries, accent, label, icon, qualityScores, onNavigateChart }: {
  entries: StrongSignalEntry[]; accent: string; label: string; icon: React.ReactNode;
  qualityScores: Record<string, number>;
  onNavigateChart: (sym: string) => void;
}) {
  const [expandedIdx, setExpandedIdx] = useState<number | null>(null);
  const signalLabel = label.replace(' Signals', '').toUpperCase() as SignalLabel;
  const isSell = signalLabel.includes('SELL');
  useEffect(() => setExpandedIdx(null), [signalLabel]);
  const sortedEntries = useMemo(() => {
    const direction = isSell ? 1 : -1;
    return [...entries].sort((a, b) => direction * ((a.exp_ret ?? 0) - (b.exp_ret ?? 0)));
  }, [entries, isSell]);

  return (
    <div className="strong-signal-inner-panel overflow-hidden" style={{ borderTop: `1px solid ${accent}26` }}>
      <div className="strong-signal-panel-head px-5 py-3.5 flex items-center gap-3"
        style={{ background: `linear-gradient(135deg, ${accent}08 0%, transparent 60%)`, borderBottom: `1px solid ${accent}15` }}>
        <div className="w-8 h-8 rounded-lg flex items-center justify-center" style={{ background: `${accent}15` }}>
          {icon}
        </div>
        <div>
          <h3 className="text-sm font-semibold" style={{ color: accent }}>{label}</h3>
        </div>
      </div>
      {sortedEntries.length === 0 ? (
        <div className="px-5 py-8 text-center">
          <Shield className="w-6 h-6 mx-auto mb-2" style={{ color: `${accent}30` }} />
          <p className="text-xs text-[var(--text-muted)]">No {label.toLowerCase()}</p>
        </div>
      ) : (
        <div className="flex flex-col gap-1.5 px-3 py-3">
          <div
            className="hidden md:flex items-center gap-3 px-2 py-1.5 text-[9px] font-bold uppercase tracking-[0.14em]"
            style={{
              color: 'var(--text-muted)',
            }}
          >
            <span className="w-5 text-center">Rank</span>
            <span className="w-1" aria-hidden />
            <span className="basis-[260px] max-w-[260px] min-w-0">Ticker / sector</span>
            <span className="w-[108px] text-center tracking-[0.08em]">Business Quality</span>
            <span className="w-[58px] text-center">Horizon</span>
            <span className="min-w-[66px] text-right">Exp. ret</span>
            <span className="min-w-[76px] text-left">{isSell ? 'P(down)' : 'P(up)'}</span>
            <span className="min-w-[68px] text-center">Momentum</span>
            <span className="w-4" aria-hidden />
          </div>
          {sortedEntries.map((s, i) => {
            const retPct = s.exp_ret != null ? s.exp_ret * 100 : null;
            const isStandout = retPct != null && Math.abs(retPct) > 5;
            const ticker = s.asset_label?.includes('(') ? s.asset_label.split('(').pop()!.replace(')', '').trim() : (s.symbol || s.asset_label || '--');
            const company = s.asset_label?.includes('(') ? s.asset_label.split('(')[0].trim() : '';
            const isExpanded = expandedIdx === i;
            const horizonKey = s.horizon || '30';
            const directionalProb = isSell ? 1 - (s.p_up ?? 0) : (s.p_up ?? 0);
            const qualityScore = lookupQualityScore(qualityScores, ticker, s.asset_label);
            return (
              <React.Fragment key={i}>
                <button
                  type="button"
                  onClick={() => setExpandedIdx(p => (p === i ? null : i))}
                  aria-expanded={isExpanded}
                  className="strong-signal-row-entry relative isolate flex w-full items-center gap-3 overflow-hidden rounded-[14px] px-3.5 py-2.5 text-left transition-all duration-200"
                  data-active={isExpanded || undefined}
                  style={{
                    ['--row-accent' as string]: accent,
                    animationDelay: `${Math.min(i, 10) * 28}ms`,
                    background: isExpanded
                      ? `linear-gradient(135deg, ${accent}13, rgba(167,139,250,0.055) 42%, rgba(255,255,255,0.018))`
                      : 'linear-gradient(180deg, rgba(255,255,255,0.030), rgba(255,255,255,0.010))',
                    border: `1px solid ${isExpanded ? `${accent}38` : 'rgba(255,255,255,0.055)'}`,
                    borderLeft: `2px solid ${isExpanded ? accent : `${accent}30`}`,
                    boxShadow: isExpanded
                      ? `0 16px 34px -30px ${accent}, inset 0 1px 0 rgba(255,255,255,0.070)`
                      : 'inset 0 1px 0 rgba(255,255,255,0.040)',
                  }}
                >
                  {/* Rank */}
                  <span
                    className="inline-flex h-7 w-7 shrink-0 items-center justify-center rounded-[9px] text-[10px] font-extrabold tabular-nums"
                    style={{
                      color: isExpanded ? accent : `${accent}95`,
                      background: isExpanded ? `${accent}16` : 'rgba(255,255,255,0.030)',
                      border: `1px solid ${isExpanded ? `${accent}34` : 'rgba(255,255,255,0.045)'}`,
                    }}
                  >
                    {i + 1}
                  </span>
                  {/* Color bar */}
                  <div className="h-9 w-1 flex-shrink-0 rounded-full" style={{ background: `linear-gradient(180deg, ${accent}, ${accent}44)` }} />
                  {/* Asset info */}
                  <div className="flex-1 min-w-0 md:flex-none md:basis-[260px] md:max-w-[260px]">
                    <div className="flex items-center gap-1.5">
                      <span className="text-[12.5px] font-extrabold tracking-[0.01em] text-[#f1f5f9]">{ticker}</span>
                      <span
                        className="truncate rounded-[7px] px-1.5 py-0.5 text-[8.5px] font-semibold"
                        style={{
                          background: `linear-gradient(180deg, ${accent}10, rgba(255,255,255,0.018))`,
                          color: 'var(--text-secondary)',
                          border: '1px solid rgba(255,255,255,0.045)',
                        }}
                      >
                        {s.sector || 'Other'}
                      </span>
                    </div>
                    {company && (
                      <span className="text-[9px] text-[var(--text-muted)] truncate max-w-[180px] block leading-tight mt-0.5">{company}</span>
                    )}
                  </div>
                  <div className="w-[108px] shrink-0">
                    <BusinessQualityRing score={qualityScore} accent={accent} />
                  </div>
                  {/* Horizon */}
                  <span
                    className="w-[58px] rounded-[8px] px-2 py-1 text-center text-[10px] font-bold"
                    style={{
                      background: 'linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.015))',
                      color: 'var(--text-secondary)',
                      border: '1px solid rgba(255,255,255,0.045)',
                    }}
                  >
                    {s.horizon || '--'}
                  </span>
                  {/* Return */}
                  <span className={`text-right min-w-[66px] tabular-nums font-bold ${isStandout ? 'text-[13px]' : 'text-[11px]'}`} style={{ color: accent }}>
                    {retPct != null ? `${retPct >= 0 ? '+' : ''}${retPct.toFixed(1)}%` : '--'}
                  </span>
                  {/* Probability bar */}
                  <div className="flex items-center gap-1.5 min-w-[76px]">
                    <div className="w-10 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                      <div className="h-full rounded-full" style={{ width: `${directionalProb * 100}%`, background: accent }} />
                    </div>
                    <span className="text-[10px] tabular-nums text-[var(--text-secondary)]">
                      {s.p_up != null ? `${(directionalProb * 100).toFixed(0)}%` : '--'}
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
                    horizonSignals={{ [horizonKey]: { exp_ret: s.exp_ret, p_up: s.p_up, label: signalLabel } }}
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

export default function StrongSignalsView({ rows, filter, qualityScores, onNavigateChart }: {
  rows: SummaryRow[];
  filter: SignalFilter;
  qualityScores: Record<string, number>;
  onNavigateChart: (sym: string) => void;
}) {
  const strongBuy = useMemo(() => entriesForLabel(rows, 'STRONG BUY'), [rows]);
  const buy = useMemo(() => entriesForLabel(rows, 'BUY'), [rows]);
  const strongSell = useMemo(() => entriesForLabel(rows, 'STRONG SELL'), [rows]);
  const sell = useMemo(() => entriesForLabel(rows, 'SELL'), [rows]);
  const onlyBuy = filter === 'bullish' || filter === 'strong_buy' || filter === 'buy';
  const onlySell = filter === 'bearish' || filter === 'strong_sell' || filter === 'sell';
  const [activeTab, setActiveTab] = useState<SignalTabKey>('strong_buy');

  const tabs = useMemo(() => [
    {
      key: 'strong_buy' as const,
      label: 'Strong Buy',
      short: 'Strong Buy',
      entries: strongBuy,
      accent: '#10b981',
      icon: <TrendingUp className="w-4 h-4" />,
    },
    {
      key: 'buy' as const,
      label: 'Buy',
      short: 'Buy',
      entries: buy,
      accent: '#34d399',
      icon: <TrendingUp className="w-4 h-4" />,
    },
    {
      key: 'strong_sell' as const,
      label: 'Strong Sell',
      short: 'Strong Sell',
      entries: strongSell,
      accent: '#f43f5e',
      icon: <TrendingDown className="w-4 h-4" />,
    },
    {
      key: 'sell' as const,
      label: 'Sell',
      short: 'Sell',
      entries: sell,
      accent: '#fb7185',
      icon: <TrendingDown className="w-4 h-4" />,
    },
  ], [buy, sell, strongBuy, strongSell]);
  const visibleTabs = useMemo(() => tabs.filter((tab) => {
    if (filter === 'strong_buy') return tab.key === 'strong_buy';
    if (filter === 'buy') return tab.key === 'buy';
    if (filter === 'strong_sell') return tab.key === 'strong_sell';
    if (filter === 'sell') return tab.key === 'sell';
    if (onlyBuy) return tab.key === 'strong_buy' || tab.key === 'buy';
    if (onlySell) return tab.key === 'strong_sell' || tab.key === 'sell';
    return true;
  }), [filter, onlyBuy, onlySell, tabs]);
  const current = visibleTabs.find((tab) => tab.key === activeTab) ?? visibleTabs[0] ?? tabs[0];
  useEffect(() => {
    if (!visibleTabs.some((tab) => tab.key === activeTab)) {
      setActiveTab(visibleTabs[0]?.key ?? 'strong_buy');
    }
  }, [activeTab, visibleTabs]);
  const activeAccent = current.accent;

  return (
    <section
      className="strong-signals-shell glass-card overflow-hidden"
      style={{
        ['--strong-accent' as string]: activeAccent,
        border: `1px solid ${activeAccent}28`,
        boxShadow: `0 22px 60px -46px ${activeAccent}, 0 12px 36px -26px rgba(0,0,0,0.86), inset 0 1px 0 rgba(255,255,255,0.04)`,
      }}
    >
      <div
        className="strong-signals-command flex flex-wrap items-center gap-3 px-4 py-3"
        style={{
          background: `radial-gradient(760px 180px at 0% -90%, ${activeAccent}18, transparent 64%), rgba(255,255,255,0.012)`,
          borderBottom: '1px solid rgba(255,255,255,0.045)',
        }}
      >
        <div className="flex min-w-0 items-center gap-2 mr-1">
          <span
            className="inline-flex h-8 w-8 items-center justify-center rounded-xl"
            style={{
              color: activeAccent,
              background: `${activeAccent}16`,
              border: `1px solid ${activeAccent}34`,
              boxShadow: `0 0 18px -10px ${activeAccent}`,
            }}
          >
            <Shield className="w-4 h-4" />
          </span>
          <div>
            <h2 className="text-[13.5px] font-semibold tracking-tight text-[var(--text-primary)]">
              Strong Signals
            </h2>
            <p className="text-[10.5px] text-[var(--text-muted)]">
              One panel · tabbed buy/sell decisions
            </p>
          </div>
        </div>

        <div
          className="grid flex-1 min-w-[520px] items-center gap-1.5 rounded-[16px] p-1"
          style={{
            gridTemplateColumns: `repeat(${visibleTabs.length}, minmax(0, 1fr))`,
            background: 'linear-gradient(180deg, rgba(255,255,255,0.040), rgba(255,255,255,0.014))',
            border: '1px solid rgba(255,255,255,0.065)',
            boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.05), 0 12px 30px -28px rgba(0,0,0,0.88)',
          }}
        >
          {visibleTabs.map((tab) => {
            const on = current.key === tab.key;
            return (
              <button
                key={tab.key}
                type="button"
                onClick={() => setActiveTab(tab.key)}
                aria-pressed={on}
                className="strong-signal-tab group relative isolate inline-flex h-9 min-w-0 items-center justify-center gap-1.5 overflow-hidden rounded-[11px] px-2 text-left transition-all duration-200 active:scale-[0.985]"
                data-active={on || undefined}
                style={{
                  color: on ? '#f8fafc' : 'var(--text-secondary)',
                  background: on
                    ? `linear-gradient(180deg, ${tab.accent}24, rgba(255,255,255,0.032) 55%, ${tab.accent}10)`
                    : 'linear-gradient(180deg, rgba(255,255,255,0.024), rgba(255,255,255,0.010))',
                  border: `1px solid ${on ? `${tab.accent}5f` : 'rgba(255,255,255,0.045)'}`,
                  boxShadow: on
                    ? `0 0 0 1px ${tab.accent}1c inset, 0 8px 20px -16px ${tab.accent}, 0 0 18px -13px ${tab.accent}`
                    : 'inset 0 1px 0 rgba(255,255,255,0.03)',
                }}
              >
                {on && (
                  <span
                    aria-hidden
                    className="absolute inset-x-2 top-0 h-px"
                    style={{ background: `linear-gradient(90deg, transparent, ${tab.accent}, rgba(255,255,255,0.65), transparent)` }}
                  />
                )}
                <span
                  className="inline-flex h-5 w-5 shrink-0 items-center justify-center rounded-[7px]"
                  style={{
                    color: on ? tab.accent : 'var(--text-muted)',
                    background: on ? `${tab.accent}17` : 'rgba(255,255,255,0.028)',
                    filter: on ? `drop-shadow(0 0 5px ${tab.accent}7a)` : 'none',
                  }}
                >
                  {tab.icon}
                </span>
                <span className="min-w-0 truncate whitespace-nowrap text-[10.5px] font-bold uppercase tracking-[0.055em]">
                  {tab.short}
                </span>
                <span
                  className="inline-flex h-[19px] min-w-[25px] items-center justify-center rounded-[7px] px-1.5 text-[10px] font-bold tabular-nums"
                  style={{
                    background: on ? `${tab.accent}22` : 'rgba(255,255,255,0.038)',
                    color: on ? '#ffffff' : tab.accent,
                    border: `1px solid ${on ? `${tab.accent}30` : 'rgba(255,255,255,0.040)'}`,
                  }}
                >
                  {tab.entries.length}
                </span>
              </button>
            );
          })}
        </div>
      </div>

      <div key={current.key} className="strong-signal-panel-stage p-3">
        <StrongSignalPanel
          key={current.key}
          entries={current.entries}
          accent={current.accent}
          label={`${current.label} Signals`}
          icon={current.key === 'strong_buy' || current.key === 'buy' ? <TrendingUp className="w-4 h-4" style={{ color: current.accent }} /> : <TrendingDown className="w-4 h-4" style={{ color: current.accent }} />}
          qualityScores={qualityScores}
          onNavigateChart={onNavigateChart}
        />
      </div>
    </section>
  );
}
