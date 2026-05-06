import React, { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, ArrowDown, ArrowUp, BarChart3, ChevronDown, ChevronUp, Clock, ExternalLink, Eye, Filter, Layers, Search, Shield, Target, TrendingDown, TrendingUp, X } from 'lucide-react';
import type { EmaState, HighConvictionSignal } from '../../../api';
import SignalDetailPanel from '../../../components/SignalDetailPanel';
import { formatHorizon } from '../../../utils/horizons';
import { smaQualityTone } from '../theme';
/* ── High Conviction Panel — full positions with rich data ────────── */
type HCSortCol = 'ticker' | 'quality' | 'exp_ret' | 'p_up' | 'strength' | 'sector';
type HCSortDir = 'asc' | 'desc';

interface GroupedTicker {
  ticker: string;
  asset_label: string;
  sector: string;
  signals: HighConvictionSignal[];
  bestReturn: number;
  avgPUp: number;
  maxStrength: number;
}

const lookupBusinessQuality = (
  scores: Record<string, number>,
  ticker: string | undefined | null,
  assetLabel?: string,
): number | null => {
  const rawTicker = String(ticker || '').trim();
  const rawLabel = String(assetLabel || '').trim();
  const fromLabel = rawLabel.match(/\(([^)]+)\)\s*$/)?.[1]?.trim() || '';
  const base = [rawTicker, fromLabel, rawLabel].filter(Boolean);
  const variants = new Set<string>();
  for (const value of base) {
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

function KpiCell({
  label,
  value,
  color,
  icon,
  dividerRight,
}: {
  label: string;
  value: string;
  color: string;
  icon?: React.ReactNode;
  dividerRight?: boolean;
}) {
  return (
    <div
      className="px-4 py-3 flex flex-col gap-1"
      style={{
        borderRight: dividerRight ? '1px solid rgba(255,255,255,0.05)' : 'none',
      }}
    >
      <div
        className="flex items-center gap-1.5 text-[9px] uppercase font-semibold"
        style={{ color: 'var(--text-muted)', letterSpacing: '0.12em' }}
      >
        {icon && <span style={{ color }}>{icon}</span>}
        {label}
      </div>
      <div
        className="text-[20px] font-bold tabular-nums tracking-tight"
        style={{ color, letterSpacing: '-0.02em', lineHeight: 1 }}
      >
        {value}
      </div>
    </div>
  );
}

function ArcGauge({
  value,
  color,
  size = 26,
}: {
  value: number; // 0..1
  color: string;
  size?: number;
}) {
  const clamped = Math.max(0, Math.min(1, value));
  const stroke = Math.max(2, Math.round(size * 0.13));
  const r = (size - stroke) / 2;
  const c = size / 2;
  const circ = 2 * Math.PI * r;
  const dash = clamped * circ;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} style={{ display: 'block' }}>
      <circle cx={c} cy={c} r={r} fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth={stroke} />
      <circle
        cx={c}
        cy={c}
        r={r}
        fill="none"
        stroke={color}
        strokeWidth={stroke}
        strokeLinecap="round"
        strokeDasharray={`${dash} ${circ - dash}`}
        transform={`rotate(-90 ${c} ${c})`}
        style={{ transition: 'stroke-dasharray 320ms cubic-bezier(0.2, 0.8, 0.2, 1)' }}
      />
    </svg>
  );
}

function BusinessQualityRing({
  score,
  compact = false,
}: {
  score: number | null | undefined;
  compact?: boolean;
}) {
  const hasScore = score != null && Number.isFinite(score);
  const pct = hasScore ? Math.max(0, Math.min(100, Math.round(score))) : 0;
  const tone = smaQualityTone(hasScore ? pct : null);
  const size = compact ? 46 : 50;
  const stroke = compact ? 4.2 : 4.6;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - pct / 100);
  const ringShadow = [
    tone.glow !== 'none' ? tone.glow : null,
    'inset 0 1px 0 rgba(255,255,255,0.08)',
    '0 12px 24px -18px rgba(0,0,0,0.95)',
  ].filter(Boolean).join(', ');

  return (
    <div
      className="inline-flex items-center gap-2"
      title={hasScore ? `Business quality: ${pct} (${tone.label})` : 'Business quality unavailable'}
    >
      <div
        className="relative shrink-0 rounded-full"
        style={{
          width: size,
          height: size,
          background:
            `radial-gradient(circle at 50% 42%, ${tone.color}18 0%, rgba(255,255,255,0.018) 50%, rgba(0,0,0,0.18) 100%)`,
          border: `1px solid ${tone.border}`,
          boxShadow: ringShadow,
        }}
      >
        <svg className="absolute inset-0 h-full w-full -rotate-90" viewBox={`0 0 ${size} ${size}`} aria-hidden>
          <circle
            cx={size / 2}
            cy={size / 2}
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.075)"
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
                filter: `drop-shadow(0 0 5px ${tone.color}66)`,
                transition: 'stroke-dashoffset 420ms cubic-bezier(.2,.8,.2,1)',
              }}
            />
          )}
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span
            className="text-[13px] font-extrabold tabular-nums leading-none"
            style={{ color: hasScore ? tone.color : 'var(--text-muted)' }}
          >
            {hasScore ? pct : '—'}
          </span>
          <span
            className="mt-[1px] text-[6.8px] font-bold uppercase leading-none"
            style={{ color: hasScore ? tone.color : 'var(--text-muted)', opacity: 0.78 }}
          >
            Q
          </span>
        </div>
      </div>
      {!compact && (
        <div className="hidden min-w-0 flex-col leading-tight lg:flex">
          <span className="text-[8px] font-semibold uppercase text-[var(--text-muted)]">
            Business
          </span>
          <span className="text-[10px] font-bold" style={{ color: tone.color }}>
            {tone.label}
          </span>
        </div>
      )}
    </div>
  );
}

function SegmentedMeter({
  value,
  color,
  segments = 5,
}: {
  value: number; // 0..1
  color: string;
  segments?: number;
}) {
  const lit = Math.round(Math.max(0, Math.min(1, value)) * segments);
  return (
    <div className="inline-flex items-center gap-[3px]">
      {Array.from({ length: segments }).map((_, i) => {
        const on = i < lit;
        return (
          <span
            key={i}
            className="rounded-sm"
            style={{
              width: 6,
              height: 10,
              background: on ? color : 'rgba(255,255,255,0.07)',
              boxShadow: on ? `0 0 6px ${color}66` : 'none',
              opacity: on ? 1 - i * 0.08 : 1,
              transition: 'background-color 200ms, box-shadow 200ms',
            }}
          />
        );
      })}
    </div>
  );
}

// ─── Premium EMA "below" filter bar ─────────────────────────────────────
//
// Three iOS-style toggle pills — Below 9 / 50 / 600 — each with a live
// match count, a soft inset glow when active, and a sliding accent dot.
// Right edge surfaces total matches and a "Clear" affordance.
function EmaFilterBar({
  accent,
  accentSoft,
  filters,
  onChange,
  counts,
  anyActive,
  onClear,
  emaLoaded,
  matchingTotal,
}: {
  accent: string;
  accentSoft: string;
  filters: { p9: boolean; p50: boolean; p600: boolean };
  onChange: (next: { p9: boolean; p50: boolean; p600: boolean }) => void;
  counts: { c9: number; c50: number; c600: number; withData: number; total: number };
  anyActive: boolean;
  onClear: () => void;
  emaLoaded: boolean;
  matchingTotal: number;
}) {
  const items: Array<{
    key: 'p9' | 'p50' | 'p600';
    label: string;
    period: string;
    count: number;
  }> = [
    { key: 'p9',   label: 'Below', period: 'EMA 9',   count: counts.c9 },
    { key: 'p50',  label: 'Below', period: 'EMA 50',  count: counts.c50 },
    { key: 'p600', label: 'Below', period: 'EMA 600', count: counts.c600 },
  ];

  return (
    <div
      className="px-6 py-3 flex items-center gap-3 flex-wrap"
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.018) 0%, rgba(255,255,255,0.005) 100%)',
        borderTop: '1px solid rgba(255,255,255,0.04)',
        borderBottom: '1px solid rgba(255,255,255,0.04)',
      }}
    >
      {/* Section label */}
      <div className="flex items-center gap-1.5 pr-1">
        <Filter className="w-3 h-3 text-[var(--text-muted)]" />
        <span
          className="text-[9.5px] font-semibold uppercase tracking-[0.14em]"
          style={{ color: 'var(--text-muted)' }}
        >
          Trend Filters
        </span>
      </div>

      {/* Pills */}
      <div className="flex items-center gap-1.5">
        {items.map((it) => {
          const on = filters[it.key];
          const empty = emaLoaded && it.count === 0;
          return (
            <button
              key={it.key}
              type="button"
              onClick={() => onChange({ ...filters, [it.key]: !on })}
              disabled={!emaLoaded}
              aria-pressed={on}
              className="group relative inline-flex items-center gap-1.5 rounded-full pl-2.5 pr-1.5 py-1 transition-all"
              style={{
                background: on
                  ? `linear-gradient(180deg, ${accent}28, ${accent}12)`
                  : 'rgba(255,255,255,0.025)',
                border: `1px solid ${on ? accent + '70' : 'rgba(255,255,255,0.06)'}`,
                boxShadow: on
                  ? `0 0 0 1px ${accent}25 inset, 0 6px 18px -8px ${accent}80, 0 0 24px -6px ${accent}55`
                  : '0 1px 0 rgba(255,255,255,0.03) inset',
                color: on ? '#fff' : 'var(--text-secondary)',
                cursor: emaLoaded ? 'pointer' : 'wait',
                opacity: emaLoaded ? 1 : 0.5,
                transition: 'background 220ms cubic-bezier(.2,.8,.2,1), border-color 220ms, box-shadow 220ms, color 220ms',
              }}
              title={
                emaLoaded
                  ? `Show only tickers trading below ${it.period}`
                  : 'Loading EMA data…'
              }
            >
              {/* Accent dot — fades + scales when active */}
              <span
                aria-hidden
                className="rounded-full"
                style={{
                  width: 6,
                  height: 6,
                  background: on ? accent : 'rgba(255,255,255,0.18)',
                  boxShadow: on ? `0 0 8px ${accent}` : 'none',
                  transform: on ? 'scale(1.05)' : 'scale(1)',
                  transition: 'background 220ms, box-shadow 220ms, transform 220ms',
                }}
              />
              <span
                className="text-[10px] font-medium uppercase tracking-[0.1em]"
                style={{ color: on ? `${accent}` : 'var(--text-muted)' }}
              >
                {it.label}
              </span>
              <span
                className="text-[11px] font-semibold tabular-nums"
                style={{ color: on ? '#fff' : 'var(--text-secondary)' }}
              >
                {it.period}
              </span>
              {/* Live count badge */}
              <span
                className="ml-0.5 inline-flex items-center justify-center rounded-full px-1.5 min-w-[20px] h-[18px] text-[10px] font-semibold tabular-nums transition-all"
                style={{
                  background: on ? `${accent}` : empty ? 'rgba(255,255,255,0.04)' : 'rgba(255,255,255,0.07)',
                  color: on ? '#0b0c12' : empty ? 'var(--text-muted)' : 'var(--text-secondary)',
                  boxShadow: on ? `0 1px 0 rgba(255,255,255,0.18) inset` : 'none',
                }}
              >
                {emaLoaded ? it.count : '—'}
              </span>
            </button>
          );
        })}
      </div>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Right edge: matches + clear */}
      {anyActive && (
        <div className="flex items-center gap-2 transition-all">
          <span className="text-[10.5px] text-[var(--text-muted)]">
            <span className="tabular-nums" style={{ color: accent }}>{matchingTotal}</span> match{matchingTotal === 1 ? '' : 'es'}
          </span>
          <button
            type="button"
            onClick={onClear}
            className="inline-flex items-center gap-1 rounded-full pl-2 pr-2 py-1 text-[10px] font-medium uppercase tracking-wider transition-all"
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: `1px solid ${accentSoft}`,
              color: 'var(--text-secondary)',
            }}
          >
            <X className="w-3 h-3" />
            Clear
          </button>
        </div>
      )}
    </div>
  );
}

type HighConvictionTabKey = 'buy' | 'sell';

function summarizeHighConvictionSignals(signals: HighConvictionSignal[]) {
  const byTicker = new Map<string, number>();
  for (const signal of signals) {
    const ticker = signal.ticker || 'UNKNOWN';
    byTicker.set(ticker, Math.max(byTicker.get(ticker) ?? -Infinity, signal.expected_return_pct ?? 0));
  }
  const bestReturns = Array.from(byTicker.values()).filter((value) => Number.isFinite(value));
  const avgReturn = bestReturns.length > 0
    ? bestReturns.reduce((sum, value) => sum + value, 0) / bestReturns.length
    : 0;
  return {
    positions: byTicker.size,
    signals: signals.length,
    avgReturn,
  };
}

export default function HighConvictionTabs({
  buySignals,
  sellSignals,
  buyLoading,
  sellLoading,
  emaStates,
  qualityScores,
}: {
  buySignals: HighConvictionSignal[];
  sellSignals: HighConvictionSignal[];
  buyLoading: boolean;
  sellLoading: boolean;
  emaStates: Record<string, EmaState>;
  qualityScores: Record<string, number>;
}) {
  const [activeTab, setActiveTab] = useState<HighConvictionTabKey>('buy');
  const buySummary = useMemo(() => summarizeHighConvictionSignals(buySignals), [buySignals]);
  const sellSummary = useMemo(() => summarizeHighConvictionSignals(sellSignals), [sellSignals]);
  const activeSummary = activeTab === 'buy' ? buySummary : sellSummary;
  const activeColor = activeTab === 'buy' ? '#10b981' : '#f43f5e';
  const activeSoft = activeTab === 'buy' ? 'rgba(16,185,129,0.13)' : 'rgba(244,63,94,0.13)';
  const tabs: Array<{
    key: HighConvictionTabKey;
    label: string;
    shortLabel: string;
    color: string;
    summary: ReturnType<typeof summarizeHighConvictionSignals>;
    icon: ReactNode;
  }> = [
    { key: 'buy', label: 'High Conviction BUY', shortLabel: 'BUY', color: '#10b981', summary: buySummary, icon: <TrendingUp className="w-4 h-4" /> },
    { key: 'sell', label: 'High Conviction SELL', shortLabel: 'SELL', color: '#f43f5e', summary: sellSummary, icon: <TrendingDown className="w-4 h-4" /> },
  ];

  return (
    <section className="mb-8 fade-up-delay-1">
      <div
        className="overflow-hidden rounded-[24px]"
        style={{
          background: 'linear-gradient(180deg, rgba(255,255,255,0.026), rgba(255,255,255,0.008))',
          border: '1px solid rgba(255,255,255,0.065)',
          boxShadow: `0 24px 72px -48px ${activeColor}99, 0 16px 48px -34px rgba(0,0,0,0.9), inset 0 1px 0 rgba(255,255,255,0.045)`,
        }}
      >
        <div
          className="flex flex-wrap items-center gap-3 px-4 py-3"
          style={{
            background: `radial-gradient(760px 180px at 10% -90%, ${activeSoft}, transparent 64%), rgba(255,255,255,0.01)`,
            borderBottom: '1px solid rgba(255,255,255,0.045)',
          }}
        >
          <div className="min-w-0 mr-1">
            <div className="flex items-center gap-2">
              <span
                className="inline-flex h-8 w-8 items-center justify-center rounded-xl"
                style={{
                  color: activeColor,
                  background: `${activeColor}18`,
                  border: `1px solid ${activeColor}36`,
                  boxShadow: `0 0 18px -10px ${activeColor}`,
                }}
              >
                <Target className="w-4 h-4" />
              </span>
              <div>
                <h2 className="text-[13.5px] font-semibold tracking-tight text-[var(--text-primary)]">
                  High Conviction
                </h2>
                <p className="text-[10.5px] text-[var(--text-muted)]">
                  Tabbed decision view · BUY opens first
                </p>
              </div>
            </div>
          </div>

          <div
            className="relative inline-flex items-center rounded-2xl p-1"
            style={{
              background: 'rgba(255,255,255,0.035)',
              border: '1px solid rgba(255,255,255,0.06)',
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.035)',
            }}
          >
            <div
              aria-hidden
              className="absolute top-1 bottom-1 rounded-xl transition-all duration-300"
              style={{
                width: 'calc((100% - 8px) / 2)',
                left: activeTab === 'buy' ? '4px' : 'calc(4px + (100% - 8px) / 2)',
                background: activeColor,
                boxShadow: `0 10px 24px -14px ${activeColor}, inset 0 1px 0 rgba(255,255,255,0.22)`,
              }}
            />
            {tabs.map((tab) => {
              const on = activeTab === tab.key;
              return (
                <button
                  key={tab.key}
                  type="button"
                  onClick={() => setActiveTab(tab.key)}
                  aria-pressed={on}
                  className="relative z-10 inline-flex min-w-[190px] items-center justify-between gap-3 rounded-xl px-3 py-2 text-left transition-colors"
                  style={{ color: on ? 'var(--void-bg)' : 'var(--text-secondary)' }}
                >
                  <span className="inline-flex min-w-0 items-center gap-2">
                    <span style={{ color: on ? 'var(--void-bg)' : tab.color }}>{tab.icon}</span>
                    <span className="truncate text-[11px] font-bold uppercase tracking-[0.08em]">{tab.shortLabel}</span>
                  </span>
                  <span
                    className="inline-flex items-center rounded-md px-1.5 py-0.5 text-[10px] font-bold tabular-nums"
                    style={{
                      background: on ? 'rgba(0,0,0,0.15)' : `${tab.color}14`,
                      color: on ? 'var(--void-bg)' : tab.color,
                    }}
                  >
                    {tab.summary.positions}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="ml-auto flex flex-wrap items-center gap-2 text-[10.5px] tabular-nums">
            <span className="rounded-lg px-2 py-1" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}>
              <span className="font-semibold text-[var(--text-secondary)]">{activeSummary.positions}</span> positions
            </span>
            <span className="rounded-lg px-2 py-1" style={{ background: 'rgba(255,255,255,0.025)', border: '1px solid rgba(255,255,255,0.05)', color: 'var(--text-muted)' }}>
              <span className="font-semibold text-[var(--text-secondary)]">{activeSummary.signals}</span> signals
            </span>
            <span className="rounded-lg px-2 py-1" style={{ background: `${activeColor}12`, border: `1px solid ${activeColor}2e`, color: activeColor }}>
              Avg {activeSummary.avgReturn >= 0 ? '+' : ''}{activeSummary.avgReturn.toFixed(1)}%
            </span>
          </div>
        </div>

        <div className="p-3">
          {activeTab === 'buy' ? (
            <HighConvictionPanel
              title="High Conviction BUY"
              signals={buySignals}
              color="green"
              isLoading={buyLoading}
              emaStates={emaStates}
              qualityScores={qualityScores}
            />
          ) : (
            <HighConvictionPanel
              title="High Conviction SELL"
              signals={sellSignals}
              color="red"
              isLoading={sellLoading}
              emaStates={emaStates}
              qualityScores={qualityScores}
            />
          )}
        </div>
      </div>
    </section>
  );
}

function HighConvictionPanel({
  title, signals, color, isLoading, emaStates, qualityScores,
}: {
  title: string;
  signals: HighConvictionSignal[];
  color: 'green' | 'red';
  isLoading: boolean;
  emaStates: Record<string, EmaState>;
  qualityScores: Record<string, number>;
}) {
  const navigate = useNavigate();
  const [sortCol, setSortCol] = useState<HCSortCol>('exp_ret');
  const [sortDir, setSortDir] = useState<HCSortDir>('desc');
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [expandedView, setExpandedView] = useState<'extended' | 'standard' | 'details'>('extended');
  const [tvRange, setTvRange] = useState<string>('3M');
  // Track which timeframes have been activated so we can keep them mounted
  // (avoids re-initializing the TradingView iframe every time the user switches)
  const [activatedRanges, setActivatedRanges] = useState<Set<string>>(() => new Set(['3M']));
  const activateRange = useCallback((label: string) => {
    setActivatedRanges((prev) => (prev.has(label) ? prev : new Set(prev).add(label)));
  }, []);
  const [searchTerm, setSearchTerm] = useState('');
  // EMA-below filters — combine with AND. null = no filter on that period.
  const [emaFilters, setEmaFilters] = useState<{ p9: boolean; p50: boolean; p600: boolean }>(
    { p9: false, p50: false, p600: false }
  );

  // TradingView range → (interval, range) param pairs
  const TV_RANGES: { label: string; range: string; interval: string }[] = [
    { label: '1D',  range: '1D',  interval: '5' },
    { label: '5D',  range: '5D',  interval: '30' },
    { label: '1M',  range: '1M',  interval: '60' },
    { label: '3M',  range: '3M',  interval: 'D' },
    { label: '6M',  range: '6M',  interval: 'D' },
    { label: '1Y',  range: '12M', interval: 'D' },
    { label: '5Y',  range: '60M', interval: 'W' },
    { label: 'ALL', range: 'ALL', interval: 'W' },
  ];

  // Map ticker to TradingView symbol format
  const toTvSymbol = useCallback((ticker: string): string => {
    if (ticker.endsWith('.TO')) return `TSX:${ticker.slice(0, -3)}`;
    if (ticker.endsWith('.V')) return `TSXV:${ticker.slice(0, -2)}`;
    if (ticker.endsWith('.L')) return `LSE:${ticker.slice(0, -2)}`;
    if (ticker.endsWith('.WA')) return `GPW:${ticker.slice(0, -3)}`;
    if (ticker.endsWith('.DE')) return `FWB:${ticker.slice(0, -3)}`;
    if (ticker.endsWith('.PA')) return `EURONEXT:${ticker.slice(0, -3)}`;
    return ticker;
  }, []);

  // Reset tab to the richer chart when row changes.
  useEffect(() => { setExpandedView('extended'); }, [expandedTicker]);

  // Esc collapses the expanded row
  useEffect(() => {
    if (!expandedTicker) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedTicker(null);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [expandedTicker]);

  const Icon = color === 'green' ? TrendingUp : TrendingDown;
  const accent = color === 'green' ? '#10b981' : '#f43f5e';
  const accentSoft = color === 'green' ? '#10b98120' : '#f43f5e20';

  // Group signals by ticker
  const grouped = useMemo(() => {
    const map = new Map<string, GroupedTicker>();
    for (const s of signals) {
      const t = s.ticker || 'UNKNOWN';
      if (!map.has(t)) {
        map.set(t, {
          ticker: t,
          asset_label: s.asset_label || t,
          sector: s.sector || 'Other',
          signals: [],
          bestReturn: -Infinity,
          avgPUp: 0,
          maxStrength: 0,
        });
      }
      const g = map.get(t)!;
      g.signals.push(s);
      g.bestReturn = Math.max(g.bestReturn, s.expected_return_pct ?? 0);
      g.maxStrength = Math.max(g.maxStrength, (s as Record<string, unknown>).signal_strength as number ?? 0);
    }
    // compute avg p_up
    for (const g of map.values()) {
      g.avgPUp = g.signals.reduce((sum, s) => sum + (s.probability_up ?? 0), 0) / g.signals.length;
      g.signals.sort((a, b) => (a.horizon_days ?? 0) - (b.horizon_days ?? 0));
    }
    return Array.from(map.values());
  }, [signals]);

  // Filter — search + EMA toggles
  const passesEma = useCallback((ticker: string) => {
    if (!emaFilters.p9 && !emaFilters.p50 && !emaFilters.p600) return true;
    const st = emaStates[ticker];
    if (!st) return false; // no EMA data → can't satisfy a "below EMA" filter
    if (emaFilters.p9 && st.below_9 !== true) return false;
    if (emaFilters.p50 && st.below_50 !== true) return false;
    if (emaFilters.p600 && st.below_600 !== true) return false;
    return true;
  }, [emaFilters, emaStates]);

  const filtered = useMemo(() => {
    const q = searchTerm.toLowerCase();
    return grouped.filter(g => {
      if (q && !(g.ticker.toLowerCase().includes(q) || g.asset_label.toLowerCase().includes(q) || g.sector.toLowerCase().includes(q))) {
        return false;
      }
      return passesEma(g.ticker);
    });
  }, [grouped, searchTerm, passesEma]);

  // Live counts per EMA period (after the search filter, ignoring other EMA toggles)
  const emaCounts = useMemo(() => {
    const q = searchTerm.toLowerCase();
    const base = q
      ? grouped.filter(g => g.ticker.toLowerCase().includes(q) || g.asset_label.toLowerCase().includes(q) || g.sector.toLowerCase().includes(q))
      : grouped;
    let c9 = 0, c50 = 0, c600 = 0, withData = 0;
    for (const g of base) {
      const st = emaStates[g.ticker];
      if (!st) continue;
      withData += 1;
      if (st.below_9 === true) c9 += 1;
      if (st.below_50 === true) c50 += 1;
      if (st.below_600 === true) c600 += 1;
    }
    return { c9, c50, c600, withData, total: base.length };
  }, [grouped, searchTerm, emaStates]);

  const anyEmaActive = emaFilters.p9 || emaFilters.p50 || emaFilters.p600;
  const clearEmaFilters = useCallback(() => setEmaFilters({ p9: false, p50: false, p600: false }), []);

  // Sort
  const sorted = useMemo(() => {
    const arr = [...filtered];
    const mult = sortDir === 'desc' ? -1 : 1;
    arr.sort((a, b) => {
      let cmp = 0;
      switch (sortCol) {
        case 'ticker': cmp = a.ticker.localeCompare(b.ticker); break;
        case 'quality': {
          const aq = lookupBusinessQuality(qualityScores, a.ticker, a.asset_label) ?? -1;
          const bq = lookupBusinessQuality(qualityScores, b.ticker, b.asset_label) ?? -1;
          cmp = aq - bq;
          break;
        }
        case 'exp_ret': cmp = a.bestReturn - b.bestReturn; break;
        case 'p_up': cmp = a.avgPUp - b.avgPUp; break;
        case 'strength': cmp = a.maxStrength - b.maxStrength; break;
        case 'sector': cmp = a.sector.localeCompare(b.sector); break;
      }
      return cmp * mult;
    });
    return arr;
  }, [filtered, sortCol, sortDir, qualityScores]);

  const handleSort = (col: HCSortCol) => {
    if (sortCol === col) setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    else { setSortCol(col); setSortDir('desc'); }
  };

  const totalSignals = signals.length;
  const uniqueTickers = grouped.length;
  const avgReturn = grouped.length > 0 ? grouped.reduce((s, g) => s + g.bestReturn, 0) / grouped.length : 0;
  const avgProb = grouped.length > 0 ? grouped.reduce((s, g) => s + g.avgPUp, 0) / grouped.length : 0;

  const SortHeader = ({ col, label, w }: { col: HCSortCol; label: string; w?: string }) => (
    <th
      className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-wider cursor-pointer select-none group"
      style={{ color: sortCol === col ? accent : 'var(--text-muted)', width: w, background: '#0b0c12' }}
      onClick={() => handleSort(col)}
    >
      <span className="inline-flex items-center gap-0.5">
        {label}
        {sortCol === col ? (
          sortDir === 'desc' ? <ChevronDown className="w-3 h-3" /> : <ChevronUp className="w-3 h-3" />
        ) : (
          <ChevronDown className="w-3 h-3 opacity-0 group-hover:opacity-30 transition-opacity" />
        )}
      </span>
    </th>
  );

  return (
    <div
      className="overflow-hidden rounded-2xl"
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, transparent 40%), var(--void-base)',
        border: `1px solid ${accentSoft}`,
        boxShadow: `0 1px 0 rgba(255,255,255,0.04) inset, 0 24px 60px -28px ${accent}55, 0 8px 24px -12px rgba(0,0,0,0.6)`,
      }}
    >
      {/* Hero header */}
      <div
        className="relative px-6 pt-5 pb-4"
        style={{
          background: `radial-gradient(1200px 200px at -10% -60%, ${accent}22, transparent 60%), radial-gradient(800px 160px at 110% -40%, ${accent}14, transparent 60%)`,
          borderBottom: `1px solid ${accentSoft}`,
        }}
      >
        {/* Top accent line */}
        <div
          aria-hidden
          className="absolute inset-x-0 top-0 h-px"
          style={{ background: `linear-gradient(90deg, transparent, ${accent}, transparent)`, opacity: 0.65 }}
        />

        <div className="flex items-start justify-between gap-6">
          {/* Title block */}
          <div className="flex items-center gap-3 min-w-0">
            <div
              className="w-11 h-11 rounded-2xl flex items-center justify-center shrink-0"
              style={{
                background: `linear-gradient(135deg, ${accent}33, ${accent}10)`,
                border: `1px solid ${accent}40`,
                boxShadow: `0 0 32px -4px ${accent}66, inset 0 1px 0 rgba(255,255,255,0.08)`,
              }}
            >
              <Icon className="w-5 h-5" style={{ color: accent }} />
            </div>
            <div className="min-w-0">
              <h3
                className="text-[15px] font-bold leading-tight tracking-tight"
                style={{ color: 'var(--text-primary)' }}
              >
                {title}
              </h3>
              <p className="text-[11px] text-[var(--text-muted)] mt-0.5">
                <span className="tabular-nums" style={{ color: accent }}>{uniqueTickers}</span> positions · <span className="tabular-nums text-[var(--text-secondary)]">{totalSignals}</span> signals
              </p>
            </div>
          </div>

          {/* Search */}
          <div
            className="flex items-center gap-1.5 pl-2.5 pr-2 py-1.5 rounded-xl transition-all shrink-0"
            style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid rgba(255,255,255,0.06)',
              width: searchTerm ? 220 : 180,
            }}
          >
            <Search className="w-3.5 h-3.5 text-[var(--text-muted)]" />
            <input
              type="text"
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
              placeholder="Filter ticker, company, sector…"
              className="bg-transparent text-[12px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none flex-1 min-w-0"
            />
            {searchTerm && (
              <button
                type="button"
                onClick={() => setSearchTerm('')}
                className="p-0.5 rounded hover:bg-white/[0.06] text-[var(--text-muted)] hover:text-[var(--text-primary)] transition-colors"
                aria-label="Clear"
              >
                <X className="w-3 h-3" />
              </button>
            )}
          </div>
        </div>

        {/* KPI strip */}
        <div
          className="mt-4 grid grid-cols-3 rounded-xl overflow-hidden"
          style={{
            background: 'rgba(255,255,255,0.025)',
            border: '1px solid rgba(255,255,255,0.05)',
          }}
        >
          <KpiCell label="Positions" value={uniqueTickers.toString()} color="var(--text-primary)" icon={<Layers className="w-3.5 h-3.5" />} dividerRight />
          <KpiCell
            label="Avg Return"
            value={`${avgReturn >= 0 ? '+' : ''}${avgReturn.toFixed(1)}%`}
            color={accent}
            icon={color === 'green' ? <ArrowUp className="w-3.5 h-3.5" /> : <ArrowDown className="w-3.5 h-3.5" />}
            dividerRight
          />
          <KpiCell label="Avg P(up)" value={`${(avgProb * 100).toFixed(0)}%`} color={accent} icon={<Target className="w-3.5 h-3.5" />} />
        </div>
      </div>

      {/* ─── Premium EMA filter bar ───────────────────────────────────── */}
      <EmaFilterBar
        accent={accent}
        accentSoft={accentSoft}
        filters={emaFilters}
        onChange={setEmaFilters}
        counts={emaCounts}
        anyActive={anyEmaActive}
        onClear={clearEmaFilters}
        emaLoaded={Object.keys(emaStates).length > 0}
        matchingTotal={sorted.length}
      />

      {/* Loading state — premium skeleton */}
      {isLoading ? (
        <div className="px-5 py-6 space-y-2">
          {[0, 1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="flex items-center gap-3 rounded-xl px-3 py-3"
              style={{
                background: 'rgba(255,255,255,0.02)',
                border: '1px solid rgba(255,255,255,0.04)',
                animation: `hcShimmer 1.4s ease-in-out ${i * 0.12}s infinite`,
              }}
            >
              <div className="w-[3px] h-7 rounded-full" style={{ background: `${accent}55` }} />
              <div className="flex-1 space-y-1.5">
                <div className="h-2.5 rounded" style={{ width: '18%', background: 'rgba(255,255,255,0.06)' }} />
                <div className="h-1.5 rounded" style={{ width: '32%', background: 'rgba(255,255,255,0.04)' }} />
              </div>
              <div className="h-2 rounded" style={{ width: 60, background: 'rgba(255,255,255,0.05)' }} />
              <div className="h-2 rounded" style={{ width: 80, background: `${accent}22` }} />
              <div className="h-2 rounded" style={{ width: 40, background: 'rgba(255,255,255,0.05)' }} />
            </div>
          ))}
          <style>{`@keyframes hcShimmer { 0%, 100% { opacity: 0.5 } 50% { opacity: 1 } }`}</style>
        </div>
      ) : sorted.length === 0 ? (
        <div className="px-5 py-14 text-center flex flex-col items-center gap-3">
          <div
            className="relative w-16 h-16 rounded-full flex items-center justify-center"
            style={{
              background: `radial-gradient(circle at center, ${accent}18, transparent 70%)`,
            }}
          >
            <div
              className="absolute inset-2 rounded-full"
              style={{
                border: `1px dashed ${accent}40`,
              }}
            />
            {searchTerm ? (
              <Search className="w-6 h-6 relative" style={{ color: `${accent}aa` }} />
            ) : (
              <Shield className="w-6 h-6 relative" style={{ color: `${accent}aa` }} />
            )}
          </div>
          <div>
            <p className="text-[13px] font-semibold" style={{ color: 'var(--text-secondary)' }}>
              {searchTerm ? 'Nothing matches your filter' : 'No high-conviction signals yet'}
            </p>
            <p className="text-[11px] text-[var(--text-muted)] mt-1">
              {searchTerm ? 'Try a different ticker, company, or sector.' : 'Run tune to generate fresh signals.'}
            </p>
          </div>
          {searchTerm && (
            <button
              type="button"
              onClick={() => setSearchTerm('')}
              className="text-[10px] uppercase tracking-wider font-semibold px-3 py-1.5 rounded-lg transition-colors"
              style={{
                color: accent,
                background: `${accent}15`,
                border: `1px solid ${accent}30`,
              }}
            >
              Clear filter
            </button>
          )}
        </div>
      ) : (
        <div
          className="overflow-x-auto overflow-y-auto"
          style={{
            maxHeight: expandedTicker ? 'none' : '420px',
            transition: 'max-height 220ms ease',
          }}
        >
          <table className="w-full" style={{ borderCollapse: 'separate', borderSpacing: 0 }}>
            <thead
              className="sticky top-0"
              style={{
                zIndex: 30,
                background: '#0b0c12',
                boxShadow: `0 1px 0 ${accentSoft}, 0 6px 12px -6px rgba(0,0,0,0.55)`,
                backdropFilter: 'saturate(140%) blur(6px)',
              }}
            >
              <tr>
                <SortHeader col="ticker" label="Asset" w="140px" />
                <SortHeader col="sector" label="Sector" />
                <SortHeader col="quality" label="Quality" w="92px" />
                <th className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]" style={{ background: '#0b0c12' }}>Horizons</th>
                <SortHeader col="exp_ret" label="Best Return" />
                <SortHeader col="p_up" label="Avg P(up)" />
                <SortHeader col="strength" label="Strength" />
                <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]" style={{ background: '#0b0c12' }}></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((g, rowIdx) => {
                const isExpanded = expandedTicker === g.ticker;
                const companyName = g.asset_label.includes('(') ? g.asset_label.split('(')[0].trim() : '';
                const returnIsUp = g.bestReturn >= 0;
                const qualityScore = lookupBusinessQuality(qualityScores, g.ticker, g.asset_label);
                return (
                  <React.Fragment key={g.ticker}>
                    <tr
                      className="cursor-pointer group transition-colors"
                      onClick={() => setExpandedTicker(isExpanded ? null : g.ticker)}
                      style={{
                        background: isExpanded ? `linear-gradient(90deg, ${accent}0f, transparent 55%)` : 'transparent',
                        borderBottom: '1px solid rgba(255,255,255,0.035)',
                        animation: `hcRowIn 320ms cubic-bezier(0.2, 0.8, 0.2, 1) ${Math.min(rowIdx, 14) * 24}ms both`,
                      }}
                      onMouseEnter={(e) => { if (!isExpanded) e.currentTarget.style.background = 'rgba(255,255,255,0.025)'; }}
                      onMouseLeave={(e) => { if (!isExpanded) e.currentTarget.style.background = 'transparent'; }}
                    >
                      {/* Asset */}
                      <td className="pl-4 pr-3 py-3">
                        <div className="flex items-center gap-3">
                          <div
                            className="rounded-full transition-all"
                            style={{
                              width: isExpanded ? 3 : 2,
                              height: 30,
                              background: isExpanded
                                ? accent
                                : `linear-gradient(to bottom, ${accent}cc, ${accent}33)`,
                              boxShadow: isExpanded ? `0 0 8px ${accent}88` : 'none',
                            }}
                          />
                          <div className="min-w-0">
                            <div className="text-[13px] font-bold tracking-tight leading-tight" style={{ color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                              {g.ticker}
                            </div>
                            {companyName && (
                              <div className="text-[10px] text-[var(--text-muted)] truncate max-w-[160px] leading-tight mt-0.5">
                                {companyName}
                              </div>
                            )}
                          </div>
                        </div>
                      </td>
                      {/* Sector */}
                      <td className="px-2 py-3">
                        <span
                          className="inline-flex items-center text-[10px] font-medium px-2 py-1 rounded-md"
                          style={{
                            color: 'var(--text-secondary)',
                            background: 'rgba(255,255,255,0.035)',
                            border: '1px solid rgba(255,255,255,0.05)',
                          }}
                        >
                          {g.sector}
                        </span>
                      </td>
                      {/* Business Quality */}
                      <td className="px-2 py-2">
                        <div className="flex items-center justify-center">
                          <BusinessQualityRing score={qualityScore} />
                        </div>
                      </td>
                      {/* Horizons pills */}
                      <td className="px-2 py-3">
                        <div className="flex items-center gap-1 flex-wrap">
                          {g.signals.map((s, i) => {
                            const d = s.horizon_days ?? 0;
                            // Opacity grades by recency
                            const op = d <= 1 ? 1 : d <= 3 ? 0.75 : d <= 7 ? 0.55 : 0.35;
                            return (
                              <span
                                key={i}
                                className="text-[10px] px-1.5 py-0.5 rounded-md font-semibold tabular-nums"
                                style={{
                                  background: `${accent}${Math.round(op * 28).toString(16).padStart(2, '0')}`,
                                  color: accent,
                                  border: `1px solid ${accent}${Math.round(op * 50).toString(16).padStart(2, '0')}`,
                                }}
                              >
                                {formatHorizon(s.horizon_days)}
                              </span>
                            );
                          })}
                        </div>
                      </td>
                      {/* Best Return */}
                      <td className="px-2 py-3 text-right">
                        <div className="inline-flex items-baseline gap-1 tabular-nums" style={{ color: accent }}>
                          <span className="text-[15px] font-bold tracking-tight" style={{ letterSpacing: '-0.02em' }}>
                            {returnIsUp ? '+' : ''}{g.bestReturn.toFixed(1)}
                          </span>
                          <span className="text-[10px] font-semibold opacity-70">%</span>
                        </div>
                      </td>
                      {/* Avg P(up) — arc gauge */}
                      <td className="px-2 py-3">
                        <div className="flex items-center gap-2">
                          <ArcGauge value={g.avgPUp} color={accent} size={26} />
                          <span className="text-[11px] font-semibold tabular-nums" style={{ color: 'var(--text-primary)' }}>
                            {(g.avgPUp * 100).toFixed(0)}%
                          </span>
                        </div>
                      </td>
                      {/* Strength — segmented meter */}
                      <td className="px-2 py-3">
                        <div className="flex items-center gap-2">
                          <SegmentedMeter value={Math.min(g.maxStrength * 2, 1)} color={accent} segments={5} />
                          <span className="text-[10px] tabular-nums text-[var(--text-muted)]">
                            {g.maxStrength.toFixed(2)}
                          </span>
                        </div>
                      </td>
                      {/* Expand */}
                      <td className="pl-2 pr-4 py-3">
                        <div className="flex items-center justify-end">
                          <span
                            className="inline-flex items-center justify-center rounded-full transition-all"
                            style={{
                              width: 22,
                              height: 22,
                              background: isExpanded ? accent : 'rgba(255,255,255,0.04)',
                              color: isExpanded ? 'var(--void-bg)' : 'var(--text-muted)',
                              boxShadow: isExpanded ? `0 0 10px ${accent}88` : 'none',
                              transform: isExpanded ? 'rotate(180deg)' : 'rotate(0deg)',
                              transition: 'transform 220ms cubic-bezier(0.2, 0.8, 0.2, 1), background-color 180ms',
                            }}
                          >
                            <ChevronDown className="w-3.5 h-3.5" />
                          </span>
                        </div>
                      </td>
                    </tr>
                    {/* Expanded detail — iOS-style segmented control + hero chart */}
                    {isExpanded && (
                      <tr>
                        <td
                          colSpan={8}
                          style={{
                            background: `linear-gradient(180deg, ${accent}0a 0%, transparent 100%)`,
                            borderBottom: `1px solid ${accentSoft}`,
                          }}
                        >
                          <div className="px-5 py-4">
                            {/* Segmented control */}
                            <div className="flex items-center justify-between mb-3">
                              <div
                                className="inline-flex items-center p-1 rounded-xl relative"
                                style={{
                                  background: 'rgba(255,255,255,0.035)',
                                  border: '1px solid rgba(255,255,255,0.06)',
                                  boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.03)',
                                }}
                              >
                                {/* Sliding indicator */}
                                <div
                                  aria-hidden
                                  className="absolute top-1 bottom-1 rounded-lg transition-all"
                                  style={{
                                    width: 'calc((100% - 8px) / 3)',
                                    left: expandedView === 'extended'
                                      ? '4px'
                                      : expandedView === 'standard'
                                        ? 'calc(4px + (100% - 8px) / 3)'
                                        : 'calc(4px + 2 * (100% - 8px) / 3)',
                                    background: accent,
                                    boxShadow: `0 2px 8px -2px ${accent}88, inset 0 1px 0 rgba(255,255,255,0.2)`,
                                    transition: 'left 260ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                                  }}
                                />
                                <button
                                  type="button"
                                  onClick={() => setExpandedView('extended')}
                                  className="relative z-10 inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-[11px] font-semibold transition-colors"
                                  style={{
                                    color: expandedView === 'extended' ? 'var(--void-bg)' : 'var(--text-secondary)',
                                    minWidth: 132,
                                    justifyContent: 'center',
                                  }}
                                >
                                  <BarChart3 className="w-3.5 h-3.5" />
                                  Extended Chart
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setExpandedView('standard')}
                                  className="relative z-10 inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-[11px] font-semibold transition-colors"
                                  style={{
                                    color: expandedView === 'standard' ? 'var(--void-bg)' : 'var(--text-secondary)',
                                    minWidth: 132,
                                    justifyContent: 'center',
                                  }}
                                >
                                  <Activity className="w-3.5 h-3.5" />
                                  Standard Chart
                                </button>
                                <button
                                  type="button"
                                  onClick={() => setExpandedView('details')}
                                  className="relative z-10 inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-[11px] font-semibold transition-colors"
                                  style={{
                                    color: expandedView === 'details' ? 'var(--void-bg)' : 'var(--text-secondary)',
                                    minWidth: 110,
                                    justifyContent: 'center',
                                  }}
                                >
                                  <Eye className="w-3.5 h-3.5" />
                                  Details
                                </button>
                              </div>
                              <div className="text-[10px] text-[var(--text-muted)] flex items-center gap-2">
                                <span className="font-mono text-[var(--text-secondary)] font-semibold tracking-wide">{g.ticker}</span>
                                {companyName && <span className="hidden sm:inline">· {companyName}</span>}
                                <span className="hidden md:inline">· {g.sector}</span>
                              </div>
                            </div>

                            {expandedView === 'extended' && (() => {
                              const horizonSignals = Object.fromEntries(
                                g.signals.map((s) => [
                                  formatHorizon(s.horizon_days),
                                  {
                                    p_up: s.probability_up,
                                    label: s.signal_type || (color === 'green' ? 'STRONG BUY' : 'STRONG SELL'),
                                  },
                                ]),
                              ) as Record<string, { p_up?: number; label?: string }>;
                              return (
                                <div className="overflow-hidden rounded-2xl" style={{ border: `1px solid ${accentSoft}` }}>
                                  <SignalDetailPanel
                                    ticker={g.ticker}
                                    signal={color === 'green' ? 'STRONG BUY' : 'STRONG SELL'}
                                    horizonSignals={horizonSignals}
                                    defaultChartType="area"
                                    defaultRange="1Y"
                                    onNavigateChart={() => navigate(`/charts/${g.ticker}`)}
                                  />
                                </div>
                              );
                            })()}

                            {expandedView === 'standard' && (() => {
                              const tvSym = toTvSymbol(g.ticker);
                              const buildSrc = (interval: string, range: string) =>
                                `https://s.tradingview.com/widgetembed/?frameElementId=tv_${encodeURIComponent(g.ticker)}_${range}&symbol=${encodeURIComponent(tvSym)}&interval=${interval}&range=${range}&hidesidetoolbar=0&hidetoptoolbar=0&symboledit=1&saveimage=0&toolbarbg=0f0f1a&theme=dark&style=8&timezone=Etc/UTC&withdateranges=1&hideideas=1&hideideasbutton=1&locale=en`;
                              return (
                                <div
                                  className="rounded-2xl overflow-hidden"
                                  style={{
                                    background: '#06070b',
                                    border: `1px solid rgba(255,255,255,0.06)`,
                                    boxShadow: `0 24px 60px -28px ${accent}44, 0 8px 24px -12px rgba(0,0,0,0.6)`,
                                  }}
                                >
                                  {/* Toolbar */}
                                  <div
                                    className="flex items-center gap-3 px-3 py-2"
                                    style={{
                                      borderBottom: '1px solid rgba(255,255,255,0.05)',
                                      background: 'rgba(255,255,255,0.015)',
                                    }}
                                  >
                                    <div className="flex items-center gap-1.5 text-[10px] uppercase tracking-wider text-[var(--text-muted)] font-semibold">
                                      <Clock className="w-3 h-3" />
                                      Timeframe
                                    </div>
                                    {/* Timeframe segmented */}
                                    <div
                                      className="inline-flex items-center p-0.5 rounded-lg"
                                      style={{
                                        background: 'rgba(255,255,255,0.04)',
                                        border: '1px solid rgba(255,255,255,0.05)',
                                      }}
                                    >
                                      {TV_RANGES.map((r) => {
                                        const isActive = r.label === tvRange;
                                        const isPreloaded = activatedRanges.has(r.label);
                                        return (
                                          <button
                                            key={r.label}
                                            type="button"
                                            onClick={() => { activateRange(r.label); setTvRange(r.label); }}
                                            onMouseEnter={() => activateRange(r.label)}
                                            onFocus={() => activateRange(r.label)}
                                            className="relative px-2.5 py-1 rounded text-[10px] font-bold tabular-nums transition-all"
                                            style={{
                                              background: isActive ? accent : 'transparent',
                                              color: isActive ? 'var(--void-bg)' : 'var(--text-secondary)',
                                              boxShadow: isActive ? `0 2px 6px -2px ${accent}aa` : 'none',
                                              letterSpacing: '0.02em',
                                            }}
                                            title={`${r.label} · ${r.interval === 'D' ? 'Daily' : r.interval === 'W' ? 'Weekly' : `${r.interval}m`} Heikin Ashi${isPreloaded && !isActive ? ' · preloaded' : ''}`}
                                          >
                                            {r.label}
                                            {isPreloaded && !isActive && (
                                              <span
                                                aria-hidden
                                                className="absolute top-0.5 right-0.5 rounded-full"
                                                style={{
                                                  width: 3,
                                                  height: 3,
                                                  background: accent,
                                                  opacity: 0.7,
                                                }}
                                              />
                                            )}
                                          </button>
                                        );
                                      })}
                                    </div>
                                    <a
                                      href={`https://www.tradingview.com/chart/?symbol=${encodeURIComponent(tvSym)}`}
                                      target="_blank"
                                      rel="noopener noreferrer"
                                      onClick={(e) => e.stopPropagation()}
                                      className="ml-auto inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-[10px] font-semibold transition-colors"
                                      style={{
                                        color: 'var(--text-secondary)',
                                        background: 'rgba(255,255,255,0.04)',
                                        border: '1px solid rgba(255,255,255,0.06)',
                                      }}
                                      title="Open on TradingView"
                                    >
                                      <span className="font-mono">{tvSym}</span>
                                      <ExternalLink className="w-3 h-3" />
                                    </a>
                                  </div>
                                  {/* Iframe stack — keeps each activated timeframe mounted so switching is instant */}
                                  <div style={{ position: 'relative', width: '100%', height: 520 }}>
                                    {TV_RANGES.map((r) => {
                                      if (!activatedRanges.has(r.label)) return null;
                                      const isActive = r.label === tvRange;
                                      return (
                                        <iframe
                                          key={`${g.ticker}_${r.label}`}
                                          title={`TradingView ${g.ticker} ${r.label}`}
                                          src={buildSrc(r.interval, r.range)}
                                          frameBorder={0}
                                          allowTransparency={true}
                                          scrolling="no"
                                          style={{
                                            position: 'absolute',
                                            inset: 0,
                                            width: '100%',
                                            height: '100%',
                                            border: 0,
                                            visibility: isActive ? 'visible' : 'hidden',
                                            pointerEvents: isActive ? 'auto' : 'none',
                                            zIndex: isActive ? 2 : 1,
                                          }}
                                        />
                                      );
                                    })}
                                  </div>
                                </div>
                              );
                            })()}

                            {expandedView === 'details' && (
                            <>
                            <div
                              className="grid gap-3"
                              style={{ gridTemplateColumns: `repeat(${Math.min(g.signals.length, 4)}, minmax(0, 1fr))` }}
                            >
                              {g.signals.map((s, i) => {
                                const strength = (s as Record<string, unknown>).signal_strength as number ?? 0;
                                const conviction = (s as Record<string, unknown>).conviction_probability as number ?? s.probability_up;
                                const profitPln = s.expected_profit_pln;
                                const expRet = s.expected_return_pct ?? 0;
                                const expSign = expRet >= 0 ? '+' : '';
                                return (
                                  <div
                                    key={i}
                                    className="relative rounded-2xl overflow-hidden"
                                    style={{
                                      background: 'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%), #06070b',
                                      border: '1px solid rgba(255,255,255,0.06)',
                                      boxShadow: `0 1px 0 rgba(255,255,255,0.03) inset, 0 12px 32px -16px ${accent}55`,
                                    }}
                                  >
                                    {/* top accent bar */}
                                    <div
                                      aria-hidden
                                      className="absolute inset-x-0 top-0 h-[2px]"
                                      style={{ background: `linear-gradient(90deg, ${accent}, ${accent}33)` }}
                                    />

                                    <div className="p-4">
                                      {/* Header: horizon + signal type chip */}
                                      <div className="flex items-start justify-between mb-3">
                                        <div>
                                          <div className="text-[9px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.12em' }}>
                                            Horizon
                                          </div>
                                          <div
                                            className="text-[16px] font-bold tabular-nums tracking-tight"
                                            style={{ color: 'var(--text-primary)', letterSpacing: '-0.02em' }}
                                          >
                                            {formatHorizon(s.horizon_days)}
                                          </div>
                                        </div>
                                        <span
                                          className="text-[9px] px-2 py-0.5 rounded-full font-bold uppercase tracking-wider"
                                          style={{
                                            background: `${accent}20`,
                                            color: accent,
                                            border: `1px solid ${accent}40`,
                                            letterSpacing: '0.08em',
                                          }}
                                        >
                                          {s.signal_type || 'STRONG'}
                                        </span>
                                      </div>

                                      {/* Hero metric: Expected Return */}
                                      <div className="mb-3">
                                        <div className="text-[9px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.12em' }}>
                                          Expected Return
                                        </div>
                                        <div
                                          className="inline-flex items-baseline gap-1 tabular-nums"
                                          style={{ color: accent }}
                                        >
                                          <span className="text-[26px] font-bold tracking-tight leading-none" style={{ letterSpacing: '-0.03em' }}>
                                            {s.expected_return_pct != null ? `${expSign}${expRet.toFixed(2)}` : '—'}
                                          </span>
                                          {s.expected_return_pct != null && (
                                            <span className="text-[12px] font-semibold opacity-60">%</span>
                                          )}
                                        </div>
                                      </div>

                                      {/* Arc gauges row */}
                                      <div
                                        className="grid grid-cols-2 gap-2 rounded-xl p-2.5 mb-3"
                                        style={{
                                          background: 'rgba(255,255,255,0.02)',
                                          border: '1px solid rgba(255,255,255,0.04)',
                                        }}
                                      >
                                        <div className="flex items-center gap-2">
                                          <ArcGauge value={s.probability_up} color={accent} size={30} />
                                          <div className="min-w-0">
                                            <div className="text-[8px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.1em' }}>
                                              P(up)
                                            </div>
                                            <div className="text-[12px] font-bold tabular-nums" style={{ color: 'var(--text-primary)' }}>
                                              {(s.probability_up * 100).toFixed(1)}%
                                            </div>
                                          </div>
                                        </div>
                                        <div className="flex items-center gap-2">
                                          <ArcGauge value={s.probability_down} color="#64748b" size={30} />
                                          <div className="min-w-0">
                                            <div className="text-[8px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.1em' }}>
                                              P(down)
                                            </div>
                                            <div className="text-[12px] font-bold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                                              {(s.probability_down * 100).toFixed(1)}%
                                            </div>
                                          </div>
                                        </div>
                                      </div>

                                      {/* Strength + Conviction rows */}
                                      <div className="space-y-2">
                                        <div className="flex items-center justify-between">
                                          <span className="text-[9px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.1em' }}>
                                            Strength
                                          </span>
                                          <div className="flex items-center gap-2">
                                            <SegmentedMeter value={Math.min(strength * 2, 1)} color={accent} segments={5} />
                                            <span className="text-[10px] font-semibold tabular-nums" style={{ color: 'var(--text-secondary)' }}>
                                              {strength.toFixed(2)}
                                            </span>
                                          </div>
                                        </div>
                                        <div className="flex items-center justify-between">
                                          <span className="text-[9px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.1em' }}>
                                            Conviction
                                          </span>
                                          <span className="text-[11px] font-bold tabular-nums" style={{ color: accent }}>
                                            {(conviction * 100).toFixed(1)}%
                                          </span>
                                        </div>
                                        {profitPln != null && profitPln !== 0 && (
                                          <div
                                            className="flex items-center justify-between pt-2 mt-2"
                                            style={{ borderTop: '1px solid rgba(255,255,255,0.06)' }}
                                          >
                                            <span className="text-[9px] uppercase font-semibold text-[var(--text-muted)]" style={{ letterSpacing: '0.1em' }}>
                                              Est. Profit
                                            </span>
                                            <span className="text-[12px] font-bold tabular-nums" style={{ color: accent }}>
                                              {profitPln > 0 ? '+' : ''}{profitPln.toLocaleString('en', { maximumFractionDigits: 0 })}
                                              <span className="text-[9px] opacity-60 ml-1">PLN</span>
                                            </span>
                                          </div>
                                        )}
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                            {/* Generated timestamp */}
                            {g.signals[0] && (g.signals[0] as Record<string, unknown>).generated_at && (
                              <div className="flex items-center gap-1.5 mt-3 px-1">
                                <Clock className="w-3 h-3 text-[var(--text-muted)]" />
                                <span className="text-[10px] text-[var(--text-muted)]" style={{ letterSpacing: '0.02em' }}>
                                  Generated <span className="font-mono text-[var(--text-secondary)]">{new Date(String((g.signals[0] as Record<string, unknown>).generated_at)).toLocaleString()}</span>
                                </span>
                              </div>
                            )}
                            </>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </React.Fragment>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
      <style>{`@keyframes hcRowIn { from { opacity: 0; transform: translateY(4px); } to { opacity: 1; transform: translateY(0); } }`}</style>
    </div>
  );
}
