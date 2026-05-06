import React, { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import { Activity, AlertTriangle, BarChart3, ChevronRight, ExternalLink, Layers, Loader2, RefreshCw, Search, ShieldCheck, Target, TrendingDown, TrendingUp, X, Zap } from 'lucide-react';
import { api, type SummaryRow, type SmaReversal, type SmaReversalsData } from '../../../api';
import MiniPriceChart, { type MiniPriceChartView } from '../../../components/MiniPriceChart';
import { Sparkline, SparklineReversalStateBadge } from '../../../components/Sparkline';
import { smaQualityTone } from '../theme';
import SegmentedControl from './SegmentedControl';

const ISO_CURRENCY_CODES = new Set([
  'AUD', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP', 'HKD',
  'HUF', 'INR', 'JPY', 'KRW', 'MXN', 'NOK', 'NZD', 'PLN', 'SEK', 'SGD',
  'TRY', 'USD', 'ZAR',
]);

const isFiatCurrencyTicker = (symbol: string | undefined | null): boolean => {
  let sym = String(symbol || '').trim().toUpperCase();
  if (sym.endsWith('_X')) sym = `${sym.slice(0, -2)}=X`;
  if (!sym.endsWith('=X')) return false;
  const pair = sym.slice(0, -2).replace(/[\/_-]/g, '');
  if (pair.length !== 6) return false;
  return ISO_CURRENCY_CODES.has(pair.slice(0, 3)) && ISO_CURRENCY_CODES.has(pair.slice(3));
};
const cleanSmaReason = (reason: string): string => reason.replace(/\s*\(n=\d+\)/gi, '');

const SMA_CHART_VIEW_PREFS_KEY = 'signals.smaReversalChartView.v1';
const SMA_CHART_RANGE_PREFS_KEY = 'signals.smaReversalChartRange.v1';
const SMA_DETAIL_MAX_TAIL = 2000;
const SMA_CHART_VIEWS: MiniPriceChartView[] = ['sma', 'heikinAshi', 'reversal', 'area'];
type SmaChartRange = '1M' | '3M' | '6M' | '1Y' | 'MAX';
const SMA_CHART_RANGES: SmaChartRange[] = ['1M', '3M', '6M', '1Y', 'MAX'];
const SMA_CHART_RANGE_DAYS: Record<SmaChartRange, number> = {
  '1M': 22,
  '3M': 66,
  '6M': 132,
  '1Y': 252,
  MAX: SMA_DETAIL_MAX_TAIL,
};

const isMiniPriceChartView = (value: unknown): value is MiniPriceChartView =>
  typeof value === 'string' && SMA_CHART_VIEWS.includes(value as MiniPriceChartView);

const isSmaChartRange = (value: unknown): value is SmaChartRange =>
  typeof value === 'string' && SMA_CHART_RANGES.includes(value as SmaChartRange);

const loadStoredSmaChartView = (): MiniPriceChartView => {
  if (typeof window === 'undefined') return 'sma';
  try {
    const raw = window.localStorage.getItem(SMA_CHART_VIEW_PREFS_KEY);
    return isMiniPriceChartView(raw) ? raw : 'sma';
  } catch {
    return 'sma';
  }
};

const saveStoredSmaChartView = (view: MiniPriceChartView): void => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SMA_CHART_VIEW_PREFS_KEY, view);
  } catch {
    // In-memory selection still works if storage is unavailable.
  }
};

const loadStoredSmaChartRange = (): SmaChartRange => {
  if (typeof window === 'undefined') return '1Y';
  try {
    const raw = window.localStorage.getItem(SMA_CHART_RANGE_PREFS_KEY);
    return isSmaChartRange(raw) ? raw : '1Y';
  } catch {
    return '1Y';
  }
};
const saveStoredSmaChartRange = (range: SmaChartRange): void => {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SMA_CHART_RANGE_PREFS_KEY, range);
  } catch {
    // In-memory selection still works if storage is unavailable.
  }
};

const lookupQualityScore = (scores: Record<string, number>, symbol: string | undefined | null): number | null => {
  const raw = String(symbol || '').trim();
  if (!raw) return null;
  const upper = raw.toUpperCase();
  const variants = [
    raw,
    upper,
    upper.replace(/-/g, '.'),
    upper.replace(/\./g, '-'),
    upper.replace(/=/g, '_'),
    upper.replace(/_/g, '='),
  ];
  for (const key of variants) {
    const score = scores[key];
    if (typeof score === 'number' && isFinite(score)) return score;
  }
  return null;
};

function SmaQualityBadge({ score, compact = false }: { score: number | null | undefined; compact?: boolean }) {
  const tone = smaQualityTone(score);
  const hasScore = score != null && isFinite(score);
  const rounded = hasScore ? Math.round(score) : null;
  return (
    <div className={`flex ${compact ? 'flex-col items-end gap-0.5' : 'items-center gap-2'}`}>
      <div
        className="inline-flex items-center justify-center rounded-lg tabular-nums"
        title={hasScore ? `Business quality score: ${rounded} (${tone.label})` : 'Business quality score unavailable'}
        style={{
          minWidth: compact ? 44 : 52,
          height: compact ? 26 : 30,
          padding: compact ? '0 8px' : '0 10px',
          background: tone.background,
          border: `1px solid ${tone.border}`,
          boxShadow: tone.glow,
          color: tone.color,
        }}
      >
        <span className={`${compact ? 'text-[10.5px]' : 'text-[12px]'} font-bold leading-none`}>
          {rounded ?? '--'}
        </span>
      </div>
      <span
        className={`${compact ? 'text-[8.5px] uppercase tracking-[0.12em]' : 'text-[10px] font-semibold'} whitespace-nowrap`}
        style={{ color: compact ? 'var(--text-muted)' : tone.color }}
      >
        {compact ? 'Quality' : tone.label}
      </span>
    </div>
  );
}
// ═══════════════════════════════════════════════════════════════════════
//   SmaReversalsPanel — world-class SMA (9/50/600) reversal dashboard
// ═══════════════════════════════════════════════════════════════════════
//
// Consumes the `/api/signals/sma-reversals` snapshot. Each reversal carries
// a composite 0-100 score built from: ATR-normalised distance, 5d SMA
// slope, volume vs 20d baseline, persistence (K of M bars on the new
// side), and freshness (days since cross ≤ 5). False-breaks are penalised
// (0.6×) but kept visible, flagged with an amber shield.
//
export default function SmaReversalsPanel({
  data,
  isLoading,
  rows,
  qualityScores,
  onNavigateChart,
}: {
  data: SmaReversalsData | undefined;
  isLoading: boolean;
  rows: SummaryRow[];
  qualityScores: Record<string, number>;
  onNavigateChart: (symbol: string) => void;
}) {
  const [periodFilter, setPeriodFilter] = useState<Set<number>>(() => new Set([9, 50, 600]));
  const [direction, setDirection] = useState<'all' | 'bull' | 'bear'>('all');
  const [gradeFilter, setGradeFilter] = useState<'all' | 'A' | 'B' | 'C'>('all');
  const [minScore, setMinScore] = useState<number>(40);
  const [hideFalseBreaks, setHideFalseBreaks] = useState<boolean>(true);
  const [search, setSearch] = useState<string>('');
  const [showAll, setShowAll] = useState<boolean>(false);
  const [expandedKey, setExpandedKey] = useState<string | null>(null);
  const [detailChartView, setDetailChartView] = useState<MiniPriceChartView>(() => loadStoredSmaChartView());
  const [detailChartRange, setDetailChartRange] = useState<SmaChartRange>(() => loadStoredSmaChartRange());

  useEffect(() => {
    if (!expandedKey) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedKey(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [expandedKey]);

  useEffect(() => {
    saveStoredSmaChartView(detailChartView);
  }, [detailChartView]);

  useEffect(() => {
    saveStoredSmaChartRange(detailChartRange);
  }, [detailChartRange]);

  const labelMap = useMemo(() => {
    const m = new Map<string, string>();
    for (const r of rows) {
      const match = r.asset_label?.match(/\(([^)]+)\)\s*$/);
      if (match) m.set(match[1].trim(), r.asset_label);
      else if (r.asset_label) m.set(r.asset_label.trim(), r.asset_label);
      // Also handle FX `=X` ↔ `_X` variance between data sides
      if (match) {
        m.set(match[1].replace(/=/g, '_').trim(), r.asset_label);
      }
    }
    return m;
  }, [rows]);

  const reversals = useMemo(
    () => (data?.reversals || []).filter((r) => !isFiatCurrencyTicker(r.symbol)),
    [data?.reversals],
  );
  const counts = useMemo(() => {
    const next: Record<string, { bull: number; bear: number }> = {};
    const periods = data?.periods || [9, 50, 600];
    for (const p of periods) next[String(p)] = { bull: 0, bear: 0 };
    for (const r of reversals) {
      const key = String(r.period);
      if (!next[key]) next[key] = { bull: 0, bear: 0 };
      if (r.direction === 'bull') next[key].bull += 1;
      if (r.direction === 'bear') next[key].bear += 1;
    }
    return next;
  }, [data?.periods, reversals]);
  const gradeCounts = useMemo(() => ({
    A: reversals.filter((r) => r.grade === 'A').length,
    B: reversals.filter((r) => r.grade === 'B').length,
    C: reversals.filter((r) => r.grade === 'C').length,
    ungraded: reversals.filter((r) => r.grade == null).length,
  }), [reversals]);
  const buySetups = useMemo(
    () => reversals.filter((r) => r.direction === 'bull' && (r.grade === 'A' || r.grade === 'B')).length,
    [reversals],
  );

  const filtered = useMemo(() => {
    const q = search.trim().toUpperCase();
    return reversals.filter((r) => {
      if (!periodFilter.has(r.period as number)) return false;
      if (direction !== 'all' && r.direction !== direction) return false;
      if (gradeFilter !== 'all' && r.grade !== gradeFilter) return false;
      if (r.score < minScore) return false;
      if (hideFalseBreaks && r.false_break) return false;
      if (q && !r.symbol.toUpperCase().includes(q)) return false;
      return true;
    });
  }, [reversals, periodFilter, direction, gradeFilter, minScore, hideFalseBreaks, search]);

  const displayed = showAll ? filtered : filtered.slice(0, 20);

  const togglePeriod = (p: number) => {
    setPeriodFilter((prev) => {
      const next = new Set(prev);
      if (next.has(p)) {
        if (next.size > 1) next.delete(p);  // never empty
      } else {
        next.add(p);
      }
      return next;
    });
  };

  // Summary: total bull/bear across selected periods
  const selectedTotals = useMemo(() => {
    let bull = 0, bear = 0;
    for (const p of Array.from(periodFilter)) {
      const c = counts[String(p)];
      if (c) { bull += c.bull; bear += c.bear; }
    }
    return { bull, bear };
  }, [counts, periodFilter]);

  return (
    <div
      className="mb-5 overflow-hidden fade-up-delay-2"
      style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.028) 0%, rgba(255,255,255,0.008) 100%)',
        border: '1px solid rgba(255,255,255,0.07)',
        borderRadius: '16px',
        boxShadow: '0 1px 0 rgba(255,255,255,0.05) inset, 0 10px 34px -18px rgba(0,0,0,0.7), 0 1px 0 rgba(0,0,0,0.4)',
        backdropFilter: 'blur(12px)',
      }}
    >
      {/* Header */}
      <div className="flex flex-wrap items-center gap-3 px-4 pt-3.5 pb-3" style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
        <div className="flex items-center gap-2.5">
          <div
            className="flex items-center justify-center rounded-lg"
            style={{
              width: 28, height: 28,
              background: 'linear-gradient(180deg, rgba(167,139,250,0.22), rgba(167,139,250,0.05))',
              border: '1px solid rgba(167,139,250,0.32)',
              boxShadow: '0 0 16px -6px rgba(167,139,250,0.6) inset',
            }}
          >
            <Zap className="w-3.5 h-3.5" style={{ color: '#a78bfa' }} />
          </div>
          <div className="flex flex-col">
            <h2 className="text-[13.5px] font-semibold text-[var(--text-primary)] tracking-tight">SMA Reversals</h2>
            <span className="text-[9.5px] uppercase tracking-[0.14em] font-semibold text-[var(--text-muted)]">
              9 · 50 · 600 crossovers
            </span>
          </div>
        </div>

        <div className="h-6 w-px bg-white/[0.05]" aria-hidden />

        {/* Buy-setups headline — the "am I a buyer today" answer */}
        {data && (
          <div
            className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-[3px]"
            style={{
              background: 'linear-gradient(180deg, rgba(16,185,129,0.22), rgba(16,185,129,0.06))',
              border: '1px solid rgba(16,185,129,0.45)',
              boxShadow: '0 0 16px -6px rgba(16,185,129,0.65) inset',
            }}
            title="High-quality long setups: Grade A or B, bull direction, regime-aligned, not overextended"
          >
            <ShieldCheck className="w-3 h-3" style={{ color: '#34d399' }} />
            <span className="text-[10.5px] font-semibold text-white tabular-nums">{buySetups}</span>
            <span className="text-[9.5px] uppercase tracking-[0.12em] font-semibold text-[#a7f3d0]">buy setups</span>
          </div>
        )}

        {/* Totals bull / bear */}
        <div className="flex items-center gap-3 text-[11px] tabular-nums">
          <div className="inline-flex items-center gap-1.5">
            <TrendingUp className="w-3 h-3" style={{ color: '#34d399' }} />
            <span className="text-[var(--text-secondary)]">Bull</span>
            <span className="font-semibold text-[var(--text-primary)]">{selectedTotals.bull}</span>
          </div>
          <div className="inline-flex items-center gap-1.5">
            <TrendingDown className="w-3 h-3" style={{ color: '#fb7185' }} />
            <span className="text-[var(--text-secondary)]">Bear</span>
            <span className="font-semibold text-[var(--text-primary)]">{selectedTotals.bear}</span>
          </div>
          {data && (
            <div className="flex items-center gap-1.5 pl-1">
              <GradeBadge grade="A" count={gradeCounts.A} />
              <GradeBadge grade="B" count={gradeCounts.B} />
              <GradeBadge grade="C" count={gradeCounts.C} />
            </div>
          )}
          <div className="text-[10px] text-[var(--text-muted)]">
            · {filtered.length} shown / {reversals.length} total
          </div>
        </div>

        <div className="flex-1 min-w-[8px]" />

        {/* Quick search */}
        <div
          className="flex items-center gap-2 px-2.5 py-[5px] transition-all duration-200"
          style={{
            background: 'rgba(255,255,255,0.02)',
            border: '1px solid rgba(255,255,255,0.06)',
            borderRadius: '10px',
          }}
        >
          <Search className="w-3 h-3 text-[var(--text-muted)]" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Filter ticker..."
            className="bg-transparent text-[11.5px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none w-32 tabular-nums"
          />
          {search && (
            <button onClick={() => setSearch('')} className="text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors">
              <X className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>

      {/* Filters row */}
      <div className="flex flex-wrap items-center gap-3 px-4 py-2.5" style={{ background: 'rgba(255,255,255,0.008)' }}>
        {/* Period multi-select */}
        <div className="flex items-center gap-1.5">
          <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)] pr-1">
            Period
          </span>
          {[9, 50, 600].map((p) => {
            const on = periodFilter.has(p);
            const c = counts[String(p)];
            const subtotal = c ? c.bull + c.bear : 0;
            return (
              <button
                key={p}
                type="button"
                onClick={() => togglePeriod(p)}
                aria-pressed={on}
                className="group inline-flex items-center gap-1.5 rounded-lg px-2 py-1 transition-all duration-200"
                style={{
                  background: on ? 'linear-gradient(180deg, rgba(167,139,250,0.28), rgba(167,139,250,0.08))' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${on ? 'rgba(167,139,250,0.55)' : 'rgba(255,255,255,0.06)'}`,
                  boxShadow: on ? '0 0 0 1px rgba(167,139,250,0.22) inset, 0 4px 14px -6px rgba(167,139,250,0.8)' : 'none',
                  color: on ? '#fff' : 'var(--text-secondary)',
                }}
              >
                <Layers className="w-3 h-3" style={{ color: on ? '#a78bfa' : 'var(--text-muted)' }} />
                <span className="text-[10.5px] font-semibold tabular-nums">SMA {p}</span>
                <span
                  className="inline-flex items-center justify-center rounded-md px-1 min-w-[18px] h-[15px] text-[9.5px] font-semibold tabular-nums"
                  style={{
                    background: on ? '#a78bfa' : 'rgba(255,255,255,0.05)',
                    color: on ? '#0b0c12' : 'var(--text-muted)',
                  }}
                >
                  {subtotal}
                </span>
              </button>
            );
          })}
        </div>

        <div className="h-5 w-px bg-white/[0.05]" aria-hidden />

        {/* Direction segmented */}
        <SegmentedControl
          options={[
            { key: 'all', label: 'All' },
            { key: 'bull', label: 'Bull', dot: '#34d399' },
            { key: 'bear', label: 'Bear', dot: '#fb7185' },
          ] as const}
          value={direction}
          onChange={(v) => setDirection(v)}
          accent="#a78bfa"
          size="sm"
        />

        <div className="h-5 w-px bg-white/[0.05]" aria-hidden />

        {/* Grade filter */}
        <SegmentedControl
          options={[
            { key: 'all', label: 'Any' },
            { key: 'A', label: 'A', dot: '#10b981' },
            { key: 'B', label: 'B', dot: '#a78bfa' },
            { key: 'C', label: 'C', dot: '#64748b' },
          ] as const}
          value={gradeFilter}
          onChange={(v) => setGradeFilter(v)}
          accent="#10b981"
          size="sm"
        />

        <div className="h-5 w-px bg-white/[0.05]" aria-hidden />

        {/* Min score slider */}
        <label className="flex items-center gap-2">
          <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">Min Score</span>
          <input
            type="range"
            min={0}
            max={100}
            step={5}
            value={minScore}
            onChange={(e) => setMinScore(parseInt(e.target.value, 10))}
            className="w-24 accent-[var(--accent-violet)]"
            aria-label="Minimum score"
          />
          <span className="text-[11px] tabular-nums text-[var(--text-primary)] font-semibold w-6 text-right">{minScore}</span>
        </label>

        <div className="h-5 w-px bg-white/[0.05]" aria-hidden />

        {/* False break toggle */}
        <button
          type="button"
          onClick={() => setHideFalseBreaks((v) => !v)}
          className="inline-flex items-center gap-1.5 px-2 py-1 rounded-lg text-[10.5px] font-medium transition-all"
          style={{
            background: hideFalseBreaks ? 'rgba(255,255,255,0.02)' : 'rgba(251,191,36,0.12)',
            border: `1px solid ${hideFalseBreaks ? 'rgba(255,255,255,0.06)' : 'rgba(251,191,36,0.35)'}`,
            color: hideFalseBreaks ? 'var(--text-secondary)' : '#fbbf24',
          }}
          title={hideFalseBreaks ? 'Show false breaks (whipsawed crossings)' : 'Hide false breaks'}
        >
          <AlertTriangle className="w-3 h-3" />
          {hideFalseBreaks ? 'Hiding false breaks' : 'Showing false breaks'}
        </button>
      </div>

      {/* Body */}
      <div className="px-2 py-2">
        {isLoading && (
          <div className="px-3 py-10 text-center text-[12px] text-[var(--text-muted)]">Loading reversals…</div>
        )}
        {!isLoading && filtered.length === 0 && (
          <div className="px-3 py-10 text-center text-[12px] text-[var(--text-muted)]">
            No reversals matching the current filters.
          </div>
        )}
        {!isLoading && displayed.length > 0 && (
          <div className="flex flex-col gap-1.5">
            {displayed.map((r) => {
              const key = `${r.symbol}-${r.period}`;
              const isExpanded = expandedKey === key;
              const label = labelMap.get(r.symbol) ?? labelMap.get(r.symbol.replace(/=/g, '_')) ?? r.symbol;
              const qualityScore = lookupQualityScore(qualityScores, r.symbol);
              return (
                <React.Fragment key={key}>
                  <ReversalRow
                    r={r}
                    label={label}
                    qualityScore={qualityScore}
                    isExpanded={isExpanded}
                    onClick={() => setExpandedKey((prev) => (prev === key ? null : key))}
                  />
                  {isExpanded && (
                    <div>
                      <ReversalDetailPanel
                        r={r}
                        label={label}
                        qualityScore={qualityScore}
                        chartView={detailChartView}
                        chartRange={detailChartRange}
                        onChartViewChange={setDetailChartView}
                        onChartRangeChange={setDetailChartRange}
                        onClose={() => setExpandedKey(null)}
                        onOpenFullChart={() => onNavigateChart(r.symbol)}
                      />
                    </div>
                  )}
                </React.Fragment>
              );
            })}
          </div>
        )}
        {!isLoading && filtered.length > 20 && (
          <div className="flex justify-center pt-2">
            <button
              onClick={() => setShowAll((v) => !v)}
              className="text-[11px] text-[var(--accent-violet)] hover:underline px-3 py-1.5"
            >
              {showAll ? `Collapse · show top 20` : `Show all ${filtered.length} reversals`}
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

// Individual row — compact premium card
// Small badge used in the header summary row ("A 51 · B 87 · C 184")
function GradeBadge({ grade, count }: { grade: 'A' | 'B' | 'C'; count: number }) {
  const palette = {
    A: { bg: 'rgba(16,185,129,0.16)', bd: 'rgba(16,185,129,0.45)', fg: '#34d399' },
    B: { bg: 'rgba(167,139,250,0.14)', bd: 'rgba(167,139,250,0.42)', fg: '#c4b5fd' },
    C: { bg: 'rgba(100,116,139,0.14)', bd: 'rgba(100,116,139,0.35)', fg: '#cbd5e1' },
  }[grade];
  return (
    <span
      className="inline-flex items-center gap-1 rounded-md px-1.5 py-[1px] text-[9.5px] font-semibold tabular-nums"
      style={{ background: palette.bg, border: `1px solid ${palette.bd}`, color: palette.fg }}
      title={`Grade ${grade}: ${count} setup${count === 1 ? '' : 's'}`}
    >
      <span style={{ fontWeight: 800 }}>{grade}</span>
      <span>{count}</span>
    </span>
  );
}

function SmaScoreRing({ score, color }: { score: number; color: string }) {
  const pct = Math.max(0, Math.min(100, score));
  const radius = 20.5;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference * (1 - pct / 100);
  const tier = pct >= 80 ? 'Elite' : pct >= 60 ? 'Strong' : pct >= 40 ? 'Watch' : 'Low';

  return (
    <div
      className="flex w-[68px] flex-shrink-0 flex-col items-center justify-center gap-1"
      title={`Setup score: ${pct.toFixed(0)} (${tier})`}
    >
      <div
        className="relative h-[54px] w-[54px] rounded-full"
        style={{
          background: `radial-gradient(circle, ${color}14 0%, ${color}08 42%, rgba(255,255,255,0.012) 72%)`,
          border: '1px solid rgba(255,255,255,0.055)',
          boxShadow: `inset 0 1px 0 rgba(255,255,255,0.055), 0 0 18px -14px ${color}`,
        }}
      >
        <svg className="absolute inset-0 h-full w-full -rotate-90" viewBox="0 0 56 56" aria-hidden>
          <circle
            cx="28"
            cy="28"
            r={radius}
            fill="none"
            stroke="rgba(255,255,255,0.075)"
            strokeWidth="4.5"
          />
          <circle
            cx="28"
            cy="28"
            r={radius}
            fill="none"
            stroke={color}
            strokeWidth="4.5"
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={dashOffset}
            style={{
              transition: 'stroke-dashoffset 360ms cubic-bezier(.2,.8,.2,1)',
            }}
          />
        </svg>
        <div className="absolute inset-0 flex flex-col items-center justify-center">
          <span className="text-[15px] font-extrabold tabular-nums leading-none" style={{ color }}>
            {pct.toFixed(0)}
          </span>
        </div>
      </div>
      <div className="flex items-center justify-center gap-1 whitespace-nowrap text-[7.5px] font-semibold uppercase tracking-[0.08em]">
        <span className="text-[var(--text-muted)]">Score</span>
        <span className="h-1 w-1 rounded-full" style={{ background: color, opacity: 0.65 }} />
        <span style={{ color, opacity: 0.9 }}>{tier}</span>
      </div>
    </div>
  );
}

function ReversalRow({ r, label, qualityScore, onClick, isExpanded = false }: {
  r: SmaReversal;
  label: string;
  qualityScore: number | null;
  onClick: () => void;
  isExpanded?: boolean;
}) {
  const isBull = r.direction === 'bull';
  const accent = isBull ? '#10b981' : '#f43f5e';
  const accentSoft = isBull ? '#34d399' : '#fb7185';
  const ArrowIcon = isBull ? TrendingUp : TrendingDown;
  const scoreColor = r.score >= 80 ? accent : r.score >= 60 ? accentSoft : r.score >= 40 ? '#fbbf24' : '#64748b';

  const persistencePips = Array.from({ length: r.persistence_window }, (_, i) => i < r.persistence);

  // Grade palette
  const gradePalette: Record<'A' | 'B' | 'C', { bg: string; bd: string; fg: string; shadow: string; title: string }> = {
    A: { bg: 'linear-gradient(180deg, rgba(16,185,129,0.30), rgba(16,185,129,0.08))', bd: 'rgba(16,185,129,0.60)', fg: '#ffffff', shadow: '0 0 14px -4px rgba(16,185,129,0.85)', title: 'Grade A — full confluence buy setup' },
    B: { bg: 'linear-gradient(180deg, rgba(167,139,250,0.28), rgba(167,139,250,0.08))', bd: 'rgba(167,139,250,0.55)', fg: '#ffffff', shadow: '0 0 12px -4px rgba(167,139,250,0.75)', title: 'Grade B — tradeable setup' },
    C: { bg: 'rgba(100,116,139,0.16)', bd: 'rgba(100,116,139,0.35)', fg: '#cbd5e1', shadow: 'none', title: 'Grade C — watch only (regime or R:R weak)' },
  };
  const g = r.grade ? gradePalette[r.grade] : null;
  const edge = r.historical_edge;
  const hasTradeGeometry = r.stop_price !== null && r.target_price !== null && r.risk_reward !== null;

  // Strip the trailing `(TICKER)` from the label for the body text
  const displayLabel = label.replace(/\s*\([^)]+\)\s*$/, '').trim() || r.symbol;
  const tooltip = r.grade_reasons && r.grade_reasons.length > 0 ? r.grade_reasons.map(cleanSmaReason).join(' · ') : undefined;
  const metricItems = [
    {
      key: 'dist',
      label: 'Dist',
      value: `${r.distance_pct > 0 ? '+' : ''}${r.distance_pct.toFixed(2)}%`,
      color: 'var(--text-primary)',
    },
    ...(r.atr_distance !== null ? [{
      key: 'atr',
      label: 'ATR',
      value: `${r.atr_distance.toFixed(2)}σ`,
      color: Math.abs(r.atr_distance) >= 2 ? '#fbbf24' : 'var(--text-primary)',
    }] : []),
    ...(r.volume_ratio !== null ? [{
      key: 'vol',
      label: 'Vol',
      value: `${r.volume_ratio.toFixed(2)}×`,
      color: r.volume_ratio >= 1.2 ? accentSoft : 'var(--text-primary)',
    }] : []),
    {
      key: 'age',
      label: 'Age',
      value: r.days_since_cross === 0 ? 'Today' : `${r.days_since_cross}d`,
      color: r.days_since_cross <= 1 ? accentSoft : 'var(--text-primary)',
    },
  ];
  const persistenceLabel = `${r.persistence}/${r.persistence_window}`;

  return (
    <button
      type="button"
      onClick={onClick}
      title={tooltip}
      aria-expanded={isExpanded}
      className="group relative w-full text-left rounded-xl px-3 py-3 transition-all duration-200"
      style={{
        background: isExpanded
          ? `linear-gradient(180deg, ${accent}16, ${accent}04)`
          : 'linear-gradient(180deg, rgba(255,255,255,0.028), rgba(255,255,255,0.008))',
        border: `1px solid ${isExpanded ? `${accent}70` : 'rgba(255,255,255,0.06)'}`,
        boxShadow: isExpanded
          ? `0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 28px -14px ${accent}80`
          : '0 1px 0 rgba(255,255,255,0.03) inset',
      }}
      onMouseEnter={(e) => {
        if (isExpanded) return;
        (e.currentTarget as HTMLButtonElement).style.borderColor = `${accent}55`;
        (e.currentTarget as HTMLButtonElement).style.boxShadow = `0 1px 0 rgba(255,255,255,0.04) inset, 0 6px 22px -12px ${accent}70`;
      }}
      onMouseLeave={(e) => {
        if (isExpanded) return;
        (e.currentTarget as HTMLButtonElement).style.borderColor = 'rgba(255,255,255,0.06)';
        (e.currentTarget as HTMLButtonElement).style.boxShadow = '0 1px 0 rgba(255,255,255,0.03) inset';
      }}
    >
      <div className="grid gap-2.5">
        <div className="grid gap-3 lg:grid-cols-[minmax(176px,0.52fr)_minmax(500px,1.82fr)_226px_18px] lg:items-center">
          <div className="flex min-h-[76px] items-center gap-3 min-w-0">
            <div
              className="flex items-center justify-center rounded-lg flex-shrink-0"
              style={{
                width: 38, height: 38,
                background: `linear-gradient(180deg, ${accent}26, ${accent}08)`,
                border: `1px solid ${accent}50`,
                boxShadow: `0 0 14px -5px ${accent}70 inset`,
              }}
            >
              <ArrowIcon className="w-4 h-4" style={{ color: accentSoft, filter: `drop-shadow(0 0 4px ${accent}90)` }} />
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex flex-wrap items-center gap-1.5">
                <span className="text-[14px] font-semibold text-[var(--text-primary)] tabular-nums tracking-tight">{r.symbol}</span>
                {g && (
                  <span
                    className="inline-flex items-center justify-center rounded px-1.5 py-[1px] text-[9.5px] font-bold tabular-nums"
                    style={{ background: g.bg, border: `1px solid ${g.bd}`, color: g.fg, boxShadow: g.shadow }}
                    title={g.title}
                  >
                    {r.grade}
                  </span>
                )}
                <span
                  className="text-[9px] font-semibold uppercase tracking-[0.1em] rounded px-1.5 py-[1px]"
                  style={{ background: 'rgba(167,139,250,0.14)', border: '1px solid rgba(167,139,250,0.28)', color: '#c4b5fd' }}
                >
                  SMA {r.period}
                </span>
                <span
                  className="text-[9px] font-semibold uppercase tracking-[0.1em] rounded px-1.5 py-[1px]"
                  style={{ background: `${accent}14`, border: `1px solid ${accent}30`, color: accentSoft }}
                >
                  {isBull ? 'Bull cross' : 'Bear cross'}
                </span>
                {!r.regime_ok && (
                  <span
                    className="inline-flex items-center gap-0.5 rounded px-1 py-[1px] text-[8.5px] font-semibold uppercase tracking-[0.1em]"
                    style={{ background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.30)', color: '#fca5a5' }}
                    title={`Against regime (price ${isBull ? 'below' : 'above'} SMA${r.regime_sma !== null ? ' 200' : ''})`}
                  >
                    vs regime
                  </span>
                )}
                {r.overextended && (
                  <span
                    className="inline-flex items-center gap-0.5 rounded px-1 py-[1px] text-[8.5px] font-semibold uppercase tracking-[0.1em]"
                    style={{ background: 'rgba(251,191,36,0.12)', border: '1px solid rgba(251,191,36,0.35)', color: '#fbbf24' }}
                    title="Price > 3 ATR from SMA"
                  >
                    overext
                  </span>
                )}
                {r.false_break && (
                  <span title="Price re-crossed within 3 bars">
                    <AlertTriangle className="w-3 h-3" style={{ color: '#fbbf24' }} />
                  </span>
                )}
              </div>
              <span className="block text-[10.5px] text-[var(--text-muted)] truncate mt-0.5">{displayLabel}</span>
            </div>
          </div>

          <div
            className="rounded-lg h-[76px] px-3 py-2 flex items-center justify-center overflow-hidden min-w-0 transition-transform duration-200 group-hover:scale-[1.006]"
            style={{
              background: `linear-gradient(180deg, ${accent}0d, rgba(255,255,255,0.012))`,
              border: `1px solid ${accent}24`,
              boxShadow: `0 0 18px -15px ${accent} inset`,
            }}
            aria-label={`${r.symbol} mini chart`}
          >
            <Sparkline ticker={r.symbol} width={560} height={62} tail={220} variant="reversal" fluid />
          </div>

          <div
            className="h-[76px] rounded-lg px-3 py-2 flex items-center justify-between gap-3"
            style={{
              background: 'linear-gradient(180deg, rgba(255,255,255,0.028), rgba(255,255,255,0.010))',
              border: '1px solid rgba(255,255,255,0.055)',
            }}
          >
            <SmaQualityBadge score={qualityScore} compact />
            <SparklineReversalStateBadge ticker={r.symbol} tail={220} />
            <SmaScoreRing score={r.score} color={scoreColor} />
          </div>

          <ChevronRight
            className="hidden lg:block w-4 h-4 flex-shrink-0 transition-transform duration-200 justify-self-end"
            style={{
              color: isExpanded ? accentSoft : 'var(--text-muted)',
              transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
            }}
          />
        </div>

        <div className="grid gap-2 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,1fr)]">
          <div
            className="rounded-lg min-h-[36px] px-3 py-2 flex items-center"
            style={{
              background: 'rgba(255,255,255,0.012)',
              border: '1px solid rgba(255,255,255,0.032)',
            }}
          >
            <div className="flex w-full flex-wrap items-center gap-x-2.5 gap-y-1 text-[10px] tabular-nums">
              {hasTradeGeometry ? (
                <>
                  <span className="text-[var(--text-muted)]">
                    Stop <span className="text-[var(--text-secondary)] font-semibold">{r.stop_price!.toFixed(2)}</span>
                  </span>
                  <span className="text-white/15">·</span>
                  <span className="text-[var(--text-muted)]">
                    Target <span className="text-[var(--text-secondary)] font-semibold">{r.target_price!.toFixed(2)}</span>
                  </span>
                  <span className="text-white/15">·</span>
                  <span className="text-[var(--text-muted)]">
                    R:R <span className="font-semibold" style={{ color: accentSoft }}>{r.risk_reward!.toFixed(1)}</span>
                  </span>
                </>
              ) : (
                <span className="text-[var(--text-muted)]">Trade geometry unavailable</span>
              )}
              {edge.win_rate !== null && (
                <>
                  <span className="text-white/15">·</span>
                  <span
                    className="inline-flex items-center gap-1"
                    title={`Historical ${r.edge_forward_days}-bar forward win-rate across ${edge.samples} past crossings. Median return ${edge.median_fwd_pct !== null ? edge.median_fwd_pct.toFixed(2) + '%' : '—'}.`}
                  >
                    <Target className="w-3 h-3 opacity-80" style={{ color: edge.win_rate >= 0.55 ? accentSoft : '#94a3b8' }} />
                    <span
                      className="font-semibold"
                      style={{ color: edge.win_rate >= 0.55 ? accentSoft : 'var(--text-secondary)' }}
                    >
                      {(edge.win_rate * 100).toFixed(0)}%
                    </span>
                    <span className="text-[var(--text-muted)]">{r.edge_forward_days}d edge</span>
                  </span>
                </>
              )}
            </div>
          </div>

          <div
            className="rounded-lg min-h-[36px] px-3 py-2 flex items-center"
            style={{
              background: 'rgba(255,255,255,0.010)',
              border: '1px solid rgba(255,255,255,0.030)',
            }}
          >
            <div className="flex w-full flex-wrap items-center gap-x-3 gap-y-1 text-[10px] tabular-nums">
              {metricItems.map((item) => (
                <span key={item.key} className="inline-flex items-baseline gap-1">
                  <span className="text-[8px] uppercase tracking-[0.12em] text-[var(--text-muted)]">{item.label}</span>
                  <span className="font-semibold" style={{ color: item.color }}>{item.value}</span>
                </span>
              ))}
              <span className="inline-flex items-center gap-1" title={`On new side: ${r.persistence} of last ${r.persistence_window} bars`}>
                <span className="text-[8px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Persist</span>
                <span className="font-semibold" style={{ color: r.persistence >= r.persistence_threshold ? accentSoft : 'var(--text-secondary)' }}>
                  {persistenceLabel}
                </span>
                <span className="hidden sm:inline-flex items-center gap-0.5">
                  {persistencePips.map((on, i) => (
                    <span
                      key={i}
                      className="rounded-full"
                      style={{
                        width: 4.5, height: 4.5,
                        background: on ? accentSoft : 'rgba(255,255,255,0.1)',
                        boxShadow: on ? `0 0 4px ${accentSoft}` : 'none',
                      }}
                    />
                  ))}
                </span>
              </span>
            </div>
          </div>
        </div>
      </div>
    </button>
  );
}


function ReversalDetailPanel({ r, label, qualityScore, chartView, chartRange, onChartViewChange, onChartRangeChange, onClose, onOpenFullChart }: {
  r: SmaReversal;
  label: string;
  qualityScore: number | null;
  chartView: MiniPriceChartView;
  chartRange: SmaChartRange;
  onChartViewChange: (view: MiniPriceChartView) => void;
  onChartRangeChange: (range: SmaChartRange) => void;
  onClose: () => void;
  onOpenFullChart: () => void;
}) {
  const ohlcvQ = useQuery({
    queryKey: ['sma-reversal-ohlcv', r.symbol, SMA_DETAIL_MAX_TAIL],
    queryFn: () => api.chartOhlcv(r.symbol, SMA_DETAIL_MAX_TAIL),
    staleTime: 120_000,
  });
  const indQ = useQuery({
    queryKey: ['sma-reversal-indicators', r.symbol, SMA_DETAIL_MAX_TAIL],
    queryFn: () => api.chartIndicators(r.symbol, SMA_DETAIL_MAX_TAIL),
    staleTime: 120_000,
  });
  const forecastQ = useQuery({
    queryKey: ['sma-reversal-forecast', r.symbol],
    queryFn: () => api.chartForecast(r.symbol),
    staleTime: 120_000,
  });

  const isBull = r.direction === 'bull';
  const accent = isBull ? '#10b981' : '#f43f5e';
  const accentSoft = isBull ? '#34d399' : '#fb7185';
  const displayLabel = label.replace(/\s*\([^)]+\)\s*$/, '').trim();
  const showLabel = displayLabel && displayLabel !== r.symbol;
  const edge = r.historical_edge;
  const qualityTone = smaQualityTone(qualityScore);
  const chartViews: Array<{ key: MiniPriceChartView; label: string; icon: ReactNode; title: string }> = [
    { key: 'sma', label: 'SMA', icon: <Activity className="w-3 h-3" />, title: 'Current SMA structure with overlays' },
    { key: 'heikinAshi', label: 'Heikin Ashi', icon: <BarChart3 className="w-3 h-3" />, title: 'Smoothed candles for trend clarity' },
    { key: 'reversal', label: 'Reversal', icon: <RefreshCw className="w-3 h-3" />, title: 'BUY and SELL reversal flips' },
    { key: 'area', label: 'Area', icon: <TrendingUp className="w-3 h-3" />, title: 'Reversal-colored trend gradient' },
  ];
  const chartRanges: Array<{ key: SmaChartRange; label: string; title: string }> = [
    { key: '1M', label: '1M', title: 'Show roughly one month' },
    { key: '3M', label: '3M', title: 'Show roughly three months' },
    { key: '6M', label: '6M', title: 'Show roughly six months' },
    { key: '1Y', label: '1Y', title: 'Show roughly one year' },
    { key: 'MAX', label: 'MAX', title: 'Show the longest available history' },
  ];

  const visibleOhlcv = useMemo(() => {
    const bars = ohlcvQ.data?.data ?? [];
    if (chartRange === 'MAX') return bars;
    const days = SMA_CHART_RANGE_DAYS[chartRange];
    return bars.length > days ? bars.slice(-days) : bars;
  }, [chartRange, ohlcvQ.data?.data]);

  const visibleIndicators = useMemo(() => {
    const raw = indQ.data?.indicators;
    if (!raw || chartRange === 'MAX') return raw;
    const visibleTimes = new Set(visibleOhlcv.map((bar) => bar.time));
    const keep = <T extends { time: string },>(series?: T[]): T[] | undefined =>
      series?.filter((point) => visibleTimes.has(point.time));
    return {
      ...raw,
      sma20: keep(raw.sma20),
      sma50: keep(raw.sma50),
      sma200: keep(raw.sma200),
      bollinger: raw.bollinger
        ? {
            upper: keep(raw.bollinger.upper) ?? [],
            lower: keep(raw.bollinger.lower) ?? [],
          }
        : undefined,
    };
  }, [chartRange, indQ.data?.indicators, visibleOhlcv]);

  const fmtPx = (v: number | null | undefined): string => {
    if (v == null || !isFinite(v)) return '—';
    const abs = Math.abs(v);
    if (abs < 1) return v.toFixed(4);
    if (abs < 100) return v.toFixed(2);
    return v.toFixed(2);
  };

  const stats: Array<{ label: string; value: string; sub?: string; color?: string }> = [
    { label: 'Entry', value: fmtPx(r.price) },
    {
      label: 'Quality',
      value: qualityScore != null ? Math.round(qualityScore).toString() : '—',
      sub: qualityScore != null ? qualityTone.label : 'unavailable',
      color: qualityTone.color,
    },
    { label: 'Stop', value: fmtPx(r.stop_price), color: '#fb7185' },
    { label: 'Target', value: fmtPx(r.target_price), color: '#34d399' },
    { label: 'R : R', value: r.risk_reward != null ? r.risk_reward.toFixed(2) : '—' },
    {
      label: 'Win %',
      value: edge?.win_rate != null ? `${(edge.win_rate * 100).toFixed(0)}%` : '—',
      sub: edge?.win_rate != null ? `${r.edge_forward_days}d edge` : undefined,
    },
    { label: 'Age', value: `${r.days_since_cross}d`, sub: `since cross` },
  ];

  return (
    <div
      className="mx-0.5 my-0.5 rounded-2xl overflow-hidden"
      style={{
        background: 'linear-gradient(160deg, rgba(13,5,30,0.96) 0%, rgba(10,18,42,0.96) 100%)',
        border: '1px solid rgba(167,139,250,0.18)',
        boxShadow: `0 20px 60px -24px rgba(0,0,0,0.85), 0 0 0 1px rgba(255,255,255,0.02), inset 0 1px 0 rgba(167,139,250,0.08), 0 0 46px -20px ${accent}4D`,
      }}
    >
      {/* Header */}
      <div
        className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid rgba(167,139,250,0.10)' }}
      >
        <div className="flex items-center gap-3 min-w-0">
          <div className="flex flex-col min-w-0">
            <span className="text-[15px] font-semibold text-[#f1f5f9] tracking-tight leading-tight">{r.symbol}</span>
            {showLabel && (
              <span className="text-[10px] text-[var(--text-muted)] font-medium truncate max-w-[260px]">{displayLabel}</span>
            )}
          </div>
          <span
            className="px-2 py-0.5 rounded text-[10px] font-semibold tracking-wide uppercase tabular-nums whitespace-nowrap"
            style={{ background: `${accent}1F`, border: `1px solid ${accent}55`, color: accent }}
          >
            {isBull ? 'Bull' : 'Bear'} · SMA {r.period}
          </span>
          {r.grade && (
            <span
              className="px-2 py-0.5 rounded text-[10px] font-bold tabular-nums whitespace-nowrap"
              style={{
                background:
                  r.grade === 'A'
                    ? 'linear-gradient(180deg, rgba(16,185,129,0.30), rgba(16,185,129,0.08))'
                    : r.grade === 'B'
                    ? 'linear-gradient(180deg, rgba(167,139,250,0.28), rgba(167,139,250,0.08))'
                    : 'rgba(100,116,139,0.20)',
                border:
                  r.grade === 'A'
                    ? '1px solid rgba(16,185,129,0.55)'
                    : r.grade === 'B'
                    ? '1px solid rgba(167,139,250,0.50)'
                    : '1px solid rgba(100,116,139,0.35)',
                color: r.grade === 'C' ? '#cbd5e1' : '#ffffff',
              }}
            >
              Grade {r.grade}
            </span>
          )}
        </div>
        <div className="flex items-center gap-1.5 flex-shrink-0">
          <button
            onClick={onOpenFullChart}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-all"
            style={{ background: 'rgba(167,139,250,0.10)', border: '1px solid rgba(167,139,250,0.25)', color: '#c4b5fd' }}
            onMouseEnter={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(167,139,250,0.18)')}
            onMouseLeave={(e) => ((e.currentTarget as HTMLButtonElement).style.background = 'rgba(167,139,250,0.10)')}
          >
            <ExternalLink className="w-3 h-3" />
            Full Chart
          </button>
          <button
            onClick={onClose}
            aria-label="Close detail"
            className="p-1.5 rounded-lg transition-all text-[var(--text-muted)] hover:text-[#f1f5f9] hover:bg-white/[0.05]"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      {/* Stats strip */}
      <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 xl:grid-cols-7 gap-1.5 px-4 pt-3 pb-1">
        {stats.map((s, i) => (
          <div
            key={i}
            className="rounded-lg px-2.5 py-2"
            style={{
              background: 'rgba(167,139,250,0.05)',
              border: '1px solid rgba(167,139,250,0.10)',
            }}
          >
            <div className="text-[8.5px] uppercase tracking-[0.08em] text-[var(--text-muted)] font-semibold mb-0.5">
              {s.label}
            </div>
            <div className="text-[13px] font-semibold tabular-nums leading-tight" style={{ color: s.color ?? '#f1f5f9' }}>
              {s.value}
            </div>
            {s.sub && <div className="text-[9px] text-[var(--text-muted)] tabular-nums mt-0.5">{s.sub}</div>}
          </div>
        ))}
      </div>

      {/* Chart */}
      <div className="px-4 pb-4 pt-2">
        <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
          <div
            className="inline-flex items-center gap-1 rounded-xl p-1"
            style={{
              background: 'rgba(255,255,255,0.028)',
              border: '1px solid rgba(255,255,255,0.06)',
              boxShadow: '0 1px 0 rgba(255,255,255,0.035) inset',
            }}
            aria-label="SMA reversal chart view"
          >
            {chartViews.map((view) => {
              const active = chartView === view.key;
              return (
                <button
                  key={view.key}
                  type="button"
                  title={view.title}
                  aria-pressed={active}
                  onClick={() => onChartViewChange(view.key)}
                  className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10.5px] font-semibold transition-all"
                  style={{
                    color: active ? '#ffffff' : 'var(--text-secondary)',
                    background: active ? `linear-gradient(180deg, ${accent}28, ${accent}0c)` : 'transparent',
                    border: `1px solid ${active ? `${accent}58` : 'transparent'}`,
                    boxShadow: active ? `0 0 0 1px ${accent}18 inset, 0 5px 18px -9px ${accent}` : 'none',
                  }}
                >
                  <span
                    className="inline-flex"
                    style={{ color: active ? accentSoft : 'var(--text-muted)', filter: active ? `drop-shadow(0 0 4px ${accent})` : 'none' }}
                  >
                    {view.icon}
                  </span>
                  <span className="whitespace-nowrap">{view.label}</span>
                </button>
              );
            })}
          </div>
          <div
            className="inline-flex items-center gap-1 rounded-xl p-1"
            style={{
              background: 'rgba(15,23,42,0.56)',
              border: '1px solid rgba(148,163,184,0.14)',
              boxShadow: '0 1px 0 rgba(255,255,255,0.035) inset',
            }}
            aria-label="SMA reversal chart range"
          >
            {chartRanges.map((range) => {
              const active = chartRange === range.key;
              return (
                <button
                  key={range.key}
                  type="button"
                  title={range.title}
                  aria-pressed={active}
                  onClick={() => onChartRangeChange(range.key)}
                  className="inline-flex min-w-[34px] justify-center rounded-lg px-2.5 py-1.5 text-[10.5px] font-bold tabular-nums transition-all"
                  style={{
                    color: active ? '#f8fafc' : 'var(--text-secondary)',
                    background: active ? `linear-gradient(180deg, ${accent}24, rgba(15,23,42,0.45))` : 'transparent',
                    border: `1px solid ${active ? `${accent}55` : 'transparent'}`,
                    boxShadow: active ? `0 0 0 1px ${accent}14 inset, 0 6px 18px -12px ${accent}` : 'none',
                  }}
                >
                  {range.label}
                </button>
              );
            })}
          </div>
        </div>
        {ohlcvQ.isLoading ? (
          <div className="h-[320px] flex items-center justify-center gap-2 text-[11px] text-[var(--text-muted)]">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading chart…
          </div>
        ) : ohlcvQ.error || !visibleOhlcv.length ? (
          <div className="h-[320px] flex items-center justify-center text-[11px] text-[var(--text-muted)]">
            No chart data available for {r.symbol}
          </div>
        ) : (
          <MiniPriceChart
            ohlcv={visibleOhlcv}
            indicators={visibleIndicators}
            forecast={forecastQ.data}
            height={320}
            viewMode={chartView}
            showOverlayControls
          />
        )}
      </div>

      {/* Grade reasons footer */}
      {r.grade_reasons && r.grade_reasons.length > 0 && (
        <div
          className="px-5 py-2.5 flex flex-wrap gap-1.5 text-[10px]"
          style={{ borderTop: '1px solid rgba(167,139,250,0.08)', background: 'rgba(0,0,0,0.25)' }}
        >
          <span className="text-[var(--text-muted)] font-medium uppercase tracking-wider text-[8.5px] self-center mr-1">Why</span>
          {r.grade_reasons.map((reason, i) => (
            <span
              key={i}
              className="px-2 py-0.5 rounded-full text-[9.5px] font-medium"
              style={{ background: 'rgba(167,139,250,0.06)', border: '1px solid rgba(167,139,250,0.15)', color: '#c4b5fd' }}
            >
              {cleanSmaReason(reason)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}
