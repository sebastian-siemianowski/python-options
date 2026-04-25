/* eslint-disable @typescript-eslint/no-explicit-any */
import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef, useCallback, Component, Fragment, type ReactNode, type ErrorInfo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { SummaryRow, SectorGroup, StrongSignalEntry, HighConvictionSignal, SignalSummaryData, SignalStats, EmaState, SmaReversal, SmaReversalsData } from '../api';
import { SignalTableSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { Sparkline, SparklinePct } from '../components/Sparkline';
import { SignalLabel, SignalStrengthMeter, MomentumBadge, CrashRiskHeat, HorizonCell, QualityCell } from '../components/SignalTableVisuals';
import { ColumnCustomizer, type ColumnDef } from '../components/ColumnCustomizer';
import SignalDetailPanel from '../components/SignalDetailPanel';
import { formatJobElapsed, useJobStore, type JobCounters, type JobMode, type JobStageMetric, type JobStatus } from '../stores/jobStore';
import {
  Filter, ChevronDown, ChevronRight,
  TrendingUp, TrendingDown, Search, X, ExternalLink, BarChart3,
  Target, Shield, ShieldCheck, ArrowUp, ArrowDown, Clock,
  Activity, Eye, Layers, ChevronUp, AlertTriangle, Zap, Loader2,
  Star, Plus, SlidersHorizontal, RefreshCw, Play, Square,
} from 'lucide-react';
import MiniPriceChart from '../components/MiniPriceChart';
import { formatHorizon, responsiveHorizons } from '../utils/horizons';

import { useWebSocket, type WSStatus } from '../hooks/useWebSocket';
import { useWatchlist } from '../hooks/useWatchlist';

/** Extract ticker from "Company Name (TICKER)" */
const extractTicker = (label: string): string => {
  if (label.includes('(')) return label.split('(').pop()!.replace(')', '').trim();
  return label;
};

/** Determine dominant horizon color for a row (majority of available horizon cells).
 *  Returns 'greens' if >50% positive exp_ret, 'reds' if >50% negative, 'mixed' otherwise. */
const rowHorizonColor = (row: SummaryRow): 'greens' | 'reds' | 'mixed' => {
  const sigs = Object.values(row.horizon_signals || {});
  if (!sigs.length) return 'mixed';
  let pos = 0;
  let neg = 0;
  for (const s of sigs) {
    const r = s?.exp_ret;
    if (typeof r !== 'number' || Number.isNaN(r)) continue;
    if (r > 0) pos++;
    else if (r < 0) neg++;
  }
  const total = pos + neg;
  if (total === 0) return 'mixed';
  if (pos / total > 0.5) return 'greens';
  if (neg / total > 0.5) return 'reds';
  return 'mixed';
};

type ViewMode = 'all' | 'sectors' | 'strong';
type SignalFilter = 'all' | 'bullish' | 'bearish' | 'greens' | 'reds' | 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';

type SortColumn = 'asset' | 'sector' | 'signal' | 'momentum' | 'crash_risk' | `horizon_${number}`;
type SortDir = 'asc' | 'desc';

/* ── Error Boundary ──────────────────────────────────────────────── */
class SignalsErrorBoundary extends Component<
  { children: ReactNode },
  { hasError: boolean; error: Error | null }
> {
  constructor(props: { children: ReactNode }) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('SignalsPage crash:', error, info.componentStack);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="p-6">
          <div className="glass-card p-6 border border-red-500/50">
            <h2 className="text-red-400 text-lg font-bold mb-2">Signals Page Error</h2>
            <p className="text-red-300 text-sm mb-3">{this.state.error?.message}</p>
            <pre className="text-[var(--text-secondary)] text-xs overflow-auto max-h-48 bg-[#0a0a1a] p-3 rounded">
              {this.state.error?.stack}
            </pre>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="mt-3 px-3 py-1 rounded text-sm"
              style={{ background: 'var(--violet-15)', color: '#b49aff' }}
            >
              Retry
            </button>
          </div>
        </div>
      );
    }
    return this.props.children;
  }
}

/* ── Story 6.1: Responsive width hook ─────────────────────────── */
function useWindowWidth(): number {
  const [width, setWidth] = useState(typeof window !== 'undefined' ? window.innerWidth : 1440);
  useEffect(() => {
    const onResize = () => setWidth(window.innerWidth);
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);
  return width;
}

function SignalsPageInner() {
  const navigate = useNavigate();
  const startJob = useJobStore((s) => s.startJob);
  const showJobSurface = useJobStore((s) => s.showSurface);
  const setJobExpanded = useJobStore((s) => s.setExpanded);
  const jobStatus = useJobStore((s) => s.status);
  const activeJobMode = useJobStore((s) => s.mode);
  const jobCounters = useJobStore((s) => s.counters);
  const jobStageMetrics = useJobStore((s) => s.stageMetrics);
  const jobActiveStageKey = useJobStore((s) => s.activeStageKey);
  const jobElapsedSec = useJobStore((s) => s.elapsedSec);
  const jobPhases = useJobStore((s) => s.phases);
  const stopJob = useJobStore((s) => s.stopJob);
  const isJobRunning = jobStatus === 'running';
  // v1 premium: default to 'all' so users see all 490 assets immediately (was 'sectors' which showed collapsed empty sectors)
  const [view, setView] = useState<ViewMode>(() => {
    try {
      const stored = localStorage.getItem('signals-view');
      if (stored === 'all' || stored === 'sectors' || stored === 'strong') return stored as ViewMode;
    } catch { /* ignore */ }
    return 'all';
  });
  useEffect(() => { try { localStorage.setItem('signals-view', view); } catch { /* ignore */ } }, [view]);
  const [filter, setFilter] = useState<SignalFilter>('all');
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [expandedSectors, setExpandedSectors] = useState<Set<string>>(new Set());
  // EMA-below filters — apply to all three views. Multi-select, AND-combined.
  const [emaFilters, setEmaFilters] = useState<{ p9: boolean; p50: boolean; p600: boolean }>(
    () => {
      try {
        const stored = localStorage.getItem('signals-ema-filters');
        if (stored) {
          const p = JSON.parse(stored);
          return { p9: !!p.p9, p50: !!p.p50, p600: !!p.p600 };
        }
      } catch { /* ignore */ }
      return { p9: false, p50: false, p600: false };
    }
  );
  useEffect(() => {
    try { localStorage.setItem('signals-ema-filters', JSON.stringify(emaFilters)); } catch { /* ignore */ }
  }, [emaFilters]);
  const [updatedAsset, setUpdatedAsset] = useState<string | null>(null);

  // Sector-view controls (lifted out of SectorPanels so the entire Signals
  // filter surface lives in one unified premium card). SectorPanels reads
  // sectorSort + sectorVisibleCols as props.
  const [sectorSort, setSectorSort] = useState<SectorSortBy>('momentum');
  const [sectorVisibleCols, setSectorVisibleCols] = useState<Set<string>>(() => loadSectorVisibleCols());
  useEffect(() => {
    try { localStorage.setItem(SECTOR_COLS_LS_KEY, JSON.stringify(Array.from(sectorVisibleCols))); } catch { /* ignore */ }
  }, [sectorVisibleCols]);
  const toggleSectorCol = (key: string) => {
    setSectorVisibleCols((prev) => {
      const def = SECTOR_COLUMN_DEFS.find((c) => c.key === key);
      if (def?.locked) return prev;
      const next = new Set(prev);
      if (next.has(key)) next.delete(key); else next.add(key);
      return next;
    });
  };
  const resetSectorCols = () => setSectorVisibleCols(new Set(DEFAULT_SECTOR_VISIBLE_COLS));

  // Story 3.4: Change tracking for aurora trails + ticker tape
  type ChangeEntry = { asset: string; from: string; to: string; time: number };
  const [changeLog, setChangeLog] = useState<ChangeEntry[]>([]);
  const [showTickerTape, setShowTickerTape] = useState(false);
  const [awayChanges, setAwayChanges] = useState<ChangeEntry[]>([]);
  const changeCountRef = useRef(0);

  // Story 3.2: Multi-axis sort (up to 3 levels, persisted in localStorage)
  type SortLevel = { col: SortColumn; dir: SortDir };
  const sortKey = `signals-sort-${view}`;
  const [sortLevels, setSortLevels] = useState<SortLevel[]>(() => {
    try {
      const stored = localStorage.getItem(sortKey);
      if (stored) return JSON.parse(stored);
    } catch { /* ignore */ }
    return [{ col: 'momentum' as SortColumn, dir: 'desc' as SortDir }];
  });
  // Persist sort state
  useEffect(() => {
    try { localStorage.setItem(sortKey, JSON.stringify(sortLevels)); } catch { /* ignore */ }
  }, [sortLevels, sortKey]);

  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  // Story 3.5: Debounce search (100ms)
  useEffect(() => {
    const timer = setTimeout(() => setDebouncedSearch(search), 100);
    return () => clearTimeout(timer);
  }, [search]);

  // Story 3.5: Cmd+K or / shortcut to focus search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey && e.key === 'k') || (e.key === '/' && !['INPUT', 'TEXTAREA'].includes((e.target as HTMLElement).tagName))) {
        e.preventDefault();
        searchRef.current?.focus();
      }
      if (e.key === 'Escape' && document.activeElement === searchRef.current) {
        setSearch('');
        searchRef.current?.blur();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const queryClient = useQueryClient();
  const { status: wsStatus, lastMessage } = useWebSocket('/ws');

  // Story 6.4 + 3.4: Real-time signal updates via WebSocket with change tracking
  useEffect(() => {
    if (!lastMessage || lastMessage.type !== 'signal_update') return;
    const summary = lastMessage.summary as SummaryRow | undefined;
    if (!summary?.asset_label) return;

    // Detect label change for aurora trail
    const oldRow = queryClient.getQueryData<SignalSummaryData>(['signalSummary'])
      ?.summary_rows.find(r => r.asset_label === summary.asset_label);
    const oldLabel = oldRow ? (oldRow.nearest_label || 'HOLD').toUpperCase() : '';
    const newLabel = (summary.nearest_label || 'HOLD').toUpperCase();
    const labelChanged = oldLabel && newLabel && oldLabel !== newLabel;

    queryClient.setQueryData<SignalSummaryData>(['signalSummary'], (old) => {
      if (!old) return old;
      const rows = old.summary_rows.map((r) =>
        r.asset_label === summary.asset_label ? { ...r, ...summary } : r
      );
      return { ...old, summary_rows: rows };
    });

    // Story 3.4: Track change
    if (labelChanged) {
      const entry: ChangeEntry = { asset: extractTicker(summary.asset_label), from: oldLabel, to: newLabel, time: Date.now() };
      if (document.hidden) {
        setAwayChanges(prev => [...prev, entry]);
      } else {
        setChangeLog(prev => [entry, ...prev].slice(0, 20));
        changeCountRef.current++;
      }
    }

    // Highlight animation
    setUpdatedAsset(summary.asset_label);
    const timer = setTimeout(() => setUpdatedAsset(null), 600);
    return () => clearTimeout(timer);
  }, [lastMessage, queryClient]);

  const { data, isLoading, error } = useQuery({
    queryKey: ['signalSummary'],
    queryFn: api.signalSummary,
  });

  const statsQ = useQuery({
    queryKey: ['signalStats'],
    queryFn: api.signalStats,
  });

  const sectorQ = useQuery({
    queryKey: ['signalsBySector'],
    queryFn: api.signalsBySector,
  });

  const strongQ = useQuery({
    queryKey: ['strongSignals'],
    queryFn: api.strongSignals,
  });

  const buyQ = useQuery({
    queryKey: ['highConvictionBuy'],
    queryFn: () => api.highConviction('buy'),
  });

  const sellQ = useQuery({
    queryKey: ['highConvictionSell'],
    queryFn: () => api.highConviction('sell'),
  });

  const qualityQ = useQuery({
    queryKey: ['qualityScores'],
    queryFn: api.qualityScores,
    staleTime: 60_000,
  });
  const qualityScores = qualityQ.data?.scores ?? {};

  const emaQ = useQuery({
    queryKey: ['emaStates'],
    queryFn: api.emaStates,
    staleTime: 5 * 60_000,
  });
  const emaStates = emaQ.data?.states ?? {};

  const reversalsQ = useQuery({
    queryKey: ['smaReversals'],
    queryFn: api.smaReversals,
    staleTime: 60_000,
  });

  // asset_label is a display label like "Euro / US Dollar (EURUSD=X)".
  // Extract the ticker from the trailing parenthetical and normalise FX
  // variants (EURUSD=X on the API ↔ EURUSD_X on disk / in emaStates).
  const emaLookup = useCallback((assetLabel: string | undefined | null) => {
    if (!assetLabel) return undefined;
    const m = assetLabel.match(/\(([^)]+)\)\s*$/);
    const raw = (m ? m[1] : assetLabel).trim();
    return (
      emaStates[raw] ??
      emaStates[raw.replace(/=/g, '_')] ??
      emaStates[raw.replace(/_/g, '=')] ??
      emaStates[raw.toUpperCase()]
    );
  }, [emaStates]);

  const rows = data?.summary_rows || [];
  const allHorizons = data?.horizons || [];
  const windowWidth = useWindowWidth();

  // Story 3.6: Horizon pill selector with localStorage override
  const [horizonOverride, setHorizonOverride] = useState<number[] | null>(() => {
    try {
      const stored = localStorage.getItem('signals-horizons');
      if (stored) return JSON.parse(stored);
    } catch { /* ignore */ }
    return null;
  });
  const autoHorizons = useMemo(() => responsiveHorizons(allHorizons, windowWidth), [allHorizons, windowWidth]);
  const horizons = horizonOverride ?? autoHorizons;

  const toggleHorizon = useCallback((h: number) => {
    setHorizonOverride(prev => {
      const current = prev ?? autoHorizons;
      const next = current.includes(h) ? current.filter(x => x !== h) : [...current, h].sort((a, b) => a - b);
      if (next.length === 0) return prev; // don't allow empty
      try { localStorage.setItem('signals-horizons', JSON.stringify(next)); } catch { /* ignore */ }
      return next;
    });
  }, [autoHorizons]);

  const resetHorizons = useCallback(() => {
    setHorizonOverride(null);
    try { localStorage.removeItem('signals-horizons'); } catch { /* ignore */ }
  }, []);
  const stats = statsQ.data;
  const rawSectors = sectorQ.data?.sectors || [];

  // v1 premium: auto-expand sectors on first load so 'sectors' view isn't an empty shell
  const sectorsAutoExpandedRef = useRef(false);
  useEffect(() => {
    if (!sectorsAutoExpandedRef.current && rawSectors.length > 0) {
      sectorsAutoExpandedRef.current = true;
      setExpandedSectors(new Set(rawSectors.map(s => s.name)));
    }
  }, [rawSectors]);

  // Story 3.5: Fuzzy match scoring
  const fuzzyMatch = useCallback((text: string, query: string): boolean => {
    if (!query) return true;
    const t = text.toLowerCase();
    const q = query.toLowerCase();
    if (t.includes(q)) return true; // substring match
    // Character-skip fuzzy
    let qi = 0;
    for (let ti = 0; ti < t.length && qi < q.length; ti++) {
      if (t[ti] === q[qi]) qi++;
    }
    return qi === q.length;
  }, []);

  // EMA-below predicate (keyed by asset_label / symbol). Missing EMA data
  // for a ticker = fail any active EMA toggle.
  const passesEma = useCallback((ticker: string | undefined | null): boolean => {
    if (!emaFilters.p9 && !emaFilters.p50 && !emaFilters.p600) return true;
    const st = emaLookup(ticker);
    if (!st) return false;
    if (emaFilters.p9 && st.below_9 !== true) return false;
    if (emaFilters.p50 && st.below_50 !== true) return false;
    if (emaFilters.p600 && st.below_600 !== true) return false;
    return true;
  }, [emaFilters, emaLookup]);

  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (debouncedSearch && !fuzzyMatch(row.asset_label, debouncedSearch)) return false;
      if (!passesEma(row.asset_label)) return false;
      if (filter === 'all') return true;
      const label = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
      if (filter === 'bullish') return label === 'STRONG_BUY' || label === 'BUY';
      if (filter === 'bearish') return label === 'STRONG_SELL' || label === 'SELL';
      if (filter === 'greens' || filter === 'reds') return rowHorizonColor(row) === filter;
      return label === filter.toUpperCase();
    });
  }, [rows, debouncedSearch, filter, fuzzyMatch, passesEma]);

  // Sectors view: apply EMA predicate at the asset level, drop empty sectors.
  const sectors = useMemo(() => {
    if (!emaFilters.p9 && !emaFilters.p50 && !emaFilters.p600) return rawSectors;
    return rawSectors
      .map(sec => ({ ...sec, assets: sec.assets.filter(a => passesEma(a.asset_label)) }))
      .filter(sec => sec.assets.length > 0);
  }, [rawSectors, emaFilters, passesEma]);

  // Global sector totals shown in the unified filter card footer.
  const sectorTotals = useMemo(() => ({
    assets: sectors.reduce((s, sec) => s + sec.asset_count, 0),
    bullish: sectors.reduce((s, sec) => s + (sec.strong_buy ?? 0) + (sec.buy ?? 0), 0),
    bearish: sectors.reduce((s, sec) => s + (sec.strong_sell ?? 0) + (sec.sell ?? 0), 0),
  }), [sectors]);

  /** Story 3.2: Multi-level sorted rows */
  const sortedRows = useMemo(() => {
    const arr = [...filteredRows];
    const signalRank = (label: string): number => {
      const m: Record<string, number> = { 'STRONG BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG SELL': 1, 'EXIT': 0 };
      return m[label.toUpperCase()] ?? 3;
    };
    const getHorizonVal = (r: SummaryRow, h: number): number => {
      const sig = r.horizon_signals[h] || r.horizon_signals[String(h)];
      return sig?.exp_ret ?? 0;
    };
    const compare = (a: SummaryRow, b: SummaryRow, col: SortColumn): number => {
      switch (col) {
        case 'asset': return a.asset_label.localeCompare(b.asset_label);
        case 'sector': return (a.sector || '').localeCompare(b.sector || '');
        case 'signal': return signalRank(a.nearest_label || 'HOLD') - signalRank(b.nearest_label || 'HOLD');
        case 'momentum': return (a.momentum_score ?? 0) - (b.momentum_score ?? 0);
        case 'crash_risk': return (a.crash_risk_score ?? 0) - (b.crash_risk_score ?? 0);
        default:
          if (col.startsWith('horizon_')) {
            const h = parseInt(col.split('_')[1], 10);
            return getHorizonVal(a, h) - getHorizonVal(b, h);
          }
          return 0;
      }
    };
    arr.sort((a, b) => {
      for (const { col, dir } of sortLevels) {
        const cmp = compare(a, b, col);
        if (cmp !== 0) return dir === 'desc' ? -cmp : cmp;
      }
      return 0;
    });
    return arr;
  }, [filteredRows, sortLevels]);

  /** Story 3.2: Handle sort click. Shift+Click adds secondary sort, plain click replaces. Triple-click on same column removes it. */
  const handleSort = useCallback((col: SortColumn, shiftKey: boolean) => {
    setSortLevels(prev => {
      const idx = prev.findIndex(s => s.col === col);
      if (idx >= 0) {
        // Column already sorted: toggle direction, or remove on third click
        const existing = prev[idx];
        if (existing.dir === 'asc') {
          // Remove this sort level
          const next = prev.filter((_, i) => i !== idx);
          return next.length > 0 ? next : [{ col: 'momentum' as SortColumn, dir: 'desc' as SortDir }];
        }
        return prev.map((s, i) => i === idx ? { ...s, dir: 'asc' as SortDir } : s);
      }
      if (shiftKey && prev.length < 3) {
        // Add as secondary/tertiary sort
        return [...prev, { col, dir: 'desc' as SortDir }];
      }
      // Replace all with single primary sort
      return [{ col, dir: 'desc' as SortDir }];
    });
  }, []);

  /** Remove a specific sort level */
  const removeSortLevel = useCallback((col: SortColumn) => {
    setSortLevels(prev => {
      const next = prev.filter(s => s.col !== col);
      return next.length > 0 ? next : [{ col: 'momentum' as SortColumn, dir: 'desc' as SortDir }];
    });
  }, []);

  const toggleSector = (name: string) => {
    setExpandedSectors(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const expandAll = () => setExpandedSectors(new Set(sectors.map(s => s.name)));

  if (isLoading) return <SignalTableSkeleton />;

  if (error) {
    return (
      <div className="p-6">
        <CosmicErrorCard title="Unable to load signals" error={error as Error} onRetry={() => window.location.reload()} />
      </div>
    );
  }
  const collapseAll = () => setExpandedSectors(new Set());

  const openOrStartJob = (mode: 'stocks' | 'retune') => {
    if (isJobRunning) {
      showJobSurface();
      setJobExpanded(true);
    } else {
      startJob(mode);
      showJobSurface();
      setJobExpanded(true);
    }
  };

  return (
    <>
      {/* ── Premium top command center: primary operations are always first. ── */}
      <SignalOperationsBar
        status={jobStatus}
        mode={activeJobMode}
        counters={jobCounters}
        stageMetrics={jobStageMetrics}
        activeStageKey={jobActiveStageKey}
        elapsedSec={jobElapsedSec}
        phaseTitle={jobPhases.length > 0 ? jobPhases[jobPhases.length - 1].title : null}
        filteredRows={filteredRows}
        totalRows={rows.length}
        onRefreshStocks={() => openOrStartJob('stocks')}
        onRunTune={() => openOrStartJob('retune')}
        onViewProgress={() => {
          showJobSurface();
          setJobExpanded(true);
        }}
        onStop={stopJob}
      />

      {/* ── v1 PREMIUM HERO BAND ─────────────────────────────────────── */}
      <SignalsHero stats={stats} rows={rows} horizons={horizons} filteredCount={filteredRows.length} wsStatus={wsStatus} />

      {/* Watchlist — always-visible, user-curated tickers persisted server-side */}
      <div className="mb-5 fade-up-delay-1">
        <WatchlistView
          allRows={sortedRows}
          horizons={horizons}
          updatedAsset={updatedAsset}
          sortLevels={sortLevels}
          onSort={handleSort}
          onRemoveSort={removeSortLevel}
          qualityScores={qualityScores}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
        />
      </div>

      {/* High Conviction Panels — full positions with rich data */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-8 fade-up-delay-1">
        <HighConvictionPanel
          title="High Conviction BUY"
          signals={buyQ.data?.signals || []}
          color="green"
          isLoading={buyQ.isLoading}
          emaStates={emaStates}
        />
        <HighConvictionPanel
          title="High Conviction SELL"
          signals={sellQ.data?.signals || []}
          color="red"
          isLoading={sellQ.isLoading}
          emaStates={emaStates}
        />
      </div>

      {/* SMA Reversals — world-class crossover detection (9 / 50 / 600) */}
      <SmaReversalsPanel
        data={reversalsQ.data}
        isLoading={reversalsQ.isLoading}
        rows={rows}
        onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
      />

      {/* ═══ Premium Filter Bar ══════════════════════════════════════════════
          Apple-grade unified filter surface:
          - Row 1: View segmented + Primary signal segmented + Search
          - Row 2: Signal-strength chips + EMA trend chips + result meta + clear
          ═══════════════════════════════════════════════════════════════════ */}
      <div className="mb-5 fade-up-delay-2 overflow-hidden" style={{
        background: 'linear-gradient(180deg, rgba(255,255,255,0.025) 0%, rgba(255,255,255,0.008) 100%)',
        border: '1px solid rgba(255,255,255,0.06)',
        borderRadius: '16px',
        boxShadow: '0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 28px -14px rgba(0,0,0,0.6), 0 1px 0 rgba(0,0,0,0.4)',
        backdropFilter: 'blur(12px)',
      }}>
        {/* ── Row 1 ─ View segmented │ Primary signal segmented │ Search ── */}
        <div className="flex flex-wrap items-center gap-3 px-4 py-3">
          {/* View segmented control with sliding indicator */}
          <SegmentedControl
            options={[
              { key: 'sectors', label: 'Sectors' },
              { key: 'strong', label: 'Strong' },
              { key: 'all', label: 'All' },
            ] as const}
            value={view}
            onChange={(v) => setView(v as ViewMode)}
            accent="var(--accent-violet)"
            size="md"
          />

          {/* Primary signal segmented */}
          <div className="h-5 w-px bg-white/[0.04]" aria-hidden />
          <SegmentedControl
            options={[
              { key: 'all', label: 'All', dot: undefined },
              { key: 'bullish', label: 'Bullish', dot: '#10b981' },
              { key: 'bearish', label: 'Bearish', dot: '#f43f5e' },
              { key: 'greens', label: 'Greens', dot: '#4ade80' },
              { key: 'reds', label: 'Reds', dot: '#f87171' },
            ] as const}
            value={filter === 'all' || filter === 'bullish' || filter === 'bearish' || filter === 'greens' || filter === 'reds' ? filter : 'all'}
            onChange={(v) => setFilter(v as SignalFilter)}
            accent="#a78bfa"
            size="md"
          />

          <div className="flex-1 min-w-[20px]" />

          {/* Premium Search */}
          <div
            className="flex items-center gap-2 px-3 py-[7px] search-cosmic focus-ring transition-all duration-200"
            style={{
              background: 'rgba(255,255,255,0.02)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '10px',
              backdropFilter: 'blur(6px)',
            }}
          >
            <Search className="w-3.5 h-3.5 text-[var(--text-muted)] group-focus-within:text-[var(--accent-violet)] transition-colors" />
            <input
              ref={searchRef}
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search assets..."
              className="bg-transparent text-[12.5px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none w-40 tabular-nums"
            />
            {!search && (
              <span className="text-[9px] text-[var(--text-muted)] border border-white/[0.08] rounded px-1 py-0.5 opacity-60 font-medium">/</span>
            )}
            {search && debouncedSearch !== search && (
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-violet)] animate-pulse" />
            )}
            {search && (
              <>
                <span className="text-[10px] text-[var(--text-muted)] tabular-nums whitespace-nowrap">
                  {filteredRows.length}/{rows.length}
                </span>
                <button onClick={() => setSearch('')} className="text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors duration-120">
                  <X className="w-3 h-3" />
                </button>
              </>
            )}
          </div>
        </div>

        {/* ── Row 2 ─ Signal chips │ EMA chips │ meta + clear ─────────────── */}
        <div
          className="flex flex-wrap items-center gap-3 px-4 py-2.5"
          style={{ borderTop: '1px solid rgba(255,255,255,0.03)', background: 'rgba(255,255,255,0.008)' }}
        >
          {/* Signal strength chips */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)] pr-1">
              Signal
            </span>
            {([
              { key: 'strong_buy' as SignalFilter, label: 'SB',   full: 'Strong Buy',   accent: '#059669' },
              { key: 'buy'         as SignalFilter, label: 'Buy',  full: 'Buy',          accent: '#34d399' },
              { key: 'hold'        as SignalFilter, label: 'Hold', full: 'Hold',         accent: '#fbbf24' },
              { key: 'sell'        as SignalFilter, label: 'Sell', full: 'Sell',         accent: '#fb7185' },
              { key: 'strong_sell' as SignalFilter, label: 'SS',   full: 'Strong Sell',  accent: '#e11d48' },
            ]).map(({ key, label, full, accent }) => {
              const on = filter === key;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setFilter(on ? 'all' : key)}
                  aria-pressed={on}
                  title={full}
                  className="group relative inline-flex items-center gap-1 rounded-lg px-2 py-1 transition-all duration-200"
                  style={{
                    background: on
                      ? `linear-gradient(180deg, ${accent}30, ${accent}14)`
                      : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${on ? accent + '75' : 'rgba(255,255,255,0.05)'}`,
                    boxShadow: on
                      ? `0 0 0 1px ${accent}25 inset, 0 4px 14px -6px ${accent}90, 0 0 18px -4px ${accent}60`
                      : '0 1px 0 rgba(255,255,255,0.02) inset',
                    color: on ? '#fff' : 'var(--text-secondary)',
                    transition: 'all 220ms cubic-bezier(.2,.8,.2,1)',
                  }}
                >
                  <span
                    aria-hidden
                    className="rounded-full"
                    style={{
                      width: 5, height: 5,
                      background: on ? accent : 'rgba(255,255,255,0.2)',
                      boxShadow: on ? `0 0 6px ${accent}` : 'none',
                      transition: 'background 220ms, box-shadow 220ms',
                    }}
                  />
                  <span className="text-[10.5px] font-semibold tracking-wide" style={{ color: on ? '#fff' : 'var(--text-secondary)' }}>
                    {label}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="h-5 w-px bg-white/[0.05]" aria-hidden />

          {/* EMA trend chips — multi-select */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)] pr-1">
              Trend
            </span>
            {([
              { key: 'p9' as const,   period: 'EMA 9',   count: rows.filter(r => emaLookup(r.asset_label)?.below_9   === true).length },
              { key: 'p50' as const,  period: 'EMA 50',  count: rows.filter(r => emaLookup(r.asset_label)?.below_50  === true).length },
              { key: 'p600' as const, period: 'EMA 600', count: rows.filter(r => emaLookup(r.asset_label)?.below_600 === true).length },
            ]).map(({ key, period, count }) => {
              const on = emaFilters[key];
              const emaLoaded = Object.keys(emaStates).length > 0;
              const accentColor = '#a78bfa';
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setEmaFilters({ ...emaFilters, [key]: !on })}
                  disabled={!emaLoaded}
                  aria-pressed={on}
                  title={emaLoaded ? `Only show assets trading below ${period}` : 'Loading EMA data…'}
                  className="group relative inline-flex items-center gap-1.5 rounded-lg pl-2 pr-1.5 py-1 transition-all duration-200"
                  style={{
                    background: on
                      ? `linear-gradient(180deg, ${accentColor}30, ${accentColor}14)`
                      : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${on ? accentColor + '75' : 'rgba(255,255,255,0.05)'}`,
                    boxShadow: on
                      ? `0 0 0 1px ${accentColor}25 inset, 0 4px 14px -6px ${accentColor}90, 0 0 18px -4px ${accentColor}60`
                      : '0 1px 0 rgba(255,255,255,0.02) inset',
                    color: on ? '#fff' : 'var(--text-secondary)',
                    cursor: emaLoaded ? 'pointer' : 'wait',
                    opacity: emaLoaded ? 1 : 0.5,
                    transition: 'all 220ms cubic-bezier(.2,.8,.2,1)',
                  }}
                >
                  <TrendingDown
                    className="w-3 h-3"
                    style={{
                      color: on ? accentColor : 'var(--text-muted)',
                      filter: on ? `drop-shadow(0 0 4px ${accentColor})` : 'none',
                      transition: 'color 220ms, filter 220ms',
                    }}
                  />
                  <span className="text-[10.5px] font-semibold tracking-wide tabular-nums" style={{ color: on ? '#fff' : 'var(--text-secondary)' }}>
                    {period}
                  </span>
                  <span
                    className="inline-flex items-center justify-center rounded-md px-1 min-w-[18px] h-[15px] text-[9.5px] font-semibold tabular-nums transition-all"
                    style={{
                      background: on ? accentColor : 'rgba(255,255,255,0.05)',
                      color: on ? '#0b0c12' : 'var(--text-muted)',
                      boxShadow: on ? '0 1px 0 rgba(255,255,255,0.2) inset' : 'none',
                    }}
                  >
                    {emaLoaded ? count : '—'}
                  </span>
                </button>
              );
            })}
          </div>

          <div className="flex-1 min-w-[8px]" />

          {/* Result count */}
          <span className="text-[10.5px] text-[var(--text-muted)] tabular-nums">
            <span className="text-[var(--text-primary)] font-medium">{view === 'sectors' ? sectors.length : filteredRows.length}</span>
            {' '}{view === 'sectors' ? 'sectors' : 'results'}
          </span>

          {/* Change counter badge */}
          {changeLog.length > 0 && (
            <button
              onClick={() => {
                const lastChange = changeLog[0];
                if (lastChange) {
                  const el = document.querySelector(`[data-ticker="${lastChange.asset}"]`);
                  el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
                }
              }}
              className="inline-flex items-center gap-1 text-[10px] px-2 py-1 rounded-md animate-pulse"
              style={{ color: 'var(--accent-violet)', background: 'var(--violet-12)', border: '1px solid rgba(167,139,250,0.18)' }}
            >
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-violet)]" />
              {changeLog.length} change{changeLog.length > 1 ? 's' : ''}
            </button>
          )}

          {/* Live Feed toggle */}
          <button
            onClick={() => setShowTickerTape(p => !p)}
            className="inline-flex items-center gap-1 text-[10px] px-2 py-1 rounded-md transition-colors"
            style={{
              color: showTickerTape ? 'var(--accent-violet)' : 'var(--text-muted)',
              background: showTickerTape ? 'var(--violet-12)' : 'rgba(255,255,255,0.02)',
              border: `1px solid ${showTickerTape ? 'rgba(167,139,250,0.22)' : 'rgba(255,255,255,0.05)'}`,
            }}
          >
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{
                background: showTickerTape ? 'var(--accent-violet)' : 'rgba(255,255,255,0.2)',
                boxShadow: showTickerTape ? '0 0 6px var(--accent-violet)' : 'none',
              }}
            />
            Live Feed
          </button>

          {/* Clear all */}
          {(filter !== 'all' || emaFilters.p9 || emaFilters.p50 || emaFilters.p600 || search) && (
            <button
              type="button"
              onClick={() => {
                setFilter('all');
                setEmaFilters({ p9: false, p50: false, p600: false });
                setSearch('');
              }}
              className="inline-flex items-center gap-1 text-[10px] px-2 py-1 rounded-md transition-colors"
              style={{
                color: 'var(--text-secondary)',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid rgba(255,255,255,0.07)',
              }}
              title="Clear all filters"
            >
              <X className="w-3 h-3" />
              Clear
            </button>
          )}
        </div>

        {/* ── Row 3 ─ Sort (sectors only) + Horizons ──
            Labels are tiny uppercase muted, pills themselves carry the active
            violet glow. This lets the entire control surface read as one
            cohesive hierarchy (what → when) with zero dividers fighting the
            content. */}
        {(view === 'sectors' || (view === 'all' && allHorizons.length > 0)) && (
          <div
            className="flex flex-wrap items-center gap-x-3 gap-y-2 px-4 py-2.5"
            style={{ borderTop: '1px solid rgba(255,255,255,0.03)', background: 'rgba(255,255,255,0.006)' }}
          >
            {view === 'sectors' && (
              <>
                <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Sort
                </span>
                <div className="flex items-center gap-1">
                  {SECTOR_SORT_OPTIONS.map(({ key, label, icon }) => {
                    const active = sectorSort === key;
                    return (
                      <button
                        key={key}
                        onClick={() => setSectorSort(key)}
                        className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-[8px] text-[11px] font-medium transition-all duration-[160ms] active:scale-[0.97]"
                        style={active
                          ? {
                              color: 'var(--accent-violet)',
                              background: 'var(--violet-15)',
                              border: '1px solid var(--border-glow)',
                              boxShadow: '0 0 0 1px rgba(167,139,250,0.20), inset 0 1px 0 rgba(255,255,255,0.05)',
                            }
                          : {
                              color: 'var(--text-secondary)',
                              background: 'rgba(255,255,255,0.02)',
                              border: '1px solid rgba(255,255,255,0.05)',
                            }
                        }
                        title={`Sort sectors by ${label}`}
                      >
                        {icon}
                        <span className="hidden sm:inline">{label}</span>
                      </button>
                    );
                  })}
                </div>
                {(view === 'sectors' && allHorizons.length > 0) && <div className="h-4 w-px bg-white/[0.05]" aria-hidden />}
              </>
            )}

            {(view === 'all' || view === 'sectors') && allHorizons.length > 0 && (
              <>
                <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)]">
                  Horizons
                </span>
                <div className="flex flex-wrap items-center gap-1.5">
                  {allHorizons.map(h => {
                    const active = horizons.includes(h);
                    return (
                      <button
                        key={h}
                        onClick={() => toggleHorizon(h)}
                        className="px-2.5 py-1 rounded-[8px] text-[11px] font-medium tabular-nums transition-all duration-[160ms] active:scale-[0.97]"
                        style={active
                          ? {
                              color: 'var(--accent-violet)',
                              background: 'var(--violet-15)',
                              border: '1px solid var(--border-glow)',
                              boxShadow: '0 0 0 1px rgba(167,139,250,0.20), inset 0 1px 0 rgba(255,255,255,0.05)',
                            }
                          : {
                              color: 'var(--text-secondary)',
                              background: 'rgba(255,255,255,0.02)',
                              border: '1px solid rgba(255,255,255,0.05)',
                            }
                        }
                        title={`Toggle ${formatHorizon(h)} column`}
                      >
                        {formatHorizon(h)}
                      </button>
                    );
                  })}
                  {horizonOverride && (
                    <button
                      onClick={resetHorizons}
                      className="text-[10px] text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors ml-0.5"
                      title="Reset horizon selection"
                    >
                      Reset
                    </button>
                  )}
                </div>
              </>
            )}
          </div>
        )}

        {/* ── Footer strip (sectors view only) ─ totals on left, utilities on
            right. Designed as a quiet status bar: monospace numerics, muted
            labels, violet reserved exclusively for active/actionable items.
            This replaces the standalone toolbar that used to live inside
            SectorPanels. */}
        {view === 'sectors' && (
          <div
            className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 py-2"
            style={{ borderTop: '1px solid rgba(255,255,255,0.03)', background: 'rgba(0,0,0,0.12)' }}
          >
            <div className="flex items-center gap-3 text-[10.5px] tabular-nums text-[var(--text-muted)]">
              <span>
                <span className="text-[var(--text-secondary)] font-medium">{sectorTotals.assets}</span> assets
              </span>
              <span className="text-white/20">·</span>
              <span>
                <span className="text-[var(--text-secondary)] font-medium">{sectors.length}</span> sectors
              </span>
              <span className="text-white/20">·</span>
              <span className="inline-flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#10b981' }} />
                <span className="text-[var(--text-muted)]">Bullish</span>
                <span className="text-[#10b981] font-medium">{sectorTotals.bullish}</span>
              </span>
              <span className="text-white/20">·</span>
              <span className="inline-flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: '#f43f5e' }} />
                <span className="text-[var(--text-muted)]">Bearish</span>
                <span className="text-[#f43f5e] font-medium">{sectorTotals.bearish}</span>
              </span>
            </div>

            <div className="flex items-center gap-2 ml-auto">
              <button
                onClick={expandAll}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10.5px] font-medium transition-all duration-[160ms]"
                style={{ color: 'var(--accent-violet)', background: 'rgba(167,139,250,0.08)', border: '1px solid rgba(167,139,250,0.18)' }}
                title="Expand all sectors"
              >
                <ChevronDown className="w-3 h-3" />
                Expand all
              </button>
              <button
                onClick={collapseAll}
                className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[10.5px] font-medium transition-all duration-[160ms]"
                style={{ color: 'var(--text-secondary)', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}
                title="Collapse all sectors"
              >
                <ChevronUp className="w-3 h-3" />
                Collapse all
              </button>
              <div className="h-4 w-px bg-white/[0.05]" aria-hidden />
              <ColumnCustomizer
                columns={SECTOR_COLUMN_DEFS}
                visible={sectorVisibleCols}
                onToggle={toggleSectorCol}
                onReset={resetSectorCols}
              />
            </div>
          </div>
        )}
      </div>

      {/* Story 3.4: Ticker tape */}
      {showTickerTape && changeLog.length > 0 && (
        <div className="h-[28px] overflow-hidden mb-2 glass-card flex items-center" style={{ background: 'var(--void-hover)' }}>
          <div className="ticker-tape-scroll flex items-center gap-6 whitespace-nowrap text-[11px] font-mono">
            {changeLog.slice(0, 5).map((c, i) => {
              const isUpgrade = ['STRONG BUY', 'BUY'].includes(c.to) && ['HOLD', 'SELL', 'STRONG SELL', 'EXIT'].includes(c.from);
              return (
                <span key={`${c.asset}-${i}`} className="signal-entry inline-flex items-center gap-1" style={{ animationDelay: `${i * 50}ms` }}>
                  <span className="text-[var(--accent-violet)]">{c.asset}</span>
                  <span className="text-[var(--text-muted)]">{c.from}</span>
                  <svg width="8" height="8" viewBox="0 0 8 8">
                    <path d="M1 4H7M5 2L7 4L5 6" stroke={isUpgrade ? 'var(--accent-emerald)' : 'var(--accent-rose)'} strokeWidth="1.5" fill="none" />
                  </svg>
                  <span style={{ color: isUpgrade ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>{c.to}</span>
                </span>
              );
            })}
          </div>
        </div>
      )}

      {/* Story 3.4: Away changes banner */}
      {awayChanges.length > 0 && (
        <div className="mb-2 glass-card px-4 py-2 flex items-center gap-3" style={{ background: 'var(--violet-6)' }}>
          <span className="text-[12px] text-[var(--accent-violet)]">{awayChanges.length} signal{awayChanges.length > 1 ? 's' : ''} changed while away</span>
          <button
            onClick={() => { setChangeLog(prev => [...awayChanges, ...prev].slice(0, 20)); setAwayChanges([]); setShowTickerTape(true); }}
            className="text-[11px] px-2 py-0.5 rounded text-[var(--accent-violet)] hover:bg-[var(--violet-10)] transition-colors"
          >
            Review
          </button>
        </div>
      )}

      {/* Horizons + colour filters are now consolidated into the main filter
          bar above. The standalone row that used to live here was removed to
          eliminate duplicated Greens/Reds controls and produce a single,
          cohesive filter surface. */}

      {/* Content */}
      {view === 'sectors' && (
        <SectorPanels
          sectors={sectors}
          expandedSectors={expandedSectors}
          toggleSector={toggleSector}
          sectorSort={sectorSort}
          sectorVisibleCols={sectorVisibleCols}
          horizons={horizons}
          search={debouncedSearch}
          filter={filter}
          updatedAsset={updatedAsset}
          qualityScores={qualityScores}
        />
      )}
      {view === 'strong' && (
        <StrongSignalsView
          strongBuy={(strongQ.data?.strong_buy || []).filter(s => passesEma(s.symbol))}
          strongSell={(strongQ.data?.strong_sell || []).filter(s => passesEma(s.symbol))}
          filter={filter}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
        />
      )}
      {view === 'all' && (
        <AllAssetsTable
          rows={sortedRows} horizons={horizons}
          updatedAsset={updatedAsset}
          sortLevels={sortLevels} onSort={handleSort} onRemoveSort={removeSortLevel}
          expandedRow={expandedRow} onExpandRow={setExpandedRow}
          qualityScores={qualityScores}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
        />
      )}
    </>
  );
}

export default function SignalsPage() {
  return (
    <SignalsErrorBoundary>
      <SignalsPageInner />
    </SignalsErrorBoundary>
  );
}

/* ── Top-of-page Signal Operations Command Center ─────────────────── */
function SignalOperationsBar({
  status,
  mode,
  counters,
  stageMetrics,
  activeStageKey,
  elapsedSec,
  phaseTitle,
  filteredRows,
  totalRows,
  onRefreshStocks,
  onRunTune,
  onViewProgress,
  onStop,
}: {
  status: JobStatus;
  mode: JobMode | null;
  counters: JobCounters;
  stageMetrics: JobStageMetric[];
  activeStageKey: string | null;
  elapsedSec: number;
  phaseTitle: string | null;
  filteredRows: SummaryRow[];
  totalRows: number;
  onRefreshStocks: () => void;
  onRunTune: () => void;
  onViewProgress: () => void;
  onStop: () => void;
}) {
  const isRunning = status === 'running';
  const isStocks = mode === 'stocks';
  const isTune = mode === 'retune' || mode === 'tune' || mode === 'calibrate';
  const activeStage = stageMetrics.find((stage) => stage.key === activeStageKey) ?? stageMetrics.find((stage) => stage.status === 'running') ?? stageMetrics[stageMetrics.length - 1] ?? null;
  const activeCounters = activeStage ? { done: activeStage.done, fail: activeStage.fail, total: activeStage.total } : counters;
  const processed = activeCounters.done + activeCounters.fail;
  const progressPct = activeCounters.total > 0 ? Math.min(100, (processed / activeCounters.total) * 100) : isRunning ? 7 : 0;
  const completionRate = processed > 0 && activeStage?.kind !== 'download' ? Math.round((activeCounters.done / processed) * 100) : null;
  const etaSec = isRunning && processed > 0 && activeCounters.total > processed && activeStage?.kind !== 'download'
    ? Math.max(0, Math.round(((activeCounters.total - processed) * elapsedSec) / processed))
    : null;
  const statusColor = status === 'running' ? '#60a5fa'
    : status === 'completed' ? '#10b981'
      : status === 'failed' || status === 'error' ? '#f43f5e'
        : status === 'stopped' ? '#94a3b8'
          : '#a78bfa';
  const statusLabel = isRunning
    ? mode === 'stocks' ? 'Refreshing market data' : 'Tuning models'
    : status === 'completed' ? 'Last job complete'
      : status === 'stopped' ? 'Stopped'
        : status === 'failed' || status === 'error' ? 'Needs attention'
          : 'Ready';
  const pipelineStages = [
    { key: 'download', label: 'Refresh data', tone: '#60a5fa' },
    { key: 'backup', label: 'Backup cache', tone: '#a78bfa' },
    { key: 'tune', label: 'Tune stocks', tone: '#c084fc' },
    { key: 'calibration', label: 'Calibration', tone: '#10b981' },
  ];
  const activeStageIndex = (() => {
    const title = (phaseTitle ?? '').toLowerCase();
    if (activeStage?.kind) {
      const stageIndex = pipelineStages.findIndex((stage) => stage.key === activeStage.kind);
      if (stageIndex >= 0) return stageIndex + 1;
    }
    if (!isRunning) return status === 'completed' ? pipelineStages.length : 0;
    if (title.includes('refresh') || title.includes('download') || isStocks) return 1;
    if (title.includes('backup')) return 2;
    if (title.includes('calibrat')) return 4;
    if (title.includes('fit') || title.includes('tune') || title.includes('model')) return 3;
    return Math.max(1, Math.min(pipelineStages.length, Math.ceil((progressPct / 100) * pipelineStages.length)));
  })();
  const activeStageLabel = activeStage?.kind === 'download'
    ? 'ready'
    : activeStage?.kind === 'backup'
      ? 'backed up'
      : activeStage?.kind === 'calibration'
        ? 'calibrated'
        : 'processed';
  const runTuneSubtitle = isRunning
    ? isTune ? 'Live fitting in progress' : 'Open the live activity drawer'
    : 'Full BMA retune, streamed live';
  const stocksSubtitle = isRunning
    ? isStocks ? 'Refreshing market data' : 'Open the live activity drawer'
    : 'Prices, cache, and signals';

  return (
    <div className="mb-6 fade-up">
      <div
        className="relative overflow-hidden rounded-[30px] px-4 py-4 md:px-6 md:py-5"
        style={{
          background: 'radial-gradient(900px 280px at 18% -20%, rgba(167,139,250,0.20), transparent 62%), radial-gradient(760px 260px at 92% 118%, rgba(56,217,245,0.13), transparent 62%), linear-gradient(135deg, rgba(25,26,44,0.82), rgba(8,9,18,0.91) 56%, rgba(24,16,42,0.82))',
          border: '1px solid rgba(255,255,255,0.09)',
          boxShadow: '0 34px 96px -58px rgba(139,92,246,0.95), 0 22px 82px -56px rgba(56,217,245,0.55), 0 18px 58px -42px rgba(0,0,0,0.95), inset 0 1px 0 rgba(255,255,255,0.11)',
          backdropFilter: 'blur(24px) saturate(1.35)',
          WebkitBackdropFilter: 'blur(24px) saturate(1.35)',
        }}
      >
        <div aria-hidden className="absolute -left-24 -top-28 h-56 w-56 rounded-full tune-orb-slow" style={{ background: 'radial-gradient(circle, rgba(139,92,246,0.22), rgba(139,92,246,0.05) 42%, transparent 72%)', filter: 'blur(4px)' }} />
        <div aria-hidden className="absolute -right-20 -bottom-28 h-64 w-64 rounded-full tune-orb-slow tune-orb-delay" style={{ background: 'radial-gradient(circle, rgba(56,217,245,0.16), rgba(56,217,245,0.035) 42%, transparent 72%)', filter: 'blur(6px)' }} />
        <div aria-hidden className="absolute inset-x-12 top-0 h-px" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.28), rgba(196,181,253,0.38), transparent)' }} />
        <div aria-hidden className="absolute inset-x-0 bottom-0 h-px" style={{ background: 'linear-gradient(90deg, transparent, rgba(56,217,245,0.18), rgba(139,92,246,0.24), transparent)' }} />

        <div className="relative flex flex-col gap-5 2xl:flex-row 2xl:items-stretch 2xl:justify-between">
          <div className="min-w-0 flex-1">
            <div className="mb-2 flex flex-wrap items-center gap-2">
              <button
                type="button"
                onClick={isRunning ? onViewProgress : undefined}
                className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[10px] font-semibold uppercase tracking-[0.10em] ${isRunning ? 'cursor-pointer hover:brightness-125' : 'cursor-default'}`}
                style={{ color: statusColor, background: `${statusColor}18`, border: `1px solid ${statusColor}36` }}
              >
                <span className={`h-1.5 w-1.5 rounded-full ${isRunning ? 'animate-pulse' : ''}`} style={{ background: statusColor, boxShadow: isRunning ? `0 0 8px ${statusColor}` : undefined }} />
                {statusLabel}
              </button>
              {isRunning && (
                <span className="text-[11px] text-[var(--text-muted)] tabular-nums">
                  {formatJobElapsed(elapsedSec)} · {processed}{activeCounters.total > 0 ? ` / ${activeCounters.total}` : ''} {activeStageLabel}{etaSec !== null ? ` · ETA ${formatJobElapsed(etaSec)}` : ''}
                </span>
              )}
            </div>
            <div className="mt-4 grid max-w-[760px] grid-cols-2 gap-2 sm:grid-cols-4" aria-label="Tune pipeline stages">
              {pipelineStages.map((stage, index) => {
                const stageNumber = index + 1;
                const active = activeStageIndex === stageNumber;
                const done = activeStageIndex > stageNumber;
                return (
                  <div
                    key={stage.label}
                    className="rounded-2xl px-3 py-2 transition-all duration-300"
                    style={{
                      background: done || active ? `${stage.tone}14` : 'rgba(255,255,255,0.025)',
                      border: `1px solid ${done || active ? `${stage.tone}3d` : 'rgba(255,255,255,0.055)'}`,
                      boxShadow: active ? `0 12px 30px -24px ${stage.tone}` : 'inset 0 1px 0 rgba(255,255,255,0.035)',
                    }}
                  >
                    <div className="mb-1 flex items-center gap-1.5">
                      <span
                        className={`h-1.5 w-1.5 rounded-full ${active && isRunning ? 'animate-pulse' : ''}`}
                        style={{ background: done || active ? stage.tone : 'rgba(255,255,255,0.18)', boxShadow: active ? `0 0 8px ${stage.tone}` : undefined }}
                      />
                      <span className="text-[9px] font-semibold uppercase tracking-[0.13em] text-[var(--text-muted)]">Step {stageNumber}</span>
                    </div>
                    <div className="truncate text-[11px] font-semibold tracking-[-0.01em]" style={{ color: done || active ? 'var(--text-primary)' : 'var(--text-secondary)' }}>{stage.label}</div>
                  </div>
                );
              })}
            </div>
            <div className="relative mt-4 h-3 max-w-[760px] overflow-hidden rounded-full p-[1px]" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.18), rgba(255,255,255,0.045))', boxShadow: 'inset 0 1px 2px rgba(0,0,0,0.42), 0 10px 34px -28px rgba(255,255,255,0.75)' }}>
              <div aria-hidden className="absolute inset-0 rounded-full" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.08), transparent 52%, rgba(0,0,0,0.16))' }} />
              <div
                className="relative h-full rounded-full transition-[width] duration-700 ease-out"
                style={{
                  width: `${progressPct}%`,
                  background: isRunning ? 'linear-gradient(90deg,#8b5cf6 0%,#38d9f5 56%,rgba(255,255,255,0.94) 100%)' : 'linear-gradient(90deg,rgba(139,92,246,0.52),rgba(56,217,245,0.26))',
                  boxShadow: isRunning ? '0 0 28px -9px rgba(139,92,246,0.95), inset 0 1px 0 rgba(255,255,255,0.48), inset 0 -1px 0 rgba(0,0,0,0.18)' : 'inset 0 1px 0 rgba(255,255,255,0.24)',
                }}
              >
                <span aria-hidden className="absolute inset-x-1 top-0 h-px rounded-full" style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.88), transparent)' }} />
                <span aria-hidden className="absolute inset-0 rounded-full" style={{ background: 'linear-gradient(180deg, rgba(255,255,255,0.16), transparent 48%)' }} />
                {isRunning && progressPct > 2 && <span aria-hidden className="absolute right-0 top-1/2 h-4 w-4 -translate-y-1/2 translate-x-1/2 rounded-full" style={{ background: '#fff', boxShadow: '0 0 24px 5px rgba(139,92,246,0.58), 0 0 0 1px rgba(255,255,255,0.65) inset' }} />}
              </div>
            </div>
            <div className="mt-3 flex flex-wrap items-center gap-2 text-[11px] text-[var(--text-muted)]">
              <span className="truncate">
                {isRunning ? `${phaseTitle ?? 'Preparing live pipeline…'} · Signals remains usable` : `${filteredRows.length.toLocaleString()} visible signals · ${totalRows.toLocaleString()} total assets`}
              </span>
              {completionRate !== null && isRunning && (
                <span className="rounded-full px-2 py-0.5 tabular-nums" style={{ color: '#a7f3d0', background: 'rgba(16,185,129,0.08)', border: '1px solid rgba(16,185,129,0.18)' }}>{completionRate}% success</span>
              )}
              {activeCounters.fail > 0 && (
                <span className="rounded-full px-2 py-0.5 tabular-nums" style={{ color: '#fb7185', background: 'rgba(244,63,94,0.10)', border: '1px solid rgba(244,63,94,0.22)' }}>{activeCounters.fail} failed</span>
              )}
            </div>
          </div>

          <div className="flex flex-col justify-center gap-3 2xl:w-[520px]">
            <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 2xl:grid-cols-1">
              <OperationButton
                icon={isTune && isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <Play className="h-5 w-5" />}
                title={isRunning ? isTune ? 'Tuning live…' : 'View live activity' : 'Run Tune'}
                subtitle={runTuneSubtitle}
                eyebrow={isRunning && isTune ? 'Streaming now' : 'Recommended'}
                color="#a78bfa"
                active={isTune && isRunning}
                primary
                onClick={onRunTune}
              />
              <OperationButton
                icon={isStocks && isRunning ? <Loader2 className="h-5 w-5 animate-spin" /> : <RefreshCw className="h-5 w-5" />}
                title={isRunning ? isStocks ? 'Refreshing…' : 'View live activity' : 'Refresh Stocks'}
                subtitle={stocksSubtitle}
                eyebrow="Market data"
                color="#60a5fa"
                active={isStocks && isRunning}
                onClick={onRefreshStocks}
              />
            </div>

            <div className="flex flex-wrap items-center justify-end gap-2">
              {isRunning && (
                <button
                  type="button"
                  onClick={onViewProgress}
                  className="inline-flex items-center justify-center gap-2 rounded-full px-3.5 py-2 text-[12px] font-semibold transition-all hover:-translate-y-0.5 active:scale-[0.98]"
                  style={{ color: '#dbeafe', background: 'rgba(96,165,250,0.10)', border: '1px solid rgba(96,165,250,0.26)', boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)' }}
                >
                  <Activity className="h-3.5 w-3.5" />
                  View Live Activity
                </button>
              )}
              {isRunning && (
                <button
                  type="button"
                  onClick={onStop}
                  className="group inline-flex items-center justify-center gap-2 rounded-full px-3.5 py-2 text-[12px] font-semibold text-white transition-all hover:-translate-y-0.5 active:scale-[0.98]"
                  style={{ background: 'linear-gradient(180deg,#fb7185,#e11d48)', boxShadow: '0 16px 34px -22px rgba(244,63,94,0.95)' }}
                >
                  <Square className="h-3.5 w-3.5" fill="currentColor" />
                  Stop
                </button>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function OperationButton({
  icon,
  title,
  subtitle,
  eyebrow,
  color,
  active,
  primary,
  onClick,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
  eyebrow?: string;
  color: string;
  active: boolean;
  primary?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`group relative overflow-hidden rounded-[24px] text-left transition-all duration-300 hover:-translate-y-0.5 active:scale-[0.985] focus-ring ${primary ? 'px-5 py-4' : 'px-4 py-3'}`}
      style={{
        background: primary
          ? `radial-gradient(520px 160px at 18% -18%, ${color}36, transparent 58%), linear-gradient(150deg, ${color}24, rgba(255,255,255,0.055) 54%, rgba(56,217,245,0.075))`
          : active
            ? `linear-gradient(150deg, ${color}20, rgba(255,255,255,0.035))`
            : 'linear-gradient(150deg, rgba(255,255,255,0.052), rgba(255,255,255,0.018))',
        border: `1px solid ${primary ? `${color}70` : active ? `${color}58` : 'rgba(255,255,255,0.085)'}`,
        boxShadow: primary
          ? `0 26px 62px -38px ${color}, 0 0 0 1px ${color}18 inset, inset 0 1px 0 rgba(255,255,255,0.14)`
          : active ? `0 16px 38px -30px ${color}, inset 0 1px 0 rgba(255,255,255,0.12)` : 'inset 0 1px 0 rgba(255,255,255,0.065)',
      }}
    >
      <div aria-hidden className="absolute inset-x-4 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}b8, rgba(255,255,255,0.42), transparent)`, opacity: primary || active ? 1 : 0.45 }} />
      {primary && <div aria-hidden className="absolute -right-16 -top-20 h-36 w-36 rounded-full tune-orb-slow" style={{ background: `radial-gradient(circle, ${color}2f, transparent 70%)`, filter: 'blur(5px)' }} />}
      <div className="relative flex items-center gap-3">
        <span
          className={`inline-flex shrink-0 items-center justify-center rounded-[17px] transition-transform duration-200 group-hover:scale-105 ${primary ? 'h-12 w-12' : 'h-10 w-10'}`}
          style={{ color, background: `${color}1d`, border: `1px solid ${color}3d`, boxShadow: primary || active ? `0 0 26px -8px ${color}` : undefined }}
        >
          {icon}
        </span>
        <span className="min-w-0">
          {eyebrow && <span className="mb-1 block text-[9px] font-semibold uppercase tracking-[0.13em]" style={{ color }}>{eyebrow}</span>}
          <span className={`block font-semibold tracking-[-0.035em] text-white ${primary ? 'text-[17px]' : 'text-[14px]'}`}>{title}</span>
          <span className={`mt-0.5 block truncate text-[var(--text-muted)] ${primary ? 'text-[12px]' : 'text-[11px]'}`}>{subtitle}</span>
        </span>
      </div>
    </button>
  );
}

/* ── v1 Premium Signals Hero Band ─────────────────────────────────── */
function SignalsHero({
  stats, rows, horizons, filteredCount, wsStatus,
}: {
  stats: SignalStats | undefined;
  rows: SummaryRow[];
  horizons: number[];
  filteredCount: number;
  wsStatus: WSStatus;
}) {
  const total = stats?.total_assets ?? rows.length;
  const strongBuy = stats?.strong_buy_signals ?? 0;
  const buy = stats?.buy_signals ?? 0;
  const hold = stats?.hold_signals ?? 0;
  const sell = stats?.sell_signals ?? 0;
  const strongSell = stats?.strong_sell_signals ?? 0;
  const conviction = strongBuy + strongSell;
  const bullishCount = buy; // buy_signals already includes strong_buy in backend convention
  const bearishCount = sell;
  const bullishPct = total > 0 ? (bullishCount / total) * 100 : 0;
  const bearishPct = total > 0 ? (bearishCount / total) * 100 : 0;
  const neutralPct = Math.max(0, 100 - bullishPct - bearishPct);

  const wsColor =
    wsStatus === 'connected' ? '#10b981' :
    wsStatus === 'connecting' ? '#f59e0b' :
    '#64748b';

  return (
    <div
      className="hero-surface fade-up relative overflow-hidden mb-6"
      style={{ borderRadius: 28, padding: '28px 32px' }}
    >
      <div className="flex items-start justify-between gap-8 flex-wrap">
        {/* Left: hero numeral */}
        <div className="flex-1 min-w-[240px]">
          <div className="label-micro mb-2 flex items-center gap-2">
            <span
              className="w-1.5 h-1.5 rounded-full"
              style={{ background: wsColor, boxShadow: wsStatus === 'connected' ? `0 0 8px ${wsColor}` : 'none' }}
            />
            <span>Signals Engine · {wsStatus === 'connected' ? 'LIVE' : wsStatus.toUpperCase()}</span>
          </div>
          <div className="flex items-baseline gap-3">
            <span className="num-hero text-white">{conviction}</span>
            <span className="text-[13px] text-[var(--text-muted)] tabular-nums">
              of {total} · high conviction
            </span>
          </div>
          <div className="mt-1 text-[12px] text-[var(--text-secondary)] tabular-nums">
            {filteredCount === rows.length
              ? <>{horizons.length} horizons active</>
              : <><span className="text-white font-medium">{filteredCount}</span> shown / {rows.length} total</>
            }
          </div>
        </div>

        {/* Right: split bar + stats strip */}
        <div className="flex-1 min-w-[320px] max-w-[560px]">
          <div
            className="flex items-stretch rounded-full overflow-hidden mb-3"
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

/* ── Sector Panels — Premium Redesign ─────────────────────────────── */
type SectorSortBy = 'momentum' | 'exp_ret' | 'signal' | 'count' | 'alpha';
const SECTOR_SORT_OPTIONS: { key: SectorSortBy; label: string; icon: React.ReactNode }[] = [
  { key: 'momentum', label: 'Momentum', icon: <TrendingUp className="w-3 h-3" /> },
  { key: 'signal', label: 'Signal Score', icon: <Target className="w-3 h-3" /> },
  { key: 'count', label: 'Asset Count', icon: <Layers className="w-3 h-3" /> },
  { key: 'alpha', label: 'Alphabetical', icon: <Filter className="w-3 h-3" /> },
];

function signalLabelColor(label: string): string {
  switch (label) {
    case 'STRONG BUY': return '#10b981';
    case 'BUY': return '#6ee7b7';
    case 'HOLD': return '#64748b';
    case 'SELL': return '#fca5a5';
    case 'STRONG SELL': return '#f43f5e';
    default: return '#64748b';
  }
}

const SECTOR_COLUMN_DEFS: ColumnDef[] = [
  { key: 'asset', label: 'Asset', locked: true },
  { key: 'chart', label: 'Chart' },
  { key: 'pct30d', label: '30D change', hint: '%' },
  { key: 'signal', label: 'Signal', locked: true },
  { key: 'strength', label: 'Strength' },
  { key: 'momentum', label: 'Momentum' },
  { key: 'quality', label: 'Quality' },
  { key: 'risk', label: 'Crash risk' },
  { key: 'horizons', label: 'Horizons' },
];
const SECTOR_COLS_LS_KEY = 'signals-sector-cols-v2';
const DEFAULT_SECTOR_VISIBLE_COLS = new Set(SECTOR_COLUMN_DEFS.map((c) => c.key));

function loadSectorVisibleCols(): Set<string> {
  try {
    const raw = localStorage.getItem(SECTOR_COLS_LS_KEY);
    if (!raw) return new Set(DEFAULT_SECTOR_VISIBLE_COLS);
    const parsed = JSON.parse(raw) as string[];
    const set = new Set(parsed);
    SECTOR_COLUMN_DEFS.forEach((c) => { if (c.locked) set.add(c.key); });
    return set;
  } catch {
    return new Set(DEFAULT_SECTOR_VISIBLE_COLS);
  }
}

function SectorPanels({
  sectors,
  expandedSectors,
  toggleSector,
  sectorSort,
  sectorVisibleCols,
  horizons,
  search,
  filter,
  updatedAsset,
  qualityScores,
}: {
  sectors: SectorGroup[];
  expandedSectors: Set<string>;
  toggleSector: (name: string) => void;
  sectorSort: SectorSortBy;
  sectorVisibleCols: Set<string>;
  horizons: number[];
  search: string;
  filter: SignalFilter;
  updatedAsset: string | null;
  qualityScores: Record<string, number>;
}) {
  const navigate = useNavigate();

  /** Per-row sort within each sector (Task 2). Single-column toggle. */
  type RowSortCol = 'asset' | 'signal' | 'strength' | 'momentum' | 'quality' | 'risk' | `horizon_${number}`;
  const [rowSortCol, setRowSortCol] = useState<RowSortCol>('momentum');
  const [rowSortDir, setRowSortDir] = useState<'asc' | 'desc'>('desc');
  const onRowSort = (col: RowSortCol) => {
    if (rowSortCol === col) {
      setRowSortDir(d => (d === 'asc' ? 'desc' : 'asc'));
    } else {
      setRowSortCol(col);
      setRowSortDir(col === 'asset' ? 'asc' : 'desc');
    }
  };
  const applyRowSort = (arr: SummaryRow[]) => {
    const signalRank: Record<string, number> = { 'STRONG BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG SELL': 1, 'EXIT': 0 };
    const getter = (r: SummaryRow): number | string => {
      const col = rowSortCol;
      if (col === 'asset') return r.asset_label;
      if (col === 'signal') return signalRank[(r.nearest_label || 'HOLD').toUpperCase()] ?? 3;
      if (col === 'strength') return r.horizon_signals[Number(Object.keys(r.horizon_signals)[0])]?.p_up ?? 0.5;
      if (col === 'momentum') return r.momentum_score ?? 0;
      if (col === 'quality') return qualityScores[extractTicker(r.asset_label)] ?? 50;
      if (col === 'risk') return r.crash_risk_score ?? 0;
      if (col.startsWith('horizon_')) {
        const h = parseInt(col.split('_')[1], 10);
        return r.horizon_signals[h]?.exp_ret ?? 0;
      }
      return 0;
    };
    const mult = rowSortDir === 'asc' ? 1 : -1;
    return [...arr].sort((a, b) => {
      const av = getter(a);
      const bv = getter(b);
      if (typeof av === 'string' && typeof bv === 'string') return av.localeCompare(bv) * mult;
      return (((av as number) - (bv as number)) || 0) * mult;
    });
  };
  const sortArrow = (col: RowSortCol) =>
    rowSortCol === col ? (rowSortDir === 'asc' ? ' \u2191' : ' \u2193') : '';
  const thSortClass = (col: RowSortCol) =>
    `cursor-pointer select-none hover:text-[var(--text-secondary)] transition-colors ${rowSortCol === col ? 'text-[var(--accent-violet)]' : ''}`;


  const sorted = useMemo(() => {
    const arr = [...sectors];
    const signalScore = (s: SectorGroup) => (s.strong_buy ?? 0) * 3 + (s.buy ?? 0) * 2 - (s.sell ?? 0) * 2 - (s.strong_sell ?? 0) * 3;
    switch (sectorSort) {
      case 'momentum': return arr.sort((a, b) => (b.avg_momentum ?? 0) - (a.avg_momentum ?? 0));
      case 'signal': return arr.sort((a, b) => signalScore(b) - signalScore(a));
      case 'count': return arr.sort((a, b) => b.asset_count - a.asset_count);
      case 'alpha': return arr.sort((a, b) => a.name.localeCompare(b.name));
      case 'exp_ret': return arr.sort((a, b) => signalScore(b) - signalScore(a));
      default: return arr;
    }
  }, [sectors, sectorSort]);

  return (
    <div className="space-y-3">
      {/* Sort / Expand / Columns / totals all moved to the unified premium
          filter card in SignalsPageInner. SectorPanels now renders sector
          content only. */}

      {sorted.map((sector) => {
        const expanded = expandedSectors.has(sector.name);
const matchesFilter = (lbl: string, row: SummaryRow) => {
      if (filter === 'all') return true;
      if (filter === 'bullish') return lbl === 'STRONG_BUY' || lbl === 'BUY';
      if (filter === 'bearish') return lbl === 'STRONG_SELL' || lbl === 'SELL';
      if (filter === 'greens' || filter === 'reds') return rowHorizonColor(row) === filter;
      return lbl === filter.toUpperCase();
    };
    const assets = sector.assets.filter(row => {
      if (search && !row.asset_label.toLowerCase().includes(search.toLowerCase())) return false;
      const lbl = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
      return matchesFilter(lbl, row);
    });
    if (assets.length === 0) return null;

        const bullish = (sector.strong_buy ?? 0) + (sector.buy ?? 0);
        const bearish = (sector.strong_sell ?? 0) + (sector.sell ?? 0);
        const neutral = sector.hold ?? 0;
        const total = bullish + bearish + neutral;
        const sentiment = bullish > bearish ? 'bullish' : bearish > bullish ? 'bearish' : 'neutral';
        const sentColor = sentiment === 'bullish' ? '#10b981' : sentiment === 'bearish' ? '#f43f5e' : '#64748b';
        const sentGlow = sentiment === 'bullish' ? '0 0 20px rgba(16,185,129,0.08)' : sentiment === 'bearish' ? '0 0 20px rgba(244,63,94,0.08)' : 'none';

        // Best performing asset
        const bestAsset = [...sector.assets].sort((a, b) => {
          const aRet = Object.values(a.horizon_signals)[0]?.exp_ret ?? 0;
          const bRet = Object.values(b.horizon_signals)[0]?.exp_ret ?? 0;
          return bRet - aRet;
        })[0];
        const bestTicker = bestAsset ? extractTicker(bestAsset.asset_label) : null;
        const bestRet = bestAsset ? (Object.values(bestAsset.horizon_signals)[0]?.exp_ret ?? 0) * 100 : 0;
        const bestLabel = bestAsset ? (bestAsset.nearest_label || 'HOLD').toUpperCase() : '';

        // Sentiment bar proportions
        const strongBuyPct = total > 0 ? ((sector.strong_buy ?? 0) / total) * 100 : 0;
        const buyPct = total > 0 ? ((sector.buy ?? 0) / total) * 100 : 0;
        const holdPct = total > 0 ? ((sector.hold ?? 0) / total) * 100 : 0;
        const sellPct = total > 0 ? ((sector.sell ?? 0) / total) * 100 : 0;
        const strongSellPct = total > 0 ? ((sector.strong_sell ?? 0) / total) * 100 : 0;

        const avgMom = sector.avg_momentum ?? 0;
        const bullishPct = total > 0 ? Math.round((bullish / total) * 100) : 0;

        return (
          <div key={sector.name} className="glass-card overflow-hidden transition-all duration-200"
            style={{
              borderLeft: `3px solid ${sentColor}40`,
              boxShadow: expanded ? sentGlow : 'none',
            }}>
            {/* Sector Header — rich, informative */}
            <button
              onClick={() => toggleSector(sector.name)}
              className="w-full px-4 py-3 hover:bg-white/[0.015] transition-all duration-200 group"
            >
              {/* Top row: Name + key stats */}
              <div className="flex items-center gap-3">
                {/* Expand indicator */}
                <div className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 transition-all duration-200"
                  style={{ background: expanded ? `${sentColor}20` : 'var(--void-active)' }}>
                  <ChevronRight
                    className="w-3 h-3 transition-transform duration-200"
                    style={{ color: expanded ? sentColor : 'var(--text-muted)', transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                  />
                </div>

                {/* Sector name */}
                <span className="font-semibold text-[13px] text-[#e2e8f0] whitespace-nowrap group-hover:text-white transition-colors">{sector.name}</span>

                {/* Asset count */}
                <span className="text-[10px] px-2 py-0.5 rounded-full font-medium tabular-nums"
                  style={{ background: `${sentColor}12`, color: sentColor }}>
                  {sector.asset_count}
                </span>

                {/* Sentiment bar — wider, more readable */}
                <div className="flex h-[5px] w-[100px] rounded-full overflow-hidden flex-shrink-0" style={{ background: 'var(--void-active)' }}>
                  <div className="transition-all duration-500" style={{ width: `${strongBuyPct}%`, background: '#10b981' }} />
                  <div className="transition-all duration-500" style={{ width: `${buyPct}%`, background: '#6ee7b7' }} />
                  <div className="transition-all duration-500" style={{ width: `${holdPct}%`, background: '#475569' }} />
                  <div className="transition-all duration-500" style={{ width: `${sellPct}%`, background: '#fca5a5' }} />
                  <div className="transition-all duration-500" style={{ width: `${strongSellPct}%`, background: '#f43f5e' }} />
                </div>

                {/* Bullish % */}
                <span className="text-[10px] font-bold tabular-nums" style={{ color: sentColor }}>
                  {bullishPct}%
                </span>

                {/* Signal counts — compact badges */}
                <div className="hidden md:flex items-center gap-1">
                  {(sector.strong_buy ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#10b98118', color: '#10b981' }}>
                      SB {sector.strong_buy}
                    </span>
                  )}
                  {(sector.buy ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#6ee7b718', color: '#6ee7b7' }}>
                      B {sector.buy}
                    </span>
                  )}
                  {neutral > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#47556918', color: '#64748b' }}>
                      H {neutral}
                    </span>
                  )}
                  {(sector.sell ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#fca5a518', color: '#fca5a5' }}>
                      S {sector.sell}
                    </span>
                  )}
                  {(sector.strong_sell ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#f43f5e18', color: '#f43f5e' }}>
                      SS {sector.strong_sell}
                    </span>
                  )}
                </div>

                {/* Spacer */}
                <div className="flex-1" />

                {/* Momentum */}
                <div className="flex items-center gap-1">
                  {avgMom > 0 ? (
                    <ArrowUp className="w-3 h-3 text-[var(--accent-emerald)]" />
                  ) : avgMom < 0 ? (
                    <ArrowDown className="w-3 h-3 text-[var(--accent-rose)]" />
                  ) : null}
                  <span className="text-[11px] font-bold font-mono tabular-nums"
                    style={{ color: avgMom > 0 ? '#10b981' : avgMom < 0 ? '#f43f5e' : '#64748b' }}>
                    {avgMom > 0 ? '+' : ''}{avgMom.toFixed(1)}%
                  </span>
                </div>

                {/* Best asset peek */}
                {!expanded && bestTicker && (
                  <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-lg" style={{ background: 'var(--void-active)' }}>
                    <span className="text-[9px] text-[var(--text-muted)]">Top</span>
                    <span className="text-[10px] font-bold text-[var(--accent-violet)]">{bestTicker}</span>
                    <span className="text-[10px] font-bold tabular-nums" style={{ color: bestRet >= 0 ? '#10b981' : '#f43f5e' }}>
                      {bestRet >= 0 ? '+' : ''}{bestRet.toFixed(1)}%
                    </span>
                    <span className="text-[8px] px-1 py-0.5 rounded font-semibold"
                      style={{ background: `${signalLabelColor(bestLabel)}18`, color: signalLabelColor(bestLabel) }}>
                      {bestLabel}
                    </span>
                  </div>
                )}
              </div>
            </button>

            {/* Expanded content — premium table */}
            {expanded && (
              <div
                style={{
                  animation: 'sectorReveal 220ms cubic-bezier(0.2,0,0,1) both',
                }}
              >
                <style>{`
                  @keyframes sectorReveal {
                    from { opacity: 0; transform: translateY(-4px); }
                    to   { opacity: 1; transform: translateY(0); }
                  }
                  @media (prefers-reduced-motion: reduce) {
                    [style*="sectorReveal"] { animation: none !important; }
                  }
                `}</style>
                {/* Sector summary strip */}
                <div className="flex items-center gap-4 px-5 py-2 text-[10px]"
                  style={{ background: `${sentColor}06`, borderTop: `1px solid ${sentColor}15`, borderBottom: '1px solid var(--border-void)' }}>
                  <div className="flex items-center gap-3">
                    <span className="text-[var(--text-muted)]">Breakdown:</span>
                    {[
                      { label: 'Strong Buy', count: sector.strong_buy ?? 0, color: '#10b981' },
                      { label: 'Buy', count: sector.buy ?? 0, color: '#6ee7b7' },
                      { label: 'Hold', count: sector.hold ?? 0, color: '#64748b' },
                      { label: 'Sell', count: sector.sell ?? 0, color: '#fca5a5' },
                      { label: 'Strong Sell', count: sector.strong_sell ?? 0, color: '#f43f5e' },
                    ].filter(x => x.count > 0).map(({ label, count, color: c }) => (
                      <span key={label} className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
                        <span style={{ color: c }} className="font-medium">{count}</span>
                        <span className="text-[var(--text-muted)]">{label}</span>
                      </span>
                    ))}
                  </div>
                  <div className="ml-auto flex items-center gap-1.5 text-[var(--text-muted)]">
                    <Activity className="w-3 h-3" />
                    <span>Avg Risk: </span>
                    <span className="font-bold tabular-nums" style={{
                      color: (sector.avg_crash_risk ?? 0) > 60 ? '#f43f5e' : (sector.avg_crash_risk ?? 0) > 30 ? '#f59e0b' : '#10b981'
                    }}>
                      {(sector.avg_crash_risk ?? 0).toFixed(0)}
                    </span>
                  </div>
                </div>

                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ background: 'var(--void-hover)' }}>
                        <th onClick={() => onRowSort('asset')} className={`text-left px-3 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[160px] ${thSortClass('asset')}`}>Asset{sortArrow('asset')}</th>
                        {sectorVisibleCols.has('chart') && (
                          <th className="text-center px-2 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[124px]">Chart</th>
                        )}
                        {sectorVisibleCols.has('pct30d') && (
                          <th className="text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px]">30D</th>
                        )}
                        <th onClick={() => onRowSort('signal')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[88px] ${thSortClass('signal')}`}>Signal{sortArrow('signal')}</th>
                        {sectorVisibleCols.has('strength') && (
                          <th onClick={() => onRowSort('strength')} className={`text-center px-1 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[64px] ${thSortClass('strength')}`}>Strength{sortArrow('strength')}</th>
                        )}
                        {sectorVisibleCols.has('momentum') && (
                          <th onClick={() => onRowSort('momentum')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass('momentum')}`}>Mom{sortArrow('momentum')}</th>
                        )}
                        {sectorVisibleCols.has('quality') && (
                          <th onClick={() => onRowSort('quality')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass('quality')}`}>Quality{sortArrow('quality')}</th>
                        )}
                        {sectorVisibleCols.has('horizons') && horizons.map(h => {
                          const col = `horizon_${h}` as RowSortCol;
                          return (
                            <th key={h} onClick={() => onRowSort(col)} className={`text-center px-1 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass(col)}`}>{formatHorizon(h)}{sortArrow(col)}</th>
                          );
                        })}
                        {sectorVisibleCols.has('risk') && (
                          <th onClick={() => onRowSort('risk')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass('risk')}`}>Risk{sortArrow('risk')}</th>
                        )}
                        <th className="w-6"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {applyRowSort(assets).map((row, i) => (
                        <SectorSignalRow
                          key={row.asset_label}
                          row={row}
                          horizons={horizons}
                          visibleCols={sectorVisibleCols}
                          qualityScore={qualityScores[extractTicker(row.asset_label)] ?? 50}
                          highlighted={row.asset_label === updatedAsset}
                          delayMs={i * 30}
                          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
                {assets.length === 0 && (
                  <div className="px-5 py-6 text-center">
                    <Shield className="w-5 h-5 mx-auto mb-1.5" style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
                    <p className="text-[11px] text-[var(--text-muted)]">No assets match current filter</p>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

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

function StrongSignalsView({ strongBuy, strongSell, filter, onNavigateChart }: {
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

/* ── Watchlist View — user-curated tickers with full detail ─────── */
function WatchlistView({
  allRows,
  horizons,
  updatedAsset,
  sortLevels,
  onSort,
  onRemoveSort,
  qualityScores,
  onNavigateChart,
}: {
  allRows: SummaryRow[];
  horizons: number[];
  updatedAsset: string | null;
  sortLevels: { col: SortColumn; dir: SortDir }[];
  onSort: (col: SortColumn, shift: boolean) => void;
  onRemoveSort: (col: SortColumn) => void;
  qualityScores: Record<string, number>;
  onNavigateChart: (sym: string) => void;
}) {
  const { symbols, proxyMap, isLoading, add, remove } = useWatchlist();
  const [input, setInput] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Build a set of tickers for O(1) lookup. Also build a ticker → row map so
  // we can tell which watchlist symbols are present in the current signal set.
  // Some watchlist tickers are proxied under the hood (e.g. DFNG → ITA,
  // XAGUSD=X → SI=F, GLDE → GLD), so a signal row labelled with the primary
  // ticker must still count as "found" for the proxied watchlist entry. We
  // build a symbol → [acceptable tickers] map using the server-provided
  // PROXY_OVERRIDES, and match a row against any of them.
  const { watchlistRows, missingSymbols } = useMemo(() => {
    const acceptableBySymbol = new Map<string, Set<string>>();
    const accepted = new Set<string>();
    for (const s of symbols) {
      const targets = new Set<string>([s]);
      const primary = proxyMap[s];
      if (primary) targets.add(primary);
      acceptableBySymbol.set(s, targets);
      for (const t of targets) accepted.add(t);
    }
    const rowsForWatchlist: SummaryRow[] = [];
    const foundRowTickers = new Set<string>();
    for (const r of allRows) {
      const ticker = extractTicker(r.asset_label);
      if (accepted.has(ticker)) {
        rowsForWatchlist.push(r);
        foundRowTickers.add(ticker);
      }
    }
    const missing = symbols.filter((s) => {
      const targets = acceptableBySymbol.get(s);
      if (!targets) return true;
      for (const t of targets) {
        if (foundRowTickers.has(t)) return false;
      }
      return true;
    });
    return { watchlistRows: rowsForWatchlist, missingSymbols: missing };
  }, [allRows, symbols, proxyMap]);

  // ── Watchlist-local filters (no pagination — user disliked paging) ───
  // Signal class: STRONG_BUY/BUY/SELL/STRONG_SELL/HOLD via nearest_label.
  // Horizon colour: whole-row majority of exp_ret signs via rowHorizonColor —
  // a ticker can be "bullish" today (label) yet "reds" across horizons, so
  // both axes are useful filters.
  type WlClass = 'bullish' | 'bearish' | 'neutral';
  type WlColor = 'greens' | 'reds' | 'mixed';
  type WlSignal = 'all' | WlClass | WlColor;
  type WlSort = 'signal' | 'momentum' | 'risk' | 'alpha';
  const [wlSignal, setWlSignal] = useState<WlSignal>('all');
  const [wlSector, setWlSector] = useState<string>('all');
  const [wlQuery, setWlQuery] = useState<string>('');
  const [wlSort, setWlSort] = useState<WlSort>('signal');
  const [wlMissingOnly, setWlMissingOnly] = useState<boolean>(false);
  // Manage drawer is HIDDEN by default once the user has a populated list —
  // it auto-opens only when the watchlist is empty (first-run experience)
  // or when the user explicitly clicks the "+ Add" button. This keeps the
  // panel's resting state clean and content-first (see Watchlist.md).
  const [manageOpen, setManageOpen] = useState<boolean>(false);
  // Chip color mode in the Manage Drawer.
  //   'signal'   — nearest-label bullish/bearish (green/red).
  //   'horizon'  — whole-row greens vs reds verdict.
  //   'bigmoves' — only large 1w movers get color; small ones are muted grey
  //                so the eye locks onto Big Greens / Big Reds instantly.
  const [chipColorMode, setChipColorMode] = useState<'signal' | 'horizon' | 'bigmoves'>('signal');
  // Refine popover — search / sector / sort are collapsed behind a single
  // trigger so the two primary rows (insight bar, segmented control) can
  // breathe.
  const [refineOpen, setRefineOpen] = useState<boolean>(false);
  // When the user clicks a column header inside the watchlist table, we
  // let the parent's sortLevels drive the row order (allRows arrive
  // pre-sorted). The wlSort dropdown becomes a no-op until the user picks
  // a preset again, at which point the override is cleared. This resolves
  // the historic "sorting doesn't work" bug where column clicks had no
  // visible effect on the watchlist table (see Watchlist.md §10).
  const [wlSortOverride, setWlSortOverride] = useState<boolean>(false);
  useEffect(() => {
    if (symbols.length === 0) setManageOpen(true);
  }, [symbols.length]);

  const sectorOptions = useMemo(() => {
    const s = new Set<string>();
    for (const r of watchlistRows) {
      if (r.sector) s.add(r.sector);
    }
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [watchlistRows]);

  const classifyRow = useCallback((row: SummaryRow): WlClass => {
    const lbl = (row.nearest_label || '').toUpperCase();
    if (lbl === 'STRONG_BUY' || lbl === 'BUY') return 'bullish';
    if (lbl === 'STRONG_SELL' || lbl === 'SELL') return 'bearish';
    return 'neutral';
  }, []);

  // Ticker (as it appears in signal rows, accounting for proxies) → row, so
  // chip rendering can look up the signal bucket for coloring.
  const rowByTicker = useMemo(() => {
    const m = new Map<string, SummaryRow>();
    for (const r of watchlistRows) {
      m.set(extractTicker(r.asset_label), r);
    }
    return m;
  }, [watchlistRows]);

  // Resolve a watchlist symbol to its signal row (via proxy if needed).
  const rowForSymbol = useCallback(
    (sym: string): SummaryRow | undefined => {
      const direct = rowByTicker.get(sym);
      if (direct) return direct;
      const primary = proxyMap[sym];
      if (primary) return rowByTicker.get(primary);
      return undefined;
    },
    [rowByTicker, proxyMap],
  );

  // 1-week implied return (exp_ret at horizon=7) per watchlist symbol.
  // Used to scale chip color intensity — stronger moves render with deeper
  // saturation so the eye can rank conviction at a glance.
  const exp1w = useCallback(
    (sym: string): number => {
      const row = rowForSymbol(sym);
      if (!row) return 0;
      const sigs = row.horizon_signals as Record<string | number, { exp_ret?: number } | undefined>;
      const sig = sigs?.[7] || sigs?.['7'];
      const r = sig?.exp_ret;
      return Number.isFinite(r) ? (r as number) : 0;
    },
    [rowForSymbol],
  );

  // Max |1w exp_ret| across the current watchlist — normalises intensity
  // so the strongest mover anchors full saturation. Floor at 0.5% to avoid
  // hyper-amplifying tiny moves on a quiet day.
  const maxAbs1w = useMemo(() => {
    let m = 0.005;
    for (const sym of symbols) {
      const v = Math.abs(exp1w(sym));
      if (v > m) m = v;
    }
    return m;
  }, [symbols, exp1w]);

  const signalCounts = useMemo(() => {
    let bull = 0, bear = 0, neut = 0, green = 0, red = 0, mixed = 0;
    for (const r of watchlistRows) {
      const c = classifyRow(r);
      if (c === 'bullish') bull++;
      else if (c === 'bearish') bear++;
      else neut++;
      const hc = rowHorizonColor(r);
      if (hc === 'greens') green++;
      else if (hc === 'reds') red++;
      else mixed++;
    }
    return { bull, bear, neut, green, red, mixed };
  }, [watchlistRows, classifyRow]);

  const filteredWatchlistRows = useMemo(() => {
    if (wlMissingOnly) return [];
    const q = wlQuery.trim().toLowerCase();
    const rows = watchlistRows.filter((r) => {
      if (wlSignal !== 'all') {
        if (wlSignal === 'bullish' || wlSignal === 'bearish' || wlSignal === 'neutral') {
          if (classifyRow(r) !== wlSignal) return false;
        } else {
          if (rowHorizonColor(r) !== wlSignal) return false;
        }
      }
      if (wlSector !== 'all' && r.sector !== wlSector) return false;
      if (q && !(r.asset_label || '').toLowerCase().includes(q)) return false;
      return true;
    });
    // When the user has clicked a column header (sort override), we want the
    // parent's sortLevels to win — and the parent already handed us rows in
    // sortLevels order. Since `watchlistRows` is derived from `allRows` with
    // `filter` (order-preserving), we can simply skip the preset sort.
    if (wlSortOverride) return rows;
    const sorted = rows.slice();
    const signalRank = (r: SummaryRow) => {
      const lbl = (r.nearest_label || '').toUpperCase();
      if (lbl === 'STRONG_BUY') return 0;
      if (lbl === 'BUY') return 1;
      if (lbl === 'HOLD' || lbl === 'EXIT' || !lbl) return 2;
      if (lbl === 'SELL') return 3;
      if (lbl === 'STRONG_SELL') return 4;
      return 2;
    };
    if (wlSort === 'signal') {
      sorted.sort((a, b) => signalRank(a) - signalRank(b) || (a.asset_label || '').localeCompare(b.asset_label || ''));
    } else if (wlSort === 'momentum') {
      sorted.sort((a, b) => (b.momentum_score ?? -Infinity) - (a.momentum_score ?? -Infinity));
    } else if (wlSort === 'risk') {
      sorted.sort((a, b) => (a.crash_risk_score ?? Infinity) - (b.crash_risk_score ?? Infinity));
    } else {
      sorted.sort((a, b) => (a.asset_label || '').localeCompare(b.asset_label || ''));
    }
    return sorted;
  }, [watchlistRows, wlSignal, wlSector, wlQuery, wlSort, wlMissingOnly, wlSortOverride, classifyRow]);

  // Wrap the parent's `onSort` so a column-header click inside the watchlist
  // table also flips the override flag. The parent handler still runs so the
  // main Signals table stays in sync — this matches the "most-recent intent
  // wins" contract documented in Watchlist.md §10.
  const handleWatchlistSort = useCallback(
    (col: SortColumn, shift: boolean) => {
      setWlSortOverride(true);
      onSort(col, shift);
    },
    [onSort],
  );

  // Changing the Sort preset is an explicit reset of the override.
  const setWlSortPreset = useCallback((key: WlSort) => {
    setWlSort(key);
    setWlSortOverride(false);
  }, []);

  // Suggested tickers for the first-run empty state. Keeping this tiny and
  // opinionated: one mega-cap tech, one AI darling, one benchmark ETF.
  const suggestedTickers = useMemo(() => ['AAPL', 'NVDA', 'SPY'], []);

  // Keyboard shortcuts. `/` focuses the ticker input (also opens the manage
  // drawer), `A` toggles the drawer, `Esc` closes it when open. We attach to
  // `window` but ignore when the user is typing into a text field.
  useEffect(() => {
    const isTypingTarget = (t: EventTarget | null) => {
      const el = t as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
    };
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && manageOpen) {
        setManageOpen(false);
        return;
      }
      if (isTypingTarget(e.target)) return;
      if (e.key === '/') {
        e.preventDefault();
        setManageOpen(true);
        setTimeout(() => inputRef.current?.focus(), 60);
      } else if (e.key === 'a' || e.key === 'A') {
        setManageOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [manageOpen]);

  const hasActiveFilter = wlSignal !== 'all' || wlSector !== 'all' || wlQuery.trim().length > 0 || wlSort !== 'signal' || wlMissingOnly || wlSortOverride;
  const hasRefinement = wlSector !== 'all' || wlQuery.trim().length > 0 || wlSort !== 'signal';
  const clearFilters = useCallback(() => {
    setWlSignal('all');
    setWlSector('all');
    setWlQuery('');
    setWlSort('signal');
    setWlMissingOnly(false);
    setWlSortOverride(false);
  }, []);

  const submit = useCallback(() => {
    const sym = input.trim().toUpperCase();
    if (!sym) return;
    add.mutate(sym, {
      onSuccess: () => setInput(''),
    });
  }, [input, add]);

  const onKey = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submit();
      }
    },
    [submit],
  );

  const addErrorMsg = add.error?.message;

  return (
    <div className="flex flex-col gap-3 fade-up-delay-3">
      {/* ─────────────────────────────────────────────────────────────────
         TIER 1 — Insight Bar
         A single headline line. Anchors the panel. Clickable count phrases
         filter the list; "+ Add" toggles the slide-up manage drawer.
         See Watchlist.md §3.
         ───────────────────────────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center gap-2 px-4 py-3"
        style={{
          background:
            'linear-gradient(180deg, rgba(255,255,255,0.028) 0%, rgba(255,255,255,0.008) 100%)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: '16px',
          boxShadow:
            '0 1px 0 rgba(255,255,255,0.05) inset, 0 10px 32px -18px rgba(0,0,0,0.55)',
          backdropFilter: 'blur(14px)',
        }}
      >
        {/* Anchor: Star + label + tracked count */}
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-[10px] flex items-center justify-center"
            style={{
              background:
                'linear-gradient(180deg, rgba(167,139,250,0.18) 0%, rgba(167,139,250,0.08) 100%)',
              color: '#c4b5fd',
              boxShadow:
                '0 0 0 1px rgba(167,139,250,0.22), 0 6px 18px -10px rgba(167,139,250,0.55)',
            }}
          >
            <Star className="w-4 h-4" />
          </div>
          <div className="leading-tight">
            <div className="text-[13px] font-semibold text-[#e2e8f0] tracking-[-0.01em]">
              Watchlist
            </div>
            <div className="text-[11px] text-[var(--text-secondary)] tabular-nums">
              {symbols.length} tracked
              {watchlistRows.length !== symbols.length && (
                <> · {watchlistRows.length} live</>
              )}
            </div>
          </div>
        </div>

        {/* Headline verdict: bullish · greens · missing. Each is a chip
            that filters the list when clicked (one-click quick filter). */}
        {symbols.length > 0 && (
          <div
            className="flex flex-wrap items-center gap-1.5 ml-1 md:ml-3"
            aria-label="Watchlist summary"
          >
            {([
              {
                key: 'bullish' as const,
                value: signalCounts.bull,
                label: 'bullish',
                accent: '#34d399',
                tint: 'rgba(52,211,153,0.10)',
                border: 'rgba(52,211,153,0.26)',
                onClick: () => {
                  setWlMissingOnly(false);
                  setWlSignal(wlSignal === 'bullish' ? 'all' : 'bullish');
                },
                active: wlSignal === 'bullish',
              },
              {
                key: 'greens' as const,
                value: signalCounts.green,
                label: 'all greens',
                accent: '#6ee7b7',
                tint: 'rgba(110,231,183,0.10)',
                border: 'rgba(110,231,183,0.26)',
                onClick: () => {
                  setWlMissingOnly(false);
                  setWlSignal(wlSignal === 'greens' ? 'all' : 'greens');
                },
                active: wlSignal === 'greens',
              },
              ...(signalCounts.bear > 0
                ? [{
                    key: 'bearish' as const,
                    value: signalCounts.bear,
                    label: 'bearish',
                    accent: '#f87171',
                    tint: 'rgba(248,113,113,0.10)',
                    border: 'rgba(248,113,113,0.26)',
                    onClick: () => {
                      setWlMissingOnly(false);
                      setWlSignal(wlSignal === 'bearish' ? 'all' : 'bearish');
                    },
                    active: wlSignal === 'bearish',
                  }]
                : []),
              ...(missingSymbols.length > 0
                ? [{
                    key: 'missing' as const,
                    value: missingSymbols.length,
                    label: 'missing',
                    accent: '#fcd34d',
                    tint: 'rgba(251,191,36,0.10)',
                    border: 'rgba(251,191,36,0.26)',
                    onClick: () => {
                      const next = !wlMissingOnly;
                      setWlMissingOnly(next);
                      if (next) setWlSignal('all');
                    },
                    active: wlMissingOnly,
                  }]
                : []),
            ]).map((phrase, idx, arr) => (
              <Fragment key={phrase.key}>
                <button
                  type="button"
                  onClick={phrase.onClick}
                  className="group inline-flex items-baseline gap-1 px-2 py-0.5 rounded-md transition-all duration-[140ms] active:scale-[0.97]"
                  style={{
                    background: phrase.active ? phrase.tint : 'transparent',
                    boxShadow: phrase.active ? `inset 0 0 0 1px ${phrase.border}` : 'none',
                  }}
                  title={phrase.active ? `Clear ${phrase.label} filter` : `Show ${phrase.label}`}
                  aria-pressed={phrase.active}
                >
                  <span
                    className="text-[15px] font-semibold tabular-nums leading-none transition-colors"
                    style={{ color: phrase.active ? phrase.accent : '#e2e8f0' }}
                  >
                    {phrase.value}
                  </span>
                  <span
                    className="text-[12px] transition-colors"
                    style={{
                      color: phrase.active
                        ? phrase.accent
                        : 'var(--text-secondary)',
                    }}
                  >
                    {phrase.label}
                  </span>
                </button>
                {idx < arr.length - 1 && (
                  <span className="text-[var(--text-secondary)] opacity-40 text-[13px] leading-none">·</span>
                )}
              </Fragment>
            ))}
          </div>
        )}

        {/* Right cluster: Refine + Add */}
        <div className="ml-auto flex items-center gap-1.5">
          {symbols.length > 0 && (
            <button
              type="button"
              onClick={() => setRefineOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 px-2.5 py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-[140ms] active:scale-[0.97]"
              style={{
                background: refineOpen || hasRefinement ? 'rgba(255,255,255,0.05)' : 'transparent',
                border: `1px solid ${refineOpen || hasRefinement ? 'rgba(255,255,255,0.10)' : 'rgba(255,255,255,0.04)'}`,
                color: hasRefinement ? '#c4b5fd' : 'var(--text-secondary)',
              }}
              title={refineOpen ? 'Hide refine controls' : 'Search, sort, filter by sector'}
              aria-expanded={refineOpen}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Refine
              {hasRefinement && (
                <span
                  className="inline-block rounded-full"
                  style={{ width: 5, height: 5, background: '#a78bfa' }}
                />
              )}
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              setManageOpen((v) => !v);
              if (!manageOpen) setTimeout(() => inputRef.current?.focus(), 80);
            }}
            className="inline-flex items-center gap-1.5 px-3 py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-[140ms] active:scale-[0.97] hover:-translate-y-[1px]"
            style={{
              background:
                'linear-gradient(180deg, rgba(167,139,250,0.20) 0%, rgba(167,139,250,0.10) 100%)',
              color: '#e9d5ff',
              border: '1px solid rgba(167,139,250,0.30)',
              boxShadow:
                '0 1px 0 rgba(255,255,255,0.08) inset, 0 6px 18px -10px rgba(167,139,250,0.55)',
            }}
            title={manageOpen ? 'Close manage drawer (A)' : 'Add / manage tickers (A)'}
            aria-expanded={manageOpen}
          >
            {manageOpen ? (
              <X className="w-3.5 h-3.5" />
            ) : (
              <Plus className="w-3.5 h-3.5" />
            )}
            {manageOpen ? 'Close' : 'Add'}
          </button>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────
         MANAGE DRAWER — slides open when manageOpen.
         Uses a CSS grid-rows trick for a smooth height transition.
         ───────────────────────────────────────────────────────────────── */}
      <div
        className="grid transition-[grid-template-rows] duration-[260ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ gridTemplateRows: manageOpen ? '1fr' : '0fr' }}
        aria-hidden={!manageOpen}
      >
        <div className="overflow-hidden">
          <div
            className="p-4"
            style={{
              background:
                'linear-gradient(180deg, rgba(255,255,255,0.022) 0%, rgba(255,255,255,0.006) 100%)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '16px',
              boxShadow:
                '0 1px 0 rgba(255,255,255,0.05) inset, 0 8px 28px -14px rgba(0,0,0,0.6)',
              backdropFilter: 'blur(12px)',
            }}
          >
            {/* Input row */}
            <div className="flex items-center gap-2">
              <div
                className="flex items-center gap-2 flex-1 px-3 py-[8px] focus-ring transition-all duration-200"
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '10px',
                }}
              >
                <Search className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value.toUpperCase())}
                  onKeyDown={onKey}
                  placeholder="Add ticker (AAPL, BTC-USD, EURUSD=X)…"
                  spellCheck={false}
                  autoCapitalize="characters"
                  autoCorrect="off"
                  className="flex-1 bg-transparent outline-none text-sm text-[#e2e8f0] placeholder:text-[var(--text-secondary)]"
                  disabled={add.isPending}
                />
                {input && (
                  <button
                    type="button"
                    onClick={() => setInput('')}
                    className="text-[var(--text-secondary)] hover:text-[#e2e8f0]"
                    title="Clear"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <button
                type="button"
                onClick={submit}
                disabled={!input.trim() || add.isPending}
                className="inline-flex items-center gap-1.5 px-3 py-[8px] rounded-[10px] text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed active:scale-[0.97]"
                style={{
                  background:
                    'linear-gradient(180deg, rgba(167,139,250,0.22) 0%, rgba(167,139,250,0.10) 100%)',
                  color: '#e9d5ff',
                  border: '1px solid rgba(167,139,250,0.30)',
                }}
              >
                {add.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Plus className="w-3.5 h-3.5" />
                )}
                Track
              </button>
            </div>
            {addErrorMsg && (
              <div
                className="mt-2 flex items-center gap-1.5 text-[11px]"
                style={{ color: '#fca5a5' }}
              >
                <AlertTriangle className="w-3 h-3" />
                {addErrorMsg}
              </div>
            )}

            {/* Suggestions when empty */}
            {symbols.length === 0 && (
              <div className="mt-3">
                <div className="text-[10px] uppercase tracking-[0.12em] font-semibold text-[var(--text-secondary)] mb-2">
                  Try one of these
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {suggestedTickers.map((sym) => (
                    <button
                      key={sym}
                      type="button"
                      onClick={() => add.mutate(sym)}
                      disabled={add.isPending}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-medium transition-all active:scale-[0.97] hover:-translate-y-[1px] disabled:opacity-40"
                      style={{
                        background: 'rgba(167,139,250,0.08)',
                        color: '#c4b5fd',
                        border: '1px solid rgba(167,139,250,0.22)',
                      }}
                      title={`Track ${sym}`}
                    >
                      <Plus className="w-3 h-3" />
                      {sym}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Chips grid — color-coded by current signal state OR by
                whole-row horizon verdict (greens vs reds) depending on
                `chipColorMode`. A small segmented toggle lets the user flip
                between the two lenses. */}
            {symbols.length > 0 && (
              <>
                <div className="mt-3 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-[0.12em] font-semibold text-[var(--text-secondary)]">
                    Color by
                  </span>
                  <div
                    className="inline-flex items-center rounded-[10px] p-0.5 gap-0.5"
                    style={{
                      background: 'rgba(255,255,255,0.025)',
                      border: '1px solid rgba(255,255,255,0.05)',
                    }}
                    role="tablist"
                    aria-label="Chip color mode"
                  >
                    {([
                      { k: 'signal' as const, label: 'Signal', accent: '#a78bfa', bg: 'rgba(167,139,250,0.14)', border: 'rgba(167,139,250,0.28)' },
                      { k: 'horizon' as const, label: 'Greens / Reds', accent: '#6ee7b7', bg: 'rgba(110,231,183,0.14)', border: 'rgba(110,231,183,0.28)' },
                      { k: 'bigmoves' as const, label: 'Big Greens / Big Reds', accent: '#fde68a', bg: 'rgba(253,230,138,0.14)', border: 'rgba(253,230,138,0.32)' },
                    ]).map((opt) => {
                      const active = chipColorMode === opt.k;
                      return (
                        <button
                          key={opt.k}
                          type="button"
                          role="tab"
                          aria-selected={active}
                          onClick={() => setChipColorMode(opt.k)}
                          className="inline-flex items-center gap-1.5 px-2.5 py-[4px] rounded-[8px] text-[11px] font-medium transition-all duration-[140ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97]"
                          style={{
                            background: active ? opt.bg : 'transparent',
                            color: active ? opt.accent : 'var(--text-secondary)',
                            boxShadow: active ? `0 0 0 1px ${opt.border}` : 'none',
                          }}
                          title={
                            opt.k === 'signal'
                              ? 'Color chips by nearest-horizon signal (bullish / bearish)'
                              : opt.k === 'horizon'
                              ? 'Color chips by whole-row verdict (all greens vs all reds)'
                              : 'Highlight only big 1-week movers (≥1.5%); small ones go neutral grey'
                          }
                        >
                          {opt.k === 'horizon' && (
                            <span className="inline-flex items-center gap-0.5">
                              <span className="inline-block rounded-full" style={{ width: 6, height: 6, background: '#34d399' }} />
                              <span className="inline-block rounded-full" style={{ width: 6, height: 6, background: '#ef4444' }} />
                            </span>
                          )}
                          {opt.k === 'bigmoves' && (
                            <span className="inline-flex items-center gap-0.5">
                              <span className="inline-block rounded-full" style={{ width: 8, height: 8, background: '#10b981', boxShadow: '0 0 0 1.5px rgba(16,185,129,0.45)' }} />
                              <span className="inline-block rounded-full" style={{ width: 8, height: 8, background: '#ef4444', boxShadow: '0 0 0 1.5px rgba(239,68,68,0.45)' }} />
                            </span>
                          )}
                          {opt.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {symbols.map((sym) => {
                    const isMissing = missingSymbols.includes(sym);
                    const row = isMissing ? undefined : rowForSymbol(sym);
                    // 1w intensity in [0..1] — |exp_ret_7d| normalised by the
                    // strongest mover. sqrt curve amplifies mid-range so the
                    // difference between a 0.5% and 2% mover is unmistakable.
                    const r7d = isMissing ? 0 : exp1w(sym);
                    const raw = Math.min(1, Math.abs(r7d) / maxAbs1w);
                    const intensity = Math.max(0.08, Math.sqrt(raw));
                    // Tier the chip into 4 visual buckets so adjacent chips
                    // are obviously distinct, not a smooth alpha gradient.
                    const tier = intensity < 0.30 ? 0 : intensity < 0.55 ? 1 : intensity < 0.80 ? 2 : 3;
                    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
                    let bg = 'rgba(148,163,184,0.06)';
                    let color = '#cbd5e1';
                    let border = 'rgba(148,163,184,0.20)';
                    let dot: string | null = null;
                    let chipTitleSuffix = '';
                    let fontWeight: 400 | 500 | 600 | 700 = 500;
                    if (isMissing) {
                      bg = 'rgba(251,191,36,0.08)';
                      color = '#fcd34d';
                      border = 'rgba(251,191,36,0.26)';
                    } else if (row && chipColorMode === 'bigmoves') {
                      // Big Greens / Big Reds lens — ignore the row verdict;
                      // gate purely on |1w move|. Below 1.5% → neutral grey.
                      // 1.5–3% → medium tint. ≥3% → vivid, bold, glowing.
                      const absR = Math.abs(r7d);
                      const sign = r7d >= 0 ? 1 : -1;
                      let bigTier: 0 | 1 | 2 = 0;
                      if (absR >= 0.03) bigTier = 2;
                      else if (absR >= 0.015) bigTier = 1;
                      if (bigTier === 0) {
                        bg = 'rgba(148,163,184,0.05)';
                        color = '#64748b';
                        border = 'rgba(148,163,184,0.16)';
                        dot = null;
                        fontWeight = 400;
                        chipTitleSuffix = ` — 1w ${(r7d * 100).toFixed(2)}% (small)`;
                      } else if (sign > 0) {
                        const bgA = bigTier === 2 ? 0.55 : 0.28;
                        const brA = bigTier === 2 ? 0.95 : 0.55;
                        bg = `rgba(16,185,129,${bgA})`;
                        color = bigTier === 2 ? '#d1fae5' : '#34d399';
                        border = `rgba(16,185,129,${brA})`;
                        dot = '#10b981';
                        fontWeight = bigTier === 2 ? 700 : 600;
                        chipTitleSuffix = ` — BIG green 1w +${(r7d * 100).toFixed(2)}%`;
                      } else {
                        const bgA = bigTier === 2 ? 0.55 : 0.28;
                        const brA = bigTier === 2 ? 0.95 : 0.55;
                        bg = `rgba(239,68,68,${bgA})`;
                        color = bigTier === 2 ? '#fee2e2' : '#f87171';
                        border = `rgba(239,68,68,${brA})`;
                        dot = '#ef4444';
                        fontWeight = bigTier === 2 ? 700 : 600;
                        chipTitleSuffix = ` — BIG red 1w ${(r7d * 100).toFixed(2)}%`;
                      }
                    } else if (row && chipColorMode === 'horizon') {
                      // Greens vs Reds lens — 4 tiers, wide alpha range so
                      // strongest movers are clearly separated from the rest.
                      const hc = rowHorizonColor(row);
                      const bgA = lerp(0.06, 0.55, intensity);
                      const brA = lerp(0.20, 0.95, intensity);
                      fontWeight = tier >= 2 ? 700 : tier === 1 ? 600 : 500;
                      if (hc === 'greens') {
                        bg = `rgba(16,185,129,${bgA.toFixed(3)})`;
                        // Step palette: tier 0 → pale, tier 3 → vivid white-green.
                        color = ['#86efac', '#6ee7b7', '#34d399', '#a7f3d0'][tier];
                        border = `rgba(16,185,129,${brA.toFixed(3)})`;
                        dot = '#10b981';
                        chipTitleSuffix = ` — all greens (1w +${(Math.abs(r7d) * 100).toFixed(2)}%)`;
                      } else if (hc === 'reds') {
                        // Pure red, not orange.
                        bg = `rgba(239,68,68,${bgA.toFixed(3)})`;
                        color = ['#fca5a5', '#f87171', '#ef4444', '#fecaca'][tier];
                        border = `rgba(239,68,68,${brA.toFixed(3)})`;
                        dot = '#ef4444';
                        chipTitleSuffix = ` — all reds (1w −${(Math.abs(r7d) * 100).toFixed(2)}%)`;
                      } else {
                        bg = 'rgba(148,163,184,0.06)';
                        color = '#94a3b8';
                        border = 'rgba(148,163,184,0.20)';
                        dot = null;
                        chipTitleSuffix = ' — mixed';
                      }
                    } else if (row) {
                      const cls = classifyRow(row);
                      // Bullish/bearish lens — same 4-tier scheme.
                      const bgA = lerp(0.04, 0.42, intensity);
                      const brA = lerp(0.16, 0.80, intensity);
                      fontWeight = tier >= 2 ? 700 : tier === 1 ? 600 : 500;
                      if (cls === 'bullish') {
                        bg = `rgba(52,211,153,${bgA.toFixed(3)})`;
                        color = ['#a7f3d0', '#6ee7b7', '#34d399', '#d1fae5'][tier];
                        border = `rgba(52,211,153,${brA.toFixed(3)})`;
                        dot = '#34d399';
                        chipTitleSuffix = ` — 1w +${(r7d * 100).toFixed(2)}%`;
                      } else if (cls === 'bearish') {
                        bg = `rgba(239,68,68,${bgA.toFixed(3)})`;
                        color = ['#fecaca', '#fca5a5', '#f87171', '#fee2e2'][tier];
                        border = `rgba(239,68,68,${brA.toFixed(3)})`;
                        dot = '#ef4444';
                        chipTitleSuffix = ` — 1w ${(r7d * 100).toFixed(2)}%`;
                      }
                    }
                    return (
                      <span
                        key={sym}
                        className="group inline-flex items-center gap-1 pl-2 pr-1 py-1 rounded-md text-[11px] tabular-nums transition-all duration-[140ms] hover:-translate-y-[1px]"
                        style={{ background: bg, color, border: `1px solid ${border}`, fontWeight }}
                        title={
                          isMissing
                            ? `${sym} — not in current signal set`
                            : row
                            ? `${sym} — ${row.nearest_label || 'HOLD'}${chipTitleSuffix}`
                            : sym
                        }
                      >
                        {isMissing ? (
                          <AlertTriangle className="w-3 h-3" />
                        ) : dot ? (
                          <span
                            className="inline-block rounded-full"
                            style={{
                              width: 6,
                              height: 6,
                              background: dot,
                              // Halo grows with 1w intensity — subtle for weak
                              // signals, prominent glow for strongest movers.
                              boxShadow: `0 0 0 2px ${dot}${Math.round(lerp(0x14, 0x55, intensity)).toString(16).padStart(2, '0')}`,
                            }}
                          />
                        ) : null}
                        <span>{sym}</span>
                        <button
                          type="button"
                          onClick={() => remove.mutate(sym)}
                          disabled={remove.isPending}
                          className="ml-0.5 w-4 h-4 inline-flex items-center justify-center rounded opacity-60 hover:opacity-100 hover:bg-white/5 disabled:opacity-30"
                          title={`Remove ${sym}`}
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </span>
                    );
                  })}
                </div>
              </>
            )}

            {/* Keyboard shortcut hint row */}
            <div className="mt-3 pt-3 border-t border-white/[0.04] flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[var(--text-secondary)]">
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">/</kbd>
                focus input
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">A</kbd>
                toggle drawer
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">Esc</kbd>
                close
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">Enter</kbd>
                track ticker
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Table / empty state ─────────────────────────────────────── */}
      {isLoading ? (
        <div className="text-center py-8 text-sm text-[var(--text-secondary)]">
          Loading watchlist…
        </div>
      ) : symbols.length === 0 ? (
        <div
          className="relative overflow-hidden flex flex-col items-center justify-center text-center py-16 px-6"
          style={{
            background:
              'linear-gradient(180deg, rgba(167,139,250,0.04) 0%, rgba(167,139,250,0.01) 60%, rgba(255,255,255,0.005) 100%)',
            border: '1px dashed rgba(167,139,250,0.18)',
            borderRadius: '20px',
          }}
        >
          <div
            aria-hidden
            className="absolute inset-0 pointer-events-none"
            style={{
              background:
                'radial-gradient(420px 180px at 50% 0%, rgba(167,139,250,0.10), transparent 70%)',
            }}
          />
          <div
            className="relative w-16 h-16 rounded-[22px] flex items-center justify-center mb-5"
            style={{
              background:
                'linear-gradient(180deg, rgba(167,139,250,0.22) 0%, rgba(167,139,250,0.08) 100%)',
              color: '#c4b5fd',
              boxShadow:
                '0 0 0 1px rgba(167,139,250,0.26), 0 12px 36px -12px rgba(167,139,250,0.55), inset 0 1px 0 rgba(255,255,255,0.08)',
            }}
          >
            <Star className="w-7 h-7" />
          </div>
          <h3 className="relative text-[17px] font-semibold text-[#e2e8f0] mb-1 tracking-[-0.01em]">
            Build your watchlist
          </h3>
          <p className="relative text-[13px] text-[var(--text-secondary)] max-w-sm mb-5 leading-relaxed">
            Pin tickers you care about for a focused view. Signals, momentum, and risk
            — just for them.
          </p>
          <div className="relative flex flex-wrap items-center justify-center gap-1.5 mb-5">
            {suggestedTickers.map((sym) => (
              <button
                key={sym}
                type="button"
                onClick={() => add.mutate(sym)}
                disabled={add.isPending}
                className="inline-flex items-center gap-1 px-3 py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-[140ms] active:scale-[0.97] hover:-translate-y-[1px] disabled:opacity-40"
                style={{
                  background: 'rgba(167,139,250,0.10)',
                  color: '#c4b5fd',
                  border: '1px solid rgba(167,139,250,0.26)',
                }}
                title={`Track ${sym}`}
              >
                <Plus className="w-3.5 h-3.5" />
                {sym}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => { setManageOpen(true); setTimeout(() => inputRef.current?.focus(), 80); }}
            className="relative inline-flex items-center gap-1.5 px-3 py-[7px] rounded-[10px] text-[12px] font-medium transition-all active:scale-[0.97]"
            style={{
              background: 'transparent',
              color: 'var(--text-secondary)',
              border: '1px solid rgba(255,255,255,0.08)',
            }}
          >
            <Search className="w-3.5 h-3.5" />
            Or add your own
          </button>
        </div>
      ) : watchlistRows.length === 0 ? (
        <div
          className="flex flex-col items-center justify-center text-center py-12 px-6"
          style={{
            background:
              'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%)',
            border: '1px dashed rgba(251,191,36,0.18)',
            borderRadius: '16px',
          }}
        >
          <AlertTriangle className="w-8 h-8 mb-3" style={{ color: '#fcd34d' }} />
          <h3 className="text-base font-semibold text-[#e2e8f0] mb-1">
            No live signals for your watchlist
          </h3>
          <p className="text-sm text-[var(--text-secondary)] max-w-md">
            None of your saved tickers are in the current signal set. They are
            still persisted — add them to the engine or refresh signals to see
            them here.
          </p>
        </div>
      ) : (
        <>
          {/* ── Unified segmented control ──────────────────────────────
             One row. Six segments. No duplicate groups. Missing is a
             segment here (only when missingSymbols.length > 0). Color
             accent per Watchlist.md §8. Sticky at top of the list for
             glanceable context as user scrolls. */}
          <div
            className="flex flex-wrap items-center gap-2 px-3 py-2 sticky top-0 z-10"
            style={{
              background:
                'linear-gradient(180deg, rgba(15,23,42,0.72) 0%, rgba(15,23,42,0.55) 100%)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '14px',
              boxShadow:
                '0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 28px -18px rgba(0,0,0,0.6)',
              backdropFilter: 'blur(14px)',
            }}
            role="tablist"
            aria-label="Watchlist view"
          >
            <div
              className="inline-flex items-center rounded-[11px] p-0.5 gap-0.5"
              style={{
                background: 'rgba(255,255,255,0.025)',
                border: '1px solid rgba(255,255,255,0.05)',
              }}
            >
              {([
                { k: 'all' as const, label: 'All', count: watchlistRows.length, accent: '#e2e8f0', bg: 'rgba(226,232,240,0.10)', border: 'rgba(226,232,240,0.20)', dot: null as string | null },
                { k: 'bullish' as const, label: 'Bullish', count: signalCounts.bull, accent: '#34d399', bg: 'rgba(52,211,153,0.12)', border: 'rgba(52,211,153,0.28)', dot: '#34d399' },
                { k: 'bearish' as const, label: 'Bearish', count: signalCounts.bear, accent: '#f87171', bg: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.28)', dot: '#f87171' },
                { k: 'greens' as const, label: 'Greens', count: signalCounts.green, accent: '#6ee7b7', bg: 'rgba(110,231,183,0.12)', border: 'rgba(110,231,183,0.28)', dot: '#6ee7b7' },
                { k: 'reds' as const, label: 'Reds', count: signalCounts.red, accent: '#fca5a5', bg: 'rgba(252,165,165,0.12)', border: 'rgba(252,165,165,0.28)', dot: '#fca5a5' },
              ]).map((seg) => {
                const active = !wlMissingOnly && wlSignal === seg.k;
                return (
                  <button
                    key={seg.k}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => {
                      setWlMissingOnly(false);
                      setWlSignal(seg.k as WlSignal);
                    }}
                    className="inline-flex items-center gap-1.5 px-2.5 py-[5px] rounded-[9px] text-[11px] font-medium transition-all duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97]"
                    style={{
                      background: active ? seg.bg : 'transparent',
                      color: active ? seg.accent : 'var(--text-secondary)',
                      boxShadow: active ? `0 0 0 1px ${seg.border}` : 'none',
                    }}
                    title={`${seg.label} (${seg.count})`}
                  >
                    {seg.dot && (
                      <span
                        className="inline-block rounded-full transition-opacity"
                        style={{
                          width: 6,
                          height: 6,
                          background: seg.dot,
                          opacity: active ? 1 : 0.55,
                        }}
                      />
                    )}
                    {seg.label}
                    <span
                      className="tabular-nums transition-opacity"
                      style={{ opacity: active ? 0.9 : 0.55 }}
                    >
                      {seg.count}
                    </span>
                  </button>
                );
              })}
              {missingSymbols.length > 0 && (
                <button
                  type="button"
                  role="tab"
                  aria-selected={wlMissingOnly}
                  onClick={() => {
                    const next = !wlMissingOnly;
                    setWlMissingOnly(next);
                    if (next) setWlSignal('all');
                  }}
                  className="inline-flex items-center gap-1.5 px-2.5 py-[5px] rounded-[9px] text-[11px] font-medium transition-all duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97]"
                  style={{
                    background: wlMissingOnly ? 'rgba(251,191,36,0.14)' : 'transparent',
                    color: wlMissingOnly ? '#fcd34d' : 'var(--text-secondary)',
                    boxShadow: wlMissingOnly ? '0 0 0 1px rgba(251,191,36,0.30)' : 'none',
                  }}
                  title={`Missing (${missingSymbols.length})`}
                >
                  <AlertTriangle className="w-3 h-3" />
                  Missing
                  <span
                    className="tabular-nums transition-opacity"
                    style={{ opacity: wlMissingOnly ? 0.9 : 0.55 }}
                  >
                    {missingSymbols.length}
                  </span>
                </button>
              )}
            </div>

            {/* Sorted-by indicator (shown only when column-click override
                is active, i.e. user clicked a column header). */}
            {wlSortOverride && sortLevels.length > 0 && (
              <div
                className="inline-flex items-center gap-1 px-2 py-[3px] rounded-[8px] text-[10px]"
                style={{
                  background: 'rgba(167,139,250,0.08)',
                  color: '#c4b5fd',
                  border: '1px solid rgba(167,139,250,0.22)',
                }}
                title="Column-click sort active. Switch the sort dropdown in Refine to reset."
              >
                <span className="opacity-80">sorted by column</span>
                <button
                  type="button"
                  onClick={() => { setWlSortOverride(false); }}
                  className="opacity-70 hover:opacity-100 -mr-0.5"
                  title="Reset to preset sort"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {/* Right cluster: result count + clear */}
            <div className="ml-auto flex items-center gap-1.5">
              {hasActiveFilter && (
                <button
                  type="button"
                  onClick={clearFilters}
                  className="inline-flex items-center gap-1 px-2 py-[4px] rounded-[8px] text-[10px] font-medium transition-all active:scale-[0.97]"
                  style={{
                    background: 'transparent',
                    border: '1px solid rgba(255,255,255,0.06)',
                    color: 'var(--text-secondary)',
                  }}
                  title="Reset all watchlist filters"
                >
                  <X className="w-3 h-3" />
                  Clear
                </button>
              )}
              <div
                className="text-[11px] tabular-nums"
                style={{ color: 'var(--text-secondary)' }}
                title={`${wlMissingOnly ? missingSymbols.length : filteredWatchlistRows.length} shown of ${watchlistRows.length} with live signals`}
              >
                {wlMissingOnly ? missingSymbols.length : filteredWatchlistRows.length}
                <span className="opacity-50"> / {watchlistRows.length}</span>
              </div>
            </div>
          </div>

          {/* ── Refine popover ─────────────────────────────────────────
             Collapsed by default. Opens from the "Refine" button in the
             Insight Bar. Holds search, sector, and preset sort. Uses the
             same grid-template-rows trick as the Manage drawer. */}
          <div
            className="grid transition-[grid-template-rows] duration-[260ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
            style={{ gridTemplateRows: refineOpen ? '1fr' : '0fr' }}
            aria-hidden={!refineOpen}
          >
            <div className="overflow-hidden">
              <div
                className="flex flex-wrap items-center gap-2 px-3 py-2 mt-1"
                style={{
                  background: 'rgba(255,255,255,0.015)',
                  border: '1px solid rgba(255,255,255,0.05)',
                  borderRadius: '12px',
                }}
              >
                <div
                  className="flex items-center gap-1.5 px-2 py-[5px] flex-1 min-w-[180px]"
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: '10px',
                  }}
                >
                  <Search className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                  <input
                    type="text"
                    value={wlQuery}
                    onChange={(e) => setWlQuery(e.target.value)}
                    placeholder="Filter by name or ticker…"
                    spellCheck={false}
                    className="flex-1 bg-transparent outline-none text-[12px] text-[#e2e8f0] placeholder:text-[var(--text-secondary)]"
                  />
                  {wlQuery && (
                    <button
                      type="button"
                      onClick={() => setWlQuery('')}
                      className="text-[var(--text-secondary)] hover:text-[#e2e8f0]"
                      title="Clear search"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>
                {sectorOptions.length > 0 && (
                  <select
                    value={wlSector}
                    onChange={(e) => setWlSector(e.target.value)}
                    className="text-[11px] px-2 py-[5px] rounded-[10px] outline-none"
                    style={{
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      color: wlSector === 'all' ? 'var(--text-secondary)' : '#e2e8f0',
                    }}
                    title="Filter by sector"
                  >
                    <option value="all">All sectors</option>
                    {sectorOptions.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                )}
                <select
                  value={wlSort}
                  onChange={(e) => setWlSortPreset(e.target.value as WlSort)}
                  className="text-[11px] px-2 py-[5px] rounded-[10px] outline-none"
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    color: '#e2e8f0',
                  }}
                  title="Preset sort order (column click overrides)"
                >
                  <option value="signal">Sort: Signal (best first)</option>
                  <option value="momentum">Sort: Momentum</option>
                  <option value="risk">Sort: Risk (low first)</option>
                  <option value="alpha">Sort: Alphabetical</option>
                </select>
              </div>
            </div>
          </div>

          {wlMissingOnly ? (
            <div
              className="flex flex-col gap-3 px-4 py-4"
              style={{
                background:
                  'linear-gradient(180deg, rgba(251,191,36,0.05) 0%, rgba(251,191,36,0.01) 100%)',
                border: '1px dashed rgba(251,191,36,0.22)',
                borderRadius: '14px',
              }}
            >
              <div className="flex items-center gap-2 text-[13px] font-medium" style={{ color: '#fcd34d' }}>
                <AlertTriangle className="w-4 h-4" />
                Missing from current signal set ({missingSymbols.length})
              </div>
              {missingSymbols.length === 0 ? (
                <div className="text-[11px] text-[var(--text-secondary)]">
                  All watchlist tickers have a live signal row.
                </div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {missingSymbols.map((sym) => (
                    <span
                      key={sym}
                      className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium tabular-nums"
                      style={{
                        background: 'rgba(251,191,36,0.08)',
                        color: '#fcd34d',
                        border: '1px solid rgba(251,191,36,0.22)',
                      }}
                      title={`${sym} — not tuned or not in latest signal snapshot`}
                    >
                      <AlertTriangle className="w-3 h-3" />
                      {sym}
                      <button
                        type="button"
                        onClick={() => remove.mutate(sym)}
                        disabled={remove.isPending}
                        className="ml-0.5 opacity-70 hover:opacity-100 disabled:opacity-30"
                        title={`Remove ${sym}`}
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
              <div className="text-[11px] text-[var(--text-secondary)]">
                These tickers exist in your watchlist but no row was emitted for
                them. That usually means they haven't been tuned yet or the
                latest snapshot is still refreshing.
              </div>
            </div>
          ) : filteredWatchlistRows.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center text-center py-10 px-6"
              style={{
                background:
                  'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%)',
                border: '1px dashed rgba(255,255,255,0.08)',
                borderRadius: '14px',
              }}
            >
              <Filter className="w-6 h-6 mb-2" style={{ color: 'var(--text-secondary)' }} />
              <div className="text-sm font-medium text-[#e2e8f0]">No matches</div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-1">
                No watchlist entries match your current filters.
              </div>
              <button
                type="button"
                onClick={clearFilters}
                className="mt-3 inline-flex items-center gap-1 px-2.5 py-1 rounded-[10px] text-[11px] font-medium"
                style={{
                  background: 'var(--violet-15)',
                  color: '#c4b5fd',
                  border: '1px solid rgba(167,139,250,0.25)',
                }}
              >
                <X className="w-3 h-3" />
                Clear filters
              </button>
            </div>
          ) : (
            <AllAssetsTable
              rows={filteredWatchlistRows}
              horizons={horizons}
              updatedAsset={updatedAsset}
              sortLevels={sortLevels}
              onSort={handleWatchlistSort}
              onRemoveSort={onRemoveSort}
              expandedRow={expanded}
              onExpandRow={setExpanded}
              qualityScores={qualityScores}
              onNavigateChart={onNavigateChart}
              disablePagination
            />
          )}
        </>
      )}
    </div>
  );
}

/* ── Story 3.2: Sort Indicator with priority badge ───────────────── */
function SortIndicator({ col, sortLevels }: { col: SortColumn; sortLevels: { col: SortColumn; dir: SortDir }[] }) {
  const idx = sortLevels.findIndex(s => s.col === col);
  if (idx < 0) {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" className="inline ml-0.5 opacity-0 group-hover:opacity-40 transition-opacity" style={{ transition: 'opacity 120ms ease' }}>
        <path d="M5 2L8 7H2L5 2Z" fill="currentColor" />
      </svg>
    );
  }
  const level = sortLevels[idx];
  return (
    <span className="inline-flex items-center gap-0.5 ml-0.5">
      <svg width="10" height="10" viewBox="0 0 10 10" className={`sort-arrow-rotate ${level.dir === 'asc' ? 'sort-arrow-asc' : ''}`}
        style={{ color: 'var(--accent-violet)', transition: 'transform 200ms cubic-bezier(0.2,0,0,1)' }}>
        <path d="M5 2L8 7H2L5 2Z" fill="currentColor" />
      </svg>
      {sortLevels.length > 1 && (
        <span className="inline-flex items-center justify-center w-[14px] h-[14px] rounded-full text-[9px] font-semibold text-white"
          style={{ background: 'var(--accent-violet)' }}>
          {idx + 1}
        </span>
      )}
    </span>
  );
}

/** Story 3.2: Human-readable sort column name */
function sortColName(col: SortColumn): string {
  if (col.startsWith('horizon_')) return formatHorizon(parseInt(col.split('_')[1], 10));
  const names: Record<string, string> = { asset: 'Asset', sector: 'Sector', signal: 'Signal', momentum: 'Momentum', crash_risk: 'Risk' };
  return names[col] || col;
}

// Column visibility — sortable headers still work; this independently controls rendering.
const ALL_ASSETS_COLUMN_DEFS: ColumnDef[] = [
  { key: 'asset', label: 'Asset', locked: true },
  { key: 'chart', label: 'Chart' },
  { key: 'pct30d', label: '30D change', hint: '%' },
  { key: 'sector', label: 'Sector' },
  { key: 'signal', label: 'Signal', locked: true },
  { key: 'strength', label: 'Strength' },
  { key: 'momentum', label: 'Momentum' },
  { key: 'quality', label: 'Quality' },
  { key: 'risk', label: 'Crash risk' },
  { key: 'horizons', label: 'Horizons' },
];
const ALL_ASSETS_COLS_LS_KEY = 'signals-visible-cols-v2';
const DEFAULT_VISIBLE_COLS = new Set(ALL_ASSETS_COLUMN_DEFS.map((c) => c.key));

function loadVisibleCols(): Set<string> {
  try {
    const raw = localStorage.getItem(ALL_ASSETS_COLS_LS_KEY);
    if (!raw) return new Set(DEFAULT_VISIBLE_COLS);
    const parsed = JSON.parse(raw) as string[];
    const set = new Set(parsed);
    // Always force locked columns on
    ALL_ASSETS_COLUMN_DEFS.forEach((c) => { if (c.locked) set.add(c.key); });
    return set;
  } catch {
    return new Set(DEFAULT_VISIBLE_COLS);
  }
}

function AllAssetsTable({ rows, horizons, updatedAsset, sortLevels, onSort, onRemoveSort, expandedRow, onExpandRow, qualityScores, onNavigateChart, disablePagination }: {
  rows: SummaryRow[]; horizons: number[]; updatedAsset: string | null;
  sortLevels: { col: SortColumn; dir: SortDir }[];
  onSort: (col: SortColumn, shiftKey: boolean) => void;
  onRemoveSort: (col: SortColumn) => void;
  expandedRow: string | null; onExpandRow: (label: string | null) => void;
  qualityScores: Record<string, number>;
  onNavigateChart: (symbol: string) => void;
  /** When true, render all rows on a single page (no pager UI). Used by the
   * Watchlist panel where the row count is small and users dislike paging. */
  disablePagination?: boolean;
}) {
  const [page, setPage] = useState(0);
  const [scrolled, setScrolled] = useState(false);
  const [visibleCols, setVisibleCols] = useState<Set<string>>(() => loadVisibleCols());
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const pageSize = 50;

  useEffect(() => {
    try {
      localStorage.setItem(
        ALL_ASSETS_COLS_LS_KEY,
        JSON.stringify(Array.from(visibleCols)),
      );
    } catch { /* ignore quota / privacy errors */ }
  }, [visibleCols]);

  const toggleCol = (key: string) => {
    setVisibleCols((prev) => {
      const def = ALL_ASSETS_COLUMN_DEFS.find((c) => c.key === key);
      if (def?.locked) return prev;
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const resetCols = () => setVisibleCols(new Set(DEFAULT_VISIBLE_COLS));
  const pageRows = disablePagination ? rows : rows.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = disablePagination ? 1 : Math.ceil(rows.length / pageSize);

  useEffect(() => { setPage(0); }, [rows.length]);

  // Detect scroll for sticky header shadow
  useEffect(() => {
    const container = tableContainerRef.current;
    if (!container) return;
    const onScroll = () => setScrolled(container.scrollTop > 0);
    container.addEventListener('scroll', onScroll, { passive: true });
    return () => container.removeEventListener('scroll', onScroll);
  }, []);

  const headerCls = `cosmic-table-header${scrolled ? ' scrolled' : ''}`;

  return (
    <div className="glass-card overflow-hidden fade-up-delay-3">
      {/* Table toolbar: column visibility (click headers to sort, use Columns to hide/show) */}
      <div className="flex items-center justify-between gap-3 px-4 h-9 border-b" style={{ borderColor: 'var(--border-void)' }}>
        <span className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-muted)' }}>
          {rows.length} asset{rows.length === 1 ? '' : 's'}
          <span className="ml-2" style={{ color: 'var(--text-muted)', opacity: 0.6 }}>
            Click a column to sort · Shift+Click to add
          </span>
        </span>
        <ColumnCustomizer
          columns={ALL_ASSETS_COLUMN_DEFS}
          visible={visibleCols}
          onToggle={toggleCol}
          onReset={resetCols}
        />
      </div>
      {/* Story 3.2 AC-5: Sort indicator bar */}
      {sortLevels.length > 0 && (
        <div className="flex items-center gap-2 px-4 h-[28px] text-[10px] text-[var(--text-secondary)]"
          style={{ background: 'var(--void-hover)' }}>
          <span>Sorted by </span>
          {sortLevels.map((s, i) => (
            <span key={s.col} className="inline-flex items-center gap-1">
              {i > 0 && <span className="text-[var(--text-muted)]">, then </span>}
              <span style={{ color: 'var(--accent-violet)' }}>{sortColName(s.col)}</span>
              <span className="text-[var(--text-muted)]">({s.dir})</span>
              <button onClick={() => onRemoveSort(s.col)}
                className="text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors text-[9px] ml-0.5">
                <X className="w-2.5 h-2.5" />
              </button>
            </span>
          ))}
          {sortLevels.length < 3 && (
            <span className="text-[var(--text-muted)] ml-1">(Shift+Click to add)</span>
          )}
        </div>
      )}
      <div ref={tableContainerRef} className="overflow-auto max-h-[calc(100vh-280px)]">
        <table className="w-full text-sm">
          <thead className={headerCls}>
            <tr>
              <th className={`text-left px-4 py-3 sortable-th group ${sortLevels.some(s => s.col === 'asset') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'asset') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('asset', e.shiftKey)}>
                Asset <SortIndicator col="asset" sortLevels={sortLevels} />
              </th>
              {visibleCols.has('chart') && (
                <th className="text-center px-2 py-3 w-[124px]">
                  <span className="text-[10px] text-[var(--text-violet)] uppercase tracking-[0.06em] font-medium">Chart</span>
                </th>
              )}
              {visibleCols.has('pct30d') && (
                <th className="text-center px-2 py-3 w-[56px]">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-[0.06em] font-medium">30D</span>
                </th>
              )}
              {visibleCols.has('sector') && (
                <th className={`text-left px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'sector') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'sector') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('sector', e.shiftKey)}>
                  Sector <SortIndicator col="sector" sortLevels={sortLevels} />
                </th>
              )}
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'signal') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'signal') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('signal', e.shiftKey)}>
                Signal <SortIndicator col="signal" sortLevels={sortLevels} />
              </th>
              {visibleCols.has('strength') && (
                <th className="text-center px-2 py-3 w-[64px]">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-[0.06em] font-medium">Strength</span>
                </th>
              )}
              {visibleCols.has('momentum') && (
                <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'momentum') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'momentum') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('momentum', e.shiftKey)}>
                  Mom <SortIndicator col="momentum" sortLevels={sortLevels} />
                </th>
              )}
              {visibleCols.has('quality') && (
                <th className="text-center px-2 py-3 w-[56px]">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-[0.06em] font-medium">Quality</span>
                </th>
              )}
              {visibleCols.has('risk') && (
                <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'crash_risk') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'crash_risk') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('crash_risk', e.shiftKey)}>
                  Risk <SortIndicator col="crash_risk" sortLevels={sortLevels} />
                </th>
              )}
              {visibleCols.has('horizons') && horizons.map((h) => {
                const hCol = `horizon_${h}` as SortColumn;
                return (
                  <th key={h} className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === hCol) ? 'active' : ''}`}
                      style={sortLevels.some(s => s.col === hCol) ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                      onClick={(e) => onSort(hCol, e.shiftKey)}>
                    {formatHorizon(h)} <SortIndicator col={hCol} sortLevels={sortLevels} />
                  </th>
                );
              })}
              <th className="w-8 px-2"></th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row) => {
              const ticker = extractTicker(row.asset_label);
              const isExpanded = expandedRow === row.asset_label;
              return (
                <CosmicSignalRow
                  key={row.asset_label}
                  row={row}
                  ticker={ticker}
                  horizons={horizons}
                  visibleCols={visibleCols}
                  qualityScore={qualityScores[ticker] ?? 50}
                  highlighted={row.asset_label === updatedAsset}
                  isExpanded={isExpanded}
                  onToggleExpand={() => onExpandRow(isExpanded ? null : row.asset_label)}
                  onNavigateChart={() => onNavigateChart(ticker)}
                />
              );
            })}
          </tbody>
        </table>
      </div>
      {!disablePagination && totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-[var(--border-void)]">
          <span className="text-xs text-[var(--text-muted)]">
            Page {page + 1} of {totalPages} ({rows.length} total)
          </span>
          <div className="flex gap-1">
            {(['First', 'Prev', 'Next', 'Last'] as const).map((label) => {
              const disabled = (label === 'First' || label === 'Prev') ? page === 0 : page >= totalPages - 1;
              const onClick = () => {
                if (label === 'First') setPage(0);
                else if (label === 'Prev') setPage(Math.max(0, page - 1));
                else if (label === 'Next') setPage(Math.min(totalPages - 1, page + 1));
                else setPage(totalPages - 1);
              };
              return (
                <button key={label} onClick={onClick} disabled={disabled}
                  className="px-2 py-0.5 rounded text-xs text-[var(--accent-violet)] hover:bg-[var(--void-hover)] disabled:opacity-30 transition">
                  {label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Story 3.1: Cosmic Signal Row ────────────────────────────────── */
function CosmicSignalRow({ row, ticker, horizons, visibleCols, qualityScore, highlighted, isExpanded, onToggleExpand, onNavigateChart }: {
  row: SummaryRow; ticker: string; horizons: number[];
  visibleCols: Set<string>;
  qualityScore: number;
  highlighted?: boolean; isExpanded: boolean;
  onToggleExpand: () => void; onNavigateChart: () => void;
}) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  // Compute composite for strength bar
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  const pUp = nearestHorizon?.p_up;
  const kellyVal = nearestHorizon?.kelly_half;

  const labelColor = signalLabelColor(label);
  return (
    <>
      <tr
        onClick={onToggleExpand}
        className={`cosmic-row cursor-pointer transition-all duration-150 ${highlighted ? 'aurora-upgrade' : ''} ${isExpanded ? 'signals-row-selected' : 'hover:bg-white/[0.015]'}`}
        style={isExpanded ? {
          borderLeft: '2px solid var(--accent-violet)',
          background: 'rgba(139,92,246,0.05)',
          boxShadow: 'inset 0 0 20px rgba(139,92,246,0.06)',
        } : {
          borderLeft: '2px solid transparent',
          borderBottom: '1px solid rgba(255,255,255,0.035)',
        }}
      >
        {/* Asset */}
        <td className="px-4 py-2.5 whitespace-nowrap" style={{ height: '40px' }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: labelColor }} />
            <div className="text-left">
              <span className="font-semibold text-white text-[12px] tabular-nums">
                {ticker}
              </span>
              {row.asset_label.includes('(') && (
                <span className="block text-[9px] text-[var(--text-muted)] truncate max-w-[140px] leading-tight">
                  {row.asset_label.split('(')[0].trim()}
                </span>
              )}
            </div>
          </div>
        </td>
        {/* AC-1: Sparkline — wider for clarity */}
        {visibleCols.has('chart') && (
          <td className="px-2 py-2 text-center">
            <Sparkline ticker={ticker} width={108} height={32} />
          </td>
        )}
        {/* 30D pct change */}
        {visibleCols.has('pct30d') && (
          <td className="px-1.5 py-2 text-center">
            <SparklinePct ticker={ticker} />
          </td>
        )}
        {/* Sector */}
        {visibleCols.has('sector') && (
          <td className="px-3 py-2 text-[10px] text-[var(--text-secondary)] max-w-[100px] truncate">{row.sector}</td>
        )}
        {/* AC-2: Signal label + strength split */}
        <td className="px-2 py-2">
          <div className="flex justify-center">
            <SignalLabel label={label.toUpperCase()} />
          </div>
        </td>
        {visibleCols.has('strength') && (
          <td className="px-1 py-2">
            <div className="flex justify-center">
              <SignalStrengthMeter label={label} pUp={pUp} kelly={kellyVal} />
            </div>
          </td>
        )}
        {/* AC-3: Momentum badge */}
        {visibleCols.has('momentum') && (
          <td className="px-3 py-2 text-center">
            <MomentumBadge value={row.momentum_score} />
          </td>
        )}
        {/* Quality score tile */}
        {visibleCols.has('quality') && (
          <td className="px-2 py-2">
            <QualityCell score={qualityScore} />
          </td>
        )}
        {/* AC-4: Crash risk heat */}
        {visibleCols.has('risk') && (
          <td className="px-3 py-2">
            <div className="flex justify-center">
              <CrashRiskHeat score={row.crash_risk_score} />
            </div>
          </td>
        )}
        {/* AC-5: Horizon cells */}
        {visibleCols.has('horizons') && horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          return (
            <td key={h} className="px-2 py-2 text-center">
              <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
            </td>
          );
        })}
        {/* Expand indicator */}
        <td className="px-2 py-2">
          <ChevronRight
            className="w-3.5 h-3.5 transition-all duration-200"
            style={{
              color: isExpanded ? 'var(--accent-violet)' : 'var(--text-muted)',
              transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
            }}
          />
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={1000} className="p-0">
            <SignalDetailPanel
              ticker={ticker}
              signal={row.nearest_label}
              momentum={row.momentum_score}
              crashRisk={row.crash_risk_score}
              horizonSignals={row.horizon_signals as any}
              onNavigateChart={onNavigateChart}
            />
          </td>
        </tr>
      )}
    </>
  );
}

/* ── Sector signal row — premium with inline expand ───────────────── */
function SectorSignalRow({ row, horizons, visibleCols, qualityScore, highlighted, delayMs = 0, onNavigateChart }: {
  row: SummaryRow; horizons: number[];
  visibleCols: Set<string>;
  qualityScore: number;
  highlighted?: boolean; delayMs?: number;
  onNavigateChart: (sym: string) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  const ticker = extractTicker(row.asset_label);
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  const labelColor = signalLabelColor(label);

  return (
    <>
      <tr
        onClick={() => setIsExpanded(p => !p)}
        className={`cursor-pointer transition-all duration-150 ${highlighted ? 'aurora-upgrade' : ''} ${isExpanded ? '' : 'hover:bg-white/[0.015]'}`}
        style={isExpanded ? {
          animationDelay: `${delayMs}ms`,
          borderLeft: '2px solid var(--accent-violet)',
          background: 'rgba(139,92,246,0.05)',
          boxShadow: 'inset 0 0 20px rgba(139,92,246,0.06)',
        } : {
          animationDelay: `${delayMs}ms`,
          borderLeft: '2px solid transparent',
          borderBottom: '1px solid rgba(255,255,255,0.035)',
        }}>
        {/* Asset */}
        <td className="px-3 py-2 whitespace-nowrap">
          <div className="flex items-center gap-2">
            <div className="w-1 h-7 rounded-full flex-shrink-0" style={{ background: `${labelColor}60` }} />
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-[12px] text-[#e2e8f0]">{ticker}</span>
                <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold leading-none"
                  style={{ background: `${labelColor}15`, color: labelColor }}>
                  {label}
                </span>
              </div>
              {row.asset_label.includes('(') && (
                <span className="text-[9px] text-[var(--text-muted)] truncate max-w-[140px] leading-tight block mt-0.5">
                  {row.asset_label.split('(')[0].trim()}
                </span>
              )}
            </div>
          </div>
        </td>
        {/* Sparkline — wider for clarity */}
        {visibleCols.has('chart') && (
          <td className="px-2 py-2 text-center">
            <Sparkline ticker={ticker} width={108} height={32} />
          </td>
        )}
        {/* 30D pct change */}
        {visibleCols.has('pct30d') && (
          <td className="px-1.5 py-2 text-center">
            <SparklinePct ticker={ticker} />
          </td>
        )}
        {/* Signal label + strength split */}
        <td className="px-1.5 py-2">
          <div className="flex justify-center">
            <SignalLabel label={label.toUpperCase()} />
          </div>
        </td>
        {visibleCols.has('strength') && (
          <td className="px-1 py-2">
            <div className="flex justify-center">
              <SignalStrengthMeter label={label} pUp={nearestHorizon?.p_up} kelly={nearestHorizon?.kelly_half} />
            </div>
          </td>
        )}
        {/* Momentum */}
        {visibleCols.has('momentum') && (
          <td className="px-1.5 py-2 text-center">
            <MomentumBadge value={row.momentum_score} />
          </td>
        )}
        {/* Quality */}
        {visibleCols.has('quality') && (
          <td className="px-1.5 py-2">
            <QualityCell score={qualityScore} />
          </td>
        )}
        {/* Horizon cells */}
        {visibleCols.has('horizons') && horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          return (
            <td key={h} className="px-1 py-2 text-center">
              <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
            </td>
          );
        })}
        {/* Risk */}
        {visibleCols.has('risk') && (
          <td className="px-1.5 py-2">
            <div className="flex justify-center">
              <CrashRiskHeat score={row.crash_risk_score} />
            </div>
          </td>
        )}
        {/* Actions */}
        <td className="px-1 py-2">
          <ChevronRight
            className="w-3.5 h-3.5 transition-all duration-200"
            style={{
              color: isExpanded ? 'var(--accent-violet)' : 'var(--text-muted)',
              transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
            }}
          />
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={1000} className="p-0">
            <SignalDetailPanel
              ticker={ticker}
              signal={row.nearest_label}
              momentum={row.momentum_score}
              crashRisk={row.crash_risk_score}
              horizonSignals={row.horizon_signals as any}
              onNavigateChart={() => onNavigateChart(ticker)}
            />
          </td>
        </tr>
      )}
    </>
  );
}

/* ── High Conviction Panel — full positions with rich data ────────── */
type HCSortCol = 'ticker' | 'exp_ret' | 'p_up' | 'strength' | 'sector';
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

// ─── Apple-grade micro components for HighConvictionPanel ───────────────

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
function SmaReversalsPanel({
  data,
  isLoading,
  rows,
  onNavigateChart,
}: {
  data: SmaReversalsData | undefined;
  isLoading: boolean;
  rows: SummaryRow[];
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

  useEffect(() => {
    if (!expandedKey) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setExpandedKey(null);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [expandedKey]);

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

  const reversals = data?.reversals || [];
  const counts = data?.counts_by_period || {};

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
        {data?.buy_setups !== undefined && (
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
            <span className="text-[10.5px] font-semibold text-white tabular-nums">{data.buy_setups}</span>
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
          {data?.grade_counts && (
            <div className="flex items-center gap-1.5 pl-1">
              <GradeBadge grade="A" count={data.grade_counts.A} />
              <GradeBadge grade="B" count={data.grade_counts.B} />
              <GradeBadge grade="C" count={data.grade_counts.C} />
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
          <div className="grid grid-cols-1 md:grid-cols-2 gap-1.5">
            {displayed.map((r) => {
              const key = `${r.symbol}-${r.period}`;
              const isExpanded = expandedKey === key;
              const label = labelMap.get(r.symbol) ?? labelMap.get(r.symbol.replace(/=/g, '_')) ?? r.symbol;
              return (
                <React.Fragment key={key}>
                  <ReversalRow
                    r={r}
                    label={label}
                    isExpanded={isExpanded}
                    onClick={() => setExpandedKey((prev) => (prev === key ? null : key))}
                  />
                  {isExpanded && (
                    <div className="md:col-span-2">
                      <ReversalDetailPanel
                        r={r}
                        label={label}
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

function ReversalRow({ r, label, onClick, isExpanded = false }: { r: SmaReversal; label: string; onClick: () => void; isExpanded?: boolean }) {
  const isBull = r.direction === 'bull';
  const accent = isBull ? '#10b981' : '#f43f5e';
  const accentSoft = isBull ? '#34d399' : '#fb7185';
  const ArrowIcon = isBull ? TrendingUp : TrendingDown;
  // Score bar fill
  const scorePct = Math.max(0, Math.min(100, r.score));
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
  const tooltip = r.grade_reasons && r.grade_reasons.length > 0 ? r.grade_reasons.join(' · ') : undefined;

  return (
    <button
      type="button"
      onClick={onClick}
      title={tooltip}
      className="group relative w-full text-left rounded-xl px-3 py-2.5 transition-all duration-200"
      style={{
        background: isExpanded
          ? `linear-gradient(180deg, ${accent}14, ${accent}04)`
          : 'linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0.008))',
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
      <div className="flex items-center gap-3">
        {/* Direction chip */}
        <div
          className="flex items-center justify-center rounded-lg flex-shrink-0"
          style={{
            width: 32, height: 32,
            background: `linear-gradient(180deg, ${accent}26, ${accent}08)`,
            border: `1px solid ${accent}50`,
            boxShadow: `0 0 14px -5px ${accent}70 inset`,
          }}
        >
          <ArrowIcon className="w-4 h-4" style={{ color: accentSoft, filter: `drop-shadow(0 0 4px ${accent}90)` }} />
        </div>

        {/* Symbol + label */}
        <div className="flex flex-col min-w-0 flex-1">
          <div className="flex items-center gap-1.5">
            <span className="text-[13px] font-semibold text-[var(--text-primary)] tabular-nums tracking-tight">{r.symbol}</span>
            {/* Grade pill */}
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
            {!r.regime_ok && (
              <span
                className="inline-flex items-center gap-0.5 rounded px-1 py-[1px] text-[8.5px] font-semibold uppercase tracking-[0.1em]"
                style={{ background: 'rgba(239,68,68,0.10)', border: '1px solid rgba(239,68,68,0.30)', color: '#fca5a5' }}
                title={`Against regime (price ${isBull ? 'below' : 'above'} SMA${r.regime_sma !== null ? ' 200' : ''}) — buying against the trend`}
              >
                vs regime
              </span>
            )}
            {r.overextended && (
              <span
                className="inline-flex items-center gap-0.5 rounded px-1 py-[1px] text-[8.5px] font-semibold uppercase tracking-[0.1em]"
                style={{ background: 'rgba(251,191,36,0.12)', border: '1px solid rgba(251,191,36,0.35)', color: '#fbbf24' }}
                title="Price > 3 ATR from SMA — chasing kills edge"
              >
                overext
              </span>
            )}
            {r.false_break && (
              <span title="Price re-crossed within 3 bars — possible whipsaw">
                <AlertTriangle className="w-3 h-3" style={{ color: '#fbbf24' }} />
              </span>
            )}
          </div>
          <span className="text-[10px] text-[var(--text-muted)] truncate">{displayLabel}</span>
          {/* Trade geometry line — "Stop · Tgt · R:R · Win% (n)" */}
          {(hasTradeGeometry || edge.win_rate !== null) && (
            <div className="flex flex-wrap items-center gap-x-2 gap-y-0.5 mt-0.5 text-[9.5px] tabular-nums">
              {hasTradeGeometry && (
                <>
                  <span className="text-[var(--text-muted)]">
                    Stop <span className="text-[var(--text-secondary)] font-medium">{r.stop_price!.toFixed(2)}</span>
                  </span>
                  <span className="text-[var(--text-muted)]">·</span>
                  <span className="text-[var(--text-muted)]">
                    Tgt <span className="text-[var(--text-secondary)] font-medium">{r.target_price!.toFixed(2)}</span>
                  </span>
                  <span className="text-[var(--text-muted)]">·</span>
                  <span className="text-[var(--text-muted)]">
                    R:R <span className="font-semibold" style={{ color: accentSoft }}>{r.risk_reward!.toFixed(1)}</span>
                  </span>
                </>
              )}
              {edge.win_rate !== null && (
                <>
                  {hasTradeGeometry && <span className="text-[var(--text-muted)]">·</span>}
                  <span
                    className="inline-flex items-center gap-0.5"
                    title={`Historical ${r.edge_forward_days}-bar forward win-rate across ${edge.samples} past crossings. Median return ${edge.median_fwd_pct !== null ? edge.median_fwd_pct.toFixed(2) + '%' : '—'}.`}
                  >
                    <Target className="w-2.5 h-2.5" style={{ color: edge.win_rate >= 0.55 ? accentSoft : '#94a3b8' }} />
                    <span
                      className="font-semibold"
                      style={{ color: edge.win_rate >= 0.55 ? accentSoft : 'var(--text-secondary)' }}
                    >
                      {(edge.win_rate * 100).toFixed(0)}%
                    </span>
                    <span className="text-[var(--text-muted)]">n={edge.samples}</span>
                  </span>
                </>
              )}
            </div>
          )}
        </div>

        {/* Metrics cluster (md+ only) */}
        <div className="hidden md:flex items-center gap-3 text-[10px] tabular-nums">
          <div className="flex flex-col items-end">
            <span className="text-[9px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Dist</span>
            <span className="text-[var(--text-primary)] font-medium">
              {r.distance_pct > 0 ? '+' : ''}{r.distance_pct.toFixed(2)}%
            </span>
          </div>
          {r.atr_distance !== null && (
            <div className="flex flex-col items-end">
              <span className="text-[9px] uppercase tracking-[0.12em] text-[var(--text-muted)]">ATR</span>
              <span className="text-[var(--text-primary)] font-medium">{r.atr_distance.toFixed(2)}σ</span>
            </div>
          )}
          {r.volume_ratio !== null && (
            <div className="flex flex-col items-end">
              <span className="text-[9px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Vol</span>
              <span
                className="font-medium"
                style={{ color: r.volume_ratio >= 1.2 ? accentSoft : 'var(--text-primary)' }}
              >
                {r.volume_ratio.toFixed(2)}×
              </span>
            </div>
          )}
          <div className="flex flex-col items-end">
            <span className="text-[9px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Age</span>
            <span className="text-[var(--text-primary)] font-medium">
              {r.days_since_cross === 0 ? 'Today' : `${r.days_since_cross}d`}
            </span>
          </div>
        </div>

        {/* Persistence pips */}
        <div className="flex items-center gap-0.5 px-1" title={`On new side: ${r.persistence} of last ${r.persistence_window} bars`}>
          {persistencePips.map((on, i) => (
            <span
              key={i}
              className="rounded-full"
              style={{
                width: 5, height: 5,
                background: on ? accentSoft : 'rgba(255,255,255,0.1)',
                boxShadow: on ? `0 0 4px ${accentSoft}` : 'none',
              }}
            />
          ))}
        </div>

        {/* Score bar */}
        <div className="flex flex-col items-end gap-0.5 w-14 flex-shrink-0">
          <span className="text-[12px] font-bold tabular-nums" style={{ color: scoreColor }}>
            {r.score.toFixed(0)}
          </span>
          <div className="w-full h-[3px] rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.06)' }}>
            <div
              className="h-full rounded-full"
              style={{
                width: `${scorePct}%`,
                background: `linear-gradient(90deg, ${scoreColor}, ${scoreColor}cc)`,
                boxShadow: `0 0 8px -2px ${scoreColor}`,
                transition: 'width 320ms cubic-bezier(.2,.8,.2,1)',
              }}
            />
          </div>
        </div>
      </div>
    </button>
  );
}


function ReversalDetailPanel({ r, label, onClose, onOpenFullChart }: {
  r: SmaReversal;
  label: string;
  onClose: () => void;
  onOpenFullChart: () => void;
}) {
  const ohlcvQ = useQuery({
    queryKey: ['sma-reversal-ohlcv', r.symbol, 365],
    queryFn: () => api.chartOhlcv(r.symbol, 365),
    staleTime: 120_000,
  });
  const indQ = useQuery({
    queryKey: ['sma-reversal-indicators', r.symbol, 365],
    queryFn: () => api.chartIndicators(r.symbol, 365),
    staleTime: 120_000,
  });
  const forecastQ = useQuery({
    queryKey: ['sma-reversal-forecast', r.symbol],
    queryFn: () => api.chartForecast(r.symbol),
    staleTime: 120_000,
  });

  const isBull = r.direction === 'bull';
  const accent = isBull ? '#10b981' : '#f43f5e';
  const displayLabel = label.replace(/\s*\([^)]+\)\s*$/, '').trim();
  const showLabel = displayLabel && displayLabel !== r.symbol;
  const edge = r.historical_edge;

  const fmtPx = (v: number | null | undefined): string => {
    if (v == null || !isFinite(v)) return '—';
    const abs = Math.abs(v);
    if (abs < 1) return v.toFixed(4);
    if (abs < 100) return v.toFixed(2);
    return v.toFixed(2);
  };

  const stats: Array<{ label: string; value: string; sub?: string; color?: string }> = [
    { label: 'Entry', value: fmtPx(r.price) },
    { label: 'Stop', value: fmtPx(r.stop_price), color: '#fb7185' },
    { label: 'Target', value: fmtPx(r.target_price), color: '#34d399' },
    { label: 'R : R', value: r.risk_reward != null ? r.risk_reward.toFixed(2) : '—' },
    {
      label: 'Win %',
      value: edge?.win_rate != null ? `${(edge.win_rate * 100).toFixed(0)}%` : '—',
      sub: edge?.samples ? `n=${edge.samples}` : undefined,
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
      <div className="grid grid-cols-3 md:grid-cols-6 gap-1.5 px-4 pt-3 pb-1">
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
        {ohlcvQ.isLoading ? (
          <div className="h-[320px] flex items-center justify-center gap-2 text-[11px] text-[var(--text-muted)]">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading chart…
          </div>
        ) : ohlcvQ.error || !ohlcvQ.data?.data?.length ? (
          <div className="h-[320px] flex items-center justify-center text-[11px] text-[var(--text-muted)]">
            No chart data available for {r.symbol}
          </div>
        ) : (
          <MiniPriceChart
            ohlcv={ohlcvQ.data.data}
            indicators={indQ.data?.indicators}
            forecast={forecastQ.data}
            height={320}
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
              {reason}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}


// iOS-style segmented control with a sliding accent indicator.
// Generic over readonly option arrays so callers get a narrow value type.
function SegmentedControl<K extends string>({
  options,
  value,
  onChange,
  accent = 'var(--accent-violet)',
  size = 'md',
}: {
  options: ReadonlyArray<{ key: K; label: string; dot?: string }>;
  value: K;
  onChange: (next: K) => void;
  accent?: string;
  size?: 'sm' | 'md';
}) {
  const btnRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [indicator, setIndicator] = useState<{ x: number; w: number } | null>(null);

  const recompute = useCallback(() => {
    const container = containerRef.current;
    const btn = btnRefs.current[value];
    if (!container || !btn) return;
    const cRect = container.getBoundingClientRect();
    const bRect = btn.getBoundingClientRect();
    setIndicator({ x: bRect.left - cRect.left, w: bRect.width });
  }, [value]);

  useEffect(() => {
    recompute();
  }, [recompute, options.length]);

  useEffect(() => {
    const onResize = () => recompute();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [recompute]);

  const pad = size === 'sm' ? 'px-2 py-[5px]' : 'px-2.5 py-[6px]';
  const textSize = size === 'sm' ? 'text-[10.5px]' : 'text-[11.5px]';

  return (
    <div
      ref={containerRef}
      className="relative inline-flex items-center gap-0 rounded-xl p-[3px]"
      style={{
        background: 'rgba(255,255,255,0.025)',
        border: '1px solid rgba(255,255,255,0.05)',
        boxShadow: '0 1px 0 rgba(255,255,255,0.03) inset',
      }}
    >
      {/* Sliding indicator */}
      {indicator && (
        <div
          aria-hidden
          className="absolute top-[3px] bottom-[3px] rounded-[9px] pointer-events-none"
          style={{
            left: indicator.x,
            width: indicator.w,
            background: `linear-gradient(180deg, ${accent}22, ${accent}0c)`,
            border: `1px solid ${accent}55`,
            boxShadow: `0 0 0 1px ${accent}18 inset, 0 4px 14px -6px ${accent}85, 0 0 18px -6px ${accent}55`,
            transition: 'left 280ms cubic-bezier(.2,.8,.2,1), width 280ms cubic-bezier(.2,.8,.2,1)',
          }}
        />
      )}
      {options.map((opt) => {
        const on = opt.key === value;
        return (
          <button
            key={opt.key}
            ref={(el) => { btnRefs.current[opt.key] = el; }}
            type="button"
            onClick={() => onChange(opt.key)}
            aria-pressed={on}
            className={`relative inline-flex items-center gap-1.5 rounded-[9px] ${pad} ${textSize} font-medium transition-colors duration-200`}
            style={{
              color: on ? '#fff' : 'var(--text-secondary)',
            }}
          >
            {opt.dot && (
              <span
                aria-hidden
                className="rounded-full"
                style={{
                  width: 5,
                  height: 5,
                  background: opt.dot,
                  boxShadow: on ? `0 0 6px ${opt.dot}` : 'none',
                  transition: 'box-shadow 220ms',
                }}
              />
            )}
            <span className="whitespace-nowrap">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}

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

function HighConvictionPanel({
  title, signals, color, isLoading, emaStates,
}: {
  title: string;
  signals: HighConvictionSignal[];
  color: 'green' | 'red';
  isLoading: boolean;
  emaStates: Record<string, EmaState>;
}) {
  const [sortCol, setSortCol] = useState<HCSortCol>('exp_ret');
  const [sortDir, setSortDir] = useState<HCSortDir>('desc');
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [expandedView, setExpandedView] = useState<'details' | 'chart'>('chart');
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

  // Reset tab to chart when row changes (chart is the default view)
  useEffect(() => { setExpandedView('chart'); }, [expandedTicker]);

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
        case 'exp_ret': cmp = a.bestReturn - b.bestReturn; break;
        case 'p_up': cmp = a.avgPUp - b.avgPUp; break;
        case 'strength': cmp = a.maxStrength - b.maxStrength; break;
        case 'sector': cmp = a.sector.localeCompare(b.sector); break;
      }
      return cmp * mult;
    });
    return arr;
  }, [filtered, sortCol, sortDir]);

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
                          colSpan={7}
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
                                    width: 'calc(50% - 4px)',
                                    left: expandedView === 'chart' ? '4px' : 'calc(50% + 0px)',
                                    background: accent,
                                    boxShadow: `0 2px 8px -2px ${accent}88, inset 0 1px 0 rgba(255,255,255,0.2)`,
                                    transition: 'left 260ms cubic-bezier(0.2, 0.8, 0.2, 1)',
                                  }}
                                />
                                <button
                                  type="button"
                                  onClick={() => setExpandedView('chart')}
                                  className="relative z-10 inline-flex items-center gap-1.5 px-4 py-1.5 rounded-lg text-[11px] font-semibold transition-colors"
                                  style={{
                                    color: expandedView === 'chart' ? 'var(--void-bg)' : 'var(--text-secondary)',
                                    minWidth: 110,
                                    justifyContent: 'center',
                                  }}
                                >
                                  <BarChart3 className="w-3.5 h-3.5" />
                                  Chart
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

                            {expandedView === 'chart' && (() => {
                              const tvSym = toTvSymbol(g.ticker);
                              const buildSrc = (interval: string, range: string) =>
                                `https://s.tradingview.com/widgetembed/?frameElementId=tv_${encodeURIComponent(g.ticker)}_${range}&symbol=${encodeURIComponent(tvSym)}&interval=${interval}&range=${range}&hidesidetoolbar=0&hidetoptoolbar=0&symboledit=1&saveimage=0&toolbarbg=0f0f1a&theme=dark&style=1&timezone=Etc/UTC&withdateranges=1&hideideas=1&hideideasbutton=1&locale=en`;
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
                                            title={`${r.label} · ${r.interval === 'D' ? 'Daily' : r.interval === 'W' ? 'Weekly' : `${r.interval}m`} candles${isPreloaded && !isActive ? ' · preloaded' : ''}`}
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
