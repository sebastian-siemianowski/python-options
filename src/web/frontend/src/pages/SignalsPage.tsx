import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { SummaryRow, SectorGroup, StrongSignalEntry, HighConvictionSignal, SignalSummaryData, SignalStats, EmaState } from '../api';
import PageHeader from '../components/PageHeader';
import { SignalTableSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { SignalsEmpty } from '../components/CosmicEmptyState';
import { ExportButton } from '../components/ExportButton';
import { Sparkline, SparklinePct } from '../components/Sparkline';
import { SignalLabel, SignalStrengthBar, SignalStrengthMeter, MomentumBadge, CrashRiskHeat, HorizonCell, QualityCell } from '../components/SignalTableVisuals';
import { ColumnCustomizer, type ColumnDef } from '../components/ColumnCustomizer';
import SignalDetailPanel from '../components/SignalDetailPanel';
import { JobRunnerModal, type JobMode } from '../components/JobRunnerModal';
import {
  ArrowUpCircle, ArrowDownCircle, Filter, ChevronDown, ChevronRight,
  TrendingUp, TrendingDown, Search, X, ExternalLink, BarChart3,
  Target, Shield, ArrowUp, ArrowDown, Clock, DollarSign,
  Activity, Eye, Layers, ChevronUp,
} from 'lucide-react';
import { formatHorizon, responsiveHorizons } from '../utils/horizons';

import { useWebSocket, type WSStatus } from '../hooks/useWebSocket';

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

/** WebSocket connection status indicator. */
function WsStatusDot({ status }: { status: WSStatus }) {
  const dotClass = status === 'connected' ? 'ws-connected' : status === 'connecting' ? 'ws-connecting' : 'ws-disconnected';
  const label = status === 'connected' ? 'Live' : status === 'connecting' ? 'Reconnecting' : 'Disconnected';
  const color = status === 'connected' ? 'var(--accent-emerald)' : status === 'connecting' ? 'var(--accent-amber)' : 'var(--accent-rose)';
  return (
    <span className="inline-flex items-center gap-1.5 ml-2" title={`WebSocket: ${status}`}>
      <span className={dotClass} />
      <span className="text-caption font-medium" style={{ color }}>{label}</span>
    </span>
  );
}

function SignalsPageInner() {
  const navigate = useNavigate();
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
  const [jobMode, setJobMode] = useState<JobMode | null>(null);

  const [updatedAsset, setUpdatedAsset] = useState<string | null>(null);

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
    const st = ticker ? emaStates[ticker] : undefined;
    if (!st) return false;
    if (emaFilters.p9 && st.below_9 !== true) return false;
    if (emaFilters.p50 && st.below_50 !== true) return false;
    if (emaFilters.p600 && st.below_600 !== true) return false;
    return true;
  }, [emaFilters, emaStates]);

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
      next.has(name) ? next.delete(name) : next.add(name);
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

  return (
    <>
      {/* ── v1 PREMIUM HERO BAND ─────────────────────────────────────── */}
      <SignalsHero stats={stats} rows={rows} horizons={horizons} filteredCount={filteredRows.length} wsStatus={wsStatus} />


      {/* High Conviction Panels — full positions with rich data */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-8 fade-up-delay-1">
        <HighConvictionPanel
          title="High Conviction BUY"
          signals={buyQ.data?.signals || []}
          color="green"
          isLoading={buyQ.isLoading}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
          emaStates={emaStates}
        />
        <HighConvictionPanel
          title="High Conviction SELL"
          signals={sellQ.data?.signals || []}
          color="red"
          isLoading={sellQ.isLoading}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
          emaStates={emaStates}
        />
      </div>

      {/* Export button row */}
      <div className="flex items-center justify-between gap-3 mb-3">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setJobMode('stocks')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-semibold transition-colors"
            style={{
              background: 'rgba(59,130,246,0.12)',
              color: '#60a5fa',
              border: '1px solid rgba(59,130,246,0.28)',
            }}
            title="Run `make stocks` (refresh prices and regenerate signals)"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/></svg>
            Refresh Stocks
          </button>
          <button
            onClick={() => setJobMode('retune')}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-[12px] font-semibold transition-colors"
            style={{
              background: 'rgba(139,92,246,0.12)',
              color: '#a78bfa',
              border: '1px solid rgba(139,92,246,0.28)',
            }}
            title="Run `make retune` (full retune pipeline)"
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"/></svg>
            Run Tune
          </button>
        </div>
        <ExportButton
          filename="signals"
          columns={[
            { key: 'asset_label', label: 'Asset' },
            { key: 'signal', label: 'Signal' },
            { key: 'momentum_score', label: 'Momentum' },
            { key: 'crash_risk_score', label: 'Crash Risk' },
          ]}
          data={filteredRows as unknown as Record<string, unknown>[]}
          filteredCount={filteredRows.length}
          totalCount={rows.length}
        />
      </div>

      {/* Inline job runner sub-panel (renders only when active) */}
      <JobRunnerModal
        open={jobMode !== null}
        mode={jobMode}
        onClose={() => setJobMode(null)}
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
              { key: 'p9' as const,   period: 'EMA 9',   count: rows.filter(r => emaStates[r.asset_label]?.below_9   === true).length },
              { key: 'p50' as const,  period: 'EMA 50',  count: rows.filter(r => emaStates[r.asset_label]?.below_50  === true).length },
              { key: 'p600' as const, period: 'EMA 600', count: rows.filter(r => emaStates[r.asset_label]?.below_600 === true).length },
            ]).map(({ key, period, count }) => {
              const on = emaFilters[key];
              const emaLoaded = Object.keys(emaStates).length > 0;
              const accent = 'var(--accent-violet)';
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

          {view === 'sectors' && (
            <>
              <div className="h-4 w-px bg-white/[0.05]" aria-hidden />
              <button onClick={expandAll} className="text-[10px] text-[var(--accent-violet)] hover:underline">Expand all</button>
              <button onClick={collapseAll} className="text-[10px] text-[var(--text-muted)] hover:underline">Collapse all</button>
            </>
          )}
        </div>
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

      {/* Story 3.6: Horizon pill selector */}
      {(view === 'all' || view === 'sectors') && allHorizons.length > 0 && (
        <div className="flex items-center gap-1.5 mb-3 fade-up-delay-2">
          <span className="text-[10px] text-[var(--text-muted)] mr-1">Horizons</span>
          {allHorizons.map(h => {
            const active = horizons.includes(h);
            return (
              <button key={h} onClick={() => toggleHorizon(h)}
                className="px-2 py-0.5 rounded-full text-[10px] font-medium transition-all duration-120"
                style={active
                  ? { color: 'var(--accent-violet)', background: 'var(--violet-15)', border: '1px solid var(--border-glow)' }
                  : { color: 'var(--text-secondary)', background: 'var(--void-active)', border: '1px solid transparent' }
                }
              >
                {formatHorizon(h)}
              </button>
            );
          })}
          {horizonOverride && (
            <button onClick={resetHorizons} className="text-[9px] text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors ml-1">
              Reset
            </button>
          )}
        </div>
      )}

      {/* Content */}
      {view === 'sectors' && (
        <SectorPanels
          sectors={sectors}
          expandedSectors={expandedSectors}
          toggleSector={toggleSector}
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

/* ── Premium stat card with progress bar ─────────────────────────── */
function StatCard({ label, value, total, color, icon }: { label: string; value: number; total: number; color: string; icon: React.ReactNode }) {
  const pct = total > 0 ? (value / total) * 100 : 0;
  return (
    <div className="glass-card px-3 py-3 hover-lift stat-shine group cursor-default"
      style={{ borderBottom: `2px solid ${color}20` }}>
      <div className="flex items-center gap-2 mb-1.5">
        <div className="w-7 h-7 rounded-lg flex items-center justify-center" style={{ background: `${color}15`, color }}>{icon}</div>
        <div className="flex-1 min-w-0">
          <p className="text-lg font-bold text-[#e2e8f0] tabular-nums leading-tight">{value}</p>
        </div>
      </div>
      <div className="flex items-center justify-between mb-1">
        <p className="text-[10px] text-[var(--text-secondary)] font-medium">{label}</p>
        <p className="text-[9px] tabular-nums" style={{ color: `${color}99` }}>{pct.toFixed(1)}%</p>
      </div>
      <div className="w-full h-1 rounded-full bg-white/[0.04] overflow-hidden">
        <div className="h-full rounded-full transition-all duration-700 ease-out" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
      </div>
    </div>
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
    wsStatus === 'error' ? '#f43f5e' : '#64748b';

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
  horizons,
  search,
  filter,
  updatedAsset,
  qualityScores,
}: {
  sectors: SectorGroup[];
  expandedSectors: Set<string>;
  toggleSector: (name: string) => void;
  horizons: number[];
  search: string;
  filter: SignalFilter;
  updatedAsset: string | null;
  qualityScores: Record<string, number>;
}) {
  const [sectorSort, setSectorSort] = useState<SectorSortBy>('momentum');
  const navigate = useNavigate();
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

  // Global sector stats
  const totalAssets = sectors.reduce((s, sec) => s + sec.asset_count, 0);
  const totalBullish = sectors.reduce((s, sec) => s + (sec.strong_buy ?? 0) + (sec.buy ?? 0), 0);
  const totalBearish = sectors.reduce((s, sec) => s + (sec.strong_sell ?? 0) + (sec.sell ?? 0), 0);

  return (
    <div className="space-y-3">
      {/* Sort bar — pill selector with global stats */}
      <div className="flex items-center gap-3 mb-1">
        <div className="flex items-center gap-1 glass-card px-2 py-1.5">
          {SECTOR_SORT_OPTIONS.map(({ key, label, icon }) => (
            <button key={key}
              onClick={() => setSectorSort(key)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all duration-200 ${
                sectorSort === key
                  ? 'bg-[var(--accent-violet)]/15 text-[var(--text-violet)] shadow-[0_0_8px_var(--violet-15)]'
                  : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)] hover:bg-white/[0.02]'
              }`}
            >
              {icon}
              <span className="hidden sm:inline">{label}</span>
            </button>
          ))}
        </div>
        <div className="flex items-center gap-4 ml-auto text-[10px]">
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-emerald)]" />
            <span className="text-[var(--text-muted)]">Bullish</span>
            <span className="font-bold text-[var(--accent-emerald)] tabular-nums">{totalBullish}</span>
          </span>
          <span className="flex items-center gap-1">
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-rose)]" />
            <span className="text-[var(--text-muted)]">Bearish</span>
            <span className="font-bold text-[var(--accent-rose)] tabular-nums">{totalBearish}</span>
          </span>
          <span className="text-[var(--text-muted)] tabular-nums">{totalAssets} assets</span>
        </div>
        <ColumnCustomizer
          columns={SECTOR_COLUMN_DEFS}
          visible={sectorVisibleCols}
          onToggle={toggleSectorCol}
          onReset={resetSectorCols}
        />
      </div>

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

function AllAssetsTable({ rows, horizons, updatedAsset, sortLevels, onSort, onRemoveSort, expandedRow, onExpandRow, qualityScores, onNavigateChart }: {
  rows: SummaryRow[]; horizons: number[]; updatedAsset: string | null;
  sortLevels: { col: SortColumn; dir: SortDir }[];
  onSort: (col: SortColumn, shiftKey: boolean) => void;
  onRemoveSort: (col: SortColumn) => void;
  expandedRow: string | null; onExpandRow: (label: string | null) => void;
  qualityScores: Record<string, number>;
  onNavigateChart: (symbol: string) => void;
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
  const pageRows = rows.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(rows.length / pageSize);

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
      {totalPages > 1 && (
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

/* ── Mini chart panel (legacy — retained for internal reference, unused) ── */
function MiniChartPanel({ ticker, onNavigateChart }: { ticker: string; onNavigateChart: () => void }) {
  const { data, isLoading, error } = useQuery({
    queryKey: ['miniChart', ticker],
    queryFn: () => api.chartOhlcv(ticker, 90),
    staleTime: 300_000,
  });

  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const bars = data?.data;
    if (!bars || bars.length < 2 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * dpr;
    canvas.height = rect.height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, rect.width, rect.height);

    if (bars.length < 2) return;

    const closes = bars.map(b => b.close);
    const minP = Math.min(...closes);
    const maxP = Math.max(...closes);
    const range = maxP - minP || 1;
    const padding = 4;
    const w = rect.width - padding * 2;
    const h = rect.height - padding * 2;

    // Gradient fill
    const isUp = closes[closes.length - 1] >= closes[0];
    const lineColor = isUp ? 'var(--accent-emerald)' : 'var(--accent-rose)';
    const gradient = ctx.createLinearGradient(0, padding, 0, rect.height);
    gradient.addColorStop(0, isUp ? 'rgba(52, 211, 153, 0.15)' : 'rgba(251, 113, 133, 0.15)');
    gradient.addColorStop(1, 'transparent');

    // Draw area fill
    ctx.beginPath();
    closes.forEach((c, i) => {
      const x = padding + (i / (closes.length - 1)) * w;
      const y = padding + h - ((c - minP) / range) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.lineTo(padding + w, padding + h);
    ctx.lineTo(padding, padding + h);
    ctx.closePath();
    ctx.fillStyle = gradient;
    ctx.fill();

    // Draw line
    ctx.beginPath();
    closes.forEach((c, i) => {
      const x = padding + (i / (closes.length - 1)) * w;
      const y = padding + h - ((c - minP) / range) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Current price dot
    const lastX = padding + w;
    const lastY = padding + h - ((closes[closes.length - 1] - minP) / range) * h;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3, 0, Math.PI * 2);
    ctx.fillStyle = lineColor;
    ctx.fill();

  }, [data]);

  const bars = data?.data || [];
  const lastPrice = bars.length > 0 ? bars[bars.length - 1].close : 0;
  const firstPrice = bars.length > 0 ? bars[0].close : 0;
  const changePct = firstPrice ? ((lastPrice - firstPrice) / firstPrice) * 100 : 0;
  const isUp = changePct >= 0;

  return (
    <div className="mini-chart-enter bg-[#0a0a1a]/60 border-t border-[var(--void-raised)]/30">
      <div className="flex items-center gap-4 px-4 py-2">
        {/* Chart area */}
        <div className="flex-1 h-[80px]">
          {isLoading ? (
            <div className="h-full flex items-center justify-center">
              <div className="w-4 h-4 border-2 border-[var(--void-raised)] border-t-[var(--accent-violet)] rounded-full animate-spin" />
            </div>
          ) : error ? (
            <div className="h-full flex items-center justify-center text-[10px] text-[var(--text-secondary)]">Chart unavailable</div>
          ) : (
            <canvas ref={canvasRef} className="w-full h-full" />
          )}
        </div>

        {/* Price info */}
        <div className="flex-shrink-0 text-right space-y-1">
          <p className="text-sm font-bold text-[#e2e8f0] tabular-nums">
            {lastPrice > 0 ? (lastPrice < 10 ? lastPrice.toFixed(4) : lastPrice.toFixed(2)) : '--'}
          </p>
          <p className={`text-xs font-semibold tabular-nums ${isUp ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]'}`}>
            {isUp ? '+' : ''}{changePct.toFixed(2)}%
          </p>
          <p className="text-[9px] text-[#6b7a90]">3M change</p>
        </div>

        {/* Navigate to full chart */}
        <button
          onClick={onNavigateChart}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-medium text-[var(--text-violet)] bg-[var(--accent-violet)]/10 hover:bg-[var(--accent-violet)]/20 transition-all"
        >
          <ExternalLink className="w-3 h-3" />
          Full Chart
        </button>
      </div>
    </div>
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

/* ── Badges & Helpers ────────────────────────────────────────────── */
function CountBadge({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <span className="inline-flex items-center gap-0.5 px-1.5 py-0.5 rounded text-[10px] font-medium"
      style={{ color, background: `${color}15` }}>
      {label}{value}
    </span>
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
  title, signals, color, isLoading, onNavigateChart, emaStates,
}: {
  title: string;
  signals: HighConvictionSignal[];
  color: 'green' | 'red';
  isLoading: boolean;
  onNavigateChart: (sym: string) => void;
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
  const accentMid = color === 'green' ? '#10b98160' : '#f43f5e60';

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
