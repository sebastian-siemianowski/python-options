/* eslint-disable @typescript-eslint/no-explicit-any */
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { SummaryRow, SignalSummaryData } from '../api';
import { SignalTableSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { ColumnCustomizer } from '../components/ColumnCustomizer';
import { useJobStore } from '../stores/jobStore';
import SignalOperationsBar from '../features/signals/components/SignalOperationsBar';
import SignalsHero from '../features/signals/components/SignalsHero';
import AllAssetsTable, { SECTOR_CHART_VIEW_LS_KEY, loadSectorChartView } from '../features/signals/components/AllAssetsTable';
import WatchlistView from '../features/signals/components/WatchlistView';
import HighConvictionTabs from '../features/signals/components/HighConvictionTabs';
import SegmentedControl from '../features/signals/components/SegmentedControl';
import SmaReversalsPanel from '../features/signals/components/SmaReversalsPanel';
import SectorPanels, { DEFAULT_SECTOR_VISIBLE_COLS, SECTOR_COLUMN_DEFS, SECTOR_COLS_LS_KEY, SECTOR_SORT_OPTIONS, loadSectorVisibleCols, type SectorSortBy } from '../features/signals/components/SectorPanels';
import StrongSignalsView from '../features/signals/components/StrongSignalsView';
import {
  extractTicker,
  isCurrencyAsset,
  isReversalQuickFilter,
  rebuildSectorFromAssets,
  rowHorizonColor,
  rowMatchesReversalFilter,
  type SignalFilter,
  type SortColumn,
  type SortDir,
} from '../features/signals/utils';
import {
  ChevronDown,
  TrendingDown, Search, X, BarChart3,
  ArrowUp, ArrowDown,
  Eye, ChevronUp, Loader2,
} from 'lucide-react';
import { formatHorizon, responsiveHorizons } from '../utils/horizons';

import { useWebSocket } from '../hooks/useWebSocket';

const SIGNALS_SHOW_CURRENCIES_LS_KEY = 'signals-show-currencies-v1';

const loadStoredShowCurrencies = (): boolean => {
  if (typeof window === 'undefined') return true;
  try {
    return window.localStorage.getItem(SIGNALS_SHOW_CURRENCIES_LS_KEY) !== '0';
  } catch {
    return true;
  }
};

type ViewMode = 'all' | 'sectors' | 'strong';

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
  const [showCurrencies, setShowCurrencies] = useState<boolean>(() => loadStoredShowCurrencies());
  useEffect(() => {
    try { localStorage.setItem(SIGNALS_SHOW_CURRENCIES_LS_KEY, showCurrencies ? '1' : '0'); } catch { /* ignore */ }
  }, [showCurrencies]);
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
  const [sectorChartView, setSectorChartView] = useState<boolean>(() => loadSectorChartView());
  useEffect(() => {
    try { localStorage.setItem(SECTOR_COLS_LS_KEY, JSON.stringify(Array.from(sectorVisibleCols))); } catch { /* ignore */ }
  }, [sectorVisibleCols]);
  useEffect(() => {
    try { localStorage.setItem(SECTOR_CHART_VIEW_LS_KEY, sectorChartView ? '1' : '0'); } catch { /* ignore */ }
  }, [sectorChartView]);
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
    queryKey: ['smaReversals', 'exclude-currencies-v2'],
    queryFn: api.smaReversals,
    staleTime: 60_000,
  });

  const reversalFlipsQ = useQuery({
    queryKey: ['reversalFlips', 4],
    queryFn: () => api.reversalFlips(4, 365),
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

  const passesCurrency = useCallback((assetLabelOrTicker: string | undefined | null): boolean => (
    showCurrencies || !isCurrencyAsset(assetLabelOrTicker)
  ), [showCurrencies]);

  const currencyVisibleRows = useMemo(() => (
    showCurrencies ? rows : rows.filter((row) => passesCurrency(row.asset_label))
  ), [rows, showCurrencies, passesCurrency]);
  const currencyAssetCount = useMemo(() => (
    rows.filter((row) => isCurrencyAsset(row.asset_label)).length
  ), [rows]);

  const filteredRows = useMemo(() => {
    return currencyVisibleRows.filter((row) => {
      if (debouncedSearch && !fuzzyMatch(row.asset_label, debouncedSearch)) return false;
      if (!passesEma(row.asset_label)) return false;
      if (filter === 'all') return true;
      if (isReversalQuickFilter(filter)) return rowMatchesReversalFilter(row, filter, reversalFlipsQ.data);
      const label = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
      if (filter === 'bullish') return label === 'STRONG_BUY' || label === 'BUY';
      if (filter === 'bearish') return label === 'STRONG_SELL' || label === 'SELL';
      if (filter === 'greens' || filter === 'reds') return rowHorizonColor(row) === filter;
      return label === filter.toUpperCase();
    });
  }, [currencyVisibleRows, debouncedSearch, filter, fuzzyMatch, passesEma, reversalFlipsQ.data]);

  // Sectors view: apply EMA predicate at the asset level, drop empty sectors.
  const sectors = useMemo(() => {
    if (showCurrencies && !emaFilters.p9 && !emaFilters.p50 && !emaFilters.p600) return rawSectors;
    return rawSectors
      .map(sec => rebuildSectorFromAssets(
        sec,
        sec.assets.filter(a => passesCurrency(a.asset_label) && passesEma(a.asset_label)),
      ))
      .filter(sec => sec.assets.length > 0);
  }, [rawSectors, showCurrencies, emaFilters, passesCurrency, passesEma]);

  // Global sector totals shown in the unified filter card footer.
  const sectorTotals = useMemo(() => ({
    assets: sectors.reduce((s, sec) => s + sec.asset_count, 0),
    bullish: sectors.reduce((s, sec) => s + (sec.strong_buy ?? 0) + (sec.buy ?? 0), 0),
    bearish: sectors.reduce((s, sec) => s + (sec.strong_sell ?? 0) + (sec.sell ?? 0), 0),
  }), [sectors]);

  const reversalQuickCounts = useMemo(() => ({
    buy: currencyVisibleRows.filter((row) => rowMatchesReversalFilter(row, 'reversal_buy', reversalFlipsQ.data)).length,
    sell: currencyVisibleRows.filter((row) => rowMatchesReversalFilter(row, 'reversal_sell', reversalFlipsQ.data)).length,
  }), [currencyVisibleRows, reversalFlipsQ.data]);

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
        case 'quality': return (qualityScores[extractTicker(a.asset_label)] ?? 50) - (qualityScores[extractTicker(b.asset_label)] ?? 50);
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
  }, [filteredRows, sortLevels, qualityScores]);

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

  const openOrStartJob = (mode: 'stocks' | 'retune' | 'tune-stocks') => {
    const currentRunBothIsRunning = isJobRunning && activeJobMode === 'tune-stocks';
    const openDetailedProgress = mode !== 'tune-stocks' && !currentRunBothIsRunning;
    if (isJobRunning) {
      showJobSurface();
      setJobExpanded(openDetailedProgress);
    } else {
      startJob(mode);
      showJobSurface();
      setJobExpanded(openDetailedProgress);
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
        totalRows={currencyVisibleRows.length}
        onRefreshStocks={() => openOrStartJob('stocks')}
        onRunTune={() => openOrStartJob('retune')}
        onRunBoth={() => openOrStartJob('tune-stocks')}
        onViewProgress={() => {
          showJobSurface();
          setJobExpanded(true);
        }}
        onStop={stopJob}
      />

      {/* ── v1 PREMIUM HERO BAND ─────────────────────────────────────── */}
      <SignalsHero stats={showCurrencies ? stats : undefined} rows={currencyVisibleRows} horizons={horizons} filteredCount={filteredRows.length} wsStatus={wsStatus} />

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
          reversalFlips={reversalFlipsQ.data}
          reversalFlipsLoading={reversalFlipsQ.isLoading}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
        />
      </div>

      {/* High Conviction — tabbed decision workspace */}
      <HighConvictionTabs
        buySignals={(buyQ.data?.signals || []).filter((signal) => passesCurrency(signal.ticker))}
        sellSignals={(sellQ.data?.signals || []).filter((signal) => passesCurrency(signal.ticker))}
        buyLoading={buyQ.isLoading}
        sellLoading={sellQ.isLoading}
        emaStates={emaStates}
      />

      {/* SMA Reversals — world-class crossover detection (9 / 50 / 600) */}
      <SmaReversalsPanel
        data={reversalsQ.data}
        isLoading={reversalsQ.isLoading}
        rows={rows}
        qualityScores={qualityScores}
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
                  {filteredRows.length}/{currencyVisibleRows.length}
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

          {/* Recent reversal chips — BUY/SELL flips in the last four trading bars */}
          <div className="flex items-center gap-1.5">
            <span className="text-[9.5px] font-semibold uppercase tracking-[0.14em] text-[var(--text-muted)] pr-1">
              Reversal
            </span>
            {([
              { key: 'reversal_buy' as SignalFilter, label: 'Buy Reversal', count: reversalQuickCounts.buy, accent: '#00f5a0', icon: <ArrowUp className="w-3 h-3" /> },
              { key: 'reversal_sell' as SignalFilter, label: 'Sell Reversal', count: reversalQuickCounts.sell, accent: '#ff375f', icon: <ArrowDown className="w-3 h-3" /> },
            ]).map(({ key, label, count, accent, icon }) => {
              const on = filter === key;
              const loaded = !!reversalFlipsQ.data && !reversalFlipsQ.isError;
              return (
                <button
                  key={key}
                  type="button"
                  onClick={() => setFilter(on ? 'all' : key)}
                  aria-pressed={on}
                  title={loaded ? `${label}: flips in the latest 4 trading bars` : 'Loading reversal flips…'}
                  className="group relative inline-flex items-center gap-1.5 rounded-lg pl-2 pr-1.5 py-1 transition-all duration-200"
                  style={{
                    background: on
                      ? `linear-gradient(180deg, ${accent}30, ${accent}12)`
                      : 'rgba(255,255,255,0.02)',
                    border: `1px solid ${on ? accent + '78' : 'rgba(255,255,255,0.05)'}`,
                    boxShadow: on
                      ? `0 0 0 1px ${accent}22 inset, 0 4px 14px -6px ${accent}90, 0 0 18px -5px ${accent}70`
                      : '0 1px 0 rgba(255,255,255,0.02) inset',
                    color: on ? '#fff' : 'var(--text-secondary)',
                    opacity: reversalFlipsQ.isLoading ? 0.72 : 1,
                    transition: 'all 220ms cubic-bezier(.2,.8,.2,1)',
                  }}
                >
                  <span style={{ color: on ? accent : 'var(--text-muted)', filter: on ? `drop-shadow(0 0 4px ${accent})` : 'none' }}>
                    {reversalFlipsQ.isLoading ? <Loader2 className="w-3 h-3 animate-spin" /> : icon}
                  </span>
                  <span className="text-[10.5px] font-semibold tracking-wide whitespace-nowrap" style={{ color: on ? '#fff' : 'var(--text-secondary)' }}>
                    {label}
                  </span>
                  <span
                    className="inline-flex items-center justify-center rounded-md px-1 min-w-[18px] h-[15px] text-[9.5px] font-semibold tabular-nums transition-all"
                    style={{
                      background: on ? accent : 'rgba(255,255,255,0.05)',
                      color: on ? '#07110d' : 'var(--text-muted)',
                      boxShadow: on ? '0 1px 0 rgba(255,255,255,0.2) inset' : 'none',
                    }}
                  >
                    {loaded ? count : '—'}
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

          <div className="h-5 w-px bg-white/[0.05]" aria-hidden />

          <button
            type="button"
            onClick={() => setShowCurrencies((prev) => !prev)}
            aria-pressed={showCurrencies}
            title={showCurrencies ? 'Hide FX currency pairs from Signals views' : 'Show FX currency pairs in Signals views'}
            className="inline-flex items-center gap-1.5 rounded-lg pl-2 pr-1.5 py-1 transition-all duration-200"
            style={{
              background: showCurrencies
                ? 'linear-gradient(180deg, rgba(56,217,245,0.18), rgba(56,217,245,0.07))'
                : 'rgba(255,255,255,0.02)',
              border: `1px solid ${showCurrencies ? 'rgba(56,217,245,0.38)' : 'rgba(255,255,255,0.05)'}`,
              color: showCurrencies ? '#bae6fd' : 'var(--text-secondary)',
              boxShadow: showCurrencies ? '0 0 14px -9px rgba(56,217,245,0.95)' : '0 1px 0 rgba(255,255,255,0.02) inset',
            }}
          >
            <Eye className="w-3 h-3" style={{ color: showCurrencies ? '#67e8f9' : 'var(--text-muted)' }} />
            <span className="text-[10.5px] font-semibold tracking-wide whitespace-nowrap">
              FX {showCurrencies ? 'On' : 'Off'}
            </span>
            <span
              className="inline-flex items-center justify-center rounded-md px-1 min-w-[18px] h-[15px] text-[9.5px] font-semibold tabular-nums"
              style={{
                background: showCurrencies ? 'rgba(56,217,245,0.24)' : 'rgba(255,255,255,0.05)',
                color: showCurrencies ? '#cffafe' : 'var(--text-muted)',
              }}
            >
              {currencyAssetCount}
            </span>
          </button>

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
          {(filter !== 'all' || emaFilters.p9 || emaFilters.p50 || emaFilters.p600 || !showCurrencies || search) && (
            <button
              type="button"
              onClick={() => {
                setFilter('all');
                setEmaFilters({ p9: false, p50: false, p600: false });
                setShowCurrencies(true);
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
                type="button"
                aria-pressed={sectorChartView}
                onClick={() => setSectorChartView((prev) => !prev)}
                className="inline-flex items-center gap-1.5 rounded-md px-2 py-1 text-[10.5px] font-medium transition-all duration-[160ms]"
                style={{
                  color: sectorChartView ? '#c4b5fd' : 'var(--text-secondary)',
                  background: sectorChartView ? 'rgba(167,139,250,0.10)' : 'rgba(255,255,255,0.02)',
                  border: `1px solid ${sectorChartView ? 'rgba(167,139,250,0.30)' : 'rgba(255,255,255,0.05)'}`,
                  boxShadow: sectorChartView ? '0 0 14px -10px rgba(167,139,250,0.95)' : 'none',
                }}
                title={sectorChartView ? 'Show sector tables' : 'Show chart-first rows inside each sector'}
              >
                <BarChart3 className="w-3 h-3" />
                Chart view
              </button>
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
              {!sectorChartView && (
                <ColumnCustomizer
                  columns={SECTOR_COLUMN_DEFS}
                  visible={sectorVisibleCols}
                  onToggle={toggleSectorCol}
                  onReset={resetSectorCols}
                />
              )}
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
          sectorChartView={sectorChartView}
          horizons={horizons}
          search={debouncedSearch}
          filter={filter}
          reversalFlips={reversalFlipsQ.data}
          updatedAsset={updatedAsset}
          qualityScores={qualityScores}
        />
      )}
      {view === 'strong' && (
        <StrongSignalsView
          strongBuy={(strongQ.data?.strong_buy || []).filter(s => passesCurrency(s.symbol) && passesEma(s.symbol))}
          strongSell={(strongQ.data?.strong_sell || []).filter(s => passesCurrency(s.symbol) && passesEma(s.symbol))}
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
          disablePagination
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
