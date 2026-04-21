import React from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { SummaryRow, SectorGroup, StrongSignalEntry, HighConvictionSignal, SignalSummaryData } from '../api';
import PageHeader from '../components/PageHeader';
import { SignalTableSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { SignalsEmpty } from '../components/CosmicEmptyState';
import { ExportButton } from '../components/ExportButton';
import { Sparkline } from '../components/Sparkline';
import { SignalStrengthBar, MomentumBadge, CrashRiskHeat, HorizonCell } from '../components/SignalTableVisuals';
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

type ViewMode = 'all' | 'sectors' | 'strong';
type SignalFilter = 'all' | 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';

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
  const [view, setView] = useState<ViewMode>('sectors');
  const [filter, setFilter] = useState<SignalFilter>('all');
  const [search, setSearch] = useState('');
  const [debouncedSearch, setDebouncedSearch] = useState('');
  const [expandedSectors, setExpandedSectors] = useState<Set<string>>(new Set());

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
  const sectors = sectorQ.data?.sectors || [];

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

  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (debouncedSearch && !fuzzyMatch(row.asset_label, debouncedSearch)) return false;
      if (filter === 'all') return true;
      const label = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
      return label === filter.toUpperCase();
    });
  }, [rows, debouncedSearch, filter, fuzzyMatch]);

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
      <PageHeader title="Signals">
        <span>{rows.length} assets across {horizons.length} horizons</span>
        <WsStatusDot status={wsStatus} />
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
      </PageHeader>

      {/* Stats bar — premium glassmorphic cards */}
      {stats && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6 fade-up">
          <StatCard label="Strong Buy" value={stats.strong_buy_signals} total={stats.total_assets} color="#10b981" icon={<ArrowUpCircle className="w-4 h-4" />} />
          <StatCard label="Buy" value={stats.buy_signals - stats.strong_buy_signals} total={stats.total_assets} color="#6ff0c0" icon={<ArrowUp className="w-4 h-4" />} />
          <StatCard label="Hold" value={stats.hold_signals} total={stats.total_assets} color="#7a8ba4" icon={<Activity className="w-4 h-4" />} />
          <StatCard label="Sell" value={stats.sell_signals - stats.strong_sell_signals} total={stats.total_assets} color="#f87171" icon={<ArrowDown className="w-4 h-4" />} />
          <StatCard label="Strong Sell" value={stats.strong_sell_signals} total={stats.total_assets} color="#f43f5e" icon={<ArrowDownCircle className="w-4 h-4" />} />
          <StatCard label="Exit" value={stats.exit_signals} total={stats.total_assets} color="#f59e0b" icon={<X className="w-4 h-4" />} />
        </div>
      )}

      {/* High Conviction Panels — full positions with rich data */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-5 mb-8 fade-up-delay-1">
        <HighConvictionPanel
          title="High Conviction BUY"
          signals={buyQ.data?.signals || []}
          color="green"
          isLoading={buyQ.isLoading}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
        />
        <HighConvictionPanel
          title="High Conviction SELL"
          signals={sellQ.data?.signals || []}
          color="red"
          isLoading={sellQ.isLoading}
          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
        />
      </div>

      {/* View mode + filters */}
      <div className="flex flex-wrap items-center gap-3 mb-5 fade-up-delay-2">
        {/* View toggle */}
        <div className="flex items-center gap-0.5 glass-card px-2.5 py-1.5">
          {([
            { key: 'sectors' as ViewMode, label: 'By Sector' },
            { key: 'strong' as ViewMode, label: 'Strong Signals' },
            { key: 'all' as ViewMode, label: 'All Assets' },
          ]).map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setView(key)}
              className={`px-3 py-1.5 rounded-lg text-[12px] font-medium transition-all duration-200 ${
                view === key ? 'bg-[var(--accent-violet)]/15 text-[var(--text-violet)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-secondary)] hover:bg-white/[0.02]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Signal filter */}
        {view !== 'strong' && (
          <div className="flex items-center gap-2 glass-card px-2.5 py-1.5">
            <Filter className="w-3 h-3 text-[var(--text-secondary)] mr-1.5" />
            {([
              { key: 'all' as SignalFilter, label: 'All' },
              { key: 'strong_buy' as SignalFilter, label: 'SB' },
              { key: 'buy' as SignalFilter, label: 'Buy' },
              { key: 'hold' as SignalFilter, label: 'Hold' },
              { key: 'sell' as SignalFilter, label: 'Sell' },
              { key: 'strong_sell' as SignalFilter, label: 'SS' },
            ]).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className="filter-pill"
                data-active={filter === key}
                data-filter={key}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* Smart Search with premium focus elevation */}
        <div className="flex items-center gap-2 glass-card px-3 py-2.5 group search-cosmic focus-ring transition-all duration-200" style={{ backdropFilter: 'blur(8px)' }}>
          <Search className="w-3.5 h-3.5 text-[var(--text-muted)] group-focus-within:text-[var(--accent-violet)] transition-colors" />
          <input
            ref={searchRef}
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets..."
            className="bg-transparent text-[13px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none w-40"
          />
          {!search && (
            <span className="text-[9px] text-[var(--text-muted)] border border-[var(--border-void)] rounded px-1 py-0.5 opacity-50">/</span>
          )}
          {search && debouncedSearch !== search && (
            <span className="w-1.5 h-1.5 rounded-full bg-[var(--accent-violet)] animate-pulse" />
          )}
          {search && (
            <>
              <span className="text-[10px] text-[var(--text-muted)] tabular-nums whitespace-nowrap">{filteredRows.length} of {rows.length}</span>
              <button onClick={() => setSearch('')} className="text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors duration-120">
                <X className="w-3 h-3" />
              </button>
            </>
          )}
        </div>

        <span className="text-xs text-[var(--text-muted)]">
          {view === 'sectors' ? `${sectors.length} sectors` : `${filteredRows.length} results`}
        </span>

        {/* Story 3.4: Change counter badge */}
        {changeLog.length > 0 && (
          <button
            onClick={() => {
              const lastChange = changeLog[0];
              if (lastChange) {
                const el = document.querySelector(`[data-ticker="${lastChange.asset}"]`);
                el?.scrollIntoView({ behavior: 'smooth', block: 'center' });
              }
            }}
            className="text-[10px] px-2 py-1 rounded-full animate-pulse"
            style={{ color: 'var(--accent-violet)', background: 'var(--violet-12)' }}
          >
            {changeLog.length} change{changeLog.length > 1 ? 's' : ''}
          </button>
        )}

        {/* Story 3.4: Live Feed toggle */}
        <button
          onClick={() => setShowTickerTape(p => !p)}
          className={`text-[10px] px-2 py-1 rounded transition-colors ${showTickerTape ? 'text-[var(--accent-violet)] bg-[var(--violet-12)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'}`}
        >
          Live Feed
        </button>

        {view === 'sectors' && (
          <div className="flex gap-2 ml-auto">
            <button onClick={expandAll} className="text-[10px] text-[var(--accent-violet)] hover:underline">Expand All</button>
            <button onClick={collapseAll} className="text-[10px] text-[var(--text-muted)] hover:underline">Collapse All</button>
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

      {/* Story 3.6: Horizon pill selector */}
      {view === 'all' && allHorizons.length > 0 && (
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
        />
      )}
      {view === 'strong' && (
        <StrongSignalsView
          strongBuy={strongQ.data?.strong_buy || []}
          strongSell={strongQ.data?.strong_sell || []}
        />
      )}
      {view === 'all' && (
        <AllAssetsTable
          rows={sortedRows} horizons={horizons}
          updatedAsset={updatedAsset}
          sortLevels={sortLevels} onSort={handleSort} onRemoveSort={removeSortLevel}
          expandedRow={expandedRow} onExpandRow={setExpandedRow}
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

function SectorPanels({
  sectors,
  expandedSectors,
  toggleSector,
  horizons,
  search,
  filter,
  updatedAsset,
}: {
  sectors: SectorGroup[];
  expandedSectors: Set<string>;
  toggleSector: (name: string) => void;
  horizons: number[];
  search: string;
  filter: SignalFilter;
  updatedAsset: string | null;
}) {
  const [sectorSort, setSectorSort] = useState<SectorSortBy>('momentum');
  const navigate = useNavigate();

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
      </div>

      {sorted.map((sector) => {
        const expanded = expandedSectors.has(sector.name);
        const assets = sector.assets.filter(row => {
          if (search && !row.asset_label.toLowerCase().includes(search.toLowerCase())) return false;
          if (filter === 'all') return true;
          const lbl = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
          return lbl === filter.toUpperCase();
        });
        if (search && assets.length === 0) return null;

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
              <div style={{ animation: 'slide-down 200ms cubic-bezier(0.2,0,0,1) both' }}>
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
                        <th className="text-left px-4 py-2.5 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[180px]">Asset</th>
                        <th className="text-center px-1 py-2.5 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[65px]">Chart</th>
                        <th className="text-center px-2 py-2.5 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[90px]">Signal</th>
                        <th className="text-center px-2 py-2.5 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[70px]">Mom</th>
                        {horizons.map(h => (
                          <th key={h} className="text-center px-2 py-2.5 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider">{formatHorizon(h)}</th>
                        ))}
                        <th className="text-center px-2 py-2.5 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[60px]">Risk</th>
                        <th className="w-8"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {assets.map((row, i) => (
                        <SectorSignalRow
                          key={row.asset_label}
                          row={row}
                          horizons={horizons}
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
function StrongSignalPanel({ entries, accent, label, icon }: {
  entries: StrongSignalEntry[]; accent: string; label: string; icon: React.ReactNode;
}) {
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
            return (
              <div key={i} className="flex items-center gap-3 px-5 py-2.5 hover:bg-white/[0.015] transition-colors">
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
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function StrongSignalsView({ strongBuy, strongSell }: { strongBuy: StrongSignalEntry[]; strongSell: StrongSignalEntry[] }) {
  return (
    <div className="grid grid-cols-1 lg:grid-cols-2 gap-5">
      <StrongSignalPanel
        entries={strongBuy}
        accent="#10b981"
        label="Strong Buy Signals"
        icon={<TrendingUp className="w-4 h-4" style={{ color: '#10b981' }} />}
      />
      <StrongSignalPanel
        entries={strongSell}
        accent="#f43f5e"
        label="Strong Sell Signals"
        icon={<TrendingDown className="w-4 h-4" style={{ color: '#f43f5e' }} />}
      />
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

function AllAssetsTable({ rows, horizons, updatedAsset, sortLevels, onSort, onRemoveSort, expandedRow, onExpandRow, onNavigateChart }: {
  rows: SummaryRow[]; horizons: number[]; updatedAsset: string | null;
  sortLevels: { col: SortColumn; dir: SortDir }[];
  onSort: (col: SortColumn, shiftKey: boolean) => void;
  onRemoveSort: (col: SortColumn) => void;
  expandedRow: string | null; onExpandRow: (label: string | null) => void;
  onNavigateChart: (symbol: string) => void;
}) {
  const [page, setPage] = useState(0);
  const [scrolled, setScrolled] = useState(false);
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const pageSize = 50;
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
              <th className="text-center px-2 py-3 w-[60px]">
                <span className="text-[10px] text-[var(--text-violet)] uppercase tracking-[0.06em] font-medium">30D</span>
              </th>
              <th className={`text-left px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'sector') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'sector') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('sector', e.shiftKey)}>
                Sector <SortIndicator col="sector" sortLevels={sortLevels} />
              </th>
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'signal') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'signal') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('signal', e.shiftKey)}>
                Signal <SortIndicator col="signal" sortLevels={sortLevels} />
              </th>
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'momentum') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'momentum') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('momentum', e.shiftKey)}>
                Mom <SortIndicator col="momentum" sortLevels={sortLevels} />
              </th>
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'crash_risk') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'crash_risk') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('crash_risk', e.shiftKey)}>
                Risk <SortIndicator col="crash_risk" sortLevels={sortLevels} />
              </th>
              {horizons.map((h) => {
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
function CosmicSignalRow({ row, ticker, horizons, highlighted, isExpanded, onToggleExpand, onNavigateChart }: {
  row: SummaryRow; ticker: string; horizons: number[];
  highlighted?: boolean; isExpanded: boolean;
  onToggleExpand: () => void; onNavigateChart: () => void;
}) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  // Compute composite for strength bar
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  const pUp = nearestHorizon?.p_up;
  const kellyVal = nearestHorizon?.kelly_half;

  return (
    <>
      <tr className={`cosmic-row border-b border-[var(--border-void)] ${highlighted ? 'aurora-upgrade' : ''} ${isExpanded ? 'bg-[var(--void-hover)]' : ''}`}>
        {/* Asset */}
        <td className="px-4 py-2 whitespace-nowrap">
          <button onClick={onToggleExpand} className="text-left group/asset">
            <span className="font-semibold text-[var(--text-primary)] text-xs group-hover/asset:text-[var(--accent-violet)] transition-colors">
              {ticker}
            </span>
            {row.asset_label.includes('(') && (
              <span className="block text-[9px] text-[var(--text-muted)] truncate max-w-[140px] leading-tight">
                {row.asset_label.split('(')[0].trim()}
              </span>
            )}
          </button>
        </td>
        {/* AC-1: Sparkline */}
        <td className="px-1 py-2 text-center">
          <Sparkline ticker={ticker} width={60} height={28} />
        </td>
        {/* Sector */}
        <td className="px-3 py-2 text-[10px] text-[var(--text-secondary)] max-w-[100px] truncate">{row.sector}</td>
        {/* AC-2: Signal with strength bar */}
        <td className="px-3 py-2 text-center">
          <SignalStrengthBar label={label} pUp={pUp} kelly={kellyVal} />
        </td>
        {/* AC-3: Momentum badge */}
        <td className="px-3 py-2 text-center">
          <MomentumBadge value={row.momentum_score} />
        </td>
        {/* AC-4: Crash risk heat */}
        <td className="px-3 py-2">
          <div className="flex justify-center">
            <CrashRiskHeat score={row.crash_risk_score} />
          </div>
        </td>
        {/* AC-5: Horizon cells */}
        {horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          return (
            <td key={h} className="px-2 py-2 text-center">
              <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
            </td>
          );
        })}
        {/* Expand button */}
        <td className="px-2 py-2">
          <button onClick={onToggleExpand} className="p-1 rounded hover:bg-[var(--void-active)] transition-colors" title="Show chart">
            <BarChart3 className={`w-3.5 h-3.5 transition-colors ${isExpanded ? 'text-[var(--accent-violet)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'}`} />
          </button>
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={horizons.length + 7} className="p-0">
            <MiniChartPanel ticker={ticker} onNavigateChart={onNavigateChart} />
          </td>
        </tr>
      )}
    </>
  );
}

/* ── Mini chart panel (expandable inline) ────────────────────────── */
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
function SectorSignalRow({ row, horizons, highlighted, delayMs = 0, onNavigateChart }: {
  row: SummaryRow; horizons: number[]; highlighted?: boolean; delayMs?: number;
  onNavigateChart: (sym: string) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  const ticker = extractTicker(row.asset_label);
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  const labelColor = signalLabelColor(label);

  return (
    <>
      <tr className={`border-b border-white/[0.03] hover:bg-white/[0.015] transition-all duration-150 ${highlighted ? 'aurora-upgrade' : ''} ${isExpanded ? 'bg-white/[0.02]' : ''}`}
        style={{ animationDelay: `${delayMs}ms` }}>
        {/* Asset */}
        <td className="px-4 py-2.5 whitespace-nowrap">
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
                <span className="text-[9px] text-[var(--text-muted)] truncate max-w-[150px] leading-tight block mt-0.5">
                  {row.asset_label.split('(')[0].trim()}
                </span>
              )}
            </div>
          </div>
        </td>
        {/* Sparkline */}
        <td className="px-1 py-2.5 text-center">
          <Sparkline ticker={ticker} width={60} height={26} />
        </td>
        {/* Signal */}
        <td className="px-2 py-2.5 text-center">
          <SignalStrengthBar label={label} pUp={nearestHorizon?.p_up} kelly={nearestHorizon?.kelly_half} />
        </td>
        {/* Momentum */}
        <td className="px-2 py-2.5 text-center">
          <MomentumBadge value={row.momentum_score} />
        </td>
        {/* Horizon cells */}
        {horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          return (
            <td key={h} className="px-2 py-2.5 text-center">
              <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
            </td>
          );
        })}
        {/* Risk */}
        <td className="px-2 py-2.5">
          <div className="flex justify-center">
            <CrashRiskHeat score={row.crash_risk_score} />
          </div>
        </td>
        {/* Actions */}
        <td className="px-1 py-2.5">
          <button
            onClick={() => setIsExpanded(p => !p)}
            className="p-1 rounded hover:bg-white/[0.05] transition-colors"
            title="Expand"
          >
            <BarChart3 className={`w-3.5 h-3.5 transition-colors ${isExpanded ? 'text-[var(--accent-violet)]' : 'text-[var(--text-muted)]'}`} />
          </button>
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={horizons.length + 5} className="p-0">
            <MiniChartPanel ticker={ticker} onNavigateChart={() => onNavigateChart(ticker)} />
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

function HighConvictionPanel({
  title, signals, color, isLoading, onNavigateChart,
}: {
  title: string;
  signals: HighConvictionSignal[];
  color: 'green' | 'red';
  isLoading: boolean;
  onNavigateChart: (sym: string) => void;
}) {
  const [sortCol, setSortCol] = useState<HCSortCol>('exp_ret');
  const [sortDir, setSortDir] = useState<HCSortDir>('desc');
  const [expandedTicker, setExpandedTicker] = useState<string | null>(null);
  const [searchTerm, setSearchTerm] = useState('');

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

  // Filter
  const filtered = useMemo(() => {
    if (!searchTerm) return grouped;
    const q = searchTerm.toLowerCase();
    return grouped.filter(g => g.ticker.toLowerCase().includes(q) || g.asset_label.toLowerCase().includes(q) || g.sector.toLowerCase().includes(q));
  }, [grouped, searchTerm]);

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
      style={{ color: sortCol === col ? accent : 'var(--text-muted)', width: w }}
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
    <div className="glass-card overflow-hidden" style={{ borderTop: `2px solid ${accent}40` }}>
      {/* Header */}
      <div className="px-5 py-4" style={{ background: `linear-gradient(135deg, ${accentSoft} 0%, transparent 60%)` }}>
        <div className="flex items-center gap-3 mb-3">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center" style={{ background: accentSoft, boxShadow: `0 0 20px ${accentSoft}` }}>
            <Icon className="w-5 h-5" style={{ color: accent }} />
          </div>
          <div>
            <h3 className="text-sm font-semibold" style={{ color: accent }}>{title}</h3>
            <p className="text-[10px] text-[var(--text-muted)]">{uniqueTickers} positions across {totalSignals} signals</p>
          </div>
        </div>
        {/* Summary stats strip */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <Layers className="w-3 h-3" style={{ color: accentMid }} />
            <span className="text-[10px] text-[var(--text-secondary)]">Positions</span>
            <span className="text-[11px] font-bold tabular-nums" style={{ color: accent }}>{uniqueTickers}</span>
          </div>
          <div className="flex items-center gap-1.5">
            <DollarSign className="w-3 h-3" style={{ color: accentMid }} />
            <span className="text-[10px] text-[var(--text-secondary)]">Avg Return</span>
            <span className="text-[11px] font-bold tabular-nums" style={{ color: accent }}>{avgReturn.toFixed(1)}%</span>
          </div>
          <div className="flex items-center gap-1.5">
            <Target className="w-3 h-3" style={{ color: accentMid }} />
            <span className="text-[10px] text-[var(--text-secondary)]">Avg P(up)</span>
            <span className="text-[11px] font-bold tabular-nums" style={{ color: accent }}>{(avgProb * 100).toFixed(0)}%</span>
          </div>
          {/* Search */}
          <div className="ml-auto flex items-center gap-1 px-2 py-1 rounded-lg" style={{ background: 'var(--void-active)' }}>
            <Search className="w-3 h-3 text-[var(--text-muted)]" />
            <input
              type="text"
              value={searchTerm}
              onChange={e => setSearchTerm(e.target.value)}
              placeholder="Filter..."
              className="bg-transparent text-[11px] text-[var(--text-primary)] placeholder:text-[var(--text-muted)] outline-none w-20"
            />
          </div>
        </div>
      </div>

      {/* Loading state */}
      {isLoading ? (
        <div className="px-5 py-10 flex items-center justify-center gap-2">
          <div className="w-4 h-4 border-2 border-[var(--void-raised)] rounded-full animate-spin" style={{ borderTopColor: accent }} />
          <span className="text-[11px] text-[var(--text-muted)]">Loading signals...</span>
        </div>
      ) : sorted.length === 0 ? (
        <div className="px-5 py-10 text-center">
          <Shield className="w-8 h-8 mx-auto mb-2" style={{ color: `${accent}40` }} />
          <p className="text-xs text-[var(--text-muted)]">{searchTerm ? 'No matching signals' : 'No active signals'}</p>
        </div>
      ) : (
        <div className="overflow-x-auto" style={{ maxHeight: '420px', overflowY: 'auto' }}>
          <table className="w-full">
            <thead className="sticky top-0 z-10" style={{ background: 'var(--void-base)' }}>
              <tr style={{ borderBottom: `1px solid ${accentSoft}` }}>
                <SortHeader col="ticker" label="Asset" w="140px" />
                <SortHeader col="sector" label="Sector" />
                <th className="px-2 py-2 text-left text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]">Horizons</th>
                <SortHeader col="exp_ret" label="Best Return" />
                <SortHeader col="p_up" label="Avg P(up)" />
                <SortHeader col="strength" label="Strength" />
                <th className="px-2 py-2 text-[10px] font-semibold uppercase tracking-wider text-[var(--text-muted)]"></th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((g) => {
                const isExpanded = expandedTicker === g.ticker;
                const companyName = g.asset_label.includes('(') ? g.asset_label.split('(')[0].trim() : '';
                return (
                  <React.Fragment key={g.ticker}>
                    <tr
                      className="border-b border-white/[0.03] hover:bg-white/[0.02] cursor-pointer transition-colors"
                      onClick={() => setExpandedTicker(isExpanded ? null : g.ticker)}
                    >
                      {/* Asset */}
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-2">
                          <div className="w-1.5 h-8 rounded-full" style={{ background: `linear-gradient(to bottom, ${accent}, ${accent}30)` }} />
                          <div>
                            <span className="text-xs font-bold text-[#e2e8f0]">{g.ticker}</span>
                            {companyName && <p className="text-[9px] text-[var(--text-muted)] truncate max-w-[110px] leading-tight">{companyName}</p>}
                          </div>
                        </div>
                      </td>
                      {/* Sector */}
                      <td className="px-2 py-2.5">
                        <span className="text-[10px] px-2 py-0.5 rounded-full" style={{ color: accent, background: accentSoft }}>
                          {g.sector}
                        </span>
                      </td>
                      {/* Horizons pills */}
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-1">
                          {g.signals.map((s, i) => (
                            <span key={i} className="text-[9px] px-1.5 py-0.5 rounded font-medium tabular-nums"
                              style={{ background: 'var(--void-active)', color: 'var(--text-secondary)' }}>
                              {formatHorizon(s.horizon_days)}
                            </span>
                          ))}
                        </div>
                      </td>
                      {/* Best Return */}
                      <td className="px-2 py-2.5 text-right">
                        <span className="text-xs font-bold tabular-nums" style={{ color: accent }}>
                          {color === 'green' ? '+' : ''}{g.bestReturn.toFixed(1)}%
                        </span>
                      </td>
                      {/* Avg P(up) */}
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-1.5">
                          <div className="w-12 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${g.avgPUp * 100}%`, background: accent }} />
                          </div>
                          <span className="text-[10px] font-medium tabular-nums text-[var(--text-secondary)]">{(g.avgPUp * 100).toFixed(0)}%</span>
                        </div>
                      </td>
                      {/* Strength */}
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-1.5">
                          <div className="w-10 h-1.5 rounded-full bg-white/[0.06] overflow-hidden">
                            <div className="h-full rounded-full" style={{ width: `${Math.min(g.maxStrength * 200, 100)}%`, background: accent }} />
                          </div>
                          <span className="text-[10px] tabular-nums text-[var(--text-muted)]">{g.maxStrength.toFixed(2)}</span>
                        </div>
                      </td>
                      {/* Expand/Chart */}
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-1">
                          <button
                            onClick={(e) => { e.stopPropagation(); onNavigateChart(g.ticker); }}
                            className="p-1 rounded hover:bg-white/[0.05] transition-colors"
                            title="Open chart"
                          >
                            <BarChart3 className="w-3.5 h-3.5 text-[var(--text-muted)] hover:text-[var(--accent-violet)]" />
                          </button>
                          {isExpanded ? <ChevronUp className="w-3.5 h-3.5" style={{ color: accent }} /> : <ChevronDown className="w-3.5 h-3.5 text-[var(--text-muted)]" />}
                        </div>
                      </td>
                    </tr>
                    {/* Expanded detail */}
                    {isExpanded && (
                      <tr>
                        <td colSpan={7} style={{ background: `${accentSoft}` }}>
                          <div className="px-4 py-3">
                            <div className="flex items-center gap-2 mb-2">
                              <Eye className="w-3 h-3" style={{ color: accentMid }} />
                              <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: accent }}>Signal Details by Horizon</span>
                            </div>
                            <div className="grid gap-2" style={{ gridTemplateColumns: `repeat(${Math.min(g.signals.length, 4)}, 1fr)` }}>
                              {g.signals.map((s, i) => {
                                const strength = (s as Record<string, unknown>).signal_strength as number ?? 0;
                                const conviction = (s as Record<string, unknown>).conviction_probability as number ?? s.probability_up;
                                const profitPln = s.expected_profit_pln;
                                return (
                                  <div key={i} className="rounded-lg p-3" style={{ background: 'var(--void-base)', border: `1px solid ${accentSoft}` }}>
                                    <div className="flex items-center justify-between mb-2">
                                      <span className="text-xs font-bold" style={{ color: accent }}>{formatHorizon(s.horizon_days)}</span>
                                      <span className="text-[9px] px-1.5 py-0.5 rounded-full font-medium" style={{ background: accentSoft, color: accent }}>
                                        {s.signal_type || 'STRONG'}
                                      </span>
                                    </div>
                                    <div className="space-y-1.5">
                                      <div className="flex items-center justify-between">
                                        <span className="text-[9px] text-[var(--text-muted)]">Expected Return</span>
                                        <span className="text-[11px] font-bold tabular-nums" style={{ color: accent }}>
                                          {s.expected_return_pct != null ? `${s.expected_return_pct >= 0 ? '+' : ''}${s.expected_return_pct.toFixed(2)}%` : '--'}
                                        </span>
                                      </div>
                                      <div className="flex items-center justify-between">
                                        <span className="text-[9px] text-[var(--text-muted)]">P(up)</span>
                                        <span className="text-[10px] font-medium tabular-nums text-[var(--text-secondary)]">{(s.probability_up * 100).toFixed(1)}%</span>
                                      </div>
                                      <div className="flex items-center justify-between">
                                        <span className="text-[9px] text-[var(--text-muted)]">P(down)</span>
                                        <span className="text-[10px] font-medium tabular-nums text-[var(--text-secondary)]">{(s.probability_down * 100).toFixed(1)}%</span>
                                      </div>
                                      <div className="flex items-center justify-between">
                                        <span className="text-[9px] text-[var(--text-muted)]">Strength</span>
                                        <span className="text-[10px] font-medium tabular-nums text-[var(--text-secondary)]">{strength.toFixed(3)}</span>
                                      </div>
                                      <div className="flex items-center justify-between">
                                        <span className="text-[9px] text-[var(--text-muted)]">Conviction</span>
                                        <span className="text-[10px] font-medium tabular-nums text-[var(--text-secondary)]">{(conviction * 100).toFixed(1)}%</span>
                                      </div>
                                      {profitPln != null && profitPln !== 0 && (
                                        <div className="flex items-center justify-between pt-1 mt-1" style={{ borderTop: `1px solid ${accentSoft}` }}>
                                          <span className="text-[9px] text-[var(--text-muted)]">Est. Profit (PLN)</span>
                                          <span className="text-[10px] font-bold tabular-nums" style={{ color: accent }}>
                                            {profitPln > 0 ? '+' : ''}{profitPln.toLocaleString('en', { maximumFractionDigits: 0 })}
                                          </span>
                                        </div>
                                      )}
                                    </div>
                                    {/* Strength bar */}
                                    <div className="mt-2">
                                      <div className="w-full h-1 rounded-full bg-white/[0.06] overflow-hidden">
                                        <div className="h-full rounded-full transition-all" style={{ width: `${Math.min(strength * 200, 100)}%`, background: accent }} />
                                      </div>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                            {/* Generated timestamp */}
                            {g.signals[0] && (g.signals[0] as Record<string, unknown>).generated_at && (
                              <div className="flex items-center gap-1 mt-2">
                                <Clock className="w-3 h-3 text-[var(--text-muted)]" />
                                <span className="text-[9px] text-[var(--text-muted)]">
                                  Generated: {new Date(String((g.signals[0] as Record<string, unknown>).generated_at)).toLocaleString()}
                                </span>
                              </div>
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
    </div>
  );
}
