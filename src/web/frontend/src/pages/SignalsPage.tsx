import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { SummaryRow, SectorGroup, StrongSignalEntry, HighConvictionSignal, SignalSummaryData } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { SignalTableSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { SignalsEmpty } from '../components/CosmicEmptyState';
import { ExportButton } from '../components/ExportButton';
import { Sparkline } from '../components/Sparkline';
import { SignalStrengthBar, MomentumBadge, CrashRiskHeat, HorizonCell } from '../components/SignalTableVisuals';
import {
  ArrowUpCircle, ArrowDownCircle, Filter, ChevronDown, ChevronRight,
  TrendingUp, TrendingDown, Search, X, ExternalLink, BarChart3,
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
            <pre className="text-[#7a8ba4] text-xs overflow-auto max-h-48 bg-[#0a0a1a] p-3 rounded">
              {this.state.error?.stack}
            </pre>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="mt-3 px-3 py-1 rounded text-sm"
              style={{ background: 'rgba(139,92,246,0.15)', color: '#b49aff' }}
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

/** Story 6.4: WebSocket connection status indicator. */
function WsStatusDot({ status }: { status: WSStatus }) {
  const color = status === 'connected' ? '#3ee8a5' : status === 'connecting' ? '#f5c542' : '#f87171';
  const label = status === 'connected' ? 'Live' : status === 'connecting' ? 'Connecting' : 'Offline';
  return (
    <span className="inline-flex items-center gap-1 ml-2 text-[10px]" title={`WebSocket: ${status}`}>
      <span className="inline-block w-2 h-2 rounded-full" style={{ backgroundColor: color }} />
      <span style={{ color }} className="font-medium">{label}</span>
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

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6 fade-up">
          <MiniStat label="Strong Buy" value={stats.strong_buy_signals} color="#3ee8a5" icon={'\u25B2\u25B2'} />
          <MiniStat label="Buy" value={stats.buy_signals - stats.strong_buy_signals} color="#6ff0c0" icon={'\u25B2'} />
          <MiniStat label="Hold" value={stats.hold_signals} color="#7a8ba4" icon={'\u2014'} />
          <MiniStat label="Sell" value={stats.sell_signals - stats.strong_sell_signals} color="#f87171" icon={'\u25BC'} />
          <MiniStat label="Strong Sell" value={stats.strong_sell_signals} color="#ff6b8a" icon={'\u25BC\u25BC'} />
          <MiniStat label="Exit" value={stats.exit_signals} color="#f5c542" icon={'\u2298'} />
        </div>
      )}

      {/* High Conviction Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8 fade-up-delay-1">
        <HighConvictionCard
          title="High Conviction BUY"
          signals={buyQ.data?.signals || []}
          color="green"
        />
        <HighConvictionCard
          title="High Conviction SELL"
          signals={sellQ.data?.signals || []}
          color="red"
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
                view === key ? 'bg-[#8b5cf6]/15 text-[#b49aff]' : 'text-[#7a8ba4] hover:text-[#94a3b8] hover:bg-white/[0.02]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Signal filter */}
        {view !== 'strong' && (
          <div className="flex items-center gap-0.5 glass-card px-2.5 py-1.5">
            <Filter className="w-3 h-3 text-[#7a8ba4] mr-1.5" />
            {([
              { key: 'all' as SignalFilter, label: 'All', c: '#b49aff' },
              { key: 'strong_buy' as SignalFilter, label: '\u25B2\u25B2 SB', c: '#3ee8a5' },
              { key: 'buy' as SignalFilter, label: '\u25B2 Buy', c: '#6ff0c0' },
              { key: 'hold' as SignalFilter, label: '\u2014 Hold', c: '#7a8ba4' },
              { key: 'sell' as SignalFilter, label: '\u25BC Sell', c: '#f87171' },
              { key: 'strong_sell' as SignalFilter, label: '\u25BC\u25BC SS', c: '#ff6b8a' },
            ]).map(({ key, label, c }) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className="px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-200"
                style={filter === key ? { color: c, background: `${c}15` } : { color: '#7a8ba4' }}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* Story 3.5: Smart Search with violet focus */}
        <div className="flex items-center gap-2 glass-card px-3 py-2 group search-cosmic transition-all duration-200">
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
            style={{ color: 'var(--accent-violet)', background: 'rgba(139,92,246,0.12)' }}
          >
            {changeLog.length} change{changeLog.length > 1 ? 's' : ''}
          </button>
        )}

        {/* Story 3.4: Live Feed toggle */}
        <button
          onClick={() => setShowTickerTape(p => !p)}
          className={`text-[10px] px-2 py-1 rounded transition-colors ${showTickerTape ? 'text-[var(--accent-violet)] bg-[rgba(139,92,246,0.12)]' : 'text-[var(--text-muted)] hover:text-[var(--text-secondary)]'}`}
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
            {changeLog.map((c, i) => {
              const isUpgrade = ['STRONG BUY', 'BUY'].includes(c.to) && ['HOLD', 'SELL', 'STRONG SELL', 'EXIT'].includes(c.from);
              return (
                <span key={`${c.asset}-${i}`} className="inline-flex items-center gap-1">
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
        <div className="mb-2 glass-card px-4 py-2 flex items-center gap-3" style={{ background: 'rgba(139,92,246,0.06)' }}>
          <span className="text-[12px] text-[var(--accent-violet)]">{awayChanges.length} signal{awayChanges.length > 1 ? 's' : ''} changed while away</span>
          <button
            onClick={() => { setChangeLog(prev => [...awayChanges, ...prev].slice(0, 20)); setAwayChanges([]); setShowTickerTape(true); }}
            className="text-[11px] px-2 py-0.5 rounded text-[var(--accent-violet)] hover:bg-[rgba(139,92,246,0.1)] transition-colors"
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
                  ? { color: 'var(--accent-violet)', background: 'rgba(139,92,246,0.15)', border: '1px solid var(--border-glow)' }
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

/* ── Mini stat card ──────────────────────────────────────────────── */
function MiniStat({ label, value, color, icon }: { label: string; value: number; color: string; icon: string }) {
  return (
    <div className="glass-card px-3 py-2.5 flex items-center gap-2 hover-lift stat-shine">
      <span className="text-lg font-bold" style={{ color }}>{icon}</span>
      <div>
        <p className="text-lg font-bold text-[#e2e8f0] tabular-nums">{value}</p>
        <p className="text-[10px] text-[#7a8ba4]">{label}</p>
      </div>
    </div>
  );
}

/* ── Story 3.3: Sector Panels Redesign with Nebula Intelligence ──── */
type SectorSortBy = 'momentum' | 'exp_ret' | 'signal' | 'count' | 'alpha';

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
  const [showSortDropdown, setShowSortDropdown] = useState(false);

  const sorted = useMemo(() => {
    const arr = [...sectors];
    const signalScore = (s: SectorGroup) => (s.strong_buy ?? 0) * 3 + (s.buy ?? 0) * 2 - (s.sell ?? 0) * 2 - (s.strong_sell ?? 0) * 3;
    switch (sectorSort) {
      case 'momentum': return arr.sort((a, b) => (b.avg_momentum ?? 0) - (a.avg_momentum ?? 0));
      case 'signal': return arr.sort((a, b) => signalScore(b) - signalScore(a));
      case 'count': return arr.sort((a, b) => b.asset_count - a.asset_count);
      case 'alpha': return arr.sort((a, b) => a.name.localeCompare(b.name));
      case 'exp_ret': return arr.sort((a, b) => signalScore(b) - signalScore(a)); // fallback
      default: return arr;
    }
  }, [sectors, sectorSort]);

  return (
    <div className="space-y-2">
      {/* Story 3.3 AC-3: Sort dropdown */}
      <div className="flex items-center gap-2 mb-2">
        <div className="relative">
          <button
            onClick={() => setShowSortDropdown(p => !p)}
            className="text-[10px] text-[var(--text-secondary)] flex items-center gap-1 hover:text-[var(--accent-violet)] transition-colors"
          >
            Sort <ChevronDown className="w-3 h-3" />
          </button>
          {showSortDropdown && (
            <div className="absolute top-6 left-0 z-20 glass-card py-1 px-1 min-w-[140px]"
              style={{ boxShadow: '0 8px 32px rgba(0,0,0,0.4)' }}>
              {([
                { key: 'momentum' as SectorSortBy, label: 'Momentum' },
                { key: 'signal' as SectorSortBy, label: 'Signal Strength' },
                { key: 'count' as SectorSortBy, label: 'Asset Count' },
                { key: 'alpha' as SectorSortBy, label: 'Alphabetical' },
              ]).map(({ key, label }) => (
                <button key={key}
                  onClick={() => { setSectorSort(key); setShowSortDropdown(false); }}
                  className={`block w-full text-left px-3 py-1.5 text-[11px] rounded transition-colors ${
                    sectorSort === key ? 'text-[var(--accent-violet)] bg-[rgba(139,92,246,0.08)]' : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--void-hover)]'
                  }`}
                >
                  {label}
                </button>
              ))}
            </div>
          )}
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
        const total = bullish + bearish + (sector.hold ?? 0);
        const sentiment = bullish > bearish ? 'bullish' : bearish > bullish ? 'bearish' : 'neutral';

        // AC-5: Left border class
        const borderClass = sentiment === 'bullish' ? 'sector-border-bullish' : sentiment === 'bearish' ? 'sector-border-bearish' : 'sector-border-mixed';

        // AC-2: Best performing asset for peek row
        const bestAsset = [...sector.assets].sort((a, b) => {
          const aRet = Object.values(a.horizon_signals)[0]?.exp_ret ?? 0;
          const bRet = Object.values(b.horizon_signals)[0]?.exp_ret ?? 0;
          return bRet - aRet;
        })[0];
        const bestTicker = bestAsset ? extractTicker(bestAsset.asset_label) : null;
        const bestRet = bestAsset ? (Object.values(bestAsset.horizon_signals)[0]?.exp_ret ?? 0) * 100 : 0;
        const bestLabel = bestAsset ? (bestAsset.nearest_label || 'HOLD').toUpperCase() : '';

        // AC-1: Sentiment bar proportions
        const strongBuyPct = total > 0 ? ((sector.strong_buy ?? 0) / total) * 100 : 0;
        const buyPct = total > 0 ? ((sector.buy ?? 0) / total) * 100 : 0;
        const holdPct = total > 0 ? ((sector.hold ?? 0) / total) * 100 : 0;
        const sellPct = total > 0 ? ((sector.sell ?? 0) / total) * 100 : 0;
        const strongSellPct = total > 0 ? ((sector.strong_sell ?? 0) / total) * 100 : 0;

        const avgMom = sector.avg_momentum ?? 0;

        return (
          <div key={sector.name} className={`glass-card overflow-hidden ${borderClass}`}>
            {/* AC-1: 48px sector header */}
            <button
              onClick={() => toggleSector(sector.name)}
              className="w-full flex items-center gap-3 px-4 h-[48px] hover:bg-[var(--void-hover)] transition"
            >
              {/* Sector name */}
              <span className="font-semibold text-[var(--text-luminous)] text-sm whitespace-nowrap">{sector.name}</span>
              {/* Asset count pill */}
              <span className="text-[9px] px-1.5 py-0.5 rounded-full text-[var(--text-muted)]" style={{ background: 'var(--void-active)' }}>
                {sector.asset_count}
              </span>

              {/* AC-1: Sector sentiment bar */}
              <div className="flex h-[4px] w-[80px] rounded-[2px] overflow-hidden flex-shrink-0" style={{ border: '1px solid var(--border-void)' }}>
                <div style={{ width: `${strongBuyPct}%`, background: 'var(--accent-emerald)' }} />
                <div style={{ width: `${buyPct}%`, background: 'rgba(62,232,165,0.5)' }} />
                <div style={{ width: `${holdPct}%`, background: 'var(--void-active)' }} />
                <div style={{ width: `${sellPct}%`, background: 'rgba(255,107,138,0.5)' }} />
                <div style={{ width: `${strongSellPct}%`, background: 'var(--accent-rose)' }} />
              </div>

              {/* Avg momentum with arrow */}
              <span className="text-[10px] font-mono tabular-nums flex items-center gap-0.5"
                style={{ color: avgMom > 0 ? 'var(--accent-emerald)' : avgMom < 0 ? 'var(--accent-rose)' : 'var(--text-muted)' }}>
                {avgMom > 0 ? (
                  <svg width="6" height="6" viewBox="0 0 6 6"><path d="M3 0L6 5H0L3 0Z" fill="currentColor" /></svg>
                ) : avgMom < 0 ? (
                  <svg width="6" height="6" viewBox="0 0 6 6"><path d="M3 6L0 1H6L3 6Z" fill="currentColor" /></svg>
                ) : null}
                {avgMom > 0 ? '+' : ''}{avgMom.toFixed(1)}%
              </span>

              {/* Spacer */}
              <div className="flex-1" />

              {/* AC-2: Peek row (best asset) */}
              {!expanded && bestTicker && (
                <span className="text-[10px] hidden sm:inline-flex items-center gap-1">
                  <span className="text-[var(--text-muted)]">Best:</span>
                  <span className="text-[var(--accent-violet)] font-medium">{bestTicker}</span>
                  <span style={{ color: bestRet >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                    {bestRet >= 0 ? '+' : ''}{bestRet.toFixed(1)}%
                  </span>
                  <span className="text-[var(--text-muted)]">({bestLabel})</span>
                </span>
              )}

              {/* Expand chevron with rotation animation */}
              <ChevronDown
                className="w-3 h-3 flex-shrink-0 transition-transform duration-200"
                style={{
                  color: 'var(--accent-violet)',
                  transform: expanded ? 'rotate(0deg)' : 'rotate(-90deg)',
                }}
              />
            </button>

            {/* AC-4: Expanded content with constellation animation */}
            {expanded && (
              <div className="border-t border-[var(--border-void)]" style={{ animation: 'slide-down 250ms cubic-bezier(0.2,0,0,1) both' }}>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead className="premium-thead">
                      <tr className="border-b border-[#2a2a4a]/50">
                        <th className="text-left px-4 py-2 text-[10px] text-[var(--text-violet)] font-medium uppercase tracking-[0.06em]">Asset</th>
                        <th className="text-center px-2 py-2 text-[10px] text-[var(--text-violet)] font-medium w-[60px]">30D</th>
                        <th className="text-center px-2 py-2 text-[10px] text-[var(--text-violet)] font-medium">Signal</th>
                        <th className="text-center px-2 py-2 text-[10px] text-[var(--text-violet)] font-medium">Mom</th>
                        {horizons.map(h => (
                          <th key={h} className="text-center px-2 py-2 text-[10px] text-[var(--text-violet)] font-medium">{formatHorizon(h)}</th>
                        ))}
                        <th className="text-center px-2 py-2 text-[10px] text-[var(--text-violet)] font-medium">Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {assets.map((row, i) => (
                        <SectorSignalRow key={row.asset_label} row={row} horizons={horizons} highlighted={row.asset_label === updatedAsset} delayMs={i * 50} />
                      ))}
                    </tbody>
                  </table>
                </div>
                {assets.length === 0 && (
                  <p className="px-4 py-3 text-xs" style={{ color: 'var(--text-muted)' }}>No assets match current filter</p>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ── Strong Signals View ─────────────────────────────────────────── */
function StrongSignalsView({ strongBuy, strongSell }: { strongBuy: StrongSignalEntry[]; strongSell: StrongSignalEntry[] }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
      <div className="glass-card overflow-hidden">
        <div className="px-5 py-3.5 flex items-center gap-2" style={{ borderBottom: '1px solid rgba(139,92,246,0.06)' }}>
          <TrendingUp className="w-4 h-4" style={{ color: '#3ee8a5' }} />
          <h3 className="premium-section-label" style={{ color: '#3ee8a5' }}>Strong Buy Signals</h3>
          <span className="ml-auto text-xs" style={{ color: 'var(--text-muted)' }}>{strongBuy.length} assets</span>
        </div>
        {strongBuy.length === 0 ? (
          <p className="px-5 py-8 text-xs text-center" style={{ color: 'var(--text-muted)' }}>No strong buy signals</p>
        ) : (
          <div>
            {strongBuy.map((s, i) => (
              <div key={i} className="px-5 py-3 flex items-center gap-3" style={{ borderBottom: '1px solid rgba(139,92,246,0.04)', transition: 'background 200ms cubic-bezier(0.2,0,0,1)' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'linear-gradient(90deg, rgba(62,232,165,0.06) 0%, transparent 60%)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                <div className="flex-1">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-luminous)' }}>{s.asset_label || '--'}</span>
                  <span className="text-[10px] ml-2" style={{ color: 'var(--text-muted)' }}>{s.sector}</span>
                </div>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{s.horizon || '--'}</span>
                <span className="text-xs font-medium" style={{ color: '#3ee8a5' }}>
                  {s.exp_ret != null ? `${s.exp_ret >= 0 ? '+' : ''}${(s.exp_ret * 100).toFixed(1)}%` : '--'}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>p={s.p_up != null ? s.p_up.toFixed(2) : '--'}</span>
                <MomentumBadge value={s.momentum} />
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="glass-card overflow-hidden">
        <div className="px-5 py-3.5 flex items-center gap-2" style={{ borderBottom: '1px solid rgba(139,92,246,0.06)' }}>
          <TrendingDown className="w-4 h-4" style={{ color: '#ff6b8a' }} />
          <h3 className="premium-section-label" style={{ color: '#ff6b8a' }}>Strong Sell Signals</h3>
          <span className="ml-auto text-xs" style={{ color: 'var(--text-muted)' }}>{strongSell.length} assets</span>
        </div>
        {strongSell.length === 0 ? (
          <p className="px-5 py-8 text-xs text-center" style={{ color: 'var(--text-muted)' }}>No strong sell signals</p>
        ) : (
          <div>
            {strongSell.map((s, i) => (
              <div key={i} className="px-5 py-3 flex items-center gap-3" style={{ borderBottom: '1px solid rgba(139,92,246,0.04)', transition: 'background 200ms cubic-bezier(0.2,0,0,1)' }}
                onMouseEnter={e => (e.currentTarget.style.background = 'linear-gradient(90deg, rgba(255,107,138,0.06) 0%, transparent 60%)')}
                onMouseLeave={e => (e.currentTarget.style.background = 'transparent')}>
                <div className="flex-1">
                  <span className="text-sm font-medium" style={{ color: 'var(--text-luminous)' }}>{s.asset_label || '--'}</span>
                  <span className="text-[10px] ml-2" style={{ color: 'var(--text-muted)' }}>{s.sector}</span>
                </div>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{s.horizon || '--'}</span>
                <span className="text-xs font-medium" style={{ color: '#ff6b8a' }}>
                  {s.exp_ret != null ? `${(s.exp_ret * 100).toFixed(1)}%` : '--'}
                </span>
                <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>p={s.p_up != null ? s.p_up.toFixed(2) : '--'}</span>
                <MomentumBadge value={s.momentum} />
              </div>
            ))}
          </div>
        )}
      </div>
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
                  style={sortLevels.some(s => s.col === 'asset') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px rgba(139,92,246,0.3)' } : {}}
                  onClick={(e) => onSort('asset', e.shiftKey)}>
                Asset <SortIndicator col="asset" sortLevels={sortLevels} />
              </th>
              <th className="text-center px-2 py-3 w-[60px]">
                <span className="text-[10px] text-[var(--text-violet)] uppercase tracking-[0.06em] font-medium">30D</span>
              </th>
              <th className={`text-left px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'sector') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'sector') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px rgba(139,92,246,0.3)' } : {}}
                  onClick={(e) => onSort('sector', e.shiftKey)}>
                Sector <SortIndicator col="sector" sortLevels={sortLevels} />
              </th>
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'signal') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'signal') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px rgba(139,92,246,0.3)' } : {}}
                  onClick={(e) => onSort('signal', e.shiftKey)}>
                Signal <SortIndicator col="signal" sortLevels={sortLevels} />
              </th>
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'momentum') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'momentum') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px rgba(139,92,246,0.3)' } : {}}
                  onClick={(e) => onSort('momentum', e.shiftKey)}>
                Mom <SortIndicator col="momentum" sortLevels={sortLevels} />
              </th>
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'crash_risk') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'crash_risk') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px rgba(139,92,246,0.3)' } : {}}
                  onClick={(e) => onSort('crash_risk', e.shiftKey)}>
                Risk <SortIndicator col="crash_risk" sortLevels={sortLevels} />
              </th>
              {horizons.map((h) => {
                const hCol = `horizon_${h}` as SortColumn;
                return (
                  <th key={h} className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === hCol) ? 'active' : ''}`}
                      style={sortLevels.some(s => s.col === hCol) ? { color: 'var(--accent-violet)', textShadow: '0 0 8px rgba(139,92,246,0.3)' } : {}}
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
    const lineColor = isUp ? '#3ee8a5' : '#ff6b8a';
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
    <div className="mini-chart-enter bg-[#0a0a1a]/60 border-t border-[#2a2a4a]/30">
      <div className="flex items-center gap-4 px-4 py-2">
        {/* Chart area */}
        <div className="flex-1 h-[80px]">
          {isLoading ? (
            <div className="h-full flex items-center justify-center">
              <div className="w-4 h-4 border-2 border-[#2a2a4a] border-t-[#8b5cf6] rounded-full animate-spin" />
            </div>
          ) : error ? (
            <div className="h-full flex items-center justify-center text-[10px] text-[#7a8ba4]">Chart unavailable</div>
          ) : (
            <canvas ref={canvasRef} className="w-full h-full" />
          )}
        </div>

        {/* Price info */}
        <div className="flex-shrink-0 text-right space-y-1">
          <p className="text-sm font-bold text-[#e2e8f0] tabular-nums">
            {lastPrice > 0 ? (lastPrice < 10 ? lastPrice.toFixed(4) : lastPrice.toFixed(2)) : '--'}
          </p>
          <p className={`text-xs font-semibold tabular-nums ${isUp ? 'text-[#3ee8a5]' : 'text-[#ff6b8a]'}`}>
            {isUp ? '+' : ''}{changePct.toFixed(2)}%
          </p>
          <p className="text-[9px] text-[#6b7a90]">3M change</p>
        </div>

        {/* Navigate to full chart */}
        <button
          onClick={onNavigateChart}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-medium text-[#b49aff] bg-[#8b5cf6]/10 hover:bg-[#8b5cf6]/20 transition-all"
        >
          <ExternalLink className="w-3 h-3" />
          Full Chart
        </button>
      </div>
    </div>
  );
}

/* ── Sector signal row (cosmic, with sparkline) ──────────────────── */
function SectorSignalRow({ row, horizons, highlighted, delayMs = 0 }: { row: SummaryRow; horizons: number[]; highlighted?: boolean; delayMs?: number }) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  const ticker = extractTicker(row.asset_label);
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  return (
    <tr className={`cosmic-row constellation-row border-b border-[var(--border-void)] ${highlighted ? 'aurora-upgrade' : ''}`}
      style={{ animationDelay: `${delayMs}ms` }}>
      <td className="px-4 py-2 whitespace-nowrap">
        <span className="font-semibold text-[var(--text-primary)] text-xs">{ticker}</span>
        {row.asset_label.includes('(') && (
          <span className="block text-[9px] text-[var(--text-muted)] truncate max-w-[140px] leading-tight">
            {row.asset_label.split('(')[0].trim()}
          </span>
        )}
      </td>
      <td className="px-1 py-2 text-center">
        <Sparkline ticker={ticker} width={60} height={24} />
      </td>
      <td className="px-3 py-2 text-center">
        <SignalStrengthBar label={label} pUp={nearestHorizon?.p_up} kelly={nearestHorizon?.kelly_half} />
      </td>
      <td className="px-3 py-2 text-center">
        <MomentumBadge value={row.momentum_score} />
      </td>
      {horizons.map((h) => {
        const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
        return (
          <td key={h} className="px-2 py-2 text-center">
            <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
          </td>
        );
      })}
      <td className="px-3 py-2">
        <div className="flex justify-center">
          <CrashRiskHeat score={row.crash_risk_score} />
        </div>
      </td>
    </tr>
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

function HighConvictionCard({
  title, signals, color,
}: {
  title: string; signals: HighConvictionSignal[]; color: 'green' | 'red';
}) {
  const Icon = color === 'green' ? ArrowUpCircle : ArrowDownCircle;
  const accent = color === 'green' ? '#3ee8a5' : '#ff6b8a';
  const top5 = signals.slice(0, 5);

  return (
    <div className={`glass-card p-5 hover-lift ${color === 'green' ? 'glow-green' : 'glow-red'}`}>
      <div className="flex items-center gap-2.5 mb-4">
        <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: `${accent}10` }}>
          <Icon className="w-4 h-4" style={{ color: accent }} />
        </div>
        <h3 className="text-[13px] font-medium" style={{ color: accent }}>{title}</h3>
        <span className="ml-auto text-[11px] text-[#7a8ba4] tabular-nums">{signals.length} signals</span>
      </div>
      {top5.length === 0 ? (
        <p className="text-xs text-[#6b7a90]">No signals</p>
      ) : (
        <div className="space-y-2">
          {top5.map((s, i) => (
            <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-white/[0.03] last:border-0">
              <span className="text-[#f1f5f9] font-semibold tracking-wide">{s.ticker || '\u2014'}</span>
              <span className="text-[#7a8ba4]">{s.horizon_days != null ? formatHorizon(s.horizon_days) : '\u2014'}</span>
              <span className="font-semibold tabular-nums" style={{ color: accent }}>{s.expected_return_pct != null ? `${s.expected_return_pct.toFixed(1)}%` : '\u2014'}</span>
              <span className="text-[#6b7a90] tabular-nums">p={s.probability_up != null ? s.probability_up.toFixed(2) : '\u2014'}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
