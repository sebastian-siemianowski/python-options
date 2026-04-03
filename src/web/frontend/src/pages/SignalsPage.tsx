import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef, useCallback, Component, type ReactNode, type ErrorInfo } from 'react';
import { useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { SummaryRow, SectorGroup, StrongSignalEntry, HighConvictionSignal, SignalSummaryData } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  ArrowUpCircle, ArrowDownCircle, Filter, ChevronDown, ChevronRight, ChevronUp,
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
            <pre className="text-[#64748b] text-xs overflow-auto max-h-48 bg-[#0a0a1a] p-3 rounded">
              {this.state.error?.stack}
            </pre>
            <button
              onClick={() => this.setState({ hasError: false, error: null })}
              className="mt-3 px-3 py-1 bg-[#42A5F5]/20 text-[#42A5F5] rounded text-sm"
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

/* ── Story 6.1: SVG directional arrows ───────────────────────── */
function DirectionArrow({ direction }: { direction: 'up' | 'down' | 'neutral' }) {
  if (direction === 'up') {
    return (
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="inline-block mr-0.5" aria-label="Up">
        <path d="M6 2L10 8H2L6 2Z" fill="#66BB6A" />
      </svg>
    );
  }
  if (direction === 'down') {
    return (
      <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="inline-block mr-0.5" aria-label="Down">
        <path d="M6 10L2 4H10L6 10Z" fill="#EF5350" />
      </svg>
    );
  }
  return (
    <svg width="12" height="12" viewBox="0 0 12 12" fill="none" className="inline-block mr-0.5" aria-label="Neutral">
      <rect x="2" y="5" width="8" height="2" rx="1" fill="#64748b" />
    </svg>
  );
}

/** Determine direction and color from expected return. */
function signalDirection(expRet: number | null | undefined): { dir: 'up' | 'down' | 'neutral'; color: string } {
  if (expRet == null || Math.abs(expRet) < 0.001) return { dir: 'neutral', color: '#64748b' };
  return expRet > 0 ? { dir: 'up', color: '#66BB6A' } : { dir: 'down', color: '#EF5350' };
}

/** Story 6.3: Compact exhaustion heat bar (ue_down=red left, ue_up=green right). */
function ExhaustionBar({ ueUp, ueDown }: { ueUp: number; ueDown: number }) {
  const up = Math.min(Math.max(ueUp || 0, 0), 1);
  const down = Math.min(Math.max(ueDown || 0, 0), 1);
  if (up < 0.01 && down < 0.01) return null;
  return (
    <div
      className="flex h-[3px] w-full mt-0.5 rounded-sm overflow-hidden"
      title={`Exhaustion: up=${(up * 100).toFixed(0)}% down=${(down * 100).toFixed(0)}%`}
    >
      <div style={{ width: '50%', background: `rgba(239,83,80,${down})` }} />
      <div style={{ width: '50%', background: `rgba(102,187,106,${up})` }} />
    </div>
  );
}

/** Story 6.4: WebSocket connection status indicator. */
function WsStatusDot({ status }: { status: WSStatus }) {
  const color = status === 'connected' ? '#66BB6A' : status === 'connecting' ? '#FFA726' : '#EF5350';
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
  const [expandedSectors, setExpandedSectors] = useState<Set<string>>(new Set());

  const [updatedAsset, setUpdatedAsset] = useState<string | null>(null);
  const [sortCol, setSortCol] = useState<SortColumn>('momentum');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [expandedRow, setExpandedRow] = useState<string | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);

  const queryClient = useQueryClient();
  const { status: wsStatus, lastMessage } = useWebSocket('/ws');

  // Story 6.4: Real-time signal updates via WebSocket
  useEffect(() => {
    if (!lastMessage || lastMessage.type !== 'signal_update') return;
    const summary = lastMessage.summary as SummaryRow | undefined;
    if (!summary?.asset_label) return;

    queryClient.setQueryData<SignalSummaryData>(['signalSummary'], (old) => {
      if (!old) return old;
      const rows = old.summary_rows.map((r) =>
        r.asset_label === summary.asset_label ? { ...r, ...summary } : r
      );
      return { ...old, summary_rows: rows };
    });

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
  const horizons = useMemo(() => responsiveHorizons(allHorizons, windowWidth), [allHorizons, windowWidth]);
  const stats = statsQ.data;
  const sectors = sectorQ.data?.sectors || [];

  const filteredRows = useMemo(() => {
    return rows.filter((row) => {
      if (search && !row.asset_label.toLowerCase().includes(search.toLowerCase())) return false;
      if (filter === 'all') return true;
      const label = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
      return label === filter.toUpperCase();
    });
  }, [rows, search, filter]);

  /** Sorted rows for "All Assets" table */
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
    arr.sort((a, b) => {
      let cmp = 0;
      switch (sortCol) {
        case 'asset': cmp = a.asset_label.localeCompare(b.asset_label); break;
        case 'sector': cmp = (a.sector || '').localeCompare(b.sector || ''); break;
        case 'signal': cmp = signalRank((a.nearest_label || 'HOLD')) - signalRank((b.nearest_label || 'HOLD')); break;
        case 'momentum': cmp = (a.momentum_score ?? 0) - (b.momentum_score ?? 0); break;
        case 'crash_risk': cmp = (a.crash_risk_score ?? 0) - (b.crash_risk_score ?? 0); break;
        default:
          if (sortCol.startsWith('horizon_')) {
            const h = parseInt(sortCol.split('_')[1], 10);
            cmp = getHorizonVal(a, h) - getHorizonVal(b, h);
          }
      }
      return sortDir === 'desc' ? -cmp : cmp;
    });
    return arr;
  }, [filteredRows, sortCol, sortDir]);

  /** Toggle sort: click same col flips direction, new col sets desc */
  const handleSort = useCallback((col: SortColumn) => {
    if (sortCol === col) {
      setSortDir(d => d === 'desc' ? 'asc' : 'desc');
    } else {
      setSortCol(col);
      setSortDir('desc');
    }
  }, [sortCol]);

  const toggleSector = (name: string) => {
    setExpandedSectors(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const expandAll = () => setExpandedSectors(new Set(sectors.map(s => s.name)));

  if (isLoading) return <LoadingSpinner text="Loading signals..." />;

  if (error) {
    return (
      <div className="p-6">
        <div className="glass-card p-6 border border-red-500/50">
          <h2 className="text-red-400 text-lg font-bold mb-2">Failed to load signals</h2>
          <p className="text-red-300 text-sm">{String(error)}</p>
        </div>
      </div>
    );
  }
  const collapseAll = () => setExpandedSectors(new Set());

  return (
    <>
      <PageHeader title="Signals">
        <span>{rows.length} assets across {horizons.length} horizons</span>
        <WsStatusDot status={wsStatus} />
      </PageHeader>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6 fade-up">
          <MiniStat label="Strong Buy" value={stats.strong_buy_signals} color="#00E676" icon={'\u25B2\u25B2'} />
          <MiniStat label="Buy" value={stats.buy_signals - stats.strong_buy_signals} color="#66BB6A" icon={'\u25B2'} />
          <MiniStat label="Hold" value={stats.hold_signals} color="#64748b" icon={'\u2014'} />
          <MiniStat label="Sell" value={stats.sell_signals - stats.strong_sell_signals} color="#EF5350" icon={'\u25BC'} />
          <MiniStat label="Strong Sell" value={stats.strong_sell_signals} color="#FF1744" icon={'\u25BC\u25BC'} />
          <MiniStat label="Exit" value={stats.exit_signals} color="#FFB300" icon={'\u2298'} />
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
                view === key ? 'bg-[#42A5F5]/15 text-[#42A5F5]' : 'text-[#64748b] hover:text-[#94a3b8] hover:bg-white/[0.02]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Signal filter */}
        {view !== 'strong' && (
          <div className="flex items-center gap-0.5 glass-card px-2.5 py-1.5">
            <Filter className="w-3 h-3 text-[#64748b] mr-1.5" />
            {([
              { key: 'all' as SignalFilter, label: 'All', c: '#42A5F5' },
              { key: 'strong_buy' as SignalFilter, label: '\u25B2\u25B2 SB', c: '#00E676' },
              { key: 'buy' as SignalFilter, label: '\u25B2 Buy', c: '#66BB6A' },
              { key: 'hold' as SignalFilter, label: '\u2014 Hold', c: '#64748b' },
              { key: 'sell' as SignalFilter, label: '\u25BC Sell', c: '#EF5350' },
              { key: 'strong_sell' as SignalFilter, label: '\u25BC\u25BC SS', c: '#FF1744' },
            ]).map(({ key, label, c }) => (
              <button
                key={key}
                onClick={() => setFilter(key)}
                className="px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-200"
                style={filter === key ? { color: c, background: `${c}15` } : { color: '#64748b' }}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* Search */}
        <div className="flex items-center gap-2 glass-card px-3 py-2 group focus-within:ring-1 focus-within:ring-[#42A5F5]/20 transition-all">
          <Search className="w-3.5 h-3.5 text-[#64748b] group-focus-within:text-[#42A5F5] transition-colors" />
          <input
            ref={searchRef}
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets..."
            className="bg-transparent text-[13px] text-[#f1f5f9] placeholder:text-[#475569] outline-none w-40"
          />
          {search && (
            <button onClick={() => setSearch('')} className="text-[#64748b] hover:text-[#e2e8f0] transition-colors">
              <X className="w-3 h-3" />
            </button>
          )}
        </div>

        <span className="text-xs text-[#64748b]">
          {view === 'sectors' ? `${sectors.length} sectors` : `${filteredRows.length} results`}
        </span>

        {view === 'sectors' && (
          <div className="flex gap-2 ml-auto">
            <button onClick={expandAll} className="text-[10px] text-[#42A5F5] hover:underline">Expand All</button>
            <button onClick={collapseAll} className="text-[10px] text-[#64748b] hover:underline">Collapse All</button>
          </div>
        )}
      </div>

      {/* Content */}
      {view === 'sectors' && (
        <SectorPanels
          sectors={sectors}
          expandedSectors={expandedSectors}
          toggleSector={toggleSector}
          horizons={horizons}
          search={search}
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
          sortCol={sortCol} sortDir={sortDir} onSort={handleSort}
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
        <p className="text-[10px] text-[#64748b]">{label}</p>
      </div>
    </div>
  );
}

/* ── Sector Panels ───────────────────────────────────────────────── */
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
  const sorted = useMemo(() =>
    [...sectors].sort((a, b) => {
      const scoreA = (a.strong_buy ?? 0) * 3 + (a.buy ?? 0) * 2 - (a.sell ?? 0) * 2 - (a.strong_sell ?? 0) * 3;
      const scoreB = (b.strong_buy ?? 0) * 3 + (b.buy ?? 0) * 2 - (b.sell ?? 0) * 2 - (b.strong_sell ?? 0) * 3;
      return scoreB - scoreA;
    }), [sectors]);

  return (
    <div className="space-y-2">
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
        const sentiment = bullish > bearish ? 'bullish' : bearish > bullish ? 'bearish' : 'neutral';
        const sentimentColor = sentiment === 'bullish' ? '#00E676' : sentiment === 'bearish' ? '#FF1744' : '#64748b';

        return (
          <div key={sector.name} className="glass-card overflow-hidden">
            {/* Sector header */}
            <button
              onClick={() => toggleSector(sector.name)}
              className="w-full flex items-center gap-3 px-4 py-3 hover:bg-[#16213e]/30 transition"
            >
              {expanded
                ? <ChevronDown className="w-4 h-4 text-[#64748b]" />
                : <ChevronRight className="w-4 h-4 text-[#64748b]" />
              }
              <span className="font-medium text-[#e2e8f0] text-sm">{sector.name}</span>
              <span className="text-xs text-[#64748b]">{sector.asset_count} assets</span>

              {/* Mini signal badges */}
              <div className="flex items-center gap-1.5 ml-auto">
                {sector.strong_buy > 0 && <CountBadge value={sector.strong_buy} label={'\u25B2\u25B2'} color="#00E676" />}
                {sector.buy > 0 && <CountBadge value={sector.buy} label={'\u25B2'} color="#66BB6A" />}
                {sector.hold > 0 && <CountBadge value={sector.hold} label={'\u2014'} color="#64748b" />}
                {sector.sell > 0 && <CountBadge value={sector.sell} label={'\u25BC'} color="#EF5350" />}
                {sector.strong_sell > 0 && <CountBadge value={sector.strong_sell} label={'\u25BC\u25BC'} color="#FF1744" />}
                {sector.exit > 0 && <CountBadge value={sector.exit} label={'\u2298'} color="#FFB300" />}
              </div>

              {/* Sentiment indicator */}
              <span className="text-[10px] font-medium px-2 py-0.5 rounded" style={{ color: sentimentColor, background: `${sentimentColor}15` }}>
                {sentiment}
              </span>

              {/* Avg momentum */}
              <span className={`text-[10px] ${(sector.avg_momentum ?? 0) > 0 ? 'text-[#00E676]' : (sector.avg_momentum ?? 0) < 0 ? 'text-[#FF1744]' : 'text-[#64748b]'}`}>
                MOM {(sector.avg_momentum ?? 0).toFixed(1)}%
              </span>
            </button>

            {/* Expanded content */}
            {expanded && (
              <div className="border-t border-[#2a2a4a]">
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b border-[#2a2a4a]/50">
                        <th className="text-left px-4 py-2 text-[10px] text-[#64748b] font-medium uppercase">Asset</th>
                        <th className="text-center px-2 py-2 text-[10px] text-[#64748b] font-medium">Signal</th>
                        <th className="text-center px-2 py-2 text-[10px] text-[#64748b] font-medium">Momentum</th>
                        {horizons.map(h => (
                          <th key={h} className="text-center px-2 py-2 text-[10px] text-[#64748b] font-medium">{formatHorizon(h)}</th>
                        ))}
                        <th className="text-center px-2 py-2 text-[10px] text-[#64748b] font-medium">Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {assets.map(row => (
                        <SectorSignalRow key={row.asset_label} row={row} horizons={horizons} highlighted={row.asset_label === updatedAsset} />
                      ))}
                    </tbody>
                  </table>
                </div>
                {assets.length === 0 && (
                  <p className="px-4 py-3 text-xs text-[#64748b]">No assets match current filter</p>
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
        <div className="px-4 py-3 border-b border-[#2a2a4a] flex items-center gap-2">
          <TrendingUp className="w-4 h-4 text-[#00E676]" />
          <h3 className="text-sm font-medium text-[#00E676]">Strong Buy Signals</h3>
          <span className="ml-auto text-xs text-[#64748b]">{strongBuy.length} assets</span>
        </div>
        {strongBuy.length === 0 ? (
          <p className="px-4 py-6 text-xs text-[#64748b] text-center">No strong buy signals</p>
        ) : (
          <div className="divide-y divide-[#2a2a4a]/50">
            {strongBuy.map((s, i) => (
              <div key={i} className="px-4 py-2.5 flex items-center gap-3 hover:bg-[#16213e]/30 transition">
                <div className="flex-1">
                  <span className="text-sm font-medium text-[#e2e8f0]">{s.asset_label || '—'}</span>
                  <span className="text-[10px] text-[#64748b] ml-2">{s.sector}</span>
                </div>
                <span className="text-[10px] text-[#64748b]">{s.horizon || '—'}</span>
                <span className="text-xs text-[#00E676] font-medium">
                  {s.exp_ret != null ? `${s.exp_ret >= 0 ? '+' : ''}${(s.exp_ret * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="text-[10px] text-[#64748b]">p={s.p_up != null ? s.p_up.toFixed(2) : '—'}</span>
                <MomentumBadge value={s.momentum} />
              </div>
            ))}
          </div>
        )}
      </div>

      <div className="glass-card overflow-hidden">
        <div className="px-4 py-3 border-b border-[#2a2a4a] flex items-center gap-2">
          <TrendingDown className="w-4 h-4 text-[#FF1744]" />
          <h3 className="text-sm font-medium text-[#FF1744]">Strong Sell Signals</h3>
          <span className="ml-auto text-xs text-[#64748b]">{strongSell.length} assets</span>
        </div>
        {strongSell.length === 0 ? (
          <p className="px-4 py-6 text-xs text-[#64748b] text-center">No strong sell signals</p>
        ) : (
          <div className="divide-y divide-[#2a2a4a]/50">
            {strongSell.map((s, i) => (
              <div key={i} className="px-4 py-2.5 flex items-center gap-3 hover:bg-[#16213e]/30 transition">
                <div className="flex-1">
                  <span className="text-sm font-medium text-[#e2e8f0]">{s.asset_label || '—'}</span>
                  <span className="text-[10px] text-[#64748b] ml-2">{s.sector}</span>
                </div>
                <span className="text-[10px] text-[#64748b]">{s.horizon || '—'}</span>
                <span className="text-xs text-[#FF1744] font-medium">
                  {s.exp_ret != null ? `${(s.exp_ret * 100).toFixed(1)}%` : '—'}
                </span>
                <span className="text-[10px] text-[#64748b]">p={s.p_up != null ? s.p_up.toFixed(2) : '—'}</span>
                <MomentumBadge value={s.momentum} />
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

/* ── All Assets Table (sortable + mini chart) ────────────────────── */
function SortIcon({ col, sortCol, sortDir }: { col: SortColumn; sortCol: SortColumn; sortDir: SortDir }) {
  if (col !== sortCol) return <ChevronDown className="w-3 h-3 opacity-0 group-hover:opacity-40 transition-opacity inline ml-0.5" />;
  return sortDir === 'desc'
    ? <ChevronDown className="w-3 h-3 text-[#42A5F5] inline ml-0.5" />
    : <ChevronUp className="w-3 h-3 text-[#42A5F5] inline ml-0.5" />;
}

function AllAssetsTable({ rows, horizons, updatedAsset, sortCol, sortDir, onSort, expandedRow, onExpandRow, onNavigateChart }: {
  rows: SummaryRow[]; horizons: number[]; updatedAsset: string | null;
  sortCol: SortColumn; sortDir: SortDir; onSort: (col: SortColumn) => void;
  expandedRow: string | null; onExpandRow: (label: string | null) => void;
  onNavigateChart: (symbol: string) => void;
}) {
  const [page, setPage] = useState(0);
  const pageSize = 50;
  const pageRows = rows.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(rows.length / pageSize);

  // Reset page when rows change
  useEffect(() => { setPage(0); }, [rows.length]);

  return (
    <div className="glass-card overflow-hidden fade-up-delay-3">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#2a2a4a]">
              <th className="text-left px-4 py-3 text-xs text-[#64748b] font-medium uppercase sortable-th group"
                  onClick={() => onSort('asset')}>
                Asset <SortIcon col="asset" sortCol={sortCol} sortDir={sortDir} />
              </th>
              <th className="text-left px-3 py-3 text-xs text-[#64748b] font-medium sortable-th group"
                  onClick={() => onSort('sector')}>
                Sector <SortIcon col="sector" sortCol={sortCol} sortDir={sortDir} />
              </th>
              <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium sortable-th group"
                  onClick={() => onSort('signal')}>
                Signal <SortIcon col="signal" sortCol={sortCol} sortDir={sortDir} />
              </th>
              <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium sortable-th group"
                  onClick={() => onSort('momentum')}>
                Momentum <SortIcon col="momentum" sortCol={sortCol} sortDir={sortDir} />
              </th>
              {horizons.map((h) => (
                <th key={h} className="text-center px-3 py-3 text-xs text-[#64748b] font-medium sortable-th group"
                    onClick={() => onSort(`horizon_${h}` as SortColumn)}>
                  {formatHorizon(h)} <SortIcon col={`horizon_${h}` as SortColumn} sortCol={sortCol} sortDir={sortDir} />
                </th>
              ))}
              <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium sortable-th group"
                  onClick={() => onSort('crash_risk')}>
                Risk <SortIcon col="crash_risk" sortCol={sortCol} sortDir={sortDir} />
              </th>
              <th className="w-8 px-2"></th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row) => {
              const ticker = extractTicker(row.asset_label);
              const isExpanded = expandedRow === row.asset_label;
              return (
                <SignalRowWithChart
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
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-[#2a2a4a]">
          <span className="text-xs text-[#64748b]">
            Page {page + 1} of {totalPages} ({rows.length} total)
          </span>
          <div className="flex gap-1">
            <button onClick={() => setPage(0)} disabled={page === 0}
              className="px-2 py-0.5 rounded text-xs text-[#42A5F5] hover:bg-[#16213e] disabled:opacity-30 transition">First</button>
            <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}
              className="px-2 py-0.5 rounded text-xs text-[#42A5F5] hover:bg-[#16213e] disabled:opacity-30 transition">Prev</button>
            <button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}
              className="px-2 py-0.5 rounded text-xs text-[#42A5F5] hover:bg-[#16213e] disabled:opacity-30 transition">Next</button>
            <button onClick={() => setPage(totalPages - 1)} disabled={page >= totalPages - 1}
              className="px-2 py-0.5 rounded text-xs text-[#42A5F5] hover:bg-[#16213e] disabled:opacity-30 transition">Last</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Signal row with expandable mini chart ───────────────────────── */
function SignalRowWithChart({ row, ticker, horizons, highlighted, isExpanded, onToggleExpand, onNavigateChart }: {
  row: SummaryRow; ticker: string; horizons: number[];
  highlighted?: boolean; isExpanded: boolean;
  onToggleExpand: () => void; onNavigateChart: () => void;
}) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  return (
    <>
      <tr className={`border-b border-[#2a2a4a]/50 row-glow transition ${highlighted ? 'animate-signal-flash' : ''} ${isExpanded ? 'bg-[#16213e]/40' : ''}`}>
        <td className="px-4 py-2 whitespace-nowrap">
          <button onClick={onToggleExpand} className="text-left group/asset">
            <span className="font-semibold text-[#e2e8f0] text-xs group-hover/asset:text-[#42A5F5] transition-colors">
              {ticker}
            </span>
            {row.asset_label.includes('(') && (
              <span className="block text-[9px] text-[#475569] truncate max-w-[140px] leading-tight">
                {row.asset_label.split('(')[0].trim()}
              </span>
            )}
          </button>
        </td>
        <td className="px-3 py-2 text-[10px] text-[#94a3b8] max-w-[120px] truncate">{row.sector}</td>
        <td className="px-3 py-2 text-center"><SignalBadge label={label} /></td>
        <td className="px-3 py-2 text-center"><MomentumBadge value={row.momentum_score} /></td>
        {horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          if (!sig) return <td key={h} className="px-2 py-2 text-center text-[#64748b] text-[10px]">{'\u2014'}</td>;
          const { dir, color } = signalDirection(sig.exp_ret);
          return (
            <td key={h} className="px-2 py-2 text-center">
              <DirectionArrow direction={dir} />
              <span className="text-[10px] font-medium" style={{ color }}>
                {sig.exp_ret != null ? `${(sig.exp_ret * 100).toFixed(1)}%` : '\u2014'}
              </span>
              <span className="block text-[9px] text-[#64748b]">
                {sig.p_up != null ? `p${(sig.p_up * 100).toFixed(0)}` : ''}
              </span>
              <ExhaustionBar ueUp={sig.ue_up} ueDown={sig.ue_down} />
            </td>
          );
        })}
        <td className="px-3 py-2 text-center"><CrashRiskBadge score={row.crash_risk_score} /></td>
        <td className="px-2 py-2">
          <button onClick={onToggleExpand} className="p-1 rounded hover:bg-[#16213e] transition-colors" title="Show chart">
            <BarChart3 className={`w-3.5 h-3.5 transition-colors ${isExpanded ? 'text-[#42A5F5]' : 'text-[#475569] hover:text-[#94a3b8]'}`} />
          </button>
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={horizons.length + 5} className="p-0">
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
    const lineColor = isUp ? '#00E676' : '#FF1744';
    const gradient = ctx.createLinearGradient(0, padding, 0, rect.height);
    gradient.addColorStop(0, isUp ? 'rgba(0, 230, 118, 0.15)' : 'rgba(255, 23, 68, 0.15)');
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
              <div className="w-4 h-4 border-2 border-[#2a2a4a] border-t-[#42A5F5] rounded-full animate-spin" />
            </div>
          ) : error ? (
            <div className="h-full flex items-center justify-center text-[10px] text-[#64748b]">Chart unavailable</div>
          ) : (
            <canvas ref={canvasRef} className="w-full h-full" />
          )}
        </div>

        {/* Price info */}
        <div className="flex-shrink-0 text-right space-y-1">
          <p className="text-sm font-bold text-[#e2e8f0] tabular-nums">
            {lastPrice > 0 ? (lastPrice < 10 ? lastPrice.toFixed(4) : lastPrice.toFixed(2)) : '--'}
          </p>
          <p className={`text-xs font-semibold tabular-nums ${isUp ? 'text-[#00E676]' : 'text-[#FF1744]'}`}>
            {isUp ? '+' : ''}{changePct.toFixed(2)}%
          </p>
          <p className="text-[9px] text-[#475569]">3M change</p>
        </div>

        {/* Navigate to full chart */}
        <button
          onClick={onNavigateChart}
          className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-[11px] font-medium text-[#42A5F5] bg-[#42A5F5]/10 hover:bg-[#42A5F5]/20 transition-all"
        >
          <ExternalLink className="w-3 h-3" />
          Full Chart
        </button>
      </div>
    </div>
  );
}

/* ── Sector signal row (compact, no mini chart) ──────────────────── */
function SectorSignalRow({ row, horizons, highlighted }: { row: SummaryRow; horizons: number[]; highlighted?: boolean }) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  const ticker = extractTicker(row.asset_label);
  return (
    <tr className={`border-b border-[#2a2a4a]/50 row-glow transition ${highlighted ? 'animate-signal-flash' : ''}`}>
      <td className="px-4 py-2 whitespace-nowrap">
        <span className="font-semibold text-[#e2e8f0] text-xs">{ticker}</span>
        {row.asset_label.includes('(') && (
          <span className="block text-[9px] text-[#475569] truncate max-w-[140px] leading-tight">
            {row.asset_label.split('(')[0].trim()}
          </span>
        )}
      </td>
      <td className="px-3 py-2 text-center"><SignalBadge label={label} /></td>
      <td className="px-3 py-2 text-center"><MomentumBadge value={row.momentum_score} /></td>
      {horizons.map((h) => {
        const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
        if (!sig) return <td key={h} className="px-2 py-2 text-center text-[#64748b] text-[10px]">{'\u2014'}</td>;
        const { dir, color } = signalDirection(sig.exp_ret);
        return (
          <td key={h} className="px-2 py-2 text-center">
            <DirectionArrow direction={dir} />
            <span className="text-[10px] font-medium" style={{ color }}>
              {sig.exp_ret != null ? `${(sig.exp_ret * 100).toFixed(1)}%` : '\u2014'}
            </span>
            <span className="block text-[9px] text-[#64748b]">
              {sig.p_up != null ? `p${(sig.p_up * 100).toFixed(0)}` : ''}
            </span>
            <ExhaustionBar ueUp={sig.ue_up} ueDown={sig.ue_down} />
          </td>
        );
      })}
      <td className="px-3 py-2 text-center"><CrashRiskBadge score={row.crash_risk_score} /></td>
    </tr>
  );
}

/* ── Badges & Helpers ────────────────────────────────────────────── */
function SignalBadge({ label }: { label: string }) {
  const config: Record<string, { icon: string; color: string; bg: string }> = {
    'STRONG BUY': { icon: '\u25B2\u25B2', color: '#00E676', bg: '#00E67620' },
    'BUY': { icon: '\u25B2', color: '#66BB6A', bg: '#66BB6A20' },
    'HOLD': { icon: '\u2014', color: '#64748b', bg: '#64748b15' },
    'SELL': { icon: '\u25BC', color: '#EF5350', bg: '#EF535020' },
    'STRONG SELL': { icon: '\u25BC\u25BC', color: '#FF1744', bg: '#FF174420' },
    'EXIT': { icon: '\u2298', color: '#FFB300', bg: '#FFB30020' },
  };
  const c = config[label] || config['HOLD'];
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded text-[10px] font-semibold"
      style={{ color: c.color, background: c.bg }}>
      {c.icon} {label}
    </span>
  );
}

function CrashRiskBadge({ score }: { score: number }) {
  const s = score ?? 0;
  const color = s < 30 ? '#00E676' : s < 60 ? '#FFB300' : '#FF1744';
  return (
    <span className="inline-block px-2 py-0.5 rounded text-[10px] font-medium"
      style={{ color, background: `${color}20` }}>
      {s.toFixed(0)}%
    </span>
  );
}

function MomentumBadge({ value }: { value: number }) {
  const v = value ?? 0;
  const color = v > 1 ? '#00E676' : v < -1 ? '#FF1744' : '#64748b';
  return (
    <span className="text-[10px] font-medium" style={{ color }}>
      {v > 0 ? '+' : ''}{v.toFixed(1)}%
    </span>
  );
}

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
  const accent = color === 'green' ? '#00E676' : '#FF1744';
  const top5 = signals.slice(0, 5);

  return (
    <div className={`glass-card p-5 hover-lift ${color === 'green' ? 'glow-green' : 'glow-red'}`}>
      <div className="flex items-center gap-2.5 mb-4">
        <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: `${accent}10` }}>
          <Icon className="w-4 h-4" style={{ color: accent }} />
        </div>
        <h3 className="text-[13px] font-medium" style={{ color: accent }}>{title}</h3>
        <span className="ml-auto text-[11px] text-[#64748b] tabular-nums">{signals.length} signals</span>
      </div>
      {top5.length === 0 ? (
        <p className="text-xs text-[#475569]">No signals</p>
      ) : (
        <div className="space-y-2">
          {top5.map((s, i) => (
            <div key={i} className="flex items-center justify-between text-xs py-1 border-b border-white/[0.03] last:border-0">
              <span className="text-[#f1f5f9] font-semibold tracking-wide">{s.ticker || '\u2014'}</span>
              <span className="text-[#64748b]">{s.horizon_days != null ? formatHorizon(s.horizon_days) : '\u2014'}</span>
              <span className="font-semibold tabular-nums" style={{ color: accent }}>{s.expected_return_pct != null ? `${s.expected_return_pct.toFixed(1)}%` : '\u2014'}</span>
              <span className="text-[#475569] tabular-nums">p={s.probability_up != null ? s.probability_up.toFixed(2) : '\u2014'}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
