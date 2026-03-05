import { useQuery } from '@tanstack/react-query';
import { useState, useMemo, Component, type ReactNode, type ErrorInfo } from 'react';
import { api } from '../api';
import type { SummaryRow, SectorGroup, StrongSignalEntry, HighConvictionSignal } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  ArrowUpCircle, ArrowDownCircle, Filter, ChevronDown, ChevronRight,
  TrendingUp, TrendingDown, Search,
} from 'lucide-react';

type ViewMode = 'all' | 'sectors' | 'strong';
type SignalFilter = 'all' | 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';

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

function SignalsPageInner() {
  const [view, setView] = useState<ViewMode>('sectors');
  const [filter, setFilter] = useState<SignalFilter>('all');
  const [search, setSearch] = useState('');
  const [expandedSectors, setExpandedSectors] = useState<Set<string>>(new Set());

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
  const horizons = data?.horizons || [];
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
        {rows.length} assets across {horizons.length} horizons
      </PageHeader>

      {/* Stats bar */}
      {stats && (
        <div className="grid grid-cols-3 md:grid-cols-6 gap-3 mb-6">
          <MiniStat label="Strong Buy" value={stats.strong_buy_signals} color="#00E676" icon={'\u25B2\u25B2'} />
          <MiniStat label="Buy" value={stats.buy_signals - stats.strong_buy_signals} color="#66BB6A" icon={'\u25B2'} />
          <MiniStat label="Hold" value={stats.hold_signals} color="#64748b" icon={'\u2014'} />
          <MiniStat label="Sell" value={stats.sell_signals - stats.strong_sell_signals} color="#EF5350" icon={'\u25BC'} />
          <MiniStat label="Strong Sell" value={stats.strong_sell_signals} color="#FF1744" icon={'\u25BC\u25BC'} />
          <MiniStat label="Exit" value={stats.exit_signals} color="#FFB300" icon={'\u2298'} />
        </div>
      )}

      {/* High Conviction Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
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
      <div className="flex flex-wrap items-center gap-3 mb-4">
        {/* View toggle */}
        <div className="flex items-center gap-0.5 glass-card px-2 py-1">
          {([
            { key: 'sectors' as ViewMode, label: 'By Sector' },
            { key: 'strong' as ViewMode, label: 'Strong Signals' },
            { key: 'all' as ViewMode, label: 'All Assets' },
          ]).map(({ key, label }) => (
            <button
              key={key}
              onClick={() => setView(key)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                view === key ? 'bg-[#42A5F5]/20 text-[#42A5F5]' : 'text-[#64748b] hover:text-[#94a3b8]'
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {/* Signal filter */}
        {view !== 'strong' && (
          <div className="flex items-center gap-0.5 glass-card px-2 py-1">
            <Filter className="w-3 h-3 text-[#64748b] mr-1" />
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
                className="px-2 py-1 rounded text-[11px] font-medium transition"
                style={filter === key ? { color: c, background: `${c}20` } : { color: '#64748b' }}
              >
                {label}
              </button>
            ))}
          </div>
        )}

        {/* Search */}
        <div className="flex items-center gap-1.5 glass-card px-2.5 py-1.5">
          <Search className="w-3 h-3 text-[#64748b]" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets..."
            className="bg-transparent text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none w-36"
          />
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
        />
      )}
      {view === 'strong' && (
        <StrongSignalsView
          strongBuy={strongQ.data?.strong_buy || []}
          strongSell={strongQ.data?.strong_sell || []}
        />
      )}
      {view === 'all' && (
        <AllAssetsTable rows={filteredRows} horizons={horizons} />
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
    <div className="glass-card px-3 py-2.5 flex items-center gap-2">
      <span className="text-lg font-bold" style={{ color }}>{icon}</span>
      <div>
        <p className="text-lg font-bold text-[#e2e8f0]">{value}</p>
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
}: {
  sectors: SectorGroup[];
  expandedSectors: Set<string>;
  toggleSector: (name: string) => void;
  horizons: number[];
  search: string;
  filter: SignalFilter;
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
                        {horizons.slice(0, 4).map(h => (
                          <th key={h} className="text-center px-2 py-2 text-[10px] text-[#64748b] font-medium">{h}D</th>
                        ))}
                        <th className="text-center px-2 py-2 text-[10px] text-[#64748b] font-medium">Crash Risk</th>
                      </tr>
                    </thead>
                    <tbody>
                      {assets.map(row => (
                        <SignalRow key={row.asset_label} row={row} horizons={horizons} />
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

/* ── All Assets Table ────────────────────────────────────────────── */
function AllAssetsTable({ rows, horizons }: { rows: SummaryRow[]; horizons: number[] }) {
  const [page, setPage] = useState(0);
  const pageSize = 50;
  const pageRows = rows.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = Math.ceil(rows.length / pageSize);

  return (
    <div className="glass-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#2a2a4a]">
              <th className="text-left px-4 py-3 text-xs text-[#64748b] font-medium uppercase">Asset</th>
              <th className="text-left px-3 py-3 text-xs text-[#64748b] font-medium">Sector</th>
              <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium">Signal</th>
              <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium">Momentum</th>
              {horizons.slice(0, 5).map((h) => (
                <th key={h} className="text-center px-3 py-3 text-xs text-[#64748b] font-medium">{h}D</th>
              ))}
              <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium">Crash Risk</th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row) => (
              <SignalRow key={row.asset_label} row={row} horizons={horizons} />
            ))}
          </tbody>
        </table>
      </div>
      {totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-2 border-t border-[#2a2a4a]">
          <span className="text-xs text-[#64748b]">
            Page {page + 1} of {totalPages} ({rows.length} total)
          </span>
          <div className="flex gap-1">
            <button onClick={() => setPage(Math.max(0, page - 1))} disabled={page === 0}
              className="px-2 py-0.5 rounded text-xs text-[#42A5F5] hover:bg-[#16213e] disabled:opacity-30">Prev</button>
            <button onClick={() => setPage(Math.min(totalPages - 1, page + 1))} disabled={page >= totalPages - 1}
              className="px-2 py-0.5 rounded text-xs text-[#42A5F5] hover:bg-[#16213e] disabled:opacity-30">Next</button>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Signal Row ──────────────────────────────────────────────────── */
function SignalRow({ row, horizons }: { row: SummaryRow; horizons: number[] }) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  return (
    <tr className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30 transition">
      <td className="px-4 py-2 font-medium text-[#e2e8f0] whitespace-nowrap text-xs">{row.asset_label}</td>
      <td className="px-3 py-2 text-[10px] text-[#94a3b8] max-w-[120px] truncate">{row.sector}</td>
      <td className="px-3 py-2 text-center"><SignalBadge label={label} /></td>
      <td className="px-3 py-2 text-center"><MomentumBadge value={row.momentum_score} /></td>
      {horizons.slice(0, (horizons.length > 5 ? 5 : 4)).map((h) => {
        const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
        if (!sig) return <td key={h} className="px-2 py-2 text-center text-[#64748b] text-[10px]">{'\u2014'}</td>;
        const sigLabel = (sig.label || 'HOLD').toUpperCase();
        return (
          <td key={h} className="px-2 py-2 text-center">
            <SignalDot label={sigLabel} />
            <span className="block text-[9px] text-[#64748b]">
              {sig.p_up != null ? `${(sig.p_up * 100).toFixed(0)}%` : '—'} | {sig.exp_ret != null ? `${(sig.exp_ret * 100).toFixed(1)}%` : '—'}
            </span>
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

function SignalDot({ label }: { label: string }) {
  const colorMap: Record<string, string> = {
    'STRONG BUY': '#00E676', 'BUY': '#66BB6A', 'HOLD': '#64748b',
    'SELL': '#EF5350', 'STRONG SELL': '#FF1744', 'EXIT': '#FFB300',
  };
  const c = colorMap[label] || '#64748b';
  return <span className="inline-block w-2 h-2 rounded-full" style={{ background: c }} title={label} />;
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
    <div className={`glass-card p-4 ${color === 'green' ? 'glow-green' : 'glow-red'}`}>
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-4 h-4" style={{ color: accent }} />
        <h3 className="text-sm font-medium" style={{ color: accent }}>{title}</h3>
        <span className="ml-auto text-xs text-[#64748b]">{signals.length} signals</span>
      </div>
      {top5.length === 0 ? (
        <p className="text-xs text-[#64748b]">No signals</p>
      ) : (
        <div className="space-y-1.5">
          {top5.map((s, i) => (
            <div key={i} className="flex items-center justify-between text-xs">
              <span className="text-[#e2e8f0] font-medium">{s.ticker || '—'}</span>
              <span className="text-[#94a3b8]">{s.horizon_days ?? '—'}D</span>
              <span style={{ color: accent }}>{s.expected_return_pct != null ? `${s.expected_return_pct.toFixed(1)}%` : '—'}</span>
              <span className="text-[#64748b]">p={s.probability_up != null ? s.probability_up.toFixed(2) : '—'}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
