import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { useState, useMemo, useEffect } from 'react';
import { api } from '../api';
import type { IndicatorStrategy, IndicatorFamily, IndicatorBacktestStatus } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  TrendingUp, Trophy, Search, ChevronDown, ChevronUp,
  ArrowUpDown, RefreshCw, Filter, Star, Target, BarChart3,
  Play, Loader2, CheckCircle, XCircle, Zap,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer, ScatterChart, Scatter, Cell,
} from 'recharts';

type SortKey = 'rank' | 'composite' | 'sharpe' | 'sortino' | 'cagr' | 'max_dd' | 'buy_hit' | 'win_rate' | 'profit_factor' | 'exposure';
type ViewTab = 'leaderboard' | 'top10' | 'families' | 'detail';

const FAMILY_COLORS: Record<string, string> = {
  'Trend & Momentum': '#8b5cf6',
  'Mean Reversion': '#06b6d4',
  'Volatility': '#f59e0b',
  'Volume & Microstructure': '#10b981',
  'Pattern & Fractal': '#ec4899',
  'Oscillator & Cycle': '#3b82f6',
  'Multi-Factor & Regime': '#ef4444',
  'Derivatives-Informed': '#f97316',
  'Cross-Asset Macro': '#14b8a6',
  'Hybrid Ensemble': '#a855f7',
};

function getFamilyColor(family: string): string {
  return FAMILY_COLORS[family] || '#6366f1';
}

function metricColor(value: number | null, good: 'high' | 'low', thresholds: [number, number]): string {
  if (value == null) return 'var(--text-muted)';
  const isGood = good === 'high' ? value >= thresholds[1] : value <= thresholds[0];
  const isBad = good === 'high' ? value <= thresholds[0] : value >= thresholds[1];
  if (isGood) return 'var(--accent-emerald)';
  if (isBad) return 'var(--accent-rose)';
  return 'var(--accent-amber)';
}

export default function IndicatorsPage() {
  const [tab, setTab] = useState<ViewTab>('leaderboard');
  const [search, setSearch] = useState('');
  const [familyFilter, setFamilyFilter] = useState<string>('');
  const [sortKey, setSortKey] = useState<SortKey>('rank');
  const [sortAsc, setSortAsc] = useState(true);
  const [selectedStrategy, setSelectedStrategy] = useState<number | null>(null);
  const [showCount, setShowCount] = useState(100);
  const queryClient = useQueryClient();

  const lbQ = useQuery({
    queryKey: ['indicatorsLeaderboard', 0, familyFilter || undefined],
    queryFn: () => api.indicatorsLeaderboard(0, familyFilter || undefined),
    staleTime: 300_000,
  });
  const top10Q = useQuery({
    queryKey: ['indicatorsTop10'],
    queryFn: api.indicatorsTop10,
    staleTime: 300_000,
    enabled: tab === 'top10',
  });
  const familiesQ = useQuery({
    queryKey: ['indicatorsFamilies'],
    queryFn: api.indicatorsFamilies,
    staleTime: 300_000,
    enabled: tab === 'families' || tab === 'leaderboard',
  });
  const detailQ = useQuery({
    queryKey: ['indicatorsDetail', selectedStrategy],
    queryFn: () => api.indicatorsStrategy(selectedStrategy!),
    staleTime: 300_000,
    enabled: tab === 'detail' && selectedStrategy != null,
  });

  // Backtest status polling
  const backtestStatusQ = useQuery({
    queryKey: ['indicatorsBacktestStatus'],
    queryFn: api.indicatorsBacktestStatus,
    refetchInterval: (query) => {
      const data = query.state.data as IndicatorBacktestStatus | undefined;
      return data?.running ? 2000 : false;
    },
    staleTime: 1000,
  });

  const runBacktestMut = useMutation({
    mutationFn: (mode: 'quick' | 'full') => api.indicatorsRunBacktest(mode),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['indicatorsBacktestStatus'] });
    },
  });

  // When backtest finishes, refresh data
  const btStatus = backtestStatusQ.data;
  useEffect(() => {
    if (btStatus && !btStatus.running && btStatus.finished_at && btStatus.exit_code === 0) {
      // Invalidate all indicator queries to get fresh data
      queryClient.invalidateQueries({ queryKey: ['indicatorsLeaderboard'] });
      queryClient.invalidateQueries({ queryKey: ['indicatorsTop10'] });
      queryClient.invalidateQueries({ queryKey: ['indicatorsFamilies'] });
    }
  }, [btStatus?.finished_at, btStatus?.exit_code, queryClient]);

  const strategies = lbQ.data?.strategies ?? [];
  const total = lbQ.data?.total ?? 0;

  const filtered = useMemo(() => {
    let items = strategies;
    if (search) {
      const q = search.toLowerCase();
      items = items.filter(s =>
        s.name.toLowerCase().includes(q) ||
        s.family.toLowerCase().includes(q) ||
        String(s.id).includes(q)
      );
    }
    // Sort
    const key = sortKey;
    items = [...items].sort((a, b) => {
      const av = (a as Record<string, unknown>)[key] as number ?? 0;
      const bv = (b as Record<string, unknown>)[key] as number ?? 0;
      return sortAsc ? av - bv : bv - av;
    });
    return items;
  }, [strategies, search, sortKey, sortAsc]);

  const visible = filtered.slice(0, showCount);

  const toggleSort = (key: SortKey) => {
    if (sortKey === key) setSortAsc(!sortAsc);
    else { setSortKey(key); setSortAsc(key === 'rank' || key === 'max_dd'); }
  };

  const openDetail = (id: number) => {
    setSelectedStrategy(id);
    setTab('detail');
  };

  if (lbQ.isLoading) return <LoadingSpinner text="Loading indicator strategies..." />;

  // Stats
  const avgComposite = strategies.length
    ? (strategies.reduce((s, x) => s + x.composite, 0) / strategies.length).toFixed(1)
    : '0';
  const avgSharpe = strategies.length
    ? (strategies.reduce((s, x) => s + (x.sharpe ?? 0), 0) / strategies.length).toFixed(3)
    : '0';
  const positiveCagr = strategies.filter(s => (s.cagr_diff ?? 0) > 0).length;

  const tabs: { id: ViewTab; label: string; icon: typeof TrendingUp }[] = [
    { id: 'leaderboard', label: 'Leaderboard', icon: BarChart3 },
    { id: 'top10', label: 'Elite Top 10', icon: Trophy },
    { id: 'families', label: 'Families', icon: Filter },
    { id: 'detail', label: 'Strategy Detail', icon: Target },
  ];

  return (
    <>
      <PageHeader
        title="Indicators"
        action={
          <div className="flex items-center gap-2">
            {/* Backtest status indicator */}
            {btStatus?.running && (
              <div className="flex items-center gap-2 px-3 py-2 rounded-xl text-xs" style={{ background: 'rgba(59, 130, 246, 0.1)', border: '1px solid rgba(59, 130, 246, 0.3)', color: '#93c5fd' }}>
                <Loader2 size={13} className="animate-spin" />
                <span>{btStatus.progress || 'Running...'}</span>
                {btStatus.elapsed_seconds != null && (
                  <span className="font-mono opacity-70">{Math.round(btStatus.elapsed_seconds)}s</span>
                )}
              </div>
            )}
            {btStatus && !btStatus.running && btStatus.finished_at && (
              <div className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs" style={{
                background: btStatus.exit_code === 0 ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)',
                border: btStatus.exit_code === 0 ? '1px solid rgba(16, 185, 129, 0.3)' : '1px solid rgba(239, 68, 68, 0.3)',
                color: btStatus.exit_code === 0 ? '#6ee7b7' : '#fca5a5',
              }}>
                {btStatus.exit_code === 0 ? <CheckCircle size={13} /> : <XCircle size={13} />}
                <span>{btStatus.exit_code === 0 ? 'Done' : 'Failed'}</span>
                {btStatus.elapsed_seconds != null && (
                  <span className="font-mono opacity-70">{Math.round(btStatus.elapsed_seconds)}s</span>
                )}
              </div>
            )}

            {/* Run backtest buttons */}
            <button
              onClick={() => runBacktestMut.mutate('quick')}
              disabled={btStatus?.running || runBacktestMut.isPending}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs transition-all duration-200 disabled:opacity-40"
              style={{ background: 'rgba(59, 130, 246, 0.15)', color: '#93c5fd', border: '1px solid rgba(59, 130, 246, 0.3)' }}
              title="Quick backtest: 10 assets, ~3 min"
            >
              <Zap size={13} />
              Quick
            </button>
            <button
              onClick={() => runBacktestMut.mutate('full')}
              disabled={btStatus?.running || runBacktestMut.isPending}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-xs transition-all duration-200 disabled:opacity-40"
              style={{ background: 'rgba(16, 185, 129, 0.15)', color: '#6ee7b7', border: '1px solid rgba(16, 185, 129, 0.3)' }}
              title="Full backtest: all assets, ~30 min"
            >
              <Play size={13} />
              Full Backtest
            </button>

            <button
              onClick={() => { lbQ.refetch(); top10Q.refetch(); familiesQ.refetch(); }}
              disabled={lbQ.isFetching}
              className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm transition-all duration-200 disabled:opacity-50"
              style={{ background: 'var(--violet-8)', color: '#b49aff', border: '1px solid var(--violet-12)' }}
            >
              <RefreshCw size={14} className={lbQ.isFetching ? 'animate-spin' : ''} />
              Refresh
            </button>
          </div>
        }
      />

      {/* Stat Cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
        <StatCard
          title="Total Strategies"
          value={total}
          subtitle="across 10 families"
          icon={<TrendingUp size={18} />}
          color="purple"
        />
        <StatCard
          title="Avg Composite"
          value={avgComposite}
          subtitle="scoring range 0-100"
          icon={<Star size={18} />}
          color="amber"
        />
        <StatCard
          title="Avg Sharpe"
          value={avgSharpe}
          subtitle="median across all assets"
          icon={<BarChart3 size={18} />}
          color="blue"
        />
        <StatCard
          title="Beat Buy & Hold"
          value={positiveCagr}
          subtitle={`of ${total} strategies`}
          icon={<Trophy size={18} />}
          color="green"
        />
      </div>

      {/* Tabs */}
      <div className="flex gap-1 mb-4 overflow-x-auto pb-1">
        {tabs.map(t => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap transition-all duration-200"
            style={{
              background: tab === t.id ? 'var(--violet-10)' : 'transparent',
              color: tab === t.id ? '#d4c4ff' : 'var(--text-muted)',
              border: tab === t.id ? '1px solid var(--violet-12)' : '1px solid transparent',
            }}
          >
            <t.icon size={13} />
            {t.label}
          </button>
        ))}
      </div>

      {/* ── Leaderboard Tab ──────────────────────────────── */}
      {tab === 'leaderboard' && (
        <div>
          {/* Filters */}
          <div className="flex flex-wrap gap-2 mb-3">
            <div className="relative flex-1 min-w-[200px]">
              <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
              <input
                type="text"
                value={search}
                onChange={e => setSearch(e.target.value)}
                placeholder="Search strategies..."
                className="w-full pl-9 pr-3 py-2 rounded-lg text-sm"
                style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: 'var(--text-primary)' }}
              />
            </div>
            <select
              value={familyFilter}
              onChange={e => setFamilyFilter(e.target.value)}
              className="px-3 py-2 rounded-lg text-sm"
              style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', color: 'var(--text-primary)' }}
            >
              <option value="">All Families</option>
              {(familiesQ.data ?? []).map(f => (
                <option key={f.name} value={f.name}>{f.name} ({f.count})</option>
              ))}
            </select>
          </div>

          {/* Table */}
          <div className="overflow-x-auto rounded-xl" style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}>
            <table className="w-full text-xs">
              <thead>
                <tr style={{ borderBottom: '1px solid var(--glass-border)' }}>
                  {([
                    ['rank', '#', '40px'],
                    ['composite', 'Score', '60px'],
                    ['', 'Strategy', ''],
                    ['', 'Family', ''],
                    ['sharpe', 'Sharpe', '65px'],
                    ['sortino', 'Sortino', '65px'],
                    ['cagr', 'CAGR%', '60px'],
                    ['max_dd', 'MaxDD%', '65px'],
                    ['buy_hit', 'Hit%', '55px'],
                    ['win_rate', 'Win%', '55px'],
                    ['profit_factor', 'PF', '50px'],
                    ['exposure', 'Exp%', '55px'],
                  ] as [SortKey | '', string, string][]).map(([key, label, w]) => (
                    <th
                      key={label}
                      className="px-2 py-2 text-left font-medium whitespace-nowrap"
                      style={{ color: 'var(--text-muted)', width: w || undefined, cursor: key ? 'pointer' : 'default' }}
                      onClick={() => key && toggleSort(key as SortKey)}
                    >
                      <span className="flex items-center gap-1">
                        {label}
                        {key && sortKey === key && (sortAsc ? <ChevronUp size={10} /> : <ChevronDown size={10} />)}
                        {key && sortKey !== key && <ArrowUpDown size={9} style={{ opacity: 0.3 }} />}
                      </span>
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {visible.map((s, i) => (
                  <tr
                    key={s.id}
                    onClick={() => openDetail(s.id)}
                    className="cursor-pointer transition-colors duration-150"
                    style={{
                      borderBottom: '1px solid var(--glass-border)',
                      background: i % 2 === 0 ? 'transparent' : 'rgba(139, 92, 246, 0.02)',
                    }}
                    onMouseEnter={e => (e.currentTarget.style.background = 'rgba(139, 92, 246, 0.08)')}
                    onMouseLeave={e => (e.currentTarget.style.background = i % 2 === 0 ? 'transparent' : 'rgba(139, 92, 246, 0.02)')}
                  >
                    <td className="px-2 py-1.5 font-mono" style={{ color: s.rank <= 10 ? 'var(--accent-amber)' : 'var(--text-muted)' }}>
                      {s.rank}
                    </td>
                    <td className="px-2 py-1.5 font-mono font-bold" style={{ color: s.composite >= 60 ? 'var(--accent-emerald)' : s.composite >= 50 ? 'var(--accent-amber)' : 'var(--text-muted)' }}>
                      {s.composite.toFixed(1)}
                    </td>
                    <td className="px-2 py-1.5 font-medium" style={{ color: 'var(--text-primary)' }}>
                      <span className="font-mono text-[10px] mr-1.5" style={{ color: 'var(--text-muted)' }}>S{String(s.id).padStart(3, '0')}</span>
                      {s.name}
                    </td>
                    <td className="px-2 py-1.5">
                      <span
                        className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                        style={{ background: getFamilyColor(s.family) + '22', color: getFamilyColor(s.family) }}
                      >
                        {s.family.split(' ')[0]}
                      </span>
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.sharpe, 'high', [0, 0.5]) }}>
                      {s.sharpe?.toFixed(3) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.sortino, 'high', [0, 0.5]) }}>
                      {s.sortino?.toFixed(3) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.cagr, 'high', [0, 10]) }}>
                      {s.cagr?.toFixed(1) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.max_dd, 'low', [-50, -20]) }}>
                      {s.max_dd?.toFixed(1) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.buy_hit, 'high', [50, 55]) }}>
                      {s.buy_hit?.toFixed(1) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.win_rate, 'high', [48, 52]) }}>
                      {s.win_rate?.toFixed(1) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: metricColor(s.profit_factor, 'high', [1.0, 1.3]) }}>
                      {s.profit_factor?.toFixed(2) ?? '-'}
                    </td>
                    <td className="px-2 py-1.5 font-mono" style={{ color: 'var(--text-muted)' }}>
                      {s.exposure?.toFixed(0) ?? '-'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {filtered.length > showCount && (
            <div className="flex justify-center mt-3 gap-2">
              <button
                onClick={() => setShowCount(c => c + 100)}
                className="px-4 py-2 rounded-lg text-xs"
                style={{ background: 'var(--violet-8)', color: '#b49aff', border: '1px solid var(--violet-12)' }}
              >
                Show More ({filtered.length - showCount} remaining)
              </button>
              <button
                onClick={() => setShowCount(filtered.length)}
                className="px-4 py-2 rounded-lg text-xs"
                style={{ background: 'var(--violet-8)', color: '#b49aff', border: '1px solid var(--violet-12)' }}
              >
                Show All
              </button>
            </div>
          )}
        </div>
      )}

      {/* ── Top 10 Tab ───────────────────────────────────── */}
      {tab === 'top10' && (
        <div>
          {top10Q.isLoading && <LoadingSpinner text="Loading elite strategies..." />}
          {top10Q.data && (
            <div className="grid gap-3">
              {top10Q.data.map((s, i) => (
                <div
                  key={s.id}
                  onClick={() => openDetail(s.id)}
                  className="cursor-pointer rounded-xl p-4 transition-all duration-200"
                  style={{
                    background: 'var(--glass-bg)',
                    border: i < 3 ? `1px solid ${['#fbbf24', '#94a3b8', '#d97706'][i]}44` : '1px solid var(--glass-border)',
                    boxShadow: i < 3 ? `0 0 20px ${['#fbbf24', '#94a3b8', '#d97706'][i]}11` : 'none',
                  }}
                  onMouseEnter={e => (e.currentTarget.style.background = 'rgba(139, 92, 246, 0.06)')}
                  onMouseLeave={e => (e.currentTarget.style.background = 'var(--glass-bg)')}
                >
                  <div className="flex items-center gap-4">
                    <div
                      className="w-10 h-10 rounded-lg flex items-center justify-center font-bold text-lg"
                      style={{
                        background: i < 3 ? `${['#fbbf24', '#94a3b8', '#d97706'][i]}22` : 'var(--violet-8)',
                        color: i < 3 ? ['#fbbf24', '#94a3b8', '#d97706'][i] : '#b49aff',
                      }}
                    >
                      {i + 1}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-mono text-xs" style={{ color: 'var(--text-muted)' }}>S{String(s.id).padStart(3, '0')}</span>
                        <span className="font-medium" style={{ color: 'var(--text-primary)' }}>{s.name}</span>
                        <span
                          className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                          style={{ background: getFamilyColor(s.family) + '22', color: getFamilyColor(s.family) }}
                        >
                          {s.family}
                        </span>
                      </div>
                      <div className="flex gap-4 mt-1 text-xs font-mono">
                        <span style={{ color: 'var(--accent-emerald)' }}>Composite: {s.composite.toFixed(1)}</span>
                        <span style={{ color: metricColor(s.sharpe, 'high', [0, 0.5]) }}>Sharpe: {s.sharpe?.toFixed(3) ?? '-'}</span>
                        <span style={{ color: metricColor(s.sortino, 'high', [0, 0.5]) }}>Sortino: {s.sortino?.toFixed(3) ?? '-'}</span>
                        <span style={{ color: metricColor(s.cagr, 'high', [0, 10]) }}>CAGR: {s.cagr?.toFixed(1) ?? '-'}%</span>
                        <span style={{ color: metricColor(s.max_dd, 'low', [-50, -20]) }}>MaxDD: {s.max_dd?.toFixed(1) ?? '-'}%</span>
                        <span style={{ color: metricColor(s.buy_hit, 'high', [50, 55]) }}>Hit: {s.buy_hit?.toFixed(1) ?? '-'}%</span>
                        <span style={{ color: metricColor(s.profit_factor, 'high', [1.0, 1.3]) }}>PF: {s.profit_factor?.toFixed(2) ?? '-'}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* ── Families Tab ─────────────────────────────────── */}
      {tab === 'families' && (
        <div>
          {familiesQ.isLoading && <LoadingSpinner text="Loading families..." />}
          {familiesQ.data && (
            <>
              {/* Bar chart */}
              <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}>
                <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Average Composite Score by Family</h3>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={familiesQ.data} layout="vertical" margin={{ left: 150 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.1)" />
                    <XAxis type="number" domain={[0, 100]} tick={{ fill: 'var(--text-muted)', fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" tick={{ fill: 'var(--text-muted)', fontSize: 11 }} width={145} />
                    <Tooltip
                      contentStyle={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', borderRadius: 8, fontSize: 12 }}
                      labelStyle={{ color: 'var(--text-primary)' }}
                    />
                    <Bar dataKey="avg_composite" radius={[0, 4, 4, 0]}>
                      {familiesQ.data.map((f: IndicatorFamily) => (
                        <Cell key={f.name} fill={getFamilyColor(f.name)} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Family cards */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {familiesQ.data.map((f: IndicatorFamily) => (
                  <div
                    key={f.name}
                    className="rounded-xl p-4 cursor-pointer transition-all duration-200"
                    style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}
                    onClick={() => { setFamilyFilter(f.name); setTab('leaderboard'); }}
                    onMouseEnter={e => (e.currentTarget.style.borderColor = getFamilyColor(f.name))}
                    onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--glass-border)')}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <span className="font-medium text-sm" style={{ color: getFamilyColor(f.name) }}>{f.name}</span>
                      <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>{f.count} strategies</span>
                    </div>
                    <div className="flex items-center gap-2">
                      <div className="flex-1 h-2 rounded-full" style={{ background: 'rgba(139,92,246,0.1)' }}>
                        <div
                          className="h-full rounded-full transition-all duration-500"
                          style={{ width: `${f.avg_composite}%`, background: getFamilyColor(f.name) }}
                        />
                      </div>
                      <span className="text-xs font-mono font-bold" style={{ color: getFamilyColor(f.name) }}>
                        {f.avg_composite.toFixed(1)}
                      </span>
                    </div>
                    <div className="mt-2 text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>
                      IDs: {f.ids[0]}..{f.ids[f.ids.length - 1]}
                    </div>
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      )}

      {/* ── Detail Tab ───────────────────────────────────── */}
      {tab === 'detail' && (
        <div>
          {!selectedStrategy && (
            <div className="text-center py-12" style={{ color: 'var(--text-muted)' }}>
              <Target size={48} className="mx-auto mb-3 opacity-30" />
              <p className="text-sm">Select a strategy from the Leaderboard or Top 10 to view details</p>
            </div>
          )}
          {selectedStrategy && detailQ.isLoading && <LoadingSpinner text="Loading strategy detail..." />}
          {selectedStrategy && detailQ.data && (() => {
            const d = detailQ.data;
            const agg = d.aggregate as Record<string, unknown>;
            const assets = d.per_asset;
            const sortedAssets = [...assets].sort((a, b) => (b.sharpe ?? 0) - (a.sharpe ?? 0));

            return (
              <div>
                {/* Header */}
                <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}>
                  <div className="flex items-center gap-3 mb-3">
                    <span className="font-mono text-sm" style={{ color: 'var(--text-muted)' }}>S{String(d.id).padStart(3, '0')}</span>
                    <h2 className="text-lg font-bold" style={{ color: 'var(--text-primary)' }}>{d.name}</h2>
                    <span
                      className="px-2 py-0.5 rounded text-xs font-medium"
                      style={{ background: getFamilyColor(d.family) + '22', color: getFamilyColor(d.family) }}
                    >
                      {d.family}
                    </span>
                  </div>
                  <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3 text-xs">
                    {[
                      ['Composite', agg.composite, 'high', [40, 60]],
                      ['Sharpe', agg.med_sharpe, 'high', [0, 0.5]],
                      ['Sortino', agg.med_sortino, 'high', [0, 0.5]],
                      ['CAGR%', agg.med_cagr, 'high', [0, 10]],
                      ['B&H CAGR%', agg.med_bh_cagr, 'high', [0, 10]],
                      ['CAGR Diff', agg.med_cagr_diff, 'high', [-5, 5]],
                      ['MaxDD%', agg.med_max_dd, 'low', [-50, -20]],
                      ['Buy Hit%', agg.med_buy_hit, 'high', [50, 55]],
                      ['Win Rate%', agg.med_win_rate, 'high', [48, 52]],
                      ['Profit Factor', agg.med_profit_factor, 'high', [1.0, 1.3]],
                      ['Exposure%', agg.med_exposure, 'high', [20, 60]],
                      ['Assets', assets.length, 'high', [50, 100]],
                    ].map(([label, val, dir, thresh]) => (
                      <div key={label as string} className="rounded-lg p-2" style={{ background: 'rgba(139,92,246,0.04)' }}>
                        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label as string}</div>
                        <div
                          className="font-mono font-bold text-sm"
                          style={{ color: metricColor(val as number, dir as 'high' | 'low', thresh as [number, number]) }}
                        >
                          {typeof val === 'number' ? (Number.isInteger(val) ? val : (val as number).toFixed(2)) : '-'}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Scatter chart: Sharpe vs CAGR per asset */}
                {assets.length > 0 && (
                  <div className="rounded-xl p-4 mb-4" style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}>
                    <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-primary)' }}>Per-Asset: Sharpe vs CAGR</h3>
                    <ResponsiveContainer width="100%" height={250}>
                      <ScatterChart margin={{ left: 10, right: 10 }}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(139,92,246,0.1)" />
                        <XAxis type="number" dataKey="sharpe" name="Sharpe" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                        <YAxis type="number" dataKey="cagr" name="CAGR%" tick={{ fill: 'var(--text-muted)', fontSize: 10 }} />
                        <Tooltip
                          contentStyle={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)', borderRadius: 8, fontSize: 11 }}
                          formatter={(v: number, name: string) => [typeof v === 'number' ? v.toFixed(2) : v, name]}
                          labelFormatter={() => ''}
                        />
                        <Scatter data={assets} fill={getFamilyColor(d.family)}>
                          {assets.map((a, i) => (
                            <Cell key={i} fill={(a.sharpe ?? 0) > 0.5 ? 'var(--accent-emerald)' : (a.sharpe ?? 0) > 0 ? 'var(--accent-amber)' : 'var(--accent-rose)'} />
                          ))}
                        </Scatter>
                      </ScatterChart>
                    </ResponsiveContainer>
                  </div>
                )}

                {/* Per-asset table */}
                <div className="rounded-xl overflow-hidden" style={{ background: 'var(--glass-bg)', border: '1px solid var(--glass-border)' }}>
                  <h3 className="text-sm font-medium p-3" style={{ color: 'var(--text-primary)' }}>Per-Asset Results ({assets.length} assets)</h3>
                  <div className="overflow-x-auto max-h-[400px] overflow-y-auto">
                    <table className="w-full text-xs">
                      <thead className="sticky top-0" style={{ background: 'var(--glass-bg)' }}>
                        <tr style={{ borderBottom: '1px solid var(--glass-border)' }}>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>Symbol</th>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>Sharpe</th>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>CAGR%</th>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>MaxDD%</th>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>Return%</th>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>Win%</th>
                          <th className="px-2 py-1.5 text-left font-medium" style={{ color: 'var(--text-muted)' }}>Trades</th>
                        </tr>
                      </thead>
                      <tbody>
                        {sortedAssets.map(a => (
                          <tr key={a.symbol} style={{ borderBottom: '1px solid var(--glass-border)' }}>
                            <td className="px-2 py-1 font-mono font-medium" style={{ color: 'var(--text-primary)' }}>{a.symbol}</td>
                            <td className="px-2 py-1 font-mono" style={{ color: metricColor(a.sharpe, 'high', [0, 0.5]) }}>{a.sharpe?.toFixed(3)}</td>
                            <td className="px-2 py-1 font-mono" style={{ color: metricColor(a.cagr, 'high', [0, 10]) }}>{a.cagr?.toFixed(1)}</td>
                            <td className="px-2 py-1 font-mono" style={{ color: metricColor(a.max_dd, 'low', [-50, -20]) }}>{a.max_dd?.toFixed(1)}</td>
                            <td className="px-2 py-1 font-mono" style={{ color: metricColor(a.total_return, 'high', [0, 50]) }}>{a.total_return?.toFixed(1)}</td>
                            <td className="px-2 py-1 font-mono" style={{ color: metricColor(a.win_rate, 'high', [48, 52]) }}>{a.win_rate?.toFixed(1) ?? '-'}</td>
                            <td className="px-2 py-1 font-mono" style={{ color: 'var(--text-muted)' }}>{a.n_trades}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              </div>
            );
          })()}
        </div>
      )}
    </>
  );
}
