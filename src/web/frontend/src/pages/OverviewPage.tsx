import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  Signal, TrendingUp, TrendingDown, Database, Settings,
  AlertTriangle, CheckCircle, Clock, HeartPulse, Zap,
} from 'lucide-react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { formatModelNameShort } from '../utils/modelNames';
import SignalHeatmap from '../components/SignalHeatmap';

export default function OverviewPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: 60_000,
  });

  const healthQ = useQuery({
    queryKey: ['servicesHealth'],
    queryFn: api.servicesHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const sectorQ = useQuery({
    queryKey: ['signalsBySector'],
    queryFn: api.signalsBySector,
    staleTime: 120_000,
  });

  const strongQ = useQuery({
    queryKey: ['strongSignals'],
    queryFn: api.strongSignals,
    staleTime: 120_000,
  });

  const summaryQ = useQuery({
    queryKey: ['signalSummary'],
    queryFn: api.signalSummary,
    staleTime: 120_000,
  });

  if (isLoading) return <LoadingSpinner text="Loading dashboard..." />;
  if (error || !data) return <div className="text-[#FF1744]">Failed to load overview</div>;

  const { signals, tuning, data: dataStatus } = data;

  // 5-slice signal pie (counts are per-asset, non-cumulative)
  const signalPieData = [
    { name: 'Strong Buy', value: signals.strong_buy_signals || 0, color: '#00E676' },
    { name: 'Buy', value: signals.buy_signals || 0, color: '#66BB6A' },
    { name: 'Hold', value: signals.hold_signals || 0, color: '#64748b' },
    { name: 'Sell', value: signals.sell_signals || 0, color: '#EF5350' },
    { name: 'Strong Sell', value: signals.strong_sell_signals || 0, color: '#FF1744' },
  ].filter(d => d.value > 0);

  const cacheAgeMin = signals.cache_age_seconds
    ? Math.round(signals.cache_age_seconds / 60)
    : null;

  const modelData = Object.entries(tuning.models_distribution || {})
    .map(([name, count]) => ({
      name: formatModelNameShort(name),
      count,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  // Top sectors for mini bar
  const topSectors = (sectorQ.data?.sectors || [])
    .sort((a, b) => (b.strong_buy + b.buy) - (a.strong_buy + a.buy))
    .slice(0, 6);

  // Strong signal counts
  const strongBuyCount = strongQ.data?.strong_buy?.length || signals.strong_buy_signals || 0;
  const strongSellCount = strongQ.data?.strong_sell?.length || signals.strong_sell_signals || 0;

  // Health status
  const healthOk = healthQ.data
    ? healthQ.data.api.status === 'ok' && healthQ.data.signal_cache.status !== 'missing' && healthQ.data.price_data.status === 'ok'
    : undefined;

  // Errors from overview
  const overviewErrors = data.errors || [];

  return (
    <>
      <PageHeader title="Dashboard">
        System-wide snapshot {'\u2014'} {signals.total_assets} assets monitored
      </PageHeader>

      {/* Errors banner */}
      {overviewErrors.length > 0 && (
        <div className="glass-card p-4 mb-5 border-l-2 border-[#FFB300] fade-up">
          <div className="flex items-center gap-2 text-xs text-[#FFB300]">
            <AlertTriangle className="w-4 h-4" />
            <span className="font-medium">Partial data returned</span>
          </div>
          <ul className="mt-1.5 space-y-0.5">
            {overviewErrors.map((e, i) => (
              <li key={i} className="text-[10px] text-[#64748b] pl-6">{e}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-8 fade-up">
        <StatCard
          title="Total Assets"
          value={signals.total_assets}
          subtitle={`${signals.failed} failed`}
          icon={<Signal className="w-5 h-5" />}
          color="blue"
        />
        <StatCard
          title="Strong Buy"
          value={strongBuyCount}
          subtitle={`${signals.buy_signals} total buy`}
          icon={<TrendingUp className="w-5 h-5" />}
          color="green"
        />
        <StatCard
          title="Hold"
          value={signals.hold_signals}
          icon={<Clock className="w-5 h-5" />}
          color="amber"
        />
        <StatCard
          title="Sell"
          value={signals.sell_signals}
          subtitle={strongSellCount > 0 ? `${strongSellCount} strong sell` : undefined}
          icon={<TrendingDown className="w-5 h-5" />}
          color="red"
        />
        <div className="glass-card px-5 py-4 flex items-center gap-3 hover-lift">
          <div className="w-9 h-9 rounded-xl flex items-center justify-center"
               style={{ background: `${healthOk === false ? '#FF1744' : '#00E676'}10` }}>
            <HeartPulse className="w-5 h-5" style={{ color: healthOk === undefined ? '#64748b' : healthOk ? '#00E676' : '#FF1744' }} />
          </div>
          <div>
            <p className="text-lg font-bold text-[#f1f5f9] tabular-nums">{healthOk === undefined ? '...' : healthOk ? 'OK' : 'Issue'}</p>
            <p className="text-[10px] text-[#64748b] tracking-wide">System Health</p>
          </div>
          <span className={`w-2 h-2 rounded-full ml-auto ${healthOk === false ? 'bg-[#FF1744]' : 'bg-[#00E676]'} pulse-dot`} />
        </div>
      </div>

      {/* Second row: Tuning + Data */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-8 fade-up-delay-1">
        <StatCard
          title="Tuned Models"
          value={tuning.total}
          icon={<Settings className="w-5 h-5" />}
          color="purple"
        />
        <StatCard
          title="PIT Pass"
          value={tuning.pit_pass}
          subtitle={`${tuning.pit_fail} failures`}
          icon={<CheckCircle className="w-5 h-5" />}
          color="green"
        />
        <StatCard
          title="Price Files"
          value={dataStatus.total_files}
          subtitle={`${dataStatus.total_size_mb} MB total`}
          icon={<Database className="w-5 h-5" />}
          color="cyan"
        />
        <StatCard
          title="Cache Age"
          value={cacheAgeMin !== null ? `${cacheAgeMin}m` : 'N/A'}
          subtitle={signals.cached ? 'Last computation' : 'No cache'}
          icon={<Clock className="w-5 h-5" />}
          color={cacheAgeMin !== null && cacheAgeMin > 360 ? 'red' : 'amber'}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8 fade-up-delay-2">
        {/* Signal Distribution 5-slice Pie */}
        <div className="glass-card p-6 hover-lift">
          <h3 className="text-[13px] font-medium text-[#94a3b8] mb-5 tracking-wide">Signal Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={signalPieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={85}
                dataKey="value"
                stroke="none"
                strokeWidth={0}
              >
                {signalPieData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} opacity={0.85} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: 'rgba(10, 10, 26, 0.95)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 12,
                  color: '#f1f5f9',
                  fontSize: 12,
                  backdropFilter: 'blur(12px)',
                  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap justify-center gap-4 mt-3">
            {signalPieData.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5 text-[10px]">
                <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                <span className="text-[#64748b]">{d.name}</span>
                <span className="text-[#f1f5f9] font-semibold tabular-nums">{d.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model Distribution Bar */}
        <div className="glass-card p-6 hover-lift">
          <h3 className="text-[13px] font-medium text-[#94a3b8] mb-5 tracking-wide">Model Distribution</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={modelData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} axisLine={false} tickLine={false} />
              <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#94a3b8', fontSize: 9 }} axisLine={false} tickLine={false} />
              <Tooltip
                contentStyle={{
                  background: 'rgba(10, 10, 26, 0.95)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: 12,
                  color: '#f1f5f9',
                  fontSize: 12,
                  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
                }}
              />
              <Bar dataKey="count" fill="#42A5F5" radius={[0, 6, 6, 0]} opacity={0.8} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top Sectors */}
        <div className="glass-card p-6 hover-lift">
          <h3 className="text-[13px] font-medium text-[#94a3b8] mb-5 tracking-wide">Top Sectors</h3>
          {topSectors.length === 0 ? (
            <div className="space-y-3">
              {[1,2,3,4].map(i => <div key={i} className="skeleton h-8" />)}
            </div>
          ) : (
            <div className="space-y-4">
              {topSectors.map((sec) => {
                const bullish = sec.strong_buy + sec.buy;
                const total = sec.asset_count;
                const pct = total > 0 ? (bullish / total * 100) : 0;
                return (
                  <div key={sec.name}>
                    <div className="flex items-center justify-between text-xs mb-1.5">
                      <span className="text-[#f1f5f9] font-medium truncate max-w-[140px]">{sec.name}</span>
                      <span className="text-[#64748b] tabular-nums">{bullish}/{total}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/[0.03] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[#00E676] to-[#66BB6A] bar-fill"
                        style={{ '--bar-width': `${pct}%`, width: `${pct}%` } as React.CSSProperties}
                      />
                    </div>
                  </div>
                );
              })}
            </div>
          )}
        </div>
      </div>

      {/* Top movers (strong signals) */}
      {(strongQ.data?.strong_buy?.length || 0) > 0 && (
        <div className="glass-card p-6 mb-8 fade-up-delay-3">
          <div className="flex items-center gap-2.5 mb-4">
            <div className="w-7 h-7 rounded-lg bg-[#00E676]/10 flex items-center justify-center">
              <Zap className="w-3.5 h-3.5 text-[#00E676]" />
            </div>
            <h3 className="text-[13px] font-medium text-[#94a3b8] tracking-wide">Top Strong Buy Signals</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-3">
            {(strongQ.data?.strong_buy || []).slice(0, 12).map((s, i) => (
              <div key={i} className="bg-white/[0.02] rounded-xl px-4 py-3 hover-lift transition-all border border-white/[0.03]">
                <p className="text-xs font-bold text-[#f1f5f9] tracking-wide">{s.symbol}</p>
                <p className="text-[10px] text-[#64748b] truncate">{s.sector}</p>
                <p className="text-xs text-[#00E676] font-medium mt-0.5">
                  {s.exp_ret >= 0 ? '+' : ''}{(s.exp_ret * 100).toFixed(1)}%
                </p>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Story 6.6: Signal Heatmap */}
      {sectorQ.data?.sectors && summaryQ.data?.horizons && (
        <SignalHeatmap
          sectors={sectorQ.data.sectors}
          horizons={summaryQ.data.horizons}
        />
      )}
    </>
  );
}
