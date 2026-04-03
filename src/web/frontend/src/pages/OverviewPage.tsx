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
      <PageHeader title="Dashboard Overview">
        System-wide snapshot {'\u2014'} {signals.total_assets} assets monitored
      </PageHeader>

      {/* Errors banner */}
      {overviewErrors.length > 0 && (
        <div className="glass-card p-3 mb-4 border-l-2 border-[#FFB300]">
          <div className="flex items-center gap-2 text-xs text-[#FFB300]">
            <AlertTriangle className="w-4 h-4" />
            <span className="font-medium">Partial data returned</span>
          </div>
          <ul className="mt-1 space-y-0.5">
            {overviewErrors.map((e, i) => (
              <li key={i} className="text-[10px] text-[#64748b] pl-6">{e}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6">
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
        <div className="glass-card px-4 py-3 flex items-center gap-3">
          <HeartPulse className="w-5 h-5" style={{ color: healthOk === undefined ? '#64748b' : healthOk ? '#00E676' : '#FF1744' }} />
          <div>
            <p className="text-lg font-bold text-[#e2e8f0]">{healthOk === undefined ? '...' : healthOk ? 'OK' : 'Issue'}</p>
            <p className="text-[10px] text-[#64748b]">System Health</p>
          </div>
          <span className={`w-2 h-2 rounded-full ml-auto ${healthOk === false ? 'bg-[#FF1744]' : 'bg-[#00E676]'} pulse-dot`} />
        </div>
      </div>

      {/* Second row: Tuning + Data */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
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
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        {/* Signal Distribution 5-slice Pie */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-4">Signal Distribution</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie
                data={signalPieData}
                cx="50%"
                cy="50%"
                innerRadius={50}
                outerRadius={80}
                dataKey="value"
                stroke="none"
              >
                {signalPieData.map((entry, i) => (
                  <Cell key={i} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip
                contentStyle={{
                  background: '#1a1a2e',
                  border: '1px solid #2a2a4a',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontSize: 12,
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-wrap justify-center gap-3 mt-2">
            {signalPieData.map((d) => (
              <div key={d.name} className="flex items-center gap-1 text-[10px]">
                <span className="w-2 h-2 rounded-full" style={{ background: d.color }} />
                <span className="text-[#94a3b8]">{d.name}:</span>
                <span className="text-[#e2e8f0] font-medium">{d.value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Model Distribution Bar */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-4">Model Distribution</h3>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={modelData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={80} tick={{ fill: '#94a3b8', fontSize: 9 }} />
              <Tooltip
                contentStyle={{
                  background: '#1a1a2e',
                  border: '1px solid #2a2a4a',
                  borderRadius: 8,
                  color: '#e2e8f0',
                  fontSize: 12,
                }}
              />
              <Bar dataKey="count" fill="#42A5F5" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Top Sectors */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-4">Top Sectors by Bullish Signals</h3>
          {topSectors.length === 0 ? (
            <p className="text-xs text-[#64748b]">Loading sectors...</p>
          ) : (
            <div className="space-y-3">
              {topSectors.map((sec) => {
                const bullish = sec.strong_buy + sec.buy;
                const total = sec.asset_count;
                const pct = total > 0 ? (bullish / total * 100) : 0;
                return (
                  <div key={sec.name}>
                    <div className="flex items-center justify-between text-xs mb-1">
                      <span className="text-[#e2e8f0] font-medium truncate max-w-[140px]">{sec.name}</span>
                      <span className="text-[#64748b]">{bullish}/{total}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-[#1a1a2e] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[#00E676] to-[#66BB6A]"
                        style={{ width: `${pct}%` }}
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
        <div className="glass-card p-5">
          <div className="flex items-center gap-2 mb-3">
            <Zap className="w-4 h-4 text-[#00E676]" />
            <h3 className="text-sm font-medium text-[#94a3b8]">Top Strong Buy Signals</h3>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-4 lg:grid-cols-6 gap-2">
            {(strongQ.data?.strong_buy || []).slice(0, 12).map((s, i) => (
              <div key={i} className="bg-[#0f0f23] rounded-lg px-3 py-2">
                <p className="text-xs font-bold text-[#e2e8f0]">{s.symbol}</p>
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
