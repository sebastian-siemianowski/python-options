import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { DashboardSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { DashboardEmpty } from '../components/CosmicEmptyState';
import BriefingCard from '../components/BriefingCard';
import SignalDistributionBar from '../components/SignalDistributionBar';
import ModelLeaderboard from '../components/ModelLeaderboard';
import ConvictionSpotlight from '../components/ConvictionSpotlight';
import {
  Signal, TrendingUp, TrendingDown, Database, Settings,
  AlertTriangle, CheckCircle, Clock, HeartPulse,
} from 'lucide-react';
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

  const riskQ = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 120_000,
  });

  if (isLoading) return <DashboardSkeleton />;
  if (error) return <CosmicErrorCard title="Unable to load dashboard" error={error as Error} onRetry={() => window.location.reload()} />;
  if (!data) return <DashboardEmpty />;

  const { signals, tuning, data: dataStatus } = data;

  // 5-slice signal pie (counts are per-asset, non-cumulative)
  const cacheAgeMin = signals.cache_age_seconds
    ? Math.round(signals.cache_age_seconds / 60)
    : null;

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
        <div className="glass-card p-4 mb-5 fade-up" style={{ borderLeft: '2px solid #f59e0b' }}>
          <div className="flex items-center gap-2 text-xs" style={{ color: '#f59e0b' }}>
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

      {/* Morning Briefing Hero Card */}
      <BriefingCard
        signals={signals}
        tuning={tuning}
        dataStatus={dataStatus}
        risk={riskQ.data}
        strongBuy={strongQ.data?.strong_buy || []}
        strongSell={strongQ.data?.strong_sell || []}
      />

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
               style={{ background: `${healthOk === false ? '#fb7185' : '#34d399'}10` }}>
            <HeartPulse className="w-5 h-5" style={{ color: healthOk === undefined ? '#64748b' : healthOk ? '#34d399' : '#fb7185' }} />
          </div>
          <div>
            <p className="text-lg font-bold text-[#f1f5f9] tabular-nums">{healthOk === undefined ? '...' : healthOk ? 'OK' : 'Issue'}</p>
            <p className="text-[10px] text-[#64748b] tracking-wide">System Health</p>
          </div>
          <span className="w-2 h-2 rounded-full ml-auto pulse-dot" style={{ background: healthOk === false ? '#fb7185' : '#34d399' }} />
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
        {/* Signal Distribution Flowing Gradient Bar */}
        <SignalDistributionBar
          signals={signals}
          sectors={sectorQ.data?.sectors}
        />

        {/* Model Confidence Leaderboard */}
        <ModelLeaderboard
          modelsDistribution={data?.tune?.models_distribution ?? {}}
          pitPass={data?.tune?.pit_pass}
          pitFail={data?.tune?.pit_fail}
          total={data?.tune?.total}
        />

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
                        className="h-full rounded-full bg-gradient-to-r from-[#34d399] to-[#6ee7b7] bar-fill"
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

      {/* Conviction Spotlight -- dual nebula panels */}
      {strongQ.data && (
        <div className="mb-8 fade-up-delay-3">
          <ConvictionSpotlight
            strongBuy={strongQ.data.strong_buy || []}
            strongSell={strongQ.data.strong_sell || []}
          />
        </div>
      )}

      {/* Signal Heatmap */}
      {sectorQ.data?.sectors && summaryQ.data?.horizons && (
        <SignalHeatmap
          sectors={sectorQ.data.sectors}
          horizons={summaryQ.data.horizons}
        />
      )}
    </>
  );
}
