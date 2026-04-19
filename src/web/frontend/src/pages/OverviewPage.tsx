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
import { useScrollReveal } from '../hooks/useScrollReveal';
import {
  Signal, TrendingUp, TrendingDown, Database, Settings,
  AlertTriangle, CheckCircle, Clock, HeartPulse,
} from 'lucide-react';


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

  const riskQ = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 120_000,
  });

  const scrollRef = useScrollReveal();

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
    <div ref={scrollRef}>
      <PageHeader title="Dashboard">
        System-wide snapshot {'\u2014'} {signals.total_assets} assets monitored
      </PageHeader>

      {/* Errors banner */}
      {overviewErrors.length > 0 && (
        <div className="glass-card p-4 mb-5 fade-up" style={{ borderLeft: '2px solid var(--accent-amber)' }}>
          <div className="flex items-center gap-2 text-xs" style={{ color: 'var(--accent-amber)' }}>
            <AlertTriangle className="w-4 h-4" />
            <span className="font-medium">Partial data returned</span>
          </div>
          <ul className="mt-1.5 space-y-0.5">
            {overviewErrors.map((e, i) => (
              <li key={i} className="text-[10px] pl-6" style={{ color: 'var(--text-muted)' }}>{e}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Morning Briefing Hero Card */}
      <div className="fade-up-hero">
        <BriefingCard
        signals={signals}
        tuning={tuning}
        dataStatus={dataStatus}
        risk={riskQ.data}
        strongBuy={strongQ.data?.strong_buy || []}
        strongSell={strongQ.data?.strong_sell || []}
      />
      </div>

      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-5 gap-5 mb-10 fade-up-delay-2">
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
        <div className="glass-card hover-lift" style={{ padding: '20px 24px' }}>
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                 style={{ background: `${healthOk === false ? 'var(--accent-rose)' : 'var(--accent-emerald)'}10` }}>
              <HeartPulse className="w-5 h-5" style={{ color: healthOk === undefined ? 'var(--text-muted)' : healthOk ? 'var(--accent-emerald)' : 'var(--accent-rose)' }} />
            </div>
            <div>
              <p className="text-lg font-bold tabular-nums" style={{ color: 'var(--text-luminous)' }}>{healthOk === undefined ? '...' : healthOk ? 'OK' : 'Issue'}</p>
              <p className="text-[10px] tracking-wide" style={{ color: 'var(--text-muted)' }}>System Health</p>
            </div>
            <span className="w-2 h-2 rounded-full ml-auto pulse-dot" style={{ background: healthOk === false ? 'var(--accent-rose)' : 'var(--accent-emerald)' }} />
          </div>
        </div>
      </div>

      {/* Second row: Tuning + Data */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-10 fade-up-delay-4">
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

      {/* Charts row -- below fold, scroll-triggered */}
      <div className="scroll-reveal grid grid-cols-1 md:grid-cols-3 gap-5 mb-10">
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
        <div className="glass-card hover-lift" style={{ padding: '24px' }}>
          <h3 className="text-[11px] font-semibold uppercase mb-5" style={{ color: 'var(--text-muted)', letterSpacing: '0.12em' }}>Top Sectors</h3>
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
                      <span className="font-medium truncate max-w-[140px]" style={{ color: 'var(--text-luminous)' }}>{sec.name}</span>
                      <span className="tabular-nums" style={{ color: 'var(--text-muted)' }}>{bullish}/{total}</span>
                    </div>
                    <div className="h-1.5 rounded-full bg-white/[0.03] overflow-hidden">
                      <div
                        className="h-full rounded-full bg-gradient-to-r from-[var(--accent-emerald)] to-[var(--accent-emerald)] bar-fill"
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
        <div className="scroll-reveal mb-10">
          <ConvictionSpotlight
            strongBuy={strongQ.data.strong_buy || []}
            strongSell={strongQ.data.strong_sell || []}
          />
        </div>
      )}


    </div>
  );
}
