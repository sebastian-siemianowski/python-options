import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api';
import type {
  RiskDashboardFull, RiskStressCategory, RiskStressIndicator,
  SectorMetrics, CurrencyMetrics,
  MarketBreadth, CorrelationStress,
} from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  ShieldAlert, Thermometer, Activity, AlertOctagon, RefreshCw,
  ChevronDown, ChevronRight,
  Gem, Globe, BarChart3, Link2,
} from 'lucide-react';

type RiskTab = 'overview' | 'cross_asset' | 'metals' | 'market' | 'sectors' | 'currencies';

export default function RiskPage() {
  const [tab, setTab] = useState<RiskTab>('overview');
  const [refreshing, setRefreshing] = useState(false);

  const summaryQ = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 5 * 60_000,
  });

  const dashQ = useQuery({
    queryKey: ['riskDashboard'],
    queryFn: () => api.riskDashboard() as Promise<RiskDashboardFull>,
    staleTime: 5 * 60_000,
  });

  const data = summaryQ.data;
  const dash = dashQ.data;

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.riskRefresh();
      summaryQ.refetch();
      dashQ.refetch();
    } finally {
      setRefreshing(false);
    }
  };

  if (summaryQ.isLoading && !data) return <LoadingSpinner text="Computing risk dashboard... this may take a minute" />;
  if (summaryQ.error) return <div className="text-[#FF1744]">Failed to compute risk dashboard</div>;
  if (!data) return null;

  const statusColor = (status: string) => {
    switch (status) {
      case 'Calm': return '#00E676';
      case 'Elevated': return '#FFB300';
      case 'Stressed': return '#FF1744';
      case 'Crisis': return '#FF1744';
      default: return '#64748b';
    }
  };

  const tempBarWidth = Math.min(data.combined_temperature / 2 * 100, 100);

  const tabs: { id: RiskTab; label: string; icon: typeof ShieldAlert }[] = [
    { id: 'overview', label: 'Overview', icon: Thermometer },
    { id: 'cross_asset', label: 'Cross-Asset Stress', icon: ShieldAlert },
    { id: 'metals', label: 'Metals', icon: Gem },
    { id: 'market', label: 'Market', icon: Globe },
    { id: 'sectors', label: 'Sectors', icon: BarChart3 },
    { id: 'currencies', label: 'Currencies', icon: Link2 },
  ];

  return (
    <>
      <PageHeader
        title="Risk Dashboard"
        action={
          <div className="flex items-center gap-2">
            {dash?._cached && (
              <span className="text-[10px] text-[#64748b]">
                cached {dash._cache_age_seconds ? `${Math.round(dash._cache_age_seconds / 60)}m ago` : ''}
              </span>
            )}
            <button
              onClick={handleRefresh}
              disabled={refreshing || summaryQ.isFetching}
              className="flex items-center gap-2 px-4 py-2 rounded-xl bg-white/[0.04] text-[13px] text-[#42A5F5] hover:bg-white/[0.06] border border-white/[0.06] transition-all duration-200 disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
              {refreshing ? 'Refreshing...' : 'Refresh Data'}
            </button>
          </div>
        }
      >
        Unified cross-asset risk assessment
      </PageHeader>

      {/* Combined Temperature Hero */}
      <div className="glass-card p-8 mb-8 fade-up ambient-glow">
        <div className="relative flex items-center gap-5 mb-5">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
               style={{ background: `${statusColor(data.status)}10` }}>
            <Thermometer className="w-7 h-7" style={{ color: statusColor(data.status) }} />
          </div>
          <div>
            <h2 className="text-4xl font-bold tabular-nums tracking-tight" style={{ color: statusColor(data.status) }}>
              {data.combined_temperature.toFixed(2)}
            </h2>
            <p className="text-[13px] font-medium mt-0.5" style={{ color: statusColor(data.status) }}>
              {data.status}
            </p>
          </div>
          <div className="flex-1 ml-8">
            <div className="h-3 rounded-full bg-[#1a1a2e] overflow-hidden relative">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out"
                style={{
                  width: `${tempBarWidth}%`,
                  background: `linear-gradient(90deg, #00E676, #FFB300, #FF1744)`,
                  backgroundSize: '200% 100%',
                }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-[#64748b] mt-1.5 font-medium tracking-wide">
              <span>Calm</span><span>Elevated</span><span>Stressed</span><span>Crisis</span>
            </div>
          </div>
        </div>
        <p className="text-[11px] text-[#475569]">
          Computed at {data.computed_at ? new Date(data.computed_at).toLocaleString() : 'N/A'}
        </p>
      </div>

      {/* Per-module cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8 fade-up-delay-1">
        <StatCard
          title="Cross-Asset Stress"
          value={data.risk_temperature.toFixed(3)}
          subtitle="FX • Equities • Duration • Commodities"
          icon={<ShieldAlert className="w-5 h-5" />}
          color={data.risk_temperature < 0.3 ? 'green' : data.risk_temperature < 0.7 ? 'amber' : 'red'}
        />
        <StatCard
          title="Metals Risk"
          value={data.metals_temperature.toFixed(3)}
          subtitle="Gold • Silver • Copper • Palladium"
          icon={<Activity className="w-5 h-5" />}
          color={data.metals_temperature < 0.3 ? 'green' : data.metals_temperature < 0.7 ? 'amber' : 'red'}
        />
        <StatCard
          title="Market Temperature"
          value={data.market_temperature.toFixed(3)}
          subtitle="Equity universe • Sectors • Currencies"
          icon={<AlertOctagon className="w-5 h-5" />}
          color={data.market_temperature < 0.3 ? 'green' : data.market_temperature < 0.7 ? 'amber' : 'red'}
        />
      </div>

      {/* Tab nav */}
      <div className="flex gap-0.5 mb-8 overflow-x-auto fade-up-delay-2">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-[13px] font-medium transition-all duration-200 rounded-xl whitespace-nowrap ${
              tab === id
                ? 'bg-white/[0.06] text-[#42A5F5]'
                : 'text-[#64748b] hover:text-[#94a3b8] hover:bg-white/[0.02]'
            }`}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'overview' && <OverviewTab data={data} />}
      {tab === 'cross_asset' && dash && <CrossAssetTab categories={dash.risk_temperature?.categories} />}
      {tab === 'metals' && dash && <MetalsTab metals={dash.metals_risk_temperature} />}
      {tab === 'market' && dash && <MarketTab market={dash.market_temperature} />}
      {tab === 'sectors' && dash && <SectorsTab sectors={dash.market_temperature?.sectors} />}
      {tab === 'currencies' && dash && <CurrenciesTab currencies={dash.market_temperature?.currencies} />}
      {dashQ.isLoading && tab !== 'overview' && <LoadingSpinner text="Loading full dashboard data..." />}
    </>
  );
}

/* ── Overview Tab ─────────────────────────────────────────────────── */

function OverviewTab({ data: _data }: { data: { combined_temperature: number; status: string } }) {
  return (
    <div className="glass-card p-5 hover-lift">
      <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Risk Interpretation</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
        {[
          { range: '< 0.3', label: 'Calm', desc: 'Full exposure permitted', color: '#00E676' },
          { range: '0.3 – 0.7', label: 'Elevated', desc: 'Monitor positions closely', color: '#FFB300' },
          { range: '0.7 – 1.2', label: 'Stressed', desc: 'Reduce risk exposure', color: '#FF1744' },
          { range: '> 1.2', label: 'Crisis', desc: 'Capital preservation mode', color: '#FF1744' },
        ].map((tier) => (
          <div key={tier.label} className="flex items-start gap-2">
            <span className="w-2 h-2 rounded-full mt-1 flex-shrink-0" style={{ background: tier.color }} />
            <div>
              <p className="font-medium text-[#e2e8f0]">{tier.label} ({tier.range})</p>
              <p className="text-[#64748b]">{tier.desc}</p>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Cross-Asset Stress Tab ───────────────────────────────────────── */

function CrossAssetTab({ categories }: { categories?: Record<string, RiskStressCategory> }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!categories) return <div className="text-[#64748b] text-sm">No cross-asset data available</div>;

  const toggle = (name: string) =>
    setExpanded(prev => { const n = new Set(prev); n.has(name) ? n.delete(name) : n.add(name); return n; });

  const sorted = Object.values(categories).sort((a, b) => b.weighted_contribution - a.weighted_contribution);

  return (
    <div className="space-y-3">
      {sorted.map(cat => (
        <div key={cat.name} className="glass-card overflow-hidden hover-lift">
          <button onClick={() => toggle(cat.name)} className="w-full px-4 py-3 flex items-center justify-between hover:bg-[#16213e]/50 transition">
            <div className="flex items-center gap-3">
              <StressBar level={cat.stress_level} />
              <span className="text-sm font-medium text-[#e2e8f0]">{cat.name}</span>
            </div>
            <div className="flex items-center gap-4 text-xs">
              <span className="text-[#94a3b8]">Weight: {(cat.weight * 100).toFixed(0)}%</span>
              <span className={stressColor(cat.stress_level)}>Stress: {cat.stress_level.toFixed(3)}</span>
              <span className="text-[#64748b]">Contrib: {cat.weighted_contribution.toFixed(4)}</span>
              {expanded.has(cat.name) ? <ChevronDown className="w-4 h-4 text-[#64748b]" /> : <ChevronRight className="w-4 h-4 text-[#64748b]" />}
            </div>
          </button>
          {expanded.has(cat.name) && cat.indicators && (
            <div className="px-4 pb-3">
              <IndicatorsTable indicators={cat.indicators} />
            </div>
          )}
        </div>
      ))}
    </div>
  );
}

/* ── Metals Tab ───────────────────────────────────────────────────── */

function MetalsTab({ metals: data }: { metals: RiskDashboardFull['metals_risk_temperature'] }) {
  if (!data?.metals) return <div className="text-[#64748b] text-sm">No metals data available</div>;

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="glass-card p-4 flex items-center justify-between hover-lift">
        <div className="flex items-center gap-3">
          <Gem className="w-5 h-5" style={{ color: data.status === 'Calm' ? '#00E676' : data.status === 'Elevated' ? '#FFB300' : '#FF1744' }} />
          <div>
            <span className="text-sm font-medium text-[#e2e8f0]">{data.status}</span>
            <span className="text-xs text-[#64748b] ml-2">{data.action_text}</span>
          </div>
        </div>
        <div className="text-xs text-[#64748b]">
          Regime: {data.regime_state} • Crash Risk: {data.crash_risk_level} ({(data.crash_risk_pct * 100).toFixed(1)}%)
        </div>
      </div>

      {/* Stress indicators */}
      {data.indicators && data.indicators.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Stress Indicators</h3>
          <IndicatorsTable indicators={data.indicators} />
        </div>
      )}

      {/* Individual metals */}
      <div className="glass-card overflow-hidden">
        <div className="px-4 py-2.5 border-b border-[#2a2a4a]">
          <h3 className="text-sm font-medium text-[#94a3b8]">Individual Metals</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-[#1a1a2e]">
              <tr className="border-b border-[#2a2a4a]">
                <th className="text-left px-3 py-2 text-[#64748b]">Metal</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Price</th>
                <th className="text-right px-3 py-2 text-[#64748b]">1D</th>
                <th className="text-right px-3 py-2 text-[#64748b]">5D</th>
                <th className="text-right px-3 py-2 text-[#64748b]">21D</th>
                <th className="text-center px-3 py-2 text-[#64748b]">Momentum</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Stress</th>
                <th className="text-right px-3 py-2 text-[#64748b]">7D</th>
                <th className="text-right px-3 py-2 text-[#64748b]">30D</th>
                <th className="text-right px-3 py-2 text-[#64748b]">90D</th>
                <th className="text-right px-3 py-2 text-[#64748b]">180D</th>
                <th className="text-right px-3 py-2 text-[#64748b]">365D</th>
                <th className="text-center px-3 py-2 text-[#64748b]">Conf</th>
              </tr>
            </thead>
            <tbody>
              {Object.entries(data.metals).map(([name, m]) => (
                <tr key={name} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30">
                  <td className="px-3 py-2 font-medium text-[#e2e8f0]">{name}</td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{m.price != null ? m.price.toFixed(2) : '—'}</td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={m.return_1d} /></td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={m.return_5d} /></td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={m.return_21d} /></td>
                  <td className="px-3 py-2 text-center"><MomentumBadge signal={m.momentum_signal} /></td>
                  <td className="px-3 py-2 text-right"><span className={stressColor(m.stress_level)}>{m.stress_level.toFixed(2)}</span></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={m.forecast_7d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={m.forecast_30d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={m.forecast_90d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={m.forecast_180d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={m.forecast_365d} /></td>
                  <td className="px-3 py-2 text-center text-[#64748b]">{m.forecast_confidence}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ── Market Tab ───────────────────────────────────────────────────── */

function MarketTab({ market }: { market: RiskDashboardFull['market_temperature'] }) {
  if (!market) return <div className="text-[#64748b] text-sm">No market data available</div>;

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="glass-card p-4 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Globe className="w-5 h-5" style={{ color: market.status === 'Calm' ? '#00E676' : market.status === 'Elevated' ? '#FFB300' : '#FF1744' }} />
          <div>
            <span className="text-sm font-medium text-[#e2e8f0]">{market.status}</span>
            <span className="text-xs text-[#64748b] ml-2">{market.action_text}</span>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs text-[#64748b]">
          <span>Momentum: {market.overall_momentum}</span>
          <span>Crash: {market.crash_risk_level} ({((market.crash_risk_pct ?? 0) * 100).toFixed(1)}%)</span>
          {market.exit_signal && <span className="text-[#FF1744] font-bold">EXIT SIGNAL: {market.exit_reason}</span>}
        </div>
      </div>

      {/* Market breadth + Correlation */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {market.breadth && <BreadthCard breadth={market.breadth} />}
        {market.correlation && <CorrelationCard corr={market.correlation} />}
      </div>

      {/* Universes */}
      {market.universes && (
        <div className="glass-card overflow-hidden">
          <div className="px-4 py-2.5 border-b border-[#2a2a4a]">
            <h3 className="text-sm font-medium text-[#94a3b8]">Market Universes</h3>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead className="bg-[#1a1a2e]">
                <tr className="border-b border-[#2a2a4a]">
                  <th className="text-left px-3 py-2 text-[#64748b]">Universe</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Level</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">1D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">5D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">21D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Vol 20D</th>
                  <th className="text-center px-3 py-2 text-[#64748b]">Momentum</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Stress</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">7D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">30D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">90D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">180D</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">365D</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(market.universes).map(([name, u]) => (
                  <tr key={name} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30">
                    <td className="px-3 py-2 font-medium text-[#e2e8f0]">{name}</td>
                    <td className="px-3 py-2 text-right text-[#94a3b8]">{u.current_level?.toFixed(1) ?? '—'}</td>
                    <td className="px-3 py-2 text-right"><ReturnCell v={u.return_1d} /></td>
                    <td className="px-3 py-2 text-right"><ReturnCell v={u.return_5d} /></td>
                    <td className="px-3 py-2 text-right"><ReturnCell v={u.return_21d} /></td>
                    <td className="px-3 py-2 text-right text-[#94a3b8]">{(u.volatility_20d * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 text-center"><MomentumBadge signal={u.momentum_signal} /></td>
                    <td className="px-3 py-2 text-right"><span className={stressColor(u.stress_level)}>{u.stress_level.toFixed(2)}</span></td>
                    <td className="px-3 py-2 text-right"><ForecastCell v={u.forecast_7d} /></td>
                    <td className="px-3 py-2 text-right"><ForecastCell v={u.forecast_30d} /></td>
                    <td className="px-3 py-2 text-right"><ForecastCell v={u.forecast_90d} /></td>
                    <td className="px-3 py-2 text-right"><ForecastCell v={u.forecast_180d} /></td>
                    <td className="px-3 py-2 text-right"><ForecastCell v={u.forecast_365d} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Sectors Tab ──────────────────────────────────────────────────── */

function SectorsTab({ sectors }: { sectors?: Record<string, SectorMetrics> }) {
  if (!sectors) return <div className="text-[#64748b] text-sm">No sector data available</div>;

  return (
    <div className="glass-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-[#1a1a2e]">
            <tr className="border-b border-[#2a2a4a]">
              <th className="text-left px-3 py-2 text-[#64748b]">Sector</th>
              <th className="text-center px-3 py-2 text-[#64748b]">ETF</th>
              <th className="text-right px-3 py-2 text-[#64748b]">1D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">5D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">21D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">Vol 20D</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Momentum</th>
              <th className="text-right px-3 py-2 text-[#64748b]">Risk</th>
              <th className="text-right px-3 py-2 text-[#64748b]">7D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">30D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">90D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">180D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">365D</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Conf</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(sectors)
              .sort(([, a], [, b]) => b.risk_score - a.risk_score)
              .map(([name, s]) => (
                <tr key={name} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30">
                  <td className="px-3 py-2 font-medium text-[#e2e8f0]">{name}</td>
                  <td className="px-3 py-2 text-center text-[#42A5F5]">{s.ticker}</td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={s.return_1d} /></td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={s.return_5d} /></td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={s.return_21d} /></td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{(s.volatility_20d * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2 text-center"><MomentumBadge signal={s.momentum_signal} /></td>
                  <td className="px-3 py-2 text-right"><RiskScoreCell score={s.risk_score} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={s.forecast_7d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={s.forecast_30d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={s.forecast_90d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={s.forecast_180d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={s.forecast_365d} /></td>
                  <td className="px-3 py-2 text-center text-[#64748b]">{s.forecast_confidence}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Currencies Tab ───────────────────────────────────────────────── */

function CurrenciesTab({ currencies }: { currencies?: Record<string, CurrencyMetrics> }) {
  if (!currencies) return <div className="text-[#64748b] text-sm">No currency data available</div>;

  return (
    <div className="glass-card overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full text-xs">
          <thead className="bg-[#1a1a2e]">
            <tr className="border-b border-[#2a2a4a]">
              <th className="text-left px-3 py-2 text-[#64748b]">Currency</th>
              <th className="text-right px-3 py-2 text-[#64748b]">Rate</th>
              <th className="text-right px-3 py-2 text-[#64748b]">1D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">5D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">21D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">Vol</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Momentum</th>
              <th className="text-right px-3 py-2 text-[#64748b]">Risk</th>
              <th className="text-right px-3 py-2 text-[#64748b]">7D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">30D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">90D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">180D</th>
              <th className="text-right px-3 py-2 text-[#64748b]">365D</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Conf</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(currencies)
              .sort(([, a], [, b]) => b.risk_score - a.risk_score)
              .map(([name, c]) => (
                <tr key={name} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30">
                  <td className="px-3 py-2 font-medium text-[#e2e8f0]">
                    {name}
                    {c.is_inverse && <span className="ml-1 text-[10px] text-[#FFB300]">inv</span>}
                  </td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{c.rate?.toFixed(4) ?? '—'}</td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={c.return_1d} /></td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={c.return_5d} /></td>
                  <td className="px-3 py-2 text-right"><ReturnCell v={c.return_21d} /></td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{(c.volatility_20d * 100).toFixed(2)}%</td>
                  <td className="px-3 py-2 text-center"><MomentumBadge signal={c.momentum_signal} /></td>
                  <td className="px-3 py-2 text-right"><RiskScoreCell score={c.risk_score} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={c.forecast_7d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={c.forecast_30d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={c.forecast_90d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={c.forecast_180d} /></td>
                  <td className="px-3 py-2 text-right"><ForecastCell v={c.forecast_365d} /></td>
                  <td className="px-3 py-2 text-center text-[#64748b]">{c.forecast_confidence}</td>
                </tr>
              ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Breadth Card ─────────────────────────────────────────────────── */

function BreadthCard({ breadth }: { breadth: MarketBreadth }) {
  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Market Breadth</h3>
      <div className="space-y-2 text-xs">
        <BreadthRow label="Above 50 MA" value={`${(breadth.pct_above_50ma * 100).toFixed(1)}%`} />
        <BreadthRow label="Above 200 MA" value={`${(breadth.pct_above_200ma * 100).toFixed(1)}%`} />
        <BreadthRow label="New Highs" value={String(breadth.new_highs)} />
        <BreadthRow label="New Lows" value={String(breadth.new_lows)} />
        <BreadthRow label="A/D Ratio" value={breadth.advance_decline_ratio.toFixed(2)} />
        {breadth.breadth_thrust && <div className="text-[#00E676] font-medium">Breadth Thrust Active</div>}
        {breadth.breadth_warning && <div className="text-[#FF1744] font-medium">Breadth Warning</div>}
        <p className="text-[#64748b] mt-2">{breadth.interpretation}</p>
      </div>
    </div>
  );
}

function BreadthRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span className="text-[#64748b]">{label}</span>
      <span className="text-[#e2e8f0] font-medium">{value}</span>
    </div>
  );
}

/* ── Correlation Card ─────────────────────────────────────────────── */

function CorrelationCard({ corr }: { corr: CorrelationStress }) {
  return (
    <div className="glass-card p-4">
      <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Correlation Stress</h3>
      <div className="space-y-2 text-xs">
        <BreadthRow label="Avg Correlation" value={corr.avg_correlation.toFixed(3)} />
        <BreadthRow label="Max Correlation" value={corr.max_correlation.toFixed(3)} />
        <BreadthRow label="Percentile" value={`${(corr.correlation_percentile * 100).toFixed(0)}%`} />
        {corr.systemic_risk_elevated && (
          <div className="text-[#FF1744] font-medium">Systemic Risk Elevated</div>
        )}
        <p className="text-[#64748b] mt-2">{corr.interpretation}</p>
      </div>
    </div>
  );
}

/* ── Indicators Table ─────────────────────────────────────────────── */

function IndicatorsTable({ indicators }: { indicators: RiskStressIndicator[] }) {
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b border-[#2a2a4a]">
          <th className="text-left px-2 py-1.5 text-[#64748b]">Indicator</th>
          <th className="text-right px-2 py-1.5 text-[#64748b]">Value</th>
          <th className="text-right px-2 py-1.5 text-[#64748b]">Z-Score</th>
          <th className="text-right px-2 py-1.5 text-[#64748b]">Contribution</th>
          <th className="text-center px-2 py-1.5 text-[#64748b]">Data</th>
        </tr>
      </thead>
      <tbody>
        {indicators.map((ind, i) => (
          <tr key={i} className="border-b border-[#2a2a4a]/30">
            <td className="px-2 py-1.5 text-[#94a3b8]">{ind.name}</td>
            <td className="px-2 py-1.5 text-right text-[#e2e8f0]">{ind.value != null ? ind.value.toFixed(4) : '—'}</td>
            <td className="px-2 py-1.5 text-right">
              <span className={ind.zscore != null && Math.abs(ind.zscore) > 2 ? 'text-[#FF1744]' : ind.zscore != null && Math.abs(ind.zscore) > 1 ? 'text-[#FFB300]' : 'text-[#94a3b8]'}>
                {ind.zscore != null ? ind.zscore.toFixed(2) : '—'}
              </span>
            </td>
            <td className="px-2 py-1.5 text-right text-[#94a3b8]">{ind.contribution.toFixed(4)}</td>
            <td className="px-2 py-1.5 text-center">
              {ind.data_available
                ? <span className="text-[#00E676]">✓</span>
                : <span className="text-[#FF1744]">✗</span>}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

/* ── Helper components ────────────────────────────────────────────── */

function StressBar({ level }: { level: number }) {
  const w = Math.min(level / 2 * 100, 100);
  const color = level < 0.3 ? '#00E676' : level < 0.7 ? '#FFB300' : '#FF1744';
  return (
    <div className="w-12 h-2 rounded-full bg-[#1a1a2e] overflow-hidden">
      <div className="h-full rounded-full" style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

function stressColor(level: number): string {
  return level < 0.3 ? 'text-[#00E676]' : level < 0.7 ? 'text-[#FFB300]' : 'text-[#FF1744]';
}

function ReturnCell({ v }: { v: number | null | undefined }) {
  if (v == null) return <span className="text-[#64748b]">—</span>;
  const pct = v * 100;
  const color = pct > 0 ? 'text-[#00E676]' : pct < 0 ? 'text-[#FF1744]' : 'text-[#94a3b8]';
  return <span className={color}>{pct > 0 ? '+' : ''}{pct.toFixed(2)}%</span>;
}

function ForecastCell({ v }: { v: number | null | undefined }) {
  if (v == null) return <span className="text-[#64748b]">—</span>;
  const pct = v * 100;
  const color = pct > 0 ? 'text-[#00E676]' : pct < 0 ? 'text-[#FF1744]' : 'text-[#94a3b8]';
  return <span className={color}>{pct > 0 ? '+' : ''}{pct.toFixed(1)}%</span>;
}

function MomentumBadge({ signal }: { signal: string }) {
  const color = signal?.includes('Strong') || signal?.includes('↑') ? 'text-[#00E676]'
    : signal?.includes('Rising') || signal?.includes('↗') ? 'text-[#66BB6A]'
    : signal?.includes('Falling') || signal?.includes('↘') ? 'text-[#FF7043]'
    : signal?.includes('Weak') || signal?.includes('↓') ? 'text-[#FF1744]'
    : 'text-[#94a3b8]';
  return <span className={`text-[10px] font-medium ${color}`}>{signal || '—'}</span>;
}

function RiskScoreCell({ score }: { score: number }) {
  const color = score < 30 ? 'text-[#00E676]' : score < 60 ? 'text-[#FFB300]' : 'text-[#FF1744]';
  return <span className={`font-medium ${color}`}>{score}</span>;
}
