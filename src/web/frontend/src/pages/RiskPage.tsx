import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { ShieldAlert, Thermometer, Activity, AlertOctagon } from 'lucide-react';

export default function RiskPage() {
  const { data, isLoading, error, refetch, isFetching } = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 5 * 60_000,
  });

  if (isLoading) return <LoadingSpinner text="Computing risk dashboard... this may take a minute" />;
  if (error) return <div className="text-[#FF1744]">Failed to compute risk dashboard</div>;
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

  return (
    <>
      <PageHeader
        title="Risk Dashboard"
        action={
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="px-4 py-2 rounded-lg bg-[#16213e] text-sm text-[#42A5F5] hover:bg-[#1a2744] border border-[#2a2a4a] transition disabled:opacity-50"
          >
            {isFetching ? 'Computing...' : 'Refresh'}
          </button>
        }
      >
        Unified cross-asset risk assessment
      </PageHeader>

      {/* Combined Temperature Hero */}
      <div className="glass-card p-6 mb-6">
        <div className="flex items-center gap-4 mb-4">
          <Thermometer className="w-8 h-8" style={{ color: statusColor(data.status) }} />
          <div>
            <h2 className="text-3xl font-bold" style={{ color: statusColor(data.status) }}>
              {data.combined_temperature.toFixed(2)}
            </h2>
            <p className="text-sm font-medium" style={{ color: statusColor(data.status) }}>
              {data.status}
            </p>
          </div>
          <div className="flex-1 ml-8">
            <div className="h-3 rounded-full bg-[#1a1a2e] overflow-hidden relative">
              <div
                className="h-full rounded-full transition-all duration-700 ease-out animate-gradient-shift"
                style={{
                  width: `${tempBarWidth}%`,
                  background: `linear-gradient(90deg, #00E676, #FFB300, #FF1744)`,
                  backgroundSize: '200% 100%',
                  animation: 'gradient-shift 3s ease infinite',
                }}
              />
            </div>
            <div className="flex justify-between text-[10px] text-[#64748b] mt-1">
              <span>Calm</span>
              <span>Elevated</span>
              <span>Stressed</span>
              <span>Crisis</span>
            </div>
          </div>
        </div>
        <p className="text-xs text-[#64748b]">
          Computed at {data.computed_at ? new Date(data.computed_at).toLocaleString() : 'N/A'}
        </p>
      </div>

      {/* Per-module cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
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

      {/* Risk interpretation */}
      <div className="glass-card p-5">
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
    </>
  );
}
