import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  Signal,
  TrendingUp,
  TrendingDown,
  Database,
  Settings,
  AlertTriangle,
  CheckCircle,
  Clock,
} from 'lucide-react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip,
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
} from 'recharts';

export default function OverviewPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['overview'],
    queryFn: api.overview,
    refetchInterval: 60_000,
  });

  if (isLoading) return <LoadingSpinner text="Loading dashboard..." />;
  if (error || !data) return <div className="text-[#FF1744]">Failed to load overview</div>;

  const { signals, tuning, data: dataStatus } = data;

  const signalPieData = [
    { name: 'Buy', value: signals.buy_signals, color: '#00E676' },
    { name: 'Sell', value: signals.sell_signals, color: '#FF1744' },
    { name: 'Hold', value: signals.hold_signals, color: '#64748b' },
  ];

  const cacheAgeMin = signals.cache_age_seconds
    ? Math.round(signals.cache_age_seconds / 60)
    : null;

  const modelData = Object.entries(tuning.models_distribution || {})
    .map(([name, count]) => ({
      name: name.replace('phi_student_t_', 'φ-t ').replace('kalman_', 'K-').replace('_unified', '').replace('_momentum', '+M'),
      count,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 8);

  return (
    <>
      <PageHeader title="Dashboard Overview">
        System-wide snapshot — {signals.total_assets} assets monitored
      </PageHeader>

      {/* Top stats row */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <StatCard
          title="Total Assets"
          value={signals.total_assets}
          subtitle={`${signals.failed} failed`}
          icon={<Signal className="w-5 h-5" />}
          color="blue"
        />
        <StatCard
          title="Buy Signals"
          value={signals.buy_signals}
          icon={<TrendingUp className="w-5 h-5" />}
          color="green"
        />
        <StatCard
          title="Sell Signals"
          value={signals.sell_signals}
          icon={<TrendingDown className="w-5 h-5" />}
          color="red"
        />
        <StatCard
          title="Cache Age"
          value={cacheAgeMin !== null ? `${cacheAgeMin}m` : 'N/A'}
          subtitle={signals.cached ? 'Last computation' : 'No cache'}
          icon={<Clock className="w-5 h-5" />}
          color="amber"
        />
      </div>

      {/* Second row: Tuning + Data */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
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
          title="Stale Data"
          value={dataStatus.stale_files}
          subtitle={dataStatus.oldest_hours ? `Oldest: ${Math.round(dataStatus.oldest_hours)}h` : undefined}
          icon={<AlertTriangle className="w-5 h-5" />}
          color={dataStatus.stale_files > 10 ? 'red' : 'amber'}
        />
      </div>

      {/* Charts row */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Signal Distribution Pie */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-4">Signal Distribution</h3>
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie
                data={signalPieData}
                cx="50%"
                cy="50%"
                innerRadius={55}
                outerRadius={85}
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
                }}
              />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex justify-center gap-6 mt-2">
            {signalPieData.map((d) => (
              <div key={d.name} className="flex items-center gap-1.5 text-xs">
                <span className="w-2.5 h-2.5 rounded-full" style={{ background: d.color }} />
                {d.name}: {d.value}
              </div>
            ))}
          </div>
        </div>

        {/* Model Distribution Bar */}
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-4">Model Distribution</h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={modelData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={90} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip
                contentStyle={{
                  background: '#1a1a2e',
                  border: '1px solid #2a2a4a',
                  borderRadius: 8,
                  color: '#e2e8f0',
                }}
              />
              <Bar dataKey="count" fill="#42A5F5" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </>
  );
}
