/**
 * Story 8.3: Continuous Profitability Monitoring Dashboard.
 *
 * Line charts for key metrics over time with target overlays.
 * Cards show current values with green (pass) / red (fail) indicators.
 */
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import type { ProfitabilityMetrics } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';

/* ── Metric Card ──────────────────────────────────────────────── */
function MetricCard({ label, value, target, unit = '', index = 0 }: {
  label: string; value: number | null; target: number; unit?: string; index?: number;
}) {
  const pass = value != null && value >= target;
  const color = value == null ? '#7a8ba4' : pass ? 'var(--accent-emerald)' : 'var(--accent-rose)';
  return (
    <div className="glass-card p-5 hover-lift stat-shine fade-up" style={{
      animationDelay: `${index * 60}ms`,
      background: pass
        ? 'linear-gradient(135deg, rgba(62,232,165,0.04) 0%, transparent 50%)'
        : value != null
          ? 'linear-gradient(135deg, rgba(255,107,138,0.04) 0%, transparent 50%)'
          : undefined,
      borderLeft: `2px solid ${pass ? 'rgba(62,232,165,0.3)' : value != null ? 'rgba(255,107,138,0.3)' : 'rgba(139,92,246,0.15)'}`,
    }}>
      <p className="text-label">{label}</p>
      <p className="text-stat-value mt-1 tabular-nums" style={{ color }}>
        {value != null ? `${(value * 100).toFixed(1)}${unit}` : '--'}
      </p>
      <p className="text-caption mt-0.5">
        Target: {(target * 100).toFixed(1)}{unit}
      </p>
      <div
        className="w-2 h-2 rounded-full mt-1"
        style={{ backgroundColor: color, boxShadow: `0 0 6px ${color}` }}
        aria-label={pass ? 'Passing' : 'Failing'}
      />
    </div>
  );
}

/* ── Chart wrapper for each metric ─────────────────────────── */
function MetricChart({ title, data, targetValue, color = '#b49aff' }: {
  title: string; data: { date: string; value: number }[];
  targetValue: number; color?: string;
}) {
  return (
    <div className="glass-card p-4 hover-lift">
      <h3 className="text-xs font-medium text-[var(--text-secondary)] mb-3">{title}</h3>
      <div aria-label={`${title} chart`} role="img" style={{ width: '100%', height: 200 }}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="var(--violet-6)" />
            <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#7a8ba4' }} />
            <YAxis
              tick={{ fontSize: 9, fill: '#7a8ba4' }}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{
                background: 'rgba(15,15,35,0.95)',
                border: '1px solid var(--violet-15)',
                borderRadius: 12,
                fontSize: 10,
                backdropFilter: 'blur(12px)',
                boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
              }}
              formatter={(v: number | undefined) => typeof v === 'number' ? `${(v * 100).toFixed(2)}%` : '—'}
            />
            <ReferenceLine
              y={targetValue}
              stroke="var(--accent-amber)"
              strokeDasharray="8 4"
              label={{ value: 'Target', position: 'right', fill: 'var(--accent-amber)', fontSize: 9 }}
            />
            <Line
              type="monotone"
              dataKey="value"
              stroke={color}
              strokeWidth={2}
              dot={{ r: 2 }}
              activeDot={{ r: 4 }}
            />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ── Main Page ────────────────────────────────────────────────── */
export default function ProfitabilityPage() {
  const { data, isLoading, error } = useQuery({
    queryKey: ['profitability'],
    queryFn: api.diagProfitability,
    staleTime: 120_000,
  });

  if (isLoading) return <LoadingSpinner text="Loading profitability metrics..." />;
  if (error || !data) return <CosmicErrorCard title="Unable to load profitability metrics" error={error as Error} onRetry={() => window.location.reload()} />;

  const m = data as ProfitabilityMetrics;
  const ts = m.timestamps || [];
  const targets = m.targets || {};

  // Build chart data arrays
  const hitRate7d = ts.map((d, i) => ({ date: d, value: m.hit_rates['7d']?.[i] ?? 0 }));
  const hitRate21d = ts.map((d, i) => ({ date: d, value: m.hit_rates['21d']?.[i] ?? 0 }));
  const signalRates = ts.map((d, i) => ({ date: d, value: m.signal_rates[i] ?? 0 }));
  const sharpe7d = ts.map((d, i) => ({ date: d, value: m.sharpe['7d']?.[i] ?? 0 }));
  const crpsData = ts.map((d, i) => ({ date: d, value: m.crps[i] ?? 0 }));
  const eceData = ts.map((d, i) => ({ date: d, value: m.ece[i] ?? 0 }));

  // Current values (last in array)
  const last = (arr: number[]) => arr.length > 0 ? arr[arr.length - 1] : null;
  const curHit7 = last(m.hit_rates['7d'] || []);
  const curHit21 = last(m.hit_rates['21d'] || []);
  const curSignal = last(m.signal_rates || []);
  const curSharpe = last(m.sharpe['7d'] || []);
  const curCrps = last(m.crps || []);
  const curEce = last(m.ece || []);

  const hasData = ts.length > 0;

  return (
    <>
      <PageHeader title="Profitability Monitor" subtitle="Continuous metric tracking" />

      {/* Summary Cards */}
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6 fade-up">
        <MetricCard label="Hit Rate 7D" value={curHit7} target={targets.hit_rate_7d || 0.55} unit="%" index={0} />
        <MetricCard label="Hit Rate 21D" value={curHit21} target={targets.hit_rate_21d || 0.53} unit="%" index={1} />
        <MetricCard label="Signal Rate" value={curSignal} target={targets.signal_rate || 0.15} unit="%" index={2} />
        <MetricCard label="Sharpe 7D" value={curSharpe} target={targets.sharpe_7d || 0.50} unit="" index={3} />
        <MetricCard label="CRPS" value={curCrps} target={targets.ece_max || 0.03} unit="%" index={4} />
        <MetricCard label="ECE" value={curEce} target={targets.ece_max || 0.03} unit="%" index={5} />
      </div>

      {!hasData ? (
        <div className="glass-card p-8 text-center text-[var(--text-secondary)] text-sm">
          No profitability history yet. Run <code style={{ color: '#b49aff' }}>make calibrate</code> to generate data.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 fade-up-delay-1">
          <MetricChart title="Hit Rate 7D" data={hitRate7d} targetValue={targets.hit_rate_7d || 0.55} color="var(--accent-emerald)" />
          <MetricChart title="Hit Rate 21D" data={hitRate21d} targetValue={targets.hit_rate_21d || 0.53} color="#6ff0c0" />
          <MetricChart title="Signal Rate" data={signalRates} targetValue={targets.signal_rate || 0.15} color="#b49aff" />
          <MetricChart title="Sharpe 7D" data={sharpe7d} targetValue={targets.sharpe_7d || 0.50} color="var(--accent-amber)" />
          <MetricChart title="CRPS" data={crpsData} targetValue={0.02} color="#f87171" />
          <MetricChart title="ECE" data={eceData} targetValue={targets.ece_max || 0.03} color="var(--accent-violet)" />
        </div>
      )}
    </>
  );
}
