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
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer,
} from 'recharts';

/* ── Metric Card ──────────────────────────────────────────────── */
function MetricCard({ label, value, target, unit = '' }: {
  label: string; value: number | null; target: number; unit?: string;
}) {
  const pass = value != null && value >= target;
  const color = value == null ? '#64748b' : pass ? '#00E676' : '#FF1744';
  return (
    <div className="glass-card p-4">
      <p className="text-[10px] text-[#64748b] uppercase tracking-wider">{label}</p>
      <p className="text-lg font-bold mt-1" style={{ color }}>
        {value != null ? `${(value * 100).toFixed(1)}${unit}` : '--'}
      </p>
      <p className="text-[9px] text-[#64748b] mt-0.5">
        Target: {(target * 100).toFixed(1)}{unit}
      </p>
      <div
        className="w-2 h-2 rounded-full mt-1"
        style={{ backgroundColor: color }}
        aria-label={pass ? 'Passing' : 'Failing'}
      />
    </div>
  );
}

/* ── Chart wrapper for each metric ─────────────────────────── */
function MetricChart({ title, data, targetValue, color = '#42A5F5' }: {
  title: string; data: { date: string; value: number }[];
  targetValue: number; color?: string;
}) {
  return (
    <div className="glass-card p-4">
      <h3 className="text-xs font-medium text-[#94a3b8] mb-3">{title}</h3>
      <div aria-label={`${title} chart`} role="img" style={{ width: '100%', height: 200 }}>
        <ResponsiveContainer width="100%" height={200}>
          <LineChart data={data} margin={{ top: 5, right: 10, bottom: 5, left: 10 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#1a1a2e" />
            <XAxis dataKey="date" tick={{ fontSize: 9, fill: '#64748b' }} />
            <YAxis
              tick={{ fontSize: 9, fill: '#64748b' }}
              tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
            />
            <Tooltip
              contentStyle={{ background: '#0f0f23', border: '1px solid #2a2a4a', fontSize: 10 }}
              formatter={(v: number) => `${(v * 100).toFixed(2)}%`}
            />
            <ReferenceLine
              y={targetValue}
              stroke="#FF1744"
              strokeDasharray="6 3"
              label={{ value: 'Target', position: 'right', fill: '#FF1744', fontSize: 9 }}
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
  if (error || !data) return <div className="text-[#FF1744]">Failed to load metrics</div>;

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
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3 mb-6">
        <MetricCard label="Hit Rate 7D" value={curHit7} target={targets.hit_rate_7d || 0.55} unit="%" />
        <MetricCard label="Hit Rate 21D" value={curHit21} target={targets.hit_rate_21d || 0.53} unit="%" />
        <MetricCard label="Signal Rate" value={curSignal} target={targets.signal_rate || 0.15} unit="%" />
        <MetricCard label="Sharpe 7D" value={curSharpe} target={targets.sharpe_7d || 0.50} unit="" />
        <MetricCard label="CRPS" value={curCrps} target={targets.ece_max || 0.03} unit="%" />
        <MetricCard label="ECE" value={curEce} target={targets.ece_max || 0.03} unit="%" />
      </div>

      {!hasData ? (
        <div className="glass-card p-8 text-center text-[#64748b] text-sm">
          No profitability history yet. Run <code className="text-[#42A5F5]">make calibrate</code> to generate data.
        </div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
          <MetricChart title="Hit Rate 7D" data={hitRate7d} targetValue={targets.hit_rate_7d || 0.55} color="#00E676" />
          <MetricChart title="Hit Rate 21D" data={hitRate21d} targetValue={targets.hit_rate_21d || 0.53} color="#66BB6A" />
          <MetricChart title="Signal Rate" data={signalRates} targetValue={targets.signal_rate || 0.15} color="#42A5F5" />
          <MetricChart title="Sharpe 7D" data={sharpe7d} targetValue={targets.sharpe_7d || 0.50} color="#FFB300" />
          <MetricChart title="CRPS" data={crpsData} targetValue={0.02} color="#EF5350" />
          <MetricChart title="ECE" data={eceData} targetValue={targets.ece_max || 0.03} color="#AB47BC" />
        </div>
      )}
    </>
  );
}
