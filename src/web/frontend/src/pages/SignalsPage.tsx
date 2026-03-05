import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api';
import type { SummaryRow, HighConvictionSignal } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { ArrowUpCircle, ArrowDownCircle, Filter } from 'lucide-react';

export default function SignalsPage() {
  const [filter, setFilter] = useState<'all' | 'buy' | 'sell'>('all');
  const [search, setSearch] = useState('');

  const { data, isLoading } = useQuery({
    queryKey: ['signalSummary'],
    queryFn: api.signalSummary,
  });

  const buyQ = useQuery({
    queryKey: ['highConvictionBuy'],
    queryFn: () => api.highConviction('buy'),
  });

  const sellQ = useQuery({
    queryKey: ['highConvictionSell'],
    queryFn: () => api.highConviction('sell'),
  });

  if (isLoading) return <LoadingSpinner text="Loading signals..." />;

  const rows = data?.summary_rows || [];
  const horizons = data?.horizons || [];

  const filteredRows = rows.filter((row) => {
    if (search && !row.asset_label.toLowerCase().includes(search.toLowerCase())) return false;
    if (filter === 'all') return true;
    return Object.values(row.horizon_signals).some(
      (s) => s.label?.toUpperCase() === filter.toUpperCase()
    );
  });

  return (
    <>
      <PageHeader title="Signals">
        {rows.length} assets across {horizons.length} horizons
      </PageHeader>

      {/* High Conviction Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <HighConvictionCard
          title="High Conviction BUY"
          signals={buyQ.data?.signals || []}
          color="green"
        />
        <HighConvictionCard
          title="High Conviction SELL"
          signals={sellQ.data?.signals || []}
          color="red"
        />
      </div>

      {/* Filters */}
      <div className="flex items-center gap-3 mb-4">
        <div className="flex items-center gap-1 glass-card px-3 py-1.5">
          <Filter className="w-3.5 h-3.5 text-[#64748b]" />
          {(['all', 'buy', 'sell'] as const).map((f) => (
            <button
              key={f}
              onClick={() => setFilter(f)}
              className={`px-2.5 py-1 rounded text-xs font-medium transition ${
                filter === f
                  ? f === 'buy' ? 'bg-[#00E676]/20 text-[#00E676]'
                    : f === 'sell' ? 'bg-[#FF1744]/20 text-[#FF1744]'
                    : 'bg-[#42A5F5]/20 text-[#42A5F5]'
                  : 'text-[#64748b] hover:text-[#94a3b8]'
              }`}
            >
              {f.toUpperCase()}
            </button>
          ))}
        </div>
        <input
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Search assets..."
          className="px-3 py-1.5 rounded-lg bg-[#1a1a2e] border border-[#2a2a4a] text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none focus:border-[#42A5F5] w-48"
        />
        <span className="text-xs text-[#64748b]">{filteredRows.length} results</span>
      </div>

      {/* Signal Table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#2a2a4a]">
                <th className="text-left px-4 py-3 text-xs text-[#64748b] font-medium uppercase">Asset</th>
                <th className="text-left px-3 py-3 text-xs text-[#64748b] font-medium">Sector</th>
                {horizons.slice(0, 5).map((h) => (
                  <th key={h} className="text-center px-3 py-3 text-xs text-[#64748b] font-medium">{h}D</th>
                ))}
                <th className="text-center px-3 py-3 text-xs text-[#64748b] font-medium">Crash Risk</th>
              </tr>
            </thead>
            <tbody>
              {filteredRows.slice(0, 100).map((row) => (
                <SignalRow key={row.asset_label} row={row} horizons={horizons} />
              ))}
            </tbody>
          </table>
        </div>
        {filteredRows.length > 100 && (
          <div className="px-4 py-2 text-xs text-[#64748b] border-t border-[#2a2a4a]">
            Showing 100 of {filteredRows.length} results
          </div>
        )}
      </div>
    </>
  );
}

function SignalRow({ row, horizons }: { row: SummaryRow; horizons: number[] }) {
  return (
    <tr className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30 transition">
      <td className="px-4 py-2.5 font-medium text-[#e2e8f0] whitespace-nowrap">{row.asset_label}</td>
      <td className="px-3 py-2.5 text-xs text-[#94a3b8]">{row.sector}</td>
      {horizons.slice(0, 5).map((h) => {
        const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
        if (!sig) return <td key={h} className="px-3 py-2.5 text-center text-[#64748b]">—</td>;
        const label = (sig.label || 'HOLD').toUpperCase();
        const cls = label === 'BUY' ? 'text-[#00E676]' : label === 'SELL' ? 'text-[#FF1744]' : 'text-[#64748b]';
        return (
          <td key={h} className={`px-3 py-2.5 text-center text-xs font-medium ${cls}`}>
            {label}
            <span className="block text-[10px] text-[#64748b]">
              {(sig.p_up * 100).toFixed(0)}% | {(sig.exp_ret * 100).toFixed(1)}%
            </span>
          </td>
        );
      })}
      <td className="px-3 py-2.5 text-center">
        <CrashRiskBadge score={row.crash_risk_score} />
      </td>
    </tr>
  );
}

function CrashRiskBadge({ score }: { score: number }) {
  const s = score ?? 0;
  const color = s < 0.3 ? 'bg-[#00E676]/20 text-[#00E676]' : s < 0.6 ? 'bg-[#FFB300]/20 text-[#FFB300]' : 'bg-[#FF1744]/20 text-[#FF1744]';
  return (
    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-medium ${color}`}>
      {(s * 100).toFixed(0)}%
    </span>
  );
}

function HighConvictionCard({
  title,
  signals,
  color,
}: {
  title: string;
  signals: HighConvictionSignal[];
  color: 'green' | 'red';
}) {
  const Icon = color === 'green' ? ArrowUpCircle : ArrowDownCircle;
  const accent = color === 'green' ? '#00E676' : '#FF1744';
  const top5 = signals.slice(0, 5);

  return (
    <div className={`glass-card p-4 ${color === 'green' ? 'glow-green' : 'glow-red'}`}>
      <div className="flex items-center gap-2 mb-3">
        <Icon className="w-4 h-4" style={{ color: accent }} />
        <h3 className="text-sm font-medium" style={{ color: accent }}>{title}</h3>
        <span className="ml-auto text-xs text-[#64748b]">{signals.length} signals</span>
      </div>
      {top5.length === 0 ? (
        <p className="text-xs text-[#64748b]">No signals</p>
      ) : (
        <div className="space-y-1.5">
          {top5.map((s, i) => (
            <div key={i} className="flex items-center justify-between text-xs">
              <span className="text-[#e2e8f0] font-medium">{s.ticker}</span>
              <span className="text-[#94a3b8]">{s.horizon_days}D</span>
              <span style={{ color: accent }}>{s.expected_return_pct?.toFixed(1)}%</span>
              <span className="text-[#64748b]">p={s.probability_up?.toFixed(2)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
