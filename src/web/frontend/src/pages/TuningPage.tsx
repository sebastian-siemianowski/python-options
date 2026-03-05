import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api';
import type { TuneAsset } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { Settings, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';

export default function TuningPage() {
  const [search, setSearch] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);

  const listQ = useQuery({ queryKey: ['tuneList'], queryFn: api.tuneList });
  const statsQ = useQuery({ queryKey: ['tuneStats'], queryFn: api.tuneStats });
  const detailQ = useQuery({
    queryKey: ['tuneDetail', selectedSymbol],
    queryFn: () => api.tuneDetail(selectedSymbol!),
    enabled: !!selectedSymbol,
  });

  if (listQ.isLoading) return <LoadingSpinner text="Loading tuning data..." />;

  const assets = listQ.data?.assets || [];
  const stats = statsQ.data;

  const filtered = assets.filter((a) =>
    a.symbol.toLowerCase().includes(search.toLowerCase())
  );

  const modelData = stats?.models_distribution
    ? Object.entries(stats.models_distribution)
        .map(([name, count]) => ({
          name: name.replace('phi_student_t_', 'φ-t ').replace('kalman_', 'K-').replace('_unified', ''),
          count,
        }))
        .sort((a, b) => b.count - a.count)
    : [];

  return (
    <>
      <PageHeader title="Model Tuning">
        {assets.length} assets tuned — BMA model competition with PIT calibration
      </PageHeader>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard title="Total Tuned" value={stats.total} icon={<Settings className="w-5 h-5" />} color="blue" />
          <StatCard title="PIT Pass" value={stats.pit_pass} icon={<CheckCircle className="w-5 h-5" />} color="green" />
          <StatCard title="PIT Fail" value={stats.pit_fail} icon={<XCircle className="w-5 h-5" />} color="red" />
          <StatCard title="Unknown" value={stats.pit_unknown} icon={<AlertCircle className="w-5 h-5" />} color="amber" />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Model distribution chart */}
        <div className="glass-card p-5 md:col-span-1">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Best Model Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 11 }} />
              <YAxis type="category" dataKey="name" width={100} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', borderRadius: 8, color: '#e2e8f0' }}
              />
              <Bar dataKey="count" fill="#AB47BC" radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Asset list */}
        <div className="glass-card md:col-span-2 overflow-hidden">
          <div className="p-3 border-b border-[#2a2a4a]">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search assets..."
              className="w-full px-3 py-1.5 rounded-lg bg-[#0f0f23] border border-[#2a2a4a] text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none focus:border-[#42A5F5]"
            />
          </div>
          <div className="overflow-y-auto max-h-[400px]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[#1a1a2e]">
                <tr className="border-b border-[#2a2a4a]">
                  <th className="text-left px-3 py-2 text-[#64748b]">Symbol</th>
                  <th className="text-left px-3 py-2 text-[#64748b]">Best Model</th>
                  <th className="text-center px-3 py-2 text-[#64748b]">Models</th>
                  <th className="text-center px-3 py-2 text-[#64748b]">PIT</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Size</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 100).map((a) => (
                  <tr
                    key={a.symbol}
                    onClick={() => setSelectedSymbol(a.symbol)}
                    className={`border-b border-[#2a2a4a]/50 cursor-pointer transition hover:bg-[#16213e]/50 ${
                      selectedSymbol === a.symbol ? 'bg-[#16213e]' : ''
                    }`}
                  >
                    <td className="px-3 py-2 font-medium text-[#e2e8f0]">{a.symbol}</td>
                    <td className="px-3 py-2 text-[#94a3b8]">
                      {a.best_model?.replace('phi_student_t_', 'φ-t ').replace('kalman_', 'K-')}
                    </td>
                    <td className="px-3 py-2 text-center text-[#94a3b8]">{a.num_models}</td>
                    <td className="px-3 py-2 text-center">
                      <PitBadge asset={a} />
                    </td>
                    <td className="px-3 py-2 text-right text-[#64748b]">{a.file_size_kb}KB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>

      {/* Detail panel */}
      {selectedSymbol && detailQ.data?.data && (
        <div className="glass-card p-5 mt-6">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">
            {selectedSymbol} — Tuning Detail
          </h3>
          <pre className="text-xs text-[#94a3b8] overflow-x-auto max-h-[400px] whitespace-pre-wrap">
            {JSON.stringify(detailQ.data.data, null, 2)}
          </pre>
        </div>
      )}
    </>
  );
}

function PitBadge({ asset }: { asset: TuneAsset }) {
  if (asset.ad_pass === true) return <span className="text-[#00E676]">✓</span>;
  if (asset.ad_pass === false) return <span className="text-[#FF1744]">✗</span>;
  return <span className="text-[#64748b]">—</span>;
}
