import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useRef, useEffect, useCallback, useSyncExternalStore, memo } from 'react';
import { api } from '../api';
import type { TuneAsset } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { Settings, CheckCircle, XCircle, AlertCircle, RefreshCw, Play, Square, Terminal } from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { formatModelName, formatModelNameShort } from '../utils/modelNames';
import {
  type RetuneStatus, type RetuneLogEntry,
  getRetuneSnapshot, subscribeRetune,
  setRetuneMode, setShowPanel,
  startRetune as storeStartRetune,
  stopRetune as storeStopRetune,
} from '../stores/retuneStore';

export default function TuningPage() {
  const [search, setSearch] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const logEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();

  // Subscribe to the module-level retune store (persists across navigation)
  const retune = useSyncExternalStore(subscribeRetune, getRetuneSnapshot);
  const retuneStatus = retune.status;
  const retuneLogs = retune.logs;
  const retuneMode = retune.mode;
  const showRetunePanel = retune.showPanel;

  const listQ = useQuery({ queryKey: ['tuneList'], queryFn: api.tuneList });
  const statsQ = useQuery({ queryKey: ['tuneStats'], queryFn: api.tuneStats });
  const detailQ = useQuery({
    queryKey: ['tuneDetail', selectedSymbol],
    queryFn: () => api.tuneDetail(selectedSymbol!),
    enabled: !!selectedSymbol,
  });

  // Auto-scroll logs
  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [retuneLogs]);

  const handleStartRetune = useCallback(() => {
    storeStartRetune(() => {
      queryClient.invalidateQueries({ queryKey: ['tuneList'] });
      queryClient.invalidateQueries({ queryKey: ['tuneStats'] });
    });
  }, [queryClient]);

  const handleStopRetune = useCallback(() => {
    storeStopRetune();
  }, []);

  if (listQ.isLoading) return <LoadingSpinner text="Loading tuning data..." />;

  const assets = listQ.data?.assets || [];
  const stats = statsQ.data;

  const filtered = assets.filter((a) =>
    a.symbol.toLowerCase().includes(search.toLowerCase())
  );

  const modelData = stats?.models_distribution
    ? Object.entries(stats.models_distribution)
        .map(([name, count]) => ({
          name: formatModelNameShort(name),
          count,
        }))
        .sort((a, b) => b.count - a.count)
    : [];

  return (
    <>
      <PageHeader
        title="Model Tuning"
        action={
          <div className="flex items-center gap-2">
            {/* Reload cache button */}
            <button
              onClick={async () => {
                await api.refreshTuneCache();
                queryClient.invalidateQueries({ queryKey: ['tuneList'] });
                queryClient.invalidateQueries({ queryKey: ['tuneStats'] });
              }}
              disabled={listQ.isFetching}
              className="flex items-center gap-1.5 px-3 py-2 rounded-lg bg-[#16213e] text-sm text-[#42A5F5] hover:bg-[#1a2744] border border-[#2a2a4a] transition disabled:opacity-50"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${listQ.isFetching ? 'animate-spin' : ''}`} />
              Reload
            </button>
            {/* Mode selector */}
            <select
              value={retuneMode}
              onChange={(e) => setRetuneMode(e.target.value as 'retune' | 'tune' | 'calibrate')}
              disabled={retuneStatus === 'running'}
              className="px-2 py-2 rounded-lg bg-[#0f0f23] border border-[#2a2a4a] text-xs text-[#94a3b8] outline-none"
            >
              <option value="retune">Full Retune</option>
              <option value="tune">Tune Only</option>
              <option value="calibrate">Calibrate Failed</option>
            </select>
            {/* Retune button */}
            {retuneStatus === 'running' ? (
              <button
                onClick={handleStopRetune}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[#FF1744]/20 text-sm text-[#FF1744] hover:bg-[#FF1744]/30 border border-[#FF1744]/30 transition"
              >
                <Square className="w-3.5 h-3.5" />
                Stop
              </button>
            ) : (
              <button
                onClick={handleStartRetune}
                className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[#00E676]/20 text-sm text-[#00E676] hover:bg-[#00E676]/30 border border-[#00E676]/30 transition"
              >
                <Play className="w-3.5 h-3.5" />
                {retuneMode === 'retune' ? 'Retune All' : retuneMode === 'tune' ? 'Tune' : 'Calibrate'}
              </button>
            )}
            {/* Toggle console */}
            {retuneLogs.length > 0 && (
              <button
                onClick={() => setShowPanel(!showRetunePanel)}
                className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-sm transition border ${
                  showRetunePanel
                    ? 'bg-[#42A5F5]/20 text-[#42A5F5] border-[#42A5F5]/30'
                    : 'bg-[#16213e] text-[#94a3b8] border-[#2a2a4a]'
                }`}
              >
                <Terminal className="w-3.5 h-3.5" />
              </button>
            )}
          </div>
        }
      >
        {assets.length} assets tuned — BMA model competition with PIT calibration
      </PageHeader>

      {/* Retune progress panel */}
      {showRetunePanel && (
        <RetunePanel
          status={retuneStatus}
          logs={retuneLogs}
          logEndRef={logEndRef}
          onClose={() => setShowPanel(false)}
        />
      )}

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 fade-up">
          <StatCard title="Total Tuned" value={stats.total} icon={<Settings className="w-5 h-5" />} color="blue" />
          <StatCard title="PIT Pass" value={stats.pit_pass} icon={<CheckCircle className="w-5 h-5" />} color="green" />
          <StatCard title="PIT Fail" value={stats.pit_fail} icon={<XCircle className="w-5 h-5" />} color="red" />
          <StatCard title="Unknown" value={stats.pit_unknown} icon={<AlertCircle className="w-5 h-5" />} color="amber" />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 fade-up-delay-1">
        {/* Model distribution chart */}
        <ModelDistributionChart data={modelData} />

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
                    className={`border-b border-[#2a2a4a]/50 cursor-pointer transition row-glow ${
                      selectedSymbol === a.symbol ? 'bg-[#16213e]' : ''
                    }`}
                  >
                    <td className="px-3 py-2 font-medium text-[#e2e8f0]">{a.symbol}</td>
                    <td className="px-3 py-2 text-[#94a3b8]">
                      {formatModelName(a.best_model)}
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

/* ── Model Distribution Chart (memoized to avoid Recharts re-render loops) ── */

const ModelDistributionChart = memo(function ModelDistributionChart({ data }: { data: { name: string; count: number }[] }) {
  return (
    <div className="glass-card p-5 md:col-span-1">
      <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Best Model Distribution</h3>
      <ResponsiveContainer width="100%" height={300}>
        <BarChart data={data} layout="vertical">
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
  );
});

/* ── Retune Progress Panel ────────────────────────────────────────── */

function RetunePanel({
  status, logs, logEndRef, onClose,
}: {
  status: RetuneStatus;
  logs: RetuneLogEntry[];
  logEndRef: React.RefObject<HTMLDivElement | null>;
  onClose: () => void;
}) {
  const progressCount = logs.filter(l => l.type === 'progress').length;
  const statusColor = status === 'running' ? '#42A5F5' : status === 'completed' ? '#00E676' : status === 'failed' ? '#FF1744' : '#64748b';

  return (
    <div className="glass-card mb-6 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-2.5 border-b border-[#2a2a4a] flex items-center justify-between">
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4" style={{ color: statusColor }} />
          <span className="text-sm font-medium text-[#e2e8f0]">Retune Output</span>
          {status === 'running' && (
            <span className="flex items-center gap-1.5 text-xs text-[#42A5F5]">
              <RefreshCw className="w-3 h-3 animate-spin" />
              Processing... ({progressCount} assets)
            </span>
          )}
          {status === 'completed' && (
            <span className="text-xs text-[#00E676]">Completed ({progressCount} assets)</span>
          )}
          {status === 'failed' && (
            <span className="text-xs text-[#FF1744]">Failed</span>
          )}
        </div>
        <button onClick={onClose} className="text-[#64748b] hover:text-[#94a3b8] text-sm">×</button>
      </div>
      {/* Log output */}
      <div className="bg-[#0a0a1a] p-3 overflow-y-auto max-h-[300px] font-mono text-[11px]">
        {logs.map((entry, i) => (
          <div key={i} className={`py-0.5 ${logColor(entry.type)}`}>
            {entry.message}
          </div>
        ))}
        <div ref={logEndRef} />
      </div>
    </div>
  );
}

function logColor(type: string): string {
  switch (type) {
    case 'progress': return 'text-[#00E676]';
    case 'phase': return 'text-[#42A5F5] font-medium';
    case 'error': return 'text-[#FF1744]';
    case 'completed': return 'text-[#00E676] font-bold';
    case 'failed': return 'text-[#FF1744] font-bold';
    default: return 'text-[#64748b]';
  }
}

function PitBadge({ asset }: { asset: TuneAsset }) {
  if (asset.ad_pass === true) return <span className="text-[#00E676]">✓</span>;
  if (asset.ad_pass === false) return <span className="text-[#FF1744]">✗</span>;
  return <span className="text-[#64748b]">—</span>;
}

