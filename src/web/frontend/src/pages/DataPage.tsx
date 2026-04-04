import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api';
import type { PriceFile } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { Database, HardDrive, Clock, AlertTriangle, FolderOpen, RefreshCw } from 'lucide-react';

export default function DataPage() {
  const [search, setSearch] = useState('');
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMsg, setRefreshMsg] = useState<string | null>(null);

  const statusQ = useQuery({ queryKey: ['dataStatus'], queryFn: api.dataStatus });
  const pricesQ = useQuery({ queryKey: ['dataPrices'], queryFn: api.dataPrices });
  const dirsQ = useQuery({ queryKey: ['dataDirectories'], queryFn: api.dataDirectories });

  const handleRefresh = async () => {
    setRefreshing(true);
    setRefreshMsg(null);
    try {
      const resp = await api.triggerDataRefresh();
      setRefreshMsg(`Data refresh started (task ${resp.task_id})`);
      // Refetch status after a delay
      setTimeout(() => {
        statusQ.refetch();
        pricesQ.refetch();
        dirsQ.refetch();
      }, 5000);
    } catch (err) {
      setRefreshMsg('Failed to start data refresh');
    } finally {
      setRefreshing(false);
    }
  };

  if (statusQ.isLoading) return <LoadingSpinner text="Loading data status..." />;
  if (statusQ.error) return <CosmicErrorCard title="Unable to load data status" error={statusQ.error as Error} onRetry={() => statusQ.refetch()} />;

  const status = statusQ.data;
  const files = pricesQ.data?.files || [];
  const dirs = dirsQ.data || {};

  const filtered = files.filter((f) =>
    f.symbol.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <PageHeader
        title="Data Management"
        action={
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm transition-all duration-200 disabled:opacity-50"
            style={{
              background: 'rgba(139,92,246,0.08)',
              color: '#b49aff',
              border: '1px solid rgba(139,92,246,0.12)',
            }}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh Data
          </button>
        }
      >
        {status?.total_files || 0} price files • {status?.total_size_mb || 0} MB
      </PageHeader>

      {refreshMsg && (
        <div className="glass-card p-3 mb-4" style={{ borderLeft: '2px solid #8b5cf6' }}>
          <p className="text-xs" style={{ color: 'var(--text-secondary)' }}>{refreshMsg}</p>
        </div>
      )}

      {/* Stats */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 fade-up">
          <StatCard title="Total Files" value={status.total_files} icon={<Database className="w-5 h-5" />} color="blue" />
          <StatCard title="Fresh" value={status.fresh_files} subtitle="< 24h old" icon={<Clock className="w-5 h-5" />} color="green" />
          <StatCard title="Stale" value={status.stale_files} subtitle="> 24h old" icon={<AlertTriangle className="w-5 h-5" />} color={status.stale_files > 10 ? 'red' : 'amber'} />
          <StatCard title="Disk Usage" value={`${status.total_size_mb} MB`} icon={<HardDrive className="w-5 h-5" />} color="purple" />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 fade-up-delay-1">
        {/* Directories */}
        <div className="glass-card p-4 hover-lift">
          <h3 className="text-sm font-medium mb-3 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
            <FolderOpen className="w-4 h-4" /> Data Directories
          </h3>
          <div className="space-y-2">
            {Object.entries(dirs).map(([name, info]) => (
              <div key={name} className="flex items-center justify-between text-xs">
                <span style={{ color: 'var(--text-luminous)' }}>{name}</span>
                <span style={{ color: info.exists ? '#3ee8a5' : '#ff6b8a' }}>
                  {info.exists ? `${info.file_count} files` : 'missing'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Price file list */}
        <div className="glass-card md:col-span-2 overflow-hidden">
          <div className="p-3" style={{ borderBottom: '1px solid rgba(139,92,246,0.08)' }}>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search symbols..."
              className="w-full px-3 py-1.5 rounded-xl text-sm outline-none transition-all duration-200"
              style={{
                background: 'rgba(10,10,26,0.6)',
                border: '1px solid rgba(139,92,246,0.08)',
                color: 'var(--text-primary)',
              }}
            />
          </div>
          <div className="overflow-y-auto max-h-[500px]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 z-10" style={{ background: 'linear-gradient(135deg, rgba(26,5,51,0.97), rgba(13,27,62,0.97))', backdropFilter: 'blur(12px)' }}>
                <tr style={{ borderBottom: '1px solid rgba(139,92,246,0.08)' }}>
                  <th className="text-left px-3 py-2" style={{ color: 'var(--text-muted)' }}>Symbol</th>
                  <th className="text-right px-3 py-2" style={{ color: 'var(--text-muted)' }}>Rows</th>
                  <th className="text-right px-3 py-2" style={{ color: 'var(--text-muted)' }}>Size</th>
                  <th className="text-right px-3 py-2" style={{ color: 'var(--text-muted)' }}>Age</th>
                  <th className="text-right px-3 py-2" style={{ color: 'var(--text-muted)' }}>Updated</th>
                </tr>
              </thead>
              <tbody>
                {filtered.slice(0, 200).map((f) => (
                  <FileRow key={f.symbol} file={f} />
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  );
}

function FileRow({ file }: { file: PriceFile }) {
  const ageColor = file.age_hours < 24 ? '#3ee8a5'
    : file.age_hours < 72 ? '#f5c542'
    : '#ff6b8a';

  return (
    <tr style={{ borderBottom: '1px solid rgba(139,92,246,0.04)' }} className="transition-all duration-150">
      <td className="px-3 py-2 font-medium" style={{ color: 'var(--text-luminous)' }}>{file.symbol}</td>
      <td className="px-3 py-2 text-right" style={{ color: 'var(--text-secondary)' }}>{file.rows.toLocaleString()}</td>
      <td className="px-3 py-2 text-right" style={{ color: 'var(--text-secondary)' }}>{file.size_kb} KB</td>
      <td className="px-3 py-2 text-right" style={{ color: ageColor }}>
        {file.age_hours < 1 ? '<1h' : `${Math.round(file.age_hours)}h`}
      </td>
      <td className="px-3 py-2 text-right" style={{ color: '#7a8ba4' }}>
        {new Date(file.last_modified).toLocaleDateString()}
      </td>
    </tr>
  );
}
