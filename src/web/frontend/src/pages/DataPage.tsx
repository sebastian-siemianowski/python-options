import { useQuery } from '@tanstack/react-query';
import { useState } from 'react';
import { api } from '../api';
import type { PriceFile } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { Database, HardDrive, Clock, AlertTriangle, FolderOpen } from 'lucide-react';

export default function DataPage() {
  const [search, setSearch] = useState('');

  const statusQ = useQuery({ queryKey: ['dataStatus'], queryFn: api.dataStatus });
  const pricesQ = useQuery({ queryKey: ['dataPrices'], queryFn: api.dataPrices });
  const dirsQ = useQuery({ queryKey: ['dataDirectories'], queryFn: api.dataDirectories });

  if (statusQ.isLoading) return <LoadingSpinner text="Loading data status..." />;

  const status = statusQ.data;
  const files = pricesQ.data?.files || [];
  const dirs = dirsQ.data || {};

  const filtered = files.filter((f) =>
    f.symbol.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <>
      <PageHeader title="Data Management">
        {status?.total_files || 0} price files • {status?.total_size_mb || 0} MB
      </PageHeader>

      {/* Stats */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <StatCard title="Total Files" value={status.total_files} icon={<Database className="w-5 h-5" />} color="blue" />
          <StatCard title="Fresh" value={status.fresh_files} subtitle="< 24h old" icon={<Clock className="w-5 h-5" />} color="green" />
          <StatCard title="Stale" value={status.stale_files} subtitle="> 24h old" icon={<AlertTriangle className="w-5 h-5" />} color={status.stale_files > 10 ? 'red' : 'amber'} />
          <StatCard title="Disk Usage" value={`${status.total_size_mb} MB`} icon={<HardDrive className="w-5 h-5" />} color="purple" />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {/* Directories */}
        <div className="glass-card p-4">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3 flex items-center gap-2">
            <FolderOpen className="w-4 h-4" /> Data Directories
          </h3>
          <div className="space-y-2">
            {Object.entries(dirs).map(([name, info]) => (
              <div key={name} className="flex items-center justify-between text-xs">
                <span className="text-[#e2e8f0]">{name}</span>
                <span className={info.exists ? 'text-[#00E676]' : 'text-[#FF1744]'}>
                  {info.exists ? `${info.file_count} files` : 'missing'}
                </span>
              </div>
            ))}
          </div>
        </div>

        {/* Price file list */}
        <div className="glass-card md:col-span-2 overflow-hidden">
          <div className="p-3 border-b border-[#2a2a4a]">
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search symbols..."
              className="w-full px-3 py-1.5 rounded-lg bg-[#0f0f23] border border-[#2a2a4a] text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none focus:border-[#42A5F5]"
            />
          </div>
          <div className="overflow-y-auto max-h-[500px]">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-[#1a1a2e]">
                <tr className="border-b border-[#2a2a4a]">
                  <th className="text-left px-3 py-2 text-[#64748b]">Symbol</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Rows</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Size</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Age</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Updated</th>
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
  const ageColor = file.age_hours < 24 ? 'text-[#00E676]'
    : file.age_hours < 72 ? 'text-[#FFB300]'
    : 'text-[#FF1744]';

  return (
    <tr className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30 transition">
      <td className="px-3 py-2 font-medium text-[#e2e8f0]">{file.symbol}</td>
      <td className="px-3 py-2 text-right text-[#94a3b8]">{file.rows.toLocaleString()}</td>
      <td className="px-3 py-2 text-right text-[#94a3b8]">{file.size_kb} KB</td>
      <td className={`px-3 py-2 text-right ${ageColor}`}>
        {file.age_hours < 1 ? '<1h' : `${Math.round(file.age_hours)}h`}
      </td>
      <td className="px-3 py-2 text-right text-[#64748b]">
        {new Date(file.last_modified).toLocaleDateString()}
      </td>
    </tr>
  );
}
