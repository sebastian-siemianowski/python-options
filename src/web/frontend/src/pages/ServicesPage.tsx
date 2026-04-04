import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import type { ServicesHealth, ServiceError } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import {
  HeartPulse, Server, Database, HardDrive, Users,
  CheckCircle, AlertTriangle, XCircle, Clock, Cpu, MemoryStick,
  RefreshCw,
} from 'lucide-react';

export default function ServicesPage() {
  const { data, isLoading, error, refetch, isFetching, dataUpdatedAt } = useQuery({
    queryKey: ['servicesHealth'],
    queryFn: api.servicesHealth,
    refetchInterval: 10_000, // Auto-refresh every 10s
  });

  const errorsQ = useQuery({
    queryKey: ['servicesErrors'],
    queryFn: api.servicesErrors,
    refetchInterval: 15_000,
  });

  if (isLoading) return <LoadingSpinner text="Checking services..." />;
  if (error || !data) return <CosmicErrorCard title="Unable to check services" error={error as Error} isNetworkError onRetry={() => window.location.reload()} />;

  const workersOk = data.workers.status === 'ok' || data.workers.status === 'degraded'
    || data.workers.redis?.status === 'not_running' || data.workers.celery?.status === 'not_running';
  const allOk = data.api.status === 'ok' && data.signal_cache.status !== 'missing'
    && data.price_data.status === 'ok' && workersOk;

  const lastRefresh = dataUpdatedAt ? new Date(dataUpdatedAt).toLocaleTimeString() : 'N/A';

  return (
    <>
      <PageHeader
        title="Services Monitor"
        action={
          <button
            onClick={() => refetch()}
            disabled={isFetching}
            className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] transition-all duration-200 disabled:opacity-50"
            style={{
              background: 'var(--violet-8)',
              color: '#b49aff',
              border: '1px solid var(--violet-12)',
            }}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isFetching ? 'animate-spin' : ''}`} />
            {isFetching ? 'Checking...' : 'Refresh'}
          </button>
        }
      >
        Real-time system health monitoring
      </PageHeader>

      {/* Status hero */}
      <div className={`glass-card p-8 mb-8 fade-up ambient-glow ${allOk ? 'glow-green' : 'glow-red'}`}>
        <div className="relative flex items-center gap-5">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center"
               style={{ background: `${allOk ? 'var(--accent-emerald)' : 'var(--accent-rose)'}10` }}>
            <HeartPulse className="w-8 h-8" style={{ color: allOk ? 'var(--accent-emerald)' : 'var(--accent-rose)' }} />
          </div>
          <div>
            <h2 className="text-2xl font-bold tracking-tight" style={{ color: allOk ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
              {allOk ? 'All Systems Operational' : 'Issues Detected'}
            </h2>
            <p className="text-[12px] text-[var(--text-secondary)] mt-1">Last check: {lastRefresh} {'\u2022'} Auto-refresh every 10s</p>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full pulse-dot" style={{ background: allOk ? 'var(--accent-emerald)' : 'var(--accent-rose)' }} />
            <span className="text-xs text-[var(--text-secondary)] font-medium">Live</span>
          </div>
        </div>
      </div>

      {/* Service cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-5 mb-8 fade-up-delay-1">
        <ApiCard data={data.api} />
        <CacheCard data={data.signal_cache} />
        <PriceDataCard data={data.price_data} />
        <WorkersCard data={data.workers} />
      </div>

      {/* Error log */}
      <ErrorLog errors={errorsQ.data?.errors || data.recent_errors || []} />
    </>
  );
}

/* ── Status helpers ──────────────────────────────────────────────── */
function StatusIcon({ status }: { status: string }) {
  if (status === 'ok' || status === 'fresh') return <CheckCircle className="w-5 h-5" style={{ color: 'var(--accent-emerald)' }} />;
  if (status === 'stale' || status === 'warning') return <AlertTriangle className="w-5 h-5" style={{ color: 'var(--accent-amber)' }} />;
  return <XCircle className="w-5 h-5" style={{ color: 'var(--accent-rose)' }} />;
}

function statusBg(status: string) {
  if (status === 'ok' || status === 'fresh') return 'border-emerald-500/10';
  if (status === 'stale' || status === 'warning') return 'border-amber-500/10';
  return 'border-rose-500/10';
}

/* ── API Card ────────────────────────────────────────────────────── */
function ApiCard({ data }: { data: ServicesHealth['api'] }) {
  return (
    <div className={`glass-card p-5 border-l-2 hover-lift ${statusBg(data.status)}`}>
      <div className="flex items-center gap-2.5 mb-4">
        <div className="w-8 h-8 rounded-xl flex items-center justify-center" style={{ background: 'var(--violet-10)' }}>
          <Server className="w-4 h-4" style={{ color: '#b49aff' }} />
        </div>
        <h3 className="text-[13px] font-medium" style={{ color: 'var(--text-luminous)' }}>API Server</h3>
        <div className="ml-auto"><StatusIcon status={data.status} /></div>
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <Metric icon={<Clock className="w-3 h-3" />} label="Uptime" value={data.uptime_human} />
        <Metric icon={<MemoryStick className="w-3 h-3" />} label="Memory" value={`${data.memory_mb.toFixed(0)} MB`} />
        <Metric icon={<Cpu className="w-3 h-3" />} label="CPU" value={`${data.cpu_percent.toFixed(1)}%`} />
        <Metric icon={<Server className="w-3 h-3" />} label="PID" value={String(data.pid)} />
      </div>
    </div>
  );
}

/* ── Cache Card ──────────────────────────────────────────────────── */
function CacheCard({ data }: { data: ServicesHealth['signal_cache'] }) {
  return (
    <div className={`glass-card p-4 border-l-2 hover-lift ${statusBg(data.status)}`}>
      <div className="flex items-center gap-2 mb-3">
        <Database className="w-4 h-4" style={{ color: '#b49aff' }} />
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-luminous)' }}>Signal Cache</h3>
        <StatusIcon status={data.status} />
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <Metric label="Status" value={data.exists ? 'Exists' : 'Missing'} />
        <Metric label="Age" value={data.age_human || 'N/A'} />
        <Metric label="Size" value={`${data.size_mb.toFixed(1)} MB`} />
        <Metric label="Last Modified" value={data.last_modified ? new Date(data.last_modified).toLocaleString() : 'N/A'} />
      </div>
    </div>
  );
}

/* ── Price Data Card ─────────────────────────────────────────────── */
function PriceDataCard({ data }: { data: ServicesHealth['price_data'] }) {
  return (
    <div className={`glass-card p-4 border-l-2 hover-lift ${statusBg(data.status)}`}>
      <div className="flex items-center gap-2 mb-3">
        <HardDrive className="w-4 h-4" style={{ color: 'var(--accent-amber)' }} />
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-luminous)' }}>Price Data</h3>
        <StatusIcon status={data.status} />
      </div>
      <div className="grid grid-cols-2 gap-3 text-xs">
        <Metric label="Files" value={String(data.total_files)} />
        <Metric label="Stale" value={`${data.stale_files} files`}
          valueColor={data.stale_files > 10 ? 'var(--accent-rose)' : data.stale_files > 0 ? 'var(--accent-amber)' : 'var(--accent-emerald)'} />
        <Metric label="Freshest" value={data.freshest_hours ? `${data.freshest_hours.toFixed(1)}h` : 'N/A'} />
        <Metric label="Size" value={data.total_size_mb ? `${data.total_size_mb.toFixed(0)} MB` : 'N/A'} />
      </div>
    </div>
  );
}

/* ── Workers Card ────────────────────────────────────────────────── */
function WorkersCard({ data }: { data: ServicesHealth['workers'] }) {
  return (
    <div className={`glass-card p-4 border-l-2 hover-lift ${statusBg(data.status)}`}>
      <div className="flex items-center gap-2 mb-3">
        <Users className="w-4 h-4" style={{ color: 'var(--accent-cyan)' }} />
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-luminous)' }}>Background Workers</h3>
        <StatusIcon status={data.status} />
      </div>
      <div className="space-y-2 text-xs">
        <div className="flex items-center justify-between">
          <span className="text-[var(--text-secondary)]">Redis</span>
          <span style={{ color: data.redis.status === 'ok' ? 'var(--accent-emerald)' : '#94a3b8' }}>
            {data.redis.status === 'ok' ? (data.redis.used_memory_human || 'Connected') : (data.redis.message || 'Not running (optional)')}
          </span>
        </div>
        <div className="flex items-center justify-between">
          <span className="text-[var(--text-secondary)]">Celery</span>
          <span style={{ color: data.celery.status === 'ok' ? 'var(--accent-emerald)' : '#94a3b8' }}>
            {data.celery.status === 'ok' ? `${data.celery.workers} worker(s)` : (data.celery.message || 'Not running (optional)')}
          </span>
        </div>
        {data.celery.worker_names?.map((w) => (
          <div key={w} className="pl-4 text-[10px] text-[var(--text-secondary)]">{w}</div>
        ))}
      </div>
    </div>
  );
}

/* ── Metric ──────────────────────────────────────────────────────── */
function Metric({ icon, label, value, valueColor }: { icon?: React.ReactNode; label: string; value: string; valueColor?: string }) {
  return (
    <div className="flex items-center gap-1.5">
      {icon && <span className="text-[var(--text-secondary)]">{icon}</span>}
      <div>
        <p className="text-[10px] text-[var(--text-secondary)]">{label}</p>
        <p className="text-[#e2e8f0] font-medium" style={valueColor ? { color: valueColor } : {}}>{value}</p>
      </div>
    </div>
  );
}

/* ── Error Log ───────────────────────────────────────────────────── */
function ErrorLog({ errors }: { errors: ServiceError[] }) {
  return (
    <div className="glass-card overflow-hidden">
      <div className="px-4 py-3 flex items-center gap-2" style={{ borderBottom: '1px solid var(--violet-8)' }}>
        <AlertTriangle className="w-4 h-4" style={{ color: 'var(--accent-amber)' }} />
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Recent Errors</h3>
        <span className="ml-auto text-xs text-[var(--text-secondary)]">{errors.length} entries</span>
      </div>
      {errors.length === 0 ? (
        <div className="px-4 py-6 text-center">
          <CheckCircle className="w-6 h-6 mx-auto mb-2" style={{ color: 'var(--accent-emerald)' }} />
          <p className="text-xs text-[var(--text-secondary)]">No recent errors</p>
        </div>
      ) : (
        <div className="max-h-64 overflow-y-auto" style={{ borderTop: errors.length ? undefined : undefined }}>
          {errors.map((e, i) => (
            <div key={i} className="px-4 py-2 flex items-start gap-3 text-xs" style={{ borderBottom: '1px solid var(--violet-4)' }}>
              <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" style={{ color: 'var(--accent-rose)' }} />
              <div className="flex-1 min-w-0">
                <span className="font-medium text-[#e2e8f0]">[{e.source}]</span>{' '}
                <span className="text-[var(--text-secondary)]">{e.message}</span>
              </div>
              <span className="text-[10px] text-[var(--text-secondary)] whitespace-nowrap">
                {new Date(e.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
