import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useRef, useEffect, useCallback, useSyncExternalStore, useMemo, memo } from 'react';
import { api } from '../api';
import type { TuneAsset } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { TuningSkeleton } from '../components/CosmicSkeleton';
import { Settings, CheckCircle, XCircle, AlertCircle, RefreshCw, Play, Square, Terminal, Copy, ArrowDown, ArrowRight } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
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

type TuneMode = 'retune' | 'tune' | 'calibrate';

export default function TuningPage() {
  const [search, setSearch] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [showFailuresOnly, setShowFailuresOnly] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('grid');
  const logEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const navigate = useNavigate();

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

  if (listQ.isLoading) return <TuningSkeleton />;

  const assets = listQ.data?.assets || [];
  const stats = statsQ.data;

  const filtered = assets.filter((a) => {
    const matchSearch = a.symbol.toLowerCase().includes(search.toLowerCase());
    const matchFilter = showFailuresOnly ? a.ad_pass === false : true;
    return matchSearch && matchFilter;
  });

  const modelData = stats?.models_distribution
    ? Object.entries(stats.models_distribution)
        .map(([name, count]) => ({ name: formatModelNameShort(name), fullName: name, count }))
        .sort((a, b) => b.count - a.count)
    : [];

  const modeLabels: { id: TuneMode; label: string }[] = [
    { id: 'retune', label: 'Full Retune' },
    { id: 'tune', label: 'Tune Only' },
    { id: 'calibrate', label: 'Calibrate Failed' },
  ];

  return (
    <>
      <PageHeader
        title="Model Tuning"
        action={
          <div className="flex items-center gap-2">
            <button
              onClick={async () => {
                await api.refreshTuneCache();
                queryClient.invalidateQueries({ queryKey: ['tuneList'] });
                queryClient.invalidateQueries({ queryKey: ['tuneStats'] });
              }}
              disabled={listQ.isFetching}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm transition disabled:opacity-50"
              style={{
                background: 'var(--violet-8)',
                color: '#b49aff',
                border: '1px solid var(--violet-12)',
              }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${listQ.isFetching ? 'animate-spin' : ''}`} />
              Reload
            </button>
          </div>
        }
      >
        {assets.length} assets tuned -- BMA model competition with PIT calibration
      </PageHeader>

      {/* ── Mission Control Panel (Story 6.1) ─────────────────────── */}
      <div className="glass-card p-6 mb-6 fade-up" style={{
        background: 'linear-gradient(135deg, var(--violet-4) 0%, rgba(99,102,241,0.02) 50%, var(--violet-4) 100%)',
        backdropFilter: 'blur(32px)',
      }}>
        <div className="flex items-center gap-4 flex-wrap">
          {/* Mode selector - segmented pills */}
          <div className="flex gap-1 p-0.5 rounded-2xl" style={{ background: 'var(--violet-4)' }}>
            {modeLabels.map(({ id, label }) => (
              <button key={id}
                onClick={() => setRetuneMode(id)}
                disabled={retuneStatus === 'running'}
                className="px-4 py-2 rounded-2xl text-[13px] font-medium transition-all duration-200 disabled:opacity-50"
                style={retuneMode === id ? {
                  background: 'var(--violet-20)',
                  color: '#b49aff',
                  border: '1px solid var(--violet-20)',
                  boxShadow: '0 0 8px var(--violet-10)',
                } : {
                  background: 'transparent',
                  color: '#94a3b8',
                  border: '1px solid transparent',
                }}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Start/Stop button */}
          {retuneStatus === 'running' ? (
            <button onClick={handleStopRetune}
              className="flex items-center gap-2 px-6 py-3 rounded-3xl text-sm font-semibold text-white transition-all duration-200"
              style={{
                background: 'linear-gradient(135deg, var(--accent-rose) 0%, #e11d48 100%)',
                minWidth: 160, height: 48,
                boxShadow: '0 0 0 4px var(--rose-15), 0 4px 20px rgba(255,107,138,0.2)',
                animation: 'pulse 1.5s ease-in-out infinite',
              }}
            >
              <Square className="w-4 h-4" /> Stop
            </button>
          ) : (
            <button onClick={handleStartRetune}
              className="flex items-center gap-2 px-6 py-3 rounded-3xl text-sm font-semibold text-white transition-all duration-200 hover:scale-[1.02]"
              style={{
                background: 'linear-gradient(135deg, var(--accent-violet) 0%, var(--accent-indigo) 100%)',
                minWidth: 160, height: 48,
                boxShadow: '0 4px 20px var(--violet-25)',
              }}
            >
              <Play className="w-4 h-4" /> Start Retune
            </button>
          )}

          {/* Status badge */}
          <StatusBadge status={retuneStatus} />

          {/* Console toggle */}
          {retuneLogs.length > 0 && (
            <button
              onClick={() => setShowPanel(!showRetunePanel)}
              className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm transition-all duration-200"
              style={{
                background: showRetunePanel ? 'var(--violet-15)' : 'var(--violet-6)',
                color: showRetunePanel ? '#b49aff' : '#94a3b8',
                border: `1px solid ${showRetunePanel ? 'var(--violet-25)' : 'var(--violet-8)'}`,
              }}
            >
              <Terminal className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>

      {/* Retune progress panel */}
      {showRetunePanel && (
        <RetunePanel
          status={retuneStatus}
          logs={retuneLogs}
          logEndRef={logEndRef}
          onClose={() => setShowPanel(false)}
        />
      )}

      {/* ── Stats cards ───────────────────────────────────────────── */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6 fade-up">
          <StatCard title="Total Tuned" value={stats.total} icon={<Settings className="w-5 h-5" />} color="blue" />
          <StatCard title="PIT Pass" value={stats.pit_pass} icon={<CheckCircle className="w-5 h-5" />} color="green" />
          <StatCard title="PIT Fail" value={stats.pit_fail} icon={<XCircle className="w-5 h-5" />} color="red" />
          <StatCard title="Unknown" value={stats.pit_unknown} icon={<AlertCircle className="w-5 h-5" />} color="amber" />
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 fade-up-delay-1">
        {/* ── Model Distribution (Story 6.2) ──────────────────────── */}
        <ModelDistributionChart data={modelData} />

        {/* ── Asset Health Grid / Table (Story 6.3) ───────────────── */}
        <div className="glass-card md:col-span-2 overflow-hidden">
          <div className="px-3 py-2.5 flex items-center gap-2" style={{ borderBottom: '1px solid var(--violet-8)' }}>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search assets..."
              className="flex-1 px-3 py-1.5 rounded-lg text-sm outline-none transition-all duration-200"
              style={{
                background: 'rgba(10,10,26,0.6)',
                border: '1px solid var(--violet-8)',
                color: 'var(--text-primary)',
              }}
            />
            {/* Failures only toggle */}
            <button
              onClick={() => setShowFailuresOnly(!showFailuresOnly)}
              className="px-2.5 py-1 rounded-lg text-[11px] font-medium transition-all duration-200"
              style={{
                background: showFailuresOnly ? 'var(--rose-15)' : 'transparent',
                color: showFailuresOnly ? 'var(--accent-rose)' : '#7a8ba4',
                border: `1px solid ${showFailuresOnly ? 'rgba(255,107,138,0.2)' : 'var(--violet-6)'}`,
              }}
            >
              {showFailuresOnly ? 'Failures' : 'All'}
            </button>
            {/* View mode toggle */}
            <div className="flex gap-0.5">
              {(['grid', 'table'] as const).map(m => (
                <button key={m}
                  onClick={() => setViewMode(m)}
                  className="px-2 py-1 rounded text-[11px] font-medium transition-all duration-150"
                  style={{
                    background: viewMode === m ? 'var(--violet-12)' : 'transparent',
                    color: viewMode === m ? '#b49aff' : '#7a8ba4',
                  }}
                >
                  {m === 'grid' ? 'Stars' : 'Table'}
                </button>
              ))}
            </div>
          </div>

          {/* Summary bar (Story 6.3 AC-4) */}
          {stats && (
            <div className="px-3 py-1.5 flex items-center gap-2 text-[10px]" style={{ background: 'var(--violet-2)' }}>
              <div className="flex-1 h-1.5 rounded-full overflow-hidden flex" style={{ background: 'var(--violet-4)' }}>
                <div style={{ flex: stats.pit_pass, background: 'linear-gradient(90deg, var(--accent-emerald), #6ff0c0)' }} />
                <div style={{ flex: stats.pit_fail, background: 'linear-gradient(90deg, var(--accent-rose), #ff5577)' }} />
                <div style={{ flex: stats.pit_unknown, background: 'rgba(100,116,139,0.3)' }} />
              </div>
              <span style={{ color: 'var(--accent-emerald)' }}>{stats.pit_pass} pass</span>
              <span style={{ color: 'var(--accent-rose)' }}>{stats.pit_fail} fail</span>
              <span style={{ color: '#7a8ba4' }}>{stats.pit_unknown} unk</span>
            </div>
          )}

          {viewMode === 'grid' ? (
            /* Star Map Grid (Story 6.3) */
            <div className="p-3 overflow-y-auto max-h-[400px]">
              <div className="flex flex-wrap gap-1.5">
                {filtered.slice(0, 200).map(a => {
                  const isPassing = a.ad_pass === true;
                  const isFailing = a.ad_pass === false;
                  const isSelected = selectedSymbol === a.symbol;
                  return (
                    <button key={a.symbol}
                      onClick={() => setSelectedSymbol(a.symbol)}
                      title={`${a.symbol} - ${formatModelName(a.best_model)} - PIT: ${a.ad_pass == null ? 'Unknown' : a.ad_pass ? 'Pass' : 'Fail'}`}
                      className="transition-all duration-150"
                      style={{
                        width: 20, height: 20, borderRadius: 4,
                        fontSize: 7, fontWeight: 600, lineHeight: '20px', textAlign: 'center',
                        color: isPassing ? 'var(--accent-emerald)' : isFailing ? 'var(--accent-rose)' : '#7a8ba4',
                        background: isPassing
                          ? 'var(--emerald-15)'
                          : isFailing
                          ? 'rgba(255,107,138,0.2)'
                          : 'rgba(100,116,139,0.08)',
                        boxShadow: isSelected
                          ? '0 0 0 2px var(--accent-violet), 0 0 8px var(--violet-30)'
                          : isPassing
                          ? '0 0 4px var(--emerald-30)'
                          : isFailing
                          ? '0 0 6px var(--rose-30)'
                          : 'none',
                        opacity: showFailuresOnly && !isFailing ? 0.15 : 1,
                        transform: isSelected ? 'scale(1.2)' : undefined,
                        animation: isFailing ? 'pulse 2s ease-in-out infinite' : undefined,
                      }}
                    >
                      {a.symbol.slice(0, 3)}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            /* Table View */
            <div className="overflow-y-auto max-h-[400px]">
              <table className="w-full text-xs">
                <thead className="sticky top-0" style={{ background: 'rgba(26,26,46,0.95)' }}>
                  <tr style={{ borderBottom: '1px solid var(--violet-8)' }}>
                    <th className="text-left px-3 py-2" style={{ color: 'var(--text-muted)' }}>Symbol</th>
                    <th className="text-left px-3 py-2" style={{ color: 'var(--text-muted)' }}>Best Model</th>
                    <th className="text-center px-3 py-2" style={{ color: 'var(--text-muted)' }}>Models</th>
                    <th className="text-center px-3 py-2" style={{ color: 'var(--text-muted)' }}>PIT</th>
                    <th className="text-right px-3 py-2" style={{ color: 'var(--text-muted)' }}>Size</th>
                  </tr>
                </thead>
                <tbody>
                  {filtered.slice(0, 100).map(a => (
                    <tr key={a.symbol}
                      onClick={() => setSelectedSymbol(a.symbol)}
                      className="cursor-pointer transition-colors duration-150"
                      style={{
                        borderBottom: '1px solid var(--violet-4)',
                        background: selectedSymbol === a.symbol ? 'var(--violet-6)' : undefined,
                      }}
                    >
                      <td className="px-3 py-2 font-medium" style={{ color: 'var(--text-luminous)' }}>{a.symbol}</td>
                      <td className="px-3 py-2" style={{ color: 'var(--text-secondary)' }}>{formatModelName(a.best_model)}</td>
                      <td className="px-3 py-2 text-center" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>{a.num_models}</td>
                      <td className="px-3 py-2 text-center"><PitBadge asset={a} /></td>
                      <td className="px-3 py-2 text-right" style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>{a.file_size_kb}KB</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ── Detail Panel (Story 6.4) ──────────────────────────────── */}
      {selectedSymbol && detailQ.data?.data && (
        <DetailPanel symbol={selectedSymbol} data={detailQ.data.data} onViewDiagnostics={() => navigate('/diagnostics')} />
      )}
    </>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 6.1: Status Badge
   ══════════════════════════════════════════════════════════════════ */

function StatusBadge({ status }: { status: RetuneStatus }) {
  const config = {
    idle: { label: 'Idle', bg: 'rgba(100,116,139,0.12)', color: '#94a3b8' },
    running: { label: 'Running', bg: 'var(--amber-12)', color: 'var(--accent-amber)' },
    completed: { label: 'Completed', bg: 'var(--emerald-12)', color: 'var(--accent-emerald)' },
    failed: { label: 'Failed', bg: 'var(--rose-12)', color: 'var(--accent-rose)' },
  }[status] ?? { label: status, bg: 'rgba(100,116,139,0.12)', color: '#94a3b8' };

  return (
    <span className="px-3 py-1 rounded-full text-[11px] font-medium" style={{
      background: config.bg, color: config.color,
    }}>
      {config.label}
    </span>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 6.1: Retune Panel (Cosmified Log Terminal)
   ══════════════════════════════════════════════════════════════════ */

function RetunePanel({
  status, logs, logEndRef, onClose,
}: {
  status: RetuneStatus;
  logs: RetuneLogEntry[];
  logEndRef: React.RefObject<HTMLDivElement | null>;
  onClose: () => void;
}) {
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const progressCount = logs.filter(l => l.type === 'progress').length;
  const currentAsset = [...logs].reverse().find(l => l.type === 'progress')?.message?.match(/\b([A-Z]{1,5})\b/)?.[1];

  const handleCopyLog = useCallback(() => {
    const text = logs.map(l => l.message).join('\n');
    navigator.clipboard.writeText(text);
  }, [logs]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 30;
    setAutoScroll(atBottom);
  }, []);

  useEffect(() => {
    if (autoScroll) logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs, autoScroll, logEndRef]);

  const statusColor = status === 'running' ? '#b49aff' : status === 'completed' ? 'var(--accent-emerald)' : status === 'failed' ? 'var(--accent-rose)' : '#7a8ba4';

  return (
    <div className="glass-card mb-6 overflow-hidden">
      <div className="px-4 py-2.5 flex items-center justify-between" style={{ borderBottom: '1px solid var(--violet-8)' }}>
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4" style={{ color: statusColor }} />
          <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Retune Output</span>
          {status === 'running' && (
            <div className="flex items-center gap-2 text-xs">
              <RefreshCw className="w-3 h-3 animate-spin" style={{ color: '#b49aff' }} />
              <span style={{ color: '#b49aff' }}>{progressCount} assets</span>
              {currentAsset && <span className="font-semibold" style={{ color: '#b49aff' }}>{currentAsset}</span>}
            </div>
          )}
          {status === 'completed' && <span className="text-xs" style={{ color: 'var(--accent-emerald)' }}>Done ({progressCount})</span>}
          {status === 'failed' && <span className="text-xs" style={{ color: 'var(--accent-rose)' }}>Failed</span>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleCopyLog} className="p-1 rounded transition" title="Copy log"
            style={{ color: '#7a8ba4' }}>
            <Copy className="w-3.5 h-3.5" />
          </button>
          <button onClick={onClose} className="text-sm transition" style={{ color: '#7a8ba4' }}>x</button>
        </div>
      </div>

      {/* Progress bar */}
      {status === 'running' && (
        <div className="px-4 py-1.5">
          <div className="h-1 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
            <div className="h-full rounded-full transition-all duration-500" style={{
              width: `${Math.min((progressCount / 147) * 100, 100)}%`,
              background: 'linear-gradient(90deg, var(--accent-violet), var(--accent-cyan))',
            }} />
          </div>
        </div>
      )}

      {/* Log terminal */}
      <div ref={scrollRef} onScroll={handleScroll}
        className="p-3 overflow-y-auto max-h-[240px] font-mono text-[11px]"
        style={{ background: 'rgba(10,10,26,0.8)', borderRadius: '0 0 12px 12px', position: 'relative' }}>
        {logs.map((entry, i) => (
          <div key={i} className={`py-0.5 ${logColor(entry.type)}`}>{entry.message}</div>
        ))}
        <div ref={logEndRef} />
        {/* Resume auto-scroll floating button */}
        {!autoScroll && (
          <button
            onClick={() => { setAutoScroll(true); logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }}
            className="sticky bottom-2 left-full ml-auto px-3 py-1 rounded-full text-[10px] font-medium flex items-center gap-1 transition"
            style={{ background: 'var(--violet-15)', color: '#b49aff', border: '1px solid var(--violet-20)' }}
          >
            <ArrowDown className="w-3 h-3" /> Resume
          </button>
        )}
      </div>
    </div>
  );
}

function logColor(type: string): string {
  switch (type) {
    case 'progress': return 'text-[var(--accent-emerald)]';
    case 'phase': return 'text-[var(--accent-cyan)] font-medium';
    case 'error': return 'text-[var(--accent-rose)] font-bold';
    case 'completed': return 'text-[var(--accent-emerald)] font-bold';
    case 'failed': return 'text-[var(--accent-rose)] font-bold';
    default: return 'text-[var(--text-secondary)]';
  }
}

/* ══════════════════════════════════════════════════════════════════
   Story 6.2: Model Distribution (Treemap + Bar toggle)
   ══════════════════════════════════════════════════════════════════ */

const MODEL_FAMILY_COLORS: Record<string, string> = {
  kalman: 'linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%)',
  phi: 'linear-gradient(135deg, #064e3b 0%, #065f46 100%)',
  momentum: 'linear-gradient(135deg, #78350f 0%, #92400e 100%)',
  default: 'linear-gradient(135deg, #581c87 0%, #7c3aed 100%)',
};

function modelFamily(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('kalman') && !lower.includes('phi')) return 'kalman';
  if (lower.includes('phi') || lower.includes('student')) return 'phi';
  if (lower.includes('momentum')) return 'momentum';
  return 'default';
}

const ModelDistributionChart = memo(function ModelDistributionChart({ data }: {
  data: { name: string; fullName?: string; count: number }[];
}) {
  const [mode, setMode] = useState<'treemap' | 'bar'>('treemap');
  const [hovered, setHovered] = useState<string | null>(null);
  const total = useMemo(() => data.reduce((s, d) => s + d.count, 0), [data]);

  return (
    <div className="glass-card p-5 md:col-span-1">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Model Distribution</h3>
        <div className="flex gap-0.5">
          {(['treemap', 'bar'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className="px-2 py-0.5 rounded text-[10px] font-medium transition-all duration-150"
              style={{
                background: mode === m ? 'var(--violet-12)' : 'transparent',
                color: mode === m ? '#b49aff' : '#7a8ba4',
              }}>
              {m === 'treemap' ? 'Map' : 'Bar'}
            </button>
          ))}
        </div>
      </div>

      {mode === 'treemap' ? (
        <div className="flex flex-wrap gap-1" style={{ minHeight: 200 }}>
          {data.map((d, i) => {
            const frac = d.count / (total || 1);
            const family = modelFamily(d.fullName ?? d.name);
            const bg = MODEL_FAMILY_COLORS[family] || MODEL_FAMILY_COLORS.default;
            const isHovered = hovered === d.name;
            const isDimmed = hovered !== null && !isHovered;
            // Width proportional to fraction, minimum 40px
            const w = Math.max(40, frac * 280);
            const h = Math.max(28, frac * 200);
            return (
              <div key={d.name}
                onMouseEnter={() => setHovered(d.name)}
                onMouseLeave={() => setHovered(null)}
                className="rounded-md flex flex-col items-center justify-center transition-all duration-200 cursor-default"
                style={{
                  width: w, height: h,
                  background: bg,
                  border: '1px solid var(--violet-10)',
                  opacity: isDimmed ? 0.4 : 1,
                  boxShadow: isHovered ? '0 0 12px var(--violet-25)' : undefined,
                  transform: isHovered ? 'scale(1.03)' : undefined,
                  animationDelay: `${i * 50}ms`,
                }}
              >
                {w > 50 && (
                  <span className="text-[8px] font-medium truncate px-1" style={{
                    color: 'rgba(255,255,255,0.8)', maxWidth: w - 8,
                  }}>
                    {d.name}
                  </span>
                )}
                <span className="text-[10px] font-bold" style={{ color: 'rgba(255,255,255,0.95)', fontFamily: 'monospace' }}>
                  {d.count}
                </span>
              </div>
            );
          })}
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={data} layout="vertical">
            <CartesianGrid strokeDasharray="3 3" stroke="var(--violet-6)" />
            <XAxis type="number" tick={{ fill: '#7a8ba4', fontSize: 11 }} />
            <YAxis type="category" dataKey="name" width={100} tick={{ fill: '#94a3b8', fontSize: 10 }} />
            <Tooltip
              contentStyle={{
                background: 'rgba(15,15,35,0.95)',
                border: '1px solid var(--violet-15)',
                borderRadius: 8,
                color: '#e2e8f0',
                backdropFilter: 'blur(12px)',
              }}
            />
            <Bar dataKey="count" fill="var(--accent-violet)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
});

/* ══════════════════════════════════════════════════════════════════
   Story 6.4: Detail Panel (Cosmified)
   ══════════════════════════════════════════════════════════════════ */

function DetailPanel({ symbol, data, onViewDiagnostics }: { symbol: string; data: Record<string, unknown>; onViewDiagnostics: () => void }) {
  const best = data.best_model as string | undefined;
  const regime = data.regime as string | undefined;
  const pitPass = data.ad_pass as boolean | undefined;
  const models = data.competing_models as Array<{
    name: string; bic?: number; crps?: number; hyv?: number; pit_p?: number; bma_weight?: number; nu?: number;
  }> | undefined;
  const kalman = data.kalman_state as Record<string, number> | undefined;

  return (
    <div className="glass-card p-5 mt-6 fade-up" style={{
      background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
    }}>
      {/* Header */}
      <div className="flex items-center gap-3 mb-4">
        <h3 className="text-xl font-bold" style={{
          background: 'linear-gradient(135deg, #b49aff, #818cf8)',
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
        }}>
          {symbol}
        </h3>
        {regime && (
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium"
            style={{ background: 'var(--violet-12)', color: '#b49aff' }}>
            {regime}
          </span>
        )}
        {pitPass != null && (
          <span className="px-2 py-0.5 rounded-full text-[10px] font-medium"
            style={{
              background: pitPass ? 'var(--emerald-12)' : 'var(--rose-12)',
              color: pitPass ? 'var(--accent-emerald)' : 'var(--accent-rose)',
            }}>
            PIT {pitPass ? 'Pass' : 'Fail'}
          </span>
        )}
      </div>

      {/* Best model */}
      {best && (
        <div className="mb-4 flex items-center justify-between">
          <div>
            <span className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Best Model</span>
            <div className="text-sm font-semibold mt-0.5" style={{ color: '#b49aff' }}>{formatModelName(best)}</div>
          </div>
          <button
            onClick={onViewDiagnostics}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-200 hover:scale-[1.02]"
            style={{
              color: '#b49aff',
              background: 'var(--violet-8)',
              border: '1px solid var(--violet-12)',
            }}
          >
            View in Diagnostics <ArrowRight className="w-3 h-3" />
          </button>
        </div>
      )}

      {/* Competing models table */}
      {models && models.length > 0 ? (
        <div className="overflow-x-auto mb-4">
          <table className="w-full text-xs">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--violet-8)' }}>
                <th className="text-left px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Model</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>BIC</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>CRPS</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Hyv</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>PIT p</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>BMA %</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => {
                const isWinner = m.name === best;
                return (
                  <tr key={i} style={{
                    borderBottom: '1px solid var(--violet-4)',
                    background: isWinner ? 'var(--emerald-6)' : undefined,
                    borderLeft: isWinner ? '2px solid var(--accent-emerald)' : '2px solid transparent',
                  }}>
                    <td className="px-2 py-1.5 font-medium" style={{ color: 'var(--text-primary)' }}>
                      {formatModelName(m.name)}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                      {m.bic?.toFixed(0) ?? '--'}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{
                      color: m.crps != null ? (m.crps < 0.02 ? 'var(--accent-emerald)' : m.crps < 0.03 ? 'var(--accent-amber)' : 'var(--accent-rose)') : 'var(--text-muted)',
                      fontVariantNumeric: 'tabular-nums',
                    }}>
                      {m.crps?.toFixed(4) ?? '--'}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{
                      color: m.hyv != null ? (m.hyv < 500 ? 'var(--accent-emerald)' : m.hyv < 1000 ? 'var(--accent-amber)' : 'var(--accent-rose)') : 'var(--text-muted)',
                      fontVariantNumeric: 'tabular-nums',
                    }}>
                      {m.hyv?.toFixed(0) ?? '--'}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{
                      color: m.pit_p != null ? (m.pit_p >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)') : 'var(--text-muted)',
                      fontVariantNumeric: 'tabular-nums',
                    }}>
                      {m.pit_p?.toFixed(3) ?? '--'}
                    </td>
                    <td className="px-2 py-1.5 text-right" style={{ position: 'relative' }}>
                      <div className="flex items-center justify-end gap-1">
                        <div className="h-1 rounded-full" style={{
                          width: `${Math.min((m.bma_weight ?? 0) * 100, 80)}px`,
                          background: 'linear-gradient(90deg, var(--accent-violet), var(--text-violet))',
                        }} />
                        <span className="font-mono text-[10px]" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                          {m.bma_weight != null ? `${(m.bma_weight * 100).toFixed(1)}%` : '--'}
                        </span>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      ) : (
        /* Raw JSON fallback for assets without structured competing_models */
        <div>
          <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Raw Parameters</h4>
          <pre className="text-xs overflow-x-auto max-h-[300px] whitespace-pre-wrap p-3 rounded-xl font-mono"
            style={{ background: 'rgba(10,10,26,0.6)', color: 'var(--text-secondary)', border: '1px solid var(--violet-6)' }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}

      {/* Kalman state grid */}
      {kalman && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mt-4">
          {Object.entries(kalman).slice(0, 8).map(([key, val]) => (
            <div key={key} className="p-2 rounded-lg" style={{ background: 'var(--violet-4)' }}>
              <div className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: 'var(--text-muted)' }}>{key}</div>
              <div className="text-sm font-mono font-medium" style={{ color: 'var(--text-luminous)', fontVariantNumeric: 'tabular-nums' }}>
                {typeof val === 'number' ? val.toFixed(6) : String(val)}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Helper: PIT Badge
   ══════════════════════════════════════════════════════════════════ */

function PitBadge({ asset }: { asset: TuneAsset }) {
  if (asset.ad_pass === true)
    return <span className="inline-block w-2 h-2 rounded-full" style={{ background: 'var(--accent-emerald)', boxShadow: '0 0 4px rgba(62,232,165,0.4)' }} />;
  if (asset.ad_pass === false)
    return <span className="inline-block w-2 h-2 rounded-full" style={{ background: 'var(--accent-rose)', boxShadow: '0 0 4px rgba(255,107,138,0.4)' }} />;
  return <span className="inline-block w-2 h-2 rounded-full" style={{ background: '#7a8ba4' }} />;
}
