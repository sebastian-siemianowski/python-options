import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useRef, useEffect, useCallback, useSyncExternalStore, useMemo, memo } from 'react';
import { api } from '../api';
import type { TuneAsset } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { TuningSkeleton } from '../components/CosmicSkeleton';
import {
  Settings, CheckCircle, XCircle, AlertCircle, RefreshCw, Play, Square, Terminal,
  Copy, ArrowDown, ArrowRight, Clock, Zap, TrendingUp, BarChart3, Activity,
  ChevronDown, ChevronUp, Search, Filter, Layers, Target, Timer, Award,
} from 'lucide-react';
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
type SortKey = 'symbol' | 'best_model' | 'num_models' | 'ad_pass' | 'bic' | 'phi' | 'nu' | 'n_obs' | 'top_weight' | 'file_size_kb';
type SortDir = 'asc' | 'desc' | null;

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const h = Math.floor(m / 60);
  if (h > 0) return `${h}h ${m % 60}m ${s % 60}s`;
  if (m > 0) return `${m}m ${s % 60}s`;
  return `${s}s`;
}

function useElapsedTime(startedAt: number | null, finishedAt: number | null, running: boolean) {
  const [now, setNow] = useState(Date.now());
  useEffect(() => {
    if (!running || !startedAt) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [running, startedAt]);
  if (!startedAt) return null;
  const end = finishedAt ?? (running ? now : startedAt);
  return end - startedAt;
}

export default function TuningPage() {
  const [search, setSearch] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [showFailuresOnly, setShowFailuresOnly] = useState(false);
  const [viewMode, setViewMode] = useState<'grid' | 'table'>('table');
  const [sortKey, setSortKey] = useState<SortKey>('symbol');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
  const logEndRef = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const navigate = useNavigate();

  const retune = useSyncExternalStore(subscribeRetune, getRetuneSnapshot);
  const retuneStatus = retune.status;
  const retuneLogs = retune.logs;
  const retuneMode = retune.mode;
  const showRetunePanel = retune.showPanel;

  const elapsed = useElapsedTime(retune.startedAt, retune.finishedAt, retuneStatus === 'running');

  const listQ = useQuery({ queryKey: ['tuneList'], queryFn: api.tuneList });
  const statsQ = useQuery({ queryKey: ['tuneStats'], queryFn: api.tuneStats });
  const detailQ = useQuery({
    queryKey: ['tuneDetail', selectedSymbol],
    queryFn: () => api.tuneDetail(selectedSymbol!),
    enabled: !!selectedSymbol,
  });

  useEffect(() => {
    if (showRetunePanel) logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [retuneLogs, showRetunePanel]);

  const handleStartRetune = useCallback(() => {
    storeStartRetune(() => {
      queryClient.invalidateQueries({ queryKey: ['tuneList'] });
      queryClient.invalidateQueries({ queryKey: ['tuneStats'] });
    });
  }, [queryClient]);

  const handleStopRetune = useCallback(() => {
    storeStopRetune();
  }, []);

  const handleSort = useCallback((key: SortKey) => {
    if (sortKey === key) {
      setSortDir(d => d === 'asc' ? 'desc' : d === 'desc' ? null : 'asc');
    } else {
      setSortKey(key);
      setSortDir('desc');
    }
  }, [sortKey]);

  if (listQ.isLoading) return <TuningSkeleton />;

  const assets = listQ.data?.assets || [];
  const stats = statsQ.data;

  const filtered = assets.filter((a: TuneAsset) => {
    const matchSearch = a.symbol.toLowerCase().includes(search.toLowerCase());
    const matchFilter = showFailuresOnly ? a.ad_pass === false : true;
    return matchSearch && matchFilter;
  });

  const sorted = sortDir ? [...filtered].sort((a: TuneAsset, b: TuneAsset) => {
    let av: number | string | boolean | null = null;
    let bv: number | string | boolean | null = null;
    switch (sortKey) {
      case 'symbol': av = a.symbol; bv = b.symbol; break;
      case 'best_model': av = a.best_model; bv = b.best_model; break;
      case 'num_models': av = a.num_models; bv = b.num_models; break;
      case 'ad_pass': av = a.ad_pass === true ? 1 : a.ad_pass === false ? -1 : 0; bv = b.ad_pass === true ? 1 : b.ad_pass === false ? -1 : 0; break;
      case 'bic': av = a.bic; bv = b.bic; break;
      case 'phi': av = a.phi; bv = b.phi; break;
      case 'nu': av = a.nu; bv = b.nu; break;
      case 'n_obs': av = a.n_obs; bv = b.n_obs; break;
      case 'top_weight': av = a.top_weight; bv = b.top_weight; break;
      case 'file_size_kb': av = a.file_size_kb; bv = b.file_size_kb; break;
    }
    if (av == null && bv == null) return 0;
    if (av == null) return 1;
    if (bv == null) return -1;
    const cmp = typeof av === 'string' ? av.localeCompare(bv as string) : (av as number) - (bv as number);
    return sortDir === 'asc' ? cmp : -cmp;
  }) : filtered;

  const modelData = stats?.models_distribution
    ? Object.entries(stats.models_distribution)
        .map(([name, count]) => ({ name: formatModelNameShort(name), fullName: name, count }))
        .sort((a, b) => b.count - a.count)
    : [];

  const modeLabels: { id: TuneMode; label: string; desc: string }[] = [
    { id: 'retune', label: 'Full Retune', desc: 'Re-estimate all models' },
    { id: 'tune', label: 'Tune Only', desc: 'Fit new assets only' },
    { id: 'calibrate', label: 'Calibrate', desc: 'Fix failing PIT only' },
  ];

  const passRate = stats ? Math.round((stats.pit_pass / Math.max(stats.total, 1)) * 100) : 0;

  return (
    <>
      <PageHeader
        title="Model Tuning"
        action={
          <button
            onClick={async () => {
              await api.refreshTuneCache();
              queryClient.invalidateQueries({ queryKey: ['tuneList'] });
              queryClient.invalidateQueries({ queryKey: ['tuneStats'] });
            }}
            disabled={listQ.isFetching}
            className="flex items-center gap-1.5 px-3 py-2 rounded-xl text-sm transition disabled:opacity-50"
            style={{ background: 'var(--violet-8)', color: '#b49aff', border: '1px solid var(--violet-12)' }}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${listQ.isFetching ? 'animate-spin' : ''}`} />
            Reload
          </button>
        }
      >
        {assets.length} assets tuned &middot; BMA model competition with PIT calibration
      </PageHeader>

      {/* ── Summary Cards ─────────────────────────────────────────── */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6 fade-up">
          <SummaryCard icon={<Layers className="w-4 h-4" />} label="Total Tuned" value={stats.total} color="#b49aff" />
          <SummaryCard icon={<CheckCircle className="w-4 h-4" />} label="PIT Pass" value={stats.pit_pass} sub={`${passRate}%`} color="var(--accent-emerald)" />
          <SummaryCard icon={<XCircle className="w-4 h-4" />} label="PIT Fail" value={stats.pit_fail} color="var(--accent-rose)" />
          <SummaryCard icon={<AlertCircle className="w-4 h-4" />} label="Unknown" value={stats.pit_unknown} color="var(--accent-amber)" />
          <SummaryCard icon={<BarChart3 className="w-4 h-4" />} label="Model Types" value={modelData.length} color="var(--accent-cyan)" />
        </div>
      )}

      {/* ── Mission Control ───────────────────────────────────────── */}
      <div className="glass-card p-5 mb-6 fade-up" style={{
        background: 'linear-gradient(135deg, var(--violet-4) 0%, rgba(99,102,241,0.03) 50%, var(--violet-4) 100%)',
      }}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3 flex-wrap">
            {/* Mode selector */}
            <div className="flex gap-1 p-0.5 rounded-2xl" style={{ background: 'var(--violet-4)' }}>
              {modeLabels.map(({ id, label }) => (
                <button key={id}
                  onClick={() => setRetuneMode(id)}
                  disabled={retuneStatus === 'running'}
                  className="px-4 py-2 rounded-2xl text-[13px] font-medium transition-all duration-200 disabled:opacity-50"
                  style={retuneMode === id ? {
                    background: 'var(--violet-20)', color: '#b49aff',
                    border: '1px solid var(--violet-20)', boxShadow: '0 0 8px var(--violet-10)',
                  } : { background: 'transparent', color: '#94a3b8', border: '1px solid transparent' }}
                >
                  {label}
                </button>
              ))}
            </div>

            {/* Start/Stop */}
            {retuneStatus === 'running' ? (
              <button onClick={handleStopRetune}
                className="flex items-center gap-2 px-5 py-2.5 rounded-2xl text-sm font-semibold text-white transition-all"
                style={{
                  background: 'linear-gradient(135deg, var(--accent-rose) 0%, #e11d48 100%)',
                  boxShadow: '0 0 0 3px var(--rose-15), 0 4px 16px rgba(255,107,138,0.2)',
                  animation: 'pulse 1.5s ease-in-out infinite',
                }}
              >
                <Square className="w-3.5 h-3.5" /> Stop
              </button>
            ) : (
              <button onClick={handleStartRetune}
                className="flex items-center gap-2 px-5 py-2.5 rounded-2xl text-sm font-semibold text-white transition-all hover:scale-[1.02]"
                style={{
                  background: 'linear-gradient(135deg, var(--accent-violet) 0%, var(--accent-indigo) 100%)',
                  boxShadow: '0 4px 16px var(--violet-25)',
                }}
              >
                <Play className="w-3.5 h-3.5" /> Start
              </button>
            )}

            <StatusBadge status={retuneStatus} />
          </div>

          {/* Live stats during retune */}
          <div className="flex items-center gap-4 text-xs">
            {elapsed != null && (
              <div className="flex items-center gap-1.5" style={{ color: '#b49aff' }}>
                <Timer className="w-3.5 h-3.5" />
                <span className="font-mono font-medium">{formatElapsed(elapsed)}</span>
              </div>
            )}
            {retuneStatus === 'running' && retune.totalAssets > 0 && (
              <>
                <div className="flex items-center gap-1.5" style={{ color: 'var(--accent-emerald)' }}>
                  <CheckCircle className="w-3 h-3" /> {retune.successCount}
                </div>
                {retune.failCount > 0 && (
                  <div className="flex items-center gap-1.5" style={{ color: 'var(--accent-rose)' }}>
                    <XCircle className="w-3 h-3" /> {retune.failCount}
                  </div>
                )}
                {retune.currentAsset && (
                  <span className="font-mono font-semibold" style={{ color: '#b49aff' }}>{retune.currentAsset}</span>
                )}
              </>
            )}
            {retuneLogs.length > 0 && (
              <button
                onClick={() => setShowPanel(!showRetunePanel)}
                className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-[11px] transition-all"
                style={{
                  background: showRetunePanel ? 'var(--violet-15)' : 'var(--violet-6)',
                  color: showRetunePanel ? '#b49aff' : '#94a3b8',
                  border: `1px solid ${showRetunePanel ? 'var(--violet-25)' : 'var(--violet-8)'}`,
                }}
              >
                <Terminal className="w-3 h-3" /> Log
              </button>
            )}
          </div>
        </div>

        {/* Progress bar */}
        {retuneStatus === 'running' && retune.totalAssets > 0 && (
          <div className="mt-3">
            <div className="flex items-center justify-between text-[10px] mb-1" style={{ color: '#94a3b8' }}>
              <span>{retune.currentPhase || 'Processing...'}</span>
              <span className="font-mono">{retune.totalAssets} assets processed</span>
            </div>
            <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
              <div className="h-full rounded-full transition-all duration-700 ease-out" style={{
                width: `${Math.min((retune.totalAssets / Math.max(assets.length, 1)) * 100, 100)}%`,
                background: 'linear-gradient(90deg, var(--accent-violet), var(--accent-cyan))',
              }} />
            </div>
          </div>
        )}
      </div>

      {/* ── Retune Log Panel ──────────────────────────────────────── */}
      {showRetunePanel && (
        <RetunePanel status={retuneStatus} logs={retuneLogs} logEndRef={logEndRef}
          onClose={() => setShowPanel(false)} elapsed={elapsed} retune={retune} />
      )}

      {/* ── Main Content Grid ─────────────────────────────────────── */}
      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 fade-up-delay-1">
        {/* Model Distribution */}
        <ModelDistributionChart data={modelData} />

        {/* Asset Table */}
        <div className="glass-card lg:col-span-3 overflow-hidden">
          {/* Controls bar */}
          <div className="px-3 py-2.5 flex items-center gap-2" style={{ borderBottom: '1px solid var(--violet-8)' }}>
            <div className="relative flex-1">
              <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5" style={{ color: '#7a8ba4' }} />
              <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
                placeholder="Search assets..."
                className="w-full pl-8 pr-3 py-2 rounded-lg text-sm outline-none transition-all focus-ring"
                style={{ background: 'rgba(10,10,26,0.6)', border: '1px solid var(--violet-8)', color: 'var(--text-primary)' }}
              />
            </div>
            <button onClick={() => setShowFailuresOnly(!showFailuresOnly)}
              className="px-2.5 py-1.5 rounded-lg text-[11px] font-medium transition-all"
              style={{
                background: showFailuresOnly ? 'var(--rose-15)' : 'transparent',
                color: showFailuresOnly ? 'var(--accent-rose)' : '#7a8ba4',
                border: `1px solid ${showFailuresOnly ? 'rgba(255,107,138,0.2)' : 'var(--violet-6)'}`,
              }}
            >
              {showFailuresOnly ? 'Failures Only' : 'All'}
            </button>
            <div className="flex gap-0.5">
              {(['grid', 'table'] as const).map(m => (
                <button key={m} onClick={() => setViewMode(m)}
                  className="px-2 py-1 rounded text-[11px] font-medium transition-all"
                  style={{ background: viewMode === m ? 'var(--violet-12)' : 'transparent', color: viewMode === m ? '#b49aff' : '#7a8ba4' }}
                >
                  {m === 'grid' ? 'Grid' : 'Table'}
                </button>
              ))}
            </div>
            <span className="text-[10px] font-mono" style={{ color: '#7a8ba4' }}>{sorted.length} assets</span>
          </div>

          {/* Health bar */}
          {stats && (
            <div className="px-3 py-1.5 flex items-center gap-2 text-[10px]" style={{ background: 'var(--violet-2)' }}>
              <div className="flex-1 h-2 rounded-md overflow-hidden flex" style={{ background: 'var(--violet-4)' }}>
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
            <div className="p-3 overflow-y-auto max-h-[500px]">
              <div className="flex flex-wrap gap-1.5">
                {sorted.slice(0, 300).map((a: TuneAsset) => {
                  const isPassing = a.ad_pass === true;
                  const isFailing = a.ad_pass === false;
                  const isSelected = selectedSymbol === a.symbol;
                  return (
                    <button key={a.symbol}
                      onClick={() => setSelectedSymbol(a.symbol)}
                      title={`${a.symbol} - ${formatModelName(a.best_model)} - PIT: ${a.ad_pass == null ? 'Unknown' : a.ad_pass ? 'Pass' : 'Fail'}${a.bic ? ` - BIC: ${a.bic.toFixed(0)}` : ''}`}
                      className="transition-all duration-150"
                      style={{
                        width: 32, height: 32, borderRadius: 6,
                        fontSize: 7, fontWeight: 600, lineHeight: '32px', textAlign: 'center',
                        color: isPassing ? 'var(--accent-emerald)' : isFailing ? 'var(--accent-rose)' : '#7a8ba4',
                        background: isPassing ? 'var(--emerald-15)' : isFailing ? 'rgba(255,107,138,0.2)' : 'rgba(100,116,139,0.08)',
                        boxShadow: isSelected ? '0 0 0 2px var(--accent-violet), 0 0 8px var(--violet-30)' : isPassing ? '0 0 4px var(--emerald-30)' : isFailing ? '0 0 6px var(--rose-30)' : 'none',
                        transform: isSelected ? 'scale(1.15)' : undefined,
                      }}
                    >
                      {a.symbol.slice(0, 3)}
                    </button>
                  );
                })}
              </div>
            </div>
          ) : (
            <div className="overflow-y-auto max-h-[500px]">
              <table className="premium-table w-full text-xs">
                <thead className="sticky top-0" style={{ background: 'rgba(10,10,26,0.95)', zIndex: 10 }}>
                  <tr style={{ borderBottom: '1px solid var(--violet-8)' }}>
                    <SortHeader label="Symbol" sortKey="symbol" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="left" />
                    <SortHeader label="Best Model" sortKey="best_model" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="left" />
                    <SortHeader label="BIC" sortKey="bic" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                    <SortHeader label="Models" sortKey="num_models" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="center" />
                    <SortHeader label="PIT" sortKey="ad_pass" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="center" />
                    <SortHeader label="phi" sortKey="phi" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                    <SortHeader label="nu" sortKey="nu" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                    <SortHeader label="Top W%" sortKey="top_weight" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                    <SortHeader label="Obs" sortKey="n_obs" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                  </tr>
                </thead>
                <tbody>
                  {sorted.slice(0, 300).map((a: TuneAsset) => (
                    <tr key={a.symbol}
                      onClick={() => setSelectedSymbol(a.symbol)}
                      className="cursor-pointer transition-colors duration-150 hover:bg-[rgba(139,92,246,0.06)]"
                      style={{
                        borderBottom: '1px solid var(--violet-4)',
                        background: selectedSymbol === a.symbol ? 'var(--violet-6)' : undefined,
                      }}
                    >
                      <td className="px-3 py-2 font-semibold" style={{ color: 'var(--text-luminous)' }}>{a.symbol}</td>
                      <td className="px-3 py-2" style={{ color: 'var(--text-secondary)' }}>{formatModelNameShort(a.best_model)}</td>
                      <td className="px-3 py-2 text-right font-mono" style={{ color: a.bic ? '#b49aff' : 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                        {a.bic ? a.bic.toFixed(0) : '--'}
                      </td>
                      <td className="px-3 py-2 text-center" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>{a.num_models}</td>
                      <td className="px-3 py-2 text-center"><PitBadge asset={a} /></td>
                      <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                        {a.phi != null ? a.phi.toFixed(4) : '--'}
                      </td>
                      <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                        {a.nu != null ? a.nu.toFixed(1) : '--'}
                      </td>
                      <td className="px-3 py-2 text-right" style={{ fontVariantNumeric: 'tabular-nums' }}>
                        <span className="font-mono text-[10px]" style={{ color: (a.top_weight ?? 0) > 0.5 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                          {a.top_weight != null ? `${(a.top_weight * 100).toFixed(1)}%` : '--'}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                        {a.n_obs ?? '--'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      </div>

      {/* ── Detail Panel ──────────────────────────────────────────── */}
      {selectedSymbol && detailQ.data?.data && (
        <DetailPanel symbol={selectedSymbol} data={detailQ.data.data} onViewDiagnostics={() => navigate('/diagnostics')} />
      )}
    </>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Summary Card
   ══════════════════════════════════════════════════════════════════ */

function SummaryCard({ icon, label, value, sub, color }: { icon: React.ReactNode; label: string; value: number; sub?: string; color: string }) {
  return (
    <div className="glass-card p-4 flex items-center gap-3" style={{
      borderLeft: `3px solid ${color}`,
      background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
    }}>
      <div className="p-2 rounded-lg" style={{ background: `${color}15`, color }}>{icon}</div>
      <div>
        <div className="text-xl font-bold font-mono" style={{ color, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label} {sub && <span style={{ color }}>{sub}</span>}</div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Sort Header
   ══════════════════════════════════════════════════════════════════ */

function SortHeader({ label, sortKey: sk, currentKey, dir, onSort, align }: {
  label: string; sortKey: SortKey; currentKey: SortKey; dir: SortDir;
  onSort: (k: SortKey) => void; align: 'left' | 'center' | 'right';
}) {
  const active = currentKey === sk && dir != null;
  return (
    <th className={`px-3 py-2 text-${align} cursor-pointer select-none transition-colors hover:text-[#b49aff]`}
      style={{ color: active ? '#b49aff' : 'var(--text-muted)' }}
      onClick={() => onSort(sk)}
    >
      <span className="inline-flex items-center gap-1">
        {label}
        {active && dir === 'asc' && <ChevronUp className="w-3 h-3" />}
        {active && dir === 'desc' && <ChevronDown className="w-3 h-3" />}
      </span>
    </th>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Status Badge
   ══════════════════════════════════════════════════════════════════ */

function StatusBadge({ status }: { status: RetuneStatus }) {
  const config = {
    idle: { label: 'Ready', bg: 'rgba(100,116,139,0.12)', color: '#94a3b8', dot: false },
    running: { label: 'Running', bg: 'var(--amber-12)', color: 'var(--accent-amber)', dot: true },
    completed: { label: 'Completed', bg: 'var(--emerald-12)', color: 'var(--accent-emerald)', dot: false },
    failed: { label: 'Failed', bg: 'var(--rose-12)', color: 'var(--accent-rose)', dot: false },
  }[status] ?? { label: status, bg: 'rgba(100,116,139,0.12)', color: '#94a3b8', dot: false };

  return (
    <span className="flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium" style={{ background: config.bg, color: config.color }}>
      {config.dot && <span className="w-1.5 h-1.5 rounded-full animate-pulse" style={{ background: config.color }} />}
      {config.label}
    </span>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Retune Panel
   ══════════════════════════════════════════════════════════════════ */

function RetunePanel({ status, logs, logEndRef, onClose, elapsed, retune }: {
  status: RetuneStatus; logs: RetuneLogEntry[];
  logEndRef: React.RefObject<HTMLDivElement | null>; onClose: () => void;
  elapsed: number | null;
  retune: ReturnType<typeof getRetuneSnapshot>;
}) {
  const [autoScroll, setAutoScroll] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const progressCount = logs.filter(l => l.type === 'progress').length;

  const handleCopyLog = useCallback(() => {
    const text = logs.map(l => l.message).join('\n');
    navigator.clipboard.writeText(text);
  }, [logs]);

  const handleScroll = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    setAutoScroll(el.scrollHeight - el.scrollTop - el.clientHeight < 30);
  }, []);

  useEffect(() => {
    if (autoScroll) logEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [logs, autoScroll, logEndRef]);

  return (
    <div className="glass-card mb-6 overflow-hidden">
      <div className="px-4 py-2.5 flex items-center justify-between" style={{ borderBottom: '1px solid var(--violet-8)' }}>
        <div className="flex items-center gap-3">
          <Terminal className="w-4 h-4" style={{ color: status === 'running' ? '#b49aff' : status === 'completed' ? 'var(--accent-emerald)' : status === 'failed' ? 'var(--accent-rose)' : '#7a8ba4' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>Console</span>
          {status === 'running' && (
            <div className="flex items-center gap-2 text-xs">
              <RefreshCw className="w-3 h-3 animate-spin" style={{ color: '#b49aff' }} />
              <span style={{ color: '#b49aff' }}>{progressCount} processed</span>
              {retune.currentAsset && <span className="font-mono font-semibold" style={{ color: '#b49aff' }}>{retune.currentAsset}</span>}
            </div>
          )}
          {status === 'completed' && <span className="text-xs" style={{ color: 'var(--accent-emerald)' }}>Done &middot; {retune.successCount} ok, {retune.failCount} fail{elapsed ? ` in ${formatElapsed(elapsed)}` : ''}</span>}
          {status === 'failed' && <span className="text-xs" style={{ color: 'var(--accent-rose)' }}>Failed{elapsed ? ` after ${formatElapsed(elapsed)}` : ''}</span>}
        </div>
        <div className="flex items-center gap-2">
          <button onClick={handleCopyLog} className="p-1 rounded transition hover:bg-[var(--violet-8)]" title="Copy log" style={{ color: '#7a8ba4' }}>
            <Copy className="w-3.5 h-3.5" />
          </button>
          <button onClick={onClose} className="p-1 rounded transition hover:bg-[var(--violet-8)]" style={{ color: '#7a8ba4' }}>
            <XCircle className="w-3.5 h-3.5" />
          </button>
        </div>
      </div>

      <div ref={scrollRef} onScroll={handleScroll}
        className="p-3 overflow-y-auto max-h-[280px] font-mono text-[11px]"
        style={{ background: 'rgba(10,10,26,0.8)' }}>
        {logs.map((entry, i) => (
          <div key={i} className={`py-0.5 ${logColor(entry.type)}`}
            style={{ borderLeft: entry.type === 'progress' ? '2px solid var(--accent-emerald)' : entry.type === 'error' || entry.type === 'failed' ? '2px solid var(--accent-rose)' : entry.type === 'phase' ? '2px solid var(--accent-cyan)' : '2px solid transparent', paddingLeft: 8 }}>
            {entry.message}
          </div>
        ))}
        <div ref={logEndRef} />
        {!autoScroll && (
          <button onClick={() => { setAutoScroll(true); logEndRef.current?.scrollIntoView({ behavior: 'smooth' }); }}
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
   Model Distribution Chart
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
    <div className="glass-card p-5 lg:col-span-1">
      <div className="flex items-center justify-between mb-3">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Model Distribution</h3>
        <div className="flex gap-0.5">
          {(['treemap', 'bar'] as const).map(m => (
            <button key={m} onClick={() => setMode(m)}
              className="px-2 py-0.5 rounded text-[10px] font-medium transition-all"
              style={{ background: mode === m ? 'var(--violet-12)' : 'transparent', color: mode === m ? '#b49aff' : '#7a8ba4' }}>
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
            const w = Math.max(44, frac * 280);
            const h = Math.max(32, frac * 200);
            return (
              <div key={d.name}
                onMouseEnter={() => setHovered(d.name)}
                onMouseLeave={() => setHovered(null)}
                className="rounded-md flex flex-col items-center justify-center transition-all duration-200 cursor-default"
                style={{
                  width: w, height: h, background: bg,
                  border: '1px solid var(--violet-10)',
                  opacity: isDimmed ? 0.4 : 1,
                  boxShadow: isHovered ? '0 0 12px var(--violet-25)' : undefined,
                  transform: isHovered ? 'scale(1.03)' : undefined,
                }}
              >
                {w > 50 && (
                  <span className="text-[8px] font-medium truncate px-1" style={{ color: 'rgba(255,255,255,0.8)', maxWidth: w - 8 }}>
                    {d.name}
                  </span>
                )}
                <span className="text-[10px] font-bold" style={{ color: 'rgba(255,255,255,0.95)', fontFamily: 'monospace' }}>{d.count}</span>
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
            <Tooltip contentStyle={{ background: 'rgba(15,15,35,0.95)', border: '1px solid var(--violet-15)', borderRadius: 8, color: '#e2e8f0' }} />
            <Bar dataKey="count" fill="var(--accent-violet)" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      )}
    </div>
  );
});

/* ══════════════════════════════════════════════════════════════════
   Detail Panel
   ══════════════════════════════════════════════════════════════════ */

function DetailPanel({ symbol, data, onViewDiagnostics }: {
  symbol: string; data: Record<string, unknown>; onViewDiagnostics: () => void;
}) {
  const g = (data.global ?? data) as Record<string, unknown>;
  const best = g.best_model as string | undefined;
  const regime = data.regime as Record<string, unknown> | string | undefined;
  const pitPass = g.ad_pass as boolean | undefined;
  const modelWeights = g.model_weights as Record<string, number> | undefined;
  const modelComparison = g.model_comparison as Record<string, { ll?: number; bic?: number; aic?: number; fit_success?: boolean }> | undefined;
  const regimeCounts = data.regime_counts as Record<string, number> | undefined;
  const diag = data.diagnostics as Record<string, unknown> | undefined;

  // Build competing models from model_weights + model_comparison
  const models = useMemo(() => {
    if (!modelWeights) return [];
    return Object.entries(modelWeights)
      .map(([name, weight]) => {
        const mc = modelComparison?.[name];
        return { name, weight, bic: mc?.bic, ll: mc?.ll, aic: mc?.aic };
      })
      .sort((a, b) => b.weight - a.weight);
  }, [modelWeights, modelComparison]);

  const bic = g.bic as number | undefined;
  const phi = g.phi as number | undefined;
  const nu = g.nu as number | undefined;
  const q = g.q as number | undefined;
  const c = g.c as number | undefined;
  const nObs = g.n_obs as number | undefined;
  const ksP = g.pit_ks_pvalue as number | undefined;
  const ksStat = g.ks_statistic as number | undefined;

  const params = [
    { label: 'BIC', value: bic?.toFixed(0), color: '#b49aff' },
    { label: 'phi', value: phi?.toFixed(4), color: 'var(--accent-cyan)' },
    { label: 'nu', value: nu?.toFixed(1), color: 'var(--accent-amber)' },
    { label: 'q', value: q != null ? q.toExponential(2) : undefined, color: '#94a3b8' },
    { label: 'c', value: c?.toFixed(4), color: '#94a3b8' },
    { label: 'N obs', value: nObs?.toString(), color: '#94a3b8' },
    { label: 'KS stat', value: ksStat?.toFixed(4), color: ksP != null ? (ksP >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)') : '#94a3b8' },
    { label: 'KS p-val', value: ksP?.toFixed(4), color: ksP != null ? (ksP >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)') : '#94a3b8' },
  ].filter(p => p.value != null);

  return (
    <div className="glass-card p-5 mt-6 fade-up" style={{
      background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
    }}>
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <h3 className="text-xl font-bold" style={{
            background: 'linear-gradient(135deg, #b49aff, #818cf8)',
            WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          }}>{symbol}</h3>
          {pitPass != null && (
            <span className="px-2 py-0.5 rounded-full text-[10px] font-medium"
              style={{ background: pitPass ? 'var(--emerald-12)' : 'var(--rose-12)', color: pitPass ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
              PIT {pitPass ? 'Pass' : 'Fail'}
            </span>
          )}
          {regimeCounts && (
            <div className="flex gap-1">
              {Object.entries(regimeCounts).map(([r, cnt]) => (
                <span key={r} className="px-1.5 py-0.5 rounded text-[9px] font-mono" style={{ background: 'var(--violet-6)', color: '#94a3b8' }}>
                  R{r}:{cnt}
                </span>
              ))}
            </div>
          )}
        </div>
        <button onClick={onViewDiagnostics}
          className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all hover:scale-[1.02]"
          style={{ color: '#b49aff', background: 'var(--violet-8)', border: '1px solid var(--violet-12)' }}
        >
          Diagnostics <ArrowRight className="w-3 h-3" />
        </button>
      </div>

      {/* Key parameters */}
      {best && (
        <div className="mb-4">
          <span className="text-[10px] uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>Best Model</span>
          <div className="text-sm font-semibold mt-0.5" style={{ color: '#b49aff' }}>{formatModelName(best)}</div>
        </div>
      )}

      {params.length > 0 && (
        <div className="grid grid-cols-4 md:grid-cols-8 gap-2 mb-4">
          {params.map(p => (
            <div key={p.label} className="p-2 rounded-lg text-center" style={{ background: 'var(--violet-4)' }}>
              <div className="text-[9px] uppercase tracking-wider mb-0.5" style={{ color: 'var(--text-muted)' }}>{p.label}</div>
              <div className="text-sm font-mono font-medium" style={{ color: p.color, fontVariantNumeric: 'tabular-nums' }}>{p.value}</div>
            </div>
          ))}
        </div>
      )}

      {/* BMA Weights Table */}
      {models.length > 0 && (
        <div className="overflow-x-auto">
          <div className="text-[10px] uppercase tracking-wider mb-2" style={{ color: 'var(--text-muted)' }}>BMA Model Weights ({models.length} competing)</div>
          <table className="w-full text-xs">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--violet-8)' }}>
                <th className="text-left px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Model</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>BMA Weight</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>BIC</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Log-Lik</th>
                <th className="px-2 py-1.5" style={{ color: 'var(--text-muted)', width: 120 }}>Weight</th>
              </tr>
            </thead>
            <tbody>
              {models.slice(0, 20).map((m) => {
                const isWinner = m.name === best;
                return (
                  <tr key={m.name} style={{
                    borderBottom: '1px solid var(--violet-4)',
                    background: isWinner ? 'var(--emerald-6)' : undefined,
                    borderLeft: isWinner ? '2px solid var(--accent-emerald)' : '2px solid transparent',
                  }}>
                    <td className="px-2 py-1.5 font-medium" style={{ color: isWinner ? 'var(--accent-emerald)' : 'var(--text-primary)' }}>
                      {formatModelName(m.name)} {isWinner && <Award className="w-3 h-3 inline ml-1" style={{ color: 'var(--accent-emerald)' }} />}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{ color: m.weight > 0.1 ? '#b49aff' : 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                      {(m.weight * 100).toFixed(2)}%
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                      {m.bic?.toFixed(0) ?? '--'}
                    </td>
                    <td className="px-2 py-1.5 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                      {m.ll?.toFixed(0) ?? '--'}
                    </td>
                    <td className="px-2 py-1.5">
                      <div className="h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
                        <div className="h-full rounded-full" style={{
                          width: `${Math.min(m.weight * 100, 100)}%`,
                          background: isWinner ? 'linear-gradient(90deg, var(--accent-emerald), #6ff0c0)' : 'linear-gradient(90deg, var(--accent-violet), #818cf8)',
                        }} />
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Fallback raw JSON if no structured data */}
      {models.length === 0 && (
        <div>
          <h4 className="text-xs font-medium mb-2" style={{ color: 'var(--text-secondary)' }}>Raw Parameters</h4>
          <pre className="text-xs overflow-x-auto max-h-[300px] whitespace-pre-wrap p-3 rounded-xl font-mono"
            style={{ background: 'rgba(10,10,26,0.6)', color: 'var(--text-secondary)', border: '1px solid var(--violet-6)' }}>
            {JSON.stringify(data, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   PIT Badge
   ══════════════════════════════════════════════════════════════════ */

function PitBadge({ asset }: { asset: TuneAsset }) {
  if (asset.ad_pass === true)
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium"
        style={{ background: 'var(--emerald-12)', color: 'var(--accent-emerald)' }}>
        <CheckCircle className="w-2.5 h-2.5" /> Pass
      </span>
    );
  if (asset.ad_pass === false)
    return (
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 rounded text-[9px] font-medium"
        style={{ background: 'var(--rose-12)', color: 'var(--accent-rose)' }}>
        <XCircle className="w-2.5 h-2.5" /> Fail
      </span>
    );
  return <span className="inline-block w-2 h-2 rounded-full" style={{ background: '#7a8ba4' }} />;
}
