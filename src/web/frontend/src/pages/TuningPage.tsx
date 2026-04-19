import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useRef, useEffect, useCallback, useSyncExternalStore, useMemo, memo } from 'react';
import { api } from '../api';
import type { TuneAsset, ModelAnalytics } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { TuningSkeleton } from '../components/CosmicSkeleton';
import {
  CheckCircle, XCircle, AlertCircle, RefreshCw, Play, Square, Terminal,
  Copy, ArrowDown, ArrowRight,
  ChevronDown, ChevronUp, Search, Filter, Layers, Target, Timer, Award,
  PieChart, Eye, X,
} from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import { formatModelName, formatModelNameShort } from '../utils/modelNames';
import {
  type RetuneStatus, type RetuneLogEntry,
  getRetuneSnapshot, subscribeRetune,
  setRetuneMode, setShowPanel,
  startRetune as storeStartRetune,
  stopRetune as storeStopRetune,
} from '../stores/retuneStore';

type TuneMode = 'retune' | 'tune' | 'calibrate';
type SortKey = 'symbol' | 'best_model' | 'num_models' | 'ad_pass' | 'bic' | 'phi' | 'nu' | 'n_obs' | 'top_weight' | 'ks_pvalue' | 'file_size_kb';
type SortDir = 'asc' | 'desc' | null;
type CardFilter = 'all' | 'pass' | 'fail' | 'unknown' | 'models';

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

/* ══════════════════════════════════════════════════════════════════
   Model family helpers
   ══════════════════════════════════════════════════════════════════ */

const MODEL_COLORS: Record<string, { bg: string; bar: string; text: string }> = {
  kalman:   { bg: 'rgba(59,130,246,0.12)', bar: '#3b82f6', text: '#60a5fa' },
  phi:      { bg: 'rgba(16,185,129,0.12)', bar: '#10b981', text: '#6ee7b7' },
  momentum: { bg: 'rgba(245,158,11,0.12)', bar: '#f59e0b', text: '#fbbf24' },
  default:  { bg: 'rgba(139,92,246,0.12)', bar: '#8b5cf6', text: '#a78bfa' },
};

function modelFamily(name: string): string {
  const lower = name.toLowerCase();
  if (lower.includes('momentum')) return 'momentum';
  if (lower.includes('kalman') && !lower.includes('phi')) return 'kalman';
  if (lower.includes('phi') || lower.includes('student')) return 'phi';
  return 'default';
}

function familyLabel(f: string): string {
  return f === 'kalman' ? 'Kalman' : f === 'phi' ? 'Phi/Student-t' : f === 'momentum' ? 'Momentum' : 'Other';
}

/* ══════════════════════════════════════════════════════════════════
   Main Component
   ══════════════════════════════════════════════════════════════════ */

export default function TuningPage() {
  const [search, setSearch] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [cardFilter, setCardFilter] = useState<CardFilter>('all');
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

  const handleCardClick = useCallback((filter: CardFilter) => {
    setCardFilter(prev => prev === filter ? 'all' : filter);
  }, []);

  if (listQ.isLoading) return <TuningSkeleton />;

  const assets = listQ.data?.assets || [];
  const stats = statsQ.data;

  // Apply card filter + search
  const filtered = assets.filter((a: TuneAsset) => {
    const matchSearch = !search || a.symbol.toLowerCase().includes(search.toLowerCase()) || a.best_model.toLowerCase().includes(search.toLowerCase());
    if (!matchSearch) return false;
    switch (cardFilter) {
      case 'pass': return a.ad_pass === true;
      case 'fail': return a.ad_pass === false;
      case 'unknown': return a.ad_pass == null;
      default: return true;
    }
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
      case 'ks_pvalue': av = a.ks_pvalue; bv = b.ks_pvalue; break;
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
        .map(([name, count]) => ({
          name: formatModelNameShort(name),
          fullName: name,
          count: count as number,
          analytics: stats.models_analytics?.[name] ?? null,
        }))
        .sort((a, b) => b.count - a.count)
    : [];

  const modeLabels: { id: TuneMode; label: string; desc: string }[] = [
    { id: 'retune', label: 'Full Retune', desc: 'Re-estimate all models' },
    { id: 'tune', label: 'Tune Only', desc: 'Fit new assets only' },
    { id: 'calibrate', label: 'Calibrate', desc: 'Fix failing PIT only' },
  ];

  const passRate = stats ? Math.round((stats.pit_pass / Math.max(stats.total, 1)) * 100) : 0;

  // Compute aggregate stats for summary cards
  const avgBic = assets.length > 0 ? assets.reduce((s: number, a: TuneAsset) => s + (a.bic ?? 0), 0) / assets.filter((a: TuneAsset) => a.bic != null).length : 0;
  const avgPhi = assets.length > 0 ? assets.reduce((s: number, a: TuneAsset) => s + (a.phi ?? 0), 0) / assets.filter((a: TuneAsset) => a.phi != null).length : 0;

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

      {/* ── Clickable Summary Cards ──────────────────────────────── */}
      {stats && (
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 mb-6 fade-up">
          <SummaryCard
            icon={<Layers className="w-4 h-4" />}
            label="Total Tuned" value={stats.total}
            sub={`Avg BIC: ${avgBic.toFixed(0)}`}
            color="#b49aff"
            active={cardFilter === 'all'}
            onClick={() => handleCardClick('all')}
          />
          <SummaryCard
            icon={<CheckCircle className="w-4 h-4" />}
            label="PIT Pass" value={stats.pit_pass}
            sub={`${passRate}% — KS p≥0.05`}
            color="var(--accent-emerald)"
            active={cardFilter === 'pass'}
            onClick={() => handleCardClick('pass')}
          />
          <SummaryCard
            icon={<XCircle className="w-4 h-4" />}
            label="PIT Fail" value={stats.pit_fail}
            sub="KS p<0.05"
            color="var(--accent-rose)"
            active={cardFilter === 'fail'}
            onClick={() => handleCardClick('fail')}
          />
          <SummaryCard
            icon={<AlertCircle className="w-4 h-4" />}
            label="Unknown" value={stats.pit_unknown}
            sub="No KS data"
            color="var(--accent-amber)"
            active={cardFilter === 'unknown'}
            onClick={() => handleCardClick('unknown')}
          />
          <SummaryCard
            icon={<PieChart className="w-4 h-4" />}
            label="Model Types" value={modelData.length}
            sub={`Avg phi: ${avgPhi.toFixed(4)}`}
            color="var(--accent-cyan)"
            active={cardFilter === 'models'}
            onClick={() => handleCardClick('models')}
          />
        </div>
      )}

      {/* ── Active Filter Indicator ────────────────────────────── */}
      {cardFilter !== 'all' && cardFilter !== 'models' && (
        <div className="mb-4 flex items-center gap-2 px-3 py-2 rounded-xl text-xs" style={{
          background: 'var(--violet-4)', border: '1px solid var(--violet-8)',
        }}>
          <Filter className="w-3.5 h-3.5" style={{ color: '#b49aff' }} />
          <span style={{ color: 'var(--text-secondary)' }}>
            Showing <strong style={{ color: '#b49aff' }}>{filtered.length}</strong> assets with PIT status: <strong style={{ color: '#b49aff' }}>{cardFilter}</strong>
          </span>
          <button onClick={() => setCardFilter('all')} className="ml-auto p-0.5 rounded hover:bg-[var(--violet-8)] transition" style={{ color: '#7a8ba4' }}>
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {/* ── Mission Control ───────────────────────────────────── */}
      <div className="glass-card p-5 mb-6 fade-up" style={{
        background: 'linear-gradient(135deg, var(--violet-4) 0%, rgba(99,102,241,0.03) 50%, var(--violet-4) 100%)',
      }}>
        <div className="flex items-center justify-between flex-wrap gap-4">
          <div className="flex items-center gap-3 flex-wrap">
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

      {/* ── Retune Log Panel ──────────────────────────────────── */}
      {showRetunePanel && (
        <RetunePanel status={retuneStatus} logs={retuneLogs} logEndRef={logEndRef}
          onClose={() => setShowPanel(false)} elapsed={elapsed} retune={retune} />
      )}

      {/* ── Model Distribution (when "Model Types" card is active) */}
      {(cardFilter === 'models' || cardFilter === 'all') && modelData.length > 0 && (
        <ModelDistributionChart data={modelData} expanded={cardFilter === 'models'} />
      )}

      {/* ── Asset Table ───────────────────────────────────────── */}
      <div className="glass-card overflow-hidden fade-up-delay-1">
        <div className="px-3 py-2.5 flex items-center gap-2" style={{ borderBottom: '1px solid var(--violet-8)' }}>
          <div className="relative flex-1 max-w-sm">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5" style={{ color: '#7a8ba4' }} />
            <input type="text" value={search} onChange={(e) => setSearch(e.target.value)}
              placeholder="Search symbol or model..."
              className="w-full pl-8 pr-3 py-2 rounded-lg text-sm outline-none transition-all focus-ring"
              style={{ background: 'rgba(10,10,26,0.6)', border: '1px solid var(--violet-8)', color: 'var(--text-primary)' }}
            />
          </div>
          <span className="text-[11px] font-mono px-2 py-1 rounded" style={{ color: '#b49aff', background: 'var(--violet-6)' }}>
            {sorted.length} / {assets.length}
          </span>
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

        <div className="overflow-y-auto max-h-[600px]">
          <table className="premium-table w-full text-xs">
            <thead className="sticky top-0" style={{ background: 'rgba(10,10,26,0.95)', zIndex: 10 }}>
              <tr style={{ borderBottom: '1px solid var(--violet-8)' }}>
                <SortHeader label="Symbol" sortKey="symbol" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="left" />
                <SortHeader label="Best Model" sortKey="best_model" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="left" />
                <SortHeader label="BIC" sortKey="bic" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortHeader label="PIT" sortKey="ad_pass" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="center" />
                <SortHeader label="KS p-val" sortKey="ks_pvalue" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortHeader label="Grade" sortKey="ad_pass" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="center" />
                <SortHeader label="phi" sortKey="phi" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortHeader label="nu" sortKey="nu" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortHeader label="Top W%" sortKey="top_weight" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
                <SortHeader label="Models" sortKey="num_models" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="center" />
                <SortHeader label="Obs" sortKey="n_obs" currentKey={sortKey} dir={sortDir} onSort={handleSort} align="right" />
              </tr>
            </thead>
            <tbody>
              {sorted.slice(0, 500).map((a: TuneAsset) => (
                <tr key={a.symbol}
                  onClick={() => setSelectedSymbol(a.symbol)}
                  className="cursor-pointer transition-colors duration-150 hover:bg-[rgba(139,92,246,0.06)]"
                  style={{
                    borderBottom: '1px solid var(--violet-4)',
                    background: selectedSymbol === a.symbol ? 'var(--violet-6)' : undefined,
                  }}
                >
                  <td className="px-3 py-2 font-semibold" style={{ color: 'var(--text-luminous)' }}>{a.symbol}</td>
                  <td className="px-3 py-2">
                    <span className="px-1.5 py-0.5 rounded text-[10px] font-medium" style={{
                      background: MODEL_COLORS[modelFamily(a.best_model)]?.bg ?? MODEL_COLORS.default.bg,
                      color: MODEL_COLORS[modelFamily(a.best_model)]?.text ?? MODEL_COLORS.default.text,
                    }}>
                      {formatModelNameShort(a.best_model)}
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right font-mono" style={{ color: a.bic ? '#b49aff' : 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    {a.bic ? a.bic.toFixed(0) : '--'}
                  </td>
                  <td className="px-3 py-2 text-center"><PitBadge asset={a} /></td>
                  <td className="px-3 py-2 text-right font-mono" style={{
                    color: a.ks_pvalue != null ? (a.ks_pvalue >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)') : 'var(--text-muted)',
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a.ks_pvalue != null ? a.ks_pvalue.toFixed(4) : '--'}
                  </td>
                  <td className="px-3 py-2 text-center">
                    <GradeBadge grade={a.pit_calibration_grade} />
                  </td>
                  <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {a.phi != null ? a.phi.toFixed(4) : '--'}
                  </td>
                  <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>
                    {a.nu != null ? a.nu.toFixed(1) : '--'}
                  </td>
                  <td className="px-3 py-2 text-right" style={{ fontVariantNumeric: 'tabular-nums' }}>
                    <TopWeightBar value={a.top_weight} />
                  </td>
                  <td className="px-3 py-2 text-center" style={{ color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums' }}>{a.num_models}</td>
                  <td className="px-3 py-2 text-right font-mono" style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}>
                    {a.n_obs ?? '--'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* ── Detail Panel ──────────────────────────────────────── */}
      {selectedSymbol && detailQ.data?.data && (
        <DetailPanel symbol={selectedSymbol} data={detailQ.data.data} onViewDiagnostics={() => navigate('/diagnostics')} onClose={() => setSelectedSymbol(null)} />
      )}
    </>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Summary Card (Clickable)
   ══════════════════════════════════════════════════════════════════ */

function SummaryCard({ icon, label, value, sub, color, active, onClick }: {
  icon: React.ReactNode; label: string; value: number; sub?: string; color: string;
  active?: boolean; onClick?: () => void;
}) {
  return (
    <button onClick={onClick} className="glass-card p-4 flex items-center gap-3 text-left transition-all duration-200 hover:scale-[1.02] w-full" style={{
      borderLeft: `3px solid ${color}`,
      background: active
        ? `linear-gradient(135deg, ${color}15, ${color}08)`
        : 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
      boxShadow: active ? `0 0 12px ${color}25, inset 0 0 20px ${color}08` : undefined,
      outline: active ? `1px solid ${color}40` : undefined,
    }}>
      <div className="p-2 rounded-lg" style={{ background: `${color}15`, color }}>{icon}</div>
      <div>
        <div className="text-xl font-bold font-mono" style={{ color, fontVariantNumeric: 'tabular-nums' }}>{value}</div>
        <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{label}</div>
        {sub && <div className="text-[9px] mt-0.5" style={{ color: `${color}99` }}>{sub}</div>}
      </div>
    </button>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Top Weight Bar
   ══════════════════════════════════════════════════════════════════ */

function TopWeightBar({ value }: { value: number | null }) {
  if (value == null) return <span className="text-[10px] font-mono" style={{ color: 'var(--text-muted)' }}>--</span>;
  const pct = value * 100;
  const color = pct > 50 ? 'var(--accent-emerald)' : pct > 25 ? 'var(--accent-amber)' : 'var(--text-secondary)';
  return (
    <div className="flex items-center gap-1.5 justify-end">
      <span className="font-mono text-[10px]" style={{ color }}>{pct.toFixed(1)}%</span>
      <div className="w-10 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
        <div className="h-full rounded-full" style={{ width: `${Math.min(pct, 100)}%`, background: color }} />
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Grade Badge
   ══════════════════════════════════════════════════════════════════ */

function GradeBadge({ grade }: { grade: string | null }) {
  if (!grade) return <span className="text-[9px]" style={{ color: '#7a8ba4' }}>--</span>;
  const colors: Record<string, { bg: string; text: string }> = {
    'A': { bg: 'var(--emerald-12)', text: 'var(--accent-emerald)' },
    'B': { bg: 'rgba(59,130,246,0.12)', text: '#60a5fa' },
    'C': { bg: 'var(--amber-12)', text: 'var(--accent-amber)' },
    'D': { bg: 'var(--rose-12)', text: 'var(--accent-rose)' },
  };
  const c = colors[grade] ?? { bg: 'var(--violet-6)', text: '#94a3b8' };
  return (
    <span className="px-1.5 py-0.5 rounded text-[9px] font-bold" style={{ background: c.bg, color: c.text }}>
      {grade}
    </span>
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
   Model Distribution — Rich Analytics Dashboard
   ══════════════════════════════════════════════════════════════════ */

const ModelDistributionChart = memo(function ModelDistributionChart({ data, expanded }: {
  data: { name: string; fullName?: string; count: number; analytics: ModelAnalytics | null }[];
  expanded?: boolean;
}) {
  const [hoveredModel, setHoveredModel] = useState<string | null>(null);
  const total = useMemo(() => data.reduce((s, d) => s + d.count, 0), [data]);

  // Group by family
  const families = useMemo(() => {
    const fam: Record<string, { count: number; models: typeof data; pitPass: number; pitFail: number; bics: number[] }> = {};
    data.forEach(d => {
      const f = modelFamily(d.fullName ?? d.name);
      if (!fam[f]) fam[f] = { count: 0, models: [], pitPass: 0, pitFail: 0, bics: [] };
      fam[f].count += d.count;
      fam[f].models.push(d);
      if (d.analytics) {
        fam[f].pitPass += d.analytics.pit_pass;
        fam[f].pitFail += d.analytics.pit_fail;
        if (d.analytics.avg_bic != null) fam[f].bics.push(d.analytics.avg_bic);
      }
    });
    return Object.entries(fam).sort((a, b) => b[1].count - a[1].count);
  }, [data]);

  // Best BIC model for highlighting
  const bestBicModel = useMemo(() => {
    let best: string | null = null;
    let bestVal = Infinity;
    data.forEach(d => {
      if (d.analytics?.avg_bic != null && d.analytics.avg_bic < bestVal) {
        bestVal = d.analytics.avg_bic;
        best = d.fullName ?? d.name;
      }
    });
    return best;
  }, [data]);

  const hoveredAnalytics = useMemo(() => {
    if (!hoveredModel) return null;
    return data.find(d => (d.fullName ?? d.name) === hoveredModel)?.analytics ?? null;
  }, [data, hoveredModel]);

  if (!expanded) {
    // Compact view — stacked bar + family legend with key metrics
    return (
      <div className="glass-card p-4 mb-6 fade-up" style={{
        background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
      }}>
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-3">
            <PieChart className="w-4 h-4" style={{ color: 'var(--accent-cyan)' }} />
            <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Model Distribution</span>
            <span className="text-[10px] font-mono" style={{ color: '#7a8ba4' }}>{data.length} types &middot; {total} assets</span>
          </div>
          <div className="flex items-center gap-3 text-[10px]">
            {families.map(([f, { count, pitPass, pitFail }]) => {
              const colors = MODEL_COLORS[f] ?? MODEL_COLORS.default;
              const pitTotal = pitPass + pitFail;
              const pitRate = pitTotal > 0 ? Math.round((pitPass / pitTotal) * 100) : null;
              return (
                <div key={f} className="flex items-center gap-1.5">
                  <div className="w-2 h-2 rounded-sm" style={{ background: colors.bar }} />
                  <span style={{ color: colors.text }}>{familyLabel(f)}</span>
                  <span className="font-mono" style={{ color: '#7a8ba4' }}>{count}</span>
                  {pitRate != null && (
                    <span className="font-mono" style={{ color: pitRate >= 90 ? 'var(--accent-emerald)' : pitRate >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)' }}>
                      {pitRate}%
                    </span>
                  )}
                </div>
              );
            })}
          </div>
        </div>
        <div className="flex gap-0.5 h-6 rounded-lg overflow-hidden">
          {data.map((d) => {
            const frac = d.count / (total || 1);
            const colors = MODEL_COLORS[modelFamily(d.fullName ?? d.name)] ?? MODEL_COLORS.default;
            const isBest = (d.fullName ?? d.name) === bestBicModel;
            return (
              <div key={d.name}
                title={`${d.name}: ${d.count} assets (${(frac * 100).toFixed(1)}%)${d.analytics?.avg_bic != null ? ` | Avg BIC: ${d.analytics.avg_bic.toFixed(0)}` : ''}${d.analytics?.pit_pass_rate != null ? ` | PIT: ${(d.analytics.pit_pass_rate * 100).toFixed(0)}%` : ''}`}
                className="transition-all duration-200 hover:brightness-125 cursor-default relative"
                style={{
                  flex: d.count,
                  background: isBest ? `linear-gradient(180deg, ${colors.bar}, ${colors.bar}cc)` : colors.bar,
                  minWidth: frac > 0.02 ? 2 : 0,
                  borderRight: '1px solid rgba(0,0,0,0.2)',
                  boxShadow: isBest ? `0 0 8px ${colors.bar}60` : undefined,
                }}
              />
            );
          })}
        </div>
      </div>
    );
  }

  // ── Expanded: Full analytics dashboard ──────────────────────────
  return (
    <div className="glass-card p-5 mb-6 fade-up" style={{
      background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
    }}>
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <PieChart className="w-4 h-4" style={{ color: 'var(--accent-cyan)' }} />
          <span className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>Model Analytics</span>
          <span className="text-[10px] font-mono" style={{ color: '#7a8ba4' }}>{data.length} models &middot; {total} assets</span>
        </div>
      </div>

      {/* Family summary cards with richer data */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-5">
        {families.map(([f, { count, models, pitPass, pitFail, bics }]) => {
          const colors = MODEL_COLORS[f] ?? MODEL_COLORS.default;
          const pct = ((count / total) * 100).toFixed(1);
          const pitTotal = pitPass + pitFail;
          const pitRate = pitTotal > 0 ? Math.round((pitPass / pitTotal) * 100) : null;
          const avgBic = bics.length > 0 ? bics.reduce((a, b) => a + b, 0) / bics.length : null;
          return (
            <div key={f} className="p-3 rounded-xl transition-all duration-200 hover:scale-[1.01]" style={{
              background: colors.bg, border: `1px solid ${colors.bar}30`,
            }}>
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-semibold" style={{ color: colors.text }}>{familyLabel(f)}</span>
                <span className="font-mono text-lg font-bold" style={{ color: colors.text }}>{count}</span>
              </div>
              <div className="h-1.5 rounded-full overflow-hidden mb-2" style={{ background: `${colors.bar}20` }}>
                <div className="h-full rounded-full transition-all duration-500" style={{ width: `${pct}%`, background: colors.bar }} />
              </div>
              <div className="grid grid-cols-2 gap-1 text-[9px]">
                <div style={{ color: `${colors.text}99` }}>{pct}% &middot; {models.length} variants</div>
                {pitRate != null && (
                  <div className="text-right font-mono" style={{
                    color: pitRate >= 90 ? 'var(--accent-emerald)' : pitRate >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)',
                  }}>
                    PIT {pitRate}%
                  </div>
                )}
                {avgBic != null && (
                  <div className="font-mono col-span-2" style={{ color: '#7a8ba4' }}>
                    Avg BIC: {avgBic.toFixed(0)}
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Full model analytics table */}
      <div className="overflow-x-auto rounded-xl" style={{ border: '1px solid var(--violet-6)' }}>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ background: 'rgba(10,10,26,0.6)', borderBottom: '1px solid var(--violet-8)' }}>
              <th className="text-left px-3 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Model</th>
              <th className="text-center px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Assets</th>
              <th className="text-right px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg BIC</th>
              <th className="text-right px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Best BIC</th>
              <th className="text-center px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>PIT Rate</th>
              <th className="text-right px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg phi</th>
              <th className="text-right px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg nu</th>
              <th className="text-right px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg KS p</th>
              <th className="text-right px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Avg Wt</th>
              <th className="px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)', minWidth: 100 }}>Share</th>
              <th className="text-left px-2 py-2.5 font-medium" style={{ color: 'var(--text-muted)' }}>Top Symbols</th>
            </tr>
          </thead>
          <tbody>
            {data.map((d) => {
              const frac = d.count / (total || 1);
              const colors = MODEL_COLORS[modelFamily(d.fullName ?? d.name)] ?? MODEL_COLORS.default;
              const a = d.analytics;
              const isBest = (d.fullName ?? d.name) === bestBicModel;
              const isHovered = (d.fullName ?? d.name) === hoveredModel;
              const pitTotal = a ? a.pit_pass + a.pit_fail : 0;
              const pitRate = pitTotal > 0 ? Math.round((a!.pit_pass / pitTotal) * 100) : null;

              return (
                <tr key={d.name}
                  onMouseEnter={() => setHoveredModel(d.fullName ?? d.name)}
                  onMouseLeave={() => setHoveredModel(null)}
                  className="transition-colors duration-100"
                  style={{
                    borderBottom: '1px solid var(--violet-4)',
                    background: isBest ? 'rgba(16,185,129,0.04)' : isHovered ? 'rgba(139,92,246,0.04)' : undefined,
                    borderLeft: isBest ? '2px solid var(--accent-emerald)' : '2px solid transparent',
                  }}
                >
                  {/* Model name */}
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-medium" style={{
                        background: colors.bg, color: colors.text,
                      }}>
                        {d.name}
                      </span>
                      {isBest && <Target className="w-3 h-3" style={{ color: 'var(--accent-emerald)' }} />}
                    </div>
                  </td>

                  {/* Asset count */}
                  <td className="px-2 py-2 text-center font-mono font-bold" style={{ color: colors.text }}>
                    {d.count}
                  </td>

                  {/* Avg BIC */}
                  <td className="px-2 py-2 text-right font-mono" style={{
                    color: a?.avg_bic != null ? '#b49aff' : 'var(--text-muted)',
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a?.avg_bic != null ? a.avg_bic.toFixed(0) : '--'}
                  </td>

                  {/* Best BIC */}
                  <td className="px-2 py-2 text-right font-mono" style={{
                    color: a?.best_bic != null ? 'var(--accent-cyan)' : 'var(--text-muted)',
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a?.best_bic != null ? a.best_bic.toFixed(0) : '--'}
                  </td>

                  {/* PIT Rate */}
                  <td className="px-2 py-2 text-center">
                    {pitRate != null ? (
                      <div className="flex items-center justify-center gap-1.5">
                        <div className="w-8 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
                          <div className="h-full rounded-full" style={{
                            width: `${pitRate}%`,
                            background: pitRate >= 90 ? 'var(--accent-emerald)' : pitRate >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)',
                          }} />
                        </div>
                        <span className="font-mono text-[10px]" style={{
                          color: pitRate >= 90 ? 'var(--accent-emerald)' : pitRate >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)',
                        }}>
                          {pitRate}%
                        </span>
                      </div>
                    ) : (
                      <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>--</span>
                    )}
                  </td>

                  {/* Avg phi */}
                  <td className="px-2 py-2 text-right font-mono" style={{
                    color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a?.avg_phi != null ? a.avg_phi.toFixed(4) : '--'}
                  </td>

                  {/* Avg nu */}
                  <td className="px-2 py-2 text-right font-mono" style={{
                    color: a?.avg_nu != null ? 'var(--accent-amber)' : 'var(--text-muted)',
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a?.avg_nu != null ? a.avg_nu.toFixed(1) : '--'}
                  </td>

                  {/* Avg KS p-value */}
                  <td className="px-2 py-2 text-right font-mono" style={{
                    color: a?.avg_ks_pvalue != null
                      ? (a.avg_ks_pvalue >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)')
                      : 'var(--text-muted)',
                    fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a?.avg_ks_pvalue != null ? a.avg_ks_pvalue.toFixed(3) : '--'}
                  </td>

                  {/* Avg top weight */}
                  <td className="px-2 py-2 text-right font-mono" style={{
                    color: 'var(--text-secondary)', fontVariantNumeric: 'tabular-nums',
                  }}>
                    {a?.avg_weight != null ? `${(a.avg_weight * 100).toFixed(1)}%` : '--'}
                  </td>

                  {/* Share bar */}
                  <td className="px-2 py-2">
                    <div className="flex items-center gap-1.5">
                      <div className="flex-1 h-2 rounded-sm overflow-hidden" style={{ background: 'var(--violet-4)' }}>
                        <div className="h-full rounded-sm transition-all duration-300" style={{
                          width: `${Math.max(frac * 100, 1)}%`,
                          background: `linear-gradient(90deg, ${colors.bar}, ${colors.bar}aa)`,
                        }} />
                      </div>
                      <span className="font-mono text-[9px] w-8 text-right" style={{ color: '#7a8ba4' }}>
                        {(frac * 100).toFixed(1)}%
                      </span>
                    </div>
                  </td>

                  {/* Top symbols */}
                  <td className="px-2 py-2">
                    <div className="flex gap-1 flex-wrap">
                      {(a?.top_symbols ?? []).slice(0, 3).map(s => (
                        <span key={s} className="px-1 py-0.5 rounded text-[8px] font-mono font-medium" style={{
                          background: 'var(--violet-4)', color: 'var(--text-secondary)',
                        }}>
                          {s}
                        </span>
                      ))}
                      {(a?.top_symbols?.length ?? 0) > 3 && (
                        <span className="text-[8px] font-mono" style={{ color: '#7a8ba4' }}>
                          +{(a?.top_symbols?.length ?? 0) - 3}
                        </span>
                      )}
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>

          {/* Summary footer */}
          <tfoot>
            <tr style={{ background: 'rgba(10,10,26,0.4)', borderTop: '1px solid var(--violet-8)' }}>
              <td className="px-3 py-2 text-[10px] font-semibold" style={{ color: 'var(--text-muted)' }}>TOTAL</td>
              <td className="px-2 py-2 text-center font-mono font-bold text-[10px]" style={{ color: '#b49aff' }}>{total}</td>
              <td colSpan={9} className="px-2 py-2">
                <div className="flex items-center gap-4 text-[9px]">
                  {families.map(([f, { count }]) => {
                    const colors = MODEL_COLORS[f] ?? MODEL_COLORS.default;
                    return (
                      <span key={f} style={{ color: colors.text }}>
                        {familyLabel(f)}: <strong>{count}</strong> ({((count / total) * 100).toFixed(0)}%)
                      </span>
                    );
                  })}
                </div>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Hover detail card */}
      {hoveredAnalytics && hoveredModel && (
        <div className="mt-4 p-4 rounded-xl transition-all duration-200" style={{
          background: 'rgba(10,10,26,0.5)', border: '1px solid var(--violet-8)',
        }}>
          <div className="flex items-center gap-2 mb-3">
            <Eye className="w-3.5 h-3.5" style={{ color: 'var(--accent-cyan)' }} />
            <span className="text-xs font-semibold" style={{ color: 'var(--text-primary)' }}>
              {formatModelName(hoveredModel)}
            </span>
            <span className="text-[10px] font-mono" style={{ color: '#7a8ba4' }}>{hoveredAnalytics.count} assets</span>
          </div>
          <div className="grid grid-cols-3 md:grid-cols-6 gap-3">
            {[
              { label: 'Avg BIC', value: hoveredAnalytics.avg_bic?.toFixed(0), color: '#b49aff' },
              { label: 'Best BIC', value: hoveredAnalytics.best_bic?.toFixed(0), color: 'var(--accent-cyan)' },
              { label: 'Worst BIC', value: hoveredAnalytics.worst_bic?.toFixed(0), color: 'var(--accent-rose)' },
              { label: 'Median KS p', value: hoveredAnalytics.median_ks_pvalue?.toFixed(4), color: hoveredAnalytics.median_ks_pvalue != null && hoveredAnalytics.median_ks_pvalue >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)' },
              { label: 'PIT Pass', value: hoveredAnalytics.pit_pass_rate != null ? `${(hoveredAnalytics.pit_pass_rate * 100).toFixed(0)}% (${hoveredAnalytics.pit_pass}/${hoveredAnalytics.pit_pass + hoveredAnalytics.pit_fail})` : null, color: hoveredAnalytics.pit_pass_rate != null && hoveredAnalytics.pit_pass_rate >= 0.9 ? 'var(--accent-emerald)' : 'var(--accent-amber)' },
              { label: 'Avg Obs', value: hoveredAnalytics.avg_n_obs?.toFixed(0), color: '#94a3b8' },
            ].map(p => (
              <div key={p.label} className="p-2 rounded-lg text-center" style={{ background: 'var(--violet-4)' }}>
                <div className="text-[8px] uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>{p.label}</div>
                <div className="text-[11px] font-mono font-medium" style={{ color: p.value ? p.color : 'var(--text-muted)' }}>
                  {p.value ?? '--'}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
});

/* ══════════════════════════════════════════════════════════════════
   Detail Panel
   ══════════════════════════════════════════════════════════════════ */

function DetailPanel({ symbol, data, onViewDiagnostics, onClose }: {
  symbol: string; data: Record<string, unknown>; onViewDiagnostics: () => void; onClose: () => void;
}) {
  const g = (data.global ?? data) as Record<string, unknown>;
  const best = g.best_model as string | undefined;
  const regime = data.regime as Record<string, unknown> | string | undefined;
  const modelWeights = g.model_weights as Record<string, number> | undefined;
  const modelComparison = g.model_comparison as Record<string, { ll?: number; bic?: number; aic?: number; fit_success?: boolean }> | undefined;
  const regimeCounts = data.regime_counts as Record<string, number> | undefined;

  // Build competing models
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
  const pitGrade = g.pit_calibration_grade as string | undefined;
  const pitPass = ksP != null ? ksP >= 0.05 : null;

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
          {pitGrade && <GradeBadge grade={pitGrade} />}
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
        <div className="flex items-center gap-2">
          <button onClick={onViewDiagnostics}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-xl text-xs font-medium transition-all hover:scale-[1.02]"
            style={{ color: '#b49aff', background: 'var(--violet-8)', border: '1px solid var(--violet-12)' }}
          >
            Diagnostics <ArrowRight className="w-3 h-3" />
          </button>
          <button onClick={onClose} className="p-1.5 rounded-lg transition hover:bg-[var(--violet-8)]" style={{ color: '#7a8ba4' }}>
            <X className="w-4 h-4" />
          </button>
        </div>
      </div>

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
