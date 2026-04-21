import { useQuery, useQueryClient } from '@tanstack/react-query';
import { useState, useRef, useEffect, useCallback, useSyncExternalStore, useMemo, memo } from 'react';
import { api } from '../api';
import type { TuneAsset, ModelAnalytics } from '../api';
import PageHeader from '../components/PageHeader';
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


/* ══════════════════════════════════════════════════════════════════
   Main Component
   ══════════════════════════════════════════════════════════════════ */

export default function TuningPage() {
  const [search, setSearch] = useState('');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [cardFilter, setCardFilter] = useState<CardFilter>('all');
  const [sortKey, setSortKey] = useState<SortKey>('symbol');
  const [sortDir, setSortDir] = useState<SortDir>('asc');
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
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-all duration-150 disabled:opacity-50 hover:brightness-110 active:scale-[0.98]"
            style={{
              background: 'linear-gradient(180deg, rgba(180,154,255,0.10), rgba(180,154,255,0.05))',
              color: '#c9b8ff',
              border: '1px solid var(--violet-15)',
              backdropFilter: 'blur(8px)',
              WebkitBackdropFilter: 'blur(8px)',
            }}
          >
            <RefreshCw className={`w-3 h-3 ${listQ.isFetching ? 'animate-spin' : ''}`} />
            <span style={{ letterSpacing: '0.01em' }}>Reload</span>
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
        <div className="mb-5 flex items-center gap-2">
          <span
            className="inline-flex items-center gap-2 pl-3 pr-1.5 py-1 rounded-full text-[11.5px]"
            style={{
              background: 'rgba(180,154,255,0.06)',
              border: '1px solid var(--violet-12)',
              color: 'var(--text-secondary)',
              backdropFilter: 'blur(8px)',
              WebkitBackdropFilter: 'blur(8px)',
            }}
          >
            <Filter className="w-3 h-3" style={{ color: '#c9b8ff' }} />
            <span>
              Filtered <span className="font-mono font-semibold" style={{ color: '#c9b8ff', fontVariantNumeric: 'tabular-nums' }}>{filtered.length}</span>
              <span style={{ color: 'var(--text-muted)' }}> · PIT </span>
              <span style={{ color: '#c9b8ff', textTransform: 'capitalize' }}>{cardFilter}</span>
            </span>
            <button
              onClick={() => setCardFilter('all')}
              className="inline-flex items-center justify-center w-5 h-5 rounded-full transition-all hover:brightness-125"
              style={{ background: 'var(--violet-10)', color: '#b49aff' }}
              aria-label="Clear filter"
            >
              <X className="w-3 h-3" />
            </button>
          </span>
        </div>
      )}

      {/* ── Mission Control ───────────────────────────────────── */}
      <MissionControl
        modes={modeLabels}
        retuneMode={retuneMode}
        retuneStatus={retuneStatus}
        retune={retune}
        elapsed={elapsed}
        totalAssetCount={assets.length}
        retuneLogs={retuneLogs}
        showRetunePanel={showRetunePanel}
        onSelectMode={setRetuneMode}
        onStart={handleStartRetune}
        onStop={handleStopRetune}
        onToggleLog={() => setShowPanel(!showRetunePanel)}
      />

      {/* ── Retune Log Panel ──────────────────────────────────── */}
      {showRetunePanel && (
        <RetunePanel status={retuneStatus} logs={retuneLogs}
          onClose={() => setShowPanel(false)} elapsed={elapsed} retune={retune} />
      )}

      {/* ── Model Distribution (when "Model Types" card is active) */}
      {(cardFilter === 'models' || cardFilter === 'all') && modelData.length > 0 && (
        <ModelDistributionChart data={modelData} expanded={cardFilter === 'models'} />
      )}

      {/* ── Asset Table ───────────────────────────────────────── */}
      <div className="glass-card overflow-hidden fade-up-delay-1" style={{ borderRadius: 18 }}>
        <div
          className="px-4 py-3 flex items-center gap-3"
          style={{
            borderBottom: '1px solid var(--violet-8)',
            background: 'linear-gradient(180deg, rgba(255,255,255,0.02), transparent)',
          }}
        >
          <div className="relative flex-1 max-w-md">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-3.5 h-3.5" style={{ color: '#7a8ba4' }} />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search symbols or models"
              className="w-full pl-9 pr-16 py-2 rounded-full text-[13px] outline-none transition-all focus-ring"
              style={{
                background: 'rgba(10,10,26,0.55)',
                border: '1px solid var(--violet-10)',
                color: 'var(--text-primary)',
                backdropFilter: 'blur(8px)',
                WebkitBackdropFilter: 'blur(8px)',
                letterSpacing: '0.005em',
              }}
            />
            <kbd
              className="absolute right-2 top-1/2 -translate-y-1/2 inline-flex items-center justify-center w-5 h-5 rounded-md text-[10px] font-mono pointer-events-none"
              style={{
                background: 'var(--violet-8)',
                color: '#7a8ba4',
                border: '1px solid var(--violet-10)',
                fontVariantNumeric: 'tabular-nums',
              }}
              title="Search"
            >
              /
            </kbd>
          </div>
          <div className="ml-auto flex items-center gap-2">
            <span
              className="text-[11px] font-mono px-2.5 py-1 rounded-full"
              style={{
                color: 'var(--text-secondary)',
                background: 'rgba(255,255,255,0.03)',
                border: '1px solid var(--violet-8)',
                fontVariantNumeric: 'tabular-nums',
                letterSpacing: '0.01em',
              }}
            >
              <span style={{ color: '#c9b8ff' }}>{sorted.length}</span>
              <span style={{ color: 'var(--text-muted)' }}> of {assets.length}</span>
            </span>
          </div>
        </div>

        {/* Health bar — restrained Apple-style */}
        {stats && (
          <div
            className="px-4 py-2.5 flex items-center gap-4"
            style={{ background: 'rgba(255,255,255,0.015)', borderBottom: '1px solid var(--violet-6)' }}
          >
            <div
              className="flex-1 h-[3px] rounded-full overflow-hidden flex"
              style={{ background: 'rgba(255,255,255,0.04)' }}
            >
              <div style={{ flex: stats.pit_pass, background: 'var(--accent-emerald)' }} />
              <div style={{ flex: stats.pit_fail, background: 'var(--accent-rose)' }} />
              <div style={{ flex: stats.pit_unknown, background: 'rgba(148,163,184,0.35)' }} />
            </div>
            <div className="flex items-center gap-3 text-[10.5px]" style={{ fontVariantNumeric: 'tabular-nums' }}>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-emerald)' }} />
                <span style={{ color: 'var(--accent-emerald)' }}>{stats.pit_pass}</span>
                <span style={{ color: 'var(--text-muted)' }}>pass</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-rose)' }} />
                <span style={{ color: 'var(--accent-rose)' }}>{stats.pit_fail}</span>
                <span style={{ color: 'var(--text-muted)' }}>fail</span>
              </span>
              <span className="inline-flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full" style={{ background: 'rgba(148,163,184,0.6)' }} />
                <span style={{ color: 'var(--text-secondary)' }}>{stats.pit_unknown}</span>
                <span style={{ color: 'var(--text-muted)' }}>unknown</span>
              </span>
            </div>
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
              {sorted.slice(0, 500).map((a: TuneAsset) => {
                const isSelected = selectedSymbol === a.symbol;
                return (
                <tr key={a.symbol}
                  onClick={() => setSelectedSymbol(a.symbol)}
                  className="cursor-pointer transition-colors duration-150 hover:bg-[rgba(180,154,255,0.04)]"
                  style={{
                    borderBottom: '1px solid rgba(255,255,255,0.04)',
                    background: isSelected ? 'rgba(180,154,255,0.07)' : undefined,
                    boxShadow: isSelected ? 'inset 2px 0 0 var(--accent-violet)' : undefined,
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
                );
              })}
              {sorted.length === 0 && (
                <tr>
                  <td colSpan={11} className="px-3 py-12 text-center">
                    <div className="inline-flex flex-col items-center gap-2">
                      <Search className="w-5 h-5" style={{ color: 'var(--text-muted)' }} />
                      <div className="text-[13px]" style={{ color: 'var(--text-secondary)' }}>No assets match</div>
                      <div className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Try clearing the filter or search</div>
                    </div>
                  </td>
                </tr>
              )}
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
   Mission Control — premium retune command center
   ══════════════════════════════════════════════════════════════════ */

function MissionControl({
  modes, retuneMode, retuneStatus, retune, elapsed, totalAssetCount,
  retuneLogs, showRetunePanel,
  onSelectMode, onStart, onStop, onToggleLog,
}: {
  modes: readonly { id: RetuneMode; label: string }[];
  retuneMode: RetuneMode;
  retuneStatus: RetuneStatus;
  retune: ReturnType<typeof getRetuneSnapshot>;
  elapsed: number | null;
  totalAssetCount: number;
  retuneLogs: RetuneLogEntry[];
  showRetunePanel: boolean;
  onSelectMode: (id: RetuneMode) => void;
  onStart: () => void;
  onStop: () => void;
  onToggleLog: () => void;
}) {
  const running = retuneStatus === 'running';
  const processed = retune.totalAssets;
  const target = Math.max(totalAssetCount, processed, 1);
  const pct = Math.min((processed / target) * 100, 100);
  const ratePerMin = elapsed && elapsed > 0 ? (processed / (elapsed / 60)) : 0;
  const etaSec = running && ratePerMin > 0
    ? Math.max(0, ((target - processed) / ratePerMin) * 60)
    : null;

  const progressColor = pct > 66
    ? 'linear-gradient(90deg, var(--accent-emerald), var(--accent-cyan))'
    : pct > 33
      ? 'linear-gradient(90deg, var(--accent-violet), var(--accent-cyan))'
      : 'linear-gradient(90deg, var(--accent-violet), var(--accent-indigo))';

  return (
    <div
      className="relative overflow-hidden rounded-[22px] p-5 mb-6 fade-up"
      style={{
        background:
          'linear-gradient(180deg, rgba(139,92,246,0.05) 0%, rgba(99,102,241,0.02) 100%)',
        border: '1px solid var(--violet-10)',
        boxShadow:
          '0 1px 0 rgba(255,255,255,0.04) inset, 0 20px 60px -30px rgba(139,92,246,0.25)',
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
      }}
    >
      {/* soft top light */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-px"
        style={{ background: 'linear-gradient(90deg, transparent, var(--violet-25), transparent)' }}
      />

      <div className="flex items-center justify-between flex-wrap gap-4">
        {/* Left: segmented modes + CTA + status */}
        <div className="flex items-center gap-3 flex-wrap">
          {/* Apple-style segmented control */}
          <div
            className="flex p-1 rounded-full"
            style={{
              background: 'rgba(20,20,30,0.5)',
              border: '1px solid var(--violet-8)',
            }}
          >
            {modes.map(({ id, label }) => {
              const active = retuneMode === id;
              return (
                <button
                  key={id}
                  onClick={() => onSelectMode(id)}
                  disabled={running}
                  className="px-4 py-1.5 rounded-full text-[12.5px] font-medium transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
                  style={{
                    background: active
                      ? 'linear-gradient(180deg, rgba(139,92,246,0.25), rgba(139,92,246,0.12))'
                      : 'transparent',
                    color: active ? '#d7ccff' : '#94a3b8',
                    boxShadow: active
                      ? '0 1px 0 rgba(255,255,255,0.08) inset, 0 4px 12px -4px rgba(139,92,246,0.45)'
                      : 'none',
                    letterSpacing: '-0.01em',
                  }}
                >
                  {label}
                </button>
              );
            })}
          </div>

          {/* Primary CTA */}
          {running ? (
            <button
              onClick={onStop}
              className="flex items-center gap-2 px-5 py-2 rounded-full text-[13px] font-semibold text-white transition-all hover:brightness-110 active:scale-[0.98]"
              style={{
                background: 'linear-gradient(180deg, #ff6a87 0%, #e11d48 100%)',
                boxShadow:
                  '0 1px 0 rgba(255,255,255,0.15) inset, 0 6px 18px -6px rgba(225,29,72,0.55)',
                letterSpacing: '-0.01em',
              }}
            >
              <Square className="w-3.5 h-3.5" fill="currentColor" />
              Stop
            </button>
          ) : (
            <button
              onClick={onStart}
              className="flex items-center gap-2 px-5 py-2 rounded-full text-[13px] font-semibold text-white transition-all hover:brightness-110 active:scale-[0.98]"
              style={{
                background: 'linear-gradient(180deg, #a78bfa 0%, #6366f1 100%)',
                boxShadow:
                  '0 1px 0 rgba(255,255,255,0.2) inset, 0 8px 22px -6px rgba(139,92,246,0.55)',
                letterSpacing: '-0.01em',
              }}
            >
              <Play className="w-3.5 h-3.5" fill="currentColor" />
              Start Retune
            </button>
          )}

          <StatusBadge status={retuneStatus} />
        </div>

        {/* Right: live stats */}
        <div className="flex items-center gap-3 text-[12px]" style={{ color: '#94a3b8' }}>
          {elapsed != null && (
            <StatChip icon={<Timer className="w-3.5 h-3.5" />} tone="violet">
              {formatElapsed(elapsed)}
            </StatChip>
          )}
          {running && processed > 0 && (
            <>
              <StatChip icon={<CheckCircle className="w-3.5 h-3.5" />} tone="emerald">
                {retune.successCount}
              </StatChip>
              {retune.failCount > 0 && (
                <StatChip icon={<XCircle className="w-3.5 h-3.5" />} tone="rose">
                  {retune.failCount}
                </StatChip>
              )}
              {retune.currentAsset && (
                <span
                  className="px-2.5 py-1 rounded-full font-mono text-[11px]"
                  style={{
                    background: 'var(--violet-10)',
                    color: '#c9b8ff',
                    border: '1px solid var(--violet-15)',
                    letterSpacing: '0.02em',
                  }}
                  title={retune.currentAsset}
                >
                  {retune.currentAsset}
                </span>
              )}
            </>
          )}
          {retuneLogs.length > 0 && (
            <button
              onClick={onToggleLog}
              className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] transition-all"
              style={{
                background: showRetunePanel ? 'var(--violet-15)' : 'rgba(255,255,255,0.02)',
                color: showRetunePanel ? '#c9b8ff' : '#94a3b8',
                border: `1px solid ${showRetunePanel ? 'var(--violet-25)' : 'var(--violet-8)'}`,
              }}
            >
              <Terminal className="w-3 h-3" /> Log
            </button>
          )}
        </div>
      </div>

      {/* Progress — only while running or with partial results */}
      {(running || processed > 0) && (
        <div className="mt-4">
          <div className="flex items-baseline justify-between mb-2">
            <div className="flex items-center gap-2 text-[12px]" style={{ color: '#c9b8ff' }}>
              <span
                className="font-medium truncate max-w-[320px]"
                style={{ letterSpacing: '-0.01em' }}
                title={retune.currentPhase || 'Processing'}
              >
                {retune.currentPhase || (running ? 'Processing…' : 'Complete')}
              </span>
              {etaSec != null && etaSec > 0 && (
                <span className="text-[11px]" style={{ color: '#7a8ba4' }}>
                  · ETA {formatElapsed(etaSec)}
                </span>
              )}
            </div>
            <div
              className="font-mono text-[12px]"
              style={{ color: '#c9b8ff', fontVariantNumeric: 'tabular-nums' }}
            >
              {processed}
              <span style={{ color: '#7a8ba4' }}> / {target}</span>
              <span style={{ color: '#7a8ba4' }}> · {pct.toFixed(0)}%</span>
              {ratePerMin > 0 && (
                <span style={{ color: '#7a8ba4' }}> · {ratePerMin.toFixed(1)}/min</span>
              )}
            </div>
          </div>
          <div
            className="h-[6px] rounded-full overflow-hidden"
            style={{
              background: 'rgba(255,255,255,0.04)',
              border: '1px solid rgba(255,255,255,0.03)',
            }}
          >
            <div
              className="h-full rounded-full transition-[width] duration-700 ease-out"
              style={{
                width: `${pct}%`,
                background: progressColor,
                boxShadow: '0 0 10px rgba(139,92,246,0.4)',
              }}
            />
          </div>
        </div>
      )}
    </div>
  );
}

function StatChip({
  icon, tone, children,
}: {
  icon: React.ReactNode;
  tone: 'violet' | 'emerald' | 'rose';
  children: React.ReactNode;
}) {
  const palette = {
    violet: { bg: 'var(--violet-10)', color: '#c9b8ff', border: 'var(--violet-15)' },
    emerald: { bg: 'var(--emerald-12)', color: 'var(--accent-emerald)', border: 'rgba(52,211,153,0.2)' },
    rose: { bg: 'var(--rose-12)', color: 'var(--accent-rose)', border: 'rgba(255,107,138,0.2)' },
  }[tone];
  return (
    <span
      className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full font-mono text-[11.5px]"
      style={{
        background: palette.bg,
        color: palette.color,
        border: `1px solid ${palette.border}`,
        fontVariantNumeric: 'tabular-nums',
      }}
    >
      {icon}
      {children}
    </span>
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
    <button
      onClick={onClick}
      className="group relative w-full text-left p-4 transition-all duration-200 hover:-translate-y-[1px] focus-ring"
      style={{
        borderRadius: 16,
        background: active
          ? `linear-gradient(160deg, ${color}10, rgba(255,255,255,0.01))`
          : 'linear-gradient(160deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01))',
        border: `1px solid ${active ? color + '30' : 'var(--violet-8)'}`,
        boxShadow: active
          ? `0 1px 0 ${color}20 inset, 0 6px 24px -12px ${color}35`
          : '0 1px 0 rgba(255,255,255,0.02) inset',
        backdropFilter: 'blur(14px)',
        WebkitBackdropFilter: 'blur(14px)',
      }}
    >
      {/* Top hairline shimmer */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-3 top-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${active ? color + '60' : 'rgba(255,255,255,0.08)'}, transparent)` }}
      />
      <div className="flex items-start justify-between mb-3">
        <div
          className="inline-flex items-center justify-center w-7 h-7 rounded-lg transition-transform duration-200 group-hover:scale-[1.04]"
          style={{
            background: `${color}12`,
            color,
            border: `1px solid ${color}22`,
          }}
        >
          {icon}
        </div>
        {active && (
          <span
            className="w-1.5 h-1.5 rounded-full"
            style={{ background: color, boxShadow: `0 0 8px ${color}` }}
          />
        )}
      </div>
      <div
        className="font-mono"
        style={{
          color: active ? color : 'var(--text-luminous)',
          fontVariantNumeric: 'tabular-nums',
          fontSize: 26,
          lineHeight: 1.1,
          letterSpacing: '-0.02em',
          fontWeight: 600,
        }}
      >
        {value}
      </div>
      <div
        className="mt-1 text-[11px]"
        style={{
          color: 'var(--text-secondary)',
          letterSpacing: '0.01em',
        }}
      >
        {label}
      </div>
      {sub && (
        <div
          className="mt-0.5 text-[10px]"
          style={{ color: 'var(--text-muted)', fontVariantNumeric: 'tabular-nums' }}
        >
          {sub}
        </div>
      )}
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
    <div className="flex items-center gap-2 justify-end">
      <span className="font-mono text-[10.5px] tabular-nums" style={{ color, fontVariantNumeric: 'tabular-nums', minWidth: 36, textAlign: 'right' }}>{pct.toFixed(1)}%</span>
      <div className="w-12 h-[3px] rounded-full overflow-hidden" style={{ background: 'rgba(255,255,255,0.05)' }}>
        <div className="h-full rounded-full transition-all duration-200" style={{ width: `${Math.min(pct, 100)}%`, background: color, boxShadow: `0 0 4px ${color}55` }} />
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Grade Badge
   ══════════════════════════════════════════════════════════════════ */

function GradeBadge({ grade }: { grade: string | null }) {
  if (!grade) return <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>--</span>;
  const colors: Record<string, { bg: string; text: string; ring: string }> = {
    'A': { bg: 'var(--emerald-12)', text: 'var(--accent-emerald)', ring: 'rgba(52,211,153,0.25)' },
    'B': { bg: 'rgba(59,130,246,0.12)', text: '#60a5fa', ring: 'rgba(96,165,250,0.25)' },
    'C': { bg: 'var(--amber-12)', text: 'var(--accent-amber)', ring: 'rgba(251,191,36,0.25)' },
    'D': { bg: 'var(--rose-12)', text: 'var(--accent-rose)', ring: 'rgba(255,107,138,0.28)' },
  };
  const c = colors[grade] ?? { bg: 'var(--violet-6)', text: '#94a3b8', ring: 'var(--violet-10)' };
  return (
    <span
      className="inline-flex items-center justify-center rounded-md text-[10px] font-bold"
      style={{
        background: c.bg,
        color: c.text,
        border: `1px solid ${c.ring}`,
        minWidth: 22,
        height: 18,
        padding: '0 5px',
        letterSpacing: '0.02em',
      }}
    >
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
    <th
      className={`px-3 py-2.5 text-${align} cursor-pointer select-none transition-colors hover:text-[#c9b8ff]`}
      style={{
        color: active ? '#c9b8ff' : 'var(--text-muted)',
        fontSize: 10,
        fontWeight: 600,
        textTransform: 'uppercase',
        letterSpacing: '0.08em',
      }}
      onClick={() => onSort(sk)}
    >
      <span className={`inline-flex items-center gap-1 ${align === 'right' ? 'justify-end' : align === 'center' ? 'justify-center' : ''}`}>
        <span>{label}</span>
        {active && dir === 'asc' && <ChevronUp className="w-3 h-3" strokeWidth={2.5} />}
        {active && dir === 'desc' && <ChevronDown className="w-3 h-3" strokeWidth={2.5} />}
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

function RetunePanel({ status, logs, onClose, elapsed, retune }: {
  status: RetuneStatus; logs: RetuneLogEntry[];
  onClose: () => void;
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

  // Keep the log view pinned to bottom WITHOUT scrolling the page.
  // Using scrollTop on the internal container avoids scrollIntoView,
  // which would bubble up to ancestor scroll containers and make the
  // whole page “bounce” on every new SSE tick.
  useEffect(() => {
    if (!autoScroll) return;
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
  }, [logs, autoScroll]);

  const jumpToBottom = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    setAutoScroll(true);
  }, []);

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
          <button onClick={handleCopyLog} className="inline-flex items-center justify-center w-7 h-7 rounded-full transition-all hover:brightness-125" title="Copy log" style={{ color: '#94a3b8', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--violet-8)' }}>
            <Copy className="w-3 h-3" />
          </button>
          <button onClick={onClose} className="inline-flex items-center justify-center w-7 h-7 rounded-full transition-all hover:brightness-125" title="Close" style={{ color: '#94a3b8', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--violet-8)' }}>
            <X className="w-3 h-3" />
          </button>
        </div>
      </div>

      <div ref={scrollRef} onScroll={handleScroll}
        className="p-3 overflow-y-auto max-h-[280px] font-mono text-[11px]"
        style={{ background: 'rgba(10,10,26,0.8)' }}>
        {logs.map((entry, i) => (
          <div key={i} className={`py-0.5 ${logColor(entry.type)}`}
            style={{ borderLeft: entry.type === 'progress' ? '1px solid var(--accent-emerald)' : entry.type === 'error' || entry.type === 'failed' ? '1px solid var(--accent-rose)' : entry.type === 'phase' ? '1px solid var(--accent-cyan)' : '1px solid transparent', paddingLeft: 10, opacity: 0.92 }}>
            {entry.message}
          </div>
        ))}
        {!autoScroll && (
          <button onClick={jumpToBottom}
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
    // Compact view — all individual models with full names
    const globalPitPass = data.reduce((s, d) => s + (d.analytics?.pit_pass ?? 0), 0);
    const globalPitFail = data.reduce((s, d) => s + (d.analytics?.pit_fail ?? 0), 0);
    const globalPitRate = (globalPitPass + globalPitFail) > 0 ? Math.round((globalPitPass / (globalPitPass + globalPitFail)) * 100) : null;
    const allBics = data.filter(d => d.analytics?.avg_bic != null).map(d => d.analytics!.avg_bic!);
    const globalAvgBic = allBics.length > 0 ? allBics.reduce((a, b) => a + b, 0) / allBics.length : null;
    const maxCount = data.length > 0 ? data[0].count : 1;

    return (
      <div
        className="glass-card p-5 mb-6 fade-up"
        style={{
          background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.02))',
          borderRadius: 16,
        }}
      >
        {/* Header — Apple-like: icon chip · title stack · hairline pill stats */}
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-3">
            <div
              className="w-8 h-8 rounded-[10px] flex items-center justify-center"
              style={{
                background: 'linear-gradient(135deg, rgba(56,217,245,0.14), rgba(56,217,245,0.04))',
                border: '1px solid rgba(56,217,245,0.22)',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06)',
              }}
            >
              <PieChart className="w-[15px] h-[15px]" style={{ color: 'var(--accent-cyan)' }} strokeWidth={2.25} />
            </div>
            <div className="flex flex-col leading-tight">
              <span className="text-[13px] font-semibold" style={{ color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                Model Distribution
              </span>
              <span
                className="text-[10px] uppercase font-semibold mt-[2px] tabular-nums"
                style={{ color: 'var(--text-muted)', letterSpacing: '0.1em' }}
              >
                {data.length} models &middot; {total} assets
              </span>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {globalPitRate != null && (
              <span
                className="inline-flex items-center gap-1 h-[22px] px-2.5 rounded-full text-[10px] font-semibold tabular-nums"
                style={{
                  background: globalPitRate >= 90 ? 'rgba(16,185,129,0.08)' : 'rgba(245,158,11,0.08)',
                  color: globalPitRate >= 90 ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                  border: `1px solid ${globalPitRate >= 90 ? 'rgba(16,185,129,0.24)' : 'rgba(245,158,11,0.24)'}`,
                  letterSpacing: '0.02em',
                }}
              >
                <span className="opacity-70">PIT</span>
                <span>{globalPitRate}%</span>
              </span>
            )}
            {globalAvgBic != null && (
              <span
                className="inline-flex items-center gap-1 h-[22px] px-2.5 rounded-full text-[10px] font-semibold tabular-nums"
                style={{
                  background: 'rgba(139,92,246,0.08)',
                  color: '#b49aff',
                  border: '1px solid rgba(139,92,246,0.22)',
                  letterSpacing: '0.02em',
                }}
              >
                <span className="opacity-70">AVG BIC</span>
                <span>{globalAvgBic.toFixed(0)}</span>
              </span>
            )}
          </div>
        </div>

        {/* Column legend */}
        <div
          className="flex items-center gap-3 px-3 pb-2 text-[9.5px] uppercase font-semibold tabular-nums"
          style={{ color: 'var(--text-muted)', letterSpacing: '0.12em', opacity: 0.7 }}
        >
          <span style={{ flex: '0 0 280px' }}>Model</span>
          <span className="flex-1">Share</span>
          <span style={{ flex: '0 0 70px' }} className="text-right">Assets</span>
          <span style={{ flex: '0 0 50px' }} className="text-right">PIT</span>
          <span style={{ flex: '0 0 65px' }} className="text-right">Avg BIC</span>
        </div>

        {/* All models — hairline rows, Apple-like */}
        <div>
          {data.map((d, i) => {
            const frac = d.count / (total || 1);
            const barPct = (d.count / maxCount) * 100;
            const colors = MODEL_COLORS[modelFamily(d.fullName ?? d.name)] ?? MODEL_COLORS.default;
            const isBest = (d.fullName ?? d.name) === bestBicModel;
            const a = d.analytics;
            const pitTotal = a ? a.pit_pass + a.pit_fail : 0;
            const pitRate = pitTotal > 0 ? Math.round((a!.pit_pass / pitTotal) * 100) : null;
            return (
              <div
                key={d.fullName ?? d.name}
                className="relative flex items-center gap-3 px-3 py-[9px] transition-colors duration-150 group"
                style={{
                  borderTop: i === 0 ? '1px solid rgba(139,92,246,0.09)' : undefined,
                  borderBottom: '1px solid rgba(139,92,246,0.09)',
                  background: isBest
                    ? 'linear-gradient(90deg, rgba(16,185,129,0.06), transparent 60%)'
                    : undefined,
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background =
                    isBest
                      ? 'linear-gradient(90deg, rgba(16,185,129,0.10), rgba(139,92,246,0.04) 60%)'
                      : 'rgba(139,92,246,0.04)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLDivElement).style.background = isBest
                    ? 'linear-gradient(90deg, rgba(16,185,129,0.06), transparent 60%)'
                    : '';
                }}
              >
                {/* Left accent rail for best */}
                {isBest && (
                  <span
                    aria-hidden
                    className="absolute left-0 top-2 bottom-2 w-[2px] rounded-full"
                    style={{ background: 'var(--accent-emerald)', boxShadow: '0 0 6px rgba(16,185,129,0.55)' }}
                  />
                )}

                {/* Model name */}
                <div className="flex items-center gap-1.5 min-w-0" style={{ flex: '0 0 280px' }}>
                  <span
                    className="w-1.5 h-1.5 rounded-full flex-shrink-0"
                    style={{ background: colors.bar, boxShadow: `0 0 6px ${colors.bar}80` }}
                  />
                  <span
                    className="text-[11.5px] font-semibold truncate"
                    style={{ color: colors.text, letterSpacing: '-0.005em' }}
                  >
                    {formatModelName(d.fullName ?? d.name)}
                  </span>
                  {isBest && (
                    <Target
                      className="w-3 h-3 flex-shrink-0"
                      style={{ color: 'var(--accent-emerald)' }}
                      strokeWidth={2.25}
                    />
                  )}
                </div>

                {/* Share bar — 3px, thin, elegant */}
                <div className="flex items-center flex-1 min-w-0">
                  <div
                    className="flex-1 h-[3px] rounded-full overflow-hidden"
                    style={{ background: 'rgba(139,92,246,0.08)' }}
                  >
                    <div
                      className="h-full rounded-full transition-all duration-500"
                      style={{
                        width: `${Math.max(barPct, 2)}%`,
                        background: `linear-gradient(90deg, ${colors.bar}, ${colors.bar}aa)`,
                        boxShadow: `0 0 6px ${colors.bar}55`,
                      }}
                    />
                  </div>
                </div>

                {/* Asset count + share */}
                <div className="flex items-baseline justify-end gap-1 tabular-nums" style={{ flex: '0 0 70px' }}>
                  <span className="text-[12px] font-semibold" style={{ color: 'var(--text-primary)', letterSpacing: '-0.01em' }}>
                    {d.count}
                  </span>
                  <span className="text-[9.5px]" style={{ color: 'var(--text-muted)' }}>
                    {(frac * 100).toFixed(1)}%
                  </span>
                </div>

                {/* PIT rate */}
                <div style={{ flex: '0 0 50px' }} className="text-right tabular-nums">
                  {pitRate != null ? (
                    <span
                      className="text-[10.5px] font-semibold"
                      style={{
                        color:
                          pitRate >= 90
                            ? 'var(--accent-emerald)'
                            : pitRate >= 70
                            ? 'var(--accent-amber)'
                            : 'var(--accent-rose)',
                      }}
                    >
                      {pitRate}%
                    </span>
                  ) : (
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      &ndash;
                    </span>
                  )}
                </div>

                {/* Avg BIC */}
                <div style={{ flex: '0 0 65px' }} className="text-right tabular-nums">
                  <span
                    className="text-[10.5px] font-semibold"
                    style={{ color: a?.avg_bic != null ? '#b49aff' : 'var(--text-muted)' }}
                  >
                    {a?.avg_bic != null ? a.avg_bic.toFixed(0) : '–'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    );
  }

  // ── Expanded: Full analytics dashboard ──────────────────────────
  const globalPitPassExp = data.reduce((s, d) => s + (d.analytics?.pit_pass ?? 0), 0);
  const globalPitFailExp = data.reduce((s, d) => s + (d.analytics?.pit_fail ?? 0), 0);
  const globalPitRateExp = (globalPitPassExp + globalPitFailExp) > 0 ? Math.round((globalPitPassExp / (globalPitPassExp + globalPitFailExp)) * 100) : null;

  return (
    <div
      className="glass-card p-5 mb-6 fade-up"
      style={{
        background: 'linear-gradient(135deg, var(--violet-3), rgba(99,102,241,0.03))',
        boxShadow: '0 4px 24px rgba(99,102,241,0.06)',
        borderRadius: 16,
      }}
    >
      {/* Header — icon chip · title · hairline pills */}
      <div className="flex items-center justify-between mb-5">
        <div className="flex items-center gap-3">
          <div
            className="w-10 h-10 rounded-[12px] flex items-center justify-center"
            style={{
              background: 'linear-gradient(135deg, rgba(56,217,245,0.16), rgba(56,217,245,0.04))',
              border: '1px solid rgba(56,217,245,0.24)',
              boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.06), 0 0 16px rgba(34,211,238,0.08)',
            }}
          >
            <Layers className="w-[18px] h-[18px]" style={{ color: 'var(--accent-cyan)' }} strokeWidth={2.25} />
          </div>
          <div className="flex flex-col leading-tight">
            <h3 className="text-[15px] font-semibold" style={{ color: 'var(--text-primary)', letterSpacing: '-0.015em' }}>
              Model Analytics
            </h3>
            <span
              className="text-[10px] uppercase font-semibold mt-[2px] tabular-nums"
              style={{ color: 'var(--text-muted)', letterSpacing: '0.12em' }}
            >
              {data.length} competing models &middot; {total} assets
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {globalPitRateExp != null && (
            <span
              className="inline-flex items-center gap-1.5 h-[26px] px-3 rounded-full text-[10.5px] font-semibold tabular-nums"
              style={{
                background: globalPitRateExp >= 90 ? 'rgba(16,185,129,0.08)' : 'rgba(245,158,11,0.08)',
                border: `1px solid ${globalPitRateExp >= 90 ? 'rgba(16,185,129,0.24)' : 'rgba(245,158,11,0.24)'}`,
                color: globalPitRateExp >= 90 ? 'var(--accent-emerald)' : 'var(--accent-amber)',
                letterSpacing: '0.02em',
              }}
            >
              <CheckCircle className="w-[12px] h-[12px]" strokeWidth={2.4} />
              <span className="opacity-70">PIT</span>
              <span>{globalPitRateExp}%</span>
            </span>
          )}
          {bestBicModel && (
            <span
              className="inline-flex items-center gap-1.5 h-[26px] px-3 rounded-full text-[10.5px] font-semibold"
              style={{
                background: 'rgba(16,185,129,0.06)',
                border: '1px solid rgba(16,185,129,0.2)',
                color: 'var(--accent-emerald)',
                letterSpacing: '0.01em',
              }}
            >
              <Award className="w-[12px] h-[12px]" strokeWidth={2.4} />
              <span className="opacity-70 uppercase" style={{ letterSpacing: '0.1em' }}>Best</span>
              <span>{formatModelName(bestBicModel)}</span>
            </span>
          )}
        </div>
      </div>

      {/* Full model analytics table — hairline header, tabular-nums body */}
      <div className="overflow-x-auto rounded-[12px]" style={{ border: '1px solid var(--violet-6)' }}>
        <table className="w-full text-xs">
          <thead>
            <tr style={{ background: 'rgba(10,10,26,0.55)', borderBottom: '1px solid var(--violet-8)' }}>
              <th className="text-left px-3 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Model</th>
              <th className="text-center px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Assets</th>
              <th className="text-right px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Avg BIC</th>
              <th className="text-right px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Best BIC</th>
              <th className="text-center px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>PIT Rate</th>
              <th className="text-right px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Avg φ</th>
              <th className="text-right px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Avg ν</th>
              <th className="text-right px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Avg KS p</th>
              <th className="text-right px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Avg Wt</th>
              <th className="px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', minWidth: 100, fontSize: 9.5, letterSpacing: '0.1em' }}>Share</th>
              <th className="text-left px-2 py-2.5 font-semibold uppercase" style={{ color: 'var(--text-muted)', fontSize: 9.5, letterSpacing: '0.1em' }}>Top Symbols</th>
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
                  {/* Model name — full name */}
                  <td className="px-3 py-2">
                    <div className="flex items-center gap-2">
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-medium" style={{
                        background: colors.bg, color: colors.text,
                      }}>
                        {formatModelName(d.fullName ?? d.name)}
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
                  {data.slice(0, 5).map((d) => {
                    const colors = MODEL_COLORS[modelFamily(d.fullName ?? d.name)] ?? MODEL_COLORS.default;
                    return (
                      <span key={d.fullName ?? d.name} style={{ color: colors.text }}>
                        {formatModelName(d.fullName ?? d.name)}: <strong>{d.count}</strong> ({((d.count / total) * 100).toFixed(0)}%)
                      </span>
                    );
                  })}
                  {data.length > 5 && <span style={{ color: '#7a8ba4' }}>+{data.length - 5} more</span>}
                </div>
              </td>
            </tr>
          </tfoot>
        </table>
      </div>

      {/* Hover detail card — premium deep-dive */}
      {hoveredAnalytics && hoveredModel && (() => {
        const hColors = MODEL_COLORS[modelFamily(hoveredModel)] ?? MODEL_COLORS.default;
        const hPitTotal = hoveredAnalytics.pit_pass + hoveredAnalytics.pit_fail;
        const hPitRate = hPitTotal > 0 ? (hoveredAnalytics.pit_pass / hPitTotal) * 100 : null;
        return (
          <div className="mt-4 p-4 rounded-xl transition-all duration-200 relative overflow-hidden" style={{
            background: `linear-gradient(135deg, rgba(10,10,26,0.6), ${hColors.bg})`,
            border: `1px solid ${hColors.bar}30`,
            boxShadow: `0 4px 20px ${hColors.bar}08`,
          }}>
            <div className="absolute -bottom-10 -right-10 w-32 h-32 rounded-full blur-3xl" style={{ background: `${hColors.bar}06` }} />
            <div className="relative">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <Eye className="w-4 h-4" style={{ color: hColors.text }} />
                  <span className="text-sm font-bold" style={{ color: 'var(--text-primary)' }}>
                    {formatModelName(hoveredModel)}
                  </span>
                  <span className="px-2 py-0.5 rounded-md text-[10px] font-mono font-medium" style={{
                    background: hColors.bg, color: hColors.text, border: `1px solid ${hColors.bar}30`,
                  }}>
                    {hoveredAnalytics.count} assets
                  </span>
                </div>
                {hPitRate != null && (
                  <div className="flex items-center gap-1.5">
                    <div className="w-16 h-2 rounded-full overflow-hidden" style={{ background: 'rgba(0,0,0,0.3)' }}>
                      <div className="h-full rounded-full" style={{
                        width: `${hPitRate}%`,
                        background: hPitRate >= 90 ? 'var(--accent-emerald)' : hPitRate >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)',
                      }} />
                    </div>
                    <span className="text-[10px] font-mono font-semibold" style={{
                      color: hPitRate >= 90 ? 'var(--accent-emerald)' : hPitRate >= 70 ? 'var(--accent-amber)' : 'var(--accent-rose)',
                    }}>PIT {hPitRate.toFixed(0)}%</span>
                  </div>
                )}
              </div>
              <div className="grid grid-cols-4 md:grid-cols-8 gap-2">
                {[
                  { label: 'Avg BIC', value: hoveredAnalytics.avg_bic?.toFixed(0), color: '#b49aff' },
                  { label: 'Best BIC', value: hoveredAnalytics.best_bic?.toFixed(0), color: 'var(--accent-cyan)' },
                  { label: 'Worst BIC', value: hoveredAnalytics.worst_bic?.toFixed(0), color: 'var(--accent-rose)' },
                  { label: 'Median BIC', value: hoveredAnalytics.median_bic?.toFixed(0), color: '#94a3b8' },
                  { label: 'Avg KS p', value: hoveredAnalytics.avg_ks_pvalue?.toFixed(4), color: hoveredAnalytics.avg_ks_pvalue != null && hoveredAnalytics.avg_ks_pvalue >= 0.05 ? 'var(--accent-emerald)' : 'var(--accent-rose)' },
                  { label: 'PIT Pass', value: hPitTotal > 0 ? `${hoveredAnalytics.pit_pass}/${hPitTotal}` : null, color: hPitRate != null && hPitRate >= 90 ? 'var(--accent-emerald)' : 'var(--accent-amber)' },
                  { label: 'Avg Weight', value: hoveredAnalytics.avg_weight != null ? `${(hoveredAnalytics.avg_weight * 100).toFixed(1)}%` : null, color: hColors.text },
                  { label: 'Avg Obs', value: hoveredAnalytics.avg_n_obs?.toFixed(0), color: '#94a3b8' },
                ].map(p => (
                  <div key={p.label} className="p-2 rounded-lg text-center" style={{ background: 'rgba(0,0,0,0.2)' }}>
                    <div className="text-[7px] uppercase tracking-widest mb-1" style={{ color: '#7a8ba4' }}>{p.label}</div>
                    <div className="text-[11px] font-mono font-bold" style={{ color: p.value ? p.color : 'var(--text-muted)' }}>
                      {p.value ?? '--'}
                    </div>
                  </div>
                ))}
              </div>
              {hoveredAnalytics.top_symbols && hoveredAnalytics.top_symbols.length > 0 && (
                <div className="mt-2.5 flex items-center gap-1.5 flex-wrap">
                  <span className="text-[9px] uppercase tracking-wider" style={{ color: '#7a8ba4' }}>Top:</span>
                  {hoveredAnalytics.top_symbols.slice(0, 6).map(s => (
                    <span key={s} className="px-1.5 py-0.5 rounded text-[9px] font-mono font-medium" style={{
                      background: 'rgba(0,0,0,0.25)', color: 'var(--text-secondary)', border: '1px solid rgba(255,255,255,0.05)',
                    }}>
                      {s}
                    </span>
                  ))}
                </div>
              )}
            </div>
          </div>
        );
      })()}
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
            className="inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-[12px] font-medium transition-all duration-150 hover:brightness-110 active:scale-[0.98]"
            style={{
              color: '#c9b8ff',
              background: 'linear-gradient(180deg, rgba(180,154,255,0.12), rgba(180,154,255,0.05))',
              border: '1px solid var(--violet-15)',
              letterSpacing: '0.01em',
            }}
          >
            Diagnostics <ArrowRight className="w-3 h-3" />
          </button>
          <button onClick={onClose} className="inline-flex items-center justify-center w-8 h-8 rounded-full transition-all hover:brightness-125" style={{ color: '#94a3b8', background: 'rgba(255,255,255,0.03)', border: '1px solid var(--violet-8)' }}>
            <X className="w-3.5 h-3.5" />
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
      <span
        className="inline-flex items-center gap-1 rounded-md text-[10px] font-medium"
        style={{
          background: 'var(--emerald-12)',
          color: 'var(--accent-emerald)',
          border: '1px solid rgba(52,211,153,0.22)',
          padding: '1px 6px',
          height: 18,
          letterSpacing: '0.01em',
        }}
      >
        <CheckCircle className="w-2.5 h-2.5" strokeWidth={2.5} /> Pass
      </span>
    );
  if (asset.ad_pass === false)
    return (
      <span
        className="inline-flex items-center gap-1 rounded-md text-[10px] font-medium"
        style={{
          background: 'var(--rose-12)',
          color: 'var(--accent-rose)',
          border: '1px solid rgba(255,107,138,0.25)',
          padding: '1px 6px',
          height: 18,
          letterSpacing: '0.01em',
        }}
      >
        <XCircle className="w-2.5 h-2.5" strokeWidth={2.5} /> Fail
      </span>
    );
  return <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: 'rgba(148,163,184,0.5)' }} />;
}
