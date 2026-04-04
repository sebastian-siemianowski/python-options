import { useQuery } from '@tanstack/react-query';
import { useState, useMemo } from 'react';
import { api } from '../api';
import type { DiagAsset, DiagModelStats, DiagCrossAssetSummary } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { DiagnosticsSkeleton } from '../components/CosmicSkeleton';
import {
  Stethoscope, CheckCircle, XCircle, AlertTriangle, ChevronDown, ChevronRight,
  Search, BarChart3, Layers, RefreshCw, Grid3X3,
} from 'lucide-react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  PieChart, Pie, Cell,
} from 'recharts';
import { formatModelNameShort } from '../utils/modelNames';

type DiagTab = 'pit' | 'models' | 'regimes' | 'failures' | 'matrix';

export default function DiagnosticsPage() {
  const [tab, setTab] = useState<DiagTab>('pit');
  const [search, setSearch] = useState('');
  const [filterPit, setFilterPit] = useState<'all' | 'pass' | 'fail'>('all');
  const [expandedAsset, setExpandedAsset] = useState<string | null>(null);

  const pitQ = useQuery({ queryKey: ['diagPitSummary'], queryFn: api.diagPitSummary, staleTime: 120_000 });
  const modelsQ = useQuery({ queryKey: ['diagModelComparison'], queryFn: api.diagModelComparison, staleTime: 120_000 });
  const regimeQ = useQuery({ queryKey: ['diagRegimeDistribution'], queryFn: api.diagRegimeDistribution, staleTime: 120_000 });
  const failQ = useQuery({ queryKey: ['diagCalibrationFailures'], queryFn: api.diagCalibrationFailures, staleTime: 120_000 });
  const matrixQ = useQuery({ queryKey: ['diagCrossAssetSummary'], queryFn: api.diagCrossAssetSummary, staleTime: 120_000, enabled: tab === 'matrix' });

  const pitData = pitQ.data;
  const modelsData = modelsQ.data;
  const regimeData = regimeQ.data;
  const failData = failQ.data;

  const filteredAssets = useMemo(() => {
    if (!pitData?.assets) return [];
    let items = pitData.assets;
    if (search) {
      items = items.filter(a => a.symbol.toLowerCase().includes(search.toLowerCase()));
    }
    if (filterPit === 'pass') items = items.filter(a => a.ad_pass === true);
    if (filterPit === 'fail') items = items.filter(a => a.ad_pass === false);
    return items;
  }, [pitData, search, filterPit]);

  const isLoading = pitQ.isLoading && modelsQ.isLoading;
  if (isLoading) return <DiagnosticsSkeleton />;

  const tabs: { id: DiagTab; label: string; icon: typeof Stethoscope }[] = [
    { id: 'pit', label: 'PIT Calibration', icon: Stethoscope },
    { id: 'models', label: 'Model Comparison', icon: BarChart3 },
    { id: 'matrix', label: 'Cross-Asset Matrix', icon: Grid3X3 },
    { id: 'regimes', label: 'Regime Distribution', icon: Layers },
    { id: 'failures', label: 'Calibration Failures', icon: AlertTriangle },
  ];

  return (
    <>
      <PageHeader
        title="Diagnostics"
        action={
          <button
            onClick={() => {
              pitQ.refetch();
              modelsQ.refetch();
              regimeQ.refetch();
              failQ.refetch();
              matrixQ.refetch();
            }}
            disabled={pitQ.isFetching}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm transition-all duration-200 disabled:opacity-50"
            style={{
              background: 'var(--violet-8)',
              color: '#b49aff',
              border: '1px solid var(--violet-12)',
            }}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${pitQ.isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        }
      >
        PIT calibration, model comparison, and regime analysis — equivalent to{' '}
        <code style={{ color: '#b49aff' }}>make diag</code>
      </PageHeader>

      {/* Stats bar */}
      {pitData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-5 mb-8 fade-up">
          <StatCard title="Total Assets" value={pitData.total} icon={<Stethoscope className="w-5 h-5" />} color="blue" />
          <StatCard title="PIT Pass" value={pitData.passing} icon={<CheckCircle className="w-5 h-5" />} color="green" />
          <StatCard title="PIT Fail" value={pitData.failing} icon={<XCircle className="w-5 h-5" />} color="red" />
          <StatCard
            title="Calib. Failures"
            value={failData?.count ?? 0}
            icon={<AlertTriangle className="w-5 h-5" />}
            color={failData?.count ? 'amber' : 'green'}
          />
        </div>
      )}

      {/* Tab nav */}
      <div className="flex gap-1 mb-8 fade-up-delay-1" style={{ borderBottom: '1px solid var(--violet-6)' }}>
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className="flex items-center gap-2 px-5 py-3 text-sm font-medium transition-all duration-200 -mb-[1px]"
            style={{
              borderBottom: tab === id ? '2px solid var(--accent-violet)' : '2px solid transparent',
              color: tab === id ? '#b49aff' : 'var(--text-muted)',
              background: tab === id ? 'var(--violet-4)' : undefined,
              borderRadius: '12px 12px 0 0',
            }}
          >
            <Icon className="w-4 h-4" />
            {label}
          </button>
        ))}
      </div>

      {/* Tab content */}
      {tab === 'pit' && <PitTab assets={filteredAssets} search={search} setSearch={setSearch} filterPit={filterPit} setFilterPit={setFilterPit} expandedAsset={expandedAsset} setExpandedAsset={setExpandedAsset} />}
      {tab === 'models' && modelsData && <ModelsTab data={modelsData} />}
      {tab === 'matrix' && <MatrixTab data={matrixQ.data} isLoading={matrixQ.isLoading} />}
      {tab === 'regimes' && regimeData && <RegimesTab data={regimeData} />}
      {tab === 'failures' && <FailuresTab data={failData} />}
    </>
  );
}

/* ── PIT Calibration Tab ──────────────────────────────────────────── */

function PitTab({
  assets, search, setSearch, filterPit, setFilterPit, expandedAsset, setExpandedAsset,
}: {
  assets: DiagAsset[];
  search: string;
  setSearch: (s: string) => void;
  filterPit: 'all' | 'pass' | 'fail';
  setFilterPit: (f: 'all' | 'pass' | 'fail') => void;
  expandedAsset: string | null;
  setExpandedAsset: (s: string | null) => void;
}) {
  return (
    <div className="glass-card overflow-hidden">
      {/* Toolbar */}
      <div className="p-3 flex items-center gap-3" style={{ borderBottom: '1px solid var(--violet-8)' }}>
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: '#7a8ba4' }} />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets..."
            className="w-full pl-9 pr-3 py-1.5 rounded-xl text-sm outline-none transition-all duration-200"
            style={{
              background: 'rgba(10,10,26,0.6)',
              border: '1px solid var(--violet-8)',
              color: 'var(--text-primary)',
            }}
          />
        </div>
        <div className="flex gap-1">
          {(['all', 'pass', 'fail'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilterPit(f)}
              className="px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-200"
              style={filterPit === f ? {
                background: f === 'pass' ? 'var(--emerald-15)' : f === 'fail' ? 'var(--rose-15)' : 'var(--violet-15)',
                color: f === 'pass' ? 'var(--accent-emerald)' : f === 'fail' ? 'var(--accent-rose)' : '#b49aff',
              } : { color: '#7a8ba4' }}
            >
              {f === 'all' ? 'All' : f === 'pass' ? 'Pass' : 'Fail'}
            </button>
          ))}
        </div>
        <span className="text-xs" style={{ color: '#7a8ba4' }}>{assets.length} assets</span>
      </div>

      {/* Table */}
      <div className="overflow-y-auto max-h-[600px]">
        <table className="w-full text-xs">
          <thead className="premium-thead">
            <tr>
              <th className="w-6"></th>
              <th className="text-left">Symbol</th>
              <th className="text-left">Best Model</th>
              <th className="text-center">PIT</th>
              <th className="text-center">AD Stat</th>
              <th className="text-center">Grade</th>
              <th className="text-center">Models</th>
              <th className="text-center">Regime</th>
            </tr>
          </thead>
          <tbody>
            {assets.slice(0, 200).map((a) => (
              <AssetRow key={a.symbol} asset={a} expanded={expandedAsset === a.symbol} onToggle={() => setExpandedAsset(expandedAsset === a.symbol ? null : a.symbol)} />
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function AssetRow({ asset, expanded, onToggle }: { asset: DiagAsset; expanded: boolean; onToggle: () => void }) {
  const pitIcon = asset.ad_pass === true ? 'P' : asset.ad_pass === false ? 'F' : '--';

  return (
    <>
      <tr
        onClick={onToggle}
        className={`premium-row premium-row-interactive ${expanded ? 'premium-row-expanded' : ''}`}
      >
        <td style={{ color: 'var(--text-muted)', width: 28 }}>
          {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </td>
        <td className="font-medium" style={{ color: 'var(--text-luminous)' }}>{asset.symbol}</td>
        <td style={{ color: 'var(--text-secondary)' }}>{formatModelNameShort(asset.best_model)}</td>
        <td className="text-center">
          <span className={asset.ad_pass === true ? 'status-pill status-pass' : asset.ad_pass === false ? 'status-pill status-fail' : ''}>
            {pitIcon}
          </span>
        </td>
        <td className="text-center" style={{ color: 'var(--text-secondary)' }}>{asset.ad_stat?.toFixed(3) ?? '--'}</td>
        <td className="text-center">
          <GradeBadge grade={asset.pit_grade} />
        </td>
        <td className="text-center" style={{ color: 'var(--text-secondary)' }}>{asset.num_models}</td>
        <td className="text-center">
          <RegimeBadge regime={asset.regime} />
        </td>
      </tr>
      {expanded && asset.models?.length > 0 && (
        <tr>
          <td colSpan={8} className="px-4 py-3" style={{ background: 'rgba(10,10,26,0.6)' }}>
            <ModelMetricsTable models={asset.models} bmaWeights={asset.bma_weights} />
          </td>
        </tr>
      )}
    </>
  );
}

function ModelMetricsTable({ models, bmaWeights }: { models: DiagAsset['models']; bmaWeights: Record<string, number> }) {
  return (
    <div className="overflow-x-auto" style={{ borderRadius: '12px', background: 'rgba(10,10,26,0.3)' }}>
      <table className="w-full text-[11px]">
        <thead className="premium-thead">
          <tr>
            <th className="text-left">Model</th>
            <th className="text-right">BIC</th>
            <th className="text-right">CRPS</th>
            <th className="text-right">Hyv</th>
            <th className="text-right">PIT p</th>
            <th className="text-right">AD p</th>
            <th className="text-right">MAD</th>
            <th className="text-right">Weight</th>
            <th className="text-right">v</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const w = bmaWeights[m.model] ?? m.weight;
            return (
              <tr key={m.model} className="premium-row">
                <td className="font-medium" style={{ color: 'var(--text-luminous)' }}>{formatModelNameShort(m.model)}</td>
                <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{m.bic != null ? m.bic.toFixed(0) : '--'}</td>
                <td className="text-right">
                  <span style={{ color: m.crps != null && m.crps < 0.02 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                    {m.crps != null ? m.crps.toFixed(4) : '--'}
                  </span>
                </td>
                <td className="text-right">
                  <span style={{ color: m.hyvarinen != null && m.hyvarinen < 1000 ? 'var(--accent-emerald)' : m.hyvarinen != null && m.hyvarinen > 2000 ? 'var(--accent-rose)' : 'var(--text-secondary)' }}>
                    {m.hyvarinen != null ? m.hyvarinen.toFixed(0) : '--'}
                  </span>
                </td>
                <td className="text-right">
                  <span style={{ color: m.pit_ks_pvalue != null && m.pit_ks_pvalue >= 0.05 ? 'var(--accent-emerald)' : m.pit_ks_pvalue != null ? 'var(--accent-rose)' : 'var(--text-muted)' }}>
                    {m.pit_ks_pvalue != null ? m.pit_ks_pvalue.toFixed(3) : '--'}
                  </span>
                </td>
                <td className="text-right">
                  <span style={{ color: m.ad_pvalue != null && m.ad_pvalue >= 0.05 ? 'var(--accent-emerald)' : m.ad_pvalue != null ? 'var(--accent-rose)' : 'var(--text-muted)' }}>
                    {m.ad_pvalue != null ? m.ad_pvalue.toFixed(3) : '--'}
                  </span>
                </td>
                <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{m.histogram_mad != null ? m.histogram_mad.toFixed(4) : '--'}</td>
                <td className="text-right">
                  <span style={{ color: w > 0.2 ? 'var(--accent-emerald)' : w > 0.05 ? 'var(--accent-amber)' : 'var(--text-muted)', fontWeight: w > 0.2 ? 700 : 400 }}>
                    {(w * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{m.nu != null ? m.nu.toFixed(1) : '--'}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/* ── Model Comparison Tab ─────────────────────────────────────────── */

function ModelsTab({ data }: { data: { models: Record<string, DiagModelStats>; total_assets: number } }) {
  const models = useMemo(() => Object.values(data.models).sort((a, b) => b.win_count - a.win_count), [data.models]);

  const chartData = models.slice(0, 15).map(m => ({
    name: formatModelNameShort(m.name),
    wins: m.win_count,
    appearances: m.appearances,
  }));

  const winRateData = models.slice(0, 15).map(m => ({
    name: formatModelNameShort(m.name),
    winRate: +(m.win_rate * 100).toFixed(1),
    avgWeight: +(m.avg_weight * 100).toFixed(1),
  }));

  return (
    <div className="space-y-6">
      {/* Charts */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="glass-card hover-lift" style={{ padding: '24px' }}>
          <h3 className="premium-section-label mb-4">Win Count (Best Model Selection)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--violet-6)" />
              <XAxis type="number" tick={{ fill: '#7a8ba4', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: 'rgba(15,15,35,0.95)', border: '1px solid var(--violet-15)', borderRadius: 8, color: '#e2e8f0', backdropFilter: 'blur(12px)' }} />
              <Bar dataKey="wins" fill="var(--accent-violet)" radius={[0, 4, 4, 0]} name="Wins" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="glass-card hover-lift" style={{ padding: '24px' }}>
          <h3 className="premium-section-label mb-4">Avg BMA Weight (%)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={winRateData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="var(--violet-6)" />
              <XAxis type="number" tick={{ fill: '#7a8ba4', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: 'rgba(15,15,35,0.95)', border: '1px solid var(--violet-15)', borderRadius: 8, color: '#e2e8f0', backdropFilter: 'blur(12px)' }} />
              <Bar dataKey="avgWeight" fill="#b49aff" radius={[0, 4, 4, 0]} name="Avg Weight %" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="premium-thead">
              <tr>
                <th className="text-left">Model</th>
                <th className="text-right">Wins</th>
                <th className="text-right">Win Rate</th>
                <th className="text-right">Appearances</th>
                <th className="text-right">Avg Wt</th>
                <th className="text-right">Max Wt</th>
                <th className="text-right">Min Wt</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="premium-row">
                  <td className="font-medium" style={{ color: 'var(--text-luminous)' }}>{formatModelNameShort(m.name)}</td>
                  <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{m.win_count}</td>
                  <td className="text-right">
                    <span style={{ color: m.win_rate > 0.1 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                      {(m.win_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{m.appearances}</td>
                  <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{(m.avg_weight * 100).toFixed(1)}%</td>
                  <td className="text-right" style={{ color: 'var(--text-secondary)' }}>{(m.max_weight * 100).toFixed(1)}%</td>
                  <td className="text-right" style={{ color: 'var(--text-muted)' }}>{(m.min_weight * 100).toFixed(1)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ── Cross-Asset Matrix Tab ───────────────────────────────────────── */

function MatrixTab({ data, isLoading }: { data?: DiagCrossAssetSummary; isLoading: boolean }) {
  const [metric, setMetric] = useState<'crps' | 'pit_ks_p' | 'weight'>('crps');
  const [matrixSearch, setMatrixSearch] = useState('');

  if (isLoading) return <LoadingSpinner text="Loading cross-asset matrix..." />;
  if (!data || data.rows.length === 0) return <div className="glass-card p-6 text-center text-[var(--text-secondary)]">No data available</div>;

  const models = data.models;
  const filteredRows = matrixSearch
    ? data.rows.filter(r => r.symbol.toLowerCase().includes(matrixSearch.toLowerCase()))
    : data.rows;

  const getCellColor = (val: number | null | undefined, _m: string): string => {
    if (val == null) return 'text-[#2a2a4a]';
    if (metric === 'crps') return val < 0.015 ? 'text-[var(--accent-emerald)]' : val < 0.025 ? 'text-[var(--text-secondary)]' : 'text-[var(--accent-orange)]';
    if (metric === 'pit_ks_p') return val >= 0.05 ? 'text-[var(--accent-emerald)]' : 'text-[var(--accent-rose)]';
    if (metric === 'weight') return val > 0.2 ? 'text-[var(--accent-emerald)] font-bold' : val > 0.05 ? 'text-[var(--accent-amber)]' : 'text-[var(--text-secondary)]';
    return 'text-[var(--text-secondary)]';
  };

  const fmtVal = (val: number | null | undefined): string => {
    if (val == null) return '·';
    if (metric === 'crps') return val.toFixed(4);
    if (metric === 'pit_ks_p') return val.toFixed(3);
    if (metric === 'weight') return (val * 100).toFixed(0) + '%';
    return val.toFixed(3);
  };

  return (
    <div className="space-y-4">
      {/* Model averages summary */}
      <div className="glass-card p-4">
        <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Model Averages (across all assets)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr style={{ borderBottom: '1px solid var(--violet-8)' }}>
                <th className="text-left px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Model</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Avg CRPS</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Avg PIT p</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Avg BIC</th>
                <th className="text-right px-2 py-1.5" style={{ color: 'var(--text-muted)' }}>Assets</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => {
                const avg = data.model_averages[m];
                return (
                  <tr key={m} style={{ borderBottom: '1px solid var(--violet-4)' }} className="transition-all duration-150">
                    <td className="px-2 py-1.5 font-medium" style={{ color: 'var(--text-luminous)' }}>{formatModelNameShort(m)}</td>
                    <td className="px-2 py-1.5 text-right" style={{ color: avg?.avg_crps != null && avg.avg_crps < 0.02 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                      {avg?.avg_crps?.toFixed(5) ?? '—'}
                    </td>
                    <td className="px-2 py-1.5 text-right" style={{ color: avg?.avg_pit_p != null && avg.avg_pit_p >= 0.05 ? 'var(--accent-emerald)' : 'var(--text-secondary)' }}>
                      {avg?.avg_pit_p?.toFixed(4) ?? '—'}
                    </td>
                    <td className="px-2 py-1.5 text-right" style={{ color: 'var(--text-secondary)' }}>{avg?.avg_bic?.toFixed(0) ?? '--'}</td>
                    <td className="px-2 py-1.5 text-right" style={{ color: '#7a8ba4' }}>{avg?.count ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Matrix controls */}
      <div className="glass-card overflow-hidden">
        <div className="p-3 flex items-center gap-3" style={{ borderBottom: '1px solid var(--violet-8)' }}>
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: '#7a8ba4' }} />
            <input
              type="text"
              value={matrixSearch}
              onChange={(e) => setMatrixSearch(e.target.value)}
              placeholder="Search assets..."
              className="w-full pl-9 pr-3 py-1.5 rounded-xl text-sm outline-none transition-all duration-200"
              style={{
                background: 'rgba(10,10,26,0.6)',
                border: '1px solid var(--violet-8)',
                color: 'var(--text-primary)',
              }}
            />
          </div>
          <div className="flex gap-1">
            {([['crps', 'CRPS'], ['pit_ks_p', 'PIT p'], ['weight', 'Weight']] as const).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setMetric(k)}
                className="px-3 py-1.5 rounded-xl text-xs font-medium transition-all duration-200"
                style={metric === k ? {
                  background: 'var(--violet-15)',
                  color: '#b49aff',
                } : { color: '#7a8ba4' }}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="text-xs" style={{ color: '#7a8ba4' }}>{filteredRows.length} assets x {models.length} models</span>
        </div>

        {/* Matrix table */}
        <div className="overflow-x-auto overflow-y-auto max-h-[600px]">
          <table className="text-[10px] border-collapse">
            <thead className="premium-thead">
              <tr>
                <th className="text-left sticky left-0 z-20 min-w-[70px]" style={{ background: 'rgba(26,5,51,0.97)' }}>Asset</th>
                <th className="text-center min-w-[20px]">PIT</th>
                {models.map(m => (
                  <th key={m} className="text-center whitespace-nowrap min-w-[50px]">
                    {formatModelNameShort(m)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.slice(0, 300).map(row => (
                <tr key={row.symbol} className="premium-row">
                  <td className="font-medium sticky left-0" style={{ color: 'var(--text-luminous)', background: 'rgba(10,10,26,0.95)' }}>{row.symbol}</td>
                  <td className="text-center">
                    <span className={row.ad_pass === true ? 'status-pill status-pass' : row.ad_pass === false ? 'status-pill status-fail' : ''}>
                      {row.ad_pass === true ? 'P' : row.ad_pass === false ? 'F' : '--'}
                    </span>
                  </td>
                  {models.map(m => {
                    const sc = row.scores[m];
                    const val = sc ? sc[metric] : null;
                    return (
                      <td key={m} className={`px-1 py-1 text-center ${getCellColor(val, m)}`}>
                        {fmtVal(val)}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}

/* ── Regime Distribution Tab ──────────────────────────────────────── */

const REGIME_COLORS: Record<string, string> = {
  LOW_VOL_TREND: 'var(--accent-emerald)',
  HIGH_VOL_TREND: 'var(--accent-amber)',
  LOW_VOL_RANGE: '#b49aff',
  HIGH_VOL_RANGE: 'var(--accent-orange)',
  CRISIS_JUMP: 'var(--accent-rose)',
  unknown: '#7a8ba4',
};

function RegimesTab({ data }: { data: { regimes: Record<string, { count: number; percentage: number; assets: string[] }>; total: number } }) {
  const pieData = Object.entries(data.regimes).map(([name, info]) => ({
    name,
    value: info.count,
    color: REGIME_COLORS[name] || '#7a8ba4',
  }));

  const [expandedRegime, setExpandedRegime] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Pie */}
      <div className="glass-card" style={{ padding: '24px' }}>
        <h3 className="premium-section-label mb-4">Regime Distribution</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }: { name?: string; percent?: number }) => `${(name ?? '').replace(/_/g, ' ')} ${((percent ?? 0) * 100).toFixed(0)}%`}>
              {pieData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ background: 'rgba(15,15,35,0.95)', border: '1px solid var(--violet-15)', borderRadius: 8, color: '#e2e8f0', fontSize: 12, backdropFilter: 'blur(12px)' }} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Regime cards */}
      <div className="md:col-span-2 space-y-3">
        {Object.entries(data.regimes).map(([name, info]) => (
          <div key={name} className="glass-card overflow-hidden">
            <button
              onClick={() => setExpandedRegime(expandedRegime === name ? null : name)}
              className="w-full px-4 py-3 flex items-center justify-between transition-all duration-200"
              style={{ background: expandedRegime === name ? 'var(--violet-4)' : undefined }}
            >
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 rounded-full" style={{ background: REGIME_COLORS[name] || '#7a8ba4' }} />
                <span className="text-sm font-medium" style={{ color: 'var(--text-luminous)' }}>{name.replace(/_/g, ' ')}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>{info.count} assets ({info.percentage}%)</span>
                {expandedRegime === name ? <ChevronDown className="w-4 h-4 text-[var(--text-secondary)]" /> : <ChevronRight className="w-4 h-4 text-[var(--text-secondary)]" />}
              </div>
            </button>
            {expandedRegime === name && (
              <div className="px-4 pb-3 flex flex-wrap gap-1.5">
                {info.assets.map(sym => (
                  <span key={sym} className="px-2 py-0.5 rounded text-[10px]" style={{ background: 'var(--violet-6)', color: 'var(--text-secondary)' }}>{sym}</span>
                ))}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Calibration Failures Tab ─────────────────────────────────────── */

function FailuresTab({ data }: { data?: { failures: Array<Record<string, unknown>>; count: number; file_exists: boolean } | undefined }) {
  if (!data) return <LoadingSpinner text="Loading failures..." />;
  if (!data.file_exists) {
    return (
      <div className="glass-card p-6 text-center">
        <AlertTriangle className="w-8 h-8 mx-auto mb-3" style={{ color: 'var(--accent-amber)' }} />
        <p style={{ color: 'var(--text-secondary)' }}>No calibration_failures.json found.</p>
        <p className="text-xs mt-1" style={{ color: '#7a8ba4' }}>Run <code style={{ color: '#b49aff' }}>make tune</code> to generate calibration data.</p>
      </div>
    );
  }

  if (data.count === 0) {
    return (
      <div className="glass-card p-6 text-center">
        <CheckCircle className="w-8 h-8 mx-auto mb-3" style={{ color: 'var(--accent-emerald)' }} />
        <p className="font-medium" style={{ color: 'var(--text-luminous)' }}>All assets passing calibration</p>
        <p className="text-xs mt-1" style={{ color: '#7a8ba4' }}>No calibration failures detected.</p>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="p-4" style={{ borderBottom: '1px solid var(--violet-6)' }}>
        <span className="text-sm font-medium" style={{ color: 'var(--accent-amber)' }}>{data.count} assets failing calibration</span>
      </div>
      <div className="overflow-y-auto max-h-[500px]">
        <table className="w-full text-xs">
          <thead className="premium-thead">
            <tr>
              <th className="text-left">Symbol</th>
              <th className="text-left">Details</th>
            </tr>
          </thead>
          <tbody>
            {data.failures.map((f, i) => (
              <tr key={i} className="premium-row">
                <td className="font-medium" style={{ color: 'var(--accent-rose)' }}>{String(f.symbol || f.asset || `Entry ${i + 1}`)}</td>
                <td className="px-3 py-2">
                  <pre className="text-[10px] whitespace-pre-wrap max-w-lg p-3 rounded-xl font-mono"
                    style={{ background: 'rgba(10,10,26,0.6)', color: 'var(--text-secondary)', border: '1px solid var(--violet-6)' }}>{JSON.stringify(f, null, 1)}</pre>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/* ── Helper Badges ────────────────────────────────────────────────── */

function GradeBadge({ grade }: { grade: string }) {
  const color =
    grade === 'A' ? 'text-[var(--accent-emerald)]' :
    grade === 'B' ? 'text-[#6ff0c0]' :
    grade === 'C' ? 'text-[var(--accent-amber)]' :
    grade === 'D' ? 'text-[var(--accent-orange)]' :
    grade === 'F' ? 'text-[var(--accent-rose)]' :
    'text-[var(--text-secondary)]';

  return <span className={`font-bold ${color}`}>{grade}</span>;
}

function RegimeBadge({ regime }: { regime: string | null }) {
  if (!regime) return <span className="text-[var(--text-secondary)]">—</span>;
  const color = REGIME_COLORS[regime] || '#7a8ba4';
  const short = regime.replace('_', ' ').replace('VOL', 'V').replace('TREND', 'T').replace('RANGE', 'R');
  return <span className="text-[10px] font-medium" style={{ color }}>{short}</span>;
}
