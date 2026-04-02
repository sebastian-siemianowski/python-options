import { useQuery } from '@tanstack/react-query';
import { useState, useMemo } from 'react';
import { api } from '../api';
import type { DiagAsset, DiagModelStats, DiagCrossAssetSummary } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
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
  if (isLoading) return <LoadingSpinner text="Loading diagnostics..." />;

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
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[#16213e] text-sm text-[#42A5F5] hover:bg-[#1a2744] border border-[#2a2a4a] transition disabled:opacity-50"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${pitQ.isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        }
      >
        PIT calibration, model comparison, and regime analysis — equivalent to{' '}
        <code className="text-[#AB47BC]">make diag</code>
      </PageHeader>

      {/* Stats bar */}
      {pitData && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
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
      <div className="flex gap-1 mb-6 border-b border-[#2a2a4a]">
        {tabs.map(({ id, label, icon: Icon }) => (
          <button
            key={id}
            onClick={() => setTab(id)}
            className={`flex items-center gap-2 px-4 py-2.5 text-sm transition border-b-2 -mb-[1px] ${
              tab === id
                ? 'border-[#42A5F5] text-[#42A5F5]'
                : 'border-transparent text-[#64748b] hover:text-[#94a3b8]'
            }`}
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
      <div className="p-3 border-b border-[#2a2a4a] flex items-center gap-3">
        <div className="relative flex-1">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#64748b]" />
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search assets..."
            className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-[#0f0f23] border border-[#2a2a4a] text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none focus:border-[#42A5F5]"
          />
        </div>
        <div className="flex gap-1">
          {(['all', 'pass', 'fail'] as const).map(f => (
            <button
              key={f}
              onClick={() => setFilterPit(f)}
              className={`px-3 py-1.5 rounded text-xs font-medium transition ${
                filterPit === f
                  ? f === 'pass' ? 'bg-[#00E676]/20 text-[#00E676]'
                    : f === 'fail' ? 'bg-[#FF1744]/20 text-[#FF1744]'
                    : 'bg-[#42A5F5]/20 text-[#42A5F5]'
                  : 'text-[#64748b] hover:text-[#94a3b8]'
              }`}
            >
              {f === 'all' ? 'All' : f === 'pass' ? 'Pass' : 'Fail'}
            </button>
          ))}
        </div>
        <span className="text-xs text-[#64748b]">{assets.length} assets</span>
      </div>

      {/* Table */}
      <div className="overflow-y-auto max-h-[600px]">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[#1a1a2e] z-10">
            <tr className="border-b border-[#2a2a4a]">
              <th className="w-6 px-2"></th>
              <th className="text-left px-3 py-2 text-[#64748b]">Symbol</th>
              <th className="text-left px-3 py-2 text-[#64748b]">Best Model</th>
              <th className="text-center px-3 py-2 text-[#64748b]">PIT</th>
              <th className="text-center px-3 py-2 text-[#64748b]">AD Stat</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Grade</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Models</th>
              <th className="text-center px-3 py-2 text-[#64748b]">Regime</th>
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
  const pitColor = asset.ad_pass === true ? 'text-[#00E676]' : asset.ad_pass === false ? 'text-[#FF1744]' : 'text-[#64748b]';
  const pitIcon = asset.ad_pass === true ? '✓' : asset.ad_pass === false ? '✗' : '—';

  return (
    <>
      <tr
        onClick={onToggle}
        className={`border-b border-[#2a2a4a]/50 cursor-pointer transition hover:bg-[#16213e]/50 ${expanded ? 'bg-[#16213e]' : ''}`}
      >
        <td className="px-2 text-[#64748b]">
          {expanded ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
        </td>
        <td className="px-3 py-2 font-medium text-[#e2e8f0]">{asset.symbol}</td>
        <td className="px-3 py-2 text-[#94a3b8]">{formatModelNameShort(asset.best_model)}</td>
        <td className={`px-3 py-2 text-center font-bold ${pitColor}`}>{pitIcon}</td>
        <td className="px-3 py-2 text-center text-[#94a3b8]">{asset.ad_stat?.toFixed(3) ?? '—'}</td>
        <td className="px-3 py-2 text-center">
          <GradeBadge grade={asset.pit_grade} />
        </td>
        <td className="px-3 py-2 text-center text-[#94a3b8]">{asset.num_models}</td>
        <td className="px-3 py-2 text-center">
          <RegimeBadge regime={asset.regime} />
        </td>
      </tr>
      {expanded && asset.models?.length > 0 && (
        <tr>
          <td colSpan={8} className="px-4 py-3 bg-[#0a0a1a]">
            <ModelMetricsTable models={asset.models} bmaWeights={asset.bma_weights} />
          </td>
        </tr>
      )}
    </>
  );
}

function ModelMetricsTable({ models, bmaWeights }: { models: DiagAsset['models']; bmaWeights: Record<string, number> }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-[11px]">
        <thead>
          <tr className="border-b border-[#2a2a4a]">
            <th className="text-left px-2 py-1.5 text-[#64748b]">Model</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">BIC</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">CRPS</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">Hyv</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">PIT p</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">AD p</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">MAD</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">Weight</th>
            <th className="text-right px-2 py-1.5 text-[#64748b]">ν</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const w = bmaWeights[m.model] ?? m.weight;
            return (
              <tr key={m.model} className="border-b border-[#2a2a4a]/30 hover:bg-[#16213e]/30">
                <td className="px-2 py-1.5 text-[#94a3b8] font-medium">{formatModelNameShort(m.model)}</td>
                <td className="px-2 py-1.5 text-right text-[#94a3b8]">{m.bic != null ? m.bic.toFixed(0) : '—'}</td>
                <td className="px-2 py-1.5 text-right">
                  <span className={m.crps != null && m.crps < 0.02 ? 'text-[#00E676]' : 'text-[#94a3b8]'}>
                    {m.crps != null ? m.crps.toFixed(4) : '—'}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right">
                  <span className={m.hyvarinen != null && m.hyvarinen < 1000 ? 'text-[#00E676]' : m.hyvarinen != null && m.hyvarinen > 2000 ? 'text-[#FF1744]' : 'text-[#94a3b8]'}>
                    {m.hyvarinen != null ? m.hyvarinen.toFixed(0) : '—'}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right">
                  <span className={m.pit_ks_pvalue != null && m.pit_ks_pvalue >= 0.05 ? 'text-[#00E676]' : m.pit_ks_pvalue != null ? 'text-[#FF1744]' : 'text-[#64748b]'}>
                    {m.pit_ks_pvalue != null ? m.pit_ks_pvalue.toFixed(3) : '—'}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right">
                  <span className={m.ad_pvalue != null && m.ad_pvalue >= 0.05 ? 'text-[#00E676]' : m.ad_pvalue != null ? 'text-[#FF1744]' : 'text-[#64748b]'}>
                    {m.ad_pvalue != null ? m.ad_pvalue.toFixed(3) : '—'}
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right text-[#94a3b8]">{m.histogram_mad != null ? m.histogram_mad.toFixed(4) : '—'}</td>
                <td className="px-2 py-1.5 text-right">
                  <span className={w > 0.2 ? 'text-[#00E676] font-bold' : w > 0.05 ? 'text-[#FFB300]' : 'text-[#64748b]'}>
                    {(w * 100).toFixed(1)}%
                  </span>
                </td>
                <td className="px-2 py-1.5 text-right text-[#94a3b8]">{m.nu != null ? m.nu.toFixed(1) : '—'}</td>
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
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Win Count (Best Model Selection)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', borderRadius: 8, color: '#e2e8f0' }} />
              <Bar dataKey="wins" fill="#AB47BC" radius={[0, 4, 4, 0]} name="Wins" />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="glass-card p-5">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Avg BMA Weight (%)</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={winRateData} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" stroke="#2a2a4a" />
              <XAxis type="number" tick={{ fill: '#64748b', fontSize: 10 }} />
              <YAxis type="category" dataKey="name" width={120} tick={{ fill: '#94a3b8', fontSize: 10 }} />
              <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', borderRadius: 8, color: '#e2e8f0' }} />
              <Bar dataKey="avgWeight" fill="#42A5F5" radius={[0, 4, 4, 0]} name="Avg Weight %" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Model table */}
      <div className="glass-card overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-xs">
            <thead className="bg-[#1a1a2e]">
              <tr className="border-b border-[#2a2a4a]">
                <th className="text-left px-3 py-2 text-[#64748b]">Model</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Wins</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Win Rate</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Appearances</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Avg Wt</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Max Wt</th>
                <th className="text-right px-3 py-2 text-[#64748b]">Min Wt</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m.name} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/50">
                  <td className="px-3 py-2 font-medium text-[#e2e8f0]">{formatModelNameShort(m.name)}</td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{m.win_count}</td>
                  <td className="px-3 py-2 text-right">
                    <span className={m.win_rate > 0.1 ? 'text-[#00E676]' : 'text-[#94a3b8]'}>
                      {(m.win_rate * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{m.appearances}</td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{(m.avg_weight * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2 text-right text-[#94a3b8]">{(m.max_weight * 100).toFixed(1)}%</td>
                  <td className="px-3 py-2 text-right text-[#64748b]">{(m.min_weight * 100).toFixed(1)}%</td>
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
  if (!data || data.rows.length === 0) return <div className="glass-card p-6 text-center text-[#64748b]">No data available</div>;

  const models = data.models;
  const filteredRows = matrixSearch
    ? data.rows.filter(r => r.symbol.toLowerCase().includes(matrixSearch.toLowerCase()))
    : data.rows;

  const getCellColor = (val: number | null | undefined, _m: string): string => {
    if (val == null) return 'text-[#2a2a4a]';
    if (metric === 'crps') return val < 0.015 ? 'text-[#00E676]' : val < 0.025 ? 'text-[#94a3b8]' : 'text-[#FF7043]';
    if (metric === 'pit_ks_p') return val >= 0.05 ? 'text-[#00E676]' : 'text-[#FF1744]';
    if (metric === 'weight') return val > 0.2 ? 'text-[#00E676] font-bold' : val > 0.05 ? 'text-[#FFB300]' : 'text-[#64748b]';
    return 'text-[#94a3b8]';
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
        <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Model Averages (across all assets)</h3>
        <div className="overflow-x-auto">
          <table className="w-full text-[11px]">
            <thead>
              <tr className="border-b border-[#2a2a4a]">
                <th className="text-left px-2 py-1.5 text-[#64748b]">Model</th>
                <th className="text-right px-2 py-1.5 text-[#64748b]">Avg CRPS</th>
                <th className="text-right px-2 py-1.5 text-[#64748b]">Avg PIT p</th>
                <th className="text-right px-2 py-1.5 text-[#64748b]">Avg BIC</th>
                <th className="text-right px-2 py-1.5 text-[#64748b]">Assets</th>
              </tr>
            </thead>
            <tbody>
              {models.map(m => {
                const avg = data.model_averages[m];
                return (
                  <tr key={m} className="border-b border-[#2a2a4a]/30 hover:bg-[#16213e]/30">
                    <td className="px-2 py-1.5 text-[#e2e8f0] font-medium">{formatModelNameShort(m)}</td>
                    <td className={`px-2 py-1.5 text-right ${avg?.avg_crps != null && avg.avg_crps < 0.02 ? 'text-[#00E676]' : 'text-[#94a3b8]'}`}>
                      {avg?.avg_crps?.toFixed(5) ?? '—'}
                    </td>
                    <td className={`px-2 py-1.5 text-right ${avg?.avg_pit_p != null && avg.avg_pit_p >= 0.05 ? 'text-[#00E676]' : 'text-[#94a3b8]'}`}>
                      {avg?.avg_pit_p?.toFixed(4) ?? '—'}
                    </td>
                    <td className="px-2 py-1.5 text-right text-[#94a3b8]">{avg?.avg_bic?.toFixed(0) ?? '—'}</td>
                    <td className="px-2 py-1.5 text-right text-[#64748b]">{avg?.count ?? 0}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Matrix controls */}
      <div className="glass-card overflow-hidden">
        <div className="p-3 border-b border-[#2a2a4a] flex items-center gap-3">
          <div className="relative flex-1 max-w-xs">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-[#64748b]" />
            <input
              type="text"
              value={matrixSearch}
              onChange={(e) => setMatrixSearch(e.target.value)}
              placeholder="Search assets..."
              className="w-full pl-9 pr-3 py-1.5 rounded-lg bg-[#0f0f23] border border-[#2a2a4a] text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none focus:border-[#42A5F5]"
            />
          </div>
          <div className="flex gap-1">
            {([['crps', 'CRPS'], ['pit_ks_p', 'PIT p'], ['weight', 'Weight']] as const).map(([k, label]) => (
              <button
                key={k}
                onClick={() => setMetric(k)}
                className={`px-3 py-1.5 rounded text-xs font-medium transition ${
                  metric === k ? 'bg-[#42A5F5]/20 text-[#42A5F5]' : 'text-[#64748b] hover:text-[#94a3b8]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>
          <span className="text-xs text-[#64748b]">{filteredRows.length} assets × {models.length} models</span>
        </div>

        {/* Matrix table */}
        <div className="overflow-x-auto overflow-y-auto max-h-[600px]">
          <table className="text-[10px] border-collapse">
            <thead className="sticky top-0 bg-[#1a1a2e] z-10">
              <tr className="border-b border-[#2a2a4a]">
                <th className="text-left px-2 py-1.5 text-[#64748b] sticky left-0 bg-[#1a1a2e] z-20 min-w-[70px]">Asset</th>
                <th className="text-center px-1 py-1.5 text-[#64748b] min-w-[20px]">PIT</th>
                {models.map(m => (
                  <th key={m} className="text-center px-1 py-1.5 text-[#64748b] whitespace-nowrap min-w-[50px]">
                    {formatModelNameShort(m)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {filteredRows.slice(0, 300).map(row => (
                <tr key={row.symbol} className="border-b border-[#2a2a4a]/20 hover:bg-[#16213e]/30">
                  <td className="px-2 py-1 text-[#e2e8f0] font-medium sticky left-0 bg-[#0f0f23]">{row.symbol}</td>
                  <td className="px-1 py-1 text-center">
                    <span className={row.ad_pass === true ? 'text-[#00E676]' : row.ad_pass === false ? 'text-[#FF1744]' : 'text-[#64748b]'}>
                      {row.ad_pass === true ? '✓' : row.ad_pass === false ? '✗' : '—'}
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
  LOW_VOL_TREND: '#00E676',
  HIGH_VOL_TREND: '#FFB300',
  LOW_VOL_RANGE: '#42A5F5',
  HIGH_VOL_RANGE: '#FF7043',
  CRISIS_JUMP: '#FF1744',
  unknown: '#64748b',
};

function RegimesTab({ data }: { data: { regimes: Record<string, { count: number; percentage: number; assets: string[] }>; total: number } }) {
  const pieData = Object.entries(data.regimes).map(([name, info]) => ({
    name,
    value: info.count,
    color: REGIME_COLORS[name] || '#64748b',
  }));

  const [expandedRegime, setExpandedRegime] = useState<string | null>(null);

  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
      {/* Pie */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Regime Distribution</h3>
        <ResponsiveContainer width="100%" height={250}>
          <PieChart>
            <Pie data={pieData} dataKey="value" nameKey="name" cx="50%" cy="50%" outerRadius={90} label={({ name, percent }: { name?: string; percent?: number }) => `${(name ?? '').replace(/_/g, ' ')} ${((percent ?? 0) * 100).toFixed(0)}%`}>
              {pieData.map((entry) => (
                <Cell key={entry.name} fill={entry.color} />
              ))}
            </Pie>
            <Tooltip contentStyle={{ background: '#1a1a2e', border: '1px solid #2a2a4a', borderRadius: 8, color: '#e2e8f0', fontSize: 12 }} />
          </PieChart>
        </ResponsiveContainer>
      </div>

      {/* Regime cards */}
      <div className="md:col-span-2 space-y-3">
        {Object.entries(data.regimes).map(([name, info]) => (
          <div key={name} className="glass-card overflow-hidden">
            <button
              onClick={() => setExpandedRegime(expandedRegime === name ? null : name)}
              className="w-full px-4 py-3 flex items-center justify-between hover:bg-[#16213e]/50 transition"
            >
              <div className="flex items-center gap-3">
                <span className="w-3 h-3 rounded-full" style={{ background: REGIME_COLORS[name] || '#64748b' }} />
                <span className="text-sm font-medium text-[#e2e8f0]">{name.replace(/_/g, ' ')}</span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-[#94a3b8]">{info.count} assets ({info.percentage}%)</span>
                {expandedRegime === name ? <ChevronDown className="w-4 h-4 text-[#64748b]" /> : <ChevronRight className="w-4 h-4 text-[#64748b]" />}
              </div>
            </button>
            {expandedRegime === name && (
              <div className="px-4 pb-3 flex flex-wrap gap-1.5">
                {info.assets.map(sym => (
                  <span key={sym} className="px-2 py-0.5 rounded bg-[#16213e] text-[10px] text-[#94a3b8]">{sym}</span>
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
        <AlertTriangle className="w-8 h-8 text-[#FFB300] mx-auto mb-3" />
        <p className="text-[#94a3b8]">No calibration_failures.json found.</p>
        <p className="text-xs text-[#64748b] mt-1">Run <code className="text-[#AB47BC]">make tune</code> to generate calibration data.</p>
      </div>
    );
  }

  if (data.count === 0) {
    return (
      <div className="glass-card p-6 text-center">
        <CheckCircle className="w-8 h-8 text-[#00E676] mx-auto mb-3" />
        <p className="text-[#e2e8f0] font-medium">All assets passing calibration</p>
        <p className="text-xs text-[#64748b] mt-1">No calibration failures detected.</p>
      </div>
    );
  }

  return (
    <div className="glass-card overflow-hidden">
      <div className="p-3 border-b border-[#2a2a4a]">
        <span className="text-sm text-[#FFB300]">{data.count} assets failing calibration</span>
      </div>
      <div className="overflow-y-auto max-h-[500px]">
        <table className="w-full text-xs">
          <thead className="sticky top-0 bg-[#1a1a2e]">
            <tr className="border-b border-[#2a2a4a]">
              <th className="text-left px-3 py-2 text-[#64748b]">Symbol</th>
              <th className="text-left px-3 py-2 text-[#64748b]">Details</th>
            </tr>
          </thead>
          <tbody>
            {data.failures.map((f, i) => (
              <tr key={i} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/50">
                <td className="px-3 py-2 font-medium text-[#FF1744]">{String(f.symbol || f.asset || `Entry ${i + 1}`)}</td>
                <td className="px-3 py-2 text-[#94a3b8]">
                  <pre className="text-[10px] whitespace-pre-wrap max-w-lg">{JSON.stringify(f, null, 1)}</pre>
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
    grade === 'A' ? 'text-[#00E676]' :
    grade === 'B' ? 'text-[#66BB6A]' :
    grade === 'C' ? 'text-[#FFB300]' :
    grade === 'D' ? 'text-[#FF7043]' :
    grade === 'F' ? 'text-[#FF1744]' :
    'text-[#64748b]';

  return <span className={`font-bold ${color}`}>{grade}</span>;
}

function RegimeBadge({ regime }: { regime: string | null }) {
  if (!regime) return <span className="text-[#64748b]">—</span>;
  const color = REGIME_COLORS[regime] || '#64748b';
  const short = regime.replace('_', ' ').replace('VOL', 'V').replace('TREND', 'T').replace('RANGE', 'R');
  return <span className="text-[10px] font-medium" style={{ color }}>{short}</span>;
}
