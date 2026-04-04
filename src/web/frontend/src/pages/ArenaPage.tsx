import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import type { SafeStorageModel } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { ArenaEmpty } from '../components/CosmicEmptyState';
import { Swords, Trophy, FlaskConical, Target, RefreshCw, Check, X, AlertTriangle, ChevronDown } from 'lucide-react';

/* ── Hard gate thresholds ────────────────────────────────────────── */
const GATES = {
  final: { pass: 70, borderline: 63 },
  bic: { pass: -29000, borderline: -26000, cmp: 'lt' as const },
  crps: { pass: 0.02, borderline: 0.022, cmp: 'lt' as const },
  hyv: { pass: 1000, borderline: 1100, cmp: 'lt' as const },
  css: { pass: 0.65, borderline: 0.585 },
  fec: { pass: 0.75, borderline: 0.675 },
};

type GateKey = keyof typeof GATES;

function gateStatus(val: number | null | undefined, gate: GateKey): 'pass' | 'fail' | 'borderline' | 'unknown' {
  if (val == null) return 'unknown';
  const g = GATES[gate];
  const isLt = 'cmp' in g && g.cmp === 'lt';
  const passes = isLt ? val < g.pass : val >= g.pass;
  if (passes) return 'pass';
  const nearThreshold = isLt ? val < g.borderline : val >= g.borderline;
  if (nearThreshold) return 'borderline';
  return 'fail';
}

export default function ArenaPage() {
  const statusQ = useQuery({ queryKey: ['arenaStatus'], queryFn: api.arenaStatus });
  const safeQ = useQuery({ queryKey: ['arenaSafeStorage'], queryFn: api.arenaSafeStorage });

  if (statusQ.isLoading) return <LoadingSpinner text="Loading arena..." />;
  if (statusQ.error) return <CosmicErrorCard title="Unable to load arena" error={statusQ.error as Error} onRetry={() => statusQ.refetch()} />;

  const status = statusQ.data;
  const models = safeQ.data?.models || [];
  const scoredModels = models.filter(m => m.has_scores);

  return (
    <>
      <PageHeader
        title="Arena"
        action={
          <button
            onClick={() => { statusQ.refetch(); safeQ.refetch(); }}
            disabled={safeQ.isFetching}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl text-sm transition-all duration-200 disabled:opacity-50"
            style={{
              background: 'var(--violet-8)',
              color: '#b49aff',
              border: '1px solid var(--violet-12)',
            }}
          >
            <RefreshCw className={`w-3.5 h-3.5 ${safeQ.isFetching ? 'animate-spin' : ''}`} />
            Refresh
          </button>
        }
      >
        Model competition sandbox — experimental vs production baselines
      </PageHeader>

      {/* Stats */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-5 mb-8 fade-up">
          <StatCard
            title="Safe Storage"
            value={status.safe_storage_count}
            subtitle="Promoted models"
            icon={<Trophy className="w-5 h-5" />}
            color="green"
          />
          <StatCard
            title="Experimental"
            value={status.experimental_count}
            subtitle="Active experiments"
            icon={<FlaskConical className="w-5 h-5" />}
            color="amber"
          />
          <StatCard
            title="Benchmark"
            value={status.benchmark_symbols.length}
            subtitle="Test symbols"
            icon={<Target className="w-5 h-5" />}
            color="blue"
          />
        </div>
      )}

      {/* Benchmark symbols */}
      {status && (
        <div className="glass-card mb-8 hover-lift" style={{ padding: '20px' }}>
          <h3 className="premium-section-label mb-4 flex items-center gap-2">
            <Swords className="w-4 h-4" style={{ color: 'var(--accent-violet)' }} /> Benchmark Universe
          </h3>
          <div className="flex flex-wrap gap-2">
            {status.benchmark_symbols.map((s) => (
              <span
                key={s}
                className="px-2.5 py-1 rounded-xl text-xs font-medium transition-colors cursor-default"
                style={{
                  background: 'var(--violet-6)',
                  color: '#b49aff',
                  border: '1px solid var(--violet-10)',
                }}
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Safe storage models with scoring */}
      <div className="glass-card overflow-hidden fade-up-delay-1">
        <div className="px-5 py-4 flex items-center justify-between" style={{ borderBottom: '1px solid var(--violet-6)' }}>
          <h3 className="premium-section-label">Safe Storage Models</h3>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
            {scoredModels.length} scored / {models.length} total
          </span>
        </div>
        {models.length === 0 ? (
          <div className="p-6 text-center text-[var(--text-secondary)] text-sm">No models in safe storage</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="premium-table">
              <thead className="premium-thead">
                <tr>
                  <th className="text-left">#</th>
                  <th className="text-left">Model Name</th>
                  <th className="text-right">Final</th>
                  <th className="text-right">BIC</th>
                  <th className="text-right">CRPS</th>
                  <th className="text-right">Hyv</th>
                  <th className="text-center">PIT</th>
                  <th className="text-right">CSS</th>
                  <th className="text-right">FEC</th>
                  <th className="text-right">Time</th>
                  <th className="text-right">Size</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <ModelRow key={m.name} model={m} rank={i + 1} index={i} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Hard gates reference */}
      <div className="glass-card mt-8 hover-lift" style={{ padding: '24px' }}>
        <h3 className="premium-section-label mb-4">Hard Gates (Promotion Criteria)</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          {[
            { gate: 'CSS >= 0.65', desc: 'Calibration stability under stress' },
            { gate: 'FEC >= 0.75', desc: 'Forecast entropy consistency' },
            { gate: 'Hyv < 1000', desc: 'Prevent variance collapse' },
            { gate: 'vs STD >= 3', desc: 'Beat best standard by 3+ pts' },
            { gate: 'PIT >= 75%', desc: 'Distributional correctness' },
            { gate: 'Final > 70', desc: 'Combined score threshold' },
            { gate: 'BIC < -29k', desc: 'Bayesian complexity penalty' },
            { gate: 'CRPS < 0.020', desc: 'Calibration + sharpness' },
          ].map((g) => (
            <div key={g.gate} className="rounded-xl p-2.5" style={{ background: 'rgba(10,10,26,0.6)', border: '1px solid var(--violet-6)' }}>
              <p className="font-mono font-bold" style={{ color: 'var(--accent-amber)' }}>{g.gate}</p>
              <p className="mt-0.5" style={{ color: '#7a8ba4' }}>{g.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

/* ── Gate Badge Component ─────────────────────────────────────────── */
function GateBadge({ val, gate, label }: { val: number | null | undefined; gate: GateKey; label: string }) {
  const status = gateStatus(val, gate);
  if (status === 'unknown') return <span className="text-caption">{label}: --</span>;
  const cls = status === 'pass' ? 'gate-pass' : status === 'borderline' ? 'gate-borderline' : 'gate-fail';
  const Icon = status === 'pass' ? Check : status === 'borderline' ? AlertTriangle : X;
  return (
    <span className={cls}>
      <Icon className="gate-icon" />
      {label}
    </span>
  );
}

/* ── Model Row ───────────────────────────────────────────────────── */

function ModelRow({ model: m, rank, index }: { model: SafeStorageModel; rank: number; index: number }) {
  const [expanded, setExpanded] = useState(false);
  const rankCls = rank === 1 ? 'rank-badge rank-gold' : rank === 2 ? 'rank-badge rank-silver' : rank === 3 ? 'rank-badge rank-bronze' : 'rank-badge rank-default';
  const isChampion = rank === 1 && m.has_scores;

  if (!m.has_scores) {
    return (
      <tr
        className={`premium-row arena-row-enter ${isChampion ? 'champion-pulse' : ''}`}
        style={{ animationDelay: `${250 + index * 30}ms` }}
      >
        <td><span className={rankCls}>{rank}</span></td>
        <td className="font-medium" style={{ color: 'var(--text-luminous)' }}>{formatName(m.name)}</td>
        <td colSpan={8} className="italic" style={{ color: 'var(--text-muted)' }}>No scoring data</td>
        <td className="text-right" style={{ color: 'var(--text-muted)' }}>{m.size_kb} KB</td>
      </tr>
    );
  }

  return (
    <>
      <tr
        className={`premium-row arena-row-enter cursor-pointer ${isChampion ? 'champion-pulse' : ''}`}
        style={{ animationDelay: `${250 + index * 30}ms` }}
        onClick={() => setExpanded(e => !e)}
      >
        <td>
          <div className="flex items-center gap-1.5">
            {isChampion && (
              <svg className="gate-icon" viewBox="0 0 14 14" fill="none" style={{ color: 'var(--accent-amber)' }}>
                <path d="M7 1L9 5L13 5.5L10 8.5L11 13L7 10.5L3 13L4 8.5L1 5.5L5 5L7 1Z" fill="currentColor" />
              </svg>
            )}
            <span className={rankCls}>{rank}</span>
          </div>
        </td>
        <td>
          <span className="font-medium" style={{ color: 'var(--text-luminous)' }}>{formatName(m.name)}</span>
        </td>
        <td className="text-right">
          <GateBadge val={m.final} gate="final" label={m.final?.toFixed(1) ?? '--'} />
        </td>
        <td className="text-right">
          <GateBadge val={m.bic} gate="bic" label={m.bic != null ? `${(m.bic / 1000).toFixed(1)}k` : '--'} />
        </td>
        <td className="text-right">
          <GateBadge val={m.crps} gate="crps" label={m.crps?.toFixed(4) ?? '--'} />
        </td>
        <td className="text-right">
          <GateBadge val={m.hyv} gate="hyv" label={m.hyv?.toFixed(0) ?? '--'} />
        </td>
        <td className="text-center">
          <span className={m.pit === 'PASS' ? 'gate-pass' : 'gate-fail'}>
            {m.pit === 'PASS' ? <Check className="gate-icon" /> : <X className="gate-icon" />}
            {m.pit ?? '--'}
          </span>
        </td>
        <td className="text-right">
          <GateBadge val={m.css} gate="css" label={m.css?.toFixed(2) ?? '--'} />
        </td>
        <td className="text-right">
          <GateBadge val={m.fec} gate="fec" label={m.fec?.toFixed(2) ?? '--'} />
        </td>
        <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
          {m.time_ms != null ? `${(m.time_ms / 1000).toFixed(1)}s` : '--'}
        </td>
        <td className="text-right">
          <div className="flex items-center justify-end gap-1">
            <span style={{ color: 'var(--text-muted)' }}>{m.size_kb} KB</span>
            <ChevronDown
              className="w-3 h-3 transition-transform duration-200"
              style={{
                color: 'var(--text-muted)',
                transform: expanded ? 'rotate(180deg)' : 'rotate(0deg)',
              }}
            />
          </div>
        </td>
      </tr>
      {/* S6.2: Expandable detail panel */}
      <tr>
        <td colSpan={11} style={{ padding: 0 }}>
          <div className={`model-detail-panel ${expanded ? 'is-open' : ''}`}>
            <div className="model-detail-inner">
              <div className="model-stat-grid">
                {([
                  { label: 'Final Score', value: m.final?.toFixed(1), gate: 'final' as GateKey },
                  { label: 'BIC', value: m.bic != null ? `${(m.bic / 1000).toFixed(1)}k` : null, gate: 'bic' as GateKey },
                  { label: 'CRPS', value: m.crps?.toFixed(4), gate: 'crps' as GateKey },
                  { label: 'Hyvarinen', value: m.hyv?.toFixed(0), gate: 'hyv' as GateKey },
                  { label: 'CSS', value: m.css?.toFixed(2), gate: 'css' as GateKey },
                  { label: 'FEC', value: m.fec?.toFixed(2), gate: 'fec' as GateKey },
                  { label: 'PIT', value: m.pit ?? null, gate: undefined },
                ]).map(s => {
                  const status = s.gate ? gateStatus(
                    s.gate === 'bic' ? m.bic : s.gate === 'crps' ? m.crps : s.gate === 'hyv' ? m.hyv : s.gate === 'css' ? m.css : s.gate === 'fec' ? m.fec : m.final,
                    s.gate
                  ) : (m.pit === 'PASS' ? 'pass' : 'fail');
                  const valColor = status === 'pass' ? 'var(--accent-emerald)' : status === 'borderline' ? 'var(--accent-amber)' : status === 'fail' ? 'var(--accent-rose)' : 'var(--text-secondary)';
                  return (
                    <div key={s.label} className="model-stat-tile">
                      <div className="stat-label">{s.label}</div>
                      <div className="stat-value" style={{ color: valColor }}>{s.value ?? '--'}</div>
                    </div>
                  );
                })}
              </div>
              {m.time_ms != null && (
                <div className="mt-3 text-caption" style={{ color: 'var(--text-muted)' }}>
                  Fit time: {(m.time_ms / 1000).toFixed(1)}s &middot; File size: {m.size_kb} KB
                </div>
              )}
            </div>
          </div>
        </td>
      </tr>
    </>
  );
}

/** Humanize model names: underscores → spaces, title case */
function formatName(raw: string): string {
  return raw
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}
