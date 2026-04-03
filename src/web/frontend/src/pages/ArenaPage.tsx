import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import type { SafeStorageModel } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { Swords, Trophy, FlaskConical, Target, RefreshCw } from 'lucide-react';

/* ── Hard gate thresholds ────────────────────────────────────────── */
const GATES = {
  final: { pass: 70 },
  bic: { pass: -29000, cmp: 'lt' as const },
  crps: { pass: 0.02, cmp: 'lt' as const },
  hyv: { pass: 1000, cmp: 'lt' as const },
  css: { pass: 0.65 },
  fec: { pass: 0.75 },
};

function gateColor(val: number | null | undefined, gate: keyof typeof GATES): string {
  if (val == null) return 'text-[#64748b]';
  const g = GATES[gate];
  if ('cmp' in g && g.cmp === 'lt') return val < g.pass ? 'text-[#00E676]' : 'text-[#FF1744]';
  return val >= g.pass ? 'text-[#00E676]' : 'text-[#FF1744]';
}

export default function ArenaPage() {
  const statusQ = useQuery({ queryKey: ['arenaStatus'], queryFn: api.arenaStatus });
  const safeQ = useQuery({ queryKey: ['arenaSafeStorage'], queryFn: api.arenaSafeStorage });

  if (statusQ.isLoading) return <LoadingSpinner text="Loading arena..." />;

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
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[#16213e] text-sm text-[#42A5F5] hover:bg-[#1a2744] border border-[#2a2a4a] transition disabled:opacity-50"
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
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6 fade-up">
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
        <div className="glass-card p-4 mb-6 hover-lift">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3 flex items-center gap-2">
            <Swords className="w-4 h-4" /> Benchmark Universe
          </h3>
          <div className="flex flex-wrap gap-2">
            {status.benchmark_symbols.map((s) => (
              <span
                key={s}
                className="px-2.5 py-1 rounded-lg bg-[#16213e] text-xs font-medium text-[#42A5F5] border border-[#2a2a4a] hover:bg-[#1a2744] transition-colors cursor-default"
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Safe storage models with scoring */}
      <div className="glass-card overflow-hidden fade-up-delay-1">
        <div className="px-4 py-3 border-b border-[#2a2a4a] flex items-center justify-between">
          <h3 className="text-sm font-medium text-[#94a3b8]">Safe Storage Models</h3>
          <span className="text-xs text-[#64748b]">
            {scoredModels.length} scored / {models.length} total
          </span>
        </div>
        {models.length === 0 ? (
          <div className="p-6 text-center text-[#64748b] text-sm">No models in safe storage</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#2a2a4a]">
                  <th className="text-left px-3 py-2 text-[#64748b]">#</th>
                  <th className="text-left px-3 py-2 text-[#64748b]">Model Name</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Final</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">BIC</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">CRPS</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Hyv</th>
                  <th className="text-center px-3 py-2 text-[#64748b]">PIT</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">CSS</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">FEC</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Time</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Size</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m, i) => (
                  <ModelRow key={m.name} model={m} rank={i + 1} />
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Hard gates reference */}
      <div className="glass-card p-5 mt-6 hover-lift">
        <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Hard Gates (Promotion Criteria)</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-xs">
          {[
            { gate: 'CSS ≥ 0.65', desc: 'Calibration stability under stress' },
            { gate: 'FEC ≥ 0.75', desc: 'Forecast entropy consistency' },
            { gate: 'Hyv < 1000', desc: 'Prevent variance collapse' },
            { gate: 'vs STD ≥ 3', desc: 'Beat best standard by 3+ pts' },
            { gate: 'PIT ≥ 75%', desc: 'Distributional correctness' },
            { gate: 'Final > 70', desc: 'Combined score threshold' },
            { gate: 'BIC < -29k', desc: 'Bayesian complexity penalty' },
            { gate: 'CRPS < 0.020', desc: 'Calibration + sharpness' },
          ].map((g) => (
            <div key={g.gate} className="bg-[#0f0f23] rounded-lg p-2.5">
              <p className="font-mono font-bold text-[#FFB300]">{g.gate}</p>
              <p className="text-[#64748b] mt-0.5">{g.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </>
  );
}

/* ── Model Row ───────────────────────────────────────────────────── */

function ModelRow({ model: m, rank }: { model: SafeStorageModel; rank: number }) {
  if (!m.has_scores) {
    return (
      <tr className="border-b border-[#2a2a4a]/50 row-glow transition">
        <td className="px-3 py-2.5 text-[#64748b]">{rank}</td>
        <td className="px-3 py-2.5 font-medium text-[#e2e8f0]">{formatName(m.name)}</td>
        <td colSpan={8} className="px-3 py-2.5 text-[#64748b] italic">No scoring data</td>
        <td className="px-3 py-2.5 text-right text-[#64748b]">{m.size_kb} KB</td>
      </tr>
    );
  }

  return (
    <tr className="border-b border-[#2a2a4a]/50 row-glow transition">
      <td className="px-3 py-2.5 text-[#64748b]">{rank}</td>
      <td className="px-3 py-2.5">
        <span className="font-medium text-[#e2e8f0]">{formatName(m.name)}</span>
      </td>
      <td className={`px-3 py-2.5 text-right font-bold ${gateColor(m.final, 'final')}`}>
        {m.final?.toFixed(1) ?? '—'}
      </td>
      <td className={`px-3 py-2.5 text-right ${gateColor(m.bic, 'bic')}`}>
        {m.bic != null ? `${(m.bic / 1000).toFixed(1)}k` : '—'}
      </td>
      <td className={`px-3 py-2.5 text-right ${gateColor(m.crps, 'crps')}`}>
        {m.crps?.toFixed(4) ?? '—'}
      </td>
      <td className={`px-3 py-2.5 text-right ${gateColor(m.hyv, 'hyv')}`}>
        {m.hyv?.toFixed(0) ?? '—'}
      </td>
      <td className="px-3 py-2.5 text-center">
        <span className={m.pit === 'PASS' ? 'text-[#00E676] font-bold' : 'text-[#FF1744]'}>
          {m.pit ?? '—'}
        </span>
      </td>
      <td className={`px-3 py-2.5 text-right ${gateColor(m.css, 'css')}`}>
        {m.css?.toFixed(2) ?? '—'}
      </td>
      <td className={`px-3 py-2.5 text-right ${gateColor(m.fec, 'fec')}`}>
        {m.fec?.toFixed(2) ?? '—'}
      </td>
      <td className="px-3 py-2.5 text-right text-[#94a3b8]">
        {m.time_ms != null ? `${(m.time_ms / 1000).toFixed(1)}s` : '—'}
      </td>
      <td className="px-3 py-2.5 text-right text-[#64748b]">{m.size_kb} KB</td>
    </tr>
  );
}

/** Humanize model names: underscores → spaces, title case */
function formatName(raw: string): string {
  return raw
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase());
}
