import { useQuery } from '@tanstack/react-query';
import { api } from '../api';
import PageHeader from '../components/PageHeader';
import StatCard from '../components/StatCard';
import LoadingSpinner from '../components/LoadingSpinner';
import { Swords, Trophy, FlaskConical, Target } from 'lucide-react';

export default function ArenaPage() {
  const statusQ = useQuery({ queryKey: ['arenaStatus'], queryFn: api.arenaStatus });
  const safeQ = useQuery({ queryKey: ['arenaSafeStorage'], queryFn: api.arenaSafeStorage });

  if (statusQ.isLoading) return <LoadingSpinner text="Loading arena..." />;

  const status = statusQ.data;
  const models = safeQ.data?.models || [];

  return (
    <>
      <PageHeader title="Arena">
        Model competition sandbox — experimental vs production baselines
      </PageHeader>

      {/* Stats */}
      {status && (
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4 mb-6">
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
        <div className="glass-card p-4 mb-6">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3 flex items-center gap-2">
            <Swords className="w-4 h-4" /> Benchmark Universe
          </h3>
          <div className="flex flex-wrap gap-2">
            {status.benchmark_symbols.map((s) => (
              <span
                key={s}
                className="px-2.5 py-1 rounded-lg bg-[#16213e] text-xs font-medium text-[#42A5F5] border border-[#2a2a4a]"
              >
                {s}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Safe storage models */}
      <div className="glass-card overflow-hidden">
        <div className="px-4 py-3 border-b border-[#2a2a4a]">
          <h3 className="text-sm font-medium text-[#94a3b8]">Safe Storage Models</h3>
        </div>
        {models.length === 0 ? (
          <div className="p-6 text-center text-[#64748b] text-sm">No models in safe storage</div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-[#2a2a4a]">
                  <th className="text-left px-4 py-2 text-[#64748b]">Model Name</th>
                  <th className="text-left px-3 py-2 text-[#64748b]">File</th>
                  <th className="text-right px-3 py-2 text-[#64748b]">Size</th>
                </tr>
              </thead>
              <tbody>
                {models.map((m) => (
                  <tr key={m.name} className="border-b border-[#2a2a4a]/50 hover:bg-[#16213e]/30 transition">
                    <td className="px-4 py-2.5 font-medium text-[#e2e8f0]">{m.name}</td>
                    <td className="px-3 py-2.5 text-[#94a3b8]">{m.filename}</td>
                    <td className="px-3 py-2.5 text-right text-[#64748b]">{m.size_kb} KB</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Hard gates reference */}
      <div className="glass-card p-5 mt-6">
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
