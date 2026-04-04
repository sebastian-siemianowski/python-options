/**
 * ModelLeaderboard -- Cosmic leaderboard replacing the old bar chart.
 *
 * Top 10 models ranked by BMA selection frequency with gold/silver/bronze
 * gradient badges, expand-on-click details, and cascade entry animation.
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { formatModelNameShort } from '../utils/modelNames';
import { ArrowRight } from 'lucide-react';

interface Props {
  modelsDistribution: Record<string, number>;
  pitPass?: number;
  pitFail?: number;
  total?: number;
}

interface ModelRow {
  name: string;
  displayName: string;
  count: number;
  rank: number;
}

const MEDAL_GRADIENTS: Record<number, string> = {
  1: 'linear-gradient(135deg, var(--accent-amber) 0%, #D97706 100%)',
  2: 'linear-gradient(135deg, #94A3B8 0%, #64748B 100%)',
  3: 'linear-gradient(135deg, #FB923C 0%, #C2410C 100%)',
};

const MEDAL_GLOW: Record<number, string> = {
  1: 'var(--amber-12)',
  2: 'rgba(148,163,184,0.08)',
  3: 'rgba(251,146,60,0.10)',
};

export default function ModelLeaderboard({ modelsDistribution, pitPass, pitFail, total }: Props) {
  const navigate = useNavigate();
  const [expandedModel, setExpandedModel] = useState<string | null>(null);
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    const raf = requestAnimationFrame(() => setVisible(true));
    return () => cancelAnimationFrame(raf);
  }, []);

  const models: ModelRow[] = Object.entries(modelsDistribution || {})
    .map(([name, count]) => ({
      name,
      displayName: formatModelNameShort(name),
      count,
      rank: 0,
    }))
    .sort((a, b) => b.count - a.count)
    .slice(0, 10)
    .map((m, i) => ({ ...m, rank: i + 1 }));

  const maxCount = models.length > 0 ? models[0].count : 1;

  // Estimate pass rate per model (rough: proportional to overall)
  const overallPassRate = total && total > 0 && pitPass != null
    ? pitPass / total
    : null;

  return (
    <div className="glass-card hover-lift" style={{ padding: '24px' }}>
      <h3
        className="text-[11px] font-semibold uppercase tracking-wide mb-5"
        style={{ color: 'var(--text-muted)', letterSpacing: '0.12em' }}
      >
        Model Leaderboard
      </h3>

      <div className="space-y-0">
        {models.map((model, i) => {
          const isExpanded = expandedModel === model.name;
          const isMedal = model.rank <= 3;
          const barWidth = (model.count / maxCount) * 100;
          const avgWeight = total && total > 0
            ? ((model.count / total) * 100).toFixed(1)
            : '?';

          return (
            <div
              key={model.name}
              className="transition-all cursor-pointer"
              style={{
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateX(0)' : 'translateX(-8px)',
                transition: `opacity 250ms cubic-bezier(0.2, 0, 0, 1) ${i * 50}ms, transform 250ms cubic-bezier(0.2, 0, 0, 1) ${i * 50}ms`,
              }}
              onClick={() => setExpandedModel(isExpanded ? null : model.name)}
            >
              <div
                className="flex items-center gap-3 px-3 py-2 rounded-lg transition-all duration-120"
                style={{
                  background: isExpanded
                    ? 'var(--void-hover, #16133a)'
                    : 'transparent',
                  ...(isMedal && !isExpanded ? {
                    background: `radial-gradient(circle at 0% 50%, ${MEDAL_GLOW[model.rank]} 0%, transparent 60%)`,
                  } : {}),
                }}
                onMouseEnter={(e) => {
                  if (!isExpanded) {
                    (e.currentTarget as HTMLElement).style.background = 'var(--void-hover, #16133a)';
                    (e.currentTarget as HTMLElement).style.boxShadow = '0 0 0 1px var(--violet-8), 0 4px 16px var(--violet-8)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (!isExpanded) {
                    (e.currentTarget as HTMLElement).style.background = isMedal
                      ? `radial-gradient(circle at 0% 50%, ${MEDAL_GLOW[model.rank]} 0%, transparent 60%)`
                      : 'transparent';
                    (e.currentTarget as HTMLElement).style.boxShadow = 'none';
                  }
                }}
              >
                {/* Rank badge */}
                {isMedal ? (
                  <div
                    className="w-7 h-7 rounded-full flex items-center justify-center flex-shrink-0 text-[11px] font-bold"
                    style={{
                      background: MEDAL_GRADIENTS[model.rank],
                      color: model.rank === 2 ? '#1e293b' : '#000',
                    }}
                  >
                    {model.rank}
                  </div>
                ) : (
                  <div className="w-7 h-7 flex items-center justify-center flex-shrink-0 text-[11px]"
                    style={{ color: 'var(--text-muted, #6b7a90)' }}>
                    {model.rank}
                  </div>
                )}

                {/* Model name */}
                <span
                  className="text-xs truncate flex-1 min-w-0"
                  title={model.name}
                  style={{ color: 'var(--text-primary, #e2e8f0)' }}
                >
                  {model.displayName}
                </span>

                {/* Count with gradient fill bar */}
                <div className="flex items-center gap-2 flex-shrink-0">
                  <div className="relative w-16 h-1.5 rounded-full overflow-hidden"
                    style={{ background: 'var(--violet-6)' }}>
                    <div
                      className="absolute left-0 top-0 h-full rounded-full"
                      style={{
                        width: `${barWidth}%`,
                        background: 'linear-gradient(90deg, var(--violet-15) 0%, var(--violet-3) 100%)',
                        transition: 'width 300ms ease',
                      }}
                    />
                  </div>
                  <span className="text-[10px] tabular-nums font-medium w-6 text-right"
                    style={{ color: 'var(--text-primary, #e2e8f0)' }}>
                    {model.count}
                  </span>
                </div>

                {/* BMA weight */}
                <span className="text-[10px] tabular-nums flex-shrink-0 w-10 text-right"
                  style={{ color: 'var(--text-secondary, #94a3b8)', fontFamily: 'SF Mono, ui-monospace, monospace' }}>
                  {avgWeight}%
                </span>

                {/* PIT badge */}
                {overallPassRate != null && (
                  <span
                    className="text-[9px] px-1.5 py-0.5 rounded flex-shrink-0"
                    style={{
                      background: overallPassRate >= 0.8
                        ? 'var(--emerald-12)'
                        : overallPassRate >= 0.6
                          ? 'var(--amber-12)'
                          : 'var(--rose-12)',
                      color: overallPassRate >= 0.8
                        ? 'var(--accent-emerald)'
                        : overallPassRate >= 0.6
                          ? 'var(--accent-amber)'
                          : 'var(--accent-rose)',
                    }}
                  >
                    {Math.round(overallPassRate * 100)}%
                  </span>
                )}
              </div>

              {/* Expanded detail */}
              <div
                className="overflow-hidden transition-all duration-250"
                style={{
                  maxHeight: isExpanded ? '120px' : '0px',
                  opacity: isExpanded ? 1 : 0,
                }}
              >
                <div className="px-3 pb-3 pt-1 ml-10 text-[10px] space-y-1.5">
                  <p style={{ color: 'var(--text-muted, #6b7a90)' }}>
                    Selected {model.count}x across {total || '?'} assets
                    ({total && total > 0 ? `${((model.count / total) * 100).toFixed(1)}%` : '?'} of universe)
                  </p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* View All Models link */}
      <button
        className="flex items-center gap-1.5 mt-4 text-[11px] font-medium cursor-pointer bg-transparent border-none group"
        style={{ color: 'var(--accent-violet, #8B5CF6)' }}
        onClick={() => navigate('/diagnostics')}
      >
        <span className="relative">
          View All Models
          <span
            className="absolute bottom-0 left-0 w-0 h-[1px] group-hover:w-full transition-all duration-200"
            style={{ background: 'linear-gradient(90deg, #8B5CF6, #6366F1)' }}
          />
        </span>
        <ArrowRight className="w-3 h-3 group-hover:translate-x-0.5 transition-transform" />
      </button>
    </div>
  );
}
