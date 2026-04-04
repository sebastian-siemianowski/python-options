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
    <div className="glass-card hover-lift p-6">
      <h3 className="premium-section-label mb-5">Model Leaderboard</h3>

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
              className="premium-row transition-all cursor-pointer"
              style={{
                opacity: visible ? 1 : 0,
                transform: visible ? 'translateX(0)' : 'translateX(-8px)',
                transition: `opacity 250ms cubic-bezier(0.2, 0, 0, 1) ${i * 50}ms, transform 250ms cubic-bezier(0.2, 0, 0, 1) ${i * 50}ms`,
              }}
              onClick={() => setExpandedModel(isExpanded ? null : model.name)}
            >
              <div
                className="flex items-center gap-3 px-3 py-2 rounded-lg"
              >
                {/* Rank badge */}
                {isMedal ? (
                  <div className={`rank-badge ${model.rank === 1 ? 'rank-gold' : model.rank === 2 ? 'rank-silver' : 'rank-bronze'}`}>
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

                {/* BMA weight bar (animated grow) */}
                <div className="flex items-center gap-2 flex-shrink-0">
                  <div className="bma-bar-track">
                    <div
                      className="bma-bar-fill"
                      style={{ width: visible ? `${barWidth}%` : '0%' }}
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
