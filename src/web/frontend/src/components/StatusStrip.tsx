import { useQuery } from '@tanstack/react-query';
import { api, type RiskSummary, type SignalStats, type ServicesHealth } from '../api';
import { useState, useMemo } from 'react';

/* ─── Regime thresholds ─────────────────────────────────────────── */
type Regime = 'calm' | 'elevated' | 'stressed' | 'crisis';

function classifyRegime(temperature: number): Regime {
  if (temperature < 0.3) return 'calm';
  if (temperature < 0.7) return 'elevated';
  if (temperature <= 1.2) return 'stressed';
  return 'crisis';
}

const REGIME_GRADIENTS: Record<Regime, { gradient: string; opacity: number; pulse: boolean }> = {
  calm: {
    gradient: 'linear-gradient(90deg, transparent 0%, var(--accent-violet) 20%, var(--accent-indigo) 50%, var(--accent-violet) 80%, transparent 100%)',
    opacity: 0.5,
    pulse: false,
  },
  elevated: {
    gradient: 'linear-gradient(90deg, transparent 0%, var(--accent-amber) 20%, var(--accent-violet) 50%, var(--accent-amber) 80%, transparent 100%)',
    opacity: 0.6,
    pulse: false,
  },
  stressed: {
    gradient: 'linear-gradient(90deg, transparent 0%, var(--accent-rose) 20%, var(--accent-amber) 50%, var(--accent-rose) 80%, transparent 100%)',
    opacity: 0.5,
    pulse: false,
  },
  crisis: {
    gradient: 'linear-gradient(90deg, transparent 0%, var(--accent-rose) 30%, #E11D48 50%, var(--accent-rose) 70%, transparent 100%)',
    opacity: 0.8,
    pulse: true,
  },
};

const REGIME_LABELS: Record<Regime, string> = {
  calm: 'Calm',
  elevated: 'Elevated',
  stressed: 'Stressed',
  crisis: 'Crisis',
};

/* ─── Component ─────────────────────────────────────────────────── */
export default function StatusStrip() {
  const [expanded, setExpanded] = useState(false);

  const riskQ = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 60_000,
    retry: false,
  });
  const signalQ = useQuery({
    queryKey: ['signalStats'],
    queryFn: api.signalStats,
    staleTime: 60_000,
    retry: false,
  });
  const healthQ = useQuery({
    queryKey: ['servicesHealth'],
    queryFn: api.servicesHealth,
    refetchInterval: 30_000,
    retry: false,
  });

  const risk = riskQ.data as RiskSummary | undefined;
  const signals = signalQ.data as SignalStats | undefined;
  const health = healthQ.data as ServicesHealth | undefined;

  const regime = useMemo(() => classifyRegime(risk?.combined_temperature ?? 0), [risk]);
  const cfg = REGIME_GRADIENTS[regime];

  const allServicesOk = health
    ? health.api.status === 'ok' && health.signal_cache.status !== 'missing' && health.price_data.status === 'ok'
    : true;

  return (
    <div
      className="fixed top-0 left-0 right-0 z-[90] transition-all"
      style={{
        height: expanded ? 36 : 3,
        transitionDuration: '300ms',
        transitionTimingFunction: 'cubic-bezier(0.2, 0, 0, 1)',
      }}
      onMouseEnter={() => setExpanded(true)}
      onMouseLeave={() => setExpanded(false)}
    >
      {/* Gradient strip */}
      <div
        className="absolute inset-0"
        style={{
          background: cfg.gradient,
          opacity: cfg.opacity,
          transition: 'background 1000ms, opacity 1000ms',
          animation: cfg.pulse ? 'strip-pulse 2s ease-in-out infinite' : 'none',
        }}
      />

      {/* Expanded content */}
      {expanded && (
        <div
          className="absolute inset-0 flex items-center justify-center gap-6 px-6 text-[11px]"
          style={{
            background: 'var(--gradient-nebula)',
            backdropFilter: 'blur(16px) saturate(1.4)',
            WebkitBackdropFilter: 'blur(16px) saturate(1.4)',
            animation: 'strip-content-in 200ms cubic-bezier(0.2,0,0,1)',
          }}
        >
          {/* Risk temperature */}
          <span className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            Risk:
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              {risk ? risk.combined_temperature.toFixed(2) : '--'}
            </span>
            <span style={{ color: regime === 'calm' ? 'var(--accent-emerald)' : regime === 'elevated' ? 'var(--accent-amber)' : 'var(--accent-rose)' }}>
              {REGIME_LABELS[regime]}
            </span>
          </span>

          <span className="w-px h-3.5" style={{ background: 'var(--border-void)' }} />

          {/* Assets count */}
          <span style={{ color: 'var(--text-secondary)' }}>
            <span className="font-semibold" style={{ color: 'var(--text-primary)' }}>
              {signals?.total_assets ?? '--'}
            </span>{' '}Assets
          </span>

          <span className="w-px h-3.5" style={{ background: 'var(--border-void)' }} />

          {/* Strong signals */}
          <span style={{ color: 'var(--text-secondary)' }}>
            <span className="font-semibold" style={{ color: 'var(--accent-emerald)' }}>
              {signals?.strong_buy_signals ?? '--'}
            </span>{' '}Strong Buys
          </span>

          <span className="w-px h-3.5" style={{ background: 'var(--border-void)' }} />

          {/* Services */}
          <span className="flex items-center gap-1.5" style={{ color: 'var(--text-secondary)' }}>
            Services:
            <span
              className="w-1.5 h-1.5 rounded-full pulse-dot"
              style={{
                background: allServicesOk ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                boxShadow: `0 0 6px ${allServicesOk ? 'rgba(62,232,165,0.45)' : 'rgba(255,107,138,0.45)'}`,
              }}
            />
            <span className="font-semibold" style={{ color: allServicesOk ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
              {allServicesOk ? 'OK' : 'Issues'}
            </span>
          </span>
        </div>
      )}
    </div>
  );
}
