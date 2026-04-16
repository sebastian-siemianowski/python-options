/**
 * Story 3.1: Signal Table visual sub-components.
 * Gradient strength bars, momentum badges, crash risk heat indicators,
 * horizon micro-arrows -- all with cosmic void aesthetic.
 */

/* ── AC-2: Gradient Signal Strength Bar ──────────────────────────── */
export function SignalStrengthBar({ label, pUp, kelly }: { label: string; pUp?: number; kelly?: number }) {
  const upper = label.toUpperCase();
  const isBuy = upper.includes('BUY');
  const isSell = upper.includes('SELL');

  // Composite confidence: blend p_up and kelly
  const confidence = Math.min(1, Math.max(0, ((pUp ?? 0.5) + (kelly ?? 0)) / 1.5));
  const fillPct = Math.round(confidence * 100);

  let gradient: string;
  let glowColor: string;
  if (isBuy) {
    gradient = 'linear-gradient(90deg, var(--emerald-30) 0%, var(--accent-emerald) 100%)';
    glowColor = 'var(--emerald-30)';
  } else if (isSell) {
    gradient = 'linear-gradient(90deg, var(--rose-30) 0%, var(--accent-rose) 100%)';
    glowColor = 'var(--rose-30)';
  } else {
    gradient = 'linear-gradient(90deg, var(--violet-20) 0%, rgba(139,92,246,0.45) 100%)';
    glowColor = 'rgba(139,92,246,0.18)';
  }

  return (
    <div className="flex items-center gap-1.5">
      <SignalLabel label={upper} />
      <div
        className="relative h-[6px] rounded-[3px] overflow-hidden"
        style={{ width: 40, background: 'var(--void-active)' }}
      >
        <div
          className="absolute inset-y-0 left-0 rounded-[3px]"
          style={{
            width: `${fillPct}%`,
            background: gradient,
            boxShadow: `0 0 6px ${glowColor}`,
            transition: 'width 300ms cubic-bezier(0.2,0,0,1)',
          }}
        />
      </div>
    </div>
  );
}

function SignalLabel({ label }: { label: string }) {
  const colorMap: Record<string, string> = {
    'STRONG BUY': 'var(--accent-emerald)',
    'BUY': 'var(--accent-emerald)',
    'HOLD': 'var(--text-muted)',
    'SELL': 'var(--accent-rose)',
    'STRONG SELL': 'var(--accent-rose)',
    'EXIT': 'var(--accent-amber)',
  };
  const bgMap: Record<string, string> = {
    'STRONG BUY': 'rgba(62,232,165,0.14)',
    'BUY': 'rgba(62,232,165,0.09)',
    'HOLD': 'rgba(74,85,104,0.14)',
    'SELL': 'rgba(255,107,138,0.09)',
    'STRONG SELL': 'rgba(255,107,138,0.14)',
    'EXIT': 'rgba(245,197,66,0.10)',
  };
  return (
    <span
      className="text-[10px] font-semibold px-2 py-0.5 rounded-md"
      style={{ color: colorMap[label] || 'var(--text-muted)', background: bgMap[label] || 'transparent' }}
    >
      {label}
    </span>
  );
}

/* ── AC-3: Momentum Colored Numeric Badge ────────────────────────── */
export function MomentumBadge({ value }: { value: number }) {
  const v = value ?? 0;
  const absV = Math.abs(v);
  const sentiment = v > 0 ? 'positive' : v < -1 ? 'negative' : 'neutral';
  const isStrong = absV >= 70;

  return (
    <span
      className="momentum-badge"
      data-sentiment={sentiment}
      data-strong={isStrong}
    >
      {v > 0 ? '+' : ''}{Math.round(v)}%
    </span>
  );
}

/* ── AC-4: Crash Risk Heat Indicator ─────────────────────────────── */
export function CrashRiskHeat({ score }: { score: number }) {
  const s = Math.min(100, Math.max(0, score ?? 0));
  // 4 segments: 0-25, 25-50, 50-75, 75-100
  const segments = [
    { threshold: 25, filled: 'var(--accent-emerald)', glow: 'rgba(62,232,165,0.35)' },
    { threshold: 50, filled: 'var(--accent-amber)', glow: 'rgba(245,197,66,0.35)' },
    { threshold: 75, filled: 'var(--accent-orange)', glow: 'rgba(249,115,22,0.35)' },
    { threshold: 100, filled: 'var(--accent-rose)', glow: 'rgba(255,107,138,0.35)' },
  ];

  return (
    <div className="flex items-center gap-[2px]" title={`Crash risk: ${s.toFixed(0)}%`}>
      {segments.map((seg, i) => {
        const isFilled = s > seg.threshold - 25;
        return (
          <div
            key={i}
            className="rounded-[2px]"
            style={{
              width: 6,
              height: 12,
              background: isFilled ? seg.filled : 'var(--void-active)',
              boxShadow: isFilled ? `0 0 4px ${seg.glow}` : 'none',
              transition: 'background 200ms ease',
            }}
          />
        );
      })}
    </div>
  );
}

/* ── AC-5: Horizon Micro-Arrow ───────────────────────────────────── */
export function HorizonArrow({ direction }: { direction: 'up' | 'down' | 'neutral' }) {
  if (direction === 'up') {
    return (
      <svg width="8" height="8" viewBox="0 0 8 8" fill="none" className="inline-block mr-0.5" aria-label="Up">
        <path d="M4 1L7 6H1L4 1Z" fill="var(--accent-emerald)" />
      </svg>
    );
  }
  if (direction === 'down') {
    return (
      <svg width="8" height="8" viewBox="0 0 8 8" fill="none" className="inline-block mr-0.5" aria-label="Down">
        <path d="M4 7L1 2H7L4 7Z" fill="var(--accent-rose)" />
      </svg>
    );
  }
  return null;
}

export function HorizonCell({ expRet, pUp }: { expRet: number | null | undefined; pUp: number | null | undefined }) {
  if (expRet == null) return <span className="text-[var(--text-muted)] text-[10px]">--</span>;

  const pct = expRet * 100;
  const absPct = Math.abs(pct);
  const isUp = pct > 0;
  const direction: 'up' | 'down' | 'flat' = absPct < 0.1 ? 'flat' : isUp ? 'up' : 'down';
  const arrowDir: 'up' | 'down' | 'neutral' = absPct < 0.1 ? 'neutral' : isUp ? 'up' : 'down';
  const isStrong = absPct > 5;

  return (
    <div className="leading-[1.2]">
      <div className="horizon-cell text-[10px] font-mono tabular-nums" data-direction={direction} data-strong={isStrong}>
        <HorizonArrow direction={arrowDir} />
        {pct >= 0 ? '+' : ''}{pct.toFixed(1)}%
      </div>
      {pUp != null && (
        <span className="block text-[9px] text-[var(--text-muted)] font-mono tabular-nums">
          p{(pUp * 100).toFixed(0)}
        </span>
      )}
    </div>
  );
}
