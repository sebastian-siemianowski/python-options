/**
 * Story 3.1: Signal Table visual sub-components.
 * Gradient strength bars, momentum badges, crash risk heat indicators,
 * horizon micro-arrows -- all with cosmic void aesthetic.
 */

/* ── AC-2: Signal Strength Meter (bar + pct) ─────────────────────── */
export function SignalStrengthMeter({ label, pUp, kelly }: { label: string; pUp?: number; kelly?: number }) {
  const upper = label.toUpperCase();
  const isBuy = upper.includes('BUY');
  const isSell = upper.includes('SELL');

  const confidence = Math.min(1, Math.max(0, ((pUp ?? 0.5) + (kelly ?? 0)) / 1.5));
  const fillPct = Math.round(confidence * 100);

  let gradient: string;
  let glowColor: string;
  let textCol: string;
  if (isBuy) {
    gradient = 'linear-gradient(90deg, var(--emerald-30) 0%, var(--accent-emerald) 100%)';
    glowColor = 'var(--emerald-30)';
    textCol = 'var(--accent-emerald)';
  } else if (isSell) {
    gradient = 'linear-gradient(90deg, var(--rose-30) 0%, var(--accent-rose) 100%)';
    glowColor = 'var(--rose-30)';
    textCol = 'var(--accent-rose)';
  } else {
    gradient = 'linear-gradient(90deg, var(--violet-20) 0%, rgba(139,92,246,0.45) 100%)';
    glowColor = 'rgba(139,92,246,0.18)';
    textCol = 'var(--text-muted)';
  }

  return (
    <div className="flex flex-col items-center gap-[3px]" style={{ width: 60 }}>
      <div
        className="relative h-[6px] rounded-[3px] overflow-hidden w-full"
        style={{ background: 'var(--void-active)' }}
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
      <span
        className="text-[9px] font-semibold tabular-nums leading-none"
        style={{ color: textCol }}
      >
        {fillPct}%
      </span>
    </div>
  );
}

/* Backward-compat composite: label + meter inline */
export function SignalStrengthBar({ label, pUp, kelly }: { label: string; pUp?: number; kelly?: number }) {
  return (
    <div className="flex items-center gap-1.5">
      <SignalLabel label={label.toUpperCase()} />
      <SignalStrengthMeter label={label} pUp={pUp} kelly={kelly} />
    </div>
  );
}

export function SignalLabel({ label }: { label: string }) {
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

/* ── AC-4: Crash Risk Heatmap Tile ───────────────────────────────── */
export function CrashRiskHeat({ score }: { score: number }) {
  const s = Math.min(100, Math.max(0, score ?? 0));

  let tier: 'LOW' | 'MOD' | 'ELEV' | 'HIGH';
  let base: string;
  let textCol: string;
  if (s < 25) {
    tier = 'LOW';
    base = '62,232,165';
    textCol = 'var(--accent-emerald)';
  } else if (s < 50) {
    tier = 'MOD';
    base = '245,197,66';
    textCol = 'var(--accent-amber)';
  } else if (s < 75) {
    tier = 'ELEV';
    base = '249,115,22';
    textCol = 'var(--accent-orange)';
  } else {
    tier = 'HIGH';
    base = '255,107,138';
    textCol = 'var(--accent-rose)';
  }

  const alpha = Math.min(0.48, 0.12 + s / 240);
  const borderAlpha = Math.min(0.55, 0.22 + s / 260);
  const isHigh = s >= 75;

  return (
    <div
      className="flex flex-col items-center justify-center rounded-md"
      title={`Crash risk: ${s.toFixed(0)}% (${tier})`}
      style={{
        minWidth: 48,
        minHeight: 32,
        padding: '2px 4px',
        background: `rgba(${base},${alpha})`,
        border: `1px solid rgba(${base},${borderAlpha})`,
        boxShadow: isHigh ? `inset 0 0 6px rgba(${base},0.35)` : 'none',
        transition: 'background 200ms ease, box-shadow 200ms ease',
      }}
    >
      <span
        className="text-[10px] font-semibold tabular-nums leading-none"
        style={{ color: textCol }}
      >
        {s.toFixed(0)}
      </span>
      <span
        className="text-[8px] font-semibold leading-none mt-[2px] tracking-wide"
        style={{ color: textCol, opacity: 0.82 }}
      >
        {tier}
      </span>
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

export function HorizonCell({ expRet }: { expRet: number | null | undefined; pUp: number | null | undefined }) {
  if (expRet == null) {
    return (
      <div
        className="flex items-center justify-center rounded-md"
        style={{
          minHeight: 32,
          background: 'rgba(74,85,104,0.06)',
          border: '1px solid rgba(74,85,104,0.12)',
        }}
      >
        <span className="text-[var(--text-muted)] text-[10px]">—</span>
      </div>
    );
  }

  const pct = expRet * 100;
  const absPct = Math.abs(pct);
  const isUp = pct > 0;
  const direction: 'up' | 'down' | 'flat' = absPct < 0.1 ? 'flat' : isUp ? 'up' : 'down';
  const isStrong = absPct > 5;

  // Heatmap intensity: 0.10 → 0.50 based on |pct| (capped at ~8%)
  const alpha = Math.min(0.48, 0.10 + absPct / 16);
  const borderAlpha = Math.min(0.55, 0.18 + absPct / 18);

  let bg: string;
  let borderCol: string;
  let textCol: string;
  if (direction === 'up') {
    bg = `rgba(62,232,165,${alpha})`;
    borderCol = `rgba(62,232,165,${borderAlpha})`;
    textCol = 'var(--accent-emerald)';
  } else if (direction === 'down') {
    bg = `rgba(255,107,138,${alpha})`;
    borderCol = `rgba(255,107,138,${borderAlpha})`;
    textCol = 'var(--accent-rose)';
  } else {
    bg = 'rgba(74,85,104,0.10)';
    borderCol = 'rgba(74,85,104,0.22)';
    textCol = 'var(--text-muted)';
  }

  return (
    <div
      className="flex flex-col items-center justify-center rounded-md"
      data-direction={direction}
      data-strong={isStrong}
      style={{
        minHeight: 32,
        padding: '3px 4px',
        background: bg,
        border: `1px solid ${borderCol}`,
        boxShadow: isStrong ? `inset 0 0 8px ${bg}` : 'none',
        transition: 'background 200ms ease, border-color 200ms ease',
      }}
    >
      <span
        className="text-[10.5px] font-mono tabular-nums leading-[1.1]"
        style={{ color: textCol, fontWeight: isStrong ? 700 : 600 }}
      >
        {pct >= 0 ? '+' : ''}{pct.toFixed(1)}%
      </span>
    </div>
  );
}

/* ── Quality Score Tile (matches HeatmapPage) ──────────────────── */
function qualityColor(score: number): string {
  if (score >= 90) return 'var(--accent-emerald)';
  if (score >= 80) return 'rgba(62,232,165,0.85)';
  if (score >= 70) return 'rgba(62,232,165,0.65)';
  if (score >= 60) return 'rgba(139,152,246,0.75)';
  if (score >= 50) return 'var(--text-muted)';
  if (score >= 40) return 'rgba(255,180,107,0.75)';
  if (score >= 30) return 'rgba(255,138,107,0.75)';
  if (score >= 20) return 'rgba(255,107,138,0.75)';
  return 'var(--accent-rose)';
}

function qualityBg(score: number): string {
  if (score >= 80) return 'rgba(6,78,59,0.25)';
  if (score >= 60) return 'rgba(30,27,75,0.25)';
  if (score >= 40) return 'rgba(60,40,10,0.2)';
  return 'rgba(76,5,25,0.2)';
}

export function QualityCell({ score }: { score: number | null | undefined }) {
  const qs = Math.round(score ?? 50);
  return (
    <div
      className="rounded-[4px] mx-auto"
      title={`Quality score: ${qs}`}
      style={{
        background: qualityBg(qs),
        height: 26,
        width: 42,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        border: '1px solid var(--violet-3)',
      }}
    >
      <span className="text-[10px] tabular-nums font-bold" style={{ color: qualityColor(qs) }}>
        {qs}
      </span>
    </div>
  );
}
