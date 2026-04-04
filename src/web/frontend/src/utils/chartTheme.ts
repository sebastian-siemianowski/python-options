/**
 * chartTheme.ts -- Centralised Recharts theme (S1.8)
 *
 * All chart components reference this shared config to ensure consistent
 * look-and-feel across every page.
 */
import { useEffect, useRef, useCallback } from 'react';

/* ── Colour Palette ──────────────────────────────────────────── */

export const CHART_COLORS = {
  violet: 'var(--accent-violet)',
  emerald: 'var(--accent-emerald)',
  rose: 'var(--accent-rose)',
  amber: 'var(--accent-amber)',
  cyan: 'var(--accent-cyan)',
  indigo: 'var(--accent-indigo)',
  muted: 'var(--text-muted)',
  grid: 'rgba(139,92,246,0.06)',
} as const;

/* ── Tooltip Style ───────────────────────────────────────────── */

export const CHART_TOOLTIP_STYLE: React.CSSProperties = {
  background: 'rgba(15,15,35,0.95)',
  border: '1px solid rgba(139,92,246,0.15)',
  borderRadius: 12,
  color: 'var(--text-primary)',
  backdropFilter: 'blur(16px)',
  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  padding: '12px 16px',
  fontSize: 12,
};

/* ── Axis Style ──────────────────────────────────────────────── */

export const CHART_AXIS_STYLE = {
  fill: 'var(--text-muted)',
  fontSize: 11,
} as const;

/* ── Grid Style ──────────────────────────────────────────────── */

export const CHART_GRID_PROPS = {
  stroke: CHART_COLORS.grid,
  strokeDasharray: '3 3',
} as const;

/* ── Gradient Area Fills ─────────────────────────────────────── */

export const CHART_GRADIENTS = {
  violetArea:  { id: 'violet-area',  from: 'rgba(139,92,246,0.15)', to: 'rgba(15,15,35,0)' },
  emeraldArea: { id: 'emerald-area', from: 'rgba(62,232,165,0.12)', to: 'rgba(15,15,35,0)' },
  roseArea:    { id: 'rose-area',    from: 'rgba(255,107,138,0.12)', to: 'rgba(15,15,35,0)' },
  amberArea:   { id: 'amber-area',   from: 'rgba(245,197,66,0.12)', to: 'rgba(15,15,35,0)' },
} as const;

/**
 * Render <defs> block with all gradient definitions.
 * Place inside any <svg> or Recharts chart.
 */
export function ChartGradientDefs() {
  return (
    <defs>
      {Object.values(CHART_GRADIENTS).map((g) => (
        <linearGradient key={g.id} id={g.id} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={g.from} />
          <stop offset="100%" stopColor={g.to} />
        </linearGradient>
      ))}
    </defs>
  );
}

/* ── Animated Line Hook ──────────────────────────────────────── */

/**
 * useAnimatedLine -- draws chart lines left-to-right on mount
 * using stroke-dashoffset animation.
 *
 * Returns a ref and style to apply to the SVG path/line element.
 */
export function useAnimatedLine(duration = 1000) {
  const ref = useRef<SVGPathElement | null>(null);
  const animated = useRef(false);

  const setRef = useCallback((node: SVGPathElement | null) => {
    ref.current = node;
    if (node && !animated.current) {
      animated.current = true;
      const length = node.getTotalLength();
      node.style.strokeDasharray = `${length}`;
      node.style.strokeDashoffset = `${length}`;
      // Force reflow
      node.getBoundingClientRect();
      node.style.transition = `stroke-dashoffset ${duration}ms cubic-bezier(0.16,1,0.3,1)`;
      node.style.strokeDashoffset = '0';
    }
  }, [duration]);

  useEffect(() => {
    return () => { animated.current = false; };
  }, []);

  return { ref: setRef };
}
