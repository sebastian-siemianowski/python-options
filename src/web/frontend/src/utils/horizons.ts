/**
 * Horizon display utilities.
 * Story 6.1: Semantic horizon labels matching terminal UX.
 */

/** Map of trading-day horizons to human-readable labels. */
export const HORIZON_LABELS: Record<number, string> = {
  1: '1D',
  3: '3D',
  7: '1W',
  21: '1M',
  63: '3M',
  126: '6M',
  252: '12M',
};

/** Convert a numeric horizon to its semantic label. Falls back to `{h}D`. */
export function formatHorizon(h: number): string {
  return HORIZON_LABELS[h] ?? `${h}D`;
}

/**
 * Responsive horizon subsets.
 * - Desktop (>= 1024): all horizons
 * - Tablet (>= 768): 5 key horizons
 * - Mobile (< 768): 3 key horizons
 */
const TABLET_HORIZONS = new Set([1, 7, 21, 63, 252]);
const MOBILE_HORIZONS = new Set([7, 21, 63]);

export function responsiveHorizons(all: number[], width: number): number[] {
  if (width >= 1024) return all;
  if (width >= 768) return all.filter(h => TABLET_HORIZONS.has(h));
  return all.filter(h => MOBILE_HORIZONS.has(h));
}
