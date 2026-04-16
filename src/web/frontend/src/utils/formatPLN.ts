/**
 * PLN formatting utilities.
 * Story 6.2: Profit & P&L Attribution Column.
 */

/**
 * Format PLN amount with sign, thousands separator, and abbreviation for large values.
 * Examples: +12,345 PLN, -3,200 PLN, +1.2M PLN
 */
export function formatPLN(value: number | null | undefined): string {
  if (value == null || !Number.isFinite(value)) return '\u2014';
  const abs = Math.abs(value);
  const sign = value >= 0 ? '+' : '\u2212';
  let formatted: string;
  if (abs >= 1_000_000) {
    formatted = `${(abs / 1_000_000).toFixed(1)}M`;
  } else if (abs >= 10_000) {
    formatted = Math.round(abs).toLocaleString('en-US');
  } else {
    formatted = Math.round(abs).toLocaleString('en-US');
  }
  return `${sign}${formatted}`;
}

/**
 * Return a Tailwind-compatible color for a profit value.
 * Green for positive, red for negative, dim for near-zero.
 */
export function profitColor(value: number | null | undefined): string {
  if (value == null || Math.abs(value) < 1) return '#64748b';
  return value > 0 ? '#34d399' : '#fb7185';
}
