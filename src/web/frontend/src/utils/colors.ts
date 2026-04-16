/**
 * Design system colour tokens -- mirrored from index.css :root
 * Use CSS variables (var(--accent-violet)) in styles whenever possible.
 * Use these JS constants only when CSS variables aren't supported
 * (SVG stopColor, Recharts stroke/fill, computed values).
 */

export const colors = {
  /* Accent palette */
  violet:       '#8b5cf6',
  violetSoft:   '#7c3aed',
  violetBright: '#a78bfa',
  indigo:       '#6366f1',
  cyan:         '#38d9f5',
  emerald:      '#3ee8a5',
  rose:         '#ff6b8a',
  amber:        '#f5c542',
  fuchsia:      '#e27af5',
  orange:       '#f97316',

  /* Text hierarchy */
  textLuminous:  '#f8fafc',
  textPrimary:   '#edf0f7',
  textSecondary: '#8e99b0',
  textMuted:     '#4a5568',
  textViolet:    '#c4b5fd',

  /* Void surfaces */
  void:        '#0a0a1a',
  voidSurface: '#0c0b1d',
  voidRaised:  '#12102a',
  voidHover:   '#1a1740',
  voidActive:  '#221e4e',
} as const;

/** rgba helper -- returns rgba string from hex + alpha */
export function rgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}
