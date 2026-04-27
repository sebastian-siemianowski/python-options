/**
 * Human-readable model name formatting for the web dashboard.
 *
 * Maps internal model identifiers (e.g. "phi_student_t_nu_8_momentum")
 * to clean display names (e.g. "Student-t ν=8 +Mom").
 */

const MODEL_DISPLAY_NAMES: Record<string, string> = {
  // ── Gaussian family ──────────────────────────────────────────
  'kalman_gaussian':               'Gaussian',
  'kalman_phi_gaussian':           'AR(1) Gaussian',
  'kalman_gaussian_unified':       'Gaussian Unified',
  'kalman_phi_gaussian_unified':   'AR(1) Gaussian Unified',
  'kalman_gaussian_momentum':      'Gaussian +Mom',
  'kalman_phi_gaussian_momentum':  'AR(1) Gaussian +Mom',

  // ── Legacy / simple ──────────────────────────────────────────
  'zero_drift':      'Zero Drift',
  'constant_drift':  'Constant Drift',
  'ewma_drift':      'EWMA Drift',
  'kalman_drift':    'Kalman Drift',
  'gaussian':        'Gaussian',
};

/**
 * Format an internal model name to a clean, human-readable display name.
 *
 * Handles:
 *  - Exact matches from the lookup table
 *  - Student-t models: phi_student_t_nu_8 → "Student-t ν=8"
 *  - VoV variants:     phi_student_t_nu_8_vov_g0.5 → "Student-t ν=8 VoV γ=0.5"
 *  - Momentum suffix:  _momentum → "+Mom"
 *  - Two-piece:        _two_piece → "2P"
 *  - Mixture:          _mixture → "Mix"
 *  - Unknown:          cleaned fallback
 */
export function formatModelName(raw: string | undefined | null): string {
  if (!raw) return '—';

  // Exact match
  if (MODEL_DISPLAY_NAMES[raw]) return MODEL_DISPLAY_NAMES[raw];

  // ── Student-t pattern matching ────────────────────────────────
  const nuMatch = raw.match(/phi_student_t_(?:(unified_improved|unified|improved)_)?nu_(mle|\d+)/);
  if (nuMatch) {
    const variant = nuMatch[1] || '';
    const nu = nuMatch[2];
    let name = `Student-t ν=${nu}`;
    if (variant === 'unified_improved') name = `Student-t Uni Improved ν=${nu}`;
    else if (variant === 'unified') name = `Student-t Uni ν=${nu}`;
    else if (variant === 'improved') name = `Student-t Improved ν=${nu}`;

    // Enhancements (order matters for display)
    if (raw.includes('_vov')) {
      const gammaMatch = raw.match(/_g(\d+\.?\d*)/);
      name += gammaMatch ? ` VoV γ=${gammaMatch[1]}` : ' VoV';
    }
    if (raw.includes('_two_piece'))  name += ' 2-Piece';
    if (raw.includes('_mixture'))    name += ' Mix';
    if (raw.includes('_momentum'))   name += ' +Mom';

    return name;
  }

  // ── Fallback: clean up underscores ────────────────────────────
  return raw
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (c) => c.toUpperCase())
    .replace(/\bNu\b/g, 'ν=')
    .replace(/\bPhi\b/g, 'φ')
    .trim();
}

/**
 * Shorter version for charts/badges where space is limited.
 * e.g. "Student-t ν=8" → "T(8)", "Gaussian Unified" → "G-Uni"
 */
export function formatModelNameShort(raw: string | undefined | null): string {
  if (!raw) return '—';

  const SHORT_MAP: Record<string, string> = {
    'kalman_gaussian':               'Gauss',
    'kalman_phi_gaussian':           'φ-Gauss',
    'kalman_gaussian_unified':       'G-Uni',
    'kalman_phi_gaussian_unified':   'φG-Uni',
    'kalman_gaussian_momentum':      'G+M',
    'kalman_phi_gaussian_momentum':  'φG+M',
    'zero_drift':                    'Zero',
    'constant_drift':                'Const',
    'ewma_drift':                    'EWMA',
    'kalman_drift':                  'K-Drift',
  };
  if (SHORT_MAP[raw]) return SHORT_MAP[raw];

  // Student-t patterns (unified and standard)
  const nuMatch = raw.match(/phi_student_t_(?:(unified_improved|unified|improved)_)?nu_(mle|\d+)/);
  if (nuMatch) {
    const variant = nuMatch[1] || '';
    const tag = variant === 'unified_improved' ? 'UI' : variant === 'unified' ? 'U' : variant === 'improved' ? 'I' : '';
    let s = `T${tag}(${nuMatch[2]})`;
    if (raw.includes('_momentum')) s += '+M';
    if (raw.includes('_vov'))     s += 'V';
    if (raw.includes('_two_piece')) s += '2P';
    if (raw.includes('_mixture')) s += 'x';
    return s;
  }

  // Fallback: abbreviate intelligently instead of truncating
  return raw
    .replace(/kalman_/g, 'K-')
    .replace(/_momentum/g, '+M')
    .replace(/_unified/g, '-U')
    .replace(/_/g, ' ')
    .replace(/\b\w/g, c => c.toUpperCase())
    .trim();
}
