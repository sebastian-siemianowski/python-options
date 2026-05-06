export function signalLabelColor(label: string): string {
  switch (label) {
    case 'STRONG BUY': return '#10b981';
    case 'BUY': return '#6ee7b7';
    case 'HOLD': return '#64748b';
    case 'SELL': return '#fca5a5';
    case 'STRONG SELL': return '#f43f5e';
    default: return '#64748b';
  }
}

export const smaQualityTone = (score: number | null | undefined) => {
  if (score == null || !isFinite(score)) {
    return {
      label: 'Unknown',
      color: 'var(--text-muted)',
      background: 'rgba(100,116,139,0.10)',
      border: 'rgba(100,116,139,0.20)',
      glow: 'none',
    };
  }
  if (score >= 80) {
    return {
      label: 'Elite',
      color: '#34d399',
      background: 'rgba(6,78,59,0.25)',
      border: 'rgba(16,185,129,0.42)',
      glow: '0 0 14px -8px rgba(16,185,129,0.9)',
    };
  }
  if (score >= 60) {
    return {
      label: 'Good',
      color: '#c4b5fd',
      background: 'rgba(30,27,75,0.28)',
      border: 'rgba(167,139,250,0.34)',
      glow: '0 0 14px -9px rgba(167,139,250,0.8)',
    };
  }
  if (score >= 40) {
    return {
      label: 'Mixed',
      color: '#fbbf24',
      background: 'rgba(60,40,10,0.22)',
      border: 'rgba(251,191,36,0.30)',
      glow: 'none',
    };
  }
  return {
    label: 'Weak',
    color: '#fb7185',
    background: 'rgba(76,5,25,0.22)',
    border: 'rgba(244,63,94,0.34)',
    glow: '0 0 14px -10px rgba(244,63,94,0.8)',
  };
};
