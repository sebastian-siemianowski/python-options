/**
 * Cosmic Empty States (Story 10.2)
 *
 * Each page gets a custom empty state with an abstract SVG illustration,
 * warm messaging, and clear action buttons. These are invitations, not errors.
 */
import { useNavigate } from 'react-router-dom';

/* ── SVG Illustrations ───────────────────────────────────────── */

function ConcentricRings() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="cosmic-float">
      <circle cx="40" cy="40" r="36" stroke="url(#ring-grad)" strokeWidth="1" opacity="0.3" />
      <circle cx="40" cy="40" r="26" stroke="url(#ring-grad)" strokeWidth="1.5" opacity="0.4" />
      <circle cx="40" cy="40" r="16" stroke="url(#ring-grad)" strokeWidth="2" opacity="0.5" />
      <circle cx="40" cy="40" r="6" fill="url(#ring-grad)" opacity="0.6" />
      <defs>
        <linearGradient id="ring-grad" x1="0" y1="0" x2="80" y2="80">
          <stop stopColor="#8B5CF6" />
          <stop offset="1" stopColor="#6366F1" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function TableDots() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="cosmic-float">
      {/* Column lines */}
      <line x1="20" y1="10" x2="20" y2="70" stroke="#8B5CF6" strokeWidth="1" opacity="0.2" />
      <line x1="40" y1="10" x2="40" y2="70" stroke="#8B5CF6" strokeWidth="1" opacity="0.15" />
      <line x1="60" y1="10" x2="60" y2="70" stroke="#8B5CF6" strokeWidth="1" opacity="0.2" />
      {/* Floating dots */}
      <circle cx="20" cy="25" r="2.5" fill="#8B5CF6" opacity="0.5" />
      <circle cx="40" cy="35" r="2" fill="#6366F1" opacity="0.4" />
      <circle cx="60" cy="20" r="3" fill="#8B5CF6" opacity="0.3" />
      <circle cx="20" cy="50" r="2" fill="#6366F1" opacity="0.4" />
      <circle cx="60" cy="55" r="2.5" fill="#8B5CF6" opacity="0.5" />
      <circle cx="40" cy="60" r="1.5" fill="#6366F1" opacity="0.3" />
    </svg>
  );
}

function CosmicWaveform() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="cosmic-float">
      <path
        d="M5 40 Q15 20 25 40 Q35 60 45 40 Q55 20 65 40 Q75 60 80 40"
        stroke="url(#wave-grad)" strokeWidth="2" fill="none" opacity="0.5"
      />
      <path
        d="M5 40 Q15 20 25 40 Q35 60 45 40 Q55 20 65 40 Q75 60 80 40 V80 H5 Z"
        fill="url(#wave-fill)" opacity="0.08"
      />
      <defs>
        <linearGradient id="wave-grad" x1="0" y1="0" x2="80" y2="0">
          <stop stopColor="#8B5CF6" />
          <stop offset="1" stopColor="#6366F1" />
        </linearGradient>
        <linearGradient id="wave-fill" x1="40" y1="40" x2="40" y2="80">
          <stop stopColor="#8B5CF6" stopOpacity="0.4" />
          <stop offset="1" stopColor="#8B5CF6" stopOpacity="0" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function EmptyGauge() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="cosmic-float">
      <path
        d="M15 55 A25 25 0 0 1 65 55"
        stroke="#8B5CF6" strokeWidth="2" strokeLinecap="round" opacity="0.2"
      />
      <path
        d="M20 53 A22 22 0 0 1 35 35"
        stroke="url(#gauge-grad)" strokeWidth="2.5" strokeLinecap="round" opacity="0.5"
      />
      <circle cx="40" cy="55" r="3" fill="#8B5CF6" opacity="0.4" />
      <defs>
        <linearGradient id="gauge-grad" x1="20" y1="55" x2="35" y2="35">
          <stop stopColor="#8B5CF6" />
          <stop offset="1" stopColor="#6366F1" />
        </linearGradient>
      </defs>
    </svg>
  );
}

function TrophyPedestal() {
  return (
    <svg width="80" height="80" viewBox="0 0 80 80" fill="none" className="cosmic-float">
      {/* Pedestal */}
      <rect x="25" y="55" width="30" height="6" rx="2" fill="#8B5CF6" opacity="0.2" />
      <rect x="30" y="45" width="20" height="12" rx="2" fill="#8B5CF6" opacity="0.15" />
      {/* Star sparkles */}
      <circle cx="40" cy="30" r="2" fill="#8B5CF6" opacity="0.5" />
      <circle cx="30" cy="22" r="1" fill="#6366F1" opacity="0.4" />
      <circle cx="50" cy="24" r="1.5" fill="#8B5CF6" opacity="0.3" />
      <circle cx="35" cy="16" r="1" fill="#6366F1" opacity="0.35" />
      <circle cx="46" cy="18" r="1" fill="#8B5CF6" opacity="0.45" />
    </svg>
  );
}

/* ── Action Button ───────────────────────────────────────────── */

function CosmicButton({ label, onClick, variant = 'primary' }: {
  label: string;
  onClick: () => void;
  variant?: 'primary' | 'ghost';
}) {
  if (variant === 'ghost') {
    return (
      <button
        onClick={onClick}
        className="h-10 px-5 rounded-xl text-sm font-medium transition-all duration-200
          hover:bg-[var(--violet-8)]"
        style={{ color: 'var(--text-secondary)', border: '1px solid var(--border-void)' }}
      >
        {label}
      </button>
    );
  }
  return (
    <button
      onClick={onClick}
      className="h-10 px-5 rounded-xl text-sm font-semibold text-white transition-all duration-200
        hover:shadow-lg hover:shadow-violet-500/20 hover:-translate-y-px"
      style={{ background: 'linear-gradient(135deg, #8B5CF6, #6366F1)' }}
    >
      {label}
    </button>
  );
}

/* ── Empty State Wrapper ─────────────────────────────────────── */

function EmptyStateShell({ illustration, heading, description, children }: {
  illustration: React.ReactNode;
  heading: string;
  description: string;
  children?: React.ReactNode;
}) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-5 fade-up" style={{ maxWidth: 400, margin: '0 auto' }}>
      <div className="empty-icon-badge mb-2">
        {illustration}
        <span className="empty-particle" />
        <span className="empty-particle" />
        <span className="empty-particle" />
        <span className="empty-particle" />
      </div>
      <h3 className="text-section text-center" style={{ color: 'var(--text-luminous)' }}>
        {heading}
      </h3>
      <p className="text-caption text-center leading-relaxed">
        {description}
      </p>
      {children && <div className="flex items-center gap-3 mt-2">{children}</div>}
    </div>
  );
}

/* ── Page-Specific Empty States ──────────────────────────────── */

export function DashboardEmpty() {
  const nav = useNavigate();
  return (
    <EmptyStateShell
      illustration={<ConcentricRings />}
      heading="No signals yet"
      description="Run your first tune to see the universe come alive."
    >
      <CosmicButton label="Start Tune" onClick={() => nav('/tuning')} />
    </EmptyStateShell>
  );
}

export function SignalsEmpty() {
  const nav = useNavigate();
  return (
    <EmptyStateShell
      illustration={<TableDots />}
      heading="No signals generated"
      description="Start tuning to populate the signal table."
    >
      <CosmicButton label="Start Tune" onClick={() => nav('/tuning')} />
      <CosmicButton label="Refresh Data" onClick={() => window.location.reload()} variant="ghost" />
    </EmptyStateShell>
  );
}

export function ChartsEmpty() {
  return (
    <EmptyStateShell
      illustration={<CosmicWaveform />}
      heading="Select an asset to begin charting"
      description="Pick from the sidebar or choose a quick-start asset below."
    >
      <div className="flex flex-wrap justify-center gap-2">
        {['SPY', 'AAPL', 'NVDA', 'TSLA'].map((t) => (
          <span
            key={t}
            className="px-3 py-1 rounded-full text-xs font-medium cursor-pointer transition-all duration-200
              hover:bg-[var(--violet-15)]"
            style={{
              background: 'var(--violet-6)',
              color: '#C4B5FD',
              border: '1px solid var(--violet-12)',
            }}
          >
            {t}
          </span>
        ))}
      </div>
    </EmptyStateShell>
  );
}

export function RiskEmpty() {
  const nav = useNavigate();
  return (
    <EmptyStateShell
      illustration={<EmptyGauge />}
      heading="No risk data available"
      description="Signals must be generated first."
    >
      <CosmicButton label="Generate Signals" onClick={() => nav('/signals')} />
    </EmptyStateShell>
  );
}

export function ArenaEmpty() {
  return (
    <EmptyStateShell
      illustration={<TrophyPedestal />}
      heading="No arena results yet"
      description="Run an arena competition to see model leaderboards."
    >
      <CosmicButton label="Start Arena" onClick={() => {}} />
    </EmptyStateShell>
  );
}
