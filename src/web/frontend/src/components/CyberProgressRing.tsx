import { useId, type CSSProperties } from 'react';

type CyberProgressRingProps = {
  percent: number;
  color: string;
  accent?: string;
  size?: number;
  stroke?: number;
  label?: string;
  caption?: string;
  stages?: Array<{ key: string; label: string; tone: string }>;
  activeStageIndex?: number;
  running?: boolean;
  status?: string | null;
  compact?: boolean;
  className?: string;
  ariaLabel?: string;
};

export default function CyberProgressRing({
  percent,
  color,
  accent = '#38d9f5',
  size = 112,
  stroke = 8,
  label = 'Complete',
  caption,
  stages = [],
  activeStageIndex = 0,
  running = false,
  status,
  compact = false,
  className = '',
  ariaLabel,
}: CyberProgressRingProps) {
  const rawId = useId().replace(/:/g, '');
  const pct = Math.max(0, Math.min(100, Number.isFinite(percent) ? percent : 0));
  const failed = status === 'failed' || status === 'error';
  const ringColor = failed ? '#fb7185' : color;
  const ringAccent = failed ? '#fecdd3' : accent;
  const trackStroke = Math.max(1, stroke - 1);
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  const dashOffset = circumference - (pct / 100) * circumference;
  const center = size / 2;
  const innerRadius = Math.max(8, radius - stroke * 1.45);
  const segmentCount = Math.max(1, stages.length);
  const segmentGap = Math.min(18, circumference / segmentCount * 0.13);
  const segmentLength = Math.max(4, circumference / segmentCount - segmentGap);
  const wrapperStyle = {
    width: size,
    height: size,
    '--ring-color': ringColor,
    '--ring-accent': ringAccent,
  } as CSSProperties;

  return (
    <div
      className={`cyber-progress-ring ${running ? 'is-running' : ''} ${compact ? 'is-compact' : ''} ${className}`}
      style={wrapperStyle}
      role="progressbar"
      aria-label={ariaLabel ?? label}
      aria-valuemin={0}
      aria-valuemax={100}
      aria-valuenow={Math.round(pct)}
    >
      <div className="cyber-progress-ring__aura" aria-hidden />
      <div className="cyber-progress-ring__grid" aria-hidden />
      <div className="cyber-progress-ring__ticks" aria-hidden />
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`} aria-hidden>
        <defs>
          <linearGradient id={`${rawId}-stroke`} x1="12%" y1="8%" x2="88%" y2="92%">
            <stop offset="0%" stopColor={ringColor} />
            <stop offset="56%" stopColor={ringAccent} />
            <stop offset="100%" stopColor="#ffffff" />
          </linearGradient>
          <filter id={`${rawId}-glow`} x="-45%" y="-45%" width="190%" height="190%">
            <feGaussianBlur stdDeviation="3.5" result="blur" />
            <feColorMatrix
              in="blur"
              type="matrix"
              values="1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 .82 0"
              result="glow"
            />
            <feMerge>
              <feMergeNode in="glow" />
              <feMergeNode in="SourceGraphic" />
            </feMerge>
          </filter>
        </defs>
        <circle
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke="rgba(255,255,255,0.065)"
          strokeWidth={trackStroke}
        />
        {stages.map((stage, index) => {
          const stageNumber = index + 1;
          const stageDone = activeStageIndex > stageNumber;
          const stageActive = activeStageIndex === stageNumber;
          return (
            <circle
              key={stage.key}
              cx={center}
              cy={center}
              r={radius}
              fill="none"
              stroke={stageDone || stageActive ? stage.tone : 'rgba(255,255,255,0.12)'}
              strokeWidth={Math.max(2, stroke * 0.42)}
              strokeLinecap="round"
              strokeDasharray={`${segmentLength} ${circumference - segmentLength}`}
              strokeDashoffset={-(index * (segmentLength + segmentGap))}
              opacity={stageActive ? 0.92 : stageDone ? 0.62 : 0.26}
              transform={`rotate(-90 ${center} ${center})`}
            />
          );
        })}
        <circle
          cx={center}
          cy={center}
          r={innerRadius}
          fill="none"
          stroke="rgba(255,255,255,0.06)"
          strokeWidth="1"
          strokeDasharray="2 8"
        />
        <circle
          className="cyber-progress-ring__arc"
          cx={center}
          cy={center}
          r={radius}
          fill="none"
          stroke={`url(#${rawId}-stroke)`}
          strokeWidth={stroke}
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          filter={`url(#${rawId}-glow)`}
          transform={`rotate(-90 ${center} ${center})`}
        />
      </svg>
      {stages.map((stage, index) => {
        const stageNumber = index + 1;
        const angle = -90 + (360 / segmentCount) * (index + 0.5);
        const radians = (angle * Math.PI) / 180;
        const orbit = compact ? 45 : 46;
        const x = 50 + Math.cos(radians) * orbit;
        const y = 50 + Math.sin(radians) * orbit;
        const stageDone = activeStageIndex > stageNumber;
        const stageActive = activeStageIndex === stageNumber;
        return (
          <span
            key={`${stage.key}-node`}
            className={`cyber-progress-ring__node ${stageActive ? 'is-active' : ''} ${stageDone ? 'is-done' : ''}`}
            title={stage.label}
            style={{
              left: `${x}%`,
              top: `${y}%`,
              '--node-color': stage.tone,
            } as CSSProperties}
            aria-hidden
          />
        );
      })}
      <div className="cyber-progress-ring__core">
        <div className="cyber-progress-ring__value">{Math.round(pct)}</div>
        {!compact && <div className="cyber-progress-ring__label">{label}</div>}
        {caption && !compact && <div className="cyber-progress-ring__caption">{caption}</div>}
      </div>
      {running && <div className="cyber-progress-ring__sweep" aria-hidden />}
    </div>
  );
}
