/**
 * MicroGauge -- 40px SVG arc gauge with animated fill and gradient stroke.
 * Used in the Morning Briefing hero card's System Pulse section.
 */
import { useEffect, useRef, useState } from 'react';

interface Props {
  /** 0-1 fraction of the arc to fill */
  value: number;
  /** Label shown inside the gauge */
  label: string;
  /** Caption shown below the gauge */
  caption: string;
  /** Gradient color stops [start, end] */
  gradient: [string, string];
  /** Unique id for SVG gradient */
  id: string;
}

export default function MicroGauge({ value, label, caption, gradient, id }: Props) {
  const [animated, setAnimated] = useState(0);
  const rafRef = useRef<number>(0);
  const startRef = useRef<number>(0);

  useEffect(() => {
    const clamped = Math.max(0, Math.min(1, value));
    startRef.current = performance.now();
    const duration = 800;

    function tick(now: number) {
      const elapsed = now - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      // spring-like overshoot: cubic-bezier(0.34, 1.56, 0.64, 1)
      const eased = progress < 1
        ? 1 - Math.pow(1 - progress, 3) * (1 + 2.2 * (1 - progress))
        : 1;
      setAnimated(clamped * Math.min(eased, 1.08));
      if (progress < 1) rafRef.current = requestAnimationFrame(tick);
      else setAnimated(clamped);
    }
    rafRef.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(rafRef.current);
  }, [value]);

  const size = 40;
  const stroke = 3;
  const radius = (size - stroke) / 2;
  const circumference = 2 * Math.PI * radius;
  // 270-degree arc
  const arcLength = circumference * 0.75;
  const dashOffset = arcLength - arcLength * animated;

  return (
    <div className="flex flex-col items-center gap-1">
      <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
        <defs>
          <linearGradient id={`gauge-grad-${id}`} x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor={gradient[0]} />
            <stop offset="100%" stopColor={gradient[1]} />
          </linearGradient>
        </defs>
        {/* Track */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke="var(--violet-8)"
          strokeWidth={stroke}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset={0}
          transform={`rotate(135 ${size / 2} ${size / 2})`}
          strokeLinecap="round"
        />
        {/* Fill */}
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          fill="none"
          stroke={`url(#gauge-grad-${id})`}
          strokeWidth={stroke}
          strokeDasharray={`${arcLength} ${circumference}`}
          strokeDashoffset={dashOffset}
          transform={`rotate(135 ${size / 2} ${size / 2})`}
          strokeLinecap="round"
          style={{ transition: 'none' }}
        />
        {/* Center label */}
        <text
          x={size / 2}
          y={size / 2 + 1}
          textAnchor="middle"
          dominantBaseline="central"
          fill="#e2e8f0"
          fontSize="9"
          fontWeight="600"
          fontFamily="SF Mono, ui-monospace, monospace"
          style={{ fontVariantNumeric: 'tabular-nums' }}
        >
          {label}
        </text>
      </svg>
      <span
        className="text-[10px] tracking-wider uppercase"
        style={{ color: 'var(--text-muted, #6b7a90)' }}
      >
        {caption}
      </span>
    </div>
  );
}
