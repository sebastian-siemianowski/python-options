import { useCallback, useEffect, useRef, useState } from 'react';

// iOS-style segmented control with a sliding accent indicator.
// Generic over readonly option arrays so callers get a narrow value type.
export default function SegmentedControl<K extends string>({
  options,
  value,
  onChange,
  accent = 'var(--accent-violet)',
  size = 'md',
}: {
  options: ReadonlyArray<{ key: K; label: string; dot?: string }>;
  value: K;
  onChange: (next: K) => void;
  accent?: string;
  size?: 'sm' | 'md';
}) {
  const btnRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [indicator, setIndicator] = useState<{ x: number; w: number } | null>(null);

  const recompute = useCallback(() => {
    const container = containerRef.current;
    const btn = btnRefs.current[value];
    if (!container || !btn) return;
    const cRect = container.getBoundingClientRect();
    const bRect = btn.getBoundingClientRect();
    setIndicator({ x: bRect.left - cRect.left, w: bRect.width });
  }, [value]);

  useEffect(() => {
    recompute();
  }, [recompute, options.length]);

  useEffect(() => {
    const onResize = () => recompute();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [recompute]);

  const pad = size === 'sm' ? 'px-2 py-[5px]' : 'px-2.5 py-[6px]';
  const textSize = size === 'sm' ? 'text-[10.5px]' : 'text-[11.5px]';

  return (
    <div
      ref={containerRef}
      className="relative inline-flex items-center gap-0 rounded-xl p-[3px]"
      style={{
        background: 'rgba(255,255,255,0.025)',
        border: '1px solid rgba(255,255,255,0.05)',
        boxShadow: '0 1px 0 rgba(255,255,255,0.03) inset',
      }}
    >
      {indicator && (
        <div
          aria-hidden
          className="pointer-events-none absolute bottom-[3px] top-[3px] rounded-[9px]"
          style={{
            left: indicator.x,
            width: indicator.w,
            background: `linear-gradient(180deg, ${accent}22, ${accent}0c)`,
            border: `1px solid ${accent}55`,
            boxShadow: `0 0 0 1px ${accent}18 inset, 0 4px 14px -6px ${accent}85, 0 0 18px -6px ${accent}55`,
            transition: 'left 280ms cubic-bezier(.2,.8,.2,1), width 280ms cubic-bezier(.2,.8,.2,1)',
          }}
        />
      )}
      {options.map((opt) => {
        const on = opt.key === value;
        return (
          <button
            key={opt.key}
            ref={(el) => { btnRefs.current[opt.key] = el; }}
            type="button"
            onClick={() => onChange(opt.key)}
            aria-pressed={on}
            className={`relative inline-flex items-center gap-1.5 rounded-[9px] ${pad} ${textSize} font-medium transition-colors duration-200`}
            style={{
              color: on ? '#fff' : 'var(--text-secondary)',
            }}
          >
            {opt.dot && (
              <span
                aria-hidden
                className="rounded-full"
                style={{
                  width: 5,
                  height: 5,
                  background: opt.dot,
                  boxShadow: on ? `0 0 6px ${opt.dot}` : 'none',
                  transition: 'box-shadow 220ms',
                }}
              />
            )}
            <span className="whitespace-nowrap">{opt.label}</span>
          </button>
        );
      })}
    </div>
  );
}
