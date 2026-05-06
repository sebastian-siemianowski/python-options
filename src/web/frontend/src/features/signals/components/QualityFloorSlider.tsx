import { useEffect, useRef, useState } from 'react';
import { RotateCcw, ShieldCheck } from 'lucide-react';

export default function QualityFloorSlider({
  value,
  onChange,
  compact = false,
  className = '',
}: {
  value: number;
  onChange: (value: number) => void;
  compact?: boolean;
  className?: string;
}) {
  const [draftValue, setDraftValue] = useState(value);
  const [isDragging, setIsDragging] = useState(false);
  const commitTimerRef = useRef<number | null>(null);
  const draggingRef = useRef(false);
  const active = draftValue > 0;
  const pct = Math.max(0, Math.min(100, draftValue));
  const handlePct = Math.max(1.5, Math.min(98.5, pct));
  const accent = pct >= 75 ? '#6ee7b7' : pct >= 50 ? '#c4b5fd' : '#38bdf8';
  const rangeHeight = compact ? 28 : 30;
  const trackHeight = compact ? 14 : 16;
  const fillHeight = compact ? 8 : 10;
  const handleWidth = compact ? 12 : 14;
  const handleHeight = compact ? 22 : 24;
  const cleanValue = (next: number) => Math.max(0, Math.min(100, Math.round(next)));
  const flushCommit = (next: number) => {
    const clean = cleanValue(next);
    if (commitTimerRef.current) window.clearTimeout(commitTimerRef.current);
    onChange(clean);
  };
  const scheduleCommit = (next: number) => {
    const clean = cleanValue(next);
    if (commitTimerRef.current) window.clearTimeout(commitTimerRef.current);
    commitTimerRef.current = window.setTimeout(() => onChange(clean), 140);
  };
  const updateDraft = (next: number) => {
    const clean = cleanValue(next);
    setDraftValue(clean);
    if (!draggingRef.current) scheduleCommit(clean);
  };
  const startDrag = () => {
    draggingRef.current = true;
    setIsDragging(true);
    if (commitTimerRef.current) window.clearTimeout(commitTimerRef.current);
  };
  const endDrag = (next: number) => {
    draggingRef.current = false;
    setIsDragging(false);
    flushCommit(next);
  };

  useEffect(() => setDraftValue(value), [value]);
  useEffect(() => () => {
    if (commitTimerRef.current) window.clearTimeout(commitTimerRef.current);
  }, []);

  return (
    <div
      className={`relative overflow-hidden rounded-[12px] ${compact ? 'px-2.5 py-1.5' : 'px-3 py-2'} ${className}`}
      style={{
        background: active
          ? `linear-gradient(135deg, ${accent}18, rgba(255,255,255,0.018) 58%, rgba(15,23,42,0.42))`
          : 'linear-gradient(135deg, rgba(255,255,255,0.032), rgba(255,255,255,0.012))',
        border: `1px solid ${active ? `${accent}4f` : 'rgba(255,255,255,0.06)'}`,
        boxShadow: active
          ? `0 0 0 1px ${accent}12, inset 0 1px 0 rgba(255,255,255,0.07), 0 12px 24px -24px ${accent}`
          : 'inset 0 1px 0 rgba(255,255,255,0.032)',
      }}
    >
      <style>{`
        .quality-floor-range {
          -webkit-appearance: none;
          appearance: none;
          height: ${rangeHeight}px;
          background: transparent;
          cursor: pointer;
          touch-action: none;
        }
        .quality-floor-range::-webkit-slider-runnable-track {
          height: ${rangeHeight}px;
          border-radius: 999px;
          background: transparent;
        }
        .quality-floor-range::-moz-range-track {
          height: ${rangeHeight}px;
          border-radius: 999px;
          background: transparent;
        }
        .quality-floor-range::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: ${Math.max(28, handleWidth + 18)}px;
          height: ${rangeHeight}px;
          margin-top: 0;
          border-radius: 999px;
          border: 0;
          background: transparent;
          box-shadow: none;
        }
        .quality-floor-range::-moz-range-thumb {
          width: ${Math.max(28, handleWidth + 18)}px;
          height: ${rangeHeight}px;
          border-radius: 999px;
          border: 0;
          background: transparent;
          box-shadow: none;
        }
      `}</style>

      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-3 top-0 h-px opacity-80"
        style={{ background: `linear-gradient(90deg, transparent, ${accent}7a, transparent)` }}
      />

      <div className="flex flex-wrap items-center gap-x-2.5 gap-y-1.5">
        <div className="flex min-w-[118px] items-center gap-2">
          <span
            className="inline-flex h-6 w-6 items-center justify-center rounded-[8px]"
            style={{
              color: active ? accent : 'var(--text-secondary)',
              background: active ? `${accent}18` : 'rgba(255,255,255,0.035)',
              border: `1px solid ${active ? `${accent}3d` : 'rgba(255,255,255,0.055)'}`,
            }}
          >
            <ShieldCheck className="w-3 h-3" />
          </span>
          <div>
            <div className="text-[9.5px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Min Quality
            </div>
          </div>
        </div>

        <div className="relative min-w-[200px] flex-1">
          <div
            aria-hidden
            className="absolute left-0 right-0 top-1/2 -translate-y-1/2 rounded-full"
            style={{
              height: trackHeight,
              background:
                'linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035))',
              border: '1px solid rgba(255,255,255,0.075)',
              boxShadow:
                'inset 0 1px 2px rgba(255,255,255,0.07), inset 0 -10px 18px rgba(2,6,23,0.22)',
            }}
          />
          <div
            aria-hidden
            className="absolute left-[3px] top-1/2 -translate-y-1/2 rounded-full"
            style={{
              height: fillHeight,
              width: `calc(${pct}% - ${pct > 0 ? 6 : 0}px)`,
              background: `linear-gradient(90deg, ${accent}cc 0%, ${accent}f2 58%, rgba(255,255,255,0.88) 100%)`,
              boxShadow: active
                ? `inset 0 1px 0 rgba(255,255,255,0.35), 0 0 18px -10px ${accent}`
                : 'none',
              transition: isDragging ? 'none' : 'width 160ms cubic-bezier(0.2,0,0,1)',
            }}
          />
          <div
            aria-hidden
            className="pointer-events-none absolute inset-x-[10px] top-1/2 h-px -translate-y-[7px] rounded-full"
            style={{ background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.22), transparent)' }}
          />
          <div
            aria-hidden
            className="pointer-events-none absolute top-1/2 z-[2] -translate-x-1/2 -translate-y-1/2"
            style={{
              left: `${handlePct}%`,
              transition: isDragging ? 'none' : 'left 160ms cubic-bezier(0.2,0,0,1)',
            }}
          >
            <div
              className="relative flex items-center justify-center rounded-full"
              style={{
                width: handleWidth,
                height: handleHeight,
                background:
                  'linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(241,245,249,0.94) 42%, rgba(203,213,225,0.82) 100%)',
                border: '1px solid rgba(255,255,255,0.78)',
                boxShadow: isDragging
                  ? `0 0 0 4px ${accent}1a, 0 10px 20px -15px rgba(0,0,0,0.86), 0 0 14px -8px ${accent}, inset 0 1px 0 rgba(255,255,255,0.96)`
                  : '0 8px 16px -14px rgba(0,0,0,0.92), inset 0 1px 0 rgba(255,255,255,0.94), inset 0 -6px 10px rgba(15,23,42,0.10)',
                transform: isDragging ? 'scale(1.035)' : 'scale(1)',
                transition: isDragging ? 'none' : 'transform 160ms ease, box-shadow 160ms ease',
              }}
            >
              <span
                className="absolute inset-x-[4px] top-[4px] h-px rounded-full"
                style={{ background: 'rgba(255,255,255,0.95)' }}
              />
              <span
                className="h-3 w-[2px] rounded-full"
                style={{
                  background: active ? accent : 'rgba(100,116,139,0.68)',
                  boxShadow: active ? `0 0 8px -2px ${accent}` : 'none',
                }}
              />
            </div>
          </div>
          <input
            type="range"
            min={0}
            max={100}
            step={1}
            value={draftValue}
            onPointerDown={startDrag}
            onChange={(event) => updateDraft(Number(event.target.value))}
            onPointerUp={(event) => endDrag(Number((event.currentTarget as HTMLInputElement).value))}
            onPointerCancel={(event) => endDrag(Number((event.currentTarget as HTMLInputElement).value))}
            onKeyUp={(event) => flushCommit(Number((event.currentTarget as HTMLInputElement).value))}
            onBlur={(event) => endDrag(Number(event.currentTarget.value))}
            className="quality-floor-range relative z-[1] w-full"
            aria-label="Minimum business quality"
            title="Filter by minimum business quality"
          />
        </div>

        <div className="flex items-center gap-2">
          <span
            className="min-w-[42px] rounded-[10px] px-2.5 py-1 text-center text-[14px] font-bold leading-none tabular-nums"
            style={{
              color: active ? accent : '#94a3b8',
              background: active ? `${accent}13` : 'rgba(255,255,255,0.025)',
              border: `1px solid ${active ? `${accent}34` : 'rgba(255,255,255,0.055)'}`,
              boxShadow: active ? `0 0 18px -14px ${accent}` : 'none',
            }}
            title="Minimum quality"
          >
            {draftValue}
          </span>
          {active && (
            <button
              type="button"
              onClick={() => {
                setDraftValue(0);
                flushCommit(0);
              }}
              className="inline-flex h-7 w-7 items-center justify-center rounded-[9px] transition-all active:scale-[0.96] hover:-translate-y-[1px]"
              style={{
                color: 'var(--text-secondary)',
                background: 'rgba(255,255,255,0.025)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
              title="Reset quality floor"
            >
              <RotateCcw className="w-3 h-3" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
