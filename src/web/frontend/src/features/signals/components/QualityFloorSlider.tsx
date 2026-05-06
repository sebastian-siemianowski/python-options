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
      className={`relative overflow-hidden rounded-[14px] ${compact ? 'px-3 py-2' : 'px-3.5 py-2.5'} ${className}`}
      style={{
        background: active
          ? `linear-gradient(135deg, ${accent}18, rgba(255,255,255,0.018) 58%, rgba(15,23,42,0.42))`
          : 'linear-gradient(135deg, rgba(255,255,255,0.032), rgba(255,255,255,0.012))',
        border: `1px solid ${active ? `${accent}4f` : 'rgba(255,255,255,0.06)'}`,
        boxShadow: active
          ? `0 0 0 1px ${accent}16, inset 0 1px 0 rgba(255,255,255,0.08), 0 18px 34px -28px ${accent}`
          : 'inset 0 1px 0 rgba(255,255,255,0.035)',
      }}
    >
      <style>{`
        .quality-floor-range {
          -webkit-appearance: none;
          appearance: none;
          height: 38px;
          background: transparent;
          cursor: pointer;
          touch-action: none;
        }
        .quality-floor-range::-webkit-slider-runnable-track {
          height: 30px;
          border-radius: 999px;
          background: transparent;
        }
        .quality-floor-range::-moz-range-track {
          height: 30px;
          border-radius: 999px;
          background: transparent;
        }
        .quality-floor-range::-webkit-slider-thumb {
          -webkit-appearance: none;
          appearance: none;
          width: 38px;
          height: 38px;
          margin-top: -4px;
          border-radius: 999px;
          border: 0;
          background: transparent;
          box-shadow: none;
        }
        .quality-floor-range::-moz-range-thumb {
          width: 38px;
          height: 38px;
          border-radius: 8px;
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

      <div className="flex flex-wrap items-center gap-x-3 gap-y-2">
        <div className="flex min-w-[132px] items-center gap-2">
          <span
            className="inline-flex h-7 w-7 items-center justify-center rounded-[9px]"
            style={{
              color: active ? accent : 'var(--text-secondary)',
              background: active ? `${accent}18` : 'rgba(255,255,255,0.035)',
              border: `1px solid ${active ? `${accent}3d` : 'rgba(255,255,255,0.055)'}`,
            }}
          >
            <ShieldCheck className="w-3.5 h-3.5" />
          </span>
          <div>
            <div className="text-[10px] font-semibold uppercase tracking-[0.16em] text-[var(--text-muted)]">
              Min Quality
            </div>
          </div>
        </div>

        <div className="relative min-w-[220px] flex-1">
          <div
            aria-hidden
            className="absolute left-0 right-0 top-1/2 h-[22px] -translate-y-1/2 rounded-full"
            style={{
              background:
                'linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035))',
              border: '1px solid rgba(255,255,255,0.075)',
              boxShadow:
                'inset 0 1px 2px rgba(255,255,255,0.07), inset 0 -10px 18px rgba(2,6,23,0.22)',
            }}
          />
          <div
            aria-hidden
            className="absolute left-[3px] top-1/2 h-[16px] -translate-y-1/2 rounded-full"
            style={{
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
                width: 18,
                height: 32,
                background:
                  'linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(241,245,249,0.94) 42%, rgba(203,213,225,0.82) 100%)',
                border: '1px solid rgba(255,255,255,0.78)',
                boxShadow: isDragging
                  ? `0 0 0 5px ${accent}1f, 0 13px 24px -15px rgba(0,0,0,0.86), 0 0 18px -7px ${accent}, inset 0 1px 0 rgba(255,255,255,0.96)`
                  : '0 9px 20px -15px rgba(0,0,0,0.92), inset 0 1px 0 rgba(255,255,255,0.94), inset 0 -7px 12px rgba(15,23,42,0.10)',
                transform: isDragging ? 'scale(1.035)' : 'scale(1)',
                transition: isDragging ? 'none' : 'transform 160ms ease, box-shadow 160ms ease',
              }}
            >
              <span
                className="absolute inset-x-[5px] top-[5px] h-px rounded-full"
                style={{ background: 'rgba(255,255,255,0.95)' }}
              />
              <span
                className="h-4 w-[2px] rounded-full"
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
            className="min-w-[54px] rounded-[12px] px-3 py-1.5 text-center text-[18px] font-bold leading-none tabular-nums"
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
              className="inline-flex h-8 w-8 items-center justify-center rounded-[10px] transition-all active:scale-[0.96] hover:-translate-y-[1px]"
              style={{
                color: 'var(--text-secondary)',
                background: 'rgba(255,255,255,0.025)',
                border: '1px solid rgba(255,255,255,0.06)',
              }}
              title="Reset quality floor"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
