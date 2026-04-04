/**
 * SignalDistributionBar -- Flowing gradient bar showing signal distribution.
 *
 * Replaces the donut chart with a continuous gradient strip where segments
 * bleed into each other. Includes 7-day history sparkline, hover tooltips,
 * and a shift summary sentence.
 */
import { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import type { SignalStats, SectorGroup } from '../api';

const HISTORY_KEY = 'signal_dist_history';
const MAX_HISTORY_DAYS = 7;

interface DistSnapshot {
  date: string; // YYYY-MM-DD
  strong_sell: number;
  sell: number;
  hold: number;
  buy: number;
  strong_buy: number;
}

interface SegmentDef {
  key: string;
  label: string;
  count: number;
  color: string;
  bgGradient: string;
}

interface Props {
  signals: SignalStats;
  sectors?: SectorGroup[];
  onFilterSignal?: (category: string | null) => void;
}

function getToday(): string {
  return new Date().toISOString().slice(0, 10);
}

function loadHistory(): DistSnapshot[] {
  try {
    const raw = localStorage.getItem(HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch {
    return [];
  }
}

function saveHistory(history: DistSnapshot[]) {
  try { localStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(-MAX_HISTORY_DAYS))); } catch { /* noop */ }
}

/**
 * Returns the top 3 tickers for a given signal category from sector data.
 */
function topTickersForCategory(sectors: SectorGroup[], category: string): string[] {
  const tickers: string[] = [];
  for (const sec of sectors) {
    for (const asset of sec.assets) {
      // Get the nearest signal label to determine category
      const nearest = asset.nearest_label?.toLowerCase() || '';
      if (category === 'Strong Buy' && nearest.includes('strong') && nearest.includes('buy')) {
        tickers.push(asset.asset_label);
      } else if (category === 'Buy' && nearest.includes('buy') && !nearest.includes('strong')) {
        tickers.push(asset.asset_label);
      } else if (category === 'Hold' && nearest.includes('hold')) {
        tickers.push(asset.asset_label);
      } else if (category === 'Sell' && nearest.includes('sell') && !nearest.includes('strong')) {
        tickers.push(asset.asset_label);
      } else if (category === 'Strong Sell' && nearest.includes('strong') && nearest.includes('sell')) {
        tickers.push(asset.asset_label);
      }
      if (tickers.length >= 3) return tickers;
    }
  }
  return tickers;
}

export default function SignalDistributionBar({ signals, sectors, onFilterSignal }: Props) {
  const navigate = useNavigate();
  const [hoveredSegment, setHoveredSegment] = useState<string | null>(null);
  const [tooltipPos, setTooltipPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [animatedWidth, setAnimatedWidth] = useState(0);
  const barRef = useRef<HTMLDivElement>(null);

  // Save today's snapshot (once per day)
  useEffect(() => {
    const today = getToday();
    const history = loadHistory();
    const alreadyHas = history.some(h => h.date === today);
    if (!alreadyHas) {
      history.push({
        date: today,
        strong_sell: signals.strong_sell_signals,
        sell: signals.sell_signals,
        hold: signals.hold_signals,
        buy: signals.buy_signals,
        strong_buy: signals.strong_buy_signals,
      });
      saveHistory(history);
    }
  }, [signals]);

  const history = useMemo(() => loadHistory(), [signals]);

  // Animate bar expansion from center
  useEffect(() => {
    const raf = requestAnimationFrame(() => {
      setTimeout(() => setAnimatedWidth(100), 50);
    });
    return () => cancelAnimationFrame(raf);
  }, []);

  // Build segments
  const segments: SegmentDef[] = useMemo(() => [
    {
      key: 'Strong Sell',
      label: 'Strong Sell',
      count: signals.strong_sell_signals,
      color: '#FB7185',
      bgGradient: 'linear-gradient(135deg, #4c0519 0%, #6b0f2a 50%, #881337 100%)',
    },
    {
      key: 'Sell',
      label: 'Sell',
      count: signals.sell_signals,
      color: 'rgba(251,113,133,0.5)',
      bgGradient: 'linear-gradient(135deg, #4c0519 0%, #881337 100%)',
    },
    {
      key: 'Hold',
      label: 'Hold',
      count: signals.hold_signals,
      color: '#1c1845',
      bgGradient: 'linear-gradient(135deg, #16133a 0%, #1c1845 100%)',
    },
    {
      key: 'Buy',
      label: 'Buy',
      count: signals.buy_signals,
      color: 'rgba(52,211,153,0.5)',
      bgGradient: 'linear-gradient(135deg, #064e3b 0%, #047857 100%)',
    },
    {
      key: 'Strong Buy',
      label: 'Strong Buy',
      count: signals.strong_buy_signals,
      color: '#34D399',
      bgGradient: 'linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%)',
    },
  ], [signals]);

  const total = useMemo(() => segments.reduce((s, seg) => s + seg.count, 0), [segments]);

  // Shift summary
  const shiftSummary = useMemo(() => {
    if (history.length < 2) return null;
    const oldest = history[0];
    const latest = history[history.length - 1];
    const oldBullish = oldest.buy + oldest.strong_buy;
    const newBullish = latest.buy + latest.strong_buy;
    const oldTotal = oldest.strong_sell + oldest.sell + oldest.hold + oldest.buy + oldest.strong_buy;
    const newTotal = latest.strong_sell + latest.sell + latest.hold + latest.buy + latest.strong_buy;
    if (oldTotal === 0 || newTotal === 0) return null;

    const oldPct = (oldBullish / oldTotal) * 100;
    const newPct = (newBullish / newTotal) * 100;
    const diff = newPct - oldPct;
    const days = history.length;

    if (Math.abs(diff) < 1) {
      return { text: `Stable this week`, direction: 'neutral' as const };
    }
    return {
      text: `Distribution shifted ${diff > 0 ? '+' : ''}${diff.toFixed(0)}% ${diff > 0 ? 'bullish' : 'bearish'} over ${days} day${days > 1 ? 's' : ''}`,
      direction: diff > 0 ? ('bullish' as const) : ('bearish' as const),
    };
  }, [history]);

  const handleSegmentHover = useCallback((key: string | null, e?: React.MouseEvent) => {
    setHoveredSegment(key);
    if (e && barRef.current) {
      const rect = barRef.current.getBoundingClientRect();
      setTooltipPos({
        x: e.clientX - rect.left,
        y: e.clientY - rect.top - 12,
      });
    }
  }, []);

  const handleSegmentClick = useCallback((key: string) => {
    onFilterSignal?.(key);
  }, [onFilterSignal]);

  // Build mini-bar data for history sparkline
  function miniBarSegments(snap: DistSnapshot) {
    const t = snap.strong_sell + snap.sell + snap.hold + snap.buy + snap.strong_buy;
    if (t === 0) return [];
    return [
      { pct: (snap.strong_sell / t) * 100, color: '#FB7185' },
      { pct: (snap.sell / t) * 100, color: 'rgba(251,113,133,0.5)' },
      { pct: (snap.hold / t) * 100, color: '#1c1845' },
      { pct: (snap.buy / t) * 100, color: 'rgba(52,211,153,0.5)' },
      { pct: (snap.strong_buy / t) * 100, color: '#34D399' },
    ];
  }

  const hoveredSeg = segments.find(s => s.key === hoveredSegment);
  const hoveredPct = hoveredSeg && total > 0 ? ((hoveredSeg.count / total) * 100).toFixed(1) : '0';
  const hoveredTickers = hoveredSeg && sectors ? topTickersForCategory(sectors, hoveredSeg.key) : [];

  return (
    <div className="glass-card p-6 hover-lift">
      <h3
        className="text-[13px] font-medium tracking-wide mb-5"
        style={{ color: 'var(--text-secondary, #94a3b8)' }}
      >
        Signal Distribution
      </h3>

      {/* Flowing gradient bar */}
      <div
        ref={barRef}
        className="relative h-3 rounded-lg overflow-hidden flex"
        style={{
          width: `${animatedWidth}%`,
          margin: '0 auto',
          transition: 'width 400ms cubic-bezier(0.16, 1, 0.3, 1)',
          background: 'var(--void-surface, #0a0a23)',
        }}
      >
        {segments.map((seg) => {
          const pct = total > 0 ? (seg.count / total) * 100 : 0;
          if (pct === 0) return null;
          return (
            <div
              key={seg.key}
              className="h-full relative cursor-pointer"
              style={{
                width: `${pct}%`,
                background: seg.color,
                transition: 'box-shadow 150ms ease',
                boxShadow: hoveredSegment === seg.key
                  ? '0 0 12px rgba(139,92,246,0.15)'
                  : 'none',
              }}
              onMouseEnter={(e) => handleSegmentHover(seg.key, e)}
              onMouseMove={(e) => handleSegmentHover(seg.key, e)}
              onMouseLeave={() => handleSegmentHover(null)}
              onClick={() => handleSegmentClick(seg.key)}
            />
          );
        })}

        {/* Tooltip */}
        {hoveredSeg && (
          <div
            className="absolute z-50 pointer-events-none"
            style={{
              left: tooltipPos.x,
              top: tooltipPos.y - 8,
              transform: 'translate(-50%, -100%)',
            }}
          >
            <div
              className="rounded-xl px-4 py-3 min-w-[160px]"
              style={{
                background: 'linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)',
                border: '1px solid rgba(139,92,246,0.25)',
                backdropFilter: 'blur(20px)',
                boxShadow: '0 8px 32px rgba(139,92,246,0.12), 0 0 80px rgba(139,92,246,0.05)',
              }}
            >
              <div className="flex items-center gap-2 mb-1.5">
                <span className="w-2 h-2 rounded-full" style={{ background: hoveredSeg.color }} />
                <span className="text-xs font-medium" style={{ color: '#e2e8f0' }}>
                  {hoveredSeg.label}
                </span>
              </div>
              <div className="flex items-baseline gap-2">
                <span className="text-lg font-bold tabular-nums" style={{ color: '#f8fafc' }}>
                  {hoveredSeg.count}
                </span>
                <span className="text-[10px]" style={{ color: '#94a3b8' }}>
                  ({hoveredPct}%)
                </span>
              </div>
              {hoveredTickers.length > 0 && (
                <div className="flex flex-wrap gap-1 mt-2">
                  {hoveredTickers.map(t => (
                    <button
                      key={t}
                      className="px-2 py-0.5 rounded text-[10px] font-medium pointer-events-auto cursor-pointer border-none"
                      style={{
                        background: 'rgba(139,92,246,0.12)',
                        color: '#C4B5FD',
                      }}
                      onClick={(e) => { e.stopPropagation(); navigate(`/charts/${t}`); }}
                    >
                      {t}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Legend row */}
      <div className="flex flex-wrap justify-center gap-4 mt-3">
        {segments.map(seg => (
          <div
            key={seg.key}
            className="flex items-center gap-1.5 text-[10px] cursor-pointer"
            onClick={() => handleSegmentClick(seg.key)}
            style={{ opacity: hoveredSegment && hoveredSegment !== seg.key ? 0.4 : 1, transition: 'opacity 150ms' }}
          >
            <span className="w-2 h-2 rounded-full" style={{ background: seg.color }} />
            <span style={{ color: 'var(--text-muted, #475569)' }}>{seg.label}</span>
            <span className="font-semibold tabular-nums" style={{ color: '#f8fafc' }}>{seg.count}</span>
          </div>
        ))}
      </div>

      {/* 7-day history sparkline */}
      {history.length > 1 && (
        <div className="mt-4 space-y-0.5">
          {history.slice(-MAX_HISTORY_DAYS).map((snap, i) => {
            const segs = miniBarSegments(snap);
            return (
              <div key={snap.date} className="flex items-center gap-2">
                <span className="text-[9px] tabular-nums w-10 text-right flex-shrink-0"
                  style={{ color: 'var(--text-muted, #475569)' }}>
                  {snap.date.slice(5)}
                </span>
                <div
                  className="flex h-[2px] rounded-full overflow-hidden flex-1"
                  style={{
                    opacity: 0.4 + 0.6 * (i / Math.max(history.length - 1, 1)),
                  }}
                >
                  {segs.map((s, j) => (
                    <div
                      key={j}
                      className="h-full"
                      style={{ width: `${s.pct}%`, background: s.color }}
                    />
                  ))}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Shift summary */}
      {shiftSummary && (
        <p className="text-xs mt-3 text-center" style={{ color: 'var(--text-secondary, #94a3b8)' }}>
          {shiftSummary.direction === 'neutral' ? (
            <span style={{ color: 'var(--text-muted, #475569)' }}>{shiftSummary.text}</span>
          ) : (
            <>
              Distribution shifted{' '}
              <span style={{
                color: shiftSummary.direction === 'bullish' ? '#34D399' : '#FB7185',
                fontWeight: 600,
              }}>
                {shiftSummary.text.match(/[+-]\d+%\s\w+/)?.[0] || shiftSummary.text}
              </span>
              {' '}over {history.length} day{history.length > 1 ? 's' : ''}
            </>
          )}
        </p>
      )}
    </div>
  );
}
