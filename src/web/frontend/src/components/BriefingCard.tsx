/**
 * BriefingCard -- Morning Briefing Hero with three-column layout:
 *   Left: Since Last Visit (signal changes)
 *   Center: Today's Conviction (highest-conviction signal)
 *   Right: System Pulse (4 micro-gauges)
 *
 * Uses aurora gradient background with sentiment-aware cosmic glow overlay.
 */
import { useMemo, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import type { SignalStats, StrongSignalEntry, RiskSummary, TuneStats, DataSummary } from '../api';
import MicroGauge from './MicroGauge';
import {
  ArrowUpRight, ArrowDownRight, CheckCircle, Sparkles,
} from 'lucide-react';

const LAST_VISIT_KEY = 'briefing_last_visit';
const PREV_SIGNALS_KEY = 'briefing_prev_signals';

interface Props {
  signals: SignalStats;
  tuning: TuneStats;
  dataStatus: DataSummary;
  risk?: RiskSummary;
  strongBuy: StrongSignalEntry[];
  strongSell: StrongSignalEntry[];
}

interface PrevSnapshot {
  timestamp: number;
  strong_buy: number;
  strong_sell: number;
  buy: number;
  sell: number;
  hold: number;
}

export default function BriefingCard({
  signals,
  tuning,
  dataStatus,
  risk,
  strongBuy,
  strongSell,
}: Props) {
  const navigate = useNavigate();
  const [visible, setVisible] = useState(false);

  useEffect(() => {
    requestAnimationFrame(() => setVisible(true));
  }, []);

  // ── Since Last Visit logic ──────────────────────────────────────
  const prevSnapshot: PrevSnapshot | null = useMemo(() => {
    try {
      const raw = localStorage.getItem(PREV_SIGNALS_KEY);
      return raw ? JSON.parse(raw) : null;
    } catch { return null; }
  }, []);

  const lastVisitTs = useMemo(() => {
    try {
      const raw = localStorage.getItem(LAST_VISIT_KEY);
      return raw ? parseInt(raw, 10) : null;
    } catch { return null; }
  }, []);

  // Save current snapshot for next visit
  useEffect(() => {
    const snap: PrevSnapshot = {
      timestamp: Date.now(),
      strong_buy: signals.strong_buy_signals,
      strong_sell: signals.strong_sell_signals,
      buy: signals.buy_signals,
      sell: signals.sell_signals,
      hold: signals.hold_signals,
    };
    try {
      localStorage.setItem(PREV_SIGNALS_KEY, JSON.stringify(snap));
      localStorage.setItem(LAST_VISIT_KEY, String(Date.now()));
    } catch { /* noop */ }
  }, [signals]);

  const signalChanges = useMemo(() => {
    if (!prevSnapshot) return null;
    const changes: { label: string; diff: number; direction: 'bull' | 'bear' | 'neutral' }[] = [];
    const diffBuy = (signals.strong_buy_signals + signals.buy_signals)
      - (prevSnapshot.strong_buy + prevSnapshot.buy);
    const diffSell = (signals.strong_sell_signals + signals.sell_signals)
      - (prevSnapshot.strong_sell + prevSnapshot.sell);
    if (diffBuy !== 0) changes.push({
      label: `${Math.abs(diffBuy)} signal${Math.abs(diffBuy) > 1 ? 's' : ''} ${diffBuy > 0 ? 'upgraded' : 'downgraded'} to Buy`,
      diff: diffBuy,
      direction: diffBuy > 0 ? 'bull' : 'bear',
    });
    if (diffSell !== 0) changes.push({
      label: `${Math.abs(diffSell)} signal${Math.abs(diffSell) > 1 ? 's' : ''} ${diffSell > 0 ? 'upgraded' : 'downgraded'} to Sell`,
      diff: diffSell,
      direction: diffSell < 0 ? 'bull' : 'bear',
    });
    return changes;
  }, [signals, prevSnapshot]);

  // Time since last visit
  const timeSinceVisit = useMemo(() => {
    if (!lastVisitTs) return null;
    const hours = Math.round((Date.now() - lastVisitTs) / (1000 * 60 * 60));
    if (hours < 1) return 'Less than an hour ago';
    if (hours < 24) return `${hours}h ago`;
    const days = Math.round(hours / 24);
    return `${days}d ago`;
  }, [lastVisitTs]);

  // ── Today's Conviction ──────────────────────────────────────────
  const topConviction = useMemo(() => {
    const allStrong = [
      ...strongBuy.map(s => ({ ...s, direction: 'buy' as const })),
      ...strongSell.map(s => ({ ...s, direction: 'sell' as const })),
    ];
    if (allStrong.length === 0) return null;
    // Sort by absolute expected return (highest conviction)
    allStrong.sort((a, b) => Math.abs(b.exp_ret) - Math.abs(a.exp_ret));
    return allStrong[0];
  }, [strongBuy, strongSell]);

  // ── Sentiment glow ──────────────────────────────────────────────
  const sentimentGlow = useMemo(() => {
    const bullish = signals.strong_buy_signals + signals.buy_signals;
    const bearish = signals.strong_sell_signals + signals.sell_signals;
    const total = bullish + bearish + signals.hold_signals;
    if (total === 0) return 'radial-gradient(ellipse at 30% 50%, var(--violet-15) 0%, transparent 70%)';
    const ratio = bullish / total;
    if (ratio > 0.55) return 'radial-gradient(ellipse at 50% 50%, rgba(62,232,165,0.07) 0%, transparent 70%)';
    if (ratio < 0.45) return 'radial-gradient(ellipse at 50% 50%, rgba(255,107,138,0.07) 0%, transparent 70%)';
    return 'radial-gradient(ellipse at 30% 50%, var(--violet-12) 0%, transparent 70%)';
  }, [signals]);

  // ── System Pulse gauges ─────────────────────────────────────────
  const riskTemp = risk?.combined_temperature ?? 0;
  const riskNorm = Math.min(riskTemp / 2.0, 1); // 0-2 scale -> 0-1
  const pitPassRate = tuning.total > 0 ? tuning.pit_pass / tuning.total : 0;
  const dataFresh = dataStatus.total_files > 0
    ? dataStatus.fresh_files / dataStatus.total_files
    : 0;
  const assetCoverage = signals.total_assets > 0
    ? (signals.total_assets - signals.failed) / signals.total_assets
    : 0;

  return (
    <div
      className="briefing-card relative overflow-hidden rounded-3xl mb-10"
      style={{
        background: 'linear-gradient(160deg, #0e0b24 0%, #1a1550 40%, #12102a 70%, #0c1445 100%)',
        border: '1px solid var(--violet-8)',
        boxShadow: '0 4px 60px rgba(0,0,0,0.35), 0 0 100px var(--violet-4), inset 0 1px 0 rgba(255,255,255,0.03)',
        opacity: visible ? 1 : 0,
        transform: visible ? 'translateY(0)' : 'translateY(12px)',
        transition: 'opacity 400ms cubic-bezier(0.16, 1, 0.3, 1), transform 400ms cubic-bezier(0.16, 1, 0.3, 1)',
      }}
    >
      {/* Top edge light */}
      <div
        className="absolute top-0 left-0 right-0 h-px pointer-events-none"
        style={{ background: 'linear-gradient(90deg, transparent 10%, var(--violet-15) 50%, transparent 90%)' }}
      />

      {/* Gradient glass edge (visible on hover) */}
      <div
        className="absolute inset-0 rounded-2xl pointer-events-none opacity-0 hover-parent-edge"
        style={{
          padding: '1px',
          background: 'linear-gradient(135deg, rgba(139,92,246,0.22) 0%, rgba(56,217,245,0.08) 50%, var(--violet-5) 100%)',
          WebkitMask: 'linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0)',
          WebkitMaskComposite: 'xor',
          maskComposite: 'exclude',
          transition: 'opacity 250ms ease',
        }}
      />

      {/* Sentiment-aware cosmic glow overlay */}
      <div
        className="absolute inset-0 pointer-events-none briefing-glow-drift"
        style={{ background: sentimentGlow }}
      />

      {/* Content: Three columns */}
      <div className="relative z-10 grid grid-cols-1 md:grid-cols-3 gap-0">
        {/* ── Left: Since Last Visit ─────────────────────────── */}
        <div
          className="p-6 md:p-8"
          style={{
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(8px)',
            transition: 'opacity 400ms cubic-bezier(0.16, 1, 0.3, 1), transform 400ms cubic-bezier(0.16, 1, 0.3, 1)',
            transitionDelay: '0ms',
          }}
        >
          <h3
            className="text-[11px] font-medium uppercase tracking-widest mb-4"
            style={{ color: 'var(--text-muted, #6b7a90)' }}
          >
            Since Last Visit
          </h3>

          {!lastVisitTs ? (
            /* First visit */
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: 'var(--violet-8)' }}
              >
                <Sparkles className="w-5 h-5" style={{ color: 'var(--accent-violet, #8B5CF6)' }} />
              </div>
              <div>
                <p className="text-sm font-medium" style={{ color: 'var(--text-violet, #C4B5FD)' }}>
                  Welcome back
                </p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted, #6b7a90)' }}>
                  Your first session this cycle
                </p>
              </div>
            </div>
          ) : signalChanges && signalChanges.length > 0 ? (
            <div className="space-y-3">
              {signalChanges.map((change, i) => (
                <div key={i} className="flex items-center gap-2.5">
                  <div
                    className="w-6 h-6 rounded-lg flex items-center justify-center flex-shrink-0"
                    style={{
                      background: change.direction === 'bull'
                        ? 'linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%)'
                        : 'linear-gradient(135deg, #4c0519 0%, #6b0f2a 50%, #881337 100%)',
                    }}
                  >
                    {change.direction === 'bull'
                      ? <ArrowUpRight className="w-3.5 h-3.5 text-emerald-400" />
                      : <ArrowDownRight className="w-3.5 h-3.5 text-rose-400" />
                    }
                  </div>
                  <span className="text-xs" style={{ color: 'var(--text-primary, #e2e8f0)' }}>
                    {change.label}
                  </span>
                </div>
              ))}
              {timeSinceVisit && (
                <p className="text-[10px] mt-2" style={{ color: 'var(--text-muted, #6b7a90)' }}>
                  Last visit: {timeSinceVisit}
                </p>
              )}
            </div>
          ) : (
            /* No changes */
            <div className="flex items-center gap-3">
              <div
                className="w-10 h-10 rounded-xl flex items-center justify-center"
                style={{ background: 'var(--emerald-8)' }}
              >
                <CheckCircle className="w-5 h-5" style={{ color: 'var(--accent-emerald, #3ee8a5)' }} />
              </div>
              <div>
                <p className="text-sm font-medium" style={{ color: 'var(--text-violet, #C4B5FD)' }}>
                  All caught up
                </p>
                <p className="text-[11px]" style={{ color: 'var(--text-muted, #6b7a90)' }}>
                  No signal changes since {timeSinceVisit}
                </p>
              </div>
            </div>
          )}
        </div>

        {/* ── Fading violet divider ─────────────────────────── */}
        <div className="hidden md:block absolute left-1/3 top-4 bottom-4 w-px"
          style={{ background: 'linear-gradient(180deg, transparent 0%, var(--violet-15) 50%, transparent 100%)' }}
        />

        {/* ── Center: Today's Conviction ─────────────────────── */}
        <div
          className="p-6 md:p-8 flex flex-col items-center justify-center text-center"
          style={{
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(8px)',
            transition: 'opacity 400ms cubic-bezier(0.16, 1, 0.3, 1), transform 400ms cubic-bezier(0.16, 1, 0.3, 1)',
            transitionDelay: '80ms',
          }}
        >
          <h3
            className="text-[11px] font-medium uppercase tracking-widest mb-4"
            style={{ color: 'var(--text-muted, #6b7a90)' }}
          >
            Today&apos;s Conviction
          </h3>

          {topConviction ? (
            <button
              className="flex flex-col items-center gap-2 group cursor-pointer bg-transparent border-none"
              onClick={() => navigate(`/charts/${topConviction.symbol}`)}
            >
              {/* Ticker in gradient text */}
              <span
                className="text-[40px] font-bold tracking-tight leading-none"
                style={{
                  background: 'linear-gradient(135deg, var(--text-luminous) 0%, var(--accent-violet) 100%)',
                  WebkitBackgroundClip: 'text',
                  WebkitTextFillColor: 'transparent',
                  letterSpacing: '-0.025em',
                }}
              >
                {topConviction.symbol}
              </span>

              {/* Signal badge */}
              <span
                className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-[11px] font-medium"
                style={{
                  background: topConviction.direction === 'buy'
                    ? 'linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%)'
                    : 'linear-gradient(135deg, #4c0519 0%, #6b0f2a 50%, #881337 100%)',
                  color: topConviction.direction === 'buy' ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                }}
              >
                {topConviction.direction === 'buy'
                  ? <ArrowUpRight className="w-3 h-3" />
                  : <ArrowDownRight className="w-3 h-3" />
                }
                Strong {topConviction.direction === 'buy' ? 'Buy' : 'Sell'}
              </span>

              {/* Expected return */}
              <span
                className="text-xl font-bold tabular-nums"
                style={{ color: topConviction.direction === 'buy' ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}
              >
                {topConviction.exp_ret >= 0 ? '+' : ''}{(topConviction.exp_ret * 100).toFixed(1)}%
              </span>

              {/* Probability arc */}
              <div className="flex items-center gap-2 mt-1">
                <svg width="20" height="20" viewBox="0 0 20 20">
                  <circle cx="10" cy="10" r="8" fill="none"
                    stroke="var(--violet-10)" strokeWidth="2"
                    strokeDasharray={`${Math.PI * 16 * 0.75} ${Math.PI * 16}`}
                    transform="rotate(135 10 10)" strokeLinecap="round"
                  />
                  <circle cx="10" cy="10" r="8" fill="none"
                    stroke={topConviction.direction === 'buy' ? 'var(--accent-emerald)' : 'var(--accent-rose)'}
                    strokeWidth="2"
                    strokeDasharray={`${Math.PI * 16 * 0.75} ${Math.PI * 16}`}
                    strokeDashoffset={Math.PI * 16 * 0.75 * (1 - topConviction.p_up)}
                    transform="rotate(135 10 10)" strokeLinecap="round"
                  />
                </svg>
                <span className="text-[11px] tabular-nums" style={{ color: 'var(--text-secondary, #94a3b8)' }}>
                  {(topConviction.p_up * 100).toFixed(0)}% probability
                </span>
              </div>

              <span className="text-[10px] mt-1 opacity-0 group-hover:opacity-100 transition-opacity"
                style={{ color: 'var(--text-muted, #6b7a90)' }}>
                Click to view chart
              </span>
            </button>
          ) : (
            <div className="flex flex-col items-center gap-2">
              <span className="text-lg font-medium" style={{ color: 'var(--text-secondary, #94a3b8)' }}>
                No strong signals
              </span>
              <span className="text-[11px]" style={{ color: 'var(--text-muted, #6b7a90)' }}>
                Markets in equilibrium
              </span>
            </div>
          )}
        </div>

        {/* ── Fading violet divider ─────────────────────────── */}
        <div className="hidden md:block absolute left-2/3 top-4 bottom-4 w-px"
          style={{ background: 'linear-gradient(180deg, transparent 0%, var(--violet-15) 50%, transparent 100%)' }}
        />

        {/* ── Right: System Pulse ────────────────────────────── */}
        <div
          className="p-6 md:p-8"
          style={{
            opacity: visible ? 1 : 0,
            transform: visible ? 'translateY(0)' : 'translateY(8px)',
            transition: 'opacity 400ms cubic-bezier(0.16, 1, 0.3, 1), transform 400ms cubic-bezier(0.16, 1, 0.3, 1)',
            transitionDelay: '160ms',
          }}
        >
          <h3
            className="text-[11px] font-medium uppercase tracking-widest mb-4"
            style={{ color: 'var(--text-muted, #6b7a90)' }}
          >
            System Pulse
          </h3>

          <div className="grid grid-cols-2 gap-4">
            <MicroGauge
              id="risk-temp"
              value={riskNorm}
              label={riskTemp.toFixed(1)}
              caption="Risk"
              gradient={['#8B5CF6', 'var(--accent-rose)']}
            />
            <MicroGauge
              id="pit-pass"
              value={pitPassRate}
              label={`${Math.round(pitPassRate * 100)}%`}
              caption="PIT"
              gradient={pitPassRate >= 0.8 ? ['var(--accent-emerald)', '#059669'] : pitPassRate >= 0.6 ? ['var(--accent-amber)', '#D97706'] : ['var(--accent-rose)', '#E11D48']}
            />
            <MicroGauge
              id="data-fresh"
              value={dataFresh}
              label={`${Math.round(dataFresh * 100)}%`}
              caption="Data"
              gradient={dataFresh >= 0.9 ? ['var(--accent-emerald)', '#059669'] : dataFresh >= 0.7 ? ['var(--accent-amber)', '#D97706'] : ['var(--accent-rose)', '#E11D48']}
            />
            <MicroGauge
              id="asset-cov"
              value={assetCoverage}
              label={`${Math.round(assetCoverage * 100)}%`}
              caption="Assets"
              gradient={['#8B5CF6', '#6366F1']}
            />
          </div>
        </div>
      </div>

      {/* Hover edge reveal */}
      <style>{`
        .briefing-card:hover .hover-parent-edge { opacity: 1 !important; }
        @keyframes briefing-glow-drift {
          0%, 100% { background-position: 30% 50%; }
          25% { background-position: 40% 30%; }
          50% { background-position: 60% 50%; }
          75% { background-position: 35% 65%; }
        }
        .briefing-glow-drift {
          background-size: 200% 200%;
          animation: briefing-glow-drift 20s ease-in-out infinite;
        }
      `}</style>
    </div>
  );
}
