/**
 * BuySellZoneCharts — Four mini charts (1M, 3M, 6M, 12M)
 * showing historical prices with forward-looking buy/sell zones.
 *
 * Each chart renders:
 * - Historical close prices with signal-colored line
 * - Buy zone (green shaded region below entry) or Sell zone (red above)
 * - Forward projection cone colored by signal
 * - Median expected price path
 * - Current price reference line
 * - Signal badge + expected return + P(up)
 */
import { useMemo, useState } from 'react';
import {
  ComposedChart, Area, Line, XAxis, YAxis, ReferenceLine, ReferenceArea,
  ResponsiveContainer, Tooltip as RTooltip,
} from 'recharts';
import type { OHLCVBar } from '../api';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';

/* ── Types ──────────────────────────────────────────────────────── */
interface Forecast {
  horizon_days: number;
  expected_return_pct: number;
  probability_up: number;
  signal_label: string;
}

interface Props {
  ohlcv: OHLCVBar[];
  forecasts: Forecast[];
  symbol: string;
  compact?: boolean;
}

interface ChartDataPoint {
  date: string;
  close: number | null;
  median: number | null;
  ciBase: number | null;
  ciBand: number | null;
}

/* ── Config ─────────────────────────────────────────────────────── */
// horizonDays = trading-day value matching API (21,63,126,252)
// calendarDays = calendar-day count for projection display
const ZONE_CONFIGS = [
  { label: '1M',  calendarDays: 30,  horizonDays: 21 },
  { label: '3M',  calendarDays: 90,  horizonDays: 63 },
  { label: '6M',  calendarDays: 180, horizonDays: 126 },
  { label: '12M', calendarDays: 365, horizonDays: 252 },
] as const;

/* ── Signal Colors ──────────────────────────────────────────────── */
const BUY_RGB = '52,211,153';
const SELL_RGB = '251,113,133';
const HOLD_RGB = '148,163,184';

interface SignalStyle {
  kind: 'buy' | 'sell' | 'hold';
  pillBg: string; pillFg: string;
  rgb: string; primary: string;
  glow: string;
}

function getSignalStyle(label: string): SignalStyle {
  const u = (label || '').toUpperCase().replace(/[\s-]/g, '_');
  if (u.includes('STRONG_BUY'))
    return { kind: 'buy', pillBg: 'linear-gradient(135deg, #064e3b, #047857)', pillFg: 'var(--accent-emerald)', rgb: BUY_RGB, primary: 'var(--accent-emerald)', glow: 'rgba(62,232,165,0.18)' };
  if (u.includes('BUY'))
    return { kind: 'buy', pillBg: 'linear-gradient(135deg, #064e3b, #065f46)', pillFg: '#6EE7B7', rgb: BUY_RGB, primary: 'var(--accent-emerald)', glow: 'var(--emerald-12)' };
  if (u.includes('STRONG_SELL'))
    return { kind: 'sell', pillBg: 'linear-gradient(135deg, #4c0519, #9f1239)', pillFg: 'var(--accent-rose)', rgb: SELL_RGB, primary: 'var(--accent-rose)', glow: 'rgba(255,107,138,0.18)' };
  if (u.includes('SELL'))
    return { kind: 'sell', pillBg: 'linear-gradient(135deg, #4c0519, #881337)', pillFg: '#FDA4AF', rgb: SELL_RGB, primary: 'var(--accent-rose)', glow: 'var(--rose-12)' };
  return { kind: 'hold', pillBg: 'linear-gradient(135deg, #1e1b4b, #312e81)', pillFg: '#A5B4FC', rgb: HOLD_RGB, primary: '#94A3B8', glow: 'rgba(148,163,184,0.08)' };
}


/* ── Seeded PRNG (Mulberry32) ── deterministic across re-renders ── */
function mulberry32(seed: number) {
  return () => {
    let t = (seed += 0x6D2B79F5);
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Box-Muller: uniform → standard normal */
function boxMuller(rng: () => number): number {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Hash a string to a seed integer */
function hashSeed(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) {
    h = ((h << 5) - h + s.charCodeAt(i)) | 0;
  }
  return h >>> 0;
}

/* ── Compute chart data with GBM simulation ─────────────────────── */
/**
 * Generates a realistic-looking forward projection using Geometric Brownian
 * Motion (GBM) conditioned on the model's expected return and realized vol.
 *
 * The median path is a *single simulated GBM trajectory* with:
 *   dS = mu*S*dt + sigma*S*dW
 *
 * - mu  = annualized drift implied by forecast expected return
 * - sigma = realized daily vol from history
 * - dW  = seeded Gaussian increments (deterministic per symbol+horizon)
 *
 * The CI envelope widens as sqrt(t) per standard diffusion theory and is
 * centered on the *deterministic drift path* (not the simulated path) so
 * the band correctly represents the model's statistical uncertainty.
 *
 * Projection granularity is daily (not 8-16 coarse steps) to produce
 * the natural oscillations the user expects. For 12M we subsample
 * every 2 days to keep point count reasonable (~180 points).
 */
function buildChartData(
  ohlcv: OHLCVBar[],
  forecast: Forecast | undefined,
  calendarDays: number,
  symbol: string,
) {
  const sliced = ohlcv.slice(-calendarDays);
  if (sliced.length === 0)
    return { data: [] as ChartDataPoint[], lastPrice: 0, targetPrice: 0, ciLow: 0, ciHigh: 0 };

  const lastPrice = sliced[sliced.length - 1].close;
  const lastDate = new Date(sliced[sliced.length - 1].time);

  const expRet = forecast ? forecast.expected_return_pct / 100 : 0;
  const targetPrice = lastPrice * (1 + expRet);

  // Realized vol from log-returns
  const logRets = sliced.slice(1).map((d, i) => Math.log(d.close / sliced[i].close));
  const dailyVol = logRets.length > 5
    ? Math.sqrt(logRets.reduce((s, r) => s + r * r, 0) / logRets.length)
    : 0.02;

  // Annualized drift implied by the forecast (mu in continuous time)
  // expRet = e^(mu*T) - 1 → mu = ln(1+expRet) / T_years
  const T_years = calendarDays / 365;
  const mu = Math.log(1 + expRet) / Math.max(T_years, 1 / 365);
  const sigmaAnn = dailyVol * Math.sqrt(252);

  // CI width: 1-sigma diffusion envelope at terminal
  const ciWidth = lastPrice * dailyVol * Math.sqrt(calendarDays) * 0.8;
  const ciHigh = targetPrice + ciWidth;
  const ciLow = Math.max(0, targetPrice - ciWidth);

  // Historical points
  const pts: ChartDataPoint[] = sliced.map(d => ({
    date: d.time, close: d.close, median: null, ciBase: null, ciBand: null,
  }));

  // Bridge
  if (pts.length > 0) {
    const last = pts[pts.length - 1];
    last.median = last.close;
    last.ciBase = last.close;
    last.ciBand = 0;
  }

  // ── GBM Simulation ──
  // Deterministic seed from symbol + horizon for stable renders
  const rng = mulberry32(hashSeed(`${symbol}-${calendarDays}`));

  // Step size (days). Subsample longer horizons to keep ~90-120 points.
  const step = calendarDays <= 90 ? 1 : calendarDays <= 180 ? 2 : 3;
  const dt = step / 365; // fraction of year per step

  let simPrice = lastPrice;

  for (let day = step; day <= calendarDays; day += step) {
    const frac = day / calendarDays;
    const dt_step = dt;

    // GBM increment: S_{t+1} = S_t * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    const Z = boxMuller(rng);
    const drift = (mu - 0.5 * sigmaAnn * sigmaAnn) * dt_step;
    const diffusion = sigmaAnn * Math.sqrt(dt_step) * Z;
    simPrice = simPrice * Math.exp(drift + diffusion);

    // Apply soft mean-reversion toward deterministic path to prevent extreme wandering
    // This is an OU bridge: pull simulated path gently toward the expected path
    const deterministicPrice = lastPrice * Math.exp(mu * (day / 365));
    const reversionStrength = 0.03; // subtle pull
    simPrice = simPrice + (deterministicPrice - simPrice) * reversionStrength;

    // Ensure positive
    simPrice = Math.max(simPrice, lastPrice * 0.01);

    // Deterministic CI envelope centered on drift path
    const ciAtStep = ciWidth * Math.sqrt(frac);
    const envCenter = lastPrice + (targetPrice - lastPrice) * frac;
    const lo = Math.max(0, envCenter - ciAtStep);
    const hi = envCenter + ciAtStep;

    const projDate = new Date(lastDate);
    projDate.setDate(projDate.getDate() + day);

    pts.push({
      date: projDate.toISOString().split('T')[0],
      close: null,
      median: simPrice,
      ciBase: lo,
      ciBand: hi - lo,
    });
  }

  // Nudge final simulated point to land near target (within 1 vol unit)
  // so the projection "arrives" at the forecast, not at a random walk endpoint.
  if (pts.length > 0) {
    const lastPt = pts[pts.length - 1];
    if (lastPt.median != null) {
      // Blend 70% toward target at terminal to anchor the endpoint
      lastPt.median = lastPt.median * 0.3 + targetPrice * 0.7;
    }
  }

  return { data: pts, lastPrice, targetPrice, ciLow, ciHigh };
}


/* ── Custom Tooltip ──────────────────────────────────────────────── */
function ZoneTooltip({ active, payload, st }: any) {
  if (!active || !payload?.length) return null;
  const d = payload[0]?.payload as ChartDataPoint;
  if (!d) return null;
  const price = d.close ?? d.median;
  if (price == null) return null;
  const s: SignalStyle = st;

  return (
    <div
      className="rounded-lg px-3 py-2"
      style={{
        background: 'rgba(8,4,20,0.96)',
        border: `1px solid rgba(${s.rgb},0.20)`,
        backdropFilter: 'blur(16px)',
        boxShadow: `0 8px 32px rgba(0,0,0,0.5), 0 0 12px ${s.glow}`,
      }}
    >
      <div className="text-[9px] mb-0.5" style={{ color: 'var(--text-muted)' }}>{d.date}</div>
      <div className="text-[12px] font-bold tabular-nums" style={{ color: 'var(--text-primary)' }}>
        ${price.toFixed(2)}
      </div>
      {d.ciBase != null && d.ciBand != null && d.ciBand > 0 && (
        <div className="text-[9px] mt-0.5 tabular-nums" style={{ color: s.primary }}>
          ${d.ciBase.toFixed(2)} &ndash; ${(d.ciBase + d.ciBand).toFixed(2)}
        </div>
      )}
    </div>
  );
}


/* ── Single Zone Chart ───────────────────────────────────────────── */
function ZoneChart({
  config, ohlcv, forecast, hovered, onHover, symbol,
}: {
  config: typeof ZONE_CONFIGS[number];
  ohlcv: OHLCVBar[];
  forecast: Forecast | undefined;
  hovered: boolean;
  onHover: (h: boolean) => void;
  symbol: string;
}) {
  const { data, lastPrice, targetPrice, ciLow, ciHigh } = useMemo(
    () => buildChartData(ohlcv, forecast, config.calendarDays, symbol),
    [ohlcv, forecast, config, symbol],
  );

  if (data.length === 0 || lastPrice === 0) return null;

  const st = forecast ? getSignalStyle(forecast.signal_label) : getSignalStyle('HOLD');
  const expRet = forecast?.expected_return_pct ?? 0;
  const pUp = forecast?.probability_up ?? 0.5;
  const signalLabel = forecast?.signal_label ?? 'N/A';
  const isPos = expRet >= 0;
  const isBuy = st.kind === 'buy';
  const isSell = st.kind === 'sell';

  // Y domain with padding
  const allPrices = data.flatMap(d => {
    const v: number[] = [];
    if (d.close != null) v.push(d.close);
    if (d.ciBase != null) v.push(d.ciBase);
    if (d.ciBase != null && d.ciBand != null) v.push(d.ciBase + d.ciBand);
    if (d.median != null) v.push(d.median);
    return v;
  });
  const lo = Math.min(...allPrices);
  const hi = Math.max(...allPrices);
  const pad = (hi - lo) * 0.10;
  const yMin = lo - pad;
  const yMax = hi + pad;

  // Buy zone: region below current price. Sell zone: region above.
  const zoneY1 = isBuy ? yMin : isSell ? lastPrice : undefined;
  const zoneY2 = isBuy ? lastPrice : isSell ? yMax : undefined;

  const uid = `z${config.calendarDays}`;

  return (
    <div
      className="overflow-hidden rounded-xl transition-all duration-300"
      style={{
        background: 'linear-gradient(170deg, rgba(8,4,22,0.97) 0%, rgba(6,12,32,0.97) 100%)',
        border: `1px solid ${hovered ? `rgba(${st.rgb},0.28)` : 'rgba(255,255,255,0.04)'}`,
        boxShadow: hovered
          ? `0 8px 32px rgba(0,0,0,0.4), 0 0 20px ${st.glow}`
          : '0 2px 10px rgba(0,0,0,0.2)',
        transform: hovered ? 'translateY(-1px)' : 'translateY(0)',
      }}
      onMouseEnter={() => onHover(true)}
      onMouseLeave={() => onHover(false)}
    >
      {/* ── Header ────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-3.5 py-2"
        style={{ borderBottom: `1px solid rgba(${st.rgb},0.08)` }}
      >
        <div className="flex items-center gap-2">
          <span className="text-[12px] font-semibold" style={{ color: 'var(--text-primary)' }}>
            {config.label}
          </span>
          <span
            className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md text-[9px] font-bold uppercase"
            style={{ background: st.pillBg, color: st.pillFg }}
          >
            {isPos
              ? <TrendingUp className="w-3 h-3" />
              : expRet < 0
                ? <TrendingDown className="w-3 h-3" />
                : <Minus className="w-3 h-3" />}
            {signalLabel}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span
            className="text-[15px] font-bold tabular-nums"
            style={{ color: st.primary, textShadow: `0 0 10px ${st.glow}` }}
          >
            {isPos ? '+' : ''}{expRet.toFixed(1)}%
          </span>
          <div className="flex items-center gap-1">
            <div className="relative h-[4px] rounded-full overflow-hidden" style={{ width: 24, background: 'rgba(255,255,255,0.05)' }}>
              <div
                className="absolute inset-y-0 left-0 rounded-full"
                style={{ width: `${Math.round(pUp * 100)}%`, background: st.primary, transition: 'width 400ms ease' }}
              />
            </div>
            <span className="text-[9px] tabular-nums" style={{ color: 'var(--text-muted)' }}>
              {(pUp * 100).toFixed(0)}%
            </span>
          </div>
        </div>
      </div>

      {/* ── Chart ─────────────────────────────────────────── */}
      <div className="px-1 pt-1 pb-0">
        <ResponsiveContainer width="100%" height={170}>
          <ComposedChart data={data} margin={{ top: 6, right: 6, bottom: 0, left: 6 }}>
            <defs>
              <linearGradient id={`${uid}-hist`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor={`rgba(${st.rgb},0.18)`} />
                <stop offset="95%" stopColor={`rgba(${st.rgb},0.01)`} />
              </linearGradient>
              <linearGradient id={`${uid}-ci`} x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor={`rgba(${st.rgb},0.32)`} />
                <stop offset="100%" stopColor={`rgba(${st.rgb},0.05)`} />
              </linearGradient>
            </defs>

            <XAxis dataKey="date" type="category" tick={false} axisLine={false} tickLine={false} />
            <YAxis domain={[yMin, yMax]} hide />

            <RTooltip
              content={<ZoneTooltip st={st} />}
              cursor={{ stroke: `rgba(${st.rgb},0.20)`, strokeDasharray: '4 4' }}
            />

            {/* ── BUY / SELL ZONE: full-width shaded band ── */}
            {zoneY1 != null && zoneY2 != null && (
              <ReferenceArea
                y1={zoneY1}
                y2={zoneY2}
                fill={isBuy ? 'rgba(62,232,165,0.07)' : 'rgba(255,107,138,0.07)'}
                fillOpacity={1}
                stroke="none"
              />
            )}

            {/* Zone label at current price level */}
            {(isBuy || isSell) && (
              <ReferenceLine
                y={lastPrice}
                stroke={`rgba(${st.rgb},0.35)`}
                strokeDasharray="3 3"
                strokeWidth={1}
                label={{
                  value: isBuy ? 'BUY ZONE' : 'SELL ZONE',
                  position: isBuy ? 'insideBottomRight' : 'insideTopRight',
                  fill: `rgba(${st.rgb},0.55)`,
                  fontSize: 8,
                  fontWeight: 700,
                }}
              />
            )}

            {/* Target price line */}
            {Math.abs(targetPrice - lastPrice) > lastPrice * 0.002 && (
              <ReferenceLine
                y={targetPrice}
                stroke={`rgba(${st.rgb},0.22)`}
                strokeDasharray="6 4"
                strokeWidth={1}
              />
            )}

            {/* CI projection band (stacked) */}
            <Area dataKey="ciBase" stackId="ci" stroke="none" fill="transparent" fillOpacity={0} isAnimationActive={false} connectNulls={false} />
            <Area dataKey="ciBand" stackId="ci" stroke="none" fill={`url(#${uid}-ci)`} fillOpacity={1} isAnimationActive={true} animationDuration={700} connectNulls={false} />

            {/* Historical price: signal-colored */}
            <Area
              dataKey="close"
              stroke={`rgba(${st.rgb},0.80)`}
              strokeWidth={1.5}
              fill={`url(#${uid}-hist)`}
              fillOpacity={1}
              isAnimationActive={true}
              animationDuration={500}
              connectNulls={false}
              dot={false}
            />

            {/* Median projection line */}
            <Line
              dataKey="median"
              stroke={st.primary}
              strokeWidth={2}
              strokeDasharray="5 4"
              dot={false}
              isAnimationActive={true}
              animationDuration={800}
              connectNulls={false}
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Footer ────────────────────────────────────────── */}
      <div
        className="flex items-center justify-between px-3.5 py-1.5"
        style={{ borderTop: `1px solid rgba(${st.rgb},0.06)` }}
      >
        <div className="flex items-center gap-1">
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Now</span>
          <span className="text-[11px] font-bold tabular-nums" style={{ color: 'var(--text-primary)' }}>
            ${lastPrice.toFixed(2)}
          </span>
          <svg width="14" height="8" viewBox="0 0 14 8" className="mx-0.5">
            <path
              d="M1 4 L10 4 L8 2 M10 4 L8 6"
              stroke={st.primary}
              strokeWidth="1.2"
              fill="none"
              strokeLinecap="round"
              strokeLinejoin="round"
            />
          </svg>
          <span className="text-[11px] font-bold tabular-nums" style={{ color: st.primary }}>
            ${targetPrice.toFixed(2)}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <span className="text-[9px]" style={{ color: 'var(--text-muted)' }}>Range</span>
          <span className="text-[10px] tabular-nums" style={{ color: 'var(--text-secondary)' }}>
            ${ciLow.toFixed(2)} &ndash; ${ciHigh.toFixed(2)}
          </span>
        </div>
      </div>
    </div>
  );
}


/* ── Main Component ──────────────────────────────────────────────── */
export default function BuySellZoneCharts({ ohlcv, forecasts, symbol, compact }: Props) {
  const [hoveredIdx, setHoveredIdx] = useState<number | null>(null);

  const forecastMap = useMemo(() => {
    const m = new Map<number, Forecast>();
    for (const f of forecasts) m.set(f.horizon_days, f);
    return m;
  }, [forecasts]);

  if (!ohlcv?.length || !forecasts?.length) return null;

  return (
    <div className={compact ? 'mt-1' : 'mt-5'}>
      {!compact && (
        <div className="flex items-center gap-2 mb-3">
          <h3 className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">
            Buy &amp; Sell Zones
          </h3>
          <div className="flex-1 h-px" style={{ background: 'var(--border-void)' }} />
        </div>
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {ZONE_CONFIGS.map((config, idx) => (
          <div key={config.calendarDays} className={`fade-up-delay-${idx + 1}`}>
            <ZoneChart
              config={config}
              ohlcv={ohlcv}
              forecast={forecastMap.get(config.horizonDays)}
              hovered={hoveredIdx === idx}
              onHover={(h) => setHoveredIdx(h ? idx : null)}
              symbol={symbol}
            />
          </div>
        ))}
      </div>
    </div>
  );
}
