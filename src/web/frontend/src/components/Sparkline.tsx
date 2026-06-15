import { useRef, useEffect, useState, memo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type OHLCVBar } from '../api';
import { isHeikinAshiUp, toHeikinAshiBars } from '../utils/heikinAshi';

interface SparklineProps {
  ticker: string;
  width?: number;
  height?: number;
  tail?: number;
  variant?: 'heikinAshi' | 'reversal' | 'extremes' | 'pullback';
  fluid?: boolean;
}

interface CanvasReversalPoint {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  trend: 1 | -1;
  color: string;
  areaColor: string;
  wickColor: string;
  isFlip: boolean;
}

interface CanvasReversalModel {
  points: CanvasReversalPoint[];
}

interface ReversalState {
  trend: 1 | -1;
  age: number | null;
}

type DecisionState = 'green' | 'red' | 'neutral';

const MAX_CONCURRENT_SPARKLINE_REQUESTS = 14;
let activeSparklineRequests = 0;
const pendingSparklineRequests: Array<() => void> = [];

function runSparklineRequest<T>(request: () => Promise<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    const run = () => {
      activeSparklineRequests += 1;
      request()
        .then(resolve, reject)
        .finally(() => {
          activeSparklineRequests = Math.max(0, activeSparklineRequests - 1);
          pendingSparklineRequests.pop()?.();
        });
    };

    if (activeSparklineRequests < MAX_CONCURRENT_SPARKLINE_REQUESTS) {
      run();
    } else {
      pendingSparklineRequests.push(run);
    }
  });
}

function useNearViewport<T extends HTMLElement>(rootMargin = '1800px 0px') {
  const ref = useRef<T | null>(null);
  const [isNearViewport, setIsNearViewport] = useState(false);

  useEffect(() => {
    if (isNearViewport) return;
    const el = ref.current;
    if (!el) return;
    if (!('IntersectionObserver' in window)) {
      setIsNearViewport(true);
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIsNearViewport(true);
          observer.disconnect();
        }
      },
      { rootMargin }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [isNearViewport, rootMargin]);

  return { ref, isNearViewport };
}

function SparklineSkeleton({ width, height, fluid, variant }: {
  width: number;
  height: number;
  fluid: boolean;
  variant: SparklineProps['variant'];
}) {
  const bullish = variant === 'reversal';
  const accent = bullish ? '#34d399' : '#8b5cf6';
  const areaFill = bullish ? 'rgba(52,211,153,0.14)' : 'rgba(139,92,246,0.13)';
  return (
    <div
      style={{
        width: fluid ? '100%' : width,
        height,
        background: 'linear-gradient(180deg, rgba(255,255,255,0.026), rgba(255,255,255,0.008))',
        border: '1px solid rgba(255,255,255,0.038)',
      }}
      className="relative overflow-hidden rounded-md"
      aria-label="Loading mini chart"
    >
      <svg
        viewBox="0 0 120 36"
        preserveAspectRatio="none"
        className="absolute inset-0 h-full w-full opacity-70"
        aria-hidden
      >
        <path
          d="M2 27 C14 24 18 18 30 20 S47 28 59 20 73 11 86 15 101 24 118 10 L118 36 L2 36 Z"
          fill={areaFill}
        />
        <path
          d="M2 27 C14 24 18 18 30 20 S47 28 59 20 73 11 86 15 101 24 118 10"
          fill="none"
          stroke={accent}
          strokeWidth="1.4"
          strokeLinecap="round"
          opacity="0.58"
        />
      </svg>
      <div
        className="absolute inset-y-0 -left-1/2 w-1/2"
        style={{
          background: 'linear-gradient(90deg, transparent, rgba(255,255,255,0.055), transparent)',
          animation: 'sparklineSkeletonSweep 1.25s ease-in-out infinite',
        }}
      />
    </div>
  );
}

function validOhlcvBars(bars: OHLCVBar[] | undefined): OHLCVBar[] {
  if (!bars?.length) return [];
  return bars.filter((bar) =>
    Number.isFinite(bar.open)
    && Number.isFinite(bar.high)
    && Number.isFinite(bar.low)
    && Number.isFinite(bar.close)
    && Number.isFinite(bar.volume),
  );
}

function computeAtr(bars: OHLCVBar[], period = 14): number[] {
  const tr = bars.map((bar, i) => {
    if (i === 0) return Math.max(0, bar.high - bar.low);
    const prevClose = bars[i - 1].close;
    return Math.max(
      Math.max(0, bar.high - bar.low),
      Math.abs(bar.high - prevClose),
      Math.abs(bar.low - prevClose),
    );
  });

  return tr.map((_, i) => {
    const start = Math.max(0, i - period + 1);
    const slice = tr.slice(start, i + 1);
    const avg = slice.reduce((sum, value) => sum + value, 0) / Math.max(1, slice.length);
    return avg || Math.max(1e-9, bars[i].high - bars[i].low);
  });
}

function buildCanvasReversalModel(bars: OHLCVBar[]): CanvasReversalModel {
  if (!bars.length) return { points: [] };

  const atr = computeAtr(bars);
  const multiplier = 2.4;
  let trend: 1 | -1 = bars[0].close >= bars[0].open ? 1 : -1;
  let finalUpper = (bars[0].high + bars[0].low) / 2 + multiplier * atr[0];
  let finalLower = (bars[0].high + bars[0].low) / 2 - multiplier * atr[0];
  const points: CanvasReversalPoint[] = [];

  bars.forEach((bar, i) => {
    const prevBar = bars[Math.max(0, i - 1)];
    const midpoint = (bar.high + bar.low) / 2;
    const basicUpper = midpoint + multiplier * atr[i];
    const basicLower = midpoint - multiplier * atr[i];
    const prevTrend = trend;

    if (i > 0) {
      finalUpper = basicUpper < finalUpper || prevBar.close > finalUpper ? basicUpper : finalUpper;
      finalLower = basicLower > finalLower || prevBar.close < finalLower ? basicLower : finalLower;

      if (prevTrend === 1 && bar.close < finalLower) trend = -1;
      else if (prevTrend === -1 && bar.close > finalUpper) trend = 1;
    }

    const lookback = bars[Math.max(0, i - 3)].close || bar.close;
    const shortMomentum = bar.close - lookback;
    const dayMomentum = i > 0 ? bar.close - prevBar.close : bar.close - bar.open;
    const isFlip = i > 0 && trend !== prevTrend;
    const isAligned = trend === 1
      ? dayMomentum >= 0 && shortMomentum >= 0
      : dayMomentum <= 0 && shortMomentum <= 0;
    const point: CanvasReversalPoint = {
      time: bar.time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      trend,
      isFlip,
      color: trend === 1
        ? isFlip
          ? '#00f5a0'
          : isAligned
            ? '#34d399'
            : '#86efac'
        : isFlip
          ? '#ff375f'
          : isAligned
            ? '#fb7185'
            : '#fca5a5',
      areaColor: trend === 1
        ? isFlip
          ? 'rgba(0,245,160,0.28)'
          : isAligned
            ? 'rgba(16,185,129,0.18)'
            : 'rgba(134,239,172,0.10)'
        : isFlip
          ? 'rgba(255,55,95,0.27)'
          : isAligned
            ? 'rgba(244,63,94,0.17)'
            : 'rgba(252,165,165,0.09)',
      wickColor: trend === 1 ? 'rgba(110,231,183,0.82)' : 'rgba(253,164,175,0.82)',
    };
    points.push(point);
  });

  return { points };
}

function getReversalState(bars: OHLCVBar[]): ReversalState | null {
  const points = buildCanvasReversalModel(bars).points;
  if (points.length < 3) return null;
  const last = points[points.length - 1];
  const lastFlipIndex = points.reduce((latest, point, index) => (point.isFlip ? index : latest), -1);
  return {
    trend: last.trend,
    age: lastFlipIndex >= 0 ? points.length - 1 - lastFlipIndex : null,
  };
}

function computeEma(values: number[], period: number): number[] {
  if (!values.length) return [];
  const alpha = 2 / (period + 1);
  const out: number[] = [];
  let prev = values[0];
  values.forEach((value, index) => {
    prev = index === 0 ? value : value * alpha + prev * (1 - alpha);
    out.push(prev);
  });
  return out;
}

function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0 ? (sorted[mid - 1] + sorted[mid]) / 2 : sorted[mid];
}

function computeAdaptiveExhaustionZ(closes: number[], lookback = 63): number[] {
  if (!closes.length) return [];
  const logCloses = closes.map((value) => Math.log(Math.max(value, 1e-9)));
  const fastLevel = computeEma(logCloses, 21);
  const slowLevel = computeEma(logCloses, 55);
  const residuals = logCloses.map((value, index) => value - (fastLevel[index] * 0.62 + slowLevel[index] * 0.38));
  const returns = logCloses.map((value, index) => (index === 0 ? 0 : value - logCloses[index - 1]));
  const fallbackVol = computeEma(returns.map((value) => Math.abs(value)), 21).map((value) => Math.max(value * Math.sqrt(21) * 1.25, 1e-5));

  return residuals.map((residual, index) => {
    const start = Math.max(0, index - lookback + 1);
    const window = residuals.slice(start, index + 1);
    const center = median(window);
    const mad = median(window.map((value) => Math.abs(value - center)));
    const robustScale = Math.max(mad * 1.4826, fallbackVol[index], 1e-5);
    return (residual - center) / robustScale;
  });
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function computeAdaptiveExhaustionScore(closes: number[], lookback = 126): number[] {
  const z = computeAdaptiveExhaustionZ(closes);
  return z.map((value, index) => {
    const start = Math.max(0, index - lookback + 1);
    const window = z.slice(start, index + 1).filter(Number.isFinite);
    if (window.length < 8) return clamp(Math.tanh(value / 1.15), -1, 1);
    const belowOrEqual = window.filter((item) => item <= value).length;
    const percentile = belowOrEqual / window.length;
    const percentileScore = percentile * 2 - 1;
    const zScore = Math.tanh(value / 1.15);
    return clamp(zScore * 0.62 + percentileScore * 0.38, -1, 1);
  });
}

function drawMiniRegimeZones(
  ctx: CanvasRenderingContext2D,
  states: DecisionState[],
  xFor: (index: number) => number,
  top: number,
  height: number,
  width: number,
) {
  if (states.length < 2) return;
  let start = 0;
  for (let i = 1; i <= states.length; i += 1) {
    if (i < states.length && states[i] === states[start]) continue;
    const state = states[start];
    if (state !== 'neutral') {
      const x0 = Math.max(0, start === 0 ? 0 : xFor(start));
      const x1 = Math.min(width, i >= states.length ? width : xFor(i));
      const green = state === 'green';
      const zone = ctx.createLinearGradient(0, top, 0, top + height);
      zone.addColorStop(0, green ? 'rgba(16,185,129,0.115)' : 'rgba(244,63,94,0.115)');
      zone.addColorStop(1, green ? 'rgba(16,185,129,0.010)' : 'rgba(244,63,94,0.010)');
      ctx.fillStyle = zone;
      ctx.fillRect(x0, top, Math.max(1, x1 - x0), height);
    }
    start = i;
  }
}

function drawBottomDecisionRail(
  ctx: CanvasRenderingContext2D,
  states: DecisionState[],
  xFor: (index: number) => number,
  width: number,
  y: number,
) {
  ctx.fillStyle = 'rgba(148,163,184,0.13)';
  ctx.fillRect(0, y, width, 2.4);
  states.forEach((state, index) => {
    if (state === 'neutral') return;
    const x0 = Math.max(0, index === 0 ? 0 : xFor(index - 0.5));
    const x1 = Math.min(width, index === states.length - 1 ? width : xFor(index + 0.5));
    ctx.fillStyle = state === 'green' ? 'rgba(16,185,129,0.72)' : 'rgba(244,63,94,0.72)';
    ctx.fillRect(x0, y, Math.max(1, x1 - x0), 2.4);
  });
}

function drawStretchHeat(
  ctx: CanvasRenderingContext2D,
  scores: number[],
  xFor: (index: number) => number,
  top: number,
  height: number,
  width: number,
) {
  scores.forEach((score, index) => {
    const abs = Math.abs(score);
    const x0 = Math.max(0, index === 0 ? 0 : xFor(index - 0.5));
    const x1 = Math.min(width, index === scores.length - 1 ? width : xFor(index + 0.5));
    const alpha = clamp(0.018 + abs * 0.145, 0.018, 0.18);
    const green = score < 0;
    const heat = ctx.createLinearGradient(0, top, 0, top + height);
    heat.addColorStop(0, green ? `rgba(16,185,129,${alpha})` : `rgba(244,63,94,${alpha})`);
    heat.addColorStop(1, green ? `rgba(16,185,129,${alpha * 0.08})` : `rgba(244,63,94,${alpha * 0.08})`);
    ctx.fillStyle = heat;
    ctx.fillRect(x0, top, Math.max(1, x1 - x0), height);
  });
}

function drawRoundedPill(ctx: CanvasRenderingContext2D, x: number, y: number, width: number, height: number, radius: number) {
  ctx.beginPath();
  ctx.roundRect(x, y, width, height, radius);
  ctx.fill();
  ctx.stroke();
}

function drawReversalSparkline(ctx: CanvasRenderingContext2D, bars: OHLCVBar[], width: number, height: number) {
  const model = buildCanvasReversalModel(bars);
  const points = model.points;
  if (points.length < 3) return;

  const minP = Math.min(...points.map((point) => point.low));
  const maxP = Math.max(...points.map((point) => point.high));
  const range = maxP - minP || 1;
  const padX = 4;
  const padTop = 4;
  const padBottom = 7;
  const chartW = width - padX * 2;
  const chartH = height - padTop - padBottom;
  const slot = chartW / Math.max(1, points.length - 1);
  const bodyW = Math.max(1.25, Math.min(3.0, slot * 0.58));

  const xFor = (index: number) => padX + (index / (points.length - 1)) * chartW;
  const yFor = (value: number) => padTop + chartH - ((value - minP) / range) * chartH;

  const segments: Array<{ start: number; end: number; trend: 1 | -1 }> = [];
  let currentStart = 0;
  for (let i = 1; i < points.length; i += 1) {
    if (points[i].trend !== points[i - 1].trend) {
      segments.push({ start: currentStart, end: i - 1, trend: points[i - 1].trend });
      currentStart = i;
    }
  }
  segments.push({ start: currentStart, end: points.length - 1, trend: points[points.length - 1].trend });

  ctx.fillStyle = 'rgba(255,255,255,0.012)';
  ctx.fillRect(0, 0, width, height);

  segments.forEach((segment) => {
    const x0 = Math.max(0, xFor(segment.start) - slot * 0.5);
    const x1 = Math.min(width, xFor(segment.end) + slot * 0.5);
    const zone = ctx.createLinearGradient(0, padTop, 0, padTop + chartH);
    zone.addColorStop(0, segment.trend === 1 ? 'rgba(0,245,160,0.105)' : 'rgba(255,55,95,0.105)');
    zone.addColorStop(1, segment.trend === 1 ? 'rgba(0,245,160,0.008)' : 'rgba(255,55,95,0.008)');
    ctx.fillStyle = zone;
    ctx.fillRect(x0, padTop, Math.max(1, x1 - x0), chartH);

    ctx.fillStyle = segment.trend === 1 ? 'rgba(0,245,160,0.62)' : 'rgba(255,55,95,0.62)';
    ctx.fillRect(x0, height - 3.5, Math.max(1, x1 - x0), 2.5);
  });

  const lastCloseY = yFor(points[points.length - 1].close);
  ctx.beginPath();
  ctx.setLineDash([3, 4]);
  ctx.moveTo(padX, lastCloseY);
  ctx.lineTo(width - padX, lastCloseY);
  ctx.strokeStyle = points[points.length - 1].trend === 1 ? 'rgba(0,245,160,0.18)' : 'rgba(255,55,95,0.18)';
  ctx.lineWidth = 0.8;
  ctx.stroke();
  ctx.setLineDash([]);

  const drawAreaSegment = (start: number, end: number) => {
    if (end <= start) return;
    const segmentTrend = points[end].trend;
    const fill = ctx.createLinearGradient(0, padTop, 0, padTop + chartH);
    fill.addColorStop(0, segmentTrend === 1 ? 'rgba(16,185,129,0.23)' : 'rgba(244,63,94,0.22)');
    fill.addColorStop(1, segmentTrend === 1 ? 'rgba(16,185,129,0)' : 'rgba(244,63,94,0)');

    ctx.beginPath();
    for (let i = start; i <= end; i += 1) {
      const x = xFor(i);
      const y = yFor(points[i].close);
      if (i === start) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.lineTo(xFor(end), padTop + chartH);
    ctx.lineTo(xFor(start), padTop + chartH);
    ctx.closePath();
    ctx.fillStyle = fill;
    ctx.fill();

    ctx.beginPath();
    for (let i = start; i <= end; i += 1) {
      const x = xFor(i);
      const y = yFor(points[i].close);
      if (i === start) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.strokeStyle = segmentTrend === 1 ? '#34d399' : '#fb7185';
    ctx.lineWidth = 1.15;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalAlpha = 0.92;
    ctx.stroke();
    ctx.globalAlpha = 1;
  };

  segments.forEach((segment) => drawAreaSegment(segment.start, segment.end));

  points.forEach((point, i) => {
    const x = xFor(i);
    const highY = yFor(point.high);
    const lowY = yFor(point.low);
    const openY = yFor(point.open);
    const closeY = yFor(point.close);
    const topY = Math.min(openY, closeY);
    const bottomY = Math.max(openY, closeY);
    const bodyH = Math.max(1.15, bottomY - topY);

    ctx.beginPath();
    ctx.moveTo(x, highY);
    ctx.lineTo(x, lowY);
    ctx.strokeStyle = point.wickColor;
    ctx.lineWidth = point.isFlip ? 1.15 : 0.78;
    ctx.lineCap = 'round';
    ctx.globalAlpha = point.isFlip ? 1 : 0.74;
    ctx.stroke();

    ctx.beginPath();
    ctx.roundRect(x - bodyW / 2, topY, bodyW, bodyH, 1.05);
    ctx.fillStyle = point.color;
    ctx.globalAlpha = point.isFlip ? 1 : 0.72;
    ctx.fill();
    ctx.globalAlpha = 1;

    if (point.isFlip) {
      ctx.beginPath();
      ctx.arc(x, closeY, 3.1, 0, Math.PI * 2);
      ctx.strokeStyle = point.trend === 1 ? 'rgba(0,245,160,0.64)' : 'rgba(255,55,95,0.64)';
      ctx.lineWidth = 1;
      ctx.stroke();
    }
  });

  const markerCandidates = points
    .map((point, index) => ({ point, index }))
    .filter(({ point, index }) => point.isFlip && index >= Math.max(0, points.length - 92))
    .reverse();
  const selectedMarkers: Array<{ point: CanvasReversalPoint; index: number }> = [];
  for (const item of markerCandidates) {
    const x = xFor(item.index);
    if (selectedMarkers.every((selected) => Math.abs(xFor(selected.index) - x) >= 38)) {
      selectedMarkers.push(item);
    }
    if (selectedMarkers.length >= 5) break;
  }

  selectedMarkers.reverse().forEach(({ point, index }) => {
    const x = xFor(index);
    const isBuy = point.trend === 1;
    const label = width >= 280 ? (isBuy ? 'BUY' : 'SELL') : (isBuy ? 'B' : 'S');
    const labelW = label.length > 1 ? 25 : 13;
    const labelH = 11;
    const labelX = Math.max(1, Math.min(width - labelW - 1, x - labelW / 2));
    const labelY = isBuy
      ? Math.min(height - labelH - 1, yFor(point.low) + 2)
      : Math.max(1, yFor(point.high) - labelH - 2);

    ctx.save();
    ctx.shadowColor = isBuy ? 'rgba(0,245,160,0.45)' : 'rgba(255,55,95,0.45)';
    ctx.shadowBlur = 8;
    ctx.fillStyle = isBuy ? 'rgba(0,245,160,0.18)' : 'rgba(255,55,95,0.18)';
    ctx.strokeStyle = isBuy ? 'rgba(0,245,160,0.72)' : 'rgba(255,55,95,0.72)';
    drawRoundedPill(ctx, labelX, labelY, labelW, labelH, 4);
    ctx.restore();

    ctx.fillStyle = isBuy ? '#a7f3d0' : '#fecdd3';
    ctx.font = '700 7px Inter, system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, labelX + labelW / 2, labelY + labelH / 2 + 0.2);
  });

  const last = points[points.length - 1];
  const lastX = xFor(points.length - 1);
  const lastY = yFor(last.close);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4.0, 0, Math.PI * 2);
  ctx.fillStyle = last.trend === 1 ? 'rgba(0,245,160,0.17)' : 'rgba(255,55,95,0.17)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(lastX, lastY, 2.0, 0, Math.PI * 2);
  ctx.fillStyle = last.trend === 1 ? '#00f5a0' : '#ff375f';
  ctx.fill();
}

function drawExtremesSparkline(ctx: CanvasRenderingContext2D, bars: OHLCVBar[], width: number, height: number) {
  if (bars.length < 8) return;
  const closes = bars.map((bar) => bar.close);
  const highs = bars.map((bar) => bar.high);
  const lows = bars.map((bar) => bar.low);
  const exhaustionZ = computeAdaptiveExhaustionZ(closes);
  const exhaustionScore = computeAdaptiveExhaustionScore(closes);
  const states = exhaustionScore.map((value) => (value <= -0.24 ? 'green' : value >= 0.24 ? 'red' : 'neutral')) as DecisionState[];
  const minP = Math.min(...lows);
  const maxP = Math.max(...highs);
  const range = maxP - minP || 1;
  const padX = 4;
  const padTop = 4;
  const padBottom = 12;
  const chartW = width - padX * 2;
  const chartH = height - padTop - padBottom;
  const xFor = (index: number) => padX + (index / Math.max(1, closes.length - 1)) * chartW;
  const yFor = (value: number) => padTop + chartH - ((value - minP) / range) * chartH;

  ctx.fillStyle = 'rgba(255,255,255,0.010)';
  ctx.fillRect(0, 0, width, height);
  drawStretchHeat(ctx, exhaustionScore, xFor, padTop, chartH, width);
  drawMiniRegimeZones(ctx, states, xFor, padTop, chartH, width);

  const area = ctx.createLinearGradient(0, padTop, 0, padTop + chartH);
  area.addColorStop(0, 'rgba(148,163,184,0.12)');
  area.addColorStop(1, 'rgba(148,163,184,0)');
  ctx.beginPath();
  closes.forEach((close, index) => {
    const x = xFor(index);
    const y = yFor(close);
    if (index === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.lineTo(xFor(closes.length - 1), padTop + chartH);
  ctx.lineTo(xFor(0), padTop + chartH);
  ctx.closePath();
  ctx.fillStyle = area;
  ctx.fill();

  for (let i = 1; i < closes.length; i += 1) {
    const state = states[i];
    const score = exhaustionScore[i] ?? 0;
    const neutralGreen = score < 0 ? clamp(Math.abs(score) * 0.72, 0.18, 0.44) : 0;
    const neutralRed = score > 0 ? clamp(Math.abs(score) * 0.72, 0.18, 0.44) : 0;
    const color = state === 'green'
      ? '#34d399'
      : state === 'red'
        ? '#fb7185'
        : score < 0
          ? `rgba(52,211,153,${neutralGreen})`
          : score > 0
            ? `rgba(251,113,133,${neutralRed})`
            : 'rgba(203,213,225,0.50)';
    ctx.beginPath();
    ctx.moveTo(xFor(i - 1), yFor(closes[i - 1]));
    ctx.lineTo(xFor(i), yFor(closes[i]));
    ctx.strokeStyle = color;
    ctx.lineWidth = state === 'neutral' ? 1.05 : 1.55;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  drawBottomDecisionRail(ctx, states, xFor, width, height - 4.2);

  const latestState = states[states.length - 1];
  const latestScore = exhaustionScore[exhaustionScore.length - 1] ?? 0;
  const latestColor = latestState === 'green' ? '#34d399' : latestState === 'red' ? '#fb7185' : latestScore < 0 ? '#86efac' : latestScore > 0 ? '#fda4af' : '#cbd5e1';
  const lastX = xFor(closes.length - 1);
  const lastY = yFor(closes[closes.length - 1]);
  ctx.beginPath();
  ctx.arc(lastX, lastY, 4.0, 0, Math.PI * 2);
  ctx.fillStyle = latestState === 'green'
    ? 'rgba(16,185,129,0.16)'
    : latestState === 'red'
      ? 'rgba(244,63,94,0.16)'
      : latestScore < 0
        ? 'rgba(16,185,129,0.09)'
        : latestScore > 0
          ? 'rgba(244,63,94,0.09)'
          : 'rgba(203,213,225,0.10)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(lastX, lastY, 1.9, 0, Math.PI * 2);
  ctx.fillStyle = latestColor;
  ctx.fill();

  const latestZ = exhaustionZ[exhaustionZ.length - 1] ?? 0;
  ctx.font = '700 7px Inter, system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillStyle = latestColor;
  ctx.globalAlpha = 0.82;
  ctx.fillText(`Z ${latestZ >= 0 ? '+' : ''}${latestZ.toFixed(1)}`, width - 5, height - 9);
  ctx.globalAlpha = 1;
}

function drawPullbackSparkline(ctx: CanvasRenderingContext2D, bars: OHLCVBar[], width: number, height: number) {
  if (bars.length < 8) return;
  const closes = bars.map((bar) => bar.close);
  const highs = bars.map((bar) => bar.high);
  const lows = bars.map((bar) => bar.low);
  const ema20 = computeEma(closes, 20);
  const ema50 = computeEma(closes, 50);
  const exhaustionZ = computeAdaptiveExhaustionZ(closes);
  const states = closes.map((close, index) => {
    const fast = ema20[index];
    const slow = ema50[index];
    const uptrend = fast >= slow;
    const downtrend = fast < slow;
    const dist = fast > 0 ? (close - fast) / fast : 0;
    const nearFast = Math.abs(dist) <= 0.035;
    if (uptrend && close >= slow && (dist <= 0.012 || (nearFast && exhaustionZ[index] <= -0.25))) return 'green';
    if (downtrend && close <= slow && (dist >= -0.012 || (nearFast && exhaustionZ[index] >= 0.25))) return 'red';
    return 'neutral';
  }) as Array<'green' | 'red' | 'neutral'>;

  const minP = Math.min(...lows, ...ema20, ...ema50);
  const maxP = Math.max(...highs, ...ema20, ...ema50);
  const range = maxP - minP || 1;
  const padX = 4;
  const padTop = 4;
  const padBottom = 12;
  const chartW = width - padX * 2;
  const chartH = height - padTop - padBottom;
  const xFor = (index: number) => padX + (index / Math.max(1, closes.length - 1)) * chartW;
  const yFor = (value: number) => padTop + chartH - ((value - minP) / range) * chartH;

  ctx.fillStyle = 'rgba(255,255,255,0.010)';
  ctx.fillRect(0, 0, width, height);
  drawMiniRegimeZones(ctx, states, xFor, padTop, chartH, width);

  const drawLine = (values: number[], color: string, lineWidth: number, alpha = 1) => {
    ctx.beginPath();
    values.forEach((value, index) => {
      const x = xFor(index);
      const y = yFor(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.globalAlpha = alpha;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
    ctx.globalAlpha = 1;
  };

  drawLine(ema50, 'rgba(148,163,184,0.42)', 0.8);
  drawLine(ema20, 'rgba(125,211,252,0.55)', 0.85);

  for (let i = 1; i < closes.length; i += 1) {
    const state = states[i];
    const trendUp = ema20[i] >= ema50[i];
    const color = state === 'green'
      ? '#34d399'
      : state === 'red'
        ? '#fb7185'
        : trendUp
          ? 'rgba(110,231,183,0.64)'
          : 'rgba(253,164,175,0.62)';
    ctx.beginPath();
    ctx.moveTo(xFor(i - 1), yFor(closes[i - 1]));
    ctx.lineTo(xFor(i), yFor(closes[i]));
    ctx.strokeStyle = color;
    ctx.lineWidth = state === 'neutral' ? 1.0 : 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();
  }

  drawBottomDecisionRail(ctx, states, xFor, width, height - 4.2);

  const last = closes.length - 1;
  const latestState = states[last];
  const trendUp = ema20[last] >= ema50[last];
  const latestColor = latestState === 'green' ? '#34d399' : latestState === 'red' ? '#fb7185' : trendUp ? '#7dd3fc' : '#fda4af';
  ctx.beginPath();
  ctx.arc(xFor(last), yFor(closes[last]), 4.0, 0, Math.PI * 2);
  ctx.fillStyle = latestState === 'green' ? 'rgba(16,185,129,0.16)' : latestState === 'red' ? 'rgba(244,63,94,0.16)' : 'rgba(125,211,252,0.10)';
  ctx.fill();
  ctx.beginPath();
  ctx.arc(xFor(last), yFor(closes[last]), 1.9, 0, Math.PI * 2);
  ctx.fillStyle = latestColor;
  ctx.fill();
}

/**
 * Story 3.1 AC-1: compact row Heikin Ashi chart showing recent trend structure.
 * The percent chip remains real close-to-close performance; the row chart is
 * intentionally visual/noise-reduced.
 */
function SparklineInner({ ticker, width = 60, height = 28, tail = 30, variant = 'heikinAshi', fluid = false }: SparklineProps) {
  const { ref: visibilityRef, isNearViewport } = useNearViewport<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [measuredWidth, setMeasuredWidth] = useState(width);
  const renderWidth = fluid ? Math.max(1, Math.round(measuredWidth || width)) : width;

  const { data } = useQuery({
    queryKey: ['sparkline', ticker, tail],
    queryFn: () => runSparklineRequest(() => api.chartOhlcv(ticker, tail)),
    enabled: isNearViewport,
    staleTime: 600_000,
    retry: 1,
  });

  useEffect(() => {
    if (!fluid) {
      setMeasuredWidth(width);
      return;
    }

    const el = visibilityRef.current;
    if (!el) return;

    const updateWidth = () => {
      const next = el.clientWidth || width;
      setMeasuredWidth((prev) => (Math.abs(prev - next) > 1 ? next : prev));
    };
    updateWidth();

    if (typeof ResizeObserver === 'undefined') {
      window.addEventListener('resize', updateWidth);
      return () => window.removeEventListener('resize', updateWidth);
    }

    const observer = new ResizeObserver(updateWidth);
    observer.observe(el);
    return () => observer.disconnect();
  }, [fluid, visibilityRef, width]);

  useEffect(() => {
    const bars = validOhlcvBars(data?.data);
    if (!bars || bars.length < 3 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = renderWidth * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, renderWidth, height);

    if (variant === 'reversal') {
      drawReversalSparkline(ctx, bars, renderWidth, height);
      return;
    }
    if (variant === 'extremes') {
      drawExtremesSparkline(ctx, bars, renderWidth, height);
      return;
    }
    if (variant === 'pullback') {
      drawPullbackSparkline(ctx, bars, renderWidth, height);
      return;
    }

    const haBars = toHeikinAshiBars(bars as OHLCVBar[]);
    const closes = haBars.map((b) => b.close);
    const minP = Math.min(...haBars.map((b) => b.low));
    const maxP = Math.max(...haBars.map((b) => b.high));
    const range = maxP - minP || 1;
    const pad = 2;
    const padTop = 3;
    const w = renderWidth - pad * 2;
    const h = height - pad - padTop;

    const smaWindow = Math.min(20, closes.length);
    const smaSlice = closes.slice(-smaWindow);
    const sma20 = smaSlice.reduce((a: number, b: number) => a + b, 0) / smaSlice.length;
    const lastClose = closes[closes.length - 1];
    const aboveSma = lastClose >= sma20;

    const trendCol = aboveSma ? '#3ee8a5' : '#ff6b8a';
    const fillStart = aboveSma ? 'rgba(62,232,165,0.16)' : 'rgba(255,107,138,0.16)';
    const fillEnd = aboveSma ? 'rgba(62,232,165,0.00)' : 'rgba(255,107,138,0.00)';

    const yFor = (value: number) => padTop + h - ((value - minP) / range) * h;
    const pts: [number, number][] = haBars.map((bar, i) => {
      const x = pad + (i / (closes.length - 1)) * w;
      const y = yFor(bar.close);
      return [x, y];
    });

    // Soft HA-close cloud so the tiny row chart keeps the old sparkline depth.
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.lineTo(pts[pts.length - 1][0], padTop + h);
    ctx.lineTo(pts[0][0], padTop + h);
    ctx.closePath();
    const grad = ctx.createLinearGradient(0, padTop, 0, padTop + h);
    grad.addColorStop(0, fillStart);
    grad.addColorStop(1, fillEnd);
    ctx.fillStyle = grad;
    ctx.fill();

    // Heikin Ashi mini-candles: wick + body, compressed for table rows.
    const slot = w / Math.max(1, haBars.length - 1);
    const bodyW = Math.max(1.35, Math.min(3.2, slot * 0.54));
    haBars.forEach((bar, i) => {
      const x = pad + (i / (haBars.length - 1)) * w;
      const up = isHeikinAshiUp(bar);
      const candleColor = up ? '#3ee8a5' : '#ff6b8a';
      const wickColor = up ? 'rgba(111,240,192,0.78)' : 'rgba(253,164,175,0.78)';
      const highY = yFor(bar.high);
      const lowY = yFor(bar.low);
      const openY = yFor(bar.open);
      const closeY = yFor(bar.close);
      const topY = Math.min(openY, closeY);
      const bottomY = Math.max(openY, closeY);
      const bodyH = Math.max(1.2, bottomY - topY);

      ctx.beginPath();
      ctx.moveTo(x, highY);
      ctx.lineTo(x, lowY);
      ctx.strokeStyle = wickColor;
      ctx.lineWidth = 0.85;
      ctx.lineCap = 'round';
      ctx.stroke();

      ctx.beginPath();
      ctx.roundRect(x - bodyW / 2, topY, bodyW, bodyH, 1.15);
      ctx.fillStyle = candleColor;
      ctx.globalAlpha = i === haBars.length - 1 ? 1 : 0.72;
      ctx.fill();
      ctx.globalAlpha = 1;
    });

    // Thin HA-close trace adds continuity without overpowering candle bodies.
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.strokeStyle = trendCol;
    ctx.lineWidth = 0.8;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalAlpha = 0.52;
    ctx.stroke();
    ctx.globalAlpha = 1;

    // End dot with glow ring
    const [lastX, lastY] = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = aboveSma ? 'rgba(62,232,165,0.22)' : 'rgba(255,107,138,0.22)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 1.8, 0, Math.PI * 2);
    ctx.fillStyle = trendCol;
    ctx.fill();
  }, [data, renderWidth, height, variant]);

  const bars = validOhlcvBars(data?.data);
  if (!bars || bars.length < 3) {
    return (
      <div
        ref={visibilityRef}
        style={{ width: fluid ? '100%' : width, height }}
        title={isNearViewport ? 'Preparing mini chart' : 'Mini chart queued'}
      >
        <SparklineSkeleton width={renderWidth} height={height} fluid={fluid} variant={variant} />
      </div>
    );
  }

  const closes = bars.map((b: { close: number }) => b.close);
  const first = closes[0];
  const last = closes[closes.length - 1];
  const pctChg = first ? ((last - first) / first) * 100 : 0;
  const up = pctChg >= 0;
  const reversalState = variant === 'reversal' ? getReversalState(bars) : null;
  const visualUp = variant === 'reversal' ? reversalState?.trend !== -1 : variant === 'extremes' || variant === 'pullback' ? true : up;
  const titleMap: Record<NonNullable<SparklineProps['variant']>, string> = {
    heikinAshi: `${tail}-bar Heikin Ashi row chart`,
    reversal: `${tail}-bar mini reversal chart`,
    extremes: `${tail}-bar overbought / oversold chart`,
    pullback: `${tail}-bar pullback trend chart`,
  };

  return (
    <div
      ref={visibilityRef}
      style={{ width: fluid ? '100%' : width, height }}
      title={titleMap[variant]}
    >
      <canvas
        ref={canvasRef}
        style={{
          width: fluid ? '100%' : renderWidth,
          height,
          display: 'block',
          filter: visualUp
            ? 'drop-shadow(0 0 2px rgba(62,232,165,0.35))'
            : 'drop-shadow(0 0 2px rgba(255,107,138,0.35))',
        }}
      />
    </div>
  );
}

export const Sparkline = memo(SparklineInner);

function SparklineReversalStateBadgeInner({ ticker, tail = 220, compact = false, tile = false }: { ticker: string; tail?: number; compact?: boolean; tile?: boolean }) {
  const { ref: visibilityRef, isNearViewport } = useNearViewport<HTMLDivElement>();
  const { data } = useQuery({
    queryKey: ['sparkline', ticker, tail],
    queryFn: () => runSparklineRequest(() => api.chartOhlcv(ticker, tail)),
    enabled: isNearViewport,
    staleTime: 600_000,
    retry: 1,
  });

  const state = getReversalState(validOhlcvBars(data?.data));
  if (!state) {
    return (
      <div ref={visibilityRef} className={`flex flex-col items-center gap-0.5 ${compact ? 'min-w-0 w-full' : 'min-w-[62px]'}`}>
        <span
          className={`inline-flex ${compact ? tile ? 'h-[38px] w-full rounded-[9px]' : 'h-[30px] w-full rounded-lg' : 'h-[26px] min-w-[58px] rounded-lg'} items-center justify-center text-[8.4px] font-bold uppercase tracking-[0.08em]`}
          style={{
            color: 'var(--text-muted)',
            background: tile ? 'linear-gradient(180deg, rgba(100,116,139,0.075), rgba(255,255,255,0.012))' : 'rgba(255,255,255,0.018)',
            border: `1px solid ${tile ? 'rgba(100,116,139,0.14)' : 'rgba(255,255,255,0.055)'}`,
            boxShadow: tile ? 'inset 0 1px 0 rgba(255,255,255,0.045)' : undefined,
          }}
        >
          —
        </span>
        {!compact && <span className="text-[8.5px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Reversal</span>}
      </div>
    );
  }

  const isBuy = state.trend === 1;
  const ageLabel = state.age === null ? '' : `${state.age}d`;
  const stateLabel = isBuy ? 'Buy' : 'Sell';
  const color = isBuy ? '#00f5a0' : '#ff375f';
  const softColor = isBuy ? '#a7f3d0' : '#fecdd3';
  const compactBadgeContent = (
    <>
      <span
        className="rounded-full"
        style={{
          width: 4,
          height: 4,
          background: color,
          boxShadow: `0 0 7px ${color}`,
        }}
      />
      <span className={`${tile ? 'text-[8.4px]' : 'text-[9px]'} font-extrabold uppercase tracking-[0.045em] leading-none`}>{stateLabel}</span>
      {ageLabel && <span className={`${tile ? 'text-[7.5px]' : 'text-[8px]'} font-bold leading-none opacity-85`}>{ageLabel}</span>}
    </>
  );

  return (
    <div ref={visibilityRef} className={`flex flex-col items-center gap-0.5 ${compact ? 'min-w-0 w-full' : 'min-w-[68px]'}`}>
      <span
        className={`inline-flex ${compact ? tile ? 'h-[38px] w-full flex-col gap-[1px] rounded-[9px] px-1' : 'h-[30px] w-full flex-col gap-0.5 rounded-lg px-1' : 'h-[26px] min-w-[64px] gap-1.5 rounded-lg px-2'} items-center justify-center tabular-nums`}
        title={`Current mini reversal state: ${stateLabel}${ageLabel ? ` for ${ageLabel}` : ''}`}
        style={{
          color: softColor,
          background: tile
            ? `linear-gradient(180deg, ${isBuy ? 'rgba(16,185,129,0.18)' : 'rgba(244,63,94,0.18)'}, rgba(255,255,255,0.012))`
            : isBuy ? 'rgba(0,245,160,0.115)' : 'rgba(255,55,95,0.115)',
          border: `1px solid ${isBuy ? tile ? 'rgba(16,185,129,0.24)' : 'rgba(0,245,160,0.42)' : tile ? 'rgba(244,63,94,0.26)' : 'rgba(255,55,95,0.42)'}`,
          boxShadow: tile ? `inset 0 1px 0 rgba(255,255,255,0.045), 0 10px 18px -18px ${color}` : `0 0 14px -9px ${color}`,
        }}
      >
        {compact ? compactBadgeContent : (
          <>
            <span
              className="rounded-full"
              style={{
                width: 5,
                height: 5,
                background: color,
                boxShadow: `0 0 7px ${color}`,
              }}
            />
            <span className="text-[9.5px] font-extrabold uppercase tracking-[0.08em]">{stateLabel}</span>
            {ageLabel && <span className="text-[9px] font-bold opacity-90">{ageLabel}</span>}
          </>
        )}
      </span>
      {!compact && <span className="text-[8.5px] uppercase tracking-[0.12em] text-[var(--text-muted)]">Reversal</span>}
    </div>
  );
}

export const SparklineReversalStateBadge = memo(SparklineReversalStateBadgeInner);

type LensBadgeVariant = 'reversal' | 'extremes' | 'pullback';

function lensBadgeState(variant: LensBadgeVariant, bars: OHLCVBar[]) {
  if (variant === 'reversal') {
    const state = getReversalState(bars);
    if (!state) return null;
    const isBuy = state.trend === 1;
    return {
      label: isBuy ? 'Buy' : 'Sell',
      detail: state.age === null ? '' : `${state.age}d`,
      color: isBuy ? '#00f5a0' : '#ff375f',
      softColor: isBuy ? '#a7f3d0' : '#fecdd3',
      bg: isBuy ? 'rgba(16,185,129,0.18)' : 'rgba(244,63,94,0.18)',
      border: isBuy ? 'rgba(16,185,129,0.24)' : 'rgba(244,63,94,0.26)',
      title: `Current reversal regime: ${isBuy ? 'BUY' : 'SELL'}`,
    };
  }

  const closes = bars.map((bar) => bar.close);
  if (closes.length < 8) return null;

  if (variant === 'extremes') {
    const exhaustionZ = computeAdaptiveExhaustionZ(closes);
    const exhaustionScore = computeAdaptiveExhaustionScore(closes);
    const latestZ = exhaustionZ[exhaustionZ.length - 1] ?? 0;
    const latestScore = exhaustionScore[exhaustionScore.length - 1] ?? 0;
    const zDetail = `Z ${latestZ >= 0 ? '+' : ''}${latestZ.toFixed(1)} · ${Math.round(Math.abs(latestScore) * 100)}`;
    if (latestScore <= -0.24) {
      return {
        label: 'Oversold',
        detail: zDetail,
        color: '#34d399',
        softColor: '#a7f3d0',
        bg: 'rgba(16,185,129,0.17)',
        border: 'rgba(16,185,129,0.25)',
        title: 'Oversold: price is statistically stretched below its adaptive equilibrium',
      };
    }
    if (latestScore >= 0.24) {
      return {
        label: 'Overbought',
        detail: zDetail,
        color: '#fb7185',
        softColor: '#fecdd3',
        bg: 'rgba(244,63,94,0.17)',
        border: 'rgba(244,63,94,0.28)',
        title: 'Overbought: price is statistically stretched above its adaptive equilibrium',
      };
    }
    return {
      label: 'Balanced',
      detail: zDetail,
      color: '#94a3b8',
      softColor: '#cbd5e1',
      bg: 'rgba(100,116,139,0.10)',
      border: 'rgba(100,116,139,0.18)',
      title: 'Balanced: price stretch is inside the normal adaptive range',
    };
  }

  const ema20 = computeEma(closes, 20);
  const ema50 = computeEma(closes, 50);
  const exhaustionZ = computeAdaptiveExhaustionZ(closes);
  const last = closes.length - 1;
  const close = closes[last];
  const fast = ema20[last];
  const slow = ema50[last];
  const uptrend = fast >= slow;
  const dist = fast > 0 ? (close - fast) / fast : 0;
  const nearFast = Math.abs(dist) <= 0.035;
  const latestZ = exhaustionZ[last] ?? 0;
  const buyDip = uptrend && close >= slow && (dist <= 0.012 || (nearFast && latestZ <= -0.25));
  const sellBounce = !uptrend && close <= slow && (dist >= -0.012 || (nearFast && latestZ >= 0.25));

  if (buyDip) {
    return {
      label: 'Buy dip',
      detail: 'Trend',
      color: '#34d399',
      softColor: '#a7f3d0',
      bg: 'rgba(16,185,129,0.17)',
      border: 'rgba(16,185,129,0.25)',
      title: 'Constructive pullback: uptrend with price near the fast trend line',
    };
  }
  if (sellBounce) {
    return {
      label: 'Sell bounce',
      detail: 'Trend',
      color: '#fb7185',
      softColor: '#fecdd3',
      bg: 'rgba(244,63,94,0.17)',
      border: 'rgba(244,63,94,0.28)',
      title: 'Risky bounce: downtrend with price near the fast trend line',
    };
  }
  return {
    label: uptrend ? 'Trend up' : 'Trend down',
    detail: uptrend ? 'Above' : 'Below',
    color: uptrend ? '#7dd3fc' : '#fda4af',
    softColor: uptrend ? '#bae6fd' : '#fecdd3',
    bg: uptrend ? 'rgba(14,165,233,0.12)' : 'rgba(244,63,94,0.11)',
    border: uptrend ? 'rgba(14,165,233,0.20)' : 'rgba(244,63,94,0.20)',
    title: uptrend ? 'Uptrend, but not a fresh pullback entry zone' : 'Downtrend, but not a fresh sell-bounce zone',
  };
}

function SparklineLensStateBadgeInner({ ticker, tail = 220, variant = 'reversal', compact = false, tile = false }: {
  ticker: string;
  tail?: number;
  variant?: LensBadgeVariant;
  compact?: boolean;
  tile?: boolean;
}) {
  const { ref: visibilityRef, isNearViewport } = useNearViewport<HTMLDivElement>();
  const { data } = useQuery({
    queryKey: ['sparkline', ticker, tail],
    queryFn: () => runSparklineRequest(() => api.chartOhlcv(ticker, tail)),
    enabled: isNearViewport,
    staleTime: 600_000,
    retry: 1,
  });

  const state = lensBadgeState(variant, validOhlcvBars(data?.data));
  if (!state) {
    return (
      <div ref={visibilityRef} className={`flex flex-col items-center gap-0.5 ${compact ? 'min-w-0 w-full' : 'min-w-[62px]'}`}>
        <span
          className={`inline-flex ${compact ? tile ? 'h-[38px] w-full rounded-[9px]' : 'h-[30px] w-full rounded-lg' : 'h-[26px] min-w-[58px] rounded-lg'} items-center justify-center text-[8.4px] font-bold uppercase tracking-[0.08em]`}
          style={{
            color: 'var(--text-muted)',
            background: tile ? 'linear-gradient(180deg, rgba(100,116,139,0.075), rgba(255,255,255,0.012))' : 'rgba(255,255,255,0.018)',
            border: `1px solid ${tile ? 'rgba(100,116,139,0.14)' : 'rgba(255,255,255,0.055)'}`,
            boxShadow: tile ? 'inset 0 1px 0 rgba(255,255,255,0.045)' : undefined,
          }}
        >
          —
        </span>
      </div>
    );
  }

  return (
    <div ref={visibilityRef} className={`flex flex-col items-center gap-0.5 ${compact ? 'min-w-0 w-full' : 'min-w-[68px]'}`}>
      <span
        className={`inline-flex ${compact ? tile ? 'h-[38px] w-full flex-col gap-[1px] rounded-[9px] px-1' : 'h-[30px] w-full flex-col gap-0.5 rounded-lg px-1' : 'h-[26px] min-w-[64px] gap-1.5 rounded-lg px-2'} items-center justify-center tabular-nums`}
        title={state.title}
        style={{
          color: state.softColor,
          background: tile ? `linear-gradient(180deg, ${state.bg}, rgba(255,255,255,0.012))` : state.bg,
          border: `1px solid ${state.border}`,
          boxShadow: tile ? `inset 0 1px 0 rgba(255,255,255,0.045), 0 10px 18px -18px ${state.color}` : `0 0 14px -9px ${state.color}`,
        }}
      >
        {compact ? (
          <>
            <span
              className="rounded-full"
              style={{ width: 4, height: 4, background: state.color, boxShadow: `0 0 7px ${state.color}` }}
            />
            <span className={`${tile ? 'text-[8.1px]' : 'text-[9px]'} font-extrabold uppercase tracking-[0.035em] leading-none`}>{state.label}</span>
            {state.detail && <span className={`${tile ? 'text-[7.1px]' : 'text-[8px]'} font-bold leading-none opacity-82`}>{state.detail}</span>}
          </>
        ) : (
          <>
            <span
              className="rounded-full"
              style={{ width: 5, height: 5, background: state.color, boxShadow: `0 0 7px ${state.color}` }}
            />
            <span className="text-[9.5px] font-extrabold uppercase tracking-[0.08em]">{state.label}</span>
          </>
        )}
      </span>
      {!compact && <span className="text-[8.5px] uppercase tracking-[0.12em] text-[var(--text-muted)]">{variant}</span>}
    </div>
  );
}

export const SparklineLensStateBadge = memo(SparklineLensStateBadgeInner);

/**
 * 30-day percent change chip, split out from the Sparkline so it can live
 * in its own table column. Shares the same React-Query cache via queryKey.
 */
function SparklinePctInner({ ticker, pct }: { ticker: string; pct?: number | null }) {
  const { ref: visibilityRef, isNearViewport } = useNearViewport<HTMLSpanElement>();
  const { data } = useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => runSparklineRequest(() => api.chartOhlcv(ticker, 30)),
    enabled: pct == null && isNearViewport,
    staleTime: 600_000,
    retry: 1,
  });
  if (typeof pct === 'number' && Number.isFinite(pct)) {
    const up = pct >= 0;
    return (
      <span
        ref={visibilityRef}
        className="inline-block text-[10px] font-mono tabular-nums font-semibold px-1.5 py-0.5 rounded-md"
        style={{
          color: up ? 'var(--accent-emerald)' : 'var(--accent-rose)',
          background: up ? 'rgba(62,232,165,0.10)' : 'rgba(255,107,138,0.10)',
        }}
      >
        {up ? '+' : ''}{pct.toFixed(1)}%
      </span>
    );
  }
  const bars = data?.data;
  if (!bars || bars.length < 3) {
    return <span ref={visibilityRef} className="text-[10px] text-[var(--text-muted)]">—</span>;
  }
  const closes = bars.map((b: { close: number }) => b.close);
  const first = closes[0];
  const last = closes[closes.length - 1];
  const pctChg = first ? ((last - first) / first) * 100 : 0;
  const up = pctChg >= 0;
  return (
    <span
      ref={visibilityRef}
      className="inline-block text-[10px] font-mono tabular-nums font-semibold px-1.5 py-0.5 rounded-md"
      style={{
        color: up ? 'var(--accent-emerald)' : 'var(--accent-rose)',
        background: up ? 'rgba(62,232,165,0.10)' : 'rgba(255,107,138,0.10)',
      }}
    >
      {up ? '+' : ''}{pctChg.toFixed(1)}%
    </span>
  );
}

export const SparklinePct = memo(SparklinePctInner);
