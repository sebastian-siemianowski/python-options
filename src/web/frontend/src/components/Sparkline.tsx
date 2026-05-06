import { useRef, useEffect, useState, memo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type OHLCVBar } from '../api';
import { isHeikinAshiUp, toHeikinAshiBars } from '../utils/heikinAshi';

interface SparklineProps {
  ticker: string;
  width?: number;
  height?: number;
  tail?: number;
  variant?: 'heikinAshi' | 'reversal';
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
  const visualUp = variant === 'reversal' ? reversalState?.trend !== -1 : up;

  return (
    <div
      ref={visibilityRef}
      style={{ width: fluid ? '100%' : width, height }}
      title={variant === 'reversal' ? `${tail}-bar mini reversal chart` : `${tail}-bar Heikin Ashi row chart`}
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
          className={`inline-flex ${compact ? tile ? 'h-[42px] w-full rounded-[9px]' : 'h-[30px] w-full rounded-lg' : 'h-[26px] min-w-[58px] rounded-lg'} items-center justify-center text-[9px] font-bold uppercase tracking-[0.08em]`}
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
      <span className="text-[9px] font-extrabold uppercase tracking-[0.05em] leading-none">{stateLabel}</span>
      {ageLabel && <span className="text-[8px] font-bold leading-none opacity-85">{ageLabel}</span>}
    </>
  );

  return (
    <div ref={visibilityRef} className={`flex flex-col items-center gap-0.5 ${compact ? 'min-w-0 w-full' : 'min-w-[68px]'}`}>
      <span
        className={`inline-flex ${compact ? tile ? 'h-[42px] w-full flex-col gap-0.5 rounded-[9px] px-1.5' : 'h-[30px] w-full flex-col gap-0.5 rounded-lg px-1' : 'h-[26px] min-w-[64px] gap-1.5 rounded-lg px-2'} items-center justify-center tabular-nums`}
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

/**
 * 30-day percent change chip, split out from the Sparkline so it can live
 * in its own table column. Shares the same React-Query cache via queryKey.
 */
function SparklinePctInner({ ticker }: { ticker: string }) {
  const { ref: visibilityRef, isNearViewport } = useNearViewport<HTMLSpanElement>();
  const { data } = useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => runSparklineRequest(() => api.chartOhlcv(ticker, 30)),
    enabled: isNearViewport,
    staleTime: 600_000,
    retry: 1,
  });
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
