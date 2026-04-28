import { useRef, useEffect, useState, memo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type OHLCVBar } from '../api';
import { isHeikinAshiUp, toHeikinAshiBars } from '../utils/heikinAshi';

interface SparklineProps {
  ticker: string;
  width?: number;
  height?: number;
}

const MAX_CONCURRENT_SPARKLINE_REQUESTS = 6;
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
          pendingSparklineRequests.shift()?.();
        });
    };

    if (activeSparklineRequests < MAX_CONCURRENT_SPARKLINE_REQUESTS) {
      run();
    } else {
      pendingSparklineRequests.push(run);
    }
  });
}

function useNearViewport<T extends HTMLElement>(rootMargin = '720px') {
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

/**
 * Story 3.1 AC-1: compact row Heikin Ashi chart showing 30-day trend structure.
 * The percent chip remains real close-to-close performance; the row chart is
 * intentionally visual/noise-reduced.
 */
function SparklineInner({ ticker, width = 60, height = 28 }: SparklineProps) {
  const { ref: visibilityRef, isNearViewport } = useNearViewport<HTMLDivElement>();
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { data } = useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => runSparklineRequest(() => api.chartOhlcv(ticker, 30)),
    enabled: isNearViewport,
    staleTime: 600_000,
  });

  useEffect(() => {
    const bars = data?.data;
    if (!bars || bars.length < 3 || !canvasRef.current) return;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = width * dpr;
    canvas.height = height * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, width, height);

    const haBars = toHeikinAshiBars(bars as OHLCVBar[]);
    const closes = haBars.map((b) => b.close);
    const minP = Math.min(...haBars.map((b) => b.low));
    const maxP = Math.max(...haBars.map((b) => b.high));
    const range = maxP - minP || 1;
    const pad = 2;
    const padTop = 3;
    const w = width - pad * 2;
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
  }, [data, width, height]);

  const bars = data?.data;
  if (!bars || bars.length < 3) {
    return (
      <div
        ref={visibilityRef}
        style={{ width, height }}
        className="opacity-20 flex items-center justify-center text-[8px] text-[var(--text-muted)]"
      >
        —
      </div>
    );
  }

  const closes = bars.map((b: { close: number }) => b.close);
  const first = closes[0];
  const last = closes[closes.length - 1];
  const pctChg = first ? ((last - first) / first) * 100 : 0;
  const up = pctChg >= 0;

  return (
    <div ref={visibilityRef} style={{ width, height }} title="30-day Heikin Ashi row chart">
      <canvas
        ref={canvasRef}
        style={{
          width,
          height,
          display: 'block',
          filter: up
            ? 'drop-shadow(0 0 2px rgba(62,232,165,0.35))'
            : 'drop-shadow(0 0 2px rgba(255,107,138,0.35))',
        }}
      />
    </div>
  );
}

export const Sparkline = memo(SparklineInner);

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
