import { useRef, useEffect, memo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';

interface SparklineProps {
  ticker: string;
  width?: number;
  height?: number;
}

/**
 * Story 3.1 AC-1: Gradient area sparkline showing 30-day price movement.
 * Emerald if above 20-day SMA, rose if below. Gradient area fill underneath,
 * end-dot with concentric glow, and a tabular % chip in the top-right.
 */
function SparklineInner({ ticker, width = 60, height = 28 }: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { data } = useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => api.chartOhlcv(ticker, 30),
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

    const closes = bars.map((b: { close: number }) => b.close);
    const minP = Math.min(...closes);
    const maxP = Math.max(...closes);
    const range = maxP - minP || 1;
    const pad = 2;
    const padTop = 4; // leave room for chip
    const w = width - pad * 2;
    const h = height - pad - padTop;

    const smaWindow = Math.min(20, closes.length);
    const smaSlice = closes.slice(-smaWindow);
    const sma20 = smaSlice.reduce((a: number, b: number) => a + b, 0) / smaSlice.length;
    const lastClose = closes[closes.length - 1];
    const aboveSma = lastClose >= sma20;

    const strokeCol = aboveSma ? '#3ee8a5' : '#ff6b8a';
    const fillStart = aboveSma ? 'rgba(62,232,165,0.32)' : 'rgba(255,107,138,0.32)';
    const fillEnd = aboveSma ? 'rgba(62,232,165,0.00)' : 'rgba(255,107,138,0.00)';

    // Build path for line
    const pts: [number, number][] = closes.map((c: number, i: number) => {
      const x = pad + (i / (closes.length - 1)) * w;
      const y = padTop + h - ((c - minP) / range) * h;
      return [x, y];
    });

    // Gradient area fill
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

    // Line
    ctx.beginPath();
    pts.forEach(([x, y], i) => (i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y)));
    ctx.strokeStyle = strokeCol;
    ctx.lineWidth = 1.75;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    // End dot with glow ring
    const [lastX, lastY] = pts[pts.length - 1];
    ctx.beginPath();
    ctx.arc(lastX, lastY, 3.2, 0, Math.PI * 2);
    ctx.fillStyle = aboveSma ? 'rgba(62,232,165,0.22)' : 'rgba(255,107,138,0.22)';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 1.8, 0, Math.PI * 2);
    ctx.fillStyle = strokeCol;
    ctx.fill();
  }, [data, width, height]);

  const bars = data?.data;
  if (!bars || bars.length < 3) {
    return (
      <div
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
  );
}

export const Sparkline = memo(SparklineInner);

/**
 * 30-day percent change chip, split out from the Sparkline so it can live
 * in its own table column. Shares the same React-Query cache via queryKey.
 */
function SparklinePctInner({ ticker }: { ticker: string }) {
  const { data } = useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => api.chartOhlcv(ticker, 30),
    staleTime: 600_000,
  });
  const bars = data?.data;
  if (!bars || bars.length < 3) {
    return <span className="text-[10px] text-[var(--text-muted)]">—</span>;
  }
  const closes = bars.map((b: { close: number }) => b.close);
  const first = closes[0];
  const last = closes[closes.length - 1];
  const pctChg = first ? ((last - first) / first) * 100 : 0;
  const up = pctChg >= 0;
  return (
    <span
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
