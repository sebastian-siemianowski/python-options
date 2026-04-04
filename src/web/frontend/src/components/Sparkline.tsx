import { useRef, useEffect, memo } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '../api';

interface SparklineProps {
  ticker: string;
  width?: number;
  height?: number;
}

/**
 * Story 3.1 AC-1: 60px-wide sparkline showing 30-day price movement.
 * Emerald line if above 20-day SMA, rose if below. Rendered on transparent bg
 * with a luminous glow filter.
 */
function SparklineInner({ ticker, width = 60, height = 28 }: SparklineProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  const { data } = useQuery({
    queryKey: ['sparkline', ticker],
    queryFn: () => api.chartOhlcv(ticker, 30),
    staleTime: 600_000, // 10 min cache for sparklines
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
    const w = width - pad * 2;
    const h = height - pad * 2;

    // Compute 20-day SMA
    const smaWindow = Math.min(20, closes.length);
    const smaSlice = closes.slice(-smaWindow);
    const sma20 = smaSlice.reduce((a: number, b: number) => a + b, 0) / smaSlice.length;
    const lastClose = closes[closes.length - 1];
    const aboveSma = lastClose >= sma20;

    const emerald = '#34D399';
    const rose = '#FB7185';
    const lineColor = aboveSma ? emerald : rose;

    // Draw line
    ctx.beginPath();
    closes.forEach((c: number, i: number) => {
      const x = pad + (i / (closes.length - 1)) * w;
      const y = pad + h - ((c - minP) / range) * h;
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    });
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.stroke();

    // End dot
    const lastX = pad + w;
    const lastY = pad + h - ((lastClose - minP) / range) * h;
    ctx.beginPath();
    ctx.arc(lastX, lastY, 1.5, 0, Math.PI * 2);
    ctx.fillStyle = lineColor;
    ctx.fill();
  }, [data, width, height]);

  const bars = data?.data;
  if (!bars || bars.length < 3) {
    return <div style={{ width, height }} className="opacity-20 flex items-center justify-center text-[8px] text-[var(--text-muted)]">--</div>;
  }

  // Compute SMA direction for the glow filter
  const closes = bars.map((b: { close: number }) => b.close);
  const smaWindow = Math.min(20, closes.length);
  const smaSlice = closes.slice(-smaWindow);
  const sma20 = smaSlice.reduce((a: number, b: number) => a + b, 0) / smaSlice.length;
  const aboveSma = closes[closes.length - 1] >= sma20;

  return (
    <canvas
      ref={canvasRef}
      style={{
        width,
        height,
        filter: aboveSma
          ? 'drop-shadow(0 0 2px rgba(52,211,153,0.4))'
          : 'drop-shadow(0 0 2px rgba(251,113,133,0.4))',
      }}
    />
  );
}

export const Sparkline = memo(SparklineInner);
