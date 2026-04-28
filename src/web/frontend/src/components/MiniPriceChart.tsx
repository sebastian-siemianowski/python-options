/**
 * MiniPriceChart – compact Heikin Ashi/candlestick chart with SMA, Bollinger, and Forecast overlays.
 *
 * Designed for inline embedding (e.g. heatmap expanded rows).
 * All overlays are enabled by default — no toggle UI.
 */
import { useEffect, useMemo, useRef } from 'react';
import {
  createChart,
  ColorType,
  LineStyle,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  HistogramSeries,
} from 'lightweight-charts';
import type { OHLCVBar, Indicators, ForecastData } from '../api';
import { isHeikinAshiUp, toHeikinAshiBars } from '../utils/heikinAshi';

interface MiniPriceChartProps {
  ohlcv: OHLCVBar[];
  indicators?: Indicators | null;
  forecast?: ForecastData | null;
  height?: number;
  candleMode?: 'standard' | 'heikinAshi';
}

export default function MiniPriceChart({
  ohlcv,
  indicators,
  forecast,
  height = 340,
  candleMode = 'standard',
}: MiniPriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartBars = useMemo(
    () => (candleMode === 'heikinAshi' ? toHeikinAshiBars(ohlcv) : ohlcv),
    [candleMode, ohlcv],
  );

  useEffect(() => {
    if (!containerRef.current || !chartBars.length) return;

    const container = containerRef.current;
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#7a8ba4',
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(139,92,246,0.04)' },
        horzLines: { color: 'rgba(139,92,246,0.04)' },
      },
      width: container.clientWidth,
      height,
      crosshair: {
        vertLine: { color: 'rgba(139,92,246,0.4)', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#0a0a1a' },
        horzLine: { color: 'rgba(139,92,246,0.4)', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#0a0a1a' },
      },
      rightPriceScale: {
        borderColor: 'rgba(42,42,74,0.5)',
        scaleMargins: { top: 0.08, bottom: 0.18 },
      },
      timeScale: {
        borderColor: 'rgba(42,42,74,0.5)',
        timeVisible: false,
        rightOffset: 12,
        barSpacing: 6,
        minBarSpacing: 3,
      },
    });

    /* Main candles */
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#3ee8a5',
      downColor: '#ff6b8a',
      borderUpColor: '#3ee8a5',
      borderDownColor: '#ff6b8a',
      wickUpColor: '#6ff0c0',
      wickDownColor: '#fda4af',
      lastValueVisible: true,
      priceLineVisible: true,
    });
    candleSeries.setData(chartBars);

    /* Volume */
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    volumeSeries.setData(
      chartBars.map((d) => ({
        time: d.time,
        value: d.volume,
        color: isHeikinAshiUp(d) ? 'rgba(0,230,118,0.18)' : 'rgba(255,23,68,0.18)',
      })),
    );

    /* SMA + Bollinger */
    const smaOpts = { crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false };
    if (indicators) {
      if (indicators.sma20?.length)
        chart.addSeries(LineSeries, { color: '#f5c542', lineWidth: 1, ...smaOpts }).setData(indicators.sma20);
      if (indicators.sma50?.length)
        chart.addSeries(LineSeries, { color: '#b49aff', lineWidth: 1, ...smaOpts }).setData(indicators.sma50);
      if (indicators.sma200?.length)
        chart.addSeries(LineSeries, { color: '#c084fc', lineWidth: 1, ...smaOpts }).setData(indicators.sma200);

      if (indicators.bollinger) {
        const bbOpts = { lineWidth: 1 as const, lineStyle: LineStyle.Dotted, ...smaOpts };
        if (indicators.bollinger.upper?.length)
          chart.addSeries(LineSeries, { color: 'rgba(139,92,246,0.25)', ...bbOpts }).setData(indicators.bollinger.upper);
        if (indicators.bollinger.lower?.length)
          chart.addSeries(LineSeries, { color: 'rgba(139,92,246,0.25)', ...bbOpts }).setData(indicators.bollinger.lower);
      }
    }

    /* Forecast + CI */
    if (forecast?.forecasts?.length && chartBars.length > 0) {
      const lastCandle = chartBars[chartBars.length - 1];
      const lastPrice = lastCandle.close;
      const lastDate = new Date(lastCandle.time as string);

      const upperData: { time: string; value: number }[] = [{ time: lastCandle.time as string, value: lastPrice }];
      const lowerData: { time: string; value: number }[] = [{ time: lastCandle.time as string, value: lastPrice }];
      const medianData: { time: string; value: number }[] = [{ time: lastCandle.time as string, value: lastPrice }];

      for (const f of forecast.forecasts) {
        const futureDate = new Date(lastDate);
        futureDate.setDate(futureDate.getDate() + f.horizon_days);
        const dateStr = futureDate.toISOString().slice(0, 10);
        const retPct = f.expected_return_pct / 100;
        const medianPrice = lastPrice * (1 + retPct);
        const ciWidth = lastPrice * Math.abs(retPct) * 0.5 + lastPrice * 0.01 * Math.sqrt(f.horizon_days);
        upperData.push({ time: dateStr, value: medianPrice + ciWidth });
        lowerData.push({ time: dateStr, value: Math.max(0, medianPrice - ciWidth) });
        medianData.push({ time: dateStr, value: medianPrice });
      }

      const fcOpts = { priceScaleId: 'right' as const, ...smaOpts };
      chart.addSeries(AreaSeries, {
        topColor: 'rgba(139,92,246,0.12)', bottomColor: 'rgba(139,92,246,0.02)',
        lineColor: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, ...fcOpts,
      }).setData(upperData);
      chart.addSeries(AreaSeries, {
        topColor: 'rgba(139,92,246,0.02)', bottomColor: 'rgba(139,92,246,0.12)',
        lineColor: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, ...fcOpts,
      }).setData(lowerData);
      chart.addSeries(LineSeries, {
        color: '#b49aff', lineWidth: 2, lineStyle: LineStyle.Dashed, ...fcOpts,
      }).setData(medianData);
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (containerRef.current) chart.applyOptions({ width: containerRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [chartBars, indicators, forecast, height]);

  /* Legend strip */
  const legendItems = [
    ...(candleMode === 'heikinAshi' ? [{ label: 'Heikin Ashi', color: '#3ee8a5' }] : []),
    { label: 'SMA 20', color: '#f5c542' },
    { label: 'SMA 50', color: '#b49aff' },
    { label: 'SMA 200', color: '#c084fc' },
    { label: 'Bollinger', color: 'rgba(139,92,246,0.6)' },
    { label: 'Forecast', color: '#b49aff' },
    { label: 'CI', color: 'rgba(139,92,246,0.4)' },
  ];

  return (
    <div>
      <div className="flex items-center gap-3 mb-1.5 px-1">
        {legendItems.map((it) => (
          <div key={it.label} className="flex items-center gap-1">
            <span className="w-2.5 h-[2px] rounded-full" style={{ backgroundColor: it.color }} />
            <span className="text-[8px] font-medium" style={{ color: 'var(--text-muted)' }}>{it.label}</span>
          </div>
        ))}
      </div>
      <div
        ref={containerRef}
        style={{ width: '100%', height, borderRadius: 8, overflow: 'hidden' }}
      />
    </div>
  );
}
