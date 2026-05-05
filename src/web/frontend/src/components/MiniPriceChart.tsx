/**
 * MiniPriceChart – compact Heikin Ashi/candlestick chart with SMA, Bollinger, and Forecast overlays.
 *
 * Designed for inline embedding (e.g. heatmap expanded rows).
 * Overlays are enabled by default; callers can opt into a compact toggle UI.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import {
  createChart,
  ColorType,
  LineStyle,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  HistogramSeries,
  createSeriesMarkers,
  type AreaData,
  type SeriesMarker,
} from 'lightweight-charts';
import type { OHLCVBar, Indicators, ForecastData } from '../api';
import { isHeikinAshiUp, toHeikinAshiBars } from '../utils/heikinAshi';

export type MiniPriceChartView = 'sma' | 'heikinAshi' | 'reversal' | 'area';
type OverlayKey = 'sma20' | 'sma50' | 'sma200' | 'bollinger' | 'forecast' | 'ci' | 'currentPrice';
type OverlayVisibility = Record<OverlayKey, boolean>;

const OVERLAY_KEYS: OverlayKey[] = ['sma20', 'sma50', 'sma200', 'bollinger', 'forecast', 'ci', 'currentPrice'];
const SMA_OVERLAY_PREFS_KEY = 'signals.smaReversalChartOverlays.v1';

const DEFAULT_OVERLAYS: OverlayVisibility = {
  sma20: true,
  sma50: true,
  sma200: true,
  bollinger: true,
  forecast: true,
  ci: true,
  currentPrice: true,
};

const ANALYSIS_OVERLAY_ITEMS: Array<{ key: OverlayKey; label: string; color: string; title: string }> = [
  { key: 'sma20', label: 'SMA 20', color: '#f5c542', title: 'Toggle the 20-day SMA overlay' },
  { key: 'sma50', label: 'SMA 50', color: '#b49aff', title: 'Toggle the 50-day SMA overlay' },
  { key: 'sma200', label: 'SMA 200', color: '#c084fc', title: 'Toggle the 200-day SMA overlay' },
  { key: 'bollinger', label: 'Bollinger', color: 'rgba(139,92,246,0.72)', title: 'Toggle Bollinger bands' },
  { key: 'forecast', label: 'Forecast', color: '#b49aff', title: 'Toggle the forecast path' },
  { key: 'ci', label: 'CI', color: 'rgba(139,92,246,0.55)', title: 'Toggle the forecast confidence interval' },
];

const CURRENT_PRICE_ITEM = {
  key: 'currentPrice' as const,
  label: 'Price line',
  color: '#60a5fa',
  title: 'Toggle the current price line',
};

function sanitizeOverlayVisibility(candidate: unknown): OverlayVisibility {
  const next = { ...DEFAULT_OVERLAYS };
  if (!candidate || typeof candidate !== 'object') return next;
  const raw = candidate as Partial<Record<OverlayKey, unknown>>;
  for (const key of OVERLAY_KEYS) {
    if (typeof raw[key] === 'boolean') next[key] = raw[key];
  }
  return next;
}

function loadStoredOverlayVisibility(): OverlayVisibility {
  if (typeof window === 'undefined') return { ...DEFAULT_OVERLAYS };
  try {
    const raw = window.localStorage.getItem(SMA_OVERLAY_PREFS_KEY);
    return raw ? sanitizeOverlayVisibility(JSON.parse(raw)) : { ...DEFAULT_OVERLAYS };
  } catch {
    return { ...DEFAULT_OVERLAYS };
  }
}

function saveStoredOverlayVisibility(overlays: OverlayVisibility): void {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(SMA_OVERLAY_PREFS_KEY, JSON.stringify(overlays));
  } catch {
    // Ignore storage failures; the in-memory controls still work for this chart.
  }
}

interface MiniPriceChartProps {
  ohlcv: OHLCVBar[];
  indicators?: Indicators | null;
  forecast?: ForecastData | null;
  height?: number;
  candleMode?: 'standard' | 'heikinAshi';
  viewMode?: MiniPriceChartView;
  showOverlayControls?: boolean;
}

interface ReversalCandle {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  color: string;
  wickColor: string;
  borderColor: string;
}

interface MiniReversalModel {
  candles: ReversalCandle[];
  area: AreaData<string>[];
  markers: SeriesMarker<string>[];
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
    const avg = slice.reduce((sum, v) => sum + v, 0) / Math.max(1, slice.length);
    return avg || Math.max(1e-9, bars[i].high - bars[i].low);
  });
}

function buildMiniReversalModel(bars: OHLCVBar[]): MiniReversalModel {
  if (bars.length === 0) return { candles: [], area: [], markers: [] };

  const atr = computeAtr(bars);
  const multiplier = 2.4;
  let trend: 1 | -1 = bars[0].close >= bars[0].open ? 1 : -1;
  let finalUpper = (bars[0].high + bars[0].low) / 2 + multiplier * atr[0];
  let finalLower = (bars[0].high + bars[0].low) / 2 - multiplier * atr[0];
  const candles: ReversalCandle[] = [];
  const area: AreaData<string>[] = [];
  const markers: SeriesMarker<string>[] = [];

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

    const color = trend === 1
      ? isFlip
        ? '#00f5a0'
        : isAligned
          ? '#34d399'
          : '#86efac'
      : isFlip
        ? '#ff375f'
        : isAligned
          ? '#fb7185'
          : '#fca5a5';
    const areaTopColor = trend === 1
      ? isFlip
        ? 'rgba(0,245,160,0.36)'
        : isAligned
          ? 'rgba(16,185,129,0.28)'
          : 'rgba(134,239,172,0.16)'
      : isFlip
        ? 'rgba(255,55,95,0.34)'
        : isAligned
          ? 'rgba(244,63,94,0.26)'
          : 'rgba(252,165,165,0.14)';

    candles.push({
      time: bar.time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      color,
      wickColor: trend === 1 ? 'rgba(110,231,183,0.92)' : 'rgba(253,164,175,0.92)',
      borderColor: isFlip ? '#f8fafc' : color,
    });
    area.push({
      time: bar.time,
      value: bar.close,
      lineColor: color,
      topColor: areaTopColor,
      bottomColor: trend === 1 ? 'rgba(16,185,129,0)' : 'rgba(244,63,94,0)',
    });

    if (isFlip) {
      markers.push({
        id: `${bar.time}-${trend === 1 ? 'buy' : 'sell'}`,
        time: bar.time,
        position: trend === 1 ? 'belowBar' : 'aboveBar',
        color: trend === 1 ? '#00f5a0' : '#ff375f',
        shape: trend === 1 ? 'arrowUp' : 'arrowDown',
        text: trend === 1 ? 'BUY' : 'SELL',
        size: 1.35,
      });
    }
  });

  return { candles, area, markers };
}

export default function MiniPriceChart({
  ohlcv,
  indicators,
  forecast,
  height = 340,
  candleMode = 'standard',
  viewMode,
  showOverlayControls = false,
}: MiniPriceChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const [overlays, setOverlays] = useState<OverlayVisibility>(() =>
    showOverlayControls ? loadStoredOverlayVisibility() : { ...DEFAULT_OVERLAYS },
  );
  const effectiveView: MiniPriceChartView = viewMode ?? (candleMode === 'heikinAshi' ? 'heikinAshi' : 'sma');
  const showAnalysisOverlays = effectiveView === 'sma' || effectiveView === 'heikinAshi';
  const chartBars = useMemo(
    () => (effectiveView === 'heikinAshi' ? toHeikinAshiBars(ohlcv) : ohlcv),
    [effectiveView, ohlcv],
  );

  useEffect(() => {
    if (showOverlayControls) setOverlays(loadStoredOverlayVisibility());
  }, [showOverlayControls]);

  useEffect(() => {
    if (showOverlayControls) saveStoredOverlayVisibility(overlays);
  }, [overlays, showOverlayControls]);

  const toggleOverlay = (key: OverlayKey) => {
    setOverlays((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const visibleToggleItems = showAnalysisOverlays
    ? [...ANALYSIS_OVERLAY_ITEMS, CURRENT_PRICE_ITEM]
    : [CURRENT_PRICE_ITEM];
  const allVisibleItemsOn = visibleToggleItems.every((item) => overlays[item.key]);

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

    const reversal = effectiveView === 'reversal' || effectiveView === 'area'
      ? buildMiniReversalModel(ohlcv)
      : null;

    if (effectiveView === 'area') {
      const areaSeries = chart.addSeries(AreaSeries, {
        topColor: 'rgba(16,185,129,0.26)',
        bottomColor: 'rgba(16,185,129,0)',
        lineColor: '#34d399',
        lineWidth: 2,
        lastValueVisible: true,
        priceLineVisible: overlays.currentPrice,
      });
      areaSeries.setData(reversal?.area ?? []);
      if (reversal?.markers.length) createSeriesMarkers(areaSeries, reversal.markers, { zOrder: 'top' });
    } else {
      const candleSeries = chart.addSeries(CandlestickSeries, {
        upColor: '#3ee8a5',
        downColor: '#ff6b8a',
        borderUpColor: '#3ee8a5',
        borderDownColor: '#ff6b8a',
        wickUpColor: '#6ff0c0',
        wickDownColor: '#fda4af',
        borderVisible: effectiveView === 'reversal',
        lastValueVisible: true,
        priceLineVisible: overlays.currentPrice,
      });
      if (effectiveView === 'reversal') {
        candleSeries.setData(reversal?.candles ?? []);
        if (reversal?.markers.length) createSeriesMarkers(candleSeries, reversal.markers, { zOrder: 'top' });
      } else {
        candleSeries.setData(chartBars);
      }
    }

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
    if (showAnalysisOverlays && indicators) {
      if (overlays.sma20 && indicators.sma20?.length)
        chart.addSeries(LineSeries, { color: '#f5c542', lineWidth: 1, ...smaOpts }).setData(indicators.sma20);
      if (overlays.sma50 && indicators.sma50?.length)
        chart.addSeries(LineSeries, { color: '#b49aff', lineWidth: 1, ...smaOpts }).setData(indicators.sma50);
      if (overlays.sma200 && indicators.sma200?.length)
        chart.addSeries(LineSeries, { color: '#c084fc', lineWidth: 1, ...smaOpts }).setData(indicators.sma200);

      if (overlays.bollinger && indicators.bollinger) {
        const bbOpts = { lineWidth: 1 as const, lineStyle: LineStyle.Dotted, ...smaOpts };
        if (indicators.bollinger.upper?.length)
          chart.addSeries(LineSeries, { color: 'rgba(139,92,246,0.25)', ...bbOpts }).setData(indicators.bollinger.upper);
        if (indicators.bollinger.lower?.length)
          chart.addSeries(LineSeries, { color: 'rgba(139,92,246,0.25)', ...bbOpts }).setData(indicators.bollinger.lower);
      }
    }

    /* Forecast + CI */
    if (showAnalysisOverlays && forecast?.forecasts?.length && chartBars.length > 0 && (overlays.forecast || overlays.ci)) {
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
      if (overlays.ci) {
        chart.addSeries(AreaSeries, {
          topColor: 'rgba(139,92,246,0.12)', bottomColor: 'rgba(139,92,246,0.02)',
          lineColor: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, ...fcOpts,
        }).setData(upperData);
        chart.addSeries(AreaSeries, {
          topColor: 'rgba(139,92,246,0.02)', bottomColor: 'rgba(139,92,246,0.12)',
          lineColor: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, ...fcOpts,
        }).setData(lowerData);
      }
      if (overlays.forecast) {
        chart.addSeries(LineSeries, {
          color: '#b49aff', lineWidth: 2, lineStyle: LineStyle.Dashed, ...fcOpts,
        }).setData(medianData);
      }
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
  }, [chartBars, effectiveView, forecast, height, indicators, ohlcv, overlays, showAnalysisOverlays]);

  /* Legend strip */
  const modeLegendItems = effectiveView === 'reversal' || effectiveView === 'area'
    ? [
        { label: 'BUY flip', color: '#00f5a0' },
        { label: 'SELL flip', color: '#ff375f' },
        { label: effectiveView === 'area' ? 'Trend gradient' : 'Reversal candles', color: '#34d399' },
      ]
    : [];
  const legendItems = effectiveView === 'reversal' || effectiveView === 'area'
    ? [
        { label: 'BUY flip', color: '#00f5a0' },
        { label: 'SELL flip', color: '#ff375f' },
        { label: effectiveView === 'area' ? 'Trend gradient' : 'Reversal candles', color: '#34d399' },
      ]
    : [
        { label: effectiveView === 'heikinAshi' ? 'Heikin Ashi' : 'Price', color: '#3ee8a5' },
        { label: 'SMA 20', color: '#f5c542' },
        { label: 'SMA 50', color: '#b49aff' },
        { label: 'SMA 200', color: '#c084fc' },
        { label: 'Bollinger', color: 'rgba(139,92,246,0.6)' },
        { label: 'Forecast', color: '#b49aff' },
        { label: 'CI', color: 'rgba(139,92,246,0.4)' },
      ];

  return (
    <div>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-1.5 px-1">
        <div className="flex flex-wrap items-center gap-1.5">
          {showOverlayControls ? (
            <>
              {modeLegendItems.map((it) => (
                <div
                  key={it.label}
                  className="inline-flex items-center gap-1 rounded-md px-1.5 py-[3px]"
                  style={{ background: 'rgba(255,255,255,0.018)', border: '1px solid rgba(255,255,255,0.04)' }}
                >
                  <span className="w-2.5 h-[2px] rounded-full" style={{ backgroundColor: it.color }} />
                  <span className="text-[8px] font-medium" style={{ color: 'var(--text-muted)' }}>{it.label}</span>
                </div>
              ))}
              {visibleToggleItems.map((it) => {
                const on = overlays[it.key];
                return (
                  <button
                    key={it.key}
                    type="button"
                    title={it.title}
                    aria-pressed={on}
                    onClick={() => toggleOverlay(it.key)}
                    className="inline-flex items-center gap-1 rounded-md px-1.5 py-[3px] text-[8px] font-semibold transition-all"
                    style={{
                      background: on ? 'rgba(255,255,255,0.045)' : 'rgba(255,255,255,0.012)',
                      border: `1px solid ${on ? it.color : 'rgba(255,255,255,0.04)'}`,
                      color: on ? 'var(--text-primary)' : 'var(--text-muted)',
                      opacity: on ? 1 : 0.58,
                      boxShadow: on ? `0 0 10px -7px ${it.color}` : 'none',
                    }}
                  >
                    <span
                      className="rounded-full"
                      style={{
                        width: 6,
                        height: 6,
                        backgroundColor: on ? it.color : 'rgba(148,163,184,0.35)',
                        boxShadow: on ? `0 0 6px ${it.color}` : 'none',
                      }}
                    />
                    <span className="whitespace-nowrap">{it.label}</span>
                  </button>
                );
              })}
            </>
          ) : (
            legendItems.map((it) => (
              <div key={it.label} className="flex items-center gap-1">
                <span className="w-2.5 h-[2px] rounded-full" style={{ backgroundColor: it.color }} />
                <span className="text-[8px] font-medium" style={{ color: 'var(--text-muted)' }}>{it.label}</span>
              </div>
            ))
          )}
        </div>

        {showOverlayControls && !allVisibleItemsOn && (
          <button
            type="button"
            title="Restore and save the default overlay set for SMA reversal charts"
            onClick={() => setOverlays({ ...DEFAULT_OVERLAYS })}
            className="rounded-md px-1.5 py-[3px] text-[8px] font-semibold transition-all"
            style={{
              color: '#c4b5fd',
              background: 'rgba(167,139,250,0.08)',
              border: '1px solid rgba(167,139,250,0.22)',
            }}
          >
            Reset
          </button>
        )}
      </div>
      <div
        ref={containerRef}
        style={{ width: '100%', height, borderRadius: 8, overflow: 'hidden' }}
      />
    </div>
  );
}
