/**
 * SignalDetailPanel — Story 3 of signals.md
 *
 * Premium TradingView-powered subpanel that expands when a user clicks a row on /signals.
 * Replaces the canvas-based MiniChartPanel with lightweight-charts v5.
 *
 * Design contract:
 *  - Apple Vision Pro visual language (hairlines, matte glass, violet→cyan accent).
 *  - Expands height 0 → 360px with cubic-bezier(0.22, 1, 0.36, 1) in 280ms.
 *  - Symbol-swap fades opacity during transition (handled by React remounting via `key={ticker}`).
 *  - No literal hex outside the TV_THEME block.
 *  - Zero chart leaks: chart.remove() always runs in cleanup.
 */
import { useEffect, useMemo, useRef, useState } from 'react';
import { useQuery, useQueryClient } from '@tanstack/react-query';
import {
  createChart,
  createSeriesMarkers,
  CandlestickSeries,
  AreaSeries,
  HistogramSeries,
  CrosshairMode,
  LineStyle,
  ColorType,
  type IChartApi,
  type AreaData,
  type SeriesMarker,
} from 'lightweight-charts';
import { ArrowUpRight, BarChart3, TrendingUp, AlertTriangle, ExternalLink, RefreshCw } from 'lucide-react';
import { api, type OHLCVBar } from '../api';
import { isHeikinAshiUp, toHeikinAshiBars } from '../utils/heikinAshi';

// Cast OHLCVBar.time (string) to Time type lightweight-charts expects.
// 'YYYY-MM-DD' strings are accepted natively.


export type SignalDetailChartType = 'candles' | 'reversal' | 'area';
type ChartType = SignalDetailChartType;
type RangeKey = '1M' | '3M' | '6M' | '1Y' | 'MAX';
type ChartOhlcvResponse = { symbol: string; data: OHLCVBar[]; count: number };
type ReversalTrend = 1 | -1;

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

interface ReversalStatePoint {
  time: string;
  trend: ReversalTrend;
  isFlip: boolean;
}

interface ReversalModel {
  candles: ReversalCandle[];
  area: AreaData<string>[];
  markers: SeriesMarker<string>[];
  states: ReversalStatePoint[];
}

const RANGE_DAYS: Record<RangeKey, number> = {
  '1M': 22, '3M': 66, '6M': 132, '1Y': 252, 'MAX': 10000,
};

/** Frozen theme object — single source of truth for chart colors. */
const TV_THEME = Object.freeze({
  layout: {
    background: { type: ColorType.Solid, color: 'transparent' },
    textColor: '#a8b2c8',
    fontFamily: 'Inter, system-ui, sans-serif',
    fontSize: 10,
  },
  grid: {
    vertLines: { color: 'rgba(255,255,255,0.025)', style: LineStyle.Solid },
    horzLines: { color: 'rgba(255,255,255,0.035)', style: LineStyle.Solid },
  },
  crosshair: {
    mode: CrosshairMode.Magnet,
    vertLine: { color: 'rgba(139,92,246,0.45)', width: 1 as const, style: LineStyle.Dashed, labelBackgroundColor: '#1a1036' },
    horzLine: { color: 'rgba(139,92,246,0.45)', width: 1 as const, style: LineStyle.Dashed, labelBackgroundColor: '#1a1036' },
  },
  rightPriceScale: {
    borderColor: 'rgba(255,255,255,0.05)',
  },
  timeScale: {
    borderColor: 'rgba(255,255,255,0.05)',
    timeVisible: true,
    secondsVisible: false,
  },
  candles: {
    upColor: '#10b981',
    downColor: '#f43f5e',
    borderVisible: false,
    wickUpColor: '#10b981',
    wickDownColor: '#f43f5e',
  },
  area: {
    lineColor: '#34d399',
    topColor: 'rgba(16,185,129,0.26)',
    bottomColor: 'rgba(16,185,129,0)',
    lineWidth: 2 as const,
    priceLineVisible: false,
    lastValueVisible: true,
  },
  volumeUp: 'rgba(16,185,129,0.35)',
  volumeDown: 'rgba(244,63,94,0.35)',
});

function formatPrice(v: number): string {
  if (v === 0 || !isFinite(v)) return '—';
  if (v >= 1000) return v.toLocaleString('en-US', { maximumFractionDigits: 2 });
  if (v >= 10) return v.toFixed(2);
  return v.toFixed(4);
}

function signalColor(label: string | undefined): string {
  const s = (label || '').toUpperCase();
  if (s === 'STRONG BUY' || s === 'STRONG_BUY') return '#10b981';
  if (s === 'BUY') return '#6ee7b7';
  if (s === 'HOLD') return '#94a3b8';
  if (s === 'SELL') return '#fca5a5';
  if (s === 'STRONG SELL' || s === 'STRONG_SELL') return '#f43f5e';
  if (s === 'EXIT') return '#f59e0b';
  return '#94a3b8';
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

function buildReversalModel(bars: OHLCVBar[]): ReversalModel {
  if (bars.length === 0) return { candles: [], area: [], markers: [], states: [] };

  const atr = computeAtr(bars);
  const multiplier = 2.4;
  let trend: ReversalTrend = bars[0].close >= bars[0].open ? 1 : -1;
  let finalUpper = (bars[0].high + bars[0].low) / 2 + multiplier * atr[0];
  let finalLower = (bars[0].high + bars[0].low) / 2 - multiplier * atr[0];
  const candles: ReversalCandle[] = [];
  const area: AreaData<string>[] = [];
  const markers: SeriesMarker<string>[] = [];
  const states: ReversalStatePoint[] = [];

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
    const areaBottomColor = trend === 1
      ? 'rgba(16,185,129,0)'
      : 'rgba(244,63,94,0)';

    const reversalPoint: ReversalCandle = {
      time: bar.time,
      open: bar.open,
      high: bar.high,
      low: bar.low,
      close: bar.close,
      color,
      wickColor: trend === 1 ? 'rgba(110,231,183,0.92)' : 'rgba(253,164,175,0.92)',
      borderColor: isFlip ? '#f8fafc' : color,
    };

    candles.push(reversalPoint);
    area.push({
      time: bar.time,
      value: bar.close,
      lineColor: color,
      topColor: areaTopColor,
      bottomColor: areaBottomColor,
    });
    states.push({ time: bar.time, trend, isFlip });

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

  return { candles, area, markers, states };
}

export interface SignalDetailPanelProps {
  ticker: string;
  /** Optional — stats strip shows these if provided. */
  signal?: string;
  momentum?: number;
  crashRisk?: number;
  /** Horizons from the row (e.g. [7, 30, 90]) — rendered as a small forecast list. */
  horizonSignals?: Record<string, { exp_ret?: number; p_up?: number; kelly_half?: number; label?: string }>;
  defaultChartType?: ChartType;
  defaultRange?: RangeKey;
  onNavigateChart: () => void;
}

export default function SignalDetailPanel({
  ticker,
  signal,
  momentum,
  crashRisk,
  horizonSignals,
  defaultChartType = 'area',
  defaultRange = '1Y',
  onNavigateChart,
}: SignalDetailPanelProps) {
  const [chartType, setChartType] = useState<ChartType>(defaultChartType);
  const [range, setRange] = useState<RangeKey>(defaultRange);
  const queryClient = useQueryClient();

  // Request a generous tail so range switches never require refetch.
  const { data, isLoading, error } = useQuery({
    queryKey: ['signalDetail', ticker],
    queryFn: () => api.chartOhlcv(ticker, 365),
    placeholderData: () => queryClient.getQueryData<ChartOhlcvResponse>(['sparkline', ticker]),
    staleTime: 300_000,
  });

  const bars = useMemo<OHLCVBar[]>(() => data?.data ?? [], [data]);

  // Slice for the current range (keeps chart responsive without refetching).
  const visibleBars = useMemo(() => {
    if (range === 'MAX') return bars;
    const n = RANGE_DAYS[range];
    return bars.length > n ? bars.slice(-n) : bars;
  }, [bars, range]);
  const chartBars = useMemo(
    () => (chartType === 'candles' ? toHeikinAshiBars(visibleBars) : visibleBars),
    [chartType, visibleBars],
  );

  const lastBar = visibleBars[visibleBars.length - 1];
  const firstBar = visibleBars[0];
  const lastPrice = lastBar?.close ?? 0;
  const delta1d = (() => {
    if (visibleBars.length < 2) return 0;
    const prev = visibleBars[visibleBars.length - 2].close;
    return prev ? ((lastPrice - prev) / prev) * 100 : 0;
  })();
  const rangePct = firstBar && firstBar.close
    ? ((lastPrice - firstBar.close) / firstBar.close) * 100
    : 0;
  const hasChartData = visibleBars.length >= 2;
  const isInitialLoading = isLoading && !hasChartData;

  return (
    <div
      className="signal-detail-panel"
      role="region"
      aria-label={`${ticker} price chart`}
      style={{
        display: 'grid',
        gridTemplateRows: '1fr',
        background: 'linear-gradient(180deg, rgba(13,13,24,0.65) 0%, rgba(8,8,18,0.75) 100%)',
        borderTop: '1px solid rgba(255,255,255,0.05)',
        borderBottom: '1px solid rgba(255,255,255,0.05)',
        animation: 'sdpExpand 320ms cubic-bezier(0.22, 1, 0.36, 1) both',
      }}
    >
      <style>{`
        @keyframes sdpExpand {
          from { grid-template-rows: 0fr; opacity: 0; }
          to   { grid-template-rows: 1fr; opacity: 1; }
        }
        .signal-detail-panel > .sdp-inner {
          min-height: 0;
          overflow: hidden;
        }
        @media (prefers-reduced-motion: reduce) {
          .signal-detail-panel { animation: none !important; }
        }
      `}</style>
      <div className="sdp-inner">

      {/* Header strip — ticker label · chart type · range · full view */}
      <div
        className="flex items-center justify-between"
        style={{
          height: 44,
          padding: '0 18px',
          borderBottom: '1px solid rgba(255,255,255,0.035)',
        }}
      >
        <div className="flex items-center gap-3">
          <span
            className="label-micro tabular-nums"
            style={{
              color: 'var(--text-muted)',
              letterSpacing: '0.18em',
              fontWeight: 600,
            }}
          >
            {ticker}
          </span>
          <span
            aria-hidden="true"
            style={{
              width: 1,
              height: 12,
              background: 'rgba(255,255,255,0.08)',
            }}
          />
          <SegmentedToggle
            value={chartType}
            onChange={(v) => setChartType(v as ChartType)}
            options={[
              { value: 'candles', label: 'Heikin Ashi', icon: <BarChart3 className="w-3 h-3" /> },
              { value: 'reversal', label: 'Reversal', icon: <RefreshCw className="w-3 h-3" /> },
              { value: 'area', label: 'Area', icon: <TrendingUp className="w-3 h-3" /> },
            ]}
          />
        </div>

        <div className="flex items-center gap-3">
          <SegmentedToggle
            value={range}
            onChange={(v) => setRange(v as RangeKey)}
            options={[
              { value: '1M', label: '1M' },
              { value: '3M', label: '3M' },
              { value: '6M', label: '6M' },
              { value: '1Y', label: '1Y' },
              { value: 'MAX', label: 'MAX' },
            ]}
            dense
          />
          <span
            aria-hidden="true"
            style={{
              width: 1,
              height: 12,
              background: 'rgba(255,255,255,0.08)',
            }}
          />
          <button
            onClick={onNavigateChart}
            className="flex items-center gap-1.5 transition-all duration-150"
            style={{
              height: 26,
              padding: '0 10px',
              borderRadius: 999,
              fontSize: 10,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              fontWeight: 600,
              color: '#e9d5ff',
              background:
                'linear-gradient(135deg, rgba(139,92,246,0.16), rgba(6,182,212,0.12))',
              border: '1px solid rgba(139,92,246,0.28)',
            }}
            title="Open in full chart view"
          >
            Full view
            <ExternalLink className="w-3 h-3" />
          </button>
        </div>
      </div>

      {/* Body: chart (72%) + stats strip (28%) */}
      <div className="flex items-stretch" style={{ height: 320 }}>
        <div
          className="flex-1 relative"
          style={{ minWidth: 0, padding: '10px 6px 10px 14px' }}
        >
          {isInitialLoading ? (
            <ChartShimmer />
          ) : !hasChartData ? (
            <ChartEmpty message={error ? 'Chart unavailable' : 'No data yet'} />
          ) : (
            <TradingViewChart
              bars={chartBars}
              reversalBars={visibleBars}
              chartType={chartType}
              key={`${ticker}-${chartType}-${range}`}
            />
          )}
        </div>

        <div
          className="flex-shrink-0"
          style={{
            width: 220,
            borderLeft: '1px solid rgba(255,255,255,0.035)',
            padding: '14px 16px',
            display: 'flex',
            flexDirection: 'column',
            gap: 14,
          }}
        >
          <StatsStrip
            signal={signal}
            momentum={momentum}
            crashRisk={crashRisk}
            lastPrice={lastPrice}
            delta1d={delta1d}
            rangePct={rangePct}
            rangeLabel={range}
            horizonSignals={horizonSignals}
          />
        </div>
      </div>
      </div>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* TradingView chart (lightweight-charts v5)                                 */
/* ──────────────────────────────────────────────────────────────────────── */

function TradingViewChart({ bars, reversalBars, chartType }: { bars: OHLCVBar[]; reversalBars: OHLCVBar[]; chartType: ChartType }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
    const reversal = buildReversalModel(reversalBars);
    const chart = createChart(container, {
      layout: TV_THEME.layout,
      grid: TV_THEME.grid,
      crosshair: TV_THEME.crosshair,
      rightPriceScale: TV_THEME.rightPriceScale,
      timeScale: TV_THEME.timeScale,
      width: container.clientWidth,
      height: container.clientHeight,
      handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true },
      handleScroll: { pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
    });
    chartRef.current = chart;

    // Main price series
    if (chartType === 'candles' || chartType === 'reversal') {
      const candleSeries = chart.addSeries(CandlestickSeries, {
        ...TV_THEME.candles,
        borderVisible: chartType === 'reversal',
      });
      if (chartType === 'reversal') {
        candleSeries.setData(reversal.candles);
        createSeriesMarkers(candleSeries, reversal.markers, { zOrder: 'top' });
      } else {
        candleSeries.setData(
          bars.map((b) => ({
            time: b.time,
            open: b.open,
            high: b.high,
            low: b.low,
            close: b.close,
          })),
        );
      }
    } else {
      const areaSeries = chart.addSeries(AreaSeries, TV_THEME.area);
      areaSeries.setData(reversal.area);
      createSeriesMarkers(areaSeries, reversal.markers, { zOrder: 'top' });
    }

    // Volume histogram on its own price scale
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
      color: TV_THEME.volumeUp,
    });
    chart.priceScale('volume').applyOptions({
      scaleMargins: { top: 0.82, bottom: 0 },
      visible: false,
    });
    volumeSeries.setData(
      bars.map((b) => ({
        time: b.time,
        value: b.volume,
        color: isHeikinAshiUp(b) ? TV_THEME.volumeUp : TV_THEME.volumeDown,
      })),
    );

    // Reversal regime rail: rendered by lightweight-charts so it shares the exact
    // same time scale as the price chart instead of relying on a separate DOM rail.
    if (reversal.states.length > 1) {
      const regimeSeries = chart.addSeries(HistogramSeries, {
        priceFormat: { type: 'volume' },
        priceScaleId: 'regime',
        color: 'rgba(16,185,129,0.70)',
        priceLineVisible: false,
        lastValueVisible: false,
      });
      chart.priceScale('regime').applyOptions({
        scaleMargins: { top: 0.958, bottom: 0.012 },
        visible: false,
      });
      regimeSeries.setData(
        reversal.states.map((state) => ({
          time: state.time,
          value: 1,
          color: state.trend === 1
            ? state.isFlip ? 'rgba(0,245,160,0.98)' : 'rgba(16,185,129,0.74)'
            : state.isFlip ? 'rgba(255,55,95,0.98)' : 'rgba(244,63,94,0.74)',
        })),
      );
    }

    chart.timeScale().fitContent();

    // Resize handler
    const ro = new ResizeObserver(() => {
      if (!chartRef.current || !containerRef.current) return;
      chartRef.current.applyOptions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
      });
    });
    ro.observe(container);

    return () => {
      ro.disconnect();
      chart.remove();
      chartRef.current = null;
    };
  }, [bars, chartType, reversalBars]);

  return (
    <div
      ref={containerRef}
      className="h-full min-h-0"
      style={{ width: '100%' }}
      title="Bottom reversal rail: green means BUY regime, red means SELL regime"
    />
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Stats strip                                                              */
/* ──────────────────────────────────────────────────────────────────────── */

function StatsStrip({
  signal,
  momentum,
  crashRisk,
  lastPrice,
  delta1d,
  rangePct,
  rangeLabel,
  horizonSignals,
}: {
  signal?: string;
  momentum?: number;
  crashRisk?: number;
  lastPrice: number;
  delta1d: number;
  rangePct: number;
  rangeLabel: RangeKey;
  horizonSignals?: SignalDetailPanelProps['horizonSignals'];
}) {
  const sigColor = signalColor(signal);
  const deltaUp = delta1d >= 0;
  const rangeUp = rangePct >= 0;

  return (
    <>
      {/* Signal */}
      <div className="flex flex-col gap-1">
        <span className="label-micro">Signal</span>
        <div className="flex items-center gap-2">
          <span
            className="w-1.5 h-1.5 rounded-full flex-shrink-0"
            style={{ background: sigColor, boxShadow: `0 0 6px ${sigColor}66` }}
          />
          <span
            className="text-white font-semibold"
            style={{ fontSize: 12, letterSpacing: '0.04em' }}
          >
            {(signal || '—').toUpperCase()}
          </span>
        </div>
      </div>

      {/* Last price + Δ 1d */}
      <div className="flex flex-col gap-1">
        <span className="label-micro">Last · Δ 1d</span>
        <div className="flex items-baseline gap-2">
          <span className="num-display text-white" style={{ fontSize: 20 }}>
            {formatPrice(lastPrice)}
          </span>
          <span
            className="tabular-nums"
            style={{
              fontSize: 11,
              color: deltaUp ? '#10b981' : '#f43f5e',
            }}
          >
            {deltaUp ? '+' : ''}
            {delta1d.toFixed(2)}%
          </span>
        </div>
        <span
          className="tabular-nums"
          style={{ fontSize: 10, color: 'var(--text-muted)' }}
        >
          {rangeLabel}: {rangeUp ? '+' : ''}
          {rangePct.toFixed(2)}%
        </span>
      </div>

      {/* Momentum + Crash risk — backend values are 0-100 integers */}
      <div className="grid grid-cols-2 gap-3">
        <MiniMetric
          label="Momentum"
          value={momentum != null ? `${Math.round(momentum)}%` : '—'}
          color={momentum != null ? (momentum >= 0 ? '#10b981' : '#f43f5e') : undefined}
        />
        <MiniMetric
          label="Crash risk"
          value={crashRisk != null ? `${Math.round(crashRisk)}%` : '—'}
          color={
            crashRisk != null
              ? crashRisk > 50
                ? '#f43f5e'
                : crashRisk > 25
                  ? '#f59e0b'
                  : '#10b981'
              : undefined
          }
        />
      </div>

      {/* Horizon forecasts */}
      {horizonSignals && Object.keys(horizonSignals).length > 0 && (
        <div className="flex flex-col gap-1.5">
          <span className="label-micro">Horizons</span>
          <div className="flex flex-col gap-1">
            {Object.entries(horizonSignals)
              .slice(0, 4)
              .map(([key, h]) => {
                const hLabel = (h?.label || 'HOLD').toUpperCase();
                const hColor = signalColor(hLabel);
                const pUp = h?.p_up;
                return (
                  <div
                    key={key}
                    className="flex items-center justify-between"
                    style={{
                      padding: '4px 0',
                      borderBottom: '1px solid rgba(255,255,255,0.03)',
                    }}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className="w-1 h-1 rounded-full flex-shrink-0"
                        style={{ background: hColor }}
                      />
                      <span
                        className="tabular-nums"
                        style={{ fontSize: 10, color: 'var(--text-secondary)' }}
                      >
                        {key}
                      </span>
                    </div>
                    <div className="flex items-center gap-2">
                      <span
                        className="tabular-nums"
                        style={{ fontSize: 10, color: 'var(--text-muted)' }}
                      >
                        {pUp != null ? `${(pUp * 100).toFixed(0)}%` : ''}
                      </span>
                      <ArrowUpRight
                        className="w-2.5 h-2.5"
                        style={{
                          color: hColor,
                          transform:
                            hLabel.includes('SELL') || hLabel === 'EXIT'
                              ? 'rotate(90deg)'
                              : hLabel === 'HOLD'
                                ? 'rotate(45deg)'
                                : 'rotate(0deg)',
                        }}
                      />
                    </div>
                  </div>
                );
              })}
          </div>
        </div>
      )}
    </>
  );
}

function MiniMetric({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="flex flex-col gap-0.5">
      <span className="label-micro">{label}</span>
      <span
        className="num-display tabular-nums"
        style={{ fontSize: 14, color: color || '#e2e8f0' }}
      >
        {value}
      </span>
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* Segmented toggle (liquid-glass sliding indicator)                         */
/* ──────────────────────────────────────────────────────────────────────── */

function SegmentedToggle<T extends string>({
  value,
  onChange,
  options,
  dense,
}: {
  value: T;
  onChange: (v: T) => void;
  options: { value: T; label: string; icon?: React.ReactNode }[];
  dense?: boolean;
}) {
  const activeIndex = Math.max(0, options.findIndex((o) => o.value === value));
  const n = options.length;
  const pad = 2; // track inner padding in px
  // Indicator is sized against the track's content box: (100% - 2*pad) / n
  // and positioned from the padding edge so it aligns pixel-perfect with buttons.
  const indicatorWidth = `calc((100% - ${pad * 2}px) / ${n})`;
  const indicatorLeft = `calc(${pad}px + ${activeIndex} * (100% - ${pad * 2}px) / ${n})`;
  const minSegmentWidth = dense ? 44 : 112;

  return (
    <div
      role="tablist"
      className="relative flex items-center select-none"
      style={{
        height: dense ? 24 : 26,
        minWidth: minSegmentWidth * n + pad * 2,
        padding: pad,
        borderRadius: 999,
        background: 'rgba(255,255,255,0.03)',
        border: '1px solid rgba(255,255,255,0.05)',
        boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.02)',
      }}
    >
      {/* Sliding indicator — pixel-aligned to button track */}
      <div
        aria-hidden="true"
        style={{
          position: 'absolute',
          top: pad,
          bottom: pad,
          left: indicatorLeft,
          width: indicatorWidth,
          transition:
            'left 260ms cubic-bezier(0.22, 1, 0.36, 1), width 260ms cubic-bezier(0.22, 1, 0.36, 1)',
          background:
            'linear-gradient(135deg, rgba(139,92,246,0.30), rgba(6,182,212,0.22))',
          border: '1px solid rgba(139,92,246,0.32)',
          borderRadius: 999,
          boxShadow:
            '0 0 14px rgba(139,92,246,0.22), inset 0 1px 0 rgba(255,255,255,0.06)',
          pointerEvents: 'none',
          willChange: 'left, width',
        }}
      />
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            role="tab"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            className="relative flex items-center justify-center transition-colors duration-150"
            style={{
              flex: '1 1 0',
              minWidth: minSegmentWidth,
              height: '100%',
              padding: 0,
              fontSize: dense ? 9.5 : 10,
              letterSpacing: '0.1em',
              textTransform: 'uppercase',
              color: active ? '#ffffff' : 'var(--text-muted)',
              fontWeight: active ? 600 : 500,
              lineHeight: 1,
              borderRadius: 999,
              zIndex: 1,
              whiteSpace: 'nowrap',
            }}
          >
            {/* Inner wrapper supplies breathing room so the track's
                natural width scales with labels, while flex:1 1 0 keeps
                segments equal-width for pixel-aligned indicator math. */}
            <span
              className="flex items-center justify-center"
              style={{
                gap: 5,
                padding: dense ? '0 12px' : '0 14px',
                minWidth: 0,
              }}
            >
              {opt.icon && (
                <span
                  className="flex items-center justify-center"
                  style={{ width: 12, height: 12 }}
                >
                  {opt.icon}
                </span>
              )}
              <span style={{ transform: 'translateY(0.5px)', whiteSpace: 'nowrap' }}>{opt.label}</span>
            </span>
          </button>
        );
      })}
    </div>
  );
}

/* ──────────────────────────────────────────────────────────────────────── */
/* States                                                                   */
/* ──────────────────────────────────────────────────────────────────────── */

function ChartShimmer() {
  return (
    <div className="w-full h-full flex items-center justify-center">
      <div className="flex items-center gap-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
        <svg width="14" height="14" viewBox="0 0 14 14" className="animate-spin">
          <circle
            cx="7"
            cy="7"
            r="5"
            stroke="rgba(139,92,246,0.25)"
            strokeWidth="1.5"
            fill="none"
          />
          <path
            d="M 7 2 A 5 5 0 0 1 12 7"
            stroke="#8b5cf6"
            strokeWidth="1.5"
            fill="none"
            strokeLinecap="round"
          />
        </svg>
        <span className="label-micro">Loading price data</span>
      </div>
    </div>
  );
}

function ChartEmpty({ message }: { message: string }) {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center gap-2">
      <AlertTriangle className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
      <span className="label-micro">{message}</span>
    </div>
  );
}
