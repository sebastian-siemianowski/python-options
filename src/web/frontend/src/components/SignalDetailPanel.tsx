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
import { useQuery } from '@tanstack/react-query';
import {
  createChart,
  CandlestickSeries,
  LineSeries,
  AreaSeries,
  HistogramSeries,
  CrosshairMode,
  LineStyle,
  ColorType,
  type IChartApi,
  type ISeriesApi,
} from 'lightweight-charts';
import { ArrowUpRight, BarChart3, Activity, TrendingUp, AlertTriangle, ExternalLink } from 'lucide-react';
import { api, type OHLCVBar } from '../api';

// Cast OHLCVBar.time (string) to Time type lightweight-charts expects.
// 'YYYY-MM-DD' strings are accepted natively.


type ChartType = 'candles' | 'line' | 'area';
type RangeKey = '1M' | '3M' | '6M' | '1Y' | 'MAX';

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
  line: {
    color: '#8b5cf6',
    lineWidth: 2 as const,
    priceLineVisible: false,
    lastValueVisible: true,
  },
  area: {
    lineColor: '#8b5cf6',
    topColor: 'rgba(139,92,246,0.24)',
    bottomColor: 'rgba(139,92,246,0)',
    lineWidth: 2 as const,
    priceLineVisible: false,
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

export interface SignalDetailPanelProps {
  ticker: string;
  /** Optional — stats strip shows these if provided. */
  signal?: string;
  momentum?: number;
  crashRisk?: number;
  /** Horizons from the row (e.g. [7, 30, 90]) — rendered as a small forecast list. */
  horizonSignals?: Record<string, { p_up?: number; kelly_half?: number; label?: string }>;
  onNavigateChart: () => void;
}

export default function SignalDetailPanel({
  ticker,
  signal,
  momentum,
  crashRisk,
  horizonSignals,
  onNavigateChart,
}: SignalDetailPanelProps) {
  const [chartType, setChartType] = useState<ChartType>('candles');
  const [range, setRange] = useState<RangeKey>('3M');

  // Request a generous tail so range switches never require refetch.
  const { data, isLoading, error } = useQuery({
    queryKey: ['signalDetail', ticker],
    queryFn: () => api.chartOhlcv(ticker, 365),
    staleTime: 300_000,
  });

  const bars = useMemo<OHLCVBar[]>(() => data?.data ?? [], [data]);

  // Slice for the current range (keeps chart responsive without refetching).
  const visibleBars = useMemo(() => {
    if (range === 'MAX') return bars;
    const n = RANGE_DAYS[range];
    return bars.length > n ? bars.slice(-n) : bars;
  }, [bars, range]);

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
              { value: 'candles', label: 'Candles', icon: <BarChart3 className="w-3 h-3" /> },
              { value: 'line', label: 'Line', icon: <Activity className="w-3 h-3" /> },
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
          {isLoading ? (
            <ChartShimmer />
          ) : error || visibleBars.length < 2 ? (
            <ChartEmpty message={error ? 'Chart unavailable' : 'No data yet'} />
          ) : (
            <TradingViewChart bars={visibleBars} chartType={chartType} key={`${ticker}-${chartType}`} />
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

function TradingViewChart({ bars, chartType }: { bars: OHLCVBar[]; chartType: ChartType }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!containerRef.current) return;

    const container = containerRef.current;
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
    let priceSeries: ISeriesApi<'Candlestick'> | ISeriesApi<'Line'> | ISeriesApi<'Area'>;
    if (chartType === 'candles') {
      priceSeries = chart.addSeries(CandlestickSeries, TV_THEME.candles);
      priceSeries.setData(
        bars.map((b) => ({
          time: b.time,
          open: b.open,
          high: b.high,
          low: b.low,
          close: b.close,
        })),
      );
    } else if (chartType === 'line') {
      priceSeries = chart.addSeries(LineSeries, TV_THEME.line);
      priceSeries.setData(bars.map((b) => ({ time: b.time, value: b.close })));
    } else {
      priceSeries = chart.addSeries(AreaSeries, TV_THEME.area);
      priceSeries.setData(bars.map((b) => ({ time: b.time, value: b.close })));
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
        color: b.close >= b.open ? TV_THEME.volumeUp : TV_THEME.volumeDown,
      })),
    );

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
  }, [bars, chartType]);

  return <div ref={containerRef} style={{ width: '100%', height: '100%' }} />;
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

  return (
    <div
      role="tablist"
      className="relative flex items-center select-none"
      style={{
        height: dense ? 24 : 26,
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
              minWidth: 0,
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
              <span style={{ transform: 'translateY(0.5px)' }}>{opt.label}</span>
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
