import { useQuery } from '@tanstack/react-query';
import { useState, useEffect, useRef, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { OHLCVBar, ChartSectorGroup } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries, AreaSeries,
  type IChartApi, LineStyle,
} from 'lightweight-charts';
import { ChevronDown, ChevronRight, Search } from 'lucide-react';
import { formatHorizon } from '../utils/horizons';

type PickerView = 'all' | 'sector' | 'strong_buy' | 'strong_sell';

export default function ChartsPage() {
  const { symbol: paramSymbol } = useParams();
  const navigate = useNavigate();
  const [symbol, setSymbol] = useState(paramSymbol || '');
  const [search, setSearch] = useState('');
  const [pickerView, setPickerView] = useState<PickerView>('sector');
  const [expandedSectors, setExpandedSectors] = useState<Set<string>>(new Set());

  const symbolsQ = useQuery({ queryKey: ['chartSymbols'], queryFn: api.chartSymbols });
  const sectorQ = useQuery({ queryKey: ['chartSymbolsBySector'], queryFn: api.chartSymbolsBySector });
  const strongQ = useQuery({ queryKey: ['strongSignals'], queryFn: api.strongSignals });

  // Set default symbol
  useEffect(() => {
    if (!symbol && symbolsQ.data?.symbols?.length) {
      const defaultSym = symbolsQ.data.symbols.find((s) => s === 'SPY') || symbolsQ.data.symbols[0];
      setSymbol(defaultSym);
    }
  }, [symbol, symbolsQ.data]);

  const symbols = symbolsQ.data?.symbols || [];
  const sectors = sectorQ.data?.sectors || [];
  const filtered = symbols.filter((s) => s.toLowerCase().includes(search.toLowerCase()));
  const strongBuySymbols = useMemo(() => (strongQ.data?.strong_buy || []).map(e => e.symbol), [strongQ.data]);
  const strongSellSymbols = useMemo(() => (strongQ.data?.strong_sell || []).map(e => e.symbol), [strongQ.data]);

  const toggleSector = (name: string) => {
    setExpandedSectors(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  };

  const selectSymbol = (s: string) => {
    setSymbol(s);
    navigate(`/charts/${s}`);
  };

  return (
    <>
      <PageHeader title="Interactive Charts">
        {symbols.length} symbols available for charting
      </PageHeader>

      <div className="flex gap-4">
        {/* Symbol picker */}
        <div className="w-56 flex-shrink-0">
          {/* Search */}
          <div className="flex items-center gap-1.5 glass-card px-2.5 py-1.5 mb-2">
            <Search className="w-3 h-3 text-[#64748b]" />
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="Search symbol..."
              className="bg-transparent text-sm text-[#e2e8f0] placeholder:text-[#64748b] outline-none w-full"
            />
          </div>

          {/* View tabs */}
          <div className="flex gap-0.5 mb-2 glass-card px-1 py-0.5">
            {([
              { key: 'sector' as PickerView, label: 'Sectors' },
              { key: 'all' as PickerView, label: 'All' },
              { key: 'strong_buy' as PickerView, label: '\u25B2\u25B2' },
              { key: 'strong_sell' as PickerView, label: '\u25BC\u25BC' },
            ]).map(({ key, label }) => (
              <button
                key={key}
                onClick={() => setPickerView(key)}
                className={`flex-1 px-1.5 py-1 rounded text-[10px] font-medium transition ${
                  pickerView === key ? 'bg-[#42A5F5]/20 text-[#42A5F5]' : 'text-[#64748b] hover:text-[#94a3b8]'
                }`}
              >
                {label}
              </button>
            ))}
          </div>

          {/* Symbol list */}
          <div className="glass-card overflow-y-auto max-h-[calc(100vh-260px)]">
            {pickerView === 'all' && (
              <FlatSymbolList symbols={filtered.slice(0, 100)} selected={symbol} onSelect={selectSymbol} />
            )}
            {pickerView === 'sector' && (
              <SectorSymbolList
                sectors={sectors}
                search={search}
                selected={symbol}
                onSelect={selectSymbol}
                expandedSectors={expandedSectors}
                toggleSector={toggleSector}
              />
            )}
            {pickerView === 'strong_buy' && (
              <FlatSymbolList
                symbols={strongBuySymbols.filter(s => !search || s.toLowerCase().includes(search.toLowerCase()))}
                selected={symbol}
                onSelect={selectSymbol}
                emptyText="No strong buy signals"
                accent="#00E676"
              />
            )}
            {pickerView === 'strong_sell' && (
              strongSellSymbols.length === 0 ? (
                <div className="px-3 py-6 text-center">
                  <p className="text-xs text-[#64748b] mb-1">No strong sell signals</p>
                  <p className="text-[10px] text-[#475569]">
                    No assets currently meet the strong sell threshold.
                    This is normal in bullish market conditions.
                  </p>
                </div>
              ) : (
                <FlatSymbolList
                  symbols={strongSellSymbols.filter(s => !search || s.toLowerCase().includes(search.toLowerCase()))}
                  selected={symbol}
                  onSelect={selectSymbol}
                  emptyText="No matching symbols"
                  accent="#FF1744"
                />
              )
            )}
          </div>
        </div>

        {/* Chart area */}
        <div className="flex-1 min-w-0">
          {symbol ? (
            <ChartPanel symbol={symbol} strongBuy={strongBuySymbols} strongSell={strongSellSymbols} />
          ) : (
            <div className="glass-card p-8 text-center text-[#64748b]">
              Select a symbol to view its chart
            </div>
          )}
        </div>
      </div>
    </>
  );
}

/* ── Flat symbol list ────────────────────────────────────────────── */
function FlatSymbolList({
  symbols, selected, onSelect, emptyText, accent,
}: {
  symbols: string[]; selected: string; onSelect: (s: string) => void;
  emptyText?: string; accent?: string;
}) {
  if (symbols.length === 0) {
    return <p className="px-3 py-4 text-xs text-[#64748b] text-center">{emptyText || 'No symbols'}</p>;
  }
  return (
    <>
      {symbols.map((s) => (
        <button
          key={s}
          onClick={() => onSelect(s)}
          className={`w-full text-left px-3 py-1.5 text-xs font-medium transition hover:bg-[#16213e] ${
            s === selected ? 'bg-[#16213e]' : ''
          }`}
          style={{ color: s === selected ? (accent || '#42A5F5') : '#94a3b8' }}
        >
          {s}
        </button>
      ))}
    </>
  );
}

/* ── Sector-grouped symbol list ──────────────────────────────────── */
function SectorSymbolList({
  sectors, search, selected, onSelect, expandedSectors, toggleSector,
}: {
  sectors: ChartSectorGroup[]; search: string; selected: string;
  onSelect: (s: string) => void;
  expandedSectors: Set<string>; toggleSector: (n: string) => void;
}) {
  const sorted = useMemo(() => [...sectors].sort((a, b) => b.count - a.count), [sectors]);

  return (
    <>
      {sorted.map((sec) => {
        const syms = search
          ? sec.symbols.filter(s => s.toLowerCase().includes(search.toLowerCase()))
          : sec.symbols;
        if (search && syms.length === 0) return null;
        const expanded = expandedSectors.has(sec.name);
        return (
          <div key={sec.name}>
            <button
              onClick={() => toggleSector(sec.name)}
              className="w-full flex items-center gap-1.5 px-2 py-1.5 hover:bg-[#16213e]/30 transition"
            >
              {expanded
                ? <ChevronDown className="w-3 h-3 text-[#64748b]" />
                : <ChevronRight className="w-3 h-3 text-[#64748b]" />}
              <span className="text-[10px] font-medium text-[#94a3b8] flex-1 text-left truncate">{sec.name}</span>
              <span className="text-[9px] text-[#64748b]">{syms.length}</span>
            </button>
            {expanded && syms.map((s) => (
              <button
                key={s}
                onClick={() => onSelect(s)}
                className={`w-full text-left pl-7 pr-3 py-1 text-xs transition hover:bg-[#16213e] ${
                  s === selected ? 'bg-[#16213e] text-[#42A5F5]' : 'text-[#94a3b8]'
                }`}
              >
                {s}
              </button>
            ))}
          </div>
        );
      })}
    </>
  );
}

/* ── Chart Panel ─────────────────────────────────────────────────── */
type OverlayKey = 'sma20' | 'sma50' | 'sma200' | 'bb' | 'rsi' | 'forecastMedian' | 'ciUpper' | 'ciLower';

const OVERLAY_DEFS: { key: OverlayKey; label: string; color: string; group: string; shortcut: string }[] = [
  { key: 'sma20',          label: 'SMA 20',          color: '#FFB300', group: 'Moving Averages', shortcut: '1' },
  { key: 'sma50',          label: 'SMA 50',          color: '#42A5F5', group: 'Moving Averages', shortcut: '2' },
  { key: 'sma200',         label: 'SMA 200',         color: '#AB47BC', group: 'Moving Averages', shortcut: '3' },
  { key: 'bb',             label: 'Bollinger Bands',  color: 'rgba(66,165,245,0.6)', group: 'Overlays', shortcut: '4' },
  { key: 'rsi',            label: 'RSI (14)',         color: '#42A5F5', group: 'Overlays', shortcut: '5' },
  { key: 'forecastMedian', label: 'Forecast Median',  color: '#2196F3', group: 'Forecast', shortcut: '6' },
  { key: 'ciUpper',        label: 'CI Upper',         color: '#26A69A', group: 'Forecast', shortcut: '7' },
  { key: 'ciLower',        label: 'CI Lower',         color: '#EF5350', group: 'Forecast', shortcut: '8' },
];

const DEFAULT_OVERLAYS: Record<OverlayKey, boolean> = {
  sma20: true, sma50: true, sma200: true, bb: true, rsi: true,
  forecastMedian: true, ciUpper: true, ciLower: true,
};

function ChartPanel({ symbol, strongBuy, strongSell }: { symbol: string; strongBuy: string[]; strongSell: string[] }) {
  const priceChartRef = useRef<HTMLDivElement>(null);
  const rsiChartRef = useRef<HTMLDivElement>(null);
  const priceChart = useRef<IChartApi | null>(null);
  const rsiChart = useRef<IChartApi | null>(null);
  const [overlays, setOverlays] = useState<Record<OverlayKey, boolean>>(DEFAULT_OVERLAYS);
  const [panelOpen, setPanelOpen] = useState(false);

  const toggle = (key: OverlayKey) => setOverlays(prev => ({ ...prev, [key]: !prev[key] }));
  const allOn = () => setOverlays(DEFAULT_OVERLAYS);
  const allOff = () => setOverlays(Object.fromEntries(Object.keys(DEFAULT_OVERLAYS).map(k => [k, false])) as Record<OverlayKey, boolean>);
  const allActive = Object.values(overlays).every(v => v);
  const noneActive = Object.values(overlays).every(v => !v);
  const activeCount = Object.values(overlays).filter(v => v).length;

  // Keyboard shortcuts: digits 1-8 toggle overlays, 0 = all off, 9 = all on, L = panel
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      const def = OVERLAY_DEFS.find(d => d.shortcut === e.key);
      if (def) { e.preventDefault(); toggle(def.key); }
      if (e.key === '0') { e.preventDefault(); allOff(); }
      if (e.key === '9') { e.preventDefault(); allOn(); }
      if (e.key === 'l' || e.key === 'L') { e.preventDefault(); setPanelOpen(v => !v); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  // Compat shims for the chart rendering
  const showSMA20 = overlays.sma20;
  const showSMA50 = overlays.sma50;
  const showSMA200 = overlays.sma200;
  const showBollinger = overlays.bb;
  const showRSI = overlays.rsi;
  const showForecastMedian = overlays.forecastMedian;
  const showCIUpper = overlays.ciUpper;
  const showCILower = overlays.ciLower;

  const ohlcvQ = useQuery({
    queryKey: ['ohlcv', symbol],
    queryFn: () => api.chartOhlcv(symbol, 365),
    enabled: !!symbol,
  });

  const indQ = useQuery({
    queryKey: ['indicators', symbol],
    queryFn: () => api.chartIndicators(symbol, 365),
    enabled: !!symbol,
  });

  const forecastQ = useQuery({
    queryKey: ['forecast', symbol],
    queryFn: () => api.chartForecast(symbol),
    enabled: !!symbol,
  });

  // Price chart
  useEffect(() => {
    if (!priceChartRef.current || !ohlcvQ.data?.data?.length) return;
    if (priceChart.current) { priceChart.current.remove(); priceChart.current = null; }

    const chart = createChart(priceChartRef.current, {
      layout: { background: { type: ColorType.Solid, color: '#1a1a2e' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#2a2a4a' }, horzLines: { color: '#2a2a4a' } },
      width: priceChartRef.current.clientWidth,
      height: 420,
      crosshair: {
        vertLine: { color: '#42A5F5', width: 1, style: 2 },
        horzLine: { color: '#42A5F5', width: 1, style: 2 },
      },
    });
    priceChart.current = chart;

    // Candlestick
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#00E676', downColor: '#FF1744',
      borderUpColor: '#00E676', borderDownColor: '#FF1744',
      wickUpColor: '#00E676', wickDownColor: '#FF1744',
    });
    candleSeries.setData(ohlcvQ.data.data);

    // Signal markers on candlestick (v5: markers on chart, not series)
    const isSB = strongBuy.includes(symbol);
    const isSS = strongSell.includes(symbol);
    if (isSB || isSS) {
      const lastBar = ohlcvQ.data.data[ohlcvQ.data.data.length - 1];
      if (lastBar) {
        try {
          (candleSeries as any).setMarkers([{
            time: lastBar.time,
            position: isSB ? 'belowBar' : 'aboveBar',
            color: isSB ? '#00E676' : '#FF1744',
            shape: isSB ? 'arrowUp' : 'arrowDown',
            text: isSB ? 'STRONG BUY' : 'STRONG SELL',
          }]);
        } catch {
          // markers not supported in this version
        }
      }
    }

    // Volume
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });
    volumeSeries.setData(
      ohlcvQ.data.data.map((d: OHLCVBar) => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(0,230,118,0.25)' : 'rgba(255,23,68,0.25)',
      }))
    );

    // Indicators — individual SMA controls
    if (indQ.data?.indicators) {
      const ind = indQ.data.indicators;
      if (showSMA20 && ind.sma20?.length) chart.addSeries(LineSeries, { color: '#FFB300', lineWidth: 1 }).setData(ind.sma20);
      if (showSMA50 && ind.sma50?.length) chart.addSeries(LineSeries, { color: '#42A5F5', lineWidth: 1 }).setData(ind.sma50);
      if (showSMA200 && ind.sma200?.length) chart.addSeries(LineSeries, { color: '#AB47BC', lineWidth: 1 }).setData(ind.sma200);
    }

    if (indQ.data?.indicators?.bollinger && showBollinger) {
      const bb = indQ.data.indicators.bollinger;
      if (bb.upper?.length) chart.addSeries(LineSeries, { color: 'rgba(66,165,245,0.3)', lineWidth: 1 }).setData(bb.upper);
      if (bb.lower?.length) chart.addSeries(LineSeries, { color: 'rgba(66,165,245,0.3)', lineWidth: 1 }).setData(bb.lower);
    }

    // Story 6.5: Forecast fan chart overlay — individual CI/median controls
    const showAnyForecast = showForecastMedian || showCIUpper || showCILower;
    if (showAnyForecast && forecastQ.data?.forecasts?.length && ohlcvQ.data.data.length > 0) {
      const lastCandle = ohlcvQ.data.data[ohlcvQ.data.data.length - 1];
      const lastPrice = lastCandle.close;
      const lastDate = new Date(lastCandle.time as string);

      const upperData: { time: string; value: number }[] = [{ time: lastCandle.time as string, value: lastPrice }];
      const lowerData: { time: string; value: number }[] = [{ time: lastCandle.time as string, value: lastPrice }];
      const medianData: { time: string; value: number }[] = [{ time: lastCandle.time as string, value: lastPrice }];

      for (const f of forecastQ.data.forecasts) {
        const futureDate = new Date(lastDate);
        futureDate.setDate(futureDate.getDate() + f.horizon_days);
        const dateStr = futureDate.toISOString().slice(0, 10);
        const retPct = f.expected_return_pct / 100;
        const medianPrice = lastPrice * (1 + retPct);
        // CI width grows with sqrt(horizon)
        const ciWidth = lastPrice * Math.abs(retPct) * 0.5 + lastPrice * 0.01 * Math.sqrt(f.horizon_days);
        upperData.push({ time: dateStr, value: medianPrice + ciWidth });
        lowerData.push({ time: dateStr, value: Math.max(0, medianPrice - ciWidth) });
        medianData.push({ time: dateStr, value: medianPrice });
      }

      // Upper CI band (green tint)
      if (showCIUpper) {
        const upperSeries = chart.addSeries(AreaSeries, {
          topColor: 'rgba(38, 166, 154, 0.25)',
          bottomColor: 'rgba(38, 166, 154, 0.0)',
          lineColor: 'rgba(38, 166, 154, 0.4)',
          lineWidth: 1,
          priceScaleId: 'right',
        });
        upperSeries.setData(upperData);
      }

      // Lower CI band (red tint)
      if (showCILower) {
        const lowerSeries = chart.addSeries(AreaSeries, {
          topColor: 'rgba(239, 83, 80, 0.0)',
          bottomColor: 'rgba(239, 83, 80, 0.25)',
          lineColor: 'rgba(239, 83, 80, 0.4)',
          lineWidth: 1,
          priceScaleId: 'right',
        });
        lowerSeries.setData(lowerData);
      }

      // Median forecast line (blue dashed)
      if (showForecastMedian) {
        const forecastLine = chart.addSeries(LineSeries, {
          color: '#2196F3',
          lineWidth: 2,
          lineStyle: LineStyle.Dashed,
          priceScaleId: 'right',
        });
        forecastLine.setData(medianData);
      }
    }

    chart.timeScale().fitContent();
    const handleResize = () => {
      if (priceChartRef.current) chart.applyOptions({ width: priceChartRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); chart.remove(); priceChart.current = null; };
  }, [ohlcvQ.data, indQ.data, forecastQ.data, showSMA20, showSMA50, showSMA200, showBollinger, showForecastMedian, showCIUpper, showCILower, symbol, strongBuy, strongSell]);

  // RSI chart
  useEffect(() => {
    if (!rsiChartRef.current || !showRSI || !indQ.data?.indicators?.rsi?.length) return;
    if (rsiChart.current) { rsiChart.current.remove(); rsiChart.current = null; }

    const chart = createChart(rsiChartRef.current, {
      layout: { background: { type: ColorType.Solid, color: '#1a1a2e' }, textColor: '#94a3b8' },
      grid: { vertLines: { color: '#2a2a4a' }, horzLines: { color: '#2a2a4a' } },
      width: rsiChartRef.current.clientWidth,
      height: 120,
      rightPriceScale: { scaleMargins: { top: 0.1, bottom: 0.1 } },
    });
    rsiChart.current = chart;

    // RSI line
    const rsiSeries = chart.addSeries(LineSeries, { color: '#42A5F5', lineWidth: 2 });
    rsiSeries.setData(indQ.data.indicators.rsi);

    // Overbought / oversold lines
    const rsiData = indQ.data.indicators.rsi;
    if (rsiData.length >= 2) {
      const times = rsiData.map(d => d.time);
      const overbought = chart.addSeries(LineSeries, { color: 'rgba(255,23,68,0.4)', lineWidth: 1, lineStyle: 2 });
      overbought.setData(times.map(t => ({ time: t, value: 70 })));
      const oversold = chart.addSeries(LineSeries, { color: 'rgba(0,230,118,0.4)', lineWidth: 1, lineStyle: 2 });
      oversold.setData(times.map(t => ({ time: t, value: 30 })));
    }

    chart.timeScale().fitContent();

    // Sync time scales
    if (priceChart.current) {
      priceChart.current.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range && rsiChart.current) rsiChart.current.timeScale().setVisibleLogicalRange(range);
      });
    }

    const handleResize = () => {
      if (rsiChartRef.current) chart.applyOptions({ width: rsiChartRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); chart.remove(); rsiChart.current = null; };
  }, [indQ.data, showRSI]);

  if (ohlcvQ.isLoading) return <LoadingSpinner text={`Loading ${symbol}...`} />;
  if (ohlcvQ.error) return <div className="text-[#FF1744]">Failed to load chart for {symbol}</div>;

  const bars = ohlcvQ.data?.data;
  const lastBar = bars?.[bars.length - 1];
  const prevBar = bars?.[bars.length - 2];
  const change = lastBar && prevBar ? ((lastBar.close - prevBar.close) / prevBar.close * 100) : 0;
  const isSB = strongBuy.includes(symbol);
  const isSS = strongSell.includes(symbol);

  return (
    <div>
      {/* Header */}
      <div className="flex items-center gap-3 mb-3">
        <h2 className="text-xl font-bold text-[#e2e8f0]">{symbol}</h2>
        {lastBar && (
          <>
            <span className="text-lg font-medium text-[#e2e8f0]">${lastBar.close.toFixed(2)}</span>
            <span className={`text-sm font-medium ${change >= 0 ? 'text-[#00E676]' : 'text-[#FF1744]'}`}>
              {change >= 0 ? '+' : ''}{change.toFixed(2)}%
            </span>
          </>
        )}
        {isSB && (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-[#00E676]/20 text-[#00E676]">
            {'\u25B2\u25B2'} STRONG BUY
          </span>
        )}
        {isSS && (
          <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-[#FF1744]/20 text-[#FF1744]">
            {'\u25BC\u25BC'} STRONG SELL
          </span>
        )}

        {/* Overlay panel toggle */}
        <div className="flex gap-2 ml-auto items-center">
          <div className="flex gap-1">
            {OVERLAY_DEFS.filter(d => overlays[d.key]).map(d => (
              <span
                key={d.key}
                className="w-2 h-2 rounded-full animate-pulse"
                style={{ backgroundColor: d.color }}
                title={d.label}
              />
            ))}
          </div>
          <button
            onClick={() => setPanelOpen(v => !v)}
            className={`px-2.5 py-1 rounded-md text-[10px] font-medium transition-all duration-200 flex items-center gap-1.5 ${
              panelOpen
                ? 'bg-[#42A5F5]/25 text-[#42A5F5] ring-1 ring-[#42A5F5]/30'
                : 'bg-[#1a1a2e] text-[#94a3b8] hover:bg-[#1e2844] hover:text-[#e2e8f0]'
            }`}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="12" cy="12" r="3" /><path d="M12 1v6m0 6v6m8.66-13-5.2 3m-6.92 4-5.2 3M22.66 17l-5.2-3m-6.92-4-5.2-3" />
            </svg>
            Layers ({activeCount}/{OVERLAY_DEFS.length})
          </button>
        </div>
      </div>

      {/* Overlay control panel */}
      <div
        className={`overflow-hidden transition-all duration-300 ease-out ${
          panelOpen ? 'max-h-[400px] opacity-100 mb-3' : 'max-h-0 opacity-0'
        }`}
      >
        <div className="glass-card p-3">
          {/* Quick actions */}
          <div className="flex items-center justify-between mb-3">
            <span className="text-[10px] text-[#64748b] uppercase tracking-wider font-medium">Chart Layers</span>
            <div className="flex gap-1.5">
              <button
                onClick={allOn}
                disabled={allActive}
                className="px-2 py-0.5 rounded text-[9px] font-medium transition-all
                  bg-[#00E676]/10 text-[#00E676] hover:bg-[#00E676]/20
                  disabled:opacity-30 disabled:cursor-default"
              >
                All On
              </button>
              <button
                onClick={allOff}
                disabled={noneActive}
                className="px-2 py-0.5 rounded text-[9px] font-medium transition-all
                  bg-[#FF1744]/10 text-[#FF1744] hover:bg-[#FF1744]/20
                  disabled:opacity-30 disabled:cursor-default"
              >
                All Off
              </button>
            </div>
          </div>

          {/* Grouped overlays */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
            {['Moving Averages', 'Overlays', 'Forecast'].map(group => (
              <div key={group}>
                <p className="text-[9px] text-[#475569] uppercase tracking-wider mb-1.5 font-medium">{group}</p>
                <div className="flex flex-col gap-1">
                  {OVERLAY_DEFS.filter(d => d.group === group).map(d => (
                    <button
                      key={d.key}
                      onClick={() => toggle(d.key)}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded-md text-[11px] font-medium transition-all duration-200 group ${
                        overlays[d.key]
                          ? 'bg-[#16213e] ring-1'
                          : 'bg-transparent hover:bg-[#16213e]/50'
                      }`}
                      style={{
                        color: overlays[d.key] ? d.color : '#64748b',
                        ringColor: overlays[d.key] ? `${d.color}33` : 'transparent',
                        borderColor: overlays[d.key] ? `${d.color}33` : 'transparent',
                        border: overlays[d.key] ? `1px solid ${d.color}33` : '1px solid transparent',
                      }}
                    >
                      {/* Color indicator with on/off animation */}
                      <span
                        className={`w-2.5 h-2.5 rounded-full transition-all duration-300 flex-shrink-0 ${
                          overlays[d.key] ? 'scale-100' : 'scale-75 opacity-40'
                        }`}
                        style={{ backgroundColor: d.color }}
                      />
                      <span className="flex-1 text-left">{d.label}</span>
                      {/* Shortcut hint */}
                      <kbd className="text-[8px] px-1 py-0.5 rounded bg-[#0f0f23] text-[#475569] font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                        {d.shortcut}
                      </kbd>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="text-[8px] text-[#3a3a5a] mt-2 text-center">
            Press 1-8 to toggle layers  |  9 all on  |  0 all off  |  L toggle panel
          </p>
        </div>
      </div>

      {/* Price chart */}
      <div className="glass-card p-2">
        <div ref={priceChartRef} />
      </div>

      {/* RSI sub-panel */}
      {showRSI && indQ.data?.indicators?.rsi?.length && (
        <div className="glass-card p-2 mt-1">
          <div className="flex items-center gap-2 px-2 mb-1">
            <span className="text-[10px] text-[#94a3b8] font-medium">RSI (14)</span>
            <span className="text-[9px] text-[#64748b]">
              {indQ.data.indicators.rsi.length > 0 && (
                <>Current: {indQ.data.indicators.rsi[indQ.data.indicators.rsi.length - 1].value.toFixed(1)}</>
              )}
            </span>
            <div className="flex gap-3 ml-auto text-[9px]">
              <span className="text-[#FF1744]/60">Overbought 70</span>
              <span className="text-[#00E676]/60">Oversold 30</span>
            </div>
          </div>
          <div ref={rsiChartRef} />
        </div>
      )}

      {/* Forecast */}
      {forecastQ.data?.forecasts?.length ? (
        <div className="glass-card p-4 mt-4">
          <h3 className="text-sm font-medium text-[#94a3b8] mb-3">Multi-Horizon Forecast</h3>
          <div className="flex gap-3 flex-wrap">
            {forecastQ.data.forecasts.map((f) => {
              const color = f.signal_label === 'BUY' || f.signal_label === 'STRONG BUY' ? '#00E676'
                : f.signal_label === 'SELL' || f.signal_label === 'STRONG SELL' ? '#FF1744' : '#64748b';
              return (
                <div key={f.horizon_days} className="bg-[#0f0f23] rounded-lg px-3 py-2 text-center min-w-[80px]">
                  <p className="text-[10px] text-[#64748b]">{formatHorizon(f.horizon_days)}</p>
                  <p className="text-sm font-bold" style={{ color }}>
                    {f.expected_return_pct >= 0 ? '+' : ''}{f.expected_return_pct.toFixed(1)}%
                  </p>
                  <p className="text-[10px]" style={{ color }}>
                    {f.signal_label} {'\u2022'} {(f.probability_up * 100).toFixed(0)}%
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}
    </div>
  );
}
