import { useQuery } from '@tanstack/react-query';
import { useState, useEffect, useRef, useMemo } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { OHLCVBar, ChartSectorGroup } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries,
  type IChartApi,
} from 'lightweight-charts';
import { ChevronDown, ChevronRight, Search } from 'lucide-react';

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
              <FlatSymbolList
                symbols={strongSellSymbols.filter(s => !search || s.toLowerCase().includes(search.toLowerCase()))}
                selected={symbol}
                onSelect={selectSymbol}
                emptyText="No strong sell signals"
                accent="#FF1744"
              />
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
function ChartPanel({ symbol, strongBuy, strongSell }: { symbol: string; strongBuy: string[]; strongSell: string[] }) {
  const priceChartRef = useRef<HTMLDivElement>(null);
  const rsiChartRef = useRef<HTMLDivElement>(null);
  const priceChart = useRef<IChartApi | null>(null);
  const rsiChart = useRef<IChartApi | null>(null);
  const [showRSI, setShowRSI] = useState(true);
  const [showBollinger, setShowBollinger] = useState(true);
  const [showSMA, setShowSMA] = useState(true);

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

    // Indicators
    if (indQ.data?.indicators && showSMA) {
      const ind = indQ.data.indicators;
      if (ind.sma20?.length) chart.addSeries(LineSeries, { color: '#FFB300', lineWidth: 1 }).setData(ind.sma20);
      if (ind.sma50?.length) chart.addSeries(LineSeries, { color: '#42A5F5', lineWidth: 1 }).setData(ind.sma50);
      if (ind.sma200?.length) chart.addSeries(LineSeries, { color: '#AB47BC', lineWidth: 1 }).setData(ind.sma200);
    }

    if (indQ.data?.indicators?.bollinger && showBollinger) {
      const bb = indQ.data.indicators.bollinger;
      if (bb.upper?.length) chart.addSeries(LineSeries, { color: 'rgba(66,165,245,0.3)', lineWidth: 1 }).setData(bb.upper);
      if (bb.lower?.length) chart.addSeries(LineSeries, { color: 'rgba(66,165,245,0.3)', lineWidth: 1 }).setData(bb.lower);
    }

    chart.timeScale().fitContent();
    const handleResize = () => {
      if (priceChartRef.current) chart.applyOptions({ width: priceChartRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); chart.remove(); priceChart.current = null; };
  }, [ohlcvQ.data, indQ.data, showSMA, showBollinger, symbol, strongBuy, strongSell]);

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

        {/* Toggles */}
        <div className="flex gap-2 ml-auto">
          <ToggleBtn label="SMA" active={showSMA} onClick={() => setShowSMA(v => !v)} />
          <ToggleBtn label="BB" active={showBollinger} onClick={() => setShowBollinger(v => !v)} />
          <ToggleBtn label="RSI" active={showRSI} onClick={() => setShowRSI(v => !v)} />
        </div>
      </div>

      {/* Legend */}
      {showSMA && (
        <div className="flex gap-4 mb-2 text-[10px]">
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#FFB300] inline-block" />SMA20</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#42A5F5] inline-block" />SMA50</span>
          <span className="flex items-center gap-1"><span className="w-3 h-0.5 bg-[#AB47BC] inline-block" />SMA200</span>
        </div>
      )}

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
                  <p className="text-[10px] text-[#64748b]">{f.horizon_days}D</p>
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

/* ── Toggle Button ───────────────────────────────────────────────── */
function ToggleBtn({ label, active, onClick }: { label: string; active: boolean; onClick: () => void }) {
  return (
    <button
      onClick={onClick}
      className={`px-2 py-0.5 rounded text-[10px] font-medium transition ${
        active ? 'bg-[#42A5F5]/20 text-[#42A5F5]' : 'bg-[#1a1a2e] text-[#64748b]'
      }`}
    >
      {label}
    </button>
  );
}
