/* eslint-disable @typescript-eslint/no-explicit-any */
import { useQuery } from '@tanstack/react-query';
import { useState, useEffect, useRef, useMemo, useCallback } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { api } from '../api';
import type { OHLCVBar, ChartSectorGroup, SummaryRow } from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import {
  createChart, ColorType, CandlestickSeries, LineSeries, HistogramSeries, AreaSeries,
  type IChartApi, LineStyle,
} from 'lightweight-charts';
import { ChevronDown, ChevronRight, Search, X } from 'lucide-react';
import { formatHorizon } from '../utils/horizons';
import BuySellZoneCharts from '../components/BuySellZoneCharts';

/* ═══════════════════════════════════════════════════════════════════
   TYPES & CONFIG
   ═══════════════════════════════════════════════════════════════════ */

/** Extract ticker symbol from "Company Name (TICKER)" format */
const extractTicker = (label: string): string => {
  if (label.includes('(')) return label.split('(').pop()!.replace(')', '').trim();
  return label;
};

/** Extract max half_kelly from kelly horizon array */
const getMaxHalfKelly = (r: SummaryRow): number => {
  if (!Array.isArray(r.kelly) || r.kelly.length === 0) return 0;
  return Math.max(...r.kelly.map(k => k.half_kelly ?? 0));
};

/** Extract max edge from kelly horizon array */
const getMaxEdge = (r: SummaryRow): number => {
  if (!Array.isArray(r.kelly) || r.kelly.length === 0) return 0;
  return Math.max(...r.kelly.map(k => k.edge ?? 0));
};
type PickerView = 'all' | 'sector' | 'strong_buy' | 'strong_sell' | 'filter';
type FilterMode = 'momentum' | 'edge' | 'exp_return' | 'low_risk' | 'kelly' | 'p_up' | 'forecast_up' | 'forecast_down';
type TimeRange = '1M' | '3M' | '6M' | '1Y' | 'ALL';
type OverlayKey = 'sma20' | 'sma50' | 'sma200' | 'bb' | 'forecastMedian' | 'ciUpper' | 'ciLower' | 'priceLine';

const TIME_RANGES: { key: TimeRange; label: string; days: number }[] = [
  { key: '1M', label: '1M', days: 30 },
  { key: '3M', label: '3M', days: 90 },
  { key: '6M', label: '6M', days: 180 },
  { key: '1Y', label: '1Y', days: 365 },
  { key: 'ALL', label: 'ALL', days: 9999 },
];

const OVERLAY_DEFS: { key: OverlayKey; label: string; color: string; group: string; shortcut: string }[] = [
  { key: 'sma20',          label: 'SMA 20',          color: '#f5c542', group: 'Moving Averages', shortcut: '1' },
  { key: 'sma50',          label: 'SMA 50',          color: '#818cf8', group: 'Moving Averages', shortcut: '2' },
  { key: 'sma200',         label: 'SMA 200',         color: '#c084fc', group: 'Moving Averages', shortcut: '3' },
  { key: 'bb',             label: 'Bollinger',        color: 'rgba(139,92,246,0.6)', group: 'Volatility', shortcut: '4' },
  { key: 'forecastMedian', label: 'Forecast',         color: '#b49aff', group: 'Forecast', shortcut: '5' },
  { key: 'ciUpper',        label: 'CI Upper',         color: '#3ee8a5', group: 'Forecast', shortcut: '6' },
  { key: 'ciLower',        label: 'CI Lower',         color: '#ff6b8a', group: 'Forecast', shortcut: '7' },
  { key: 'priceLine',      label: 'Price Line',       color: '#e2e8f0', group: 'Overlays', shortcut: 'p' },
];

const DEFAULT_OVERLAYS: Record<OverlayKey, boolean> = {
  sma20: true, sma50: true, sma200: true, bb: true,
  forecastMedian: true, ciUpper: true, ciLower: true, priceLine: true,
};

/* ── Sub-chart indicator definitions ──────────────────────── */
type SubIndicatorKey = 'composite' | 'rsi' | 'macd' | 'stochastic' | 'adx' | 'atr' | 'obv' | 'cci' | 'mfi' | 'cmf' | 'roc' | 'bbpctb';

interface SubIndicatorDef {
  key: SubIndicatorKey;
  label: string;
  color: string;
  group: string;
}

const SUB_INDICATOR_DEFS: SubIndicatorDef[] = [
  { key: 'composite',  label: 'Composite Signal', color: '#facc15', group: 'Signal' },
  { key: 'rsi',        label: 'RSI (14)',       color: '#b49aff', group: 'Momentum' },
  { key: 'macd',       label: 'MACD',          color: '#3ee8a5', group: 'Momentum' },
  { key: 'stochastic', label: 'Stochastic',    color: '#f5c542', group: 'Momentum' },
  { key: 'roc',        label: 'ROC (12)',       color: '#ff9f43', group: 'Momentum' },
  { key: 'adx',        label: 'ADX (14)',       color: '#26A69A', group: 'Trend' },
  { key: 'cci',        label: 'CCI (20)',       color: '#b49aff', group: 'Trend' },
  { key: 'bbpctb',     label: 'BB %B',         color: '#818cf8', group: 'Volatility' },
  { key: 'atr',        label: 'ATR (14)',       color: '#c084fc', group: 'Volatility' },
  { key: 'obv',        label: 'OBV',            color: '#64b5f6', group: 'Volume' },
  { key: 'mfi',        label: 'MFI (14)',       color: '#4dd0e1', group: 'Volume' },
  { key: 'cmf',        label: 'CMF (20)',       color: '#7986cb', group: 'Volume' },
];

const DEFAULT_SUB_INDICATORS: Record<SubIndicatorKey, boolean> = {
  composite: false, rsi: true, macd: false, stochastic: false, adx: false, atr: false, obv: false,
  cci: false, mfi: false, cmf: false, roc: false, bbpctb: false,
};

/* ═══════════════════════════════════════════════════════════════════
   MAIN PAGE
   ═══════════════════════════════════════════════════════════════════ */
export default function ChartsPage() {
  const { symbol: paramSymbol } = useParams();
  const navigate = useNavigate();
  const [symbol, setSymbol] = useState(paramSymbol || '');
  const [search, setSearch] = useState('');
  const [pickerView, setPickerView] = useState<PickerView>('sector');
  const [expandedSectors, setExpandedSectors] = useState<Set<string>>(new Set());
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [filterMode, setFilterMode] = useState<FilterMode>('momentum');
  const searchRef = useRef<HTMLInputElement>(null);
  const sidebarScrollRef = useRef<HTMLDivElement>(null);

  const symbolsQ = useQuery({ queryKey: ['chartSymbols'], queryFn: api.chartSymbols });
  const sectorQ = useQuery({ queryKey: ['chartSymbolsBySector'], queryFn: api.chartSymbolsBySector });
  const strongQ = useQuery({ queryKey: ['strongSignals'], queryFn: api.strongSignals });
  const summaryQ = useQuery({ queryKey: ['signalSummary'], queryFn: api.signalSummary, staleTime: 60_000 });

  useEffect(() => {
    if (!symbol && symbolsQ.data?.symbols?.length) {
      const defaultSym = symbolsQ.data.symbols.find((s) => s === 'SPY') || symbolsQ.data.symbols[0];
      setSymbol(defaultSym);
    }
  }, [symbol, symbolsQ.data]);

  // Global keyboard: Cmd+K focuses search
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  const symbols = symbolsQ.data?.symbols || [];
  const sectors = sectorQ.data?.sectors || [];
  const filtered = symbols.filter((s) => s.toLowerCase().includes(search.toLowerCase()));
  const strongBuySymbols = useMemo(() => (strongQ.data?.strong_buy || []).map(e => e.symbol), [strongQ.data]);
  const strongSellSymbols = useMemo(() => (strongQ.data?.strong_sell || []).map(e => e.symbol), [strongQ.data]);

  // Ranked filter lists derived from signal summary
  const rankedSymbols = useMemo(() => {
    const rows = summaryQ.data?.summary_rows || [];
    if (rows.length === 0) return [];
    const getBestExpRet = (r: SummaryRow): number => {
      const sigs = Object.values(r.horizon_signals || {});
      return sigs.length ? Math.max(...sigs.map(s => s.exp_ret ?? 0)) : 0;
    };
    const getBestPUp = (r: SummaryRow): number => {
      const sigs = Object.values(r.horizon_signals || {});
      return sigs.length ? Math.max(...sigs.map(s => s.p_up ?? 0)) : 0;
    };
    const sorted = [...rows];
    switch (filterMode) {
      case 'momentum':    sorted.sort((a, b) => (b.momentum_score ?? 0) - (a.momentum_score ?? 0)); break;
      case 'edge':        sorted.sort((a, b) => getMaxEdge(b) - getMaxEdge(a)); break;
      case 'exp_return':  sorted.sort((a, b) => getBestExpRet(b) - getBestExpRet(a)); break;
      case 'low_risk':    sorted.sort((a, b) => (a.crash_risk_score ?? 1) - (b.crash_risk_score ?? 1)); break;
      case 'kelly':       sorted.sort((a, b) => getMaxHalfKelly(b) - getMaxHalfKelly(a)); break;
      case 'p_up':        sorted.sort((a, b) => getBestPUp(b) - getBestPUp(a)); break;
      case 'forecast_up':  sorted.sort((a, b) => getBestExpRet(b) - getBestExpRet(a)); break;
      case 'forecast_down': sorted.sort((a, b) => getBestExpRet(a) - getBestExpRet(b)); break;
    }
    return sorted.slice(0, 30);
  }, [summaryQ.data, filterMode]);

  const toggleSector = (name: string) => {
    setExpandedSectors(prev => {
      const next = new Set(prev);
      if (next.has(name)) next.delete(name);
      else next.add(name);
      return next;
    });
  };

  const selectSymbol = useCallback((s: string) => {
    setSymbol(s);
    navigate(`/charts/${s}`);
  }, [navigate]);

  // Arrow Up / Down to cycle through the filtered symbol list
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Skip if user is typing in an input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      e.preventDefault();
      const list = filtered.length > 0 ? filtered : symbols;
      if (list.length === 0) return;
      const idx = list.indexOf(symbol);
      let next: number;
      if (idx === -1) {
        next = 0;
      } else if (e.key === 'ArrowDown') {
        next = idx + 1 >= list.length ? 0 : idx + 1;
      } else {
        next = idx - 1 < 0 ? list.length - 1 : idx - 1;
      }
      selectSymbol(list[next]);
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [symbol, filtered, symbols, selectSymbol]);

  // Scroll the active symbol button into view in the sidebar
  useEffect(() => {
    if (!symbol || !sidebarScrollRef.current) return;
    const btn = sidebarScrollRef.current.querySelector(`[data-symbol="${symbol}"]`);
    if (btn) btn.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }, [symbol]);

  return (
    <>
      <PageHeader title="Charts">
        {symbols.length} assets
      </PageHeader>

      <div className="flex gap-0">
        {/* ── Sidebar ──────────────────────────────────────────── */}
        <div className={`flex-shrink-0 transition-all duration-300 ease-out ${sidebarCollapsed ? 'w-10' : 'w-56'}`}>
          {sidebarCollapsed ? (
            <button
              onClick={() => setSidebarCollapsed(false)}
              className="w-10 h-10 flex items-center justify-center glass-card text-[#7a8ba4] hover:text-[#b49aff] transition-colors"
              title="Expand sidebar"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          ) : (
            <div className="pr-3">
              {/* Search */}
              <div className="flex items-center gap-1.5 glass-card px-2.5 py-1.5 mb-2 group focus-within:ring-1 focus-within:ring-[#8b5cf6]/30 transition-all">
                <Search className="w-3 h-3 text-[#7a8ba4] group-focus-within:text-[#b49aff] transition-colors" />
                <input
                  ref={searchRef}
                  type="text"
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search..."
                  className="bg-transparent text-xs text-[#e2e8f0] placeholder:text-[#6b7a90] outline-none w-full"
                  onKeyDown={(e) => {
                    if (e.key === 'Escape') { setSearch(''); searchRef.current?.blur(); }
                    if (e.key === 'Enter' && filtered.length > 0) {
                      selectSymbol(filtered[0]);
                      setSearch('');
                      searchRef.current?.blur();
                    }
                  }}
                />
                {search && (
                  <button onClick={() => setSearch('')} className="text-[#7a8ba4] hover:text-[#e2e8f0]">
                    <X className="w-3 h-3" />
                  </button>
                )}
                <kbd className="text-[8px] text-[#6b7a90] font-mono hidden sm:inline">
                  {navigator.platform.includes('Mac') ? '\u2318' : 'Ctrl+'}K
                </kbd>
              </div>

              {/* View tabs */}
              <div className="flex gap-0.5 mb-2 glass-card p-0.5">
                {([
                  { key: 'sector' as PickerView, label: 'Sectors', accent: undefined },
                  { key: 'all' as PickerView, label: 'All', accent: undefined },
                  { key: 'strong_buy' as PickerView, label: 'Buy', accent: '#3ee8a5' },
                  { key: 'strong_sell' as PickerView, label: 'Sell', accent: '#ff6b8a' },
                  { key: 'filter' as PickerView, label: 'Top', accent: '#f5c542' },
                ]).map(({ key, label, accent }) => (
                  <button
                    key={key}
                    onClick={() => setPickerView(key)}
                    className={`flex-1 px-1.5 py-1 rounded text-[10px] font-medium transition-all duration-200 ${
                      pickerView === key
                        ? 'text-[#e2e8f0] shadow-sm'
                        : 'text-[#7a8ba4] hover:text-[#94a3b8]'
                    }`}
                    style={pickerView === key ? {
                      background: accent ? `${accent}20` : 'rgba(139,92,246,0.15)',
                      color: accent || '#b49aff',
                    } : {}}
                  >
                    {label}
                    {key === 'strong_buy' && strongBuySymbols.length > 0 && (
                      <span className="ml-0.5 text-[8px] opacity-60">{strongBuySymbols.length}</span>
                    )}
                    {key === 'strong_sell' && strongSellSymbols.length > 0 && (
                      <span className="ml-0.5 text-[8px] opacity-60">{strongSellSymbols.length}</span>
                    )}
                  </button>
                ))}
              </div>

              {/* Symbol list */}
              <div ref={sidebarScrollRef} className="glass-card overflow-y-auto max-h-[calc(100vh-260px)] scrollbar-thin">
                {pickerView === 'all' && (
                  <SymbolList symbols={filtered.slice(0, 150)} selected={symbol} onSelect={selectSymbol}
                    strongBuy={strongBuySymbols} strongSell={strongSellSymbols} />
                )}
                {pickerView === 'sector' && (
                  <SectorSymbolList
                    sectors={sectors} search={search} selected={symbol} onSelect={selectSymbol}
                    expandedSectors={expandedSectors} toggleSector={toggleSector}
                    strongBuy={strongBuySymbols} strongSell={strongSellSymbols}
                  />
                )}
                {pickerView === 'strong_buy' && (
                  <SymbolList
                    symbols={strongBuySymbols.filter(s => !search || s.toLowerCase().includes(search.toLowerCase()))}
                    selected={symbol} onSelect={selectSymbol} emptyText="No strong buy signals" accent="#3ee8a5"
                    strongBuy={strongBuySymbols} strongSell={strongSellSymbols}
                  />
                )}
                {pickerView === 'strong_sell' && (
                  strongSellSymbols.length === 0 ? (
                    <div className="px-3 py-8 text-center">
                      <p className="text-xs text-[#7a8ba4] mb-1">No strong sell signals</p>
                      <p className="text-[10px] text-[#6b7a90] leading-relaxed">
                        No assets currently meet the strong sell threshold.
                      </p>
                    </div>
                  ) : (
                    <SymbolList
                      symbols={strongSellSymbols.filter(s => !search || s.toLowerCase().includes(search.toLowerCase()))}
                      selected={symbol} onSelect={selectSymbol} emptyText="No matching symbols" accent="#ff6b8a"
                      strongBuy={strongBuySymbols} strongSell={strongSellSymbols}
                    />
                  )
                )}
                {pickerView === 'filter' && (
                  <RankedFilterList
                    filterMode={filterMode} setFilterMode={setFilterMode}
                    rows={rankedSymbols} selected={symbol} onSelect={selectSymbol}
                    search={search} loading={summaryQ.isLoading}
                    strongBuy={strongBuySymbols} strongSell={strongSellSymbols}
                  />
                )}
              </div>

              <button
                onClick={() => setSidebarCollapsed(true)}
                className="mt-2 w-full py-1 text-[9px] text-[#6b7a90] hover:text-[#94a3b8] transition-colors text-center"
              >
                Collapse sidebar
              </button>
            </div>
          )}
        </div>

        {/* ── Chart Area ───────────────────────────────────────── */}
        <div className="flex-1 min-w-0">
          {symbol ? (
            <ChartPanel symbol={symbol} strongBuy={strongBuySymbols} strongSell={strongSellSymbols} />
          ) : (
            <div className="glass-card p-16 text-center">
              <div className="text-[#2a2a4a] mb-4">
                <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1" className="mx-auto">
                  <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
                </svg>
              </div>
              <p className="text-sm text-[#7a8ba4]">Select a symbol to begin</p>
              <p className="text-[10px] text-[#6b7a90] mt-1">
                Use the sidebar or press {navigator.platform.includes('Mac') ? '\u2318' : 'Ctrl+'}K to search
              </p>
            </div>
          )}
        </div>
      </div>
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SYMBOL LIST with signal badges
   ═══════════════════════════════════════════════════════════════════ */
function SymbolList({
  symbols, selected, onSelect, emptyText, accent, strongBuy, strongSell,
}: {
  symbols: string[]; selected: string; onSelect: (s: string) => void;
  emptyText?: string; accent?: string;
  strongBuy: string[]; strongSell: string[];
}) {
  if (symbols.length === 0) {
    return <p className="px-3 py-6 text-xs text-[#7a8ba4] text-center">{emptyText || 'No symbols'}</p>;
  }
  return (
    <>
      {symbols.map((s) => {
        const isSelected = s === selected;
        const isBuy = strongBuy.includes(s);
        const isSell = strongSell.includes(s);
        return (
          <button
            key={s}
            data-symbol={s}
            onClick={() => onSelect(s)}
            className={`w-full text-left px-3 py-1.5 text-xs font-medium transition-all duration-150 flex items-center gap-1.5 outline-none
              ${isSelected ? 'bg-[#8b5cf6]/10' : 'hover:bg-[#8b5cf6]/5'}`}
            style={{ color: isSelected ? (accent || '#b49aff') : '#94a3b8' }}
          >
            <span className={`w-0.5 h-3 rounded-full transition-all duration-200 ${isSelected ? 'opacity-100' : 'opacity-0'}`}
              style={{ backgroundColor: accent || '#b49aff' }} />
            <span className="flex-1">{s}</span>
            {isBuy && <span className="w-1.5 h-1.5 rounded-full bg-[#3ee8a5]" title="Strong Buy" />}
            {isSell && <span className="w-1.5 h-1.5 rounded-full bg-[#ff6b8a]" title="Strong Sell" />}
          </button>
        );
      })}
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SECTOR LIST
   ═══════════════════════════════════════════════════════════════════ */
function SectorSymbolList({
  sectors, search, selected, onSelect, expandedSectors, toggleSector, strongBuy, strongSell,
}: {
  sectors: ChartSectorGroup[]; search: string; selected: string;
  onSelect: (s: string) => void;
  expandedSectors: Set<string>; toggleSector: (n: string) => void;
  strongBuy: string[]; strongSell: string[];
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
        const buyCount = syms.filter(s => strongBuy.includes(s)).length;
        const sellCount = syms.filter(s => strongSell.includes(s)).length;
        return (
          <div key={sec.name}>
            <button
              onClick={() => toggleSector(sec.name)}
              className="w-full flex items-center gap-1.5 px-2 py-1.5 hover:bg-[#8b5cf6]/5 transition-colors outline-none"
            >
              {expanded
                ? <ChevronDown className="w-3 h-3 text-[#7a8ba4]" />
                : <ChevronRight className="w-3 h-3 text-[#6b7a90]" />}
              <span className={`text-[10px] font-medium flex-1 text-left truncate ${expanded ? 'text-[#94a3b8]' : 'text-[#7a8ba4]'}`}>
                {sec.name}
              </span>
              <span className="flex items-center gap-1">
                {buyCount > 0 && <span className="text-[8px] text-[#3ee8a5]">{buyCount}</span>}
                {sellCount > 0 && <span className="text-[8px] text-[#ff6b8a]">{sellCount}</span>}
                <span className="text-[9px] text-[#6b7a90]">{syms.length}</span>
              </span>
            </button>
            {expanded && (
              <div className="chart-sector-expand">
                {syms.map((s) => {
                  const isSelected = s === selected;
                  const isBuy = strongBuy.includes(s);
                  const isSell = strongSell.includes(s);
                  return (
                    <button
                      key={s}
                      data-symbol={s}
                      onClick={() => onSelect(s)}
                      className={`w-full text-left pl-7 pr-3 py-1 text-xs transition-all duration-150 flex items-center gap-1.5 outline-none
                        ${isSelected ? 'bg-[#8b5cf6]/10 text-[#b49aff]' : 'text-[#7a8599] hover:text-[#94a3b8] hover:bg-[#8b5cf6]/5'}`}
                    >
                      <span className="flex-1">{s}</span>
                      {isBuy && <span className="w-1.5 h-1.5 rounded-full bg-[#3ee8a5]" />}
                      {isSell && <span className="w-1.5 h-1.5 rounded-full bg-[#ff6b8a]" />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>
        );
      })}
    </>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   RANKED FILTER LIST — fast filters for most profitable charts
   ═══════════════════════════════════════════════════════════════════ */
const FILTER_DEFS: { key: FilterMode; label: string; desc: string; color: string; icon: string;
  metric: (r: SummaryRow) => string; rawMetric: (r: SummaryRow) => number; format: 'pct' | 'num' | 'pct100'; }[] = [
  { key: 'momentum', label: 'Momentum', desc: 'Strongest trend signal', color: '#3ee8a5', icon: 'M',
    metric: (r) => { const v = r.momentum_score ?? 0; return (v >= 0 ? '+' : '') + v.toFixed(0); },
    rawMetric: (r) => r.momentum_score ?? 0, format: 'num' },
  { key: 'edge', label: 'Edge', desc: 'Best signal edge (Kelly)', color: '#b49aff', icon: 'E',
    metric: (r) => { const v = getMaxEdge(r); return (v >= 0 ? '+' : '') + (v * 100).toFixed(1) + '%'; },
    rawMetric: (r) => getMaxEdge(r), format: 'pct' },
  { key: 'exp_return', label: 'Return', desc: 'Best expected return', color: '#f5c542', icon: 'R',
    metric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      const best = sigs.length ? Math.max(...sigs.map(s => s.exp_ret ?? 0)) : 0;
      return (best >= 0 ? '+' : '') + best.toFixed(1) + '%';
    },
    rawMetric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      return sigs.length ? Math.max(...sigs.map(s => s.exp_ret ?? 0)) : 0;
    }, format: 'pct' },
  { key: 'p_up', label: 'P(Up)', desc: 'Highest upside probability', color: '#26A69A', icon: 'P',
    metric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      const best = sigs.length ? Math.max(...sigs.map(s => s.p_up ?? 0)) : 0;
      return (best * 100).toFixed(0) + '%';
    },
    rawMetric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      return sigs.length ? Math.max(...sigs.map(s => s.p_up ?? 0)) : 0;
    }, format: 'pct100' },
  { key: 'low_risk', label: 'Safe', desc: 'Lowest crash risk', color: '#c084fc', icon: 'S',
    metric: (r) => (r.crash_risk_score ?? 0).toFixed(1),
    rawMetric: (r) => r.crash_risk_score ?? 0, format: 'num' },
  { key: 'kelly', label: 'Kelly', desc: 'Best half-Kelly sizing', color: '#f87171', icon: 'K',
    metric: (r) => (getMaxHalfKelly(r) * 100).toFixed(1) + '%',
    rawMetric: (r) => getMaxHalfKelly(r), format: 'pct' },
  { key: 'forecast_up', label: 'Upside', desc: 'Most forecast upside', color: '#3ee8a5', icon: '\u2191',
    metric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      const best = sigs.length ? Math.max(...sigs.map(s => s.exp_ret ?? 0)) : 0;
      return (best >= 0 ? '+' : '') + (best * 100).toFixed(1) + '%';
    },
    rawMetric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      return sigs.length ? Math.max(...sigs.map(s => s.exp_ret ?? 0)) : 0;
    }, format: 'pct' },
  { key: 'forecast_down', label: 'Downside', desc: 'Most forecast downside', color: '#ff6b8a', icon: '\u2193',
    metric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      const worst = sigs.length ? Math.min(...sigs.map(s => s.exp_ret ?? 0)) : 0;
      return (worst * 100).toFixed(1) + '%';
    },
    rawMetric: (r) => {
      const sigs = Object.values(r.horizon_signals || {});
      return sigs.length ? Math.min(...sigs.map(s => s.exp_ret ?? 0)) : 0;
    }, format: 'pct' },
];

/** Compute bar width % for a metric relative to the best in the list */
function computeBarWidths(rows: SummaryRow[], rawMetric: (r: SummaryRow) => number, isInverse: boolean): number[] {
  if (rows.length === 0) return [];
  const values = rows.map(rawMetric);
  const absMax = Math.max(...values.map(v => Math.abs(v)), 0.001);
  if (isInverse) {
    const maxVal = Math.max(...values, 0.001);
    return values.map(v => Math.max(5, ((maxVal - v) / maxVal) * 100));
  }
  return values.map(v => Math.max(5, (Math.abs(v) / absMax) * 100));
}

function RankedFilterList({
  filterMode, setFilterMode, rows, selected, onSelect, search, loading, strongBuy, strongSell,
}: {
  filterMode: FilterMode; setFilterMode: (m: FilterMode) => void;
  rows: SummaryRow[]; selected: string; onSelect: (s: string) => void;
  search: string; loading: boolean;
  strongBuy: string[]; strongSell: string[];
}) {
  const currentFilter = FILTER_DEFS.find(f => f.key === filterMode)!;
  const filteredRows = search
    ? rows.filter(r => r.asset_label.toLowerCase().includes(search.toLowerCase()))
    : rows;
  const barWidths = computeBarWidths(filteredRows, currentFilter.rawMetric, filterMode === 'low_risk');

  return (
    <div>
      {/* Filter mode pills */}
      <div className="flex flex-wrap gap-0.5 p-1.5 border-b border-[#2a2a4a]/50">
        {FILTER_DEFS.map(f => {
          const active = filterMode === f.key;
          return (
            <button
              key={f.key}
              onClick={() => setFilterMode(f.key)}
              className={`px-1.5 py-1 rounded text-[9px] font-semibold transition-all duration-200 outline-none ${
                active ? 'shadow-sm' : 'text-[#6b7a90] hover:text-[#94a3b8] bg-transparent'
              }`}
              style={active ? { background: `${f.color}18`, color: f.color } : {}}
              title={f.desc}
            >
              {f.label}
            </button>
          );
        })}
      </div>

      {/* Header with count */}
      <div className="flex items-center justify-between px-2.5 py-1.5 border-b border-[#2a2a4a]/30">
        <p className="text-[9px] text-[#6b7a90]">{currentFilter.desc}</p>
        <span className="text-[8px] tabular-nums text-[#3a3a5a]">{filteredRows.length} assets</span>
      </div>

      {loading ? (
        <div className="px-3 py-8 text-center">
          <div className="inline-block w-4 h-4 border-2 border-[#2a2a4a] border-t-[#8b5cf6] rounded-full animate-spin mb-2" />
          <p className="text-[10px] text-[#6b7a90]">Loading signals...</p>
        </div>
      ) : filteredRows.length === 0 ? (
        <div className="px-3 py-8 text-center">
          <p className="text-[10px] text-[#6b7a90]">No matching assets</p>
        </div>
      ) : (
        filteredRows.map((r, idx) => {
          const ticker = extractTicker(r.asset_label);
          const companyName = r.asset_label.includes('(') ? r.asset_label.split('(')[0].trim() : '';
          const isSelected = ticker === selected;
          const isBuy = strongBuy.includes(ticker);
          const isSell = strongSell.includes(ticker);
          const metricStr = currentFilter.metric(r);
          const barW = barWidths[idx] ?? 0;
          const isTop3 = idx < 3;
          return (
            <button
              key={ticker}
              data-symbol={ticker}
              onClick={() => onSelect(ticker)}
              className={`group relative w-full text-left px-2 py-1.5 text-xs transition-all duration-150 overflow-hidden outline-none
                ${isSelected
                  ? 'bg-[#8b5cf6]/10'
                  : 'hover:bg-[#8b5cf6]/5'
                }`}
            >
              {/* Metric bar background */}
              <div
                className="absolute inset-y-0 left-0 transition-all duration-500 ease-out opacity-[0.06]"
                style={{ width: `${barW}%`, backgroundColor: currentFilter.color }}
              />
              <div className="relative flex items-center gap-1.5">
                {/* Rank */}
                <span className={`text-[9px] w-4 text-right tabular-nums font-mono flex-shrink-0 ${
                  isTop3 ? 'font-bold' : ''
                }`} style={{ color: isTop3 ? currentFilter.color : '#3a3a5a' }}>
                  {idx + 1}
                </span>
                {/* Selection indicator */}
                <span className={`w-0.5 h-3.5 rounded-full transition-all duration-200 flex-shrink-0 ${
                  isSelected ? 'opacity-100' : 'opacity-0'
                }`} style={{ backgroundColor: currentFilter.color }} />
                {/* Ticker + company */}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-1">
                    <span className={`font-semibold text-[11px] tracking-wide ${
                      isSelected ? 'text-[#e2e8f0]' : 'text-[#cbd5e1] group-hover:text-[#e2e8f0]'
                    }`}>
                      {ticker}
                    </span>
                    {isBuy && (
                      <span className="px-1 py-px rounded text-[7px] font-bold bg-[#3ee8a5]/15 text-[#3ee8a5]">BUY</span>
                    )}
                    {isSell && (
                      <span className="px-1 py-px rounded text-[7px] font-bold bg-[#ff6b8a]/15 text-[#ff6b8a]">SELL</span>
                    )}
                  </div>
                  {companyName && (
                    <p className="text-[8px] text-[#6b7a90] truncate leading-tight mt-px">
                      {companyName}
                    </p>
                  )}
                </div>
                {/* Metric value */}
                <span className={`text-[10px] tabular-nums font-bold flex-shrink-0 ${
                  isTop3 ? '' : 'opacity-80'
                }`} style={{ color: currentFilter.color }}>
                  {metricStr}
                </span>
              </div>
            </button>
          );
        })
      )}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   CHART PANEL
   ═══════════════════════════════════════════════════════════════════ */
function ChartPanel({ symbol, strongBuy, strongSell }: { symbol: string; strongBuy: string[]; strongSell: string[] }) {
  const priceChartRef = useRef<HTMLDivElement>(null);
  const priceChart = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<any>(null);
  const volumeSeriesRef = useRef<any>(null);
  const [overlays, setOverlays] = useState<Record<OverlayKey, boolean>>(DEFAULT_OVERLAYS);
  const [subIndicators, setSubIndicators] = useState<Record<SubIndicatorKey, boolean>>(DEFAULT_SUB_INDICATORS);
  const [panelOpen, setPanelOpen] = useState(false);
  const [indicatorPanelOpen, setIndicatorPanelOpen] = useState(false);
  const [timeRange, setTimeRange] = useState<TimeRange>('1Y');
  const [crosshairData, setCrosshairData] = useState<{
    time: string; o: number; h: number; l: number; c: number; v: number;
  } | null>(null);

  const toggle = useCallback((key: OverlayKey) => setOverlays(prev => ({ ...prev, [key]: !prev[key] })), []);
  const toggleSub = useCallback((key: SubIndicatorKey) => setSubIndicators(prev => ({ ...prev, [key]: !prev[key] })), []);
  const allOn = useCallback(() => setOverlays(DEFAULT_OVERLAYS), []);
  const allOff = useCallback(() => setOverlays(
    Object.fromEntries(Object.keys(DEFAULT_OVERLAYS).map(k => [k, false])) as Record<OverlayKey, boolean>
  ), []);
  const allActive = Object.values(overlays).every(v => v);
  const noneActive = Object.values(overlays).every(v => !v);
  const activeCount = Object.values(overlays).filter(v => v).length;
  const activeSubCount = Object.values(subIndicators).filter(v => v).length;

  // Keyboard shortcuts
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
  }, [toggle, allOn, allOff]);

  const showSMA20 = overlays.sma20;
  const showSMA50 = overlays.sma50;
  const showSMA200 = overlays.sma200;
  const showBollinger = overlays.bb;
  const showForecastMedian = overlays.forecastMedian;
  const showCIUpper = overlays.ciUpper;
  const showCILower = overlays.ciLower;

  const rangeDays = TIME_RANGES.find(r => r.key === timeRange)?.days || 365;

  const ohlcvQ = useQuery({
    queryKey: ['ohlcv', symbol, rangeDays],
    queryFn: () => api.chartOhlcv(symbol, rangeDays),
    enabled: !!symbol,
  });

  const indQ = useQuery({
    queryKey: ['indicators', symbol, rangeDays],
    queryFn: () => api.chartIndicators(symbol, rangeDays),
    enabled: !!symbol,
  });

  const forecastQ = useQuery({
    queryKey: ['forecast', symbol],
    queryFn: () => api.chartForecast(symbol),
    enabled: !!symbol,
  });

  /* Zone charts always need 365d of data regardless of current time range */
  const zoneOhlcvQ = useQuery({
    queryKey: ['ohlcv', symbol, 365],
    queryFn: () => api.chartOhlcv(symbol, 365),
    enabled: !!symbol,
    staleTime: 120_000,
  });

  /* ── Price chart ─────────────────────────────────────────── */
  useEffect(() => {
    if (!priceChartRef.current || !ohlcvQ.data?.data?.length) return;
    if (priceChart.current) { priceChart.current.remove(); priceChart.current = null; }

    const container = priceChartRef.current;
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#7a8ba4',
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: 'rgba(139,92,246,0.04)' },
        horzLines: { color: 'rgba(139,92,246,0.04)' },
      },
      width: container.clientWidth,
      height: 480,
      crosshair: {
        vertLine: { color: 'rgba(139,92,246,0.4)', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#0a0a1a' },
        horzLine: { color: 'rgba(139,92,246,0.4)', width: 1, style: LineStyle.Dashed, labelBackgroundColor: '#0a0a1a' },
      },
      rightPriceScale: {
        borderColor: 'rgba(42, 42, 74, 0.5)',
        scaleMargins: { top: 0.08, bottom: 0.18 },
      },
      timeScale: {
        borderColor: 'rgba(42, 42, 74, 0.5)',
        timeVisible: false,
        rightOffset: 12,
        barSpacing: 8,
        minBarSpacing: 4,
      },
    });
    priceChart.current = chart;

    // Candlesticks
    const showPriceLine = overlays.priceLine;
    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: '#3ee8a5',
      downColor: '#ff6b8a',
      borderUpColor: '#3ee8a5',
      borderDownColor: '#ff6b8a',
      wickUpColor: '#6ff0c0',
      wickDownColor: '#fda4af',
      lastValueVisible: showPriceLine,
      priceLineVisible: showPriceLine,
    });
    candleSeries.setData(ohlcvQ.data.data);
    candleSeriesRef.current = candleSeries;

    // Volume
    const volumeSeries = chart.addSeries(HistogramSeries, {
      priceFormat: { type: 'volume' },
      priceScaleId: 'volume',
    });
    chart.priceScale('volume').applyOptions({ scaleMargins: { top: 0.84, bottom: 0 } });
    volumeSeries.setData(
      ohlcvQ.data.data.map((d: OHLCVBar) => ({
        time: d.time,
        value: d.volume,
        color: d.close >= d.open ? 'rgba(0,230,118,0.18)' : 'rgba(255,23,68,0.18)',
      }))
    );
    volumeSeriesRef.current = volumeSeries;

    // Crosshair move for live OHLCV data
    chart.subscribeCrosshairMove((param) => {
      if (!param.time || !param.seriesData) {
        setCrosshairData(null);
        return;
      }
      const candle = param.seriesData.get(candleSeries) as any;
      const vol = param.seriesData.get(volumeSeries) as any;
      if (candle) {
        setCrosshairData({
          time: String(param.time),
          o: candle.open, h: candle.high, l: candle.low, c: candle.close,
          v: vol?.value || 0,
        });
      }
    });

    // Signal markers
    const isSB = strongBuy.includes(symbol);
    const isSS = strongSell.includes(symbol);
    if (isSB || isSS) {
      const lastBar = ohlcvQ.data.data[ohlcvQ.data.data.length - 1];
      if (lastBar) {
        try {
          (candleSeries as any).setMarkers([{
            time: lastBar.time,
            position: isSB ? 'belowBar' : 'aboveBar',
            color: isSB ? '#3ee8a5' : '#ff6b8a',
            shape: isSB ? 'arrowUp' : 'arrowDown',
            text: isSB ? 'STRONG BUY' : 'STRONG SELL',
          }]);
        } catch { /* markers not supported in this version */ }
      }
    }

    // SMAs with clean presentation (no price line, no last value label)
    const smaOpts = { crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false };
    if (indQ.data?.indicators) {
      const ind = indQ.data.indicators;
      if (showSMA20 && ind.sma20?.length)
        chart.addSeries(LineSeries, { color: '#f5c542', lineWidth: 1, ...smaOpts }).setData(ind.sma20);
      if (showSMA50 && ind.sma50?.length)
        chart.addSeries(LineSeries, { color: '#b49aff', lineWidth: 1, ...smaOpts }).setData(ind.sma50);
      if (showSMA200 && ind.sma200?.length)
        chart.addSeries(LineSeries, { color: '#c084fc', lineWidth: 1, ...smaOpts }).setData(ind.sma200);
    }

    // Bollinger Bands with dotted style
    if (indQ.data?.indicators?.bollinger && showBollinger) {
      const bb = indQ.data.indicators.bollinger;
      const bbOpts = { lineWidth: 1 as const, lineStyle: LineStyle.Dotted, ...smaOpts };
      if (bb.upper?.length) chart.addSeries(LineSeries, { color: 'rgba(139,92,246,0.25)', ...bbOpts }).setData(bb.upper);
      if (bb.lower?.length) chart.addSeries(LineSeries, { color: 'rgba(139,92,246,0.25)', ...bbOpts }).setData(bb.lower);
    }

    // Forecast overlay
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
        const ciWidth = lastPrice * Math.abs(retPct) * 0.5 + lastPrice * 0.01 * Math.sqrt(f.horizon_days);
        upperData.push({ time: dateStr, value: medianPrice + ciWidth });
        lowerData.push({ time: dateStr, value: Math.max(0, medianPrice - ciWidth) });
        medianData.push({ time: dateStr, value: medianPrice });
      }

      const fcOpts = { priceScaleId: 'right' as const, ...smaOpts };
      if (showCIUpper) {
        chart.addSeries(AreaSeries, {
          topColor: 'rgba(139,92,246,0.12)', bottomColor: 'rgba(139,92,246,0.02)',
          lineColor: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, ...fcOpts,
        }).setData(upperData);
      }
      if (showCILower) {
        chart.addSeries(AreaSeries, {
          topColor: 'rgba(139,92,246,0.02)', bottomColor: 'rgba(139,92,246,0.12)',
          lineColor: 'rgba(139,92,246,0.25)', lineWidth: 1, lineStyle: LineStyle.Dashed, ...fcOpts,
        }).setData(lowerData);
      }
      if (showForecastMedian) {
        chart.addSeries(LineSeries, {
          color: '#b49aff', lineWidth: 2, lineStyle: LineStyle.Dashed, ...fcOpts,
        }).setData(medianData);
      }
    }

    chart.timeScale().fitContent();
    const handleResize = () => {
      if (priceChartRef.current) chart.applyOptions({ width: priceChartRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); chart.remove(); priceChart.current = null; };
  }, [ohlcvQ.data, indQ.data, forecastQ.data, showSMA20, showSMA50, showSMA200, showBollinger, showForecastMedian, showCIUpper, showCILower, symbol, strongBuy, strongSell, overlays.priceLine]);

  /* ── Derived data ────────────────────────────────────────── */
  if (ohlcvQ.isLoading) {
    return (
      <div className="glass-card p-16 text-center chart-fade-in">
        <LoadingSpinner text={`Loading ${symbol}...`} />
      </div>
    );
  }
  if (ohlcvQ.error) {
    return (
      <div className="glass-card p-8 text-center">
        <p className="text-[#ff6b8a] text-sm">Failed to load chart for {symbol}</p>
        <p className="text-[10px] text-[#6b7a90] mt-1">Check data availability or try another symbol</p>
      </div>
    );
  }

  const bars = ohlcvQ.data?.data;
  const lastBar = bars?.[bars.length - 1];
  const prevBar = bars?.[bars.length - 2];
  const change = lastBar && prevBar ? ((lastBar.close - prevBar.close) / prevBar.close * 100) : 0;
  const isSB = strongBuy.includes(symbol);
  const isSS = strongSell.includes(symbol);

  const high = lastBar?.high || 0;
  const low = lastBar?.low || 0;
  const open = lastBar?.open || 0;
  const volume = lastBar?.volume || 0;

  const rsiData = indQ.data?.indicators?.rsi;
  const rsiValue = rsiData?.length ? rsiData[rsiData.length - 1].value : null;

  // Performance summary line
  const firstBar = bars?.[0];
  const periodReturn = firstBar && lastBar ? ((lastBar.close - firstBar.close) / firstBar.close * 100) : 0;

  return (
    <div className="chart-fade-in">
      {/* ── Ticker Header ───────────────────────────────────── */}
      <div className="flex items-start gap-4 mb-3">
        <div className="flex-1 min-w-0">
          <div className="flex items-baseline gap-3 mb-1">
            <h2 className="text-2xl font-bold tracking-tight" style={{ color: 'var(--text-luminous)' }}>{symbol}</h2>
            {lastBar && (
              <>
                <span className="text-xl font-semibold text-[var(--text-primary)] tabular-nums">${lastBar.close.toFixed(2)}</span>
                <span className="text-sm font-semibold tabular-nums px-1.5 py-0.5 rounded"
                  style={{
                    color: change >= 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                    background: change >= 0 ? 'rgba(62,232,165,0.08)' : 'rgba(255,107,138,0.08)',
                  }}>
                  {change >= 0 ? '+' : ''}{change.toFixed(2)}%
                </span>
              </>
            )}
            {isSB && (
              <span className="px-2 py-0.5 rounded-md text-[10px] font-bold" style={{ background: 'rgba(62,232,165,0.15)', color: 'var(--accent-emerald)', border: '1px solid rgba(62,232,165,0.2)' }}>
                STRONG BUY
              </span>
            )}
            {isSS && (
              <span className="px-2 py-0.5 rounded-md text-[10px] font-bold" style={{ background: 'rgba(255,107,138,0.15)', color: 'var(--accent-rose)', border: '1px solid rgba(255,107,138,0.2)' }}>
                STRONG SELL
              </span>
            )}
          </div>

          {/* OHLCV data bar (live with crosshair) */}
          <div className="flex items-center gap-4 text-[10px] text-[#7a8ba4] tabular-nums">
            <span>O <span className="text-[#94a3b8]">{(crosshairData?.o ?? open).toFixed(2)}</span></span>
            <span>H <span className="text-[#3ee8a5]/70">{(crosshairData?.h ?? high).toFixed(2)}</span></span>
            <span>L <span className="text-[#ff6b8a]/70">{(crosshairData?.l ?? low).toFixed(2)}</span></span>
            <span>C <span className="text-[#94a3b8]">{(crosshairData?.c ?? lastBar?.close ?? 0).toFixed(2)}</span></span>
            <span className="text-[#2a2a4a]">|</span>
            <span>Vol <span className="text-[#94a3b8]">{formatVolume(crosshairData?.v ?? volume)}</span></span>
            {rsiValue != null && (
              <>
                <span className="text-[#2a2a4a]">|</span>
                <span>RSI <span className={rsiValue > 70 ? 'text-[#ff6b8a]' : rsiValue < 30 ? 'text-[#3ee8a5]' : 'text-[#94a3b8]'}>
                  {rsiValue.toFixed(1)}
                </span></span>
              </>
            )}
            <span className="text-[#2a2a4a]">|</span>
            <span>Period <span className={periodReturn >= 0 ? 'text-[#3ee8a5]/70' : 'text-[#ff6b8a]/70'}>
              {periodReturn >= 0 ? '+' : ''}{periodReturn.toFixed(1)}%
            </span></span>
            {crosshairData && (
              <>
                <span className="text-[#2a2a4a]">|</span>
                <span className="text-[#6b7a90]">{crosshairData.time}</span>
              </>
            )}
          </div>
        </div>

        {/* ── Controls ──────────────────────────────────────── */}
        <div className="flex items-center gap-2 flex-shrink-0 pt-1">
          {/* Time range pills */}
          <div className="flex rounded-lg p-0.5" style={{ background: 'var(--void)', border: '1px solid var(--border-void)' }}>
            {TIME_RANGES.map(r => (
              <button
                key={r.key}
                onClick={() => setTimeRange(r.key)}
                className="px-2 py-1 rounded-md text-[10px] font-semibold transition-all duration-200"
                style={timeRange === r.key
                  ? { background: 'rgba(139,92,246,0.15)', color: 'var(--accent-violet)', boxShadow: '0 0 8px rgba(139,92,246,0.15)' }
                  : { color: 'var(--text-muted)' }
                }
              >
                {r.label}
              </button>
            ))}
          </div>

          {/* Layers button */}
          <button
            onClick={() => setPanelOpen(v => !v)}
            className="px-2.5 py-1.5 rounded-lg text-[10px] font-semibold transition-all duration-200 flex items-center gap-1.5"
            style={panelOpen
              ? { background: 'rgba(139,92,246,0.15)', color: 'var(--accent-violet)', border: '1px solid rgba(139,92,246,0.3)' }
              : { background: 'var(--void)', color: 'var(--text-muted)', border: '1px solid var(--border-void)' }
            }
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
            {activeCount}/{OVERLAY_DEFS.length}
          </button>

          {/* Indicators button */}
          <button
            onClick={() => setIndicatorPanelOpen(v => !v)}
            className="px-2.5 py-1.5 rounded-lg text-[10px] font-semibold transition-all duration-200 flex items-center gap-1.5"
            style={indicatorPanelOpen
              ? { background: 'rgba(62,232,165,0.12)', color: '#3ee8a5', border: '1px solid rgba(62,232,165,0.3)' }
              : activeSubCount > 0
                ? { background: 'rgba(62,232,165,0.06)', color: '#3ee8a5', border: '1px solid var(--border-void)' }
                : { background: 'var(--void)', color: 'var(--text-muted)', border: '1px solid var(--border-void)' }
            }
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
            </svg>
            {activeSubCount > 0 ? `${activeSubCount} on` : 'Indicators'}
          </button>
        </div>
      </div>

      {/* ── Overlay Control Panel ────────────────────────────── */}
      <div className={`overflow-hidden transition-all duration-300 ease-out ${
        panelOpen ? 'max-h-[600px] opacity-100 mb-3' : 'max-h-0 opacity-0'
      }`}>
        <div className="glass-card p-3 backdrop-blur-xl">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider font-semibold">Chart Layers</span>
            <div className="flex gap-1.5">
              <button onClick={allOn} disabled={allActive}
                className="px-2 py-0.5 rounded text-[9px] font-semibold transition-all bg-[#3ee8a5]/8 text-[#3ee8a5] hover:bg-[#3ee8a5]/15 disabled:opacity-20 disabled:cursor-default">
                All On
              </button>
              <button onClick={allOff} disabled={noneActive}
                className="px-2 py-0.5 rounded text-[9px] font-semibold transition-all bg-[#ff6b8a]/8 text-[#ff6b8a] hover:bg-[#ff6b8a]/15 disabled:opacity-20 disabled:cursor-default">
                All Off
              </button>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-4 gap-2">
            {['Moving Averages', 'Volatility', 'Forecast', 'Overlays'].map(group => (
              <div key={group}>
                <p className="text-[8px] text-[#3a3a5a] uppercase tracking-widest mb-1.5 font-bold">{group}</p>
                <div className="flex flex-col gap-0.5">
                  {OVERLAY_DEFS.filter(d => d.group === group).map(d => (
                    <button
                      key={d.key}
                      onClick={() => toggle(d.key)}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-200 group ${
                        overlays[d.key] ? 'bg-[#8b5cf6]/10' : 'bg-transparent hover:bg-[#8b5cf6]/5'
                      }`}
                      style={{
                        color: overlays[d.key] ? d.color : '#6b7a90',
                        border: overlays[d.key] ? `1px solid ${d.color}22` : '1px solid transparent',
                      }}
                    >
                      <span
                        className={`w-2 h-2 rounded-full transition-all duration-300 flex-shrink-0 ${
                          overlays[d.key] ? 'scale-100' : 'scale-75 opacity-30'
                        }`}
                        style={{
                          backgroundColor: d.color,
                          boxShadow: overlays[d.key] ? `0 0 6px ${d.color}40` : 'none',
                        }}
                      />
                      <span className="flex-1 text-left">{d.label}</span>
                      <kbd className="text-[7px] px-1 py-0.5 rounded bg-[#0a0a1a] text-[#3a3a5a] font-mono opacity-0 group-hover:opacity-100 transition-opacity">
                        {d.shortcut}
                      </kbd>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
          <p className="text-[8px] text-[#2a2a3a] mt-2 text-center font-medium">
            1-8 toggle  /  9 all on  /  0 all off  /  L panel
          </p>
        </div>
      </div>

      {/* ── Indicator Control Panel ──────────────────────────── */}
      <div className={`overflow-hidden transition-all duration-300 ease-out ${
        indicatorPanelOpen ? 'max-h-[400px] opacity-100 mb-3' : 'max-h-0 opacity-0'
      }`}>
        <div className="glass-card p-3 backdrop-blur-xl">
          <div className="flex items-center justify-between mb-2.5">
            <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-wider font-semibold">
              Indicators {activeSubCount > 0 && <span className="text-[#3ee8a5] ml-1">{activeSubCount} active</span>}
            </span>
            <div className="flex gap-1.5">
              <button
                onClick={() => setSubIndicators(Object.fromEntries(SUB_INDICATOR_DEFS.map(d => [d.key, true])) as Record<SubIndicatorKey, boolean>)}
                disabled={activeSubCount === SUB_INDICATOR_DEFS.length}
                className="px-2 py-0.5 rounded text-[9px] font-semibold transition-all bg-[#3ee8a5]/8 text-[#3ee8a5] hover:bg-[#3ee8a5]/15 disabled:opacity-20 disabled:cursor-default">
                All On
              </button>
              <button
                onClick={() => setSubIndicators(DEFAULT_SUB_INDICATORS)}
                disabled={activeSubCount === 0}
                className="px-2 py-0.5 rounded text-[9px] font-semibold transition-all bg-[#ff6b8a]/8 text-[#ff6b8a] hover:bg-[#ff6b8a]/15 disabled:opacity-20 disabled:cursor-default">
                All Off
              </button>
            </div>
          </div>
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-2">
            {['Signal', 'Momentum', 'Trend', 'Volatility', 'Volume'].map(group => (
              <div key={group}>
                <p className="text-[8px] text-[#3a3a5a] uppercase tracking-widest mb-1.5 font-bold">{group}</p>
                <div className="flex flex-col gap-0.5">
                  {SUB_INDICATOR_DEFS.filter(d => d.group === group).map(d => (
                    <button
                      key={d.key}
                      onClick={() => toggleSub(d.key)}
                      className={`flex items-center gap-2 px-2 py-1.5 rounded-lg text-[11px] font-medium transition-all duration-200 ${
                        subIndicators[d.key] ? 'bg-[#8b5cf6]/10' : 'bg-transparent hover:bg-[#8b5cf6]/5'
                      }`}
                      style={{
                        color: subIndicators[d.key] ? d.color : '#6b7a90',
                        border: subIndicators[d.key] ? `1px solid ${d.color}22` : '1px solid transparent',
                      }}
                    >
                      <span
                        className={`w-2 h-2 rounded-full transition-all duration-300 flex-shrink-0 ${
                          subIndicators[d.key] ? 'scale-100' : 'scale-75 opacity-30'
                        }`}
                        style={{
                          backgroundColor: d.color,
                          boxShadow: subIndicators[d.key] ? `0 0 6px ${d.color}40` : 'none',
                        }}
                      />
                      <span className="flex-1 text-left">{d.label}</span>
                    </button>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Overlay legend (always visible, click to toggle) ── */}
      <div className="flex items-center gap-3 mb-1 min-h-[18px]">
        {OVERLAY_DEFS.map(d => {
          const active = overlays[d.key];
          return (
            <button
              key={d.key}
              onClick={() => toggle(d.key)}
              className={`flex items-center gap-1 text-[9px] transition-all cursor-pointer ${
                active ? 'hover:opacity-60' : 'opacity-30 hover:opacity-50'
              }`}
              style={{ color: d.color }}
              title={active ? `Click to hide ${d.label}` : `Click to show ${d.label}`}
            >
              <span
                className="inline-block w-3 h-[2px] rounded-full"
                style={{ backgroundColor: d.color, opacity: active ? 1 : 0.4 }}
              />
              <span className={active ? '' : 'line-through'}>{d.label}</span>
            </button>
          );
        })}
      </div>

      {/* ── Price Chart ──────────────────────────────────────── */}
      <div className="chart-container rounded-xl overflow-hidden ring-1 ring-[#2a2a4a]/30">
        <div ref={priceChartRef} />
      </div>

      {/* ── Additional Indicator Sub-charts ──────────────────── */}
      {indQ.data?.indicators && SUB_INDICATOR_DEFS.map(d => {
        if (!subIndicators[d.key]) return null;
        // Check if data exists for this indicator
        const ind = indQ.data.indicators;
        const hasData = d.key === 'macd' ? ind.macd?.macd?.length
          : d.key === 'stochastic' ? ind.stochastic?.k?.length
          : d.key === 'adx' ? ind.adx?.adx?.length
          : (ind as any)[d.key]?.length;
        if (!hasData) return null;
        return (
          <SubIndicatorPane
            key={d.key}
            indicatorKey={d.key}
            indicators={ind}
            priceChartApi={priceChart.current}
            onToggle={() => toggleSub(d.key)}
          />
        );
      })}

      {/* ── Multi-Horizon Forecast Cards ─────────────────────── */}
      {forecastQ.data?.forecasts?.length ? (
        <div className="mt-4">
          <div className="flex items-center gap-2 mb-2">
            <h3 className="text-[11px] font-semibold text-[var(--text-muted)] uppercase tracking-wider">Forecast</h3>
            <div className="flex-1 h-px" style={{ background: 'var(--border-void)' }} />
          </div>
          <div className="grid grid-cols-7 gap-1.5">
            {forecastQ.data.forecasts.map((f) => {
              const isPositive = f.expected_return_pct >= 0;
              const isBuy = f.signal_label === 'BUY' || f.signal_label === 'STRONG BUY';
              const isSell = f.signal_label === 'SELL' || f.signal_label === 'STRONG SELL';
              const signalColor = isBuy ? 'var(--accent-emerald)' : isSell ? 'var(--accent-rose)' : 'var(--text-muted)';
              return (
                <div
                  key={f.horizon_days}
                  className="glass-card p-2.5 text-center transition-all hover:ring-1"
                  style={{ borderColor: 'var(--border-void)' }}
                >
                  <p className="text-[9px] text-[var(--text-muted)] font-semibold uppercase tracking-wider mb-1.5">
                    {formatHorizon(f.horizon_days)}
                  </p>
                  <p className="text-base font-bold tabular-nums leading-none"
                    style={{ color: isPositive ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                    {isPositive ? '+' : ''}{f.expected_return_pct.toFixed(1)}%
                  </p>
                  <div className="mt-1.5 flex items-center justify-center gap-1">
                    <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: signalColor }} />
                    <span className="text-[8px] font-semibold uppercase" style={{ color: signalColor }}>
                      {f.signal_label}
                    </span>
                  </div>
                  <p className="text-[9px] text-[var(--text-muted)] mt-0.5 tabular-nums">
                    {(f.probability_up * 100).toFixed(0)}% up
                  </p>
                </div>
              );
            })}
          </div>
        </div>
      ) : null}

      {/* ── Buy & Sell Zone Mini Charts ──────────────────────── */}
      {zoneOhlcvQ.data?.data?.length && forecastQ.data?.forecasts?.length ? (
        <BuySellZoneCharts
          ohlcv={zoneOhlcvQ.data.data}
          forecasts={forecastQ.data.forecasts}
          symbol={symbol}
        />
      ) : null}
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   SUB-INDICATOR PANE — reusable mini chart for indicators
   ═══════════════════════════════════════════════════════════════════ */
function SubIndicatorPane({
  indicatorKey,
  indicators,
  priceChartApi,
  onToggle,
}: {
  indicatorKey: SubIndicatorKey;
  indicators: any;
  priceChartApi: IChartApi | null;
  onToggle: () => void;
}) {
  const chartRef = useRef<HTMLDivElement>(null);
  const chartApi = useRef<IChartApi | null>(null);
  const def = SUB_INDICATOR_DEFS.find(d => d.key === indicatorKey)!;

  useEffect(() => {
    if (!chartRef.current || !indicators) return;
    if (chartApi.current) { chartApi.current.remove(); chartApi.current = null; }

    const container = chartRef.current;
    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#7a8ba4',
        fontFamily: "'Inter', system-ui, sans-serif",
        fontSize: 10,
      },
      grid: {
        vertLines: { color: 'rgba(42, 42, 74, 0.3)' },
        horzLines: { color: 'rgba(42, 42, 74, 0.3)' },
      },
      width: container.clientWidth,
      height: indicatorKey === 'composite' ? 130 : 100,
      rightPriceScale: { scaleMargins: { top: 0.08, bottom: 0.08 }, borderColor: 'rgba(42, 42, 74, 0.5)' },
      timeScale: { borderColor: 'rgba(42, 42, 74, 0.5)', visible: false },
    });
    chartApi.current = chart;

    const lineOpts = { crosshairMarkerVisible: false, lastValueVisible: false, priceLineVisible: false };
    const refOpts = { lineWidth: 1 as const, lineStyle: LineStyle.Dotted, ...lineOpts };

    switch (indicatorKey) {
      case 'composite': {
        const d = indicators.composite;
        if (!d?.length) break;
        // Color-coded histogram: green (buy) / red (sell) / muted (neutral)
        chart.addSeries(HistogramSeries, {
          ...lineOpts,
        }).setData(d.map((p: any) => ({
          time: p.time,
          value: p.value,
          color: p.value >= 30 ? 'rgba(62,232,165,0.75)'
            : p.value >= 10 ? 'rgba(62,232,165,0.35)'
            : p.value <= -30 ? 'rgba(255,107,138,0.75)'
            : p.value <= -10 ? 'rgba(255,107,138,0.35)'
            : 'rgba(122,139,164,0.25)',
        })));
        // Buy/Sell zone reference lines
        const times = d.map((p: any) => p.time);
        chart.addSeries(LineSeries, { color: 'rgba(62,232,165,0.4)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 30 })));
        chart.addSeries(LineSeries, { color: 'rgba(255,107,138,0.4)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: -30 })));
        chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0 })));
        break;
      }
      case 'rsi': {
        const d = indicators.rsi;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#b49aff', lineWidth: 2, ...lineOpts }).setData(d);
        const times = d.map((p: any) => p.time);
        chart.addSeries(LineSeries, { color: 'rgba(255,107,138,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 70 })));
        chart.addSeries(LineSeries, { color: 'rgba(62,232,165,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 30 })));
        chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 50 })));
        break;
      }
      case 'macd': {
        const d = indicators.macd;
        if (!d) break;
        if (d.histogram?.length) {
          chart.addSeries(HistogramSeries, {
            ...lineOpts,
          }).setData(d.histogram.map((p: any) => ({
            time: p.time,
            value: p.value,
            color: p.value >= 0 ? 'rgba(62,232,165,0.6)' : 'rgba(255,107,138,0.6)',
          })));
        }
        if (d.macd?.length)
          chart.addSeries(LineSeries, { color: '#3ee8a5', lineWidth: 2, ...lineOpts }).setData(d.macd);
        if (d.signal?.length)
          chart.addSeries(LineSeries, { color: '#ff6b8a', lineWidth: 1, ...lineOpts }).setData(d.signal);
        // Zero line
        if (d.macd?.length) {
          const times = d.macd.map((p: any) => p.time);
          chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0 })));
        }
        break;
      }
      case 'stochastic': {
        const d = indicators.stochastic;
        if (!d) break;
        if (d.k?.length)
          chart.addSeries(LineSeries, { color: '#f5c542', lineWidth: 2, ...lineOpts }).setData(d.k);
        if (d.d?.length)
          chart.addSeries(LineSeries, { color: '#ff6b8a', lineWidth: 1, ...lineOpts }).setData(d.d);
        // Reference lines
        if (d.k?.length) {
          const times = d.k.map((p: any) => p.time);
          chart.addSeries(LineSeries, { color: 'rgba(255,107,138,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 80 })));
          chart.addSeries(LineSeries, { color: 'rgba(62,232,165,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 20 })));
        }
        break;
      }
      case 'adx': {
        const d = indicators.adx;
        if (!d) break;
        if (d.adx?.length)
          chart.addSeries(LineSeries, { color: '#26A69A', lineWidth: 2, ...lineOpts }).setData(d.adx);
        if (d.plus_di?.length)
          chart.addSeries(LineSeries, { color: '#3ee8a5', lineWidth: 1, ...lineOpts }).setData(d.plus_di);
        if (d.minus_di?.length)
          chart.addSeries(LineSeries, { color: '#ff6b8a', lineWidth: 1, ...lineOpts }).setData(d.minus_di);
        // 25 threshold
        if (d.adx?.length) {
          const times = d.adx.map((p: any) => p.time);
          chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.2)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 25 })));
        }
        break;
      }
      case 'atr': {
        const d = indicators.atr;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#c084fc', lineWidth: 2, ...lineOpts }).setData(d);
        break;
      }
      case 'obv': {
        const d = indicators.obv;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#64b5f6', lineWidth: 2, ...lineOpts }).setData(d);
        break;
      }
      case 'cci': {
        const d = indicators.cci;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#b49aff', lineWidth: 2, ...lineOpts }).setData(d);
        // Reference lines
        const times = d.map((p: any) => p.time);
        chart.addSeries(LineSeries, { color: 'rgba(255,107,138,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 100 })));
        chart.addSeries(LineSeries, { color: 'rgba(62,232,165,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: -100 })));
        chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0 })));
        break;
      }
      case 'mfi': {
        const d = indicators.mfi;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#4dd0e1', lineWidth: 2, ...lineOpts }).setData(d);
        const times = d.map((p: any) => p.time);
        chart.addSeries(LineSeries, { color: 'rgba(255,107,138,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 80 })));
        chart.addSeries(LineSeries, { color: 'rgba(62,232,165,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 20 })));
        break;
      }
      case 'cmf': {
        const d = indicators.cmf;
        if (!d?.length) break;
        chart.addSeries(HistogramSeries, {
          ...lineOpts,
        }).setData(d.map((p: any) => ({
          time: p.time,
          value: p.value,
          color: p.value >= 0 ? 'rgba(62,232,165,0.5)' : 'rgba(255,107,138,0.5)',
        })));
        if (d.length) {
          const times = d.map((p: any) => p.time);
          chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0 })));
        }
        break;
      }
      case 'roc': {
        const d = indicators.roc;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#ff9f43', lineWidth: 2, ...lineOpts }).setData(d);
        const times = d.map((p: any) => p.time);
        chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0 })));
        break;
      }
      case 'bbpctb': {
        const d = indicators.bbpctb;
        if (!d?.length) break;
        chart.addSeries(LineSeries, { color: '#818cf8', lineWidth: 2, ...lineOpts }).setData(d);
        const times = d.map((p: any) => p.time);
        chart.addSeries(LineSeries, { color: 'rgba(255,107,138,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 1 })));
        chart.addSeries(LineSeries, { color: 'rgba(62,232,165,0.25)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0 })));
        chart.addSeries(LineSeries, { color: 'rgba(100,116,139,0.15)', ...refOpts }).setData(times.map((t: string) => ({ time: t, value: 0.5 })));
        break;
      }
    }

    chart.timeScale().fitContent();

    // Sync time scale with main chart
    if (priceChartApi) {
      priceChartApi.timeScale().subscribeVisibleLogicalRangeChange((range) => {
        if (range && chartApi.current) chartApi.current.timeScale().setVisibleLogicalRange(range);
      });
    }

    const handleResize = () => {
      if (chartRef.current) chart.applyOptions({ width: chartRef.current.clientWidth });
    };
    window.addEventListener('resize', handleResize);
    return () => { window.removeEventListener('resize', handleResize); chart.remove(); chartApi.current = null; };
  }, [indicators, indicatorKey, priceChartApi]);

  // Header labels per indicator
  const headerInfo = useMemo(() => {
    switch (indicatorKey) {
      case 'rsi': return { refs: [{ label: '70 Overbought', color: '#ff6b8a40' }, { label: '50', color: '#7a8ba430' }, { label: '30 Oversold', color: '#3ee8a540' }] };
      case 'macd': return { refs: [{ label: 'MACD', color: '#3ee8a5' }, { label: 'Signal', color: '#ff6b8a' }, { label: 'Histogram', color: '#7a8ba4' }] };
      case 'stochastic': return { refs: [{ label: '%K', color: '#f5c542' }, { label: '%D', color: '#ff6b8a' }, { label: '80', color: '#ff6b8a40' }, { label: '20', color: '#3ee8a540' }] };
      case 'adx': return { refs: [{ label: 'ADX', color: '#26A69A' }, { label: '+DI', color: '#3ee8a5' }, { label: '-DI', color: '#ff6b8a' }, { label: '25 Trend', color: '#7a8ba430' }] };
      case 'atr': return { refs: [] };
      case 'obv': return { refs: [] };
      case 'cci': return { refs: [{ label: '+100', color: '#ff6b8a40' }, { label: '0', color: '#7a8ba430' }, { label: '-100', color: '#3ee8a540' }] };
      case 'mfi': return { refs: [{ label: '80 Overbought', color: '#ff6b8a40' }, { label: '20 Oversold', color: '#3ee8a540' }] };
      case 'cmf': return { refs: [{ label: '0', color: '#7a8ba430' }] };
      case 'roc': return { refs: [{ label: '0%', color: '#7a8ba430' }] };
      case 'bbpctb': return { refs: [{ label: '1.0 Upper', color: '#ff6b8a40' }, { label: '0.5', color: '#7a8ba430' }, { label: '0.0 Lower', color: '#3ee8a540' }] };
      case 'composite': return { refs: [{ label: '+30 Buy', color: '#3ee8a5' }, { label: '0', color: '#7a8ba430' }, { label: '-30 Sell', color: '#ff6b8a' }] };
      default: return { refs: [] };
    }
  }, [indicatorKey]);

  return (
    <div className="chart-container rounded-xl overflow-hidden ring-1 ring-[#2a2a4a]/30 mt-1">
      <div className="flex items-center gap-3 px-3 py-1.5">
        <button
          onClick={onToggle}
          className="text-[10px] font-semibold transition-colors hover:opacity-70"
          style={{ color: def.color }}
          title={`Hide ${def.label}`}
        >
          {def.label}
        </button>
        <div className="flex gap-3 ml-auto text-[8px] font-medium">
          {headerInfo.refs.map(r => (
            <span key={r.label} style={{ color: r.color }}>{r.label}</span>
          ))}
        </div>
      </div>
      <div ref={chartRef} />
    </div>
  );
}

/* ═══════════════════════════════════════════════════════════════════
   HELPERS
   ═══════════════════════════════════════════════════════════════════ */
function formatVolume(v: number): string {
  if (v >= 1_000_000_000) return (v / 1_000_000_000).toFixed(1) + 'B';
  if (v >= 1_000_000) return (v / 1_000_000).toFixed(1) + 'M';
  if (v >= 1_000) return (v / 1_000).toFixed(1) + 'K';
  return v.toFixed(0);
}
