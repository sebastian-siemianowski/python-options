/**
 * Heatmap Page — Full-screen signal star-map.
 *
 * A dedicated immersive view of the signal heatmap with filtering,
 * search, sector analytics, and a top-bar summary strip.
 * Click any asset row to expand inline zone charts (1M/3M/6M/12M).
 */
import { useQuery } from '@tanstack/react-query';
import { api, type SectorGroup, type SummaryRow, type HorizonSignal } from '../api';
import PageHeader from '../components/PageHeader';
import { DashboardSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { DashboardEmpty } from '../components/CosmicEmptyState';
import BuySellZoneCharts from '../components/BuySellZoneCharts';
import MiniPriceChart from '../components/MiniPriceChart';
import { formatHorizon } from '../utils/horizons';
import React, {
  useState, useMemo, useCallback, useEffect, useRef,
} from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ChevronDown, ChevronRight, ChevronUp, Search, X, Filter,
  TrendingUp, TrendingDown, Minus, Maximize2, Minimize2,
  ExternalLink, Loader2, ArrowUpDown,
} from 'lucide-react';

/* ── Constants ──────────────────────────────────────────────────── */
const COLLAPSE_KEY = 'heatmap_v2_collapse';
const FILTER_KEY = 'heatmap_v2_filter';

type SignalFilter = 'all' | 'strong_buy' | 'buy' | 'hold' | 'sell' | 'strong_sell';

/** Sorting state for heatmap columns */
type SortKey = 'momentum' | number;  // number = horizon key
type SortDir = 'asc' | 'desc';
interface SortState { key: SortKey; dir: SortDir; }

/** Get the sortable value for a given asset row and sort key */
function getSortValue(row: SummaryRow, key: SortKey): number {
  if (key === 'momentum') return row.momentum_score ?? 0;
  const sig = row.horizon_signals[key] || row.horizon_signals[String(key)];
  return sig?.exp_ret ?? 0;
}

/** Signal label badge colors */
function getSignalBadge(label: string): { text: string; bg: string; fg: string } | null {
  const norm = (label || '').toUpperCase().replace(/[\s-]/g, '_');
  if (norm.includes('STRONG_BUY')) return { text: 'Strong Buy', bg: 'rgba(6,78,59,0.6)', fg: 'var(--accent-emerald)' };
  if (norm.includes('BUY')) return { text: 'Buy', bg: 'rgba(6,78,59,0.4)', fg: 'rgba(62,232,165,0.85)' };
  if (norm.includes('STRONG_SELL')) return { text: 'Strong Sell', bg: 'rgba(76,5,25,0.6)', fg: 'var(--accent-rose)' };
  if (norm.includes('SELL')) return { text: 'Sell', bg: 'rgba(76,5,25,0.4)', fg: 'rgba(255,107,138,0.85)' };
  if (norm.includes('HOLD')) return { text: 'Hold', bg: 'var(--violet-8)', fg: 'var(--text-muted)' };
  return null;
}

const FILTER_OPTIONS: { value: SignalFilter; label: string; color: string }[] = [
  { value: 'all', label: 'All Signals', color: 'var(--text-secondary)' },
  { value: 'strong_buy', label: 'Strong Buy', color: 'var(--accent-emerald)' },
  { value: 'buy', label: 'Buy', color: 'rgba(62,232,165,0.7)' },
  { value: 'hold', label: 'Hold', color: 'var(--text-muted)' },
  { value: 'sell', label: 'Sell', color: 'rgba(255,107,138,0.7)' },
  { value: 'strong_sell', label: 'Strong Sell', color: 'var(--accent-rose)' },
];

/** Extract ticker from "Company Name (TICKER)" */
const extractTicker = (label: string): string => {
  if (label.includes('(')) return label.split('(').pop()!.replace(')', '').trim();
  return label;
};

/* ── Expanded Asset Row (inline zone charts) ────────────────────── */
function ExpandedAssetRow({
  assetLabel, colSpan, onClose, asset, horizons,
}: {
  assetLabel: string; colSpan: number; onClose: () => void;
  asset: SummaryRow; horizons: number[];
}) {
  const ticker = extractTicker(assetLabel);
  const navigate = useNavigate();

  const ohlcvQ = useQuery({
    queryKey: ['ohlcv', ticker, 365],
    queryFn: () => api.chartOhlcv(ticker, 365),
    staleTime: 120_000,
  });
  const forecastQ = useQuery({
    queryKey: ['forecast', ticker],
    queryFn: () => api.chartForecast(ticker),
    staleTime: 120_000,
  });
  const indQ = useQuery({
    queryKey: ['indicators', ticker, 365],
    queryFn: () => api.chartIndicators(ticker, 365),
    staleTime: 120_000,
  });

  const ohlcvLoading = ohlcvQ.isLoading;
  const hasOhlcv = !!ohlcvQ.data?.data?.length;
  const hasForecast = !!forecastQ.data?.forecasts?.length;

  return (
    <tr>
      <td colSpan={colSpan} className="px-0 py-0">
        <div
          className="mx-2 my-1.5 rounded-xl overflow-hidden"
          style={{
            background: 'linear-gradient(160deg, rgba(13,5,30,0.95) 0%, rgba(10,18,42,0.95) 100%)',
            border: '1px solid var(--violet-15)',
            boxShadow: '0 8px 40px rgba(0,0,0,0.3), inset 0 1px 0 var(--violet-6)',
          }}
        >
          {/* Header bar */}
          <div
            className="flex items-center justify-between px-5 py-2.5"
            style={{ borderBottom: '1px solid var(--violet-10)' }}
          >
            <div className="flex items-center gap-2.5">
              <span className="text-[13px] font-semibold" style={{ color: 'var(--text-primary)' }}>
                {assetLabel}
              </span>
              <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Forecast &amp; Zones</span>
            </div>
            <div className="flex items-center gap-2">
              <button
                onClick={(e) => { e.stopPropagation(); navigate(`/charts/${ticker}`); }}
                className="flex items-center gap-1 px-2.5 py-1 rounded-lg text-[10px] font-medium transition-colors"
                style={{ background: 'var(--violet-10)', color: 'var(--text-violet)' }}
              >
                Full Chart <ExternalLink className="w-3 h-3" />
              </button>
              <button
                onClick={(e) => { e.stopPropagation(); onClose(); }}
                className="p-1 rounded-lg transition-colors"
                style={{ color: 'var(--text-muted)' }}
              >
                <X className="w-3.5 h-3.5" />
              </button>
            </div>
          </div>

          {/* Content */}
          <div className="px-4 pb-4 pt-2">
            {/* Forecast summary cards from heatmap signal data */}
            <div className={`grid gap-1.5 mb-3`} style={{ gridTemplateColumns: `repeat(${horizons.length}, minmax(0, 1fr))` }}>
              {horizons.map(h => {
                const sig = asset.horizon_signals[h] || asset.horizon_signals[String(h)];
                if (!sig) return <div key={h} />;
                const ret = sig.exp_ret * 100;
                const isPos = ret >= 0;
                const lbl = (sig.label || 'HOLD').toUpperCase().replace(/[\s-]/g, '_');
                const isBuy = lbl.includes('BUY');
                const isSell = lbl.includes('SELL');
                const dotColor = isBuy ? 'var(--accent-emerald)' : isSell ? 'var(--accent-rose)' : 'var(--text-muted)';
                return (
                  <div
                    key={h}
                    className="rounded-lg p-2 text-center transition-all"
                    style={{ background: 'var(--violet-4)', border: '1px solid var(--violet-8)' }}
                  >
                    <p className="text-[9px] font-semibold uppercase tracking-wider mb-1" style={{ color: 'var(--text-muted)' }}>
                      {formatHorizon(h)}
                    </p>
                    <p className="text-[14px] font-bold tabular-nums leading-none"
                      style={{ color: isPos ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                      {isPos ? '+' : ''}{ret.toFixed(1)}%
                    </p>
                    <div className="mt-1 flex items-center justify-center gap-1">
                      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: dotColor }} />
                      <span className="text-[8px] font-semibold uppercase" style={{ color: dotColor }}>
                        {sig.label || 'HOLD'}
                      </span>
                    </div>
                    <p className="text-[9px] mt-0.5 tabular-nums" style={{ color: 'var(--text-muted)' }}>
                      {(sig.p_up * 100).toFixed(0)}% up
                    </p>
                  </div>
                );
              })}
            </div>

            {/* Main price chart with SMA / Bollinger / Forecast */}
            {ohlcvLoading && (
              <div className="flex items-center justify-center py-10 gap-2">
                <Loader2 className="w-4 h-4 animate-spin" style={{ color: 'var(--accent-violet)' }} />
                <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>Loading chart...</span>
              </div>
            )}
            {!ohlcvLoading && !hasOhlcv && (
              <div className="flex items-center justify-center py-6">
                <span className="text-[11px]" style={{ color: 'var(--text-muted)' }}>No chart data available for {ticker}</span>
              </div>
            )}
            {!ohlcvLoading && hasOhlcv && (
              <MiniPriceChart
                ohlcv={ohlcvQ.data!.data}
                indicators={indQ.data?.indicators ?? null}
                forecast={forecastQ.data ?? null}
                height={340}
              />
            )}

            {/* Zone charts */}
            {!ohlcvLoading && hasOhlcv && hasForecast && (
              <div className="mt-3">
                <BuySellZoneCharts
                  ohlcv={ohlcvQ.data!.data}
                  forecasts={forecastQ.data!.forecasts}
                  symbol={ticker}
                  compact
                />
              </div>
            )}
          </div>
        </div>
      </td>
    </tr>
  );
}

/* ── Cell styling ───────────────────────────────────────────────── */
function heatColor(expRet: number | null | undefined): string {
  if (expRet == null) return 'var(--void-surface)';
  const c = Math.max(-0.10, Math.min(0.10, expRet));
  const mag = Math.abs(c) / 0.10;
  if (Math.abs(c) < 0.003) return 'var(--void-surface)';
  return c > 0
    ? `rgba(62,232,165,${(0.06 + mag * 0.50).toFixed(3)})`
    : `rgba(255,107,138,${(0.06 + mag * 0.50).toFixed(3)})`;
}

function signalMatchesFilter(row: SummaryRow, filter: SignalFilter): boolean {
  if (filter === 'all') return true;
  const label = (row.nearest_label || '').toUpperCase().replace(/[\s-]/g, '_');
  return label === filter.toUpperCase();
}

/* ── Sector Sentiment Micro-Bar ─────────────────────────────────── */
function SentimentStrip({ sector }: { sector: SectorGroup }) {
  const total = sector.asset_count || 1;
  const segs = [
    { pct: (sector.strong_sell / total) * 100, c: 'var(--accent-rose)' },
    { pct: (sector.sell / total) * 100, c: 'rgba(255,107,138,0.35)' },
    { pct: (sector.hold / total) * 100, c: 'var(--violet-10)' },
    { pct: (sector.buy / total) * 100, c: 'rgba(62,232,165,0.35)' },
    { pct: (sector.strong_buy / total) * 100, c: 'var(--accent-emerald)' },
  ];
  return (
    <div className="flex h-[12px] rounded-md overflow-hidden" style={{ width: 80, background: 'var(--void-active)', boxShadow: '0 2px 8px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.04)' }}>
      {segs.map((s, i) => s.pct > 0 ? (
        <div key={i} className="h-full" style={{ width: `${s.pct}%`, background: s.c }} />
      ) : null)}
    </div>
  );
}

/* ── Summary Statistics Strip ──────────────────────────────────── */
function SummaryStrip({ sectors }: { sectors: SectorGroup[] }) {
  const stats = useMemo(() => {
    let totalAssets = 0, strongBuys = 0, buys = 0, holds = 0, sells = 0, strongSells = 0, exits = 0;
    for (const s of sectors) {
      totalAssets += s.asset_count;
      strongBuys += s.strong_buy;
      buys += s.buy;
      holds += s.hold;
      sells += s.sell;
      strongSells += s.strong_sell;
      exits += s.exit;
    }
    return { totalAssets, strongBuys, buys, holds, sells, strongSells, exits,
             sectors: sectors.length, active: strongBuys + buys + sells + strongSells };
  }, [sectors]);

  const items = [
    { label: 'Assets', value: stats.totalAssets, color: 'var(--text-primary)' },
    { label: 'Sectors', value: stats.sectors, color: 'var(--accent-violet)' },
    { label: 'Strong Buy', value: stats.strongBuys, color: 'var(--accent-emerald)' },
    { label: 'Buy', value: stats.buys, color: 'rgba(62,232,165,0.7)' },
    { label: 'Hold', value: stats.holds, color: 'var(--text-muted)' },
    { label: 'Sell', value: stats.sells, color: 'rgba(255,107,138,0.7)' },
    { label: 'Strong Sell', value: stats.strongSells, color: 'var(--accent-rose)' },
  ];

  return (
    <div className="flex items-center gap-1.5 flex-wrap">
      {items.map((it, idx) => (
        <div key={it.label}
          className="glass-card hover-lift flex items-center gap-1.5 px-3 py-2 rounded-lg"
          style={{ animationDelay: `${idx * 50}ms` }}>
          <span className="text-label">{it.label}</span>
          <span className="text-stat-value tabular-nums" style={{ color: it.color, fontSize: 13 }}>{it.value}</span>
        </div>
      ))}
    </div>
  );
}

/* ── Tooltip ───────────────────────────────────────────────────── */
interface TooltipInfo {
  asset: string; horizon: string; expRet: number; pUp: number;
  kelly?: number; label: string; x: number; y: number;
  momentum: number; crashRisk: number;
}

function HeatTooltip({ info }: { info: TooltipInfo }) {
  const labelColors: Record<string, { bg: string; fg: string }> = {
    STRONG_BUY: { bg: 'linear-gradient(135deg, #064e3b, #047857)', fg: 'var(--accent-emerald)' },
    BUY: { bg: 'linear-gradient(135deg, #064e3b, #065f46)', fg: '#6EE7B7' },
    HOLD: { bg: 'linear-gradient(135deg, #1e1b4b, #312e81)', fg: '#A5B4FC' },
    SELL: { bg: 'linear-gradient(135deg, #4c0519, #881337)', fg: '#FDA4AF' },
    STRONG_SELL: { bg: 'linear-gradient(135deg, #4c0519, #9f1239)', fg: 'var(--accent-rose)' },
  };
  const labelKey = (info.label || 'HOLD').toUpperCase().replace(/[\s-]/g, '_');
  const lc = labelColors[labelKey] || labelColors.HOLD;

  return (
    <div
      className="absolute z-50 pointer-events-none"
      style={{ left: info.x, top: info.y, transform: 'translate(-50%, -100%)' }}
    >
      <div
        className="rounded-2xl px-5 py-4 min-w-[220px]"
        style={{
          background: 'linear-gradient(160deg, rgba(26,5,51,0.97) 0%, rgba(13,27,62,0.97) 40%, rgba(10,37,64,0.97) 100%)',
          border: '1px solid var(--violet-30)',
          backdropFilter: 'blur(24px)',
          boxShadow: '0 12px 48px rgba(0,0,0,0.5), 0 0 80px var(--violet-8)',
        }}
      >
        {/* Asset + Horizon */}
        <div className="text-[12px] font-semibold mb-2" style={{ color: '#e2e8f0' }}>
          {info.asset}
          <span className="ml-2 px-1.5 py-0.5 rounded text-[9px] font-medium"
            style={{ background: 'var(--violet-12)', color: 'var(--text-violet)' }}>
            {info.horizon}
          </span>
        </div>

        {/* Expected return - hero number */}
        <div className="mb-3">
          <span
            className="text-[22px] font-black tabular-nums tracking-tight"
            style={{
              background: info.expRet >= 0
                ? 'linear-gradient(135deg, var(--text-luminous) 0%, var(--accent-emerald) 100%)'
                : 'linear-gradient(135deg, var(--text-luminous) 0%, var(--accent-rose) 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
            }}
          >
            {info.expRet >= 0 ? '+' : ''}{(info.expRet * 100).toFixed(2)}%
          </span>
        </div>

        {/* Metrics row */}
        <div className="flex items-center gap-4 mb-2">
          {/* p(up) radial */}
          <div className="flex items-center gap-1.5">
            <svg width="20" height="20" viewBox="0 0 20 20">
              <circle cx="10" cy="10" r="8" fill="none"
                stroke="var(--violet-12)" strokeWidth="2"
                strokeDasharray={`${Math.PI * 16 * 0.75} ${Math.PI * 16}`}
                transform="rotate(135 10 10)" strokeLinecap="round" />
              <circle cx="10" cy="10" r="8" fill="none"
                stroke={info.pUp >= 0.5 ? 'var(--accent-emerald)' : 'var(--accent-rose)'}
                strokeWidth="2"
                strokeDasharray={`${Math.PI * 16 * 0.75} ${Math.PI * 16}`}
                strokeDashoffset={Math.PI * 16 * 0.75 * (1 - info.pUp)}
                transform="rotate(135 10 10)" strokeLinecap="round" />
            </svg>
            <div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>P(up)</div>
              <div className="text-[12px] font-bold tabular-nums"
                style={{ color: info.pUp >= 0.5 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                {(info.pUp * 100).toFixed(0)}%
              </div>
            </div>
          </div>

          {/* Kelly */}
          {info.kelly != null && (
            <div>
              <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Kelly</div>
              <div className="text-[12px] font-bold tabular-nums" style={{ color: 'var(--accent-violet)' }}>
                {(info.kelly * 100).toFixed(0)}%
              </div>
            </div>
          )}

          {/* Momentum */}
          <div>
            <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Mom</div>
            <div className="text-[12px] font-bold tabular-nums"
              style={{ color: info.momentum > 0 ? 'var(--accent-emerald)' : info.momentum < 0 ? 'var(--accent-rose)' : 'var(--text-muted)' }}>
              {info.momentum > 0 ? '+' : ''}{Math.round(info.momentum)}%
            </div>
          </div>
        </div>

        {/* Signal label pill */}
        <span
          className="inline-block px-2.5 py-1 rounded-lg text-[10px] font-semibold tracking-wide"
          style={{ background: lc.bg, color: lc.fg }}
        >
          {info.label}
        </span>
      </div>
    </div>
  );
}

/* ── Color Scale Legend ─────────────────────────────────────────── */
function ColorScaleLegend() {
  return (
    <div className="flex items-center gap-3">
      <span className="text-[10px] font-medium tabular-nums" style={{ color: 'var(--accent-rose)' }}>-10%</span>
      <div className="relative" style={{ width: 160, height: 10 }}>
        <div className="absolute inset-0 rounded-full overflow-hidden flex"
          style={{ background: 'var(--void-surface)' }}>
          <div className="h-full w-1/2" style={{
            background: 'linear-gradient(90deg, rgba(255,107,138,0.55) 0%, var(--rose-6) 85%, transparent 100%)',
          }} />
          <div className="h-full w-1/2" style={{
            background: 'linear-gradient(90deg, transparent 0%, var(--emerald-6) 15%, rgba(62,232,165,0.55) 100%)',
          }} />
        </div>
        {/* Center tick */}
        <div className="absolute top-0 left-1/2 -translate-x-1/2 w-px h-full" style={{ background: 'var(--violet-30)' }} />
      </div>
      <span className="text-[10px] font-medium tabular-nums" style={{ color: 'var(--accent-emerald)' }}>+10%</span>
    </div>
  );
}

/* ── Main Page ──────────────────────────────────────────────────── */
export default function HeatmapPage() {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);

  /* Data fetching */
  const sectorQ = useQuery({
    queryKey: ['signalsBySector'],
    queryFn: api.signalsBySector,
    staleTime: 120_000,
  });
  const summaryQ = useQuery({
    queryKey: ['signalSummary'],
    queryFn: api.signalSummary,
    staleTime: 120_000,
  });

  /* State */
  const [search, setSearch] = useState('');
  const [filter, setFilter] = useState<SignalFilter>(() => {
    try { return (localStorage.getItem(FILTER_KEY) as SignalFilter) || 'all'; } catch { return 'all'; }
  });
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    try {
      const raw = localStorage.getItem(COLLAPSE_KEY);
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch { return new Set(); }
  });
  const [focusRow, setFocusRow] = useState(-1);
  const [focusCol, setFocusCol] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [tooltip, setTooltip] = useState<TooltipInfo | null>(null);
  const [flashCell, setFlashCell] = useState<string | null>(null);
  const [isFullWidth, setIsFullWidth] = useState(false);
  const [expandedAsset, setExpandedAsset] = useState<string | null>(null);
  const [sort, setSort] = useState<SortState | null>(null);

  /* Persist state */
  useEffect(() => {
    try { localStorage.setItem(COLLAPSE_KEY, JSON.stringify([...collapsed])); } catch { /* noop */ }
  }, [collapsed]);
  useEffect(() => {
    try { localStorage.setItem(FILTER_KEY, filter); } catch { /* noop */ }
  }, [filter]);

  const sectors = sectorQ.data?.sectors ?? [];
  const horizons = summaryQ.data?.horizons ?? [];

  /* Filter + search + sort */
  const filteredSectors = useMemo(() => {
    const q = search.toLowerCase().trim();
    return sectors.map(s => {
      let assets = s.assets;
      if (filter !== 'all') {
        assets = assets.filter(a => signalMatchesFilter(a, filter));
      }
      if (q) {
        assets = assets.filter(a => a.asset_label.toLowerCase().includes(q) || (a.sector || '').toLowerCase().includes(q));
      }
      if (sort) {
        assets = [...assets].sort((a, b) => {
          const va = getSortValue(a, sort.key);
          const vb = getSortValue(b, sort.key);
          return sort.dir === 'desc' ? vb - va : va - vb;
        });
      }
      return { ...s, assets, asset_count: assets.length };
    }).filter(s => s.assets.length > 0);
  }, [sectors, search, filter, sort]);

  /* Flatten for keyboard navigation */
  const flatRows = useMemo(() => {
    const result: { type: 'sector' | 'asset'; label: string; row?: SummaryRow; sectorName?: string }[] = [];
    for (const s of filteredSectors) {
      result.push({ type: 'sector', label: s.name });
      if (!collapsed.has(s.name)) {
        for (const a of s.assets) {
          result.push({ type: 'asset', label: a.asset_label, row: a, sectorName: s.name });
        }
      }
    }
    return result;
  }, [filteredSectors, collapsed]);

  const toggleSector = useCallback((name: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }, []);

  const expandAll = useCallback(() => setCollapsed(new Set()), []);
  const collapseAll = useCallback(() => {
    setCollapsed(new Set(filteredSectors.map(s => s.name)));
  }, [filteredSectors]);

  /* Keyboard navigation */
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      if (target.tagName === 'INPUT') return;
      if (!el.contains(document.activeElement) && document.activeElement !== el) return;

      switch (e.key) {
        case 'j': case 'ArrowDown':
          e.preventDefault();
          setFocusRow(r => Math.min(r + 1, flatRows.length - 1));
          break;
        case 'k': case 'ArrowUp':
          e.preventDefault();
          setFocusRow(r => Math.max(r - 1, 0));
          break;
        case 'l': case 'ArrowRight':
          e.preventDefault();
          setFocusCol(c => Math.min(c + 1, horizons.length - 1));
          break;
        case 'h': case 'ArrowLeft':
          e.preventDefault();
          setFocusCol(c => Math.max(c - 1, 0));
          break;
        case 'Enter': {
          const item = flatRows[focusRow];
          if (item?.type === 'asset') {
            setExpandedAsset(prev => prev === item.label ? null : item.label);
          } else if (item?.type === 'sector') {
            toggleSector(item.label);
          }
          break;
        }
        case 'o': {
          const item = flatRows[focusRow];
          if (item?.type === 'asset') navigate(`/charts/${extractTicker(item.label)}`);
          break;
        }
        case 'Escape':
          if (expandedAsset) {
            setExpandedAsset(null);
          } else {
            setFocusRow(-1);
            el.blur();
          }
          break;
        case '/':
          e.preventDefault();
          document.getElementById('heatmap-search')?.focus();
          break;
      }
    };
    el.addEventListener('keydown', handler);
    return () => el.removeEventListener('keydown', handler);
  }, [flatRows, focusRow, focusCol, horizons.length, navigate, toggleSector, expandedAsset]);

  /* Cell handlers */
  const handleCellHover = useCallback((asset: SummaryRow, hKey: string, hLabel: string, e: React.MouseEvent, rIdx: number, cIdx: number) => {
    setHoveredCell({ row: rIdx, col: cIdx });
    const sig = asset.horizon_signals[hKey] || asset.horizon_signals[String(hKey)];
    if (!sig) return;
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const cRect = containerRef.current?.getBoundingClientRect();
    setTooltip({
      asset: asset.asset_label, horizon: hLabel,
      expRet: sig.exp_ret, pUp: sig.p_up,
      kelly: sig.kelly_half, label: sig.label || '',
      momentum: asset.momentum_score, crashRisk: asset.crash_risk_score,
      x: rect.left - (cRect?.left ?? 0) + rect.width / 2,
      y: rect.top - (cRect?.top ?? 0) - 8,
    });
  }, []);

  const handleAssetClick = useCallback((assetLabel: string) => {
    setExpandedAsset(prev => prev === assetLabel ? null : assetLabel);
  }, []);

  const handleCellClick = useCallback((assetLabel: string, cellKey: string) => {
    setFlashCell(cellKey);
    setTimeout(() => setFlashCell(null), 180);
    setExpandedAsset(prev => prev === assetLabel ? null : assetLabel);
  }, []);

  /* Loading / Error */
  if (sectorQ.isLoading || summaryQ.isLoading) return <DashboardSkeleton />;
  if (sectorQ.error) return <CosmicErrorCard title="Failed to load sector data" error={sectorQ.error as Error} onRetry={() => sectorQ.refetch()} />;
  if (!sectorQ.data?.sectors?.length) return <DashboardEmpty />;

  let globalRowIdx = -1;

  return (
    <>
      <PageHeader title="Signal Heatmap" action={
        <div className="flex items-center gap-2">
          <button
            onClick={() => setIsFullWidth(f => !f)}
            className="p-2 rounded-lg transition-colors"
            style={{ background: 'var(--void-surface)', color: 'var(--text-secondary)' }}
            title={isFullWidth ? 'Normal width' : 'Full width'}
          >
            {isFullWidth ? <Minimize2 className="w-4 h-4" /> : <Maximize2 className="w-4 h-4" />}
          </button>
        </div>
      }>
        Star-map of expected returns across all assets and horizons.
        Use <kbd className="px-1.5 py-0.5 rounded text-[10px] font-mono" style={{ background: 'var(--void-surface)', color: 'var(--text-violet)' }}>j/k</kbd> to navigate,
        <kbd className="px-1.5 py-0.5 rounded text-[10px] font-mono ml-1" style={{ background: 'var(--void-surface)', color: 'var(--text-violet)' }}>/</kbd> to search,
        <kbd className="px-1.5 py-0.5 rounded text-[10px] font-mono ml-1" style={{ background: 'var(--void-surface)', color: 'var(--text-violet)' }}>Enter</kbd> to expand zones,
        <kbd className="px-1.5 py-0.5 rounded text-[10px] font-mono ml-1" style={{ background: 'var(--void-surface)', color: 'var(--text-violet)' }}>o</kbd> to open chart.
      </PageHeader>

      {/* Summary strip */}
      <div className="mb-4 fade-up-delay-1">
        <SummaryStrip sectors={sectors} />
      </div>

      {/* Controls bar */}
      <div className="mb-4 flex items-center gap-3 flex-wrap fade-up-delay-2">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px] max-w-[360px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4" style={{ color: 'var(--text-muted)' }} />
          <input
            id="heatmap-search"
            type="text"
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Search assets or sectors..."
            className="w-full pl-9 pr-8 py-2.5 rounded-xl text-[12px] outline-none transition-all focus-ring"
            style={{
              background: 'var(--void-surface)',
              border: '1px solid var(--border-void)',
              color: 'var(--text-primary)',
              backdropFilter: 'blur(8px)',
            }}
            onFocus={e => (e.target.style.borderColor = 'rgba(139,92,246,0.35)')}
            onBlur={e => (e.target.style.borderColor = 'var(--border-void)')}
          />
          {search && (
            <button
              onClick={() => setSearch('')}
              className="absolute right-2.5 top-1/2 -translate-y-1/2"
              style={{ color: 'var(--text-muted)' }}
            >
              <X className="w-3.5 h-3.5" />
            </button>
          )}
        </div>

        {/* Signal filter pills */}
        <div className="flex items-center gap-1">
          <Filter className="w-3.5 h-3.5 mr-1" style={{ color: 'var(--text-muted)' }} />
          {FILTER_OPTIONS.map(opt => (
            <button
              key={opt.value}
              onClick={() => setFilter(opt.value)}
              className="filter-pill"
              data-active={filter === opt.value || undefined}
              data-filter={opt.value}
            >
              {opt.label}
            </button>
          ))}
        </div>

        {/* Expand/Collapse */}
        <div className="flex items-center gap-1 ml-auto">
          <button onClick={expandAll}
            className="px-2.5 py-1 rounded-lg text-[10px] font-medium transition-colors"
            style={{ background: 'var(--void-surface)', color: 'var(--text-secondary)' }}>
            Expand All
          </button>
          <button onClick={collapseAll}
            className="px-2.5 py-1 rounded-lg text-[10px] font-medium transition-colors"
            style={{ background: 'var(--void-surface)', color: 'var(--text-secondary)' }}>
            Collapse All
          </button>
        </div>

        {/* Color legend */}
        <ColorScaleLegend />
      </div>

      {/* Main heatmap */}
      <div
        ref={containerRef}
        tabIndex={0}
        className={`glass-card overflow-hidden relative fade-up-delay-3 ${isFullWidth ? '-mx-6' : ''}`}
        style={{ outline: 'none' }}
      >
        {/* Tooltip */}
        {tooltip && <HeatTooltip info={tooltip} />}

        {/* No results */}
        {filteredSectors.length === 0 && (
          <div className="flex flex-col items-center justify-center py-16 gap-3">
            <Search className="w-8 h-8" style={{ color: 'var(--text-muted)' }} />
            <span className="text-[13px]" style={{ color: 'var(--text-muted)' }}>
              No assets match "{search}" with filter "{filter}"
            </span>
            <button onClick={() => { setSearch(''); setFilter('all'); }}
              className="text-[11px] px-3 py-1.5 rounded-lg transition-colors"
              style={{ background: 'var(--violet-12)', color: 'var(--text-violet)' }}>
              Clear filters
            </button>
          </div>
        )}

        {filteredSectors.length > 0 && (
          <div className="overflow-x-auto">
            <table className="w-full border-collapse" style={{ minWidth: 600 }}>
              <thead>
                <tr>
                  <th className="text-left px-4 py-3 font-semibold text-[11px]"
                    style={{
                      color: 'var(--text-muted)',
                      background: 'var(--void)',
                      position: 'sticky', left: 0, zIndex: 20,
                      borderBottom: '1px solid var(--violet-10)',
                      width: 220,
                    }}>
                    Asset
                  </th>
                  {horizons.map(h => {
                    const isActive = sort?.key === h;
                    return (
                      <th key={h}
                        className="text-center px-1 py-3 font-semibold text-[10px] cursor-pointer select-none group/sort"
                        style={{
                          color: isActive ? 'var(--accent-violet)' : 'var(--text-muted)',
                          background: isActive ? 'var(--violet-4)' : 'var(--void)',
                          borderBottom: '1px solid var(--violet-10)',
                          minWidth: 56,
                          transition: 'color 0.15s, background 0.15s',
                        }}
                        onClick={() => setSort(prev =>
                          prev?.key === h
                            ? prev.dir === 'desc' ? { key: h, dir: 'asc' } : null
                            : { key: h, dir: 'desc' }
                        )}
                        title={`Sort by ${formatHorizon(h)}`}
                      >
                        <div className="flex items-center justify-center gap-0.5">
                          {formatHorizon(h)}
                          {isActive
                            ? (sort.dir === 'desc'
                                ? <ChevronDown className="w-3 h-3" />
                                : <ChevronUp className="w-3 h-3" />)
                            : <ArrowUpDown className="w-2.5 h-2.5 opacity-0 group-hover/sort:opacity-40 transition-opacity" />
                          }
                        </div>
                      </th>
                    );
                  })}
                  {(() => {
                    const isActive = sort?.key === 'momentum';
                    return (
                      <th
                        className="text-center px-2 py-3 font-semibold text-[11px] cursor-pointer select-none group/sort"
                        style={{
                          color: isActive ? 'var(--accent-violet)' : 'var(--text-muted)',
                          background: isActive ? 'var(--violet-4)' : 'var(--void)',
                          borderBottom: '1px solid var(--violet-10)',
                          minWidth: 56,
                          transition: 'color 0.15s, background 0.15s',
                        }}
                        onClick={() => setSort(prev =>
                          prev?.key === 'momentum'
                            ? prev.dir === 'desc' ? { key: 'momentum', dir: 'asc' } : null
                            : { key: 'momentum', dir: 'desc' }
                        )}
                        title="Sort by Momentum"
                      >
                        <div className="flex items-center justify-center gap-0.5">
                          Mom
                          {isActive
                            ? (sort.dir === 'desc'
                                ? <ChevronDown className="w-3 h-3" />
                                : <ChevronUp className="w-3 h-3" />)
                            : <ArrowUpDown className="w-2.5 h-2.5 opacity-0 group-hover/sort:opacity-40 transition-opacity" />
                          }
                        </div>
                      </th>
                    );
                  })()}
                </tr>
              </thead>
              {filteredSectors.map(sector => {
                const isCollapsed = collapsed.has(sector.name);
                globalRowIdx++;
                const sectorRowIdx = globalRowIdx;
                const avgMom = sector.avg_momentum ?? 0;

                return (
                  <tbody key={sector.name}>
                    {/* Sector header row */}
                    <tr
                      className="cursor-pointer group"
                      onClick={() => toggleSector(sector.name)}
                      style={{
                        background: sectorRowIdx === focusRow
                          ? 'var(--violet-8)'
                          : 'var(--void-hover)',
                      }}
                    >
                      <td
                        className="px-4 py-2.5 whitespace-nowrap"
                        colSpan={horizons.length + 2}
                        style={{ borderBottom: '1px solid var(--violet-6)' }}
                      >
                        <div className="flex items-center gap-3">
                          {/* Chevron */}
                          <span className="transition-transform" style={{ display: 'flex' }}>
                            {isCollapsed
                              ? <ChevronRight className="w-4 h-4" style={{ color: 'var(--text-muted)' }} />
                              : <ChevronDown className="w-4 h-4" style={{ color: 'var(--accent-violet)' }} />
                            }
                          </span>

                          {/* Sector name */}
                          <span className="text-[12px] font-semibold" style={{ color: 'var(--text-violet)' }}>
                            {sector.name}
                          </span>

                          {/* Asset count */}
                          <span
                            className="px-1.5 py-0.5 rounded-md text-[9px] font-medium"
                            style={{ background: 'var(--violet-10)', color: 'var(--text-secondary)' }}
                          >
                            {sector.asset_count}
                          </span>

                          {/* Sentiment bar */}
                          <SentimentStrip sector={sector} />

                          {/* Buy / Sell counts */}
                          {sector.strong_buy > 0 && (
                            <span className="flex items-center gap-0.5 text-[10px] font-medium" style={{ color: 'var(--accent-emerald)' }}>
                              <TrendingUp className="w-3 h-3" />{sector.strong_buy + sector.buy}
                            </span>
                          )}
                          {sector.strong_sell > 0 && (
                            <span className="flex items-center gap-0.5 text-[10px] font-medium" style={{ color: 'var(--accent-rose)' }}>
                              <TrendingDown className="w-3 h-3" />{sector.strong_sell + sector.sell}
                            </span>
                          )}

                          {/* Average momentum */}
                          <span className="text-[10px] tabular-nums font-semibold ml-auto"
                            style={{ color: avgMom > 5 ? 'var(--accent-emerald)' : avgMom < -5 ? 'var(--accent-rose)' : 'var(--text-muted)' }}>
                            {avgMom > 0 ? '+' : ''}{avgMom.toFixed(1)}
                          </span>
                        </div>
                      </td>
                    </tr>

                    {/* Asset rows */}
                    {!isCollapsed && sector.assets.map(asset => {
                      globalRowIdx++;
                      const aIdx = globalRowIdx;
                      const isFocusedRow = aIdx === focusRow;
                      const mom = asset.momentum_score ?? 0;

                      return (
                        <React.Fragment key={asset.asset_label}>
                          <tr
                            className="transition-colors"
                            style={{
                              borderBottom: '1px solid var(--violet-3)',
                              background: isFocusedRow ? 'var(--violet-6)' : 'transparent',
                            }}
                          >
                            {/* Asset name - sticky, click to expand */}
                            <td
                              className="px-4 py-1 whitespace-nowrap cursor-pointer group/name"
                              style={{
                                position: 'sticky', left: 0, zIndex: 10,
                                background: expandedAsset === asset.asset_label
                                  ? 'var(--violet-8)'
                                  : isFocusedRow ? 'rgba(3,0,20,0.95)' : 'var(--void)',
                              }}
                              onClick={() => handleAssetClick(asset.asset_label)}
                            >
                              <div className="flex items-center gap-1.5">
                                <span className="transition-transform" style={{ display: 'flex' }}>
                                  {expandedAsset === asset.asset_label
                                    ? <ChevronDown className="w-3 h-3" style={{ color: 'var(--accent-violet)' }} />
                                    : <ChevronRight className="w-3 h-3" style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
                                  }
                                </span>
                                <span
                                  className="text-[11px] transition-colors"
                                  style={{
                                    color: expandedAsset === asset.asset_label
                                      ? 'var(--text-violet)'
                                      : isFocusedRow ? 'var(--text-luminous)' : 'var(--text-primary)',
                                  }}
                                >
                                  {asset.asset_label}
                                </span>
                                {(() => {
                                  const badge = getSignalBadge(asset.nearest_label);
                                  if (!badge) return null;
                                  return (
                                    <span
                                      className="px-1.5 py-[1px] rounded text-[8px] font-semibold uppercase tracking-wide whitespace-nowrap"
                                      style={{ background: badge.bg, color: badge.fg }}
                                    >
                                      {badge.text}
                                    </span>
                                  );
                                })()}
                              </div>
                            </td>

                            {/* Heat cells */}
                            {horizons.map((h, ci) => {
                              const sig = asset.horizon_signals[h] || asset.horizon_signals[String(h)];
                              const bg = heatColor(sig?.exp_ret);
                              const isFocused = isFocusedRow && ci === focusCol;
                              const isHovered = hoveredCell?.row === aIdx && hoveredCell?.col === ci;
                              const cellKey = `${asset.asset_label}-${h}`;
                              const isFlashing = flashCell === cellKey;

                              return (
                                <td key={h} className="text-center px-0.5 py-[2px] cursor-pointer"
                                  onClick={() => handleCellClick(asset.asset_label, cellKey)}
                                  onMouseEnter={e => handleCellHover(asset, String(h), formatHorizon(h), e, aIdx, ci)}
                                  onMouseLeave={() => { setHoveredCell(null); setTooltip(null); }}
                                >
                                  <div
                                    className="rounded-[4px] transition-all duration-100"
                                    style={{
                                      background: bg,
                                      height: 26,
                                      display: 'flex',
                                      alignItems: 'center',
                                      justifyContent: 'center',
                                      border: `1px solid ${
                                        isFocused ? 'rgba(139,92,246,0.35)'
                                          : isHovered ? 'var(--violet-20)'
                                          : 'var(--violet-3)'
                                      }`,
                                      boxShadow: isFlashing
                                        ? '0 0 20px rgba(139,92,246,0.5), inset 0 0 8px var(--violet-30)'
                                        : isHovered
                                          ? '0 0 16px var(--violet-15)'
                                          : isFocused
                                            ? '0 0 8px var(--violet-12)'
                                            : 'none',
                                      transform: isHovered ? 'scale(1.12)' : 'scale(1)',
                                    }}
                                  >
                                    <span
                                      className="text-[9px] tabular-nums font-medium"
                                      style={{
                                        color: sig?.exp_ret != null
                                          ? (Math.abs(sig.exp_ret) > 0.02 ? 'var(--text-luminous)' : 'var(--text-secondary)')
                                          : 'var(--text-muted)',
                                        opacity: sig?.exp_ret != null && Math.abs(sig.exp_ret) < 0.005 ? 0.4 : 1,
                                      }}
                                    >
                                      {sig?.exp_ret != null ? `${(sig.exp_ret * 100).toFixed(1)}` : '\u2014'}
                                    </span>
                                  </div>
                                </td>
                              );
                            })}

                            {/* Momentum column */}
                            <td className="text-center px-1 py-[2px]">
                              <span
                                className="text-[10px] tabular-nums font-medium"
                                style={{
                                  color: mom > 30 ? 'var(--accent-emerald)' : mom > 0 ? 'rgba(62,232,165,0.7)'
                                    : mom < -30 ? 'var(--accent-rose)' : mom < 0 ? 'rgba(255,107,138,0.7)'
                                    : 'var(--text-muted)',
                                }}
                              >
                                {mom > 0 ? '+' : ''}{Math.round(mom)}%
                              </span>
                            </td>
                          </tr>

                          {/* Expanded zone charts row */}
                          {expandedAsset === asset.asset_label && (
                            <ExpandedAssetRow
                              assetLabel={asset.asset_label}
                              colSpan={horizons.length + 2}
                              onClose={() => setExpandedAsset(null)}
                              asset={asset}
                              horizons={horizons}
                            />
                          )}
                        </React.Fragment>
                      );
                    })}
                  </tbody>
                );
              })}
            </table>
          </div>
        )}

        {/* Bottom status bar */}
        <div
          className="flex items-center justify-between px-5 py-2"
          style={{ borderTop: '1px solid var(--violet-8)', background: 'var(--void)' }}
        >
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            {filteredSectors.reduce((a, s) => a + s.assets.length, 0)} assets across {filteredSectors.length} sectors
            {search && <> matching "<span style={{ color: 'var(--text-violet)' }}>{search}</span>"</>}
          </span>
          <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
            {horizons.length} horizons
          </span>
        </div>
      </div>
    </>
  );
}
