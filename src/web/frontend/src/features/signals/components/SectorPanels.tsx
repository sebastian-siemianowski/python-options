import React, { useMemo, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Activity, ArrowDown, ArrowUp, ChevronRight, Filter, Layers, Shield, Target, TrendingUp } from 'lucide-react';
import type { ReversalFlipsData, SectorGroup, SummaryRow } from '../../../api';
import type { ColumnDef } from '../../../components/ColumnCustomizer';
import { formatHorizon } from '../../../utils/horizons';
import ChartAssetRow from './ChartAssetRow';
import { SectorSignalRow } from './AllAssetsTable';
import { signalLabelColor } from '../theme';
import {
  extractTicker,
  isReversalQuickFilter,
  rowHorizonColor,
  rowMatchesReversalFilter,
  type SignalFilter,
  type SortColumn,
  type SortDir,
} from '../utils';

/* ── Sector Panels — Premium Redesign ─────────────────────────────── */
export type SectorSortBy = 'momentum' | 'exp_ret' | 'signal' | 'count' | 'alpha';
export type SectorRowSortColumn = SortColumn | 'strength';
export const SECTOR_SORT_OPTIONS: { key: SectorSortBy; label: string; icon: React.ReactNode }[] = [
  { key: 'momentum', label: 'Momentum', icon: <TrendingUp className="w-3 h-3" /> },
  { key: 'signal', label: 'Signal Score', icon: <Target className="w-3 h-3" /> },
  { key: 'count', label: 'Asset Count', icon: <Layers className="w-3 h-3" /> },
  { key: 'alpha', label: 'Alphabetical', icon: <Filter className="w-3 h-3" /> },
];
const SECTOR_DISPLAY_PRIORITY: Record<string, number> = {
  'Core Markets, Commodities & Crypto': 0,
  Currencies: 1,
};

function sectorDisplayPriority(name: string): number {
  return SECTOR_DISPLAY_PRIORITY[name] ?? 50;
}


export const SECTOR_COLUMN_DEFS: ColumnDef[] = [
  { key: 'asset', label: 'Asset', locked: true },
  { key: 'chart', label: 'Chart' },
  { key: 'pct30d', label: '30D change', hint: '%' },
  { key: 'signal', label: 'Signal', locked: true },
  { key: 'strength', label: 'Strength' },
  { key: 'momentum', label: 'Momentum' },
  { key: 'quality', label: 'Quality' },
  { key: 'risk', label: 'Crash risk' },
  { key: 'horizons', label: 'Horizons' },
];
export const SECTOR_COLS_LS_KEY = 'signals-sector-cols-v2';
export const DEFAULT_SECTOR_VISIBLE_COLS = new Set(SECTOR_COLUMN_DEFS.map((c) => c.key));

export function loadSectorVisibleCols(): Set<string> {
  try {
    const raw = localStorage.getItem(SECTOR_COLS_LS_KEY);
    if (!raw) return new Set(DEFAULT_SECTOR_VISIBLE_COLS);
    const parsed = JSON.parse(raw) as string[];
    const set = new Set(parsed);
    SECTOR_COLUMN_DEFS.forEach((c) => { if (c.locked) set.add(c.key); });
    return set;
  } catch {
    return new Set(DEFAULT_SECTOR_VISIBLE_COLS);
  }
}

export default function SectorPanels({
  sectors,
  expandedSectors,
  toggleSector,
  sectorSort,
  rowSortCol,
  rowSortDir,
  onRowSort,
  sectorVisibleCols,
  sectorChartView,
  horizons,
  search,
  filter,
  reversalFlips,
  updatedAsset,
  qualityScores,
}: {
  sectors: SectorGroup[];
  expandedSectors: Set<string>;
  toggleSector: (name: string) => void;
  sectorSort: SectorSortBy;
  rowSortCol: SectorRowSortColumn;
  rowSortDir: SortDir;
  onRowSort: (col: SectorRowSortColumn) => void;
  sectorVisibleCols: Set<string>;
  sectorChartView: boolean;
  horizons: number[];
  search: string;
  filter: SignalFilter;
  reversalFlips?: ReversalFlipsData;
  updatedAsset: string | null;
  qualityScores: Record<string, number>;
}) {
  const navigate = useNavigate();
  const [chartExpandedRow, setChartExpandedRow] = useState<string | null>(null);

  const applyRowSort = (arr: SummaryRow[]) => {
    const signalRank: Record<string, number> = { 'STRONG BUY': 5, 'BUY': 4, 'HOLD': 3, 'SELL': 2, 'STRONG SELL': 1, 'EXIT': 0 };
    const getter = (r: SummaryRow): number | string => {
      const col = rowSortCol;
      if (col === 'asset') return r.asset_label;
      if (col === 'sector') return r.sector || '';
      if (col === 'signal') return signalRank[(r.nearest_label || 'HOLD').toUpperCase()] ?? 3;
      if (col === 'strength') return r.horizon_signals[Number(Object.keys(r.horizon_signals)[0])]?.p_up ?? 0.5;
      if (col === 'momentum') return r.momentum_score ?? 0;
      if (col === 'quality') return qualityScores[extractTicker(r.asset_label)] ?? 50;
      if (col === 'crash_risk') return r.crash_risk_score ?? 0;
      if (col.startsWith('horizon_')) {
        const h = parseInt(col.split('_')[1], 10);
        return r.horizon_signals[h]?.exp_ret ?? 0;
      }
      return 0;
    };
    const mult = rowSortDir === 'asc' ? 1 : -1;
    return [...arr].sort((a, b) => {
      const av = getter(a);
      const bv = getter(b);
      if (typeof av === 'string' && typeof bv === 'string') return av.localeCompare(bv) * mult;
      return (((av as number) - (bv as number)) || 0) * mult;
    });
  };
  const sortArrow = (col: SectorRowSortColumn) =>
    rowSortCol === col ? (rowSortDir === 'asc' ? ' \u2191' : ' \u2193') : '';
  const thSortClass = (col: SectorRowSortColumn) =>
    `cursor-pointer select-none hover:text-[var(--text-secondary)] transition-colors ${rowSortCol === col ? 'text-[var(--accent-violet)]' : ''}`;


  const sorted = useMemo(() => {
    const arr = [...sectors];
    const signalScore = (s: SectorGroup) => (s.strong_buy ?? 0) * 3 + (s.buy ?? 0) * 2 - (s.sell ?? 0) * 2 - (s.strong_sell ?? 0) * 3;
    const pinned = (a: SectorGroup, b: SectorGroup) => sectorDisplayPriority(a.name) - sectorDisplayPriority(b.name);
    switch (sectorSort) {
      case 'momentum': return arr.sort((a, b) => pinned(a, b) || (b.avg_momentum ?? 0) - (a.avg_momentum ?? 0));
      case 'signal': return arr.sort((a, b) => pinned(a, b) || signalScore(b) - signalScore(a));
      case 'count': return arr.sort((a, b) => pinned(a, b) || b.asset_count - a.asset_count);
      case 'alpha': return arr.sort((a, b) => pinned(a, b) || a.name.localeCompare(b.name));
      case 'exp_ret': return arr.sort((a, b) => pinned(a, b) || signalScore(b) - signalScore(a));
      default: return arr;
    }
  }, [sectors, sectorSort]);

  return (
    <div className="space-y-3">
      {/* Sort / Expand / Columns / totals all moved to the unified premium
          filter card in SignalsPageInner. SectorPanels now renders sector
          content only. */}

      {sorted.map((sector) => {
        const expanded = expandedSectors.has(sector.name);
        const matchesFilter = (lbl: string, row: SummaryRow) => {
          if (filter === 'all') return true;
          if (isReversalQuickFilter(filter)) return rowMatchesReversalFilter(row, filter, reversalFlips);
          if (filter === 'bullish') return lbl === 'STRONG_BUY' || lbl === 'BUY';
          if (filter === 'bearish') return lbl === 'STRONG_SELL' || lbl === 'SELL';
          if (filter === 'greens' || filter === 'reds') return rowHorizonColor(row) === filter;
          return lbl === filter.toUpperCase();
        };
        const assets = sector.assets.filter(row => {
          if (search && !row.asset_label.toLowerCase().includes(search.toLowerCase())) return false;
          const lbl = (row.nearest_label || '').toUpperCase().replace(/\s+/g, '_');
          return matchesFilter(lbl, row);
        });
        if (assets.length === 0) return null;

        const bullish = (sector.strong_buy ?? 0) + (sector.buy ?? 0);
        const bearish = (sector.strong_sell ?? 0) + (sector.sell ?? 0);
        const neutral = sector.hold ?? 0;
        const total = bullish + bearish + neutral;
        const sentiment = bullish > bearish ? 'bullish' : bearish > bullish ? 'bearish' : 'neutral';
        const sentColor = sentiment === 'bullish' ? '#10b981' : sentiment === 'bearish' ? '#f43f5e' : '#64748b';
        const sentGlow = sentiment === 'bullish' ? '0 0 20px rgba(16,185,129,0.08)' : sentiment === 'bearish' ? '0 0 20px rgba(244,63,94,0.08)' : 'none';

        // Best performing asset
        const bestAsset = [...sector.assets].sort((a, b) => {
          const aRet = Object.values(a.horizon_signals)[0]?.exp_ret ?? 0;
          const bRet = Object.values(b.horizon_signals)[0]?.exp_ret ?? 0;
          return bRet - aRet;
        })[0];
        const bestTicker = bestAsset ? extractTicker(bestAsset.asset_label) : null;
        const bestRet = bestAsset ? (Object.values(bestAsset.horizon_signals)[0]?.exp_ret ?? 0) * 100 : 0;
        const bestLabel = bestAsset ? (bestAsset.nearest_label || 'HOLD').toUpperCase() : '';

        // Sentiment bar proportions
        const strongBuyPct = total > 0 ? ((sector.strong_buy ?? 0) / total) * 100 : 0;
        const buyPct = total > 0 ? ((sector.buy ?? 0) / total) * 100 : 0;
        const holdPct = total > 0 ? ((sector.hold ?? 0) / total) * 100 : 0;
        const sellPct = total > 0 ? ((sector.sell ?? 0) / total) * 100 : 0;
        const strongSellPct = total > 0 ? ((sector.strong_sell ?? 0) / total) * 100 : 0;

        const avgMom = sector.avg_momentum ?? 0;
        const bullishPct = total > 0 ? Math.round((bullish / total) * 100) : 0;
        const sortedAssets = applyRowSort(assets);

        return (
          <div key={sector.name} className="glass-card overflow-hidden transition-all duration-200"
            style={{
              borderLeft: `3px solid ${sentColor}40`,
              boxShadow: expanded ? sentGlow : 'none',
            }}>
            {/* Sector Header — rich, informative */}
            <button
              onClick={() => toggleSector(sector.name)}
              className="w-full px-4 py-3 hover:bg-white/[0.015] transition-all duration-200 group"
            >
              {/* Top row: Name + key stats */}
              <div className="flex items-center gap-3">
                {/* Expand indicator */}
                <div className="w-5 h-5 rounded-md flex items-center justify-center flex-shrink-0 transition-all duration-200"
                  style={{ background: expanded ? `${sentColor}20` : 'var(--void-active)' }}>
                  <ChevronRight
                    className="w-3 h-3 transition-transform duration-200"
                    style={{ color: expanded ? sentColor : 'var(--text-muted)', transform: expanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
                  />
                </div>

                {/* Sector name */}
                <span className="font-semibold text-[13px] text-[#e2e8f0] whitespace-nowrap group-hover:text-white transition-colors">{sector.name}</span>

                {/* Asset count */}
                <span className="text-[10px] px-2 py-0.5 rounded-full font-medium tabular-nums"
                  style={{ background: `${sentColor}12`, color: sentColor }}>
                  {sector.asset_count}
                </span>

                {/* Sentiment bar — wider, more readable */}
                <div className="flex h-[5px] w-[100px] rounded-full overflow-hidden flex-shrink-0" style={{ background: 'var(--void-active)' }}>
                  <div className="transition-all duration-500" style={{ width: `${strongBuyPct}%`, background: '#10b981' }} />
                  <div className="transition-all duration-500" style={{ width: `${buyPct}%`, background: '#6ee7b7' }} />
                  <div className="transition-all duration-500" style={{ width: `${holdPct}%`, background: '#475569' }} />
                  <div className="transition-all duration-500" style={{ width: `${sellPct}%`, background: '#fca5a5' }} />
                  <div className="transition-all duration-500" style={{ width: `${strongSellPct}%`, background: '#f43f5e' }} />
                </div>

                {/* Bullish % */}
                <span className="text-[10px] font-bold tabular-nums" style={{ color: sentColor }}>
                  {bullishPct}%
                </span>

                {/* Signal counts — compact badges */}
                <div className="hidden md:flex items-center gap-1">
                  {(sector.strong_buy ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#10b98118', color: '#10b981' }}>
                      SB {sector.strong_buy}
                    </span>
                  )}
                  {(sector.buy ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#6ee7b718', color: '#6ee7b7' }}>
                      B {sector.buy}
                    </span>
                  )}
                  {neutral > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#47556918', color: '#64748b' }}>
                      H {neutral}
                    </span>
                  )}
                  {(sector.sell ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#fca5a518', color: '#fca5a5' }}>
                      S {sector.sell}
                    </span>
                  )}
                  {(sector.strong_sell ?? 0) > 0 && (
                    <span className="text-[9px] px-1.5 py-0.5 rounded font-semibold tabular-nums" style={{ background: '#f43f5e18', color: '#f43f5e' }}>
                      SS {sector.strong_sell}
                    </span>
                  )}
                </div>

                {/* Spacer */}
                <div className="flex-1" />

                {/* Momentum */}
                <div className="flex items-center gap-1">
                  {avgMom > 0 ? (
                    <ArrowUp className="w-3 h-3 text-[var(--accent-emerald)]" />
                  ) : avgMom < 0 ? (
                    <ArrowDown className="w-3 h-3 text-[var(--accent-rose)]" />
                  ) : null}
                  <span className="text-[11px] font-bold font-mono tabular-nums"
                    style={{ color: avgMom > 0 ? '#10b981' : avgMom < 0 ? '#f43f5e' : '#64748b' }}>
                    {avgMom > 0 ? '+' : ''}{avgMom.toFixed(1)}%
                  </span>
                </div>

                {/* Best asset peek */}
                {!expanded && bestTicker && (
                  <div className="hidden lg:flex items-center gap-1.5 px-2.5 py-1 rounded-lg" style={{ background: 'var(--void-active)' }}>
                    <span className="text-[9px] text-[var(--text-muted)]">Top</span>
                    <span className="text-[10px] font-bold text-[var(--accent-violet)]">{bestTicker}</span>
                    <span className="text-[10px] font-bold tabular-nums" style={{ color: bestRet >= 0 ? '#10b981' : '#f43f5e' }}>
                      {bestRet >= 0 ? '+' : ''}{bestRet.toFixed(1)}%
                    </span>
                    <span className="text-[8px] px-1 py-0.5 rounded font-semibold"
                      style={{ background: `${signalLabelColor(bestLabel)}18`, color: signalLabelColor(bestLabel) }}>
                      {bestLabel}
                    </span>
                  </div>
                )}
              </div>
            </button>

            {/* Expanded content — premium table */}
            {expanded && (
              <div
                style={{
                  animation: 'sectorReveal 220ms cubic-bezier(0.2,0,0,1) both',
                }}
              >
                <style>{`
                  @keyframes sectorReveal {
                    from { opacity: 0; transform: translateY(-4px); }
                    to   { opacity: 1; transform: translateY(0); }
                  }
                  @media (prefers-reduced-motion: reduce) {
                    [style*="sectorReveal"] { animation: none !important; }
                  }
                `}</style>
                {/* Sector summary strip */}
                <div className="flex items-center gap-4 px-5 py-2 text-[10px]"
                  style={{ background: `${sentColor}06`, borderTop: `1px solid ${sentColor}15`, borderBottom: '1px solid var(--border-void)' }}>
                  <div className="flex items-center gap-3">
                    <span className="text-[var(--text-muted)]">Breakdown:</span>
                    {[
                      { label: 'Strong Buy', count: sector.strong_buy ?? 0, color: '#10b981' },
                      { label: 'Buy', count: sector.buy ?? 0, color: '#6ee7b7' },
                      { label: 'Hold', count: sector.hold ?? 0, color: '#64748b' },
                      { label: 'Sell', count: sector.sell ?? 0, color: '#fca5a5' },
                      { label: 'Strong Sell', count: sector.strong_sell ?? 0, color: '#f43f5e' },
                    ].filter(x => x.count > 0).map(({ label, count, color: c }) => (
                      <span key={label} className="flex items-center gap-1">
                        <span className="w-1.5 h-1.5 rounded-full" style={{ background: c }} />
                        <span style={{ color: c }} className="font-medium">{count}</span>
                        <span className="text-[var(--text-muted)]">{label}</span>
                      </span>
                    ))}
                  </div>
                  <div className="ml-auto flex items-center gap-1.5 text-[var(--text-muted)]">
                    <Activity className="w-3 h-3" />
                    <span>Avg Risk: </span>
                    <span className="font-bold tabular-nums" style={{
                      color: (sector.avg_crash_risk ?? 0) > 60 ? '#f43f5e' : (sector.avg_crash_risk ?? 0) > 30 ? '#f59e0b' : '#10b981'
                    }}>
                      {(sector.avg_crash_risk ?? 0).toFixed(0)}
                    </span>
                  </div>
                </div>

                {sectorChartView ? (
                  <div className="space-y-1.5 p-2">
                    {sortedAssets.map((row) => {
                      const ticker = extractTicker(row.asset_label);
                      const isExpandedRow = chartExpandedRow === row.asset_label;
                      return (
                        <ChartAssetRow
                          key={row.asset_label}
                          row={row}
                          ticker={ticker}
                          horizons={horizons}
                          qualityScore={qualityScores[ticker] ?? 50}
                          highlighted={row.asset_label === updatedAsset}
                          isExpanded={isExpandedRow}
                          onToggleExpand={() => setChartExpandedRow(isExpandedRow ? null : row.asset_label)}
                          onNavigateChart={() => navigate(`/charts/${ticker}`)}
                        />
                      );
                    })}
                  </div>
                ) : (
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr style={{ background: 'var(--void-hover)' }}>
                        <th onClick={() => onRowSort('asset')} className={`text-left px-3 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[160px] ${thSortClass('asset')}`}>Asset{sortArrow('asset')}</th>
                        {sectorVisibleCols.has('chart') && (
                          <th className="text-center px-2 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[124px]">Chart</th>
                        )}
                        {sectorVisibleCols.has('pct30d') && (
                          <th className="text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px]">30D</th>
                        )}
                        <th onClick={() => onRowSort('signal')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[88px] ${thSortClass('signal')}`}>Signal{sortArrow('signal')}</th>
                        {sectorVisibleCols.has('strength') && (
                          <th onClick={() => onRowSort('strength')} className={`text-center px-1 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[64px] ${thSortClass('strength')}`}>Strength{sortArrow('strength')}</th>
                        )}
                        {sectorVisibleCols.has('momentum') && (
                          <th onClick={() => onRowSort('momentum')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass('momentum')}`}>Mom{sortArrow('momentum')}</th>
                        )}
                        {sectorVisibleCols.has('quality') && (
                          <th onClick={() => onRowSort('quality')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass('quality')}`}>Quality{sortArrow('quality')}</th>
                        )}
                        {sectorVisibleCols.has('horizons') && horizons.map(h => {
                          const col = `horizon_${h}` as SectorRowSortColumn;
                          return (
                            <th key={h} onClick={() => onRowSort(col)} className={`text-center px-1 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass(col)}`}>{formatHorizon(h)}{sortArrow(col)}</th>
                          );
                        })}
                        {sectorVisibleCols.has('risk') && (
                          <th onClick={() => onRowSort('crash_risk')} className={`text-center px-1.5 py-2 text-[10px] text-[var(--text-muted)] font-semibold uppercase tracking-wider w-[56px] ${thSortClass('crash_risk')}`}>Risk{sortArrow('crash_risk')}</th>
                        )}
                        <th className="w-6"></th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedAssets.map((row, i) => (
                        <SectorSignalRow
                          key={row.asset_label}
                          row={row}
                          horizons={horizons}
                          visibleCols={sectorVisibleCols}
                          qualityScore={qualityScores[extractTicker(row.asset_label)] ?? 50}
                          highlighted={row.asset_label === updatedAsset}
                          delayMs={i * 30}
                          onNavigateChart={(sym) => navigate(`/charts/${sym}`)}
                        />
                      ))}
                    </tbody>
                  </table>
                </div>
                )}
                {assets.length === 0 && (
                  <div className="px-5 py-6 text-center">
                    <Shield className="w-5 h-5 mx-auto mb-1.5" style={{ color: 'var(--text-muted)', opacity: 0.4 }} />
                    <p className="text-[11px] text-[var(--text-muted)]">No assets match current filter</p>
                  </div>
                )}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
