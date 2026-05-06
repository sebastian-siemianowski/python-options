import { useEffect, useRef, useState } from 'react';
import { BarChart3, ChevronRight, X } from 'lucide-react';
import type { SummaryRow } from '../../../api';
import { Sparkline, SparklinePct } from '../../../components/Sparkline';
import { SignalLabel, SignalStrengthMeter, MomentumBadge, CrashRiskHeat, HorizonCell, QualityCell } from '../../../components/SignalTableVisuals';
import { ColumnCustomizer, type ColumnDef } from '../../../components/ColumnCustomizer';
import SignalDetailPanel, { type SignalDetailChartType } from '../../../components/SignalDetailPanel';
import { formatHorizon } from '../../../utils/horizons';
import ChartAssetRow from './ChartAssetRow';
import { signalLabelColor } from '../theme';
import { extractTicker, type SortColumn, type SortDir } from '../utils';

/* ── Story 3.2: Sort Indicator with priority badge ───────────────── */
function SortIndicator({ col, sortLevels }: { col: SortColumn; sortLevels: { col: SortColumn; dir: SortDir }[] }) {
  const idx = sortLevels.findIndex(s => s.col === col);
  if (idx < 0) {
    return (
      <svg width="10" height="10" viewBox="0 0 10 10" className="inline ml-0.5 opacity-0 group-hover:opacity-40 transition-opacity" style={{ transition: 'opacity 120ms ease' }}>
        <path d="M5 2L8 7H2L5 2Z" fill="currentColor" />
      </svg>
    );
  }
  const level = sortLevels[idx];
  return (
    <span className="inline-flex items-center gap-0.5 ml-0.5">
      <svg width="10" height="10" viewBox="0 0 10 10" className={`sort-arrow-rotate ${level.dir === 'asc' ? 'sort-arrow-asc' : ''}`}
        style={{ color: 'var(--accent-violet)', transition: 'transform 200ms cubic-bezier(0.2,0,0,1)' }}>
        <path d="M5 2L8 7H2L5 2Z" fill="currentColor" />
      </svg>
      {sortLevels.length > 1 && (
        <span className="inline-flex items-center justify-center w-[14px] h-[14px] rounded-full text-[9px] font-semibold text-white"
          style={{ background: 'var(--accent-violet)' }}>
          {idx + 1}
        </span>
      )}
    </span>
  );
}

/** Story 3.2: Human-readable sort column name */
function sortColName(col: SortColumn): string {
  if (col.startsWith('horizon_')) return formatHorizon(parseInt(col.split('_')[1], 10));
  const names: Record<string, string> = { asset: 'Asset', sector: 'Sector', signal: 'Signal', momentum: 'Momentum', quality: 'Quality', crash_risk: 'Risk' };
  return names[col] || col;
}

// Column visibility — sortable headers still work; this independently controls rendering.
const ALL_ASSETS_COLUMN_DEFS: ColumnDef[] = [
  { key: 'asset', label: 'Asset', locked: true },
  { key: 'chart', label: 'Chart' },
  { key: 'pct30d', label: '30D change', hint: '%' },
  { key: 'sector', label: 'Sector' },
  { key: 'signal', label: 'Signal', locked: true },
  { key: 'strength', label: 'Strength' },
  { key: 'momentum', label: 'Momentum' },
  { key: 'quality', label: 'Quality' },
  { key: 'risk', label: 'Crash risk' },
  { key: 'horizons', label: 'Horizons' },
];
const ALL_ASSETS_COLS_LS_KEY = 'signals-visible-cols-v2';
const ALL_ASSETS_CHART_VIEW_LS_KEY = 'signals-all-assets-chart-view-v1';
export const SECTOR_CHART_VIEW_LS_KEY = 'signals-sector-chart-view-v1';
const DEFAULT_VISIBLE_COLS = new Set(ALL_ASSETS_COLUMN_DEFS.map((c) => c.key));
function loadVisibleCols(): Set<string> {
  try {
    const raw = localStorage.getItem(ALL_ASSETS_COLS_LS_KEY);
    if (!raw) return new Set(DEFAULT_VISIBLE_COLS);
    const parsed = JSON.parse(raw) as string[];
    const set = new Set(parsed);
    // Always force locked columns on
    ALL_ASSETS_COLUMN_DEFS.forEach((c) => { if (c.locked) set.add(c.key); });
    return set;
  } catch {
    return new Set(DEFAULT_VISIBLE_COLS);
  }
}

function loadAllAssetsChartView(): boolean {
  try {
    return localStorage.getItem(ALL_ASSETS_CHART_VIEW_LS_KEY) === '1';
  } catch {
    return false;
  }
}

export function loadSectorChartView(): boolean {
  try {
    return localStorage.getItem(SECTOR_CHART_VIEW_LS_KEY) === '1';
  } catch {
    return false;
  }
}

export default function AllAssetsTable({ rows, horizons, updatedAsset, sortLevels, onSort, onRemoveSort, expandedRow, onExpandRow, qualityScores, onNavigateChart, disablePagination, detailDefaultChartType }: {
  rows: SummaryRow[]; horizons: number[]; updatedAsset: string | null;
  sortLevels: { col: SortColumn; dir: SortDir }[];
  onSort: (col: SortColumn, shiftKey: boolean) => void;
  onRemoveSort: (col: SortColumn) => void;
  expandedRow: string | null; onExpandRow: (label: string | null) => void;
  qualityScores: Record<string, number>;
  onNavigateChart: (symbol: string) => void;
  /** When true, render all rows in one scrollable table with no pager UI. */
  disablePagination?: boolean;
  detailDefaultChartType?: SignalDetailChartType;
}) {
  const [page, setPage] = useState(0);
  const [scrolled, setScrolled] = useState(false);
  const [visibleCols, setVisibleCols] = useState<Set<string>>(() => loadVisibleCols());
  const [chartView, setChartView] = useState<boolean>(() => loadAllAssetsChartView());
  const tableContainerRef = useRef<HTMLDivElement>(null);
  const pageSize = 50;

  useEffect(() => {
    try {
      localStorage.setItem(
        ALL_ASSETS_COLS_LS_KEY,
        JSON.stringify(Array.from(visibleCols)),
      );
    } catch { /* ignore quota / privacy errors */ }
  }, [visibleCols]);

  useEffect(() => {
    try {
      localStorage.setItem(ALL_ASSETS_CHART_VIEW_LS_KEY, chartView ? '1' : '0');
    } catch { /* ignore quota / privacy errors */ }
  }, [chartView]);

  const toggleCol = (key: string) => {
    setVisibleCols((prev) => {
      const def = ALL_ASSETS_COLUMN_DEFS.find((c) => c.key === key);
      if (def?.locked) return prev;
      const next = new Set(prev);
      if (next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };
  const resetCols = () => setVisibleCols(new Set(DEFAULT_VISIBLE_COLS));
  const pageRows = disablePagination ? rows : rows.slice(page * pageSize, (page + 1) * pageSize);
  const totalPages = disablePagination ? 1 : Math.ceil(rows.length / pageSize);

  useEffect(() => { setPage(0); }, [rows.length]);

  // Detect scroll for sticky header shadow
  useEffect(() => {
    const container = tableContainerRef.current;
    if (!container) return;
    const onScroll = () => setScrolled(container.scrollTop > 0);
    container.addEventListener('scroll', onScroll, { passive: true });
    return () => container.removeEventListener('scroll', onScroll);
  }, []);

  const headerCls = `cosmic-table-header${scrolled ? ' scrolled' : ''}`;

  return (
    <div className="glass-card overflow-hidden fade-up-delay-3">
      {/* Table toolbar: column visibility (click headers to sort, use Columns to hide/show) */}
      <div className="flex items-center justify-between gap-3 px-4 h-10 border-b" style={{ borderColor: 'var(--border-void)' }}>
        <span className="text-[10px] uppercase tracking-[0.08em]" style={{ color: 'var(--text-muted)' }}>
          {rows.length} asset{rows.length === 1 ? '' : 's'}
          <span className="ml-2" style={{ color: 'var(--text-muted)', opacity: 0.6 }}>
            {chartView ? 'Chart-first decision view' : 'Click a column to sort · Shift+Click to add'}
          </span>
        </span>
        <div className="flex items-center gap-2">
          <button
            type="button"
            aria-pressed={chartView}
            title={chartView ? 'Switch to the current table view' : 'Switch to chart-first decision view'}
            onClick={() => setChartView((prev) => !prev)}
            className="inline-flex items-center gap-1.5 rounded-lg px-2.5 py-1.5 text-[10px] font-semibold uppercase tracking-[0.08em] transition-all"
            style={{
              color: chartView ? '#c4b5fd' : 'var(--text-muted)',
              background: chartView ? 'rgba(167,139,250,0.12)' : 'rgba(255,255,255,0.018)',
              border: `1px solid ${chartView ? 'rgba(167,139,250,0.42)' : 'rgba(255,255,255,0.055)'}`,
              boxShadow: chartView ? '0 0 16px -10px rgba(167,139,250,0.9)' : 'none',
            }}
          >
            <BarChart3 className="w-3.5 h-3.5" />
            Chart view
            <span
              className="ml-0.5 inline-flex h-4 w-7 items-center rounded-full p-[2px] transition-colors"
              style={{ background: chartView ? 'rgba(167,139,250,0.38)' : 'rgba(100,116,139,0.24)' }}
            >
              <span
                className="h-3 w-3 rounded-full transition-transform"
                style={{
                  background: chartView ? '#c4b5fd' : '#64748b',
                  transform: chartView ? 'translateX(12px)' : 'translateX(0)',
                  boxShadow: chartView ? '0 0 7px rgba(196,181,253,0.8)' : 'none',
                }}
              />
            </span>
          </button>
          {!chartView && (
            <ColumnCustomizer
              columns={ALL_ASSETS_COLUMN_DEFS}
              visible={visibleCols}
              onToggle={toggleCol}
              onReset={resetCols}
            />
          )}
        </div>
      </div>
      {/* Story 3.2 AC-5: Sort indicator bar */}
      {sortLevels.length > 0 && (
        <div className="flex items-center gap-2 px-4 h-[28px] text-[10px] text-[var(--text-secondary)]"
          style={{ background: 'var(--void-hover)' }}>
          <span>Sorted by </span>
          {sortLevels.map((s, i) => (
            <span key={s.col} className="inline-flex items-center gap-1">
              {i > 0 && <span className="text-[var(--text-muted)]">, then </span>}
              <span style={{ color: 'var(--accent-violet)' }}>{sortColName(s.col)}</span>
              <span className="text-[var(--text-muted)]">({s.dir})</span>
              <button onClick={() => onRemoveSort(s.col)}
                className="text-[var(--text-muted)] hover:text-[var(--accent-rose)] transition-colors text-[9px] ml-0.5">
                <X className="w-2.5 h-2.5" />
              </button>
            </span>
          ))}
          {sortLevels.length < 3 && (
            <span className="text-[var(--text-muted)] ml-1">(Shift+Click to add)</span>
          )}
        </div>
      )}
      <div ref={tableContainerRef} className="overflow-auto max-h-[calc(100vh-280px)]">
        {chartView ? (
          <div className="space-y-1.5 p-2">
            {pageRows.map((row) => {
              const ticker = extractTicker(row.asset_label);
              const isExpanded = expandedRow === row.asset_label;
              return (
                <ChartAssetRow
                  key={row.asset_label}
                  row={row}
                  ticker={ticker}
                  horizons={horizons}
                  qualityScore={qualityScores[ticker] ?? 50}
                  highlighted={row.asset_label === updatedAsset}
                  isExpanded={isExpanded}
                  onToggleExpand={() => onExpandRow(isExpanded ? null : row.asset_label)}
                  onNavigateChart={() => onNavigateChart(ticker)}
                  detailDefaultChartType={detailDefaultChartType}
                />
              );
            })}
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead className={headerCls}>
              <tr>
                <th className={`text-left px-4 py-3 sortable-th group ${sortLevels.some(s => s.col === 'asset') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'asset') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('asset', e.shiftKey)}>
                Asset <SortIndicator col="asset" sortLevels={sortLevels} />
              </th>
              {visibleCols.has('chart') && (
                <th className="text-center px-2 py-3 w-[124px]">
                  <span className="text-[10px] text-[var(--text-violet)] uppercase tracking-[0.06em] font-medium">Chart</span>
                </th>
              )}
              {visibleCols.has('pct30d') && (
                <th className="text-center px-2 py-3 w-[56px]">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-[0.06em] font-medium">30D</span>
                </th>
              )}
              {visibleCols.has('sector') && (
                <th className={`text-left px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'sector') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'sector') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('sector', e.shiftKey)}>
                  Sector <SortIndicator col="sector" sortLevels={sortLevels} />
                </th>
              )}
              <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'signal') ? 'active' : ''}`}
                  style={sortLevels.some(s => s.col === 'signal') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                  onClick={(e) => onSort('signal', e.shiftKey)}>
                Signal <SortIndicator col="signal" sortLevels={sortLevels} />
              </th>
              {visibleCols.has('strength') && (
                <th className="text-center px-2 py-3 w-[64px]">
                  <span className="text-[10px] text-[var(--text-muted)] uppercase tracking-[0.06em] font-medium">Strength</span>
                </th>
              )}
              {visibleCols.has('momentum') && (
                <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'momentum') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'momentum') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('momentum', e.shiftKey)}>
                  Mom <SortIndicator col="momentum" sortLevels={sortLevels} />
                </th>
              )}
              {visibleCols.has('quality') && (
                <th className={`text-center px-3 py-3 sortable-th group w-[72px] ${sortLevels.some(s => s.col === 'quality') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'quality') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('quality', e.shiftKey)}>
                  Quality <SortIndicator col="quality" sortLevels={sortLevels} />
                </th>
              )}
              {visibleCols.has('risk') && (
                <th className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === 'crash_risk') ? 'active' : ''}`}
                    style={sortLevels.some(s => s.col === 'crash_risk') ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                    onClick={(e) => onSort('crash_risk', e.shiftKey)}>
                  Risk <SortIndicator col="crash_risk" sortLevels={sortLevels} />
                </th>
              )}
              {visibleCols.has('horizons') && horizons.map((h) => {
                const hCol = `horizon_${h}` as SortColumn;
                return (
                  <th key={h} className={`text-center px-3 py-3 sortable-th group ${sortLevels.some(s => s.col === hCol) ? 'active' : ''}`}
                      style={sortLevels.some(s => s.col === hCol) ? { color: 'var(--accent-violet)', textShadow: '0 0 8px var(--violet-30)' } : {}}
                      onClick={(e) => onSort(hCol, e.shiftKey)}>
                    {formatHorizon(h)} <SortIndicator col={hCol} sortLevels={sortLevels} />
                  </th>
                );
              })}
              <th className="w-8 px-2"></th>
            </tr>
          </thead>
          <tbody>
            {pageRows.map((row) => {
              const ticker = extractTicker(row.asset_label);
              const isExpanded = expandedRow === row.asset_label;
              return (
                <CosmicSignalRow
                  key={row.asset_label}
                  row={row}
                  ticker={ticker}
                  horizons={horizons}
                  visibleCols={visibleCols}
                  qualityScore={qualityScores[ticker] ?? 50}
                  highlighted={row.asset_label === updatedAsset}
                  isExpanded={isExpanded}
                  onToggleExpand={() => onExpandRow(isExpanded ? null : row.asset_label)}
                  onNavigateChart={() => onNavigateChart(ticker)}
                  detailDefaultChartType={detailDefaultChartType}
                />
              );
            })}
            </tbody>
          </table>
        )}
      </div>
      {!disablePagination && totalPages > 1 && (
        <div className="flex items-center justify-between px-4 py-2.5 border-t border-[var(--border-void)]">
          <span className="text-xs text-[var(--text-muted)]">
            Page {page + 1} of {totalPages} ({rows.length} total)
          </span>
          <div className="flex gap-1">
            {(['First', 'Prev', 'Next', 'Last'] as const).map((label) => {
              const disabled = (label === 'First' || label === 'Prev') ? page === 0 : page >= totalPages - 1;
              const onClick = () => {
                if (label === 'First') setPage(0);
                else if (label === 'Prev') setPage(Math.max(0, page - 1));
                else if (label === 'Next') setPage(Math.min(totalPages - 1, page + 1));
                else setPage(totalPages - 1);
              };
              return (
                <button key={label} onClick={onClick} disabled={disabled}
                  className="px-2 py-0.5 rounded text-xs text-[var(--accent-violet)] hover:bg-[var(--void-hover)] disabled:opacity-30 transition">
                  {label}
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Story 3.1: Cosmic Signal Row ────────────────────────────────── */
function CosmicSignalRow({ row, ticker, horizons, visibleCols, qualityScore, highlighted, isExpanded, onToggleExpand, onNavigateChart, detailDefaultChartType }: {
  row: SummaryRow; ticker: string; horizons: number[];
  visibleCols: Set<string>;
  qualityScore: number;
  highlighted?: boolean; isExpanded: boolean;
  onToggleExpand: () => void; onNavigateChart: () => void;
  detailDefaultChartType?: SignalDetailChartType;
}) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  // Compute composite for strength bar
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  const pUp = nearestHorizon?.p_up;
  const kellyVal = nearestHorizon?.kelly_half;

  const labelColor = signalLabelColor(label);
  return (
    <>
      <tr
        onClick={onToggleExpand}
        className={`cosmic-row cursor-pointer transition-all duration-150 ${highlighted ? 'aurora-upgrade' : ''} ${isExpanded ? 'signals-row-selected' : 'hover:bg-white/[0.015]'}`}
        style={isExpanded ? {
          borderLeft: '2px solid var(--accent-violet)',
          background: 'rgba(139,92,246,0.05)',
          boxShadow: 'inset 0 0 20px rgba(139,92,246,0.06)',
        } : {
          borderLeft: '2px solid transparent',
          borderBottom: '1px solid rgba(255,255,255,0.035)',
        }}
      >
        {/* Asset */}
        <td className="px-4 py-2.5 whitespace-nowrap" style={{ height: '40px' }}>
          <div className="flex items-center gap-2">
            <span className="w-1.5 h-1.5 rounded-full flex-shrink-0" style={{ background: labelColor }} />
            <div className="text-left">
              <span className="font-semibold text-white text-[12px] tabular-nums">
                {ticker}
              </span>
              {row.asset_label.includes('(') && (
                <span className="block text-[9px] text-[var(--text-muted)] truncate max-w-[140px] leading-tight">
                  {row.asset_label.split('(')[0].trim()}
                </span>
              )}
            </div>
          </div>
        </td>
        {/* AC-1: Sparkline — wider for clarity */}
        {visibleCols.has('chart') && (
          <td className="px-2 py-2 text-center">
            <Sparkline ticker={ticker} width={108} height={32} />
          </td>
        )}
        {/* 30D pct change */}
        {visibleCols.has('pct30d') && (
          <td className="px-1.5 py-2 text-center">
            <SparklinePct ticker={ticker} />
          </td>
        )}
        {/* Sector */}
        {visibleCols.has('sector') && (
          <td className="px-3 py-2 text-[10px] text-[var(--text-secondary)] max-w-[100px] truncate">{row.sector}</td>
        )}
        {/* AC-2: Signal label + strength split */}
        <td className="px-2 py-2">
          <div className="flex justify-center">
            <SignalLabel label={label.toUpperCase()} />
          </div>
        </td>
        {visibleCols.has('strength') && (
          <td className="px-1 py-2">
            <div className="flex justify-center">
              <SignalStrengthMeter label={label} pUp={pUp} kelly={kellyVal} />
            </div>
          </td>
        )}
        {/* AC-3: Momentum badge */}
        {visibleCols.has('momentum') && (
          <td className="px-3 py-2 text-center">
            <MomentumBadge value={row.momentum_score} />
          </td>
        )}
        {/* Quality score tile */}
        {visibleCols.has('quality') && (
          <td className="px-2 py-2">
            <QualityCell score={qualityScore} />
          </td>
        )}
        {/* AC-4: Crash risk heat */}
        {visibleCols.has('risk') && (
          <td className="px-3 py-2">
            <div className="flex justify-center">
              <CrashRiskHeat score={row.crash_risk_score} />
            </div>
          </td>
        )}
        {/* AC-5: Horizon cells */}
        {visibleCols.has('horizons') && horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          return (
            <td key={h} className="px-2 py-2 text-center">
              <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
            </td>
          );
        })}
        {/* Expand indicator */}
        <td className="px-2 py-2">
          <ChevronRight
            className="w-3.5 h-3.5 transition-all duration-200"
            style={{
              color: isExpanded ? 'var(--accent-violet)' : 'var(--text-muted)',
              transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
            }}
          />
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={1000} className="p-0">
            <SignalDetailPanel
              ticker={ticker}
              signal={row.nearest_label}
              momentum={row.momentum_score}
              crashRisk={row.crash_risk_score}
              horizonSignals={row.horizon_signals as any}
              defaultChartType={detailDefaultChartType}
              onNavigateChart={onNavigateChart}
            />
          </td>
        </tr>
      )}
    </>
  );
}

/* ── Sector signal row — premium with inline expand ───────────────── */
export function SectorSignalRow({ row, horizons, visibleCols, qualityScore, highlighted, delayMs = 0, onNavigateChart }: {
  row: SummaryRow; horizons: number[];
  visibleCols: Set<string>;
  qualityScore: number;
  highlighted?: boolean; delayMs?: number;
  onNavigateChart: (sym: string) => void;
}) {
  const [isExpanded, setIsExpanded] = useState(false);
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  const ticker = extractTicker(row.asset_label);
  const nearestHorizon = Object.values(row.horizon_signals)[0];
  const labelColor = signalLabelColor(label);

  return (
    <>
      <tr
        onClick={() => setIsExpanded(p => !p)}
        className={`cursor-pointer transition-all duration-150 ${highlighted ? 'aurora-upgrade' : ''} ${isExpanded ? '' : 'hover:bg-white/[0.015]'}`}
        style={isExpanded ? {
          animationDelay: `${delayMs}ms`,
          borderLeft: '2px solid var(--accent-violet)',
          background: 'rgba(139,92,246,0.05)',
          boxShadow: 'inset 0 0 20px rgba(139,92,246,0.06)',
        } : {
          animationDelay: `${delayMs}ms`,
          borderLeft: '2px solid transparent',
          borderBottom: '1px solid rgba(255,255,255,0.035)',
        }}>
        {/* Asset */}
        <td className="px-3 py-2 whitespace-nowrap">
          <div className="flex items-center gap-2">
            <div className="w-1 h-7 rounded-full flex-shrink-0" style={{ background: `${labelColor}60` }} />
            <div>
              <div className="flex items-center gap-1.5">
                <span className="font-bold text-[12px] text-[#e2e8f0]">{ticker}</span>
                <span className="text-[8px] px-1.5 py-0.5 rounded font-semibold leading-none"
                  style={{ background: `${labelColor}15`, color: labelColor }}>
                  {label}
                </span>
              </div>
              {row.asset_label.includes('(') && (
                <span className="text-[9px] text-[var(--text-muted)] truncate max-w-[140px] leading-tight block mt-0.5">
                  {row.asset_label.split('(')[0].trim()}
                </span>
              )}
            </div>
          </div>
        </td>
        {/* Sparkline — wider for clarity */}
        {visibleCols.has('chart') && (
          <td className="px-2 py-2 text-center">
            <Sparkline ticker={ticker} width={108} height={32} />
          </td>
        )}
        {/* 30D pct change */}
        {visibleCols.has('pct30d') && (
          <td className="px-1.5 py-2 text-center">
            <SparklinePct ticker={ticker} />
          </td>
        )}
        {/* Signal label + strength split */}
        <td className="px-1.5 py-2">
          <div className="flex justify-center">
            <SignalLabel label={label.toUpperCase()} />
          </div>
        </td>
        {visibleCols.has('strength') && (
          <td className="px-1 py-2">
            <div className="flex justify-center">
              <SignalStrengthMeter label={label} pUp={nearestHorizon?.p_up} kelly={nearestHorizon?.kelly_half} />
            </div>
          </td>
        )}
        {/* Momentum */}
        {visibleCols.has('momentum') && (
          <td className="px-1.5 py-2 text-center">
            <MomentumBadge value={row.momentum_score} />
          </td>
        )}
        {/* Quality */}
        {visibleCols.has('quality') && (
          <td className="px-1.5 py-2">
            <QualityCell score={qualityScore} />
          </td>
        )}
        {/* Horizon cells */}
        {visibleCols.has('horizons') && horizons.map((h) => {
          const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
          return (
            <td key={h} className="px-1 py-2 text-center">
              <HorizonCell expRet={sig?.exp_ret} pUp={sig?.p_up} />
            </td>
          );
        })}
        {/* Risk */}
        {visibleCols.has('risk') && (
          <td className="px-1.5 py-2">
            <div className="flex justify-center">
              <CrashRiskHeat score={row.crash_risk_score} />
            </div>
          </td>
        )}
        {/* Actions */}
        <td className="px-1 py-2">
          <ChevronRight
            className="w-3.5 h-3.5 transition-all duration-200"
            style={{
              color: isExpanded ? 'var(--accent-violet)' : 'var(--text-muted)',
              transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)',
            }}
          />
        </td>
      </tr>
      {isExpanded && (
        <tr>
          <td colSpan={1000} className="p-0">
            <SignalDetailPanel
              ticker={ticker}
              signal={row.nearest_label}
              momentum={row.momentum_score}
              crashRisk={row.crash_risk_score}
              horizonSignals={row.horizon_signals as any}
              onNavigateChart={() => onNavigateChart(ticker)}
            />
          </td>
        </tr>
      )}
    </>
  );
}
