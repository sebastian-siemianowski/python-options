/**
 * Signal Heatmap -- Living star-map of assets x horizons.
 *
 * Each cell glows against the void with the intensity of its signal:
 * strong signals burn bright (emerald/rose), holds are invisible dimples.
 * Sector groups are collapsible with sentiment bars and momentum.
 */
import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import type { SummaryRow, SectorGroup } from '../api';
import { formatHorizon } from '../utils/horizons';
import { ChevronDown, ChevronRight } from 'lucide-react';

const SECTOR_COLLAPSE_KEY = 'heatmap_sector_collapse';

/**
 * Perceptually uniform cell color using opacity against the void.
 * Bearish: rose at opacity mapped to magnitude
 * Neutral: void-surface (nearly invisible)
 * Bullish: emerald at opacity mapped to magnitude
 */
function heatCellStyle(expRet: number | null | undefined): { background: string; opacity: number } {
  if (expRet == null) return { background: 'var(--void-surface, #0a0a23)', opacity: 0.5 };
  const clamped = Math.max(-0.10, Math.min(0.10, expRet));
  const magnitude = Math.abs(clamped) / 0.10; // 0-1
  if (Math.abs(clamped) < 0.005) {
    return { background: 'var(--void-surface, #0a0a23)', opacity: 0.6 };
  }
  if (clamped > 0) {
    // Emerald: opacity proportional to magnitude
    return {
      background: `rgba(52, 211, 153, ${0.08 + magnitude * 0.45})`,
      opacity: 1,
    };
  }
  // Rose: opacity proportional to magnitude
  return {
    background: `rgba(251, 113, 133, ${0.08 + magnitude * 0.45})`,
    opacity: 1,
  };
}

/** Opacity from position_strength (conviction). */
function cellOpacity(sig: { position_strength?: number } | undefined): number {
  if (!sig || sig.position_strength == null) return 0.7;
  return 0.3 + 0.7 * Math.min(Math.max(sig.position_strength, 0), 1);
}

/** Build a mini sentiment bar from sector counts. */
function SectorSentimentBar({ sector }: { sector: SectorGroup }) {
  const total = sector.asset_count || 1;
  const segments = [
    { pct: (sector.strong_sell / total) * 100, color: '#FB7185' },
    { pct: (sector.sell / total) * 100, color: 'rgba(251,113,133,0.4)' },
    { pct: (sector.hold / total) * 100, color: '#1c1845' },
    { pct: (sector.buy / total) * 100, color: 'rgba(52,211,153,0.4)' },
    { pct: (sector.strong_buy / total) * 100, color: '#34D399' },
  ];
  return (
    <div className="flex h-1 w-20 rounded-full overflow-hidden" style={{ background: 'var(--void-active, #1c1845)' }}>
      {segments.map((s, i) => s.pct > 0 ? (
        <div key={i} className="h-full" style={{ width: `${s.pct}%`, background: s.color }} />
      ) : null)}
    </div>
  );
}

interface Props {
  sectors: SectorGroup[];
  horizons: number[];
}

export default function SignalHeatmap({ sectors, horizons }: Props) {
  const navigate = useNavigate();
  const containerRef = useRef<HTMLDivElement>(null);
  const [focusRow, setFocusRow] = useState(-1);
  const [focusCol, setFocusCol] = useState(0);
  const [hoveredCell, setHoveredCell] = useState<{ row: number; col: number } | null>(null);
  const [tooltipData, setTooltipData] = useState<{
    asset: string; horizon: string; expRet: number; pUp: number;
    kelly?: number; label: string; x: number; y: number;
  } | null>(null);
  const [flashCell, setFlashCell] = useState<string | null>(null);

  // Persist collapsed sectors
  const [collapsed, setCollapsed] = useState<Set<string>>(() => {
    try {
      const raw = localStorage.getItem(SECTOR_COLLAPSE_KEY);
      return raw ? new Set(JSON.parse(raw)) : new Set();
    } catch { return new Set(); }
  });

  useEffect(() => {
    try { localStorage.setItem(SECTOR_COLLAPSE_KEY, JSON.stringify([...collapsed])); } catch { /* noop */ }
  }, [collapsed]);

  // Flatten for keyboard navigation
  const flatRows = useMemo(() => {
    const result: { type: 'sector' | 'asset'; label: string; row?: SummaryRow; sector?: string }[] = [];
    for (const s of sectors) {
      result.push({ type: 'sector', label: s.name });
      if (!collapsed.has(s.name)) {
        for (const a of s.assets) {
          result.push({ type: 'asset', label: a.asset_label, row: a, sector: s.name });
        }
      }
    }
    return result;
  }, [sectors, collapsed]);

  const toggleSector = useCallback((name: string) => {
    setCollapsed(prev => {
      const next = new Set(prev);
      next.has(name) ? next.delete(name) : next.add(name);
      return next;
    });
  }, []);

  // Keyboard navigation (j/k for rows, h/l for columns, Enter to navigate, Esc to deselect)
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: KeyboardEvent) => {
      if (!el.contains(document.activeElement) && document.activeElement !== el) return;
      if (e.key === 'j' || e.key === 'ArrowDown') {
        e.preventDefault();
        setFocusRow(r => Math.min(r + 1, flatRows.length - 1));
      } else if (e.key === 'k' || e.key === 'ArrowUp') {
        e.preventDefault();
        setFocusRow(r => Math.max(r - 1, 0));
      } else if (e.key === 'l' || e.key === 'ArrowRight') {
        e.preventDefault();
        setFocusCol(c => Math.min(c + 1, horizons.length - 1));
      } else if (e.key === 'h' || e.key === 'ArrowLeft') {
        e.preventDefault();
        setFocusCol(c => Math.max(c - 1, 0));
      } else if (e.key === 'Enter') {
        const item = flatRows[focusRow];
        if (item?.type === 'asset' && item.row) {
          navigate(`/charts/${item.label}`);
        } else if (item?.type === 'sector') {
          toggleSector(item.label);
        }
      } else if (e.key === 'Escape') {
        setFocusRow(-1);
        el.blur();
      }
    };
    el.addEventListener('keydown', handler);
    return () => el.removeEventListener('keydown', handler);
  }, [flatRows, focusRow, focusCol, horizons.length, navigate, toggleSector]);

  const handleCellHover = useCallback((asset: SummaryRow, horizonKey: string, horizonLabel: string, e: React.MouseEvent, rowIdx: number, colIdx: number) => {
    setHoveredCell({ row: rowIdx, col: colIdx });
    const sig = asset.horizon_signals[horizonKey] || asset.horizon_signals[String(horizonKey)];
    if (!sig) return;
    const rect = (e.target as HTMLElement).getBoundingClientRect();
    const containerRect = containerRef.current?.getBoundingClientRect();
    setTooltipData({
      asset: asset.asset_label,
      horizon: horizonLabel,
      expRet: sig.exp_ret,
      pUp: sig.p_up,
      kelly: sig.kelly_half,
      label: sig.label || '',
      x: rect.left - (containerRect?.left ?? 0) + rect.width / 2,
      y: rect.top - (containerRect?.top ?? 0) - 8,
    });
  }, []);

  const handleCellClick = useCallback((symbol: string, cellKey: string) => {
    setFlashCell(cellKey);
    setTimeout(() => {
      setFlashCell(null);
      navigate(`/charts/${symbol}`);
    }, 200);
  }, [navigate]);

  const totalSignals = useMemo(() => {
    let count = 0;
    for (const s of sectors) {
      count += (s.strong_buy ?? 0) + (s.buy ?? 0) + (s.sell ?? 0) + (s.strong_sell ?? 0);
    }
    return count;
  }, [sectors]);

  let rowIdx = -1;

  return (
    <div className="glass-card overflow-hidden relative" ref={containerRef} tabIndex={0}
      style={{ outline: 'none' }}>
      {/* Header */}
      <div className="flex items-center justify-between px-5 py-3"
        style={{ borderBottom: '1px solid rgba(139,92,246,0.08)' }}>
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-primary, #e2e8f0)' }}>
          Signal Heatmap
        </h3>
        <div className="flex items-center gap-4">
          {/* Color scale legend */}
          <div className="flex items-center gap-2">
            <span className="text-[9px] tabular-nums" style={{ color: 'var(--text-muted, #475569)' }}>-10%</span>
            <div className="w-[120px] h-2 rounded-full overflow-hidden flex"
              style={{ background: 'var(--void-surface, #0a0a23)' }}>
              <div className="h-full w-1/2" style={{
                background: 'linear-gradient(90deg, rgba(251,113,133,0.5) 0%, rgba(251,113,133,0.08) 80%, transparent 100%)',
              }} />
              <div className="h-full w-1/2" style={{
                background: 'linear-gradient(90deg, transparent 0%, rgba(52,211,153,0.08) 20%, rgba(52,211,153,0.5) 100%)',
              }} />
            </div>
            <span className="text-[9px] tabular-nums" style={{ color: 'var(--text-muted, #475569)' }}>+10%</span>
          </div>
          <span className="text-[10px]" style={{ color: 'var(--text-muted, #475569)' }}>
            {totalSignals} active signals
          </span>
        </div>
      </div>

      {/* Tooltip */}
      {tooltipData && (
        <div
          className="absolute z-50 pointer-events-none"
          style={{
            left: tooltipData.x,
            top: tooltipData.y,
            transform: 'translate(-50%, -100%)',
          }}
        >
          <div
            className="rounded-xl px-4 py-3 min-w-[180px]"
            style={{
              background: 'linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)',
              border: '1px solid rgba(139,92,246,0.25)',
              backdropFilter: 'blur(20px)',
              boxShadow: '0 8px 32px rgba(139,92,246,0.12), 0 0 80px rgba(139,92,246,0.05)',
            }}
          >
            <div className="text-xs font-medium mb-1" style={{ color: '#e2e8f0' }}>
              {tooltipData.asset} <span style={{ color: 'var(--text-muted, #475569)' }}>{tooltipData.horizon}</span>
            </div>
            <div className="flex items-baseline gap-2 mb-1">
              <span
                className="text-base font-bold tabular-nums"
                style={{
                  color: tooltipData.expRet >= 0 ? '#34D399' : '#FB7185',
                  ...(Math.abs(tooltipData.expRet) > 0.05 ? {
                    background: tooltipData.expRet > 0
                      ? 'linear-gradient(135deg, #f8fafc 0%, #34D399 100%)'
                      : 'linear-gradient(135deg, #f8fafc 0%, #FB7185 100%)',
                    WebkitBackgroundClip: 'text',
                    WebkitTextFillColor: 'transparent',
                  } : {}),
                }}
              >
                {tooltipData.expRet >= 0 ? '+' : ''}{(tooltipData.expRet * 100).toFixed(2)}%
              </span>
            </div>
            <div className="flex items-center gap-3 text-[10px]">
              <div className="flex items-center gap-1">
                <svg width="16" height="16" viewBox="0 0 16 16">
                  <circle cx="8" cy="8" r="6" fill="none"
                    stroke="rgba(139,92,246,0.15)" strokeWidth="1.5"
                    strokeDasharray={`${Math.PI * 12 * 0.75} ${Math.PI * 12}`}
                    transform="rotate(135 8 8)" strokeLinecap="round"
                  />
                  <circle cx="8" cy="8" r="6" fill="none"
                    stroke={tooltipData.expRet >= 0 ? '#34D399' : '#FB7185'}
                    strokeWidth="1.5"
                    strokeDasharray={`${Math.PI * 12 * 0.75} ${Math.PI * 12}`}
                    strokeDashoffset={Math.PI * 12 * 0.75 * (1 - tooltipData.pUp)}
                    transform="rotate(135 8 8)" strokeLinecap="round"
                  />
                </svg>
                <span style={{ color: 'var(--text-secondary, #94a3b8)' }}>
                  {(tooltipData.pUp * 100).toFixed(0)}%
                </span>
              </div>
              {tooltipData.kelly != null && (
                <div className="flex items-center gap-1">
                  <div className="w-10 h-[3px] rounded-full overflow-hidden"
                    style={{ background: 'var(--void-active, #1c1845)' }}>
                    <div className="h-full rounded-full"
                      style={{
                        width: `${Math.min(Math.abs(tooltipData.kelly) * 100, 100)}%`,
                        background: 'linear-gradient(90deg, #8B5CF6, #6366F1)',
                      }}
                    />
                  </div>
                  <span style={{ color: 'var(--text-muted, #475569)' }}>
                    Kelly {(tooltipData.kelly * 100).toFixed(0)}%
                  </span>
                </div>
              )}
            </div>
            {tooltipData.label && (
              <div className="mt-1.5">
                <span
                  className="inline-block px-2 py-0.5 rounded text-[9px] font-medium"
                  style={{
                    background: tooltipData.expRet >= 0
                      ? 'linear-gradient(135deg, #064e3b 0%, #047857 100%)'
                      : 'linear-gradient(135deg, #4c0519 0%, #881337 100%)',
                    color: tooltipData.expRet >= 0 ? '#34D399' : '#FB7185',
                  }}
                >
                  {tooltipData.label}
                </span>
              </div>
            )}
          </div>
        </div>
      )}

      <div className="overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr style={{ borderBottom: '1px solid rgba(139,92,246,0.08)' }}>
              <th className="text-left px-4 py-2 font-medium w-36"
                style={{
                  color: 'var(--text-muted, #475569)',
                  background: 'linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)',
                  backdropFilter: 'blur(12px)',
                  position: 'sticky', left: 0, zIndex: 10,
                }}>
                Asset
              </th>
              {horizons.map(h => (
                <th key={h} className="text-center px-1 py-2 font-medium w-16"
                  style={{
                    color: 'var(--text-muted, #475569)',
                    background: 'linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)',
                    backdropFilter: 'blur(12px)',
                  }}>
                  {formatHorizon(h)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sectors.map(sector => {
              const isCollapsed = collapsed.has(sector.name);
              rowIdx++;
              const sectorIdx = rowIdx;
              const avgMom = sector.avg_momentum ?? 0;
              return (
                <tbody key={sector.name}>
                  {/* Sector header */}
                  <tr
                    className="cursor-pointer transition-colors"
                    style={{
                      background: 'var(--void-hover, #16133a)',
                      borderBottom: '1px solid rgba(139,92,246,0.06)',
                    }}
                    onClick={() => toggleSector(sector.name)}
                  >
                    <td className="px-4 py-2 whitespace-nowrap" colSpan={horizons.length + 1}>
                      <div className="flex items-center gap-3">
                        <span className="flex items-center gap-1.5">
                          <span style={{
                            transform: isCollapsed ? 'rotate(0deg)' : 'rotate(0deg)',
                            transition: 'transform 200ms ease',
                            display: 'flex',
                          }}>
                            {isCollapsed
                              ? <ChevronRight className="w-3.5 h-3.5" style={{ color: 'var(--text-muted, #475569)' }} />
                              : <ChevronDown className="w-3.5 h-3.5" style={{ color: 'var(--text-muted, #475569)' }} />
                            }
                          </span>
                          <span className="text-[11px] font-medium" style={{ color: 'var(--text-violet, #C4B5FD)' }}>
                            {sector.name}
                          </span>
                        </span>
                        <span
                          className="px-1.5 py-0.5 rounded text-[9px]"
                          style={{ background: 'rgba(139,92,246,0.08)', color: 'var(--text-muted, #475569)' }}
                        >
                          {sector.asset_count}
                        </span>
                        <SectorSentimentBar sector={sector} />
                        <span
                          className="text-[10px] tabular-nums font-medium"
                          style={{ color: avgMom > 5 ? '#34D399' : avgMom < -5 ? '#FB7185' : 'var(--text-muted, #475569)' }}
                        >
                          {avgMom > 0 ? '+' : ''}{avgMom.toFixed(1)}
                        </span>
                      </div>
                    </td>
                  </tr>
                  {/* Asset rows */}
                  {!isCollapsed && sector.assets.map(asset => {
                    rowIdx++;
                    const aIdx = rowIdx;
                    return (
                      <tr
                        key={asset.asset_label}
                        className="transition-all"
                        style={{
                          borderBottom: '1px solid rgba(139,92,246,0.04)',
                          ...(aIdx === focusRow ? {
                            boxShadow: '0 0 0 1px rgba(139,92,246,0.25), 0 0 12px rgba(139,92,246,0.08)',
                          } : {}),
                        }}
                      >
                        <td
                          className="px-4 py-1 whitespace-nowrap cursor-pointer"
                          style={{
                            color: 'var(--text-primary, #e2e8f0)',
                            fontSize: '10px',
                            position: 'sticky', left: 0, zIndex: 5,
                            background: 'var(--void, #030014)',
                          }}
                          onClick={() => handleCellClick(asset.asset_label, `${asset.asset_label}-nav`)}
                        >
                          {asset.asset_label}
                        </td>
                        {horizons.map((h, ci) => {
                          const sig = asset.horizon_signals[h] || asset.horizon_signals[String(h)];
                          const { background, opacity } = heatCellStyle(sig?.exp_ret);
                          const isFocused = aIdx === focusRow && ci === focusCol;
                          const isHovered = hoveredCell?.row === aIdx && hoveredCell?.col === ci;
                          const cellKey = `${asset.asset_label}-${h}`;
                          const isFlashing = flashCell === cellKey;
                          return (
                            <td
                              key={h}
                              className="text-center px-0.5 py-0.5 cursor-pointer"
                              onClick={() => handleCellClick(asset.asset_label, cellKey)}
                              onMouseEnter={(e) => handleCellHover(asset, String(h), formatHorizon(h), e, aIdx, ci)}
                              onMouseLeave={() => { setHoveredCell(null); setTooltipData(null); }}
                            >
                              <div
                                className="rounded transition-all duration-150"
                                style={{
                                  background,
                                  opacity,
                                  height: '24px',
                                  display: 'flex',
                                  alignItems: 'center',
                                  justifyContent: 'center',
                                  border: `1px solid ${isFocused ? 'rgba(139,92,246,0.3)' : 'rgba(139,92,246,0.04)'}`,
                                  borderRadius: '3px',
                                  boxShadow: isFlashing
                                    ? '0 0 16px rgba(139,92,246,0.4)'
                                    : isHovered
                                      ? '0 0 16px rgba(139,92,246,0.2)'
                                      : isFocused
                                        ? '0 0 8px rgba(139,92,246,0.15)'
                                        : 'none',
                                  transform: isHovered ? 'scale(1.08)' : 'scale(1)',
                                }}
                              >
                                <span className="text-[9px] tabular-nums font-medium"
                                  style={{
                                    color: sig?.exp_ret != null
                                      ? (Math.abs(sig.exp_ret) > 0.03 ? '#f8fafc' : 'var(--text-secondary, #94a3b8)')
                                      : 'var(--text-muted, #475569)',
                                  }}>
                                  {sig?.exp_ret != null ? `${(sig.exp_ret * 100).toFixed(1)}` : '\u2014'}
                                </span>
                              </div>
                            </td>
                          );
                        })}
                      </tr>
                    );
                  })}
                </tbody>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
