/**
 * Story 6.6: Signal Heatmap — full-screen overview of assets x horizons.
 *
 * Rows: assets grouped by sector. Columns: horizons.
 * Cell color: continuous gradient (deep red -> gray -> deep green) from exp_ret.
 * Cell opacity: proportional to position_strength (conviction).
 */
import { useState, useMemo, useCallback, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import type { SummaryRow, SectorGroup } from '../api';
import { formatHorizon } from '../utils/horizons';
import { ChevronDown, ChevronRight } from 'lucide-react';

/**
 * CIELAB-inspired perceptually uniform interpolation.
 * Maps exp_ret (-0.10 .. +0.10) to a color gradient:
 *   deep red -> gray -> deep green.
 */
function heatColor(expRet: number | null | undefined): string {
  if (expRet == null) return 'rgba(100, 116, 139, 0.15)';
  // Clamp to [-0.10, +0.10] then normalize to [-1, 1]
  const clamped = Math.max(-0.10, Math.min(0.10, expRet));
  const t = clamped / 0.10; // -1 to +1
  if (t >= 0) {
    // Gray (100,116,139) -> Green (0,230,118)
    const r = Math.round(100 * (1 - t));
    const g = Math.round(116 + (230 - 116) * t);
    const b = Math.round(139 * (1 - t) + 118 * t);
    return `rgb(${r}, ${g}, ${b})`;
  } else {
    const s = -t;
    // Gray (100,116,139) -> Red (239,83,80)
    const r = Math.round(100 + (239 - 100) * s);
    const g = Math.round(116 * (1 - s) + 83 * s);
    const b = Math.round(139 * (1 - s) + 80 * s);
    return `rgb(${r}, ${g}, ${b})`;
  }
}

/** Opacity from position_strength (conviction). */
function cellOpacity(sig: { position_strength?: number } | undefined): number {
  if (!sig || sig.position_strength == null) return 0.7;
  return 0.3 + 0.7 * Math.min(Math.max(sig.position_strength, 0), 1);
}

interface Props {
  sectors: SectorGroup[];
  horizons: number[];
}

export default function SignalHeatmap({ sectors, horizons }: Props) {
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set());
  const [focusRow, setFocusRow] = useState(0);
  const [focusCol, setFocusCol] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

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

  // Keyboard navigation
  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'ArrowDown') { e.preventDefault(); setFocusRow(r => Math.min(r + 1, flatRows.length - 1)); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); setFocusRow(r => Math.max(r - 1, 0)); }
      else if (e.key === 'ArrowRight') { e.preventDefault(); setFocusCol(c => Math.min(c + 1, horizons.length - 1)); }
      else if (e.key === 'ArrowLeft') { e.preventDefault(); setFocusCol(c => Math.max(c - 1, 0)); }
      else if (e.key === 'Enter') {
        const item = flatRows[focusRow];
        if (item?.type === 'asset' && item.row) {
          navigate(`/charts/${item.label}`);
        } else if (item?.type === 'sector') {
          toggleSector(item.label);
        }
      }
      else if (e.key === 'Escape') { el.blur(); }
    };
    el.addEventListener('keydown', handler);
    return () => el.removeEventListener('keydown', handler);
  }, [flatRows, focusRow, focusCol, horizons.length, navigate, toggleSector]);

  // Total expected P&L
  const totalSignals = useMemo(() => {
    let count = 0;
    for (const s of sectors) {
      count += (s.strong_buy ?? 0) + (s.buy ?? 0) + (s.sell ?? 0) + (s.strong_sell ?? 0);
    }
    return count;
  }, [sectors]);

  let rowIdx = -1;

  return (
    <div className="glass-card overflow-hidden" ref={containerRef} tabIndex={0}>
      <div className="flex items-center justify-between px-4 py-3 border-b border-[#2a2a4a]">
        <h3 className="text-sm font-medium text-[#e2e8f0]">Signal Heatmap</h3>
        <span className="text-[10px] text-[#64748b]">{totalSignals} active signals</span>
      </div>
      <div className="overflow-x-auto">
        <table className="w-full text-[10px]">
          <thead>
            <tr className="border-b border-[#2a2a4a]">
              <th className="text-left px-3 py-2 text-[#64748b] font-medium w-32">Asset</th>
              {horizons.map(h => (
                <th key={h} className="text-center px-1 py-2 text-[#64748b] font-medium w-16">{formatHorizon(h)}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {sectors.map(sector => {
              const isCollapsed = collapsed.has(sector.name);
              rowIdx++;
              const sectorIdx = rowIdx;
              return (
                <tbody key={sector.name}>
                  <tr
                    className={`border-b border-[#2a2a4a]/50 cursor-pointer transition hover:bg-[#16213e]/40 ${sectorIdx === focusRow ? 'ring-1 ring-[#42A5F5]/40' : ''}`}
                    onClick={() => toggleSector(sector.name)}
                  >
                    <td className="px-3 py-1.5 font-medium text-[#94a3b8] whitespace-nowrap" colSpan={horizons.length + 1}>
                      <span className="inline-flex items-center gap-1">
                        {isCollapsed ? <ChevronRight className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
                        {sector.name}
                        <span className="text-[9px] text-[#64748b] ml-1">{sector.asset_count} assets</span>
                      </span>
                    </td>
                  </tr>
                  {!isCollapsed && sector.assets.map(asset => {
                    rowIdx++;
                    const aIdx = rowIdx;
                    return (
                      <tr
                        key={asset.asset_label}
                        className={`border-b border-[#1a1a2e]/50 cursor-pointer transition hover:scale-[1.005] hover:shadow-md ${aIdx === focusRow ? 'ring-1 ring-[#42A5F5]/40' : ''}`}
                        onClick={() => navigate(`/charts/${asset.asset_label}`)}
                      >
                        <td className="px-3 py-1 text-[#e2e8f0] whitespace-nowrap">{asset.asset_label}</td>
                        {horizons.map((h, ci) => {
                          const sig = asset.horizon_signals[h] || asset.horizon_signals[String(h)];
                          const color = heatColor(sig?.exp_ret);
                          const opacity = cellOpacity(sig as any);
                          const isFocused = aIdx === focusRow && ci === focusCol;
                          return (
                            <td
                              key={h}
                              className={`text-center px-1 py-1 transition-colors duration-300 ${isFocused ? 'ring-1 ring-white/30' : ''}`}
                              style={{ backgroundColor: color, opacity }}
                              title={sig ? `${asset.asset_label} ${formatHorizon(h)}: ${(sig.exp_ret * 100).toFixed(1)}% | p_up=${(sig.p_up * 100).toFixed(0)}%` : '\u2014'}
                            >
                              {sig ? `${(sig.exp_ret * 100).toFixed(1)}` : '\u2014'}
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
