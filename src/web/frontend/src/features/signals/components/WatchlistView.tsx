import React, { Fragment, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, ArrowDownRight, ArrowUpRight, Filter, Layers, Loader2, Plus, Search, SlidersHorizontal, Star, X } from 'lucide-react';
import type { ReversalFlipsData, SummaryRow } from '../../../api';
import { useWatchlist } from '../../../hooks/useWatchlist';
import AllAssetsTable from './AllAssetsTable';
import SortPillStrip from './SortPillStrip';
import QualityFloorSlider from './QualityFloorSlider';
import {
  nextSortLevels,
  removeSortLevel,
  sortSummaryRows,
  extractTicker,
  isReversalQuickFilter,
  reversalFlipForAsset,
  rowHorizonColor,
  rowMatchesReversalFilter,
  type ReversalQuickFilter,
  type SortColumn,
  type SortDir,
} from '../utils';

const WATCHLIST_BIG_MOVE_ABS_THRESHOLD = 0.03;
const WATCHLIST_BIG_MOVE_HORIZON = 7;

/* ── Watchlist View — user-curated tickers with full detail ─────── */
export default function WatchlistView({
  allRows,
  horizons,
  sortHorizons,
  minQuality,
  onMinQualityChange,
  updatedAsset,
  qualityScores,
  reversalFlips,
  reversalFlipsLoading = false,
  onNavigateChart,
}: {
  allRows: SummaryRow[];
  horizons: number[];
  sortHorizons: number[];
  minQuality: number;
  onMinQualityChange: (value: number) => void;
  updatedAsset: string | null;
  qualityScores: Record<string, number>;
  reversalFlips?: ReversalFlipsData;
  reversalFlipsLoading?: boolean;
  onNavigateChart: (sym: string) => void;
}) {
  const { symbols, proxyMap, isLoading, add, remove } = useWatchlist();
  const [input, setInput] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  // Build a set of tickers for O(1) lookup. Also build a ticker → row map so
  // we can tell which watchlist symbols are present in the current signal set.
  // Some watchlist tickers are proxied under the hood (e.g. DFNG → ITA,
  // XAGUSD=X → SI=F, GLDE → GLD), so a signal row labelled with the primary
  // ticker must still count as "found" for the proxied watchlist entry. We
  // build a symbol → [acceptable tickers] map using the server-provided
  // PROXY_OVERRIDES, and match a row against any of them.
  const { watchlistRows, missingSymbols } = useMemo(() => {
    const acceptableBySymbol = new Map<string, Set<string>>();
    const accepted = new Set<string>();
    for (const s of symbols) {
      const targets = new Set<string>([s]);
      const primary = proxyMap[s];
      if (primary) targets.add(primary);
      acceptableBySymbol.set(s, targets);
      for (const t of targets) accepted.add(t);
    }
    const rowsForWatchlist: SummaryRow[] = [];
    const foundRowTickers = new Set<string>();
    for (const r of allRows) {
      const ticker = extractTicker(r.asset_label);
      if (accepted.has(ticker)) {
        rowsForWatchlist.push(r);
        foundRowTickers.add(ticker);
      }
    }
    const missing = symbols.filter((s) => {
      const targets = acceptableBySymbol.get(s);
      if (!targets) return true;
      for (const t of targets) {
        if (foundRowTickers.has(t)) return false;
      }
      return true;
    });
    return { watchlistRows: rowsForWatchlist, missingSymbols: missing };
  }, [allRows, symbols, proxyMap]);

  // ── Watchlist-local filters (no pagination — user disliked paging) ───
  // Signal class: STRONG_BUY/BUY/SELL/STRONG_SELL/HOLD via nearest_label.
  // Horizon colour: whole-row majority of exp_ret signs via rowHorizonColor —
  // a ticker can be "bullish" today (label) yet "reds" across horizons, so
  // both axes are useful filters.
  type WlClass = 'bullish' | 'bearish' | 'neutral';
  type WlColor = 'greens' | 'reds' | 'mixed';
  type WlBigMove = 'big_green' | 'big_red';
  type WlSignal = 'all' | WlClass | WlColor | WlBigMove | ReversalQuickFilter;
  type WlSort = 'signal' | 'momentum' | 'quality' | 'risk' | 'alpha';
  const [wlSignal, setWlSignal] = useState<WlSignal>('all');
  const [wlSector, setWlSector] = useState<string>('all');
  const [wlQuery, setWlQuery] = useState<string>('');
  const [wlSort, setWlSort] = useState<WlSort>('signal');
  const [wlMissingOnly, setWlMissingOnly] = useState<boolean>(false);
  const [wlSortLevels, setWlSortLevels] = useState<{ col: SortColumn; dir: SortDir }[]>([]);
  // Manage drawer is HIDDEN by default once the user has a populated list —
  // it auto-opens only when the watchlist is empty (first-run experience)
  // or when the user explicitly clicks the "+ Add" button. This keeps the
  // panel's resting state clean and content-first (see Watchlist.md).
  const [manageOpen, setManageOpen] = useState<boolean>(false);
  // Chip color mode in the Manage Drawer.
  //   'signal'   — nearest-label bullish/bearish (green/red).
  //   'horizon'  — whole-row greens vs reds verdict.
  //   'bigmoves' — groups 1w movers by signed intensity, so Big Greens /
  //                Big Reds become impossible to miss.
  const [chipColorMode, setChipColorMode] = useState<'signal' | 'horizon' | 'bigmoves'>('horizon');
  // Refine popover — search / sector / sort are collapsed behind a single
  // trigger so the two primary rows (insight bar, segmented control) can
  // breathe.
  const [refineOpen, setRefineOpen] = useState<boolean>(false);
  // Column-header and pill sorts are local to the watchlist. That keeps sort
  // clicks fast because we rank only the curated list, not the full signal set.
  const [wlSortOverride, setWlSortOverride] = useState<boolean>(false);
  useEffect(() => {
    if (symbols.length === 0) setManageOpen(true);
  }, [symbols.length]);

  const sectorOptions = useMemo(() => {
    const s = new Set<string>();
    for (const r of watchlistRows) {
      if (r.sector) s.add(r.sector);
    }
    return Array.from(s).sort((a, b) => a.localeCompare(b));
  }, [watchlistRows]);

  const classifyRow = useCallback((row: SummaryRow): WlClass => {
    const lbl = (row.nearest_label || '').toUpperCase();
    if (lbl === 'STRONG_BUY' || lbl === 'BUY') return 'bullish';
    if (lbl === 'STRONG_SELL' || lbl === 'SELL') return 'bearish';
    return 'neutral';
  }, []);

  // Ticker (as it appears in signal rows, accounting for proxies) → row, so
  // chip rendering can look up the signal bucket for coloring.
  const rowByTicker = useMemo(() => {
    const m = new Map<string, SummaryRow>();
    for (const r of watchlistRows) {
      m.set(extractTicker(r.asset_label), r);
    }
    return m;
  }, [watchlistRows]);

  // Resolve a watchlist symbol to its signal row (via proxy if needed).
  const rowForSymbol = useCallback(
    (sym: string): SummaryRow | undefined => {
      const direct = rowByTicker.get(sym);
      if (direct) return direct;
      const primary = proxyMap[sym];
      if (primary) return rowByTicker.get(primary);
      return undefined;
    },
    [rowByTicker, proxyMap],
  );

  const expAtHorizon = useCallback((row: SummaryRow, horizon = WATCHLIST_BIG_MOVE_HORIZON): number => {
    const sigs = row.horizon_signals as Record<string | number, { exp_ret?: number } | undefined>;
    const sig = sigs?.[horizon] || sigs?.[String(horizon)];
    const r = sig?.exp_ret;
    return Number.isFinite(r) ? (r as number) : 0;
  }, []);

  // 1-week implied return (exp_ret at horizon=7) per watchlist symbol.
  // Used to scale chip color intensity — stronger moves render with deeper
  // saturation so the eye can rank conviction at a glance.
  const exp1w = useCallback(
    (sym: string): number => {
      const row = rowForSymbol(sym);
      if (!row) return 0;
      return expAtHorizon(row);
    },
    [rowForSymbol, expAtHorizon],
  );

  // Max |1w exp_ret| across the current watchlist — normalises intensity
  // so the strongest mover anchors full saturation. Floor at 0.5% to avoid
  // hyper-amplifying tiny moves on a quiet day.
  const maxAbs1w = useMemo(() => {
    let m = 0.005;
    for (const sym of symbols) {
      const v = Math.abs(exp1w(sym));
      if (v > m) m = v;
    }
    return m;
  }, [symbols, exp1w]);

  const signalCounts = useMemo(() => {
    let bull = 0, bear = 0, neut = 0, green = 0, red = 0, mixed = 0, bigGreen = 0, bigRed = 0, reversalBuy = 0, reversalSell = 0;
    const qualityRows = watchlistRows.filter((row) => (qualityScores[extractTicker(row.asset_label)] ?? 50) >= minQuality);
    for (const r of qualityRows) {
      const c = classifyRow(r);
      if (c === 'bullish') bull++;
      else if (c === 'bearish') bear++;
      else neut++;
      const hc = rowHorizonColor(r);
      if (hc === 'greens') green++;
      else if (hc === 'reds') red++;
      else mixed++;
      const move1w = expAtHorizon(r);
      if (move1w >= WATCHLIST_BIG_MOVE_ABS_THRESHOLD) bigGreen++;
      else if (move1w <= -WATCHLIST_BIG_MOVE_ABS_THRESHOLD) bigRed++;
      const flip = reversalFlipForAsset(reversalFlips, r.asset_label);
      if (flip?.signal === 'buy') reversalBuy++;
      else if (flip?.signal === 'sell') reversalSell++;
    }
    return { bull, bear, neut, green, red, mixed, bigGreen, bigRed, reversalBuy, reversalSell };
  }, [watchlistRows, classifyRow, reversalFlips, expAtHorizon, minQuality, qualityScores]);

  const filteredWatchlistRows = useMemo(() => {
    if (wlMissingOnly) return [];
    const q = wlQuery.trim().toLowerCase();
    const rows = watchlistRows.filter((r) => {
      if ((qualityScores[extractTicker(r.asset_label)] ?? 50) < minQuality) return false;
      if (wlSignal !== 'all') {
        if (isReversalQuickFilter(wlSignal)) {
          if (!rowMatchesReversalFilter(r, wlSignal, reversalFlips)) return false;
        } else if (wlSignal === 'bullish' || wlSignal === 'bearish' || wlSignal === 'neutral') {
          if (classifyRow(r) !== wlSignal) return false;
        } else if (wlSignal === 'big_green' || wlSignal === 'big_red') {
          const move1w = expAtHorizon(r);
          if (wlSignal === 'big_green' && move1w < WATCHLIST_BIG_MOVE_ABS_THRESHOLD) return false;
          if (wlSignal === 'big_red' && move1w > -WATCHLIST_BIG_MOVE_ABS_THRESHOLD) return false;
        } else {
          if (rowHorizonColor(r) !== wlSignal) return false;
        }
      }
      if (wlSector !== 'all' && r.sector !== wlSector) return false;
      if (q && !(r.asset_label || '').toLowerCase().includes(q)) return false;
      return true;
    });
    if (wlSortOverride) return sortSummaryRows(rows, wlSortLevels, qualityScores);
    if (wlSort === 'signal') return sortSummaryRows(rows, [{ col: 'signal', dir: 'desc' }, { col: 'asset', dir: 'asc' }], qualityScores);
    if (wlSort === 'momentum') return sortSummaryRows(rows, [{ col: 'momentum', dir: 'desc' }], qualityScores);
    if (wlSort === 'quality') return sortSummaryRows(rows, [{ col: 'quality', dir: 'desc' }], qualityScores);
    if (wlSort === 'risk') return sortSummaryRows(rows, [{ col: 'crash_risk', dir: 'asc' }], qualityScores);
    return sortSummaryRows(rows, [{ col: 'asset', dir: 'asc' }], qualityScores);
  }, [watchlistRows, wlSignal, wlSector, wlQuery, wlSort, wlMissingOnly, wlSortOverride, wlSortLevels, classifyRow, qualityScores, reversalFlips, expAtHorizon, minQuality]);

  // Watchlist sorts are deliberately local now. Sorting 20-100 curated rows is
  // much cheaper than re-sorting and re-rendering the whole Signals surface.
  const handleWatchlistSort = useCallback(
    (col: SortColumn, shift: boolean) => {
      setWlSortOverride(true);
      setWlSortLevels((prev) => nextSortLevels(prev, col, shift));
    },
    [],
  );
  const handleWatchlistPillSort = useCallback(
    (col: SortColumn) => {
      setWlSortOverride(true);
      setWlSortLevels((prev) => nextSortLevels(prev, col, false));
    },
    [],
  );

  // Changing the Sort preset is an explicit reset of the override.
  const setWlSortPreset = useCallback((key: WlSort) => {
    setWlSort(key);
    setWlSortOverride(false);
  }, []);
  const watchlistPresetSortLevels = useMemo((): { col: SortColumn; dir: SortDir }[] => {
    if (wlSort === 'momentum') return [{ col: 'momentum', dir: 'desc' }];
    if (wlSort === 'quality') return [{ col: 'quality', dir: 'desc' }];
    if (wlSort === 'risk') return [{ col: 'crash_risk', dir: 'asc' }];
    return [];
  }, [wlSort]);
  const resetWatchlistSort = useCallback(() => {
    setWlSort('signal');
    setWlSortOverride(false);
    setWlSortLevels([]);
  }, []);
  const removeWatchlistSort = useCallback((col: SortColumn) => {
    setWlSortLevels((prev) => removeSortLevel(prev, col));
  }, []);
  const activeWatchlistSortLevels = wlSortOverride ? wlSortLevels : watchlistPresetSortLevels;

  // Suggested tickers for the first-run empty state. Keeping this tiny and
  // opinionated: one mega-cap tech, one AI darling, one benchmark ETF.
  const suggestedTickers = useMemo(() => ['AAPL', 'NVDA', 'SPY'], []);

  // Keyboard shortcuts. `/` focuses the ticker input (also opens the manage
  // drawer), `A` toggles the drawer, `Esc` closes it when open. We attach to
  // `window` but ignore when the user is typing into a text field.
  useEffect(() => {
    const isTypingTarget = (t: EventTarget | null) => {
      const el = t as HTMLElement | null;
      if (!el) return false;
      const tag = el.tagName;
      return tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT' || el.isContentEditable;
    };
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && manageOpen) {
        setManageOpen(false);
        return;
      }
      if (isTypingTarget(e.target)) return;
      if (e.key === '/') {
        e.preventDefault();
        setManageOpen(true);
        setTimeout(() => inputRef.current?.focus(), 60);
      } else if (e.key === 'a' || e.key === 'A') {
        setManageOpen((v) => !v);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [manageOpen]);

  const hasActiveFilter = wlSignal !== 'all' || wlSector !== 'all' || wlQuery.trim().length > 0 || wlSort !== 'signal' || wlMissingOnly || wlSortOverride || minQuality > 0;
  const hasRefinement = wlSector !== 'all' || wlQuery.trim().length > 0 || wlSort !== 'signal';
  const clearFilters = useCallback(() => {
    setWlSignal('all');
    setWlSector('all');
    setWlQuery('');
    setWlSort('signal');
    setWlMissingOnly(false);
    setWlSortOverride(false);
    onMinQualityChange(0);
  }, [onMinQualityChange]);

  const submit = useCallback(() => {
    const sym = input.trim().toUpperCase();
    if (!sym) return;
    add.mutate(sym, {
      onSuccess: () => setInput(''),
    });
  }, [input, add]);

  const onKey = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (e.key === 'Enter') {
        e.preventDefault();
        submit();
      }
    },
    [submit],
  );

  const addErrorMsg = add.error?.message;

  const watchlistChipSections = useMemo(() => {
    type ChipTone = 'green' | 'red' | 'neutral' | 'missing';
    type ChipSectionKey =
      | 'big_green'
      | 'green'
      | 'soft_green'
      | 'mixed'
      | 'soft_red'
      | 'red'
      | 'big_red'
      | 'missing';
    type ChipMeta = {
      symbol: string;
      title: string;
      tone: ChipTone;
      bg: string;
      color: string;
      border: string;
      dot: string | null;
      fontWeight: 400 | 500 | 600 | 700;
      intensity: number;
      isMissing: boolean;
      percentLabel: string;
    };
    type ChipSection = {
      key: ChipSectionKey;
      label: string;
      hint: string;
      accent: string;
      tint: string;
      border: string;
      items: ChipMeta[];
    };

    const sectionSpecs: Omit<ChipSection, 'items'>[] = [
      {
        key: 'big_green',
        label: 'Big Greens',
        hint: 'largest upside pressure',
        accent: '#34d399',
        tint: 'rgba(16,185,129,0.105)',
        border: 'rgba(16,185,129,0.38)',
      },
      {
        key: 'green',
        label: 'Greens',
        hint: 'positive and building',
        accent: '#6ee7b7',
        tint: 'rgba(52,211,153,0.070)',
        border: 'rgba(52,211,153,0.26)',
      },
      {
        key: 'soft_green',
        label: 'Soft Greens',
        hint: 'positive, low intensity',
        accent: '#86efac',
        tint: 'rgba(134,239,172,0.045)',
        border: 'rgba(134,239,172,0.18)',
      },
      {
        key: 'mixed',
        label: 'Mixed / Flat',
        hint: 'direction still undecided',
        accent: '#cbd5e1',
        tint: 'rgba(148,163,184,0.035)',
        border: 'rgba(148,163,184,0.16)',
      },
      {
        key: 'soft_red',
        label: 'Soft Reds',
        hint: 'negative, low intensity',
        accent: '#fca5a5',
        tint: 'rgba(252,165,165,0.040)',
        border: 'rgba(252,165,165,0.18)',
      },
      {
        key: 'red',
        label: 'Reds',
        hint: 'negative and building',
        accent: '#f87171',
        tint: 'rgba(248,113,113,0.065)',
        border: 'rgba(248,113,113,0.26)',
      },
      {
        key: 'big_red',
        label: 'Big Reds',
        hint: 'largest downside pressure',
        accent: '#ef4444',
        tint: 'rgba(239,68,68,0.095)',
        border: 'rgba(239,68,68,0.36)',
      },
      {
        key: 'missing',
        label: 'Missing',
        hint: 'not in the live signal set',
        accent: '#fcd34d',
        tint: 'rgba(251,191,36,0.060)',
        border: 'rgba(251,191,36,0.28)',
      },
    ];

    const groups = new Map<ChipSectionKey, ChipSection>(
      sectionSpecs.map((spec) => [spec.key, { ...spec, items: [] }]),
    );
    const alphaSort = (a: ChipMeta, b: ChipMeta) =>
      a.symbol.localeCompare(b.symbol, undefined, { numeric: true, sensitivity: 'base' });
    const lerp = (a: number, b: number, t: number) => a + (b - a) * t;
    const pctLabel = (value: number) => {
      if (!Number.isFinite(value) || value === 0) return '0.00%';
      const sign = value > 0 ? '+' : '';
      return `${sign}${(value * 100).toFixed(2)}%`;
    };
    const bucketFor = (tone: ChipTone, absReturn: number, tier: number): ChipSectionKey => {
      if (tone === 'missing') return 'missing';
      if (tone === 'neutral') return 'mixed';
      const size = absReturn >= 0.03 || tier === 3 ? 'big' : absReturn >= 0.015 || tier >= 2 ? 'mid' : 'soft';
      if (tone === 'green') {
        if (size === 'big') return 'big_green';
        if (size === 'mid') return 'green';
        return 'soft_green';
      }
      if (size === 'big') return 'big_red';
      if (size === 'mid') return 'red';
      return 'soft_red';
    };
    const paint = (tone: ChipTone, tier: number, intensity: number) => {
      if (tone === 'missing') {
        return {
          bg: 'rgba(251,191,36,0.08)',
          color: '#fcd34d',
          border: 'rgba(251,191,36,0.26)',
          dot: null,
          fontWeight: 600 as const,
        };
      }
      if (tone === 'neutral') {
        return {
          bg: 'rgba(148,163,184,0.06)',
          color: '#cbd5e1',
          border: 'rgba(148,163,184,0.20)',
          dot: null,
          fontWeight: 500 as const,
        };
      }
      const isGreen = tone === 'green';
      const bgA = lerp(0.06, isGreen ? 0.52 : 0.50, intensity);
      const brA = lerp(0.20, 0.92, intensity);
      return {
        bg: `rgba(${isGreen ? '16,185,129' : '239,68,68'},${bgA.toFixed(3)})`,
        color: isGreen
          ? ['#bbf7d0', '#86efac', '#34d399', '#d1fae5'][tier]
          : ['#fecaca', '#fca5a5', '#f87171', '#fee2e2'][tier],
        border: `rgba(${isGreen ? '16,185,129' : '239,68,68'},${brA.toFixed(3)})`,
        dot: isGreen ? '#10b981' : '#ef4444',
        fontWeight: (tier >= 2 ? 700 : tier === 1 ? 600 : 500) as 500 | 600 | 700,
      };
    };

    for (const sym of symbols) {
      const isMissing = missingSymbols.includes(sym);
      const row = isMissing ? undefined : rowForSymbol(sym);
      const r7d = isMissing ? 0 : exp1w(sym);
      const absReturn = Math.abs(r7d);
      const raw = Math.min(1, absReturn / maxAbs1w);
      const intensity = Math.max(0.08, Math.sqrt(raw));
      const tier = intensity < 0.30 ? 0 : intensity < 0.55 ? 1 : intensity < 0.80 ? 2 : 3;
      let tone: ChipTone = 'neutral';
      let titleDetail = 'mixed';

      if (isMissing) {
        tone = 'missing';
        titleDetail = 'not in current signal set';
      } else if (row && chipColorMode === 'horizon') {
        const horizonTone = rowHorizonColor(row);
        tone = horizonTone === 'greens' ? 'green' : horizonTone === 'reds' ? 'red' : 'neutral';
        titleDetail =
          horizonTone === 'greens'
            ? `green horizon tape, 1w ${pctLabel(r7d)}`
            : horizonTone === 'reds'
            ? `red horizon tape, 1w ${pctLabel(r7d)}`
            : `mixed horizon tape, 1w ${pctLabel(r7d)}`;
      } else if (row && chipColorMode === 'bigmoves') {
        tone = r7d > 0 ? 'green' : r7d < 0 ? 'red' : 'neutral';
        titleDetail = `1w ${pctLabel(r7d)}`;
      } else if (row) {
        const cls = classifyRow(row);
        tone = cls === 'bullish' ? 'green' : cls === 'bearish' ? 'red' : 'neutral';
        titleDetail = `${row.nearest_label || 'HOLD'}, 1w ${pctLabel(r7d)}`;
      }

      const style = paint(tone, tier, intensity);
      if (chipColorMode === 'bigmoves' && absReturn < WATCHLIST_BIG_MOVE_ABS_THRESHOLD) {
        continue;
      }

      const sectionKey = bucketFor(tone, absReturn, tier);
      if (chipColorMode === 'bigmoves' && sectionKey !== 'big_green' && sectionKey !== 'big_red') {
        continue;
      }
      groups.get(sectionKey)?.items.push({
        symbol: sym,
        title: `${sym} - ${titleDetail}`,
        tone,
        bg: style.bg,
        color: style.color,
        border: style.border,
        dot: style.dot,
        fontWeight: style.fontWeight,
        intensity,
        isMissing,
        percentLabel: isMissing ? '' : pctLabel(r7d),
      });
    }

    return sectionSpecs
      .map((spec) => {
        const section = groups.get(spec.key)!;
        section.items.sort(alphaSort);
        return section;
      })
      .filter((section) =>
        section.items.length > 0 &&
        (chipColorMode !== 'bigmoves' || section.key === 'big_green' || section.key === 'big_red')
      );
  }, [
    symbols,
    missingSymbols,
    rowForSymbol,
    exp1w,
    maxAbs1w,
    chipColorMode,
    classifyRow,
  ]);

  const chipToneCounts = useMemo(() => {
    let green = 0;
    let red = 0;
    for (const section of watchlistChipSections) {
      for (const chip of section.items) {
        if (chip.tone === 'green') green++;
        else if (chip.tone === 'red') red++;
      }
    }
    return { green, red };
  }, [watchlistChipSections]);

  return (
    <div className="flex flex-col gap-3 fade-up-delay-3">
      {/* ─────────────────────────────────────────────────────────────────
         TIER 1 — Insight Bar
         A single headline line. Anchors the panel. Clickable count phrases
         filter the list; "+ Add" toggles the slide-up manage drawer.
         See Watchlist.md §3.
         ───────────────────────────────────────────────────────────────── */}
      <div
        className="flex flex-wrap items-center gap-2 px-4 py-3"
        style={{
          background:
            'linear-gradient(180deg, rgba(255,255,255,0.028) 0%, rgba(255,255,255,0.008) 100%)',
          border: '1px solid rgba(255,255,255,0.06)',
          borderRadius: '16px',
          boxShadow:
            '0 1px 0 rgba(255,255,255,0.05) inset, 0 10px 32px -18px rgba(0,0,0,0.55)',
          backdropFilter: 'blur(14px)',
        }}
      >
        {/* Anchor: Star + label + tracked count */}
        <div className="flex items-center gap-2.5">
          <div
            className="w-8 h-8 rounded-[10px] flex items-center justify-center"
            style={{
              background:
                'linear-gradient(180deg, rgba(167,139,250,0.18) 0%, rgba(167,139,250,0.08) 100%)',
              color: '#c4b5fd',
              boxShadow:
                '0 0 0 1px rgba(167,139,250,0.22), 0 6px 18px -10px rgba(167,139,250,0.55)',
            }}
          >
            <Star className="w-4 h-4" />
          </div>
          <div className="leading-tight">
            <div className="text-[13px] font-semibold text-[#e2e8f0] tracking-[-0.01em]">
              Watchlist
            </div>
            <div className="text-[11px] text-[var(--text-secondary)] tabular-nums">
              {symbols.length} tracked
              {watchlistRows.length !== symbols.length && (
                <> · {watchlistRows.length} live</>
              )}
            </div>
          </div>
        </div>

        {/* Headline verdict: bullish · greens · missing. Each is a chip
            that filters the list when clicked (one-click quick filter). */}
        {symbols.length > 0 && (
          <div
            className="flex flex-wrap items-center gap-1.5 ml-1 md:ml-3"
            aria-label="Watchlist summary"
          >
            {([
              {
                key: 'bullish' as const,
                value: signalCounts.bull,
                label: 'bullish',
                accent: '#34d399',
                tint: 'rgba(52,211,153,0.10)',
                border: 'rgba(52,211,153,0.26)',
                onClick: () => {
                  setWlMissingOnly(false);
                  setWlSignal(wlSignal === 'bullish' ? 'all' : 'bullish');
                },
                active: wlSignal === 'bullish',
              },
              {
                key: 'greens' as const,
                value: signalCounts.green,
                label: 'all greens',
                accent: '#6ee7b7',
                tint: 'rgba(110,231,183,0.10)',
                border: 'rgba(110,231,183,0.26)',
                onClick: () => {
                  setWlMissingOnly(false);
                  setWlSignal(wlSignal === 'greens' ? 'all' : 'greens');
                },
                active: wlSignal === 'greens',
              },
              ...(signalCounts.bear > 0
                ? [{
                    key: 'bearish' as const,
                    value: signalCounts.bear,
                    label: 'bearish',
                    accent: '#f87171',
                    tint: 'rgba(248,113,113,0.10)',
                    border: 'rgba(248,113,113,0.26)',
                    onClick: () => {
                      setWlMissingOnly(false);
                      setWlSignal(wlSignal === 'bearish' ? 'all' : 'bearish');
                    },
                    active: wlSignal === 'bearish',
                  }]
                : []),
              ...(missingSymbols.length > 0
                ? [{
                    key: 'missing' as const,
                    value: missingSymbols.length,
                    label: 'missing',
                    accent: '#fcd34d',
                    tint: 'rgba(251,191,36,0.10)',
                    border: 'rgba(251,191,36,0.26)',
                    onClick: () => {
                      const next = !wlMissingOnly;
                      setWlMissingOnly(next);
                      if (next) setWlSignal('all');
                    },
                    active: wlMissingOnly,
                  }]
                : []),
            ]).map((phrase, idx, arr) => (
              <Fragment key={phrase.key}>
                <button
                  type="button"
                  onClick={phrase.onClick}
                  className="group inline-flex items-baseline gap-1 px-2 py-0.5 rounded-md transition-all duration-[140ms] active:scale-[0.97]"
                  style={{
                    background: phrase.active ? phrase.tint : 'transparent',
                    boxShadow: phrase.active ? `inset 0 0 0 1px ${phrase.border}` : 'none',
                  }}
                  title={phrase.active ? `Clear ${phrase.label} filter` : `Show ${phrase.label}`}
                  aria-pressed={phrase.active}
                >
                  <span
                    className="text-[15px] font-semibold tabular-nums leading-none transition-colors"
                    style={{ color: phrase.active ? phrase.accent : '#e2e8f0' }}
                  >
                    {phrase.value}
                  </span>
                  <span
                    className="text-[12px] transition-colors"
                    style={{
                      color: phrase.active
                        ? phrase.accent
                        : 'var(--text-secondary)',
                    }}
                  >
                    {phrase.label}
                  </span>
                </button>
                {idx < arr.length - 1 && (
                  <span className="text-[var(--text-secondary)] opacity-40 text-[13px] leading-none">·</span>
                )}
              </Fragment>
            ))}
          </div>
        )}

        {/* Right cluster: Refine + Add */}
        <div className="ml-auto flex items-center gap-1.5">
          {symbols.length > 0 && (
            <button
              type="button"
              onClick={() => setRefineOpen((v) => !v)}
              className="inline-flex items-center gap-1.5 px-2.5 py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-[140ms] active:scale-[0.97]"
              style={{
                background: refineOpen || hasRefinement ? 'rgba(255,255,255,0.05)' : 'transparent',
                border: `1px solid ${refineOpen || hasRefinement ? 'rgba(255,255,255,0.10)' : 'rgba(255,255,255,0.04)'}`,
                color: hasRefinement ? '#c4b5fd' : 'var(--text-secondary)',
              }}
              title={refineOpen ? 'Hide refine controls' : 'Search, sort, filter by sector'}
              aria-expanded={refineOpen}
            >
              <SlidersHorizontal className="w-3.5 h-3.5" />
              Refine
              {hasRefinement && (
                <span
                  className="inline-block rounded-full"
                  style={{ width: 5, height: 5, background: '#a78bfa' }}
                />
              )}
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              setManageOpen((v) => !v);
              if (!manageOpen) setTimeout(() => inputRef.current?.focus(), 80);
            }}
            className="inline-flex items-center gap-1.5 px-3 py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-[140ms] active:scale-[0.97] hover:-translate-y-[1px]"
            style={{
              background:
                'linear-gradient(180deg, rgba(167,139,250,0.20) 0%, rgba(167,139,250,0.10) 100%)',
              color: '#e9d5ff',
              border: '1px solid rgba(167,139,250,0.30)',
              boxShadow:
                '0 1px 0 rgba(255,255,255,0.08) inset, 0 6px 18px -10px rgba(167,139,250,0.55)',
            }}
            title={manageOpen ? 'Close manage drawer (A)' : 'Add / manage tickers (A)'}
            aria-expanded={manageOpen}
          >
            {manageOpen ? (
              <X className="w-3.5 h-3.5" />
            ) : (
              <Plus className="w-3.5 h-3.5" />
            )}
            {manageOpen ? 'Close' : 'Add'}
          </button>
        </div>
      </div>

      {/* ─────────────────────────────────────────────────────────────────
         MANAGE DRAWER — slides open when manageOpen.
         Uses a CSS grid-rows trick for a smooth height transition.
         ───────────────────────────────────────────────────────────────── */}
      <div
        className="grid transition-[grid-template-rows] duration-[260ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
        style={{ gridTemplateRows: manageOpen ? '1fr' : '0fr' }}
        aria-hidden={!manageOpen}
      >
        <div className="overflow-hidden">
          <div
            className="p-4"
            style={{
              background:
                'linear-gradient(180deg, rgba(255,255,255,0.022) 0%, rgba(255,255,255,0.006) 100%)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '16px',
              boxShadow:
                '0 1px 0 rgba(255,255,255,0.05) inset, 0 8px 28px -14px rgba(0,0,0,0.6)',
              backdropFilter: 'blur(12px)',
            }}
          >
            {/* Input row */}
            <div className="flex items-center gap-2">
              <div
                className="flex items-center gap-2 flex-1 px-3 py-[8px] focus-ring transition-all duration-200"
                style={{
                  background: 'rgba(255,255,255,0.02)',
                  border: '1px solid rgba(255,255,255,0.06)',
                  borderRadius: '10px',
                }}
              >
                <Search className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                <input
                  ref={inputRef}
                  type="text"
                  value={input}
                  onChange={(e) => setInput(e.target.value.toUpperCase())}
                  onKeyDown={onKey}
                  placeholder="Add ticker (AAPL, BTC-USD, EURUSD=X)…"
                  spellCheck={false}
                  autoCapitalize="characters"
                  autoCorrect="off"
                  className="flex-1 bg-transparent outline-none text-sm text-[#e2e8f0] placeholder:text-[var(--text-secondary)]"
                  disabled={add.isPending}
                />
                {input && (
                  <button
                    type="button"
                    onClick={() => setInput('')}
                    className="text-[var(--text-secondary)] hover:text-[#e2e8f0]"
                    title="Clear"
                  >
                    <X className="w-3.5 h-3.5" />
                  </button>
                )}
              </div>
              <button
                type="button"
                onClick={submit}
                disabled={!input.trim() || add.isPending}
                className="inline-flex items-center gap-1.5 px-3 py-[8px] rounded-[10px] text-sm font-medium transition-all disabled:opacity-40 disabled:cursor-not-allowed active:scale-[0.97]"
                style={{
                  background:
                    'linear-gradient(180deg, rgba(167,139,250,0.22) 0%, rgba(167,139,250,0.10) 100%)',
                  color: '#e9d5ff',
                  border: '1px solid rgba(167,139,250,0.30)',
                }}
              >
                {add.isPending ? (
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                ) : (
                  <Plus className="w-3.5 h-3.5" />
                )}
                Track
              </button>
            </div>
            {addErrorMsg && (
              <div
                className="mt-2 flex items-center gap-1.5 text-[11px]"
                style={{ color: '#fca5a5' }}
              >
                <AlertTriangle className="w-3 h-3" />
                {addErrorMsg}
              </div>
            )}

            {/* Suggestions when empty */}
            {symbols.length === 0 && (
              <div className="mt-3">
                <div className="text-[10px] uppercase tracking-[0.12em] font-semibold text-[var(--text-secondary)] mb-2">
                  Try one of these
                </div>
                <div className="flex flex-wrap gap-1.5">
                  {suggestedTickers.map((sym) => (
                    <button
                      key={sym}
                      type="button"
                      onClick={() => add.mutate(sym)}
                      disabled={add.isPending}
                      className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md text-[11px] font-medium transition-all active:scale-[0.97] hover:-translate-y-[1px] disabled:opacity-40"
                      style={{
                        background: 'rgba(167,139,250,0.08)',
                        color: '#c4b5fd',
                        border: '1px solid rgba(167,139,250,0.22)',
                      }}
                      title={`Track ${sym}`}
                    >
                      <Plus className="w-3 h-3" />
                      {sym}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {/* Chips grid — color-coded by current signal state OR by
                whole-row horizon verdict (greens vs reds) depending on
                `chipColorMode`. A small segmented toggle lets the user flip
                between the two lenses. */}
            {symbols.length > 0 && (
              <>
                <div className="mt-3 flex items-center gap-2">
                  <span className="text-[10px] uppercase tracking-[0.12em] font-semibold text-[var(--text-secondary)]">
                    Color by
                  </span>
                  <div
                    className="inline-flex items-center rounded-[10px] p-0.5 gap-0.5"
                    style={{
                      background: 'rgba(255,255,255,0.025)',
                      border: '1px solid rgba(255,255,255,0.05)',
                    }}
                    role="tablist"
                    aria-label="Chip color mode"
                  >
                    {([
                      { k: 'signal' as const, label: 'Signal', accent: '#a78bfa', bg: 'rgba(167,139,250,0.14)', border: 'rgba(167,139,250,0.28)' },
                      { k: 'horizon' as const, label: 'Greens / Reds', accent: '#6ee7b7', bg: 'rgba(110,231,183,0.14)', border: 'rgba(110,231,183,0.28)' },
                      { k: 'bigmoves' as const, label: 'Big Greens / Big Reds', accent: '#fde68a', bg: 'rgba(253,230,138,0.14)', border: 'rgba(253,230,138,0.32)' },
                    ]).map((opt) => {
                      const active = chipColorMode === opt.k;
                      return (
                        <button
                          key={opt.k}
                          type="button"
                          role="tab"
                          aria-selected={active}
                          onClick={() => setChipColorMode(opt.k)}
                          className="inline-flex items-center gap-1.5 px-2.5 py-[4px] rounded-[8px] text-[11px] font-medium transition-all duration-[140ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97]"
                          style={{
                            background: active ? opt.bg : 'transparent',
                            color: active ? opt.accent : 'var(--text-secondary)',
                            boxShadow: active ? `0 0 0 1px ${opt.border}` : 'none',
                          }}
                          title={
                            opt.k === 'signal'
                              ? 'Color chips by nearest-horizon signal (bullish / bearish)'
                              : opt.k === 'horizon'
                              ? 'Color chips by whole-row verdict (all greens vs all reds)'
                              : 'Show only big 1-week green/red moves (>=3%)'
                          }
                        >
                          {opt.k === 'horizon' && (
                            <span className="inline-flex items-center gap-0.5">
                              <span className="inline-block rounded-full" style={{ width: 6, height: 6, background: '#34d399' }} />
                              <span className="inline-block rounded-full" style={{ width: 6, height: 6, background: '#ef4444' }} />
                            </span>
                          )}
                          {opt.k === 'bigmoves' && (
                            <span className="inline-flex items-center gap-0.5">
                              <span className="inline-block rounded-full" style={{ width: 8, height: 8, background: '#10b981', boxShadow: '0 0 0 1.5px rgba(16,185,129,0.45)' }} />
                              <span className="inline-block rounded-full" style={{ width: 8, height: 8, background: '#ef4444', boxShadow: '0 0 0 1.5px rgba(239,68,68,0.45)' }} />
                            </span>
                          )}
                          {opt.label}
                        </button>
                      );
                    })}
                  </div>
                </div>
                <div
                  className="mt-3 overflow-hidden"
                  style={{
                    background:
                      'linear-gradient(180deg, rgba(255,255,255,0.026) 0%, rgba(255,255,255,0.006) 100%)',
                    border: '1px solid rgba(255,255,255,0.065)',
                    borderRadius: '14px',
                    boxShadow:
                      '0 1px 0 rgba(255,255,255,0.05) inset, 0 12px 32px -20px rgba(0,0,0,0.7)',
                  }}
                  aria-label="Watchlist tickers grouped by intensity"
                >
                  <div className="flex flex-wrap items-center justify-between gap-2 px-3 py-2 border-b border-white/[0.045]">
                    <div className="flex items-center gap-2">
                      <Layers className="w-3.5 h-3.5" style={{ color: '#c4b5fd' }} />
                      <div className="leading-tight">
                        <div className="text-[11px] font-semibold text-[#e2e8f0]">
                          {chipColorMode === 'bigmoves' ? 'Big move map' : 'Intensity map'}
                        </div>
                        <div className="text-[10px] text-[var(--text-secondary)]">
                          {chipColorMode === 'bigmoves'
                            ? 'only big green/red moves, alphabetized inside each band'
                            : 'grouped by move strength, alphabetized inside each band'}
                        </div>
                      </div>
                    </div>
                    <div className="inline-flex items-center gap-1 text-[10px] text-[var(--text-secondary)] tabular-nums">
                      <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: '#10b981' }} />
                      {chipToneCounts.green}
                      <span className="mx-1 opacity-40">/</span>
                      <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: '#ef4444' }} />
                      {chipToneCounts.red}
                    </div>
                  </div>

                  <div className="divide-y divide-white/[0.04]">
                    {watchlistChipSections.length === 0 ? (
                      <div className="px-3 py-4 text-[11px] text-[var(--text-secondary)]">
                        No big green/red moves right now.
                      </div>
                    ) : watchlistChipSections.map((section) => (
                      <section
                        key={section.key}
                        className="px-3 py-2.5"
                        style={{
                          background: `linear-gradient(90deg, ${section.tint} 0%, transparent 54%)`,
                        }}
                      >
                        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
                          <div className="flex items-center gap-2 min-w-0">
                            <span
                              className="inline-block w-1 h-4 rounded-full"
                              style={{
                                background: section.accent,
                                boxShadow: `0 0 16px ${section.border}`,
                              }}
                            />
                            <div className="min-w-0">
                              <div className="flex items-baseline gap-1.5">
                                <span
                                  className="text-[11px] font-semibold"
                                  style={{ color: section.accent }}
                                >
                                  {section.label}
                                </span>
                                <span className="text-[10px] tabular-nums text-[var(--text-secondary)]">
                                  {section.items.length}
                                </span>
                              </div>
                              <div className="text-[10px] text-[var(--text-secondary)] truncate">
                                {section.hint}
                              </div>
                            </div>
                          </div>
                          <span
                            className="text-[9px] uppercase tracking-[0.10em] font-semibold"
                            style={{ color: 'rgba(148,163,184,0.62)' }}
                          >
                            A-Z
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-1.5">
                          {section.items.map((chip) => (
                            <span
                              key={chip.symbol}
                              className="group inline-flex items-center gap-1 pl-2 pr-1 py-1 rounded-md text-[11px] tabular-nums transition-all duration-[140ms] hover:-translate-y-[1px]"
                              style={{
                                background: chip.bg,
                                color: chip.color,
                                border: `1px solid ${chip.border}`,
                                fontWeight: chip.fontWeight,
                              }}
                              title={chip.title}
                            >
                              {chip.isMissing ? (
                                <AlertTriangle className="w-3 h-3" />
                              ) : chip.dot ? (
                                <span
                                  className="inline-block rounded-full"
                                  style={{
                                    width: 6,
                                    height: 6,
                                    background: chip.dot,
                                    boxShadow: `0 0 0 2px ${chip.dot}${Math.round((0x14 + (0x55 - 0x14) * chip.intensity)).toString(16).padStart(2, '0')}`,
                                  }}
                                />
                              ) : null}
                              <span>{chip.symbol}</span>
                              {chip.percentLabel && (
                                <span className="hidden sm:inline opacity-55 text-[10px]">
                                  {chip.percentLabel}
                                </span>
                              )}
                              <button
                                type="button"
                                onClick={() => remove.mutate(chip.symbol)}
                                disabled={remove.isPending}
                                className="ml-0.5 w-4 h-4 inline-flex items-center justify-center rounded opacity-60 hover:opacity-100 hover:bg-white/5 disabled:opacity-30"
                                title={`Remove ${chip.symbol}`}
                              >
                                <X className="w-3 h-3" />
                              </button>
                            </span>
                          ))}
                        </div>
                      </section>
                    ))}
                  </div>
                </div>
              </>
            )}

            {/* Keyboard shortcut hint row */}
            <div className="mt-3 pt-3 border-t border-white/[0.04] flex flex-wrap items-center gap-x-3 gap-y-1 text-[10px] text-[var(--text-secondary)]">
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">/</kbd>
                focus input
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">A</kbd>
                toggle drawer
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">Esc</kbd>
                close
              </span>
              <span className="inline-flex items-center gap-1">
                <kbd className="px-1.5 py-px rounded bg-white/[0.05] border border-white/[0.06] text-[10px] font-mono">Enter</kbd>
                track ticker
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* ── Table / empty state ─────────────────────────────────────── */}
      {isLoading ? (
        <div className="text-center py-8 text-sm text-[var(--text-secondary)]">
          Loading watchlist…
        </div>
      ) : symbols.length === 0 ? (
        <div
          className="relative overflow-hidden flex flex-col items-center justify-center text-center py-16 px-6"
          style={{
            background:
              'linear-gradient(180deg, rgba(167,139,250,0.04) 0%, rgba(167,139,250,0.01) 60%, rgba(255,255,255,0.005) 100%)',
            border: '1px dashed rgba(167,139,250,0.18)',
            borderRadius: '20px',
          }}
        >
          <div
            aria-hidden
            className="absolute inset-0 pointer-events-none"
            style={{
              background:
                'radial-gradient(420px 180px at 50% 0%, rgba(167,139,250,0.10), transparent 70%)',
            }}
          />
          <div
            className="relative w-16 h-16 rounded-[22px] flex items-center justify-center mb-5"
            style={{
              background:
                'linear-gradient(180deg, rgba(167,139,250,0.22) 0%, rgba(167,139,250,0.08) 100%)',
              color: '#c4b5fd',
              boxShadow:
                '0 0 0 1px rgba(167,139,250,0.26), 0 12px 36px -12px rgba(167,139,250,0.55), inset 0 1px 0 rgba(255,255,255,0.08)',
            }}
          >
            <Star className="w-7 h-7" />
          </div>
          <h3 className="relative text-[17px] font-semibold text-[#e2e8f0] mb-1 tracking-[-0.01em]">
            Build your watchlist
          </h3>
          <p className="relative text-[13px] text-[var(--text-secondary)] max-w-sm mb-5 leading-relaxed">
            Pin tickers you care about for a focused view. Signals, momentum, and risk
            — just for them.
          </p>
          <div className="relative flex flex-wrap items-center justify-center gap-1.5 mb-5">
            {suggestedTickers.map((sym) => (
              <button
                key={sym}
                type="button"
                onClick={() => add.mutate(sym)}
                disabled={add.isPending}
                className="inline-flex items-center gap-1 px-3 py-[7px] rounded-[10px] text-[12px] font-medium transition-all duration-[140ms] active:scale-[0.97] hover:-translate-y-[1px] disabled:opacity-40"
                style={{
                  background: 'rgba(167,139,250,0.10)',
                  color: '#c4b5fd',
                  border: '1px solid rgba(167,139,250,0.26)',
                }}
                title={`Track ${sym}`}
              >
                <Plus className="w-3.5 h-3.5" />
                {sym}
              </button>
            ))}
          </div>
          <button
            type="button"
            onClick={() => { setManageOpen(true); setTimeout(() => inputRef.current?.focus(), 80); }}
            className="relative inline-flex items-center gap-1.5 px-3 py-[7px] rounded-[10px] text-[12px] font-medium transition-all active:scale-[0.97]"
            style={{
              background: 'transparent',
              color: 'var(--text-secondary)',
              border: '1px solid rgba(255,255,255,0.08)',
            }}
          >
            <Search className="w-3.5 h-3.5" />
            Or add your own
          </button>
        </div>
      ) : watchlistRows.length === 0 ? (
        <div
          className="flex flex-col items-center justify-center text-center py-12 px-6"
          style={{
            background:
              'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%)',
            border: '1px dashed rgba(251,191,36,0.18)',
            borderRadius: '16px',
          }}
        >
          <AlertTriangle className="w-8 h-8 mb-3" style={{ color: '#fcd34d' }} />
          <h3 className="text-base font-semibold text-[#e2e8f0] mb-1">
            No live signals for your watchlist
          </h3>
          <p className="text-sm text-[var(--text-secondary)] max-w-md">
            None of your saved tickers are in the current signal set. They are
            still persisted — add them to the engine or refresh signals to see
            them here.
          </p>
        </div>
      ) : (
        <>
          {/* ── Unified segmented control ──────────────────────────────
             One row. Six segments. No duplicate groups. Missing is a
             segment here (only when missingSymbols.length > 0). Color
             accent per Watchlist.md §8. */}
          <div
            className="flex flex-wrap items-center gap-2 px-3 py-2"
            style={{
              background:
                'linear-gradient(180deg, rgba(15,23,42,0.72) 0%, rgba(15,23,42,0.55) 100%)',
              border: '1px solid rgba(255,255,255,0.06)',
              borderRadius: '14px',
              boxShadow:
                '0 1px 0 rgba(255,255,255,0.04) inset, 0 8px 28px -18px rgba(0,0,0,0.6)',
              backdropFilter: 'blur(14px)',
            }}
            role="tablist"
            aria-label="Watchlist view"
          >
            <div
              className="inline-flex items-center rounded-[11px] p-0.5 gap-0.5"
              style={{
                background: 'rgba(255,255,255,0.025)',
                border: '1px solid rgba(255,255,255,0.05)',
              }}
            >
              {([
                { k: 'all' as const, label: 'All', count: watchlistRows.length, accent: '#e2e8f0', bg: 'rgba(226,232,240,0.10)', border: 'rgba(226,232,240,0.20)', dot: null as string | null },
                { k: 'bullish' as const, label: 'Bullish', count: signalCounts.bull, accent: '#34d399', bg: 'rgba(52,211,153,0.12)', border: 'rgba(52,211,153,0.28)', dot: '#34d399' },
                { k: 'bearish' as const, label: 'Bearish', count: signalCounts.bear, accent: '#f87171', bg: 'rgba(248,113,113,0.12)', border: 'rgba(248,113,113,0.28)', dot: '#f87171' },
                { k: 'greens' as const, label: 'Greens', count: signalCounts.green, accent: '#6ee7b7', bg: 'rgba(110,231,183,0.12)', border: 'rgba(110,231,183,0.28)', dot: '#6ee7b7' },
                { k: 'reds' as const, label: 'Reds', count: signalCounts.red, accent: '#fca5a5', bg: 'rgba(252,165,165,0.12)', border: 'rgba(252,165,165,0.28)', dot: '#fca5a5' },
                { k: 'reversal_buy' as const, label: 'Buy Rev', count: reversalFlipsLoading ? '—' : signalCounts.reversalBuy, accent: '#00f5a0', bg: 'rgba(0,245,160,0.12)', border: 'rgba(0,245,160,0.30)', dot: '#00f5a0' },
                { k: 'reversal_sell' as const, label: 'Sell Rev', count: reversalFlipsLoading ? '—' : signalCounts.reversalSell, accent: '#ff375f', bg: 'rgba(255,55,95,0.12)', border: 'rgba(255,55,95,0.30)', dot: '#ff375f' },
              ]).map((seg) => {
                const active = !wlMissingOnly && wlSignal === seg.k;
                return (
                  <button
                    key={seg.k}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => {
                      setWlMissingOnly(false);
                      setWlSignal(seg.k as WlSignal);
                    }}
                    className="inline-flex items-center gap-1.5 px-2.5 py-[5px] rounded-[9px] text-[11px] font-medium transition-all duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97]"
                    style={{
                      background: active ? seg.bg : 'transparent',
                      color: active ? seg.accent : 'var(--text-secondary)',
                      boxShadow: active ? `0 0 0 1px ${seg.border}` : 'none',
                    }}
                    title={`${seg.label} (${seg.count})`}
                  >
                    {seg.dot && (
                      <span
                        className="inline-block rounded-full transition-opacity"
                        style={{
                          width: 6,
                          height: 6,
                          background: seg.dot,
                          opacity: active ? 1 : 0.55,
                        }}
                      />
                    )}
                    {seg.label}
                    <span
                      className="tabular-nums transition-opacity"
                      style={{ opacity: active ? 0.9 : 0.55 }}
                    >
                      {seg.count}
                    </span>
                  </button>
                );
              })}
              {missingSymbols.length > 0 && (
                <button
                  type="button"
                  role="tab"
                  aria-selected={wlMissingOnly}
                  onClick={() => {
                    const next = !wlMissingOnly;
                    setWlMissingOnly(next);
                    if (next) setWlSignal('all');
                  }}
                  className="inline-flex items-center gap-1.5 px-2.5 py-[5px] rounded-[9px] text-[11px] font-medium transition-all duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97]"
                  style={{
                    background: wlMissingOnly ? 'rgba(251,191,36,0.14)' : 'transparent',
                    color: wlMissingOnly ? '#fcd34d' : 'var(--text-secondary)',
                    boxShadow: wlMissingOnly ? '0 0 0 1px rgba(251,191,36,0.30)' : 'none',
                  }}
                  title={`Missing (${missingSymbols.length})`}
                >
                  <AlertTriangle className="w-3 h-3" />
                  Missing
                  <span
                    className="tabular-nums transition-opacity"
                    style={{ opacity: wlMissingOnly ? 0.9 : 0.55 }}
                  >
                    {missingSymbols.length}
                  </span>
                </button>
              )}
            </div>

            <div
              className="inline-flex items-center rounded-[12px] p-0.5 gap-0.5"
              style={{
                background:
                  'linear-gradient(135deg, rgba(15,23,42,0.58), rgba(255,255,255,0.032))',
                border: '1px solid rgba(255,255,255,0.07)',
                boxShadow:
                  '0 1px 0 rgba(255,255,255,0.05) inset, 0 10px 26px -22px rgba(0,0,0,0.8)',
              }}
              aria-label="Big move filters"
            >
              <span className="hidden sm:inline-flex items-center gap-1.5 px-2 text-[10px] font-semibold uppercase text-[var(--text-secondary)]">
                <Filter className="w-3 h-3" />
                1W &gt;= 3%
              </span>
              {([
                {
                  k: 'big_green' as const,
                  label: 'Big Greens',
                  count: signalCounts.bigGreen,
                  accent: '#34d399',
                  bg: 'linear-gradient(135deg, rgba(16,185,129,0.20), rgba(20,184,166,0.08))',
                  border: 'rgba(52,211,153,0.34)',
                  glow: 'rgba(16,185,129,0.24)',
                  icon: <ArrowUpRight className="w-3.5 h-3.5" />,
                },
                {
                  k: 'big_red' as const,
                  label: 'Big Reds',
                  count: signalCounts.bigRed,
                  accent: '#fb7185',
                  bg: 'linear-gradient(135deg, rgba(244,63,94,0.20), rgba(251,113,133,0.08))',
                  border: 'rgba(251,113,133,0.34)',
                  glow: 'rgba(244,63,94,0.24)',
                  icon: <ArrowDownRight className="w-3.5 h-3.5" />,
                },
              ]).map((seg) => {
                const active = !wlMissingOnly && wlSignal === seg.k;
                return (
                  <button
                    key={seg.k}
                    type="button"
                    role="tab"
                    aria-selected={active}
                    onClick={() => {
                      setWlMissingOnly(false);
                      setWlSignal(active ? 'all' : seg.k);
                    }}
                    className="inline-flex items-center gap-1.5 px-2.5 py-[5px] rounded-[9px] text-[11px] font-semibold transition-all duration-[180ms] ease-[cubic-bezier(0.16,1,0.3,1)] active:scale-[0.97] hover:-translate-y-[1px]"
                    style={{
                      background: active ? seg.bg : 'transparent',
                      color: active ? seg.accent : 'var(--text-secondary)',
                      boxShadow: active
                        ? `0 0 0 1px ${seg.border}, 0 10px 24px -18px ${seg.glow}`
                        : 'none',
                    }}
                    title={`${seg.label}: 1-week expected move at least ${(WATCHLIST_BIG_MOVE_ABS_THRESHOLD * 100).toFixed(0)}% in magnitude`}
                  >
                    <span
                      className="inline-flex items-center justify-center rounded-full"
                      style={{
                        width: 18,
                        height: 18,
                        background: active ? 'rgba(255,255,255,0.08)' : 'rgba(255,255,255,0.035)',
                        color: active ? seg.accent : 'rgba(148,163,184,0.78)',
                      }}
                    >
                      {seg.icon}
                    </span>
                    <span>{seg.label}</span>
                    <span className="tabular-nums" style={{ opacity: active ? 0.95 : 0.58 }}>
                      {seg.count}
                    </span>
                  </button>
                );
              })}
            </div>

            {/* Sorted-by indicator (shown only when column-click override
                is active, i.e. user clicked a column header). */}
            {wlSortOverride && activeWatchlistSortLevels.length > 0 && (
              <div
                className="inline-flex items-center gap-1 px-2 py-[3px] rounded-[8px] text-[10px]"
                style={{
                  background: 'rgba(167,139,250,0.08)',
                  color: '#c4b5fd',
                  border: '1px solid rgba(167,139,250,0.22)',
                }}
                title="Column-click sort active. Switch the sort dropdown in Refine to reset."
              >
                <span className="opacity-80">sorted by column</span>
                <button
                  type="button"
                  onClick={() => { setWlSortOverride(false); }}
                  className="opacity-70 hover:opacity-100 -mr-0.5"
                  title="Reset to preset sort"
                >
                  <X className="w-3 h-3" />
                </button>
              </div>
            )}

            {/* Right cluster: result count + clear */}
            <div className="ml-auto flex items-center gap-1.5">
              {hasActiveFilter && (
                <button
                  type="button"
                  onClick={clearFilters}
                  className="inline-flex items-center gap-1 px-2 py-[4px] rounded-[8px] text-[10px] font-medium transition-all active:scale-[0.97]"
                  style={{
                    background: 'transparent',
                    border: '1px solid rgba(255,255,255,0.06)',
                    color: 'var(--text-secondary)',
                  }}
                  title="Reset all watchlist filters"
                >
                  <X className="w-3 h-3" />
                  Clear
                </button>
              )}
              <div
                className="text-[11px] tabular-nums"
                style={{ color: 'var(--text-secondary)' }}
                title={`${wlMissingOnly ? missingSymbols.length : filteredWatchlistRows.length} shown of ${watchlistRows.length} with live signals`}
              >
                {wlMissingOnly ? missingSymbols.length : filteredWatchlistRows.length}
                <span className="opacity-50"> / {watchlistRows.length}</span>
              </div>
            </div>
          </div>

          <div className="mt-2">
            <QualityFloorSlider
              value={minQuality}
              onChange={onMinQualityChange}
              compact
            />
          </div>

          <div
            className="mt-2 px-3 py-2"
            style={{
              background:
                'linear-gradient(135deg, rgba(15,23,42,0.46), rgba(255,255,255,0.018))',
              border: '1px solid rgba(255,255,255,0.055)',
              borderRadius: '14px',
              boxShadow: '0 1px 0 rgba(255,255,255,0.035) inset',
            }}
          >
            <SortPillStrip
              sortLevels={activeWatchlistSortLevels}
              onSort={handleWatchlistPillSort}
              onClear={resetWatchlistSort}
              title="Sort"
              subtitle="watchlist"
              horizons={sortHorizons}
            />
          </div>

          {/* ── Refine popover ─────────────────────────────────────────
             Collapsed by default. Opens from the "Refine" button in the
             Insight Bar. Holds search, sector, and preset sort. Uses the
             same grid-template-rows trick as the Manage drawer. */}
          <div
            className="grid transition-[grid-template-rows] duration-[260ms] ease-[cubic-bezier(0.16,1,0.3,1)]"
            style={{ gridTemplateRows: refineOpen ? '1fr' : '0fr' }}
            aria-hidden={!refineOpen}
          >
            <div className="overflow-hidden">
              <div
                className="flex flex-wrap items-center gap-2 px-3 py-2 mt-1"
                style={{
                  background: 'rgba(255,255,255,0.015)',
                  border: '1px solid rgba(255,255,255,0.05)',
                  borderRadius: '12px',
                }}
              >
                <div
                  className="flex items-center gap-1.5 px-2 py-[5px] flex-1 min-w-[180px]"
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    borderRadius: '10px',
                  }}
                >
                  <Search className="w-3.5 h-3.5 text-[var(--text-secondary)]" />
                  <input
                    type="text"
                    value={wlQuery}
                    onChange={(e) => setWlQuery(e.target.value)}
                    placeholder="Filter by name or ticker…"
                    spellCheck={false}
                    className="flex-1 bg-transparent outline-none text-[12px] text-[#e2e8f0] placeholder:text-[var(--text-secondary)]"
                  />
                  {wlQuery && (
                    <button
                      type="button"
                      onClick={() => setWlQuery('')}
                      className="text-[var(--text-secondary)] hover:text-[#e2e8f0]"
                      title="Clear search"
                    >
                      <X className="w-3 h-3" />
                    </button>
                  )}
                </div>
                {sectorOptions.length > 0 && (
                  <select
                    value={wlSector}
                    onChange={(e) => setWlSector(e.target.value)}
                    className="text-[11px] px-2 py-[5px] rounded-[10px] outline-none"
                    style={{
                      background: 'rgba(255,255,255,0.02)',
                      border: '1px solid rgba(255,255,255,0.06)',
                      color: wlSector === 'all' ? 'var(--text-secondary)' : '#e2e8f0',
                    }}
                    title="Filter by sector"
                  >
                    <option value="all">All sectors</option>
                    {sectorOptions.map((s) => (
                      <option key={s} value={s}>{s}</option>
                    ))}
                  </select>
                )}
                <select
                  value={wlSort}
                  onChange={(e) => setWlSortPreset(e.target.value as WlSort)}
                  className="text-[11px] px-2 py-[5px] rounded-[10px] outline-none"
                  style={{
                    background: 'rgba(255,255,255,0.02)',
                    border: '1px solid rgba(255,255,255,0.06)',
                    color: '#e2e8f0',
                  }}
                  title="Preset sort order (column click overrides)"
                >
                  <option value="signal">Sort: Signal (best first)</option>
                  <option value="momentum">Sort: Momentum</option>
                  <option value="quality">Sort: Quality</option>
                  <option value="risk">Sort: Risk (low first)</option>
                  <option value="alpha">Sort: Alphabetical</option>
                </select>
              </div>
            </div>
          </div>

          {wlMissingOnly ? (
            <div
              className="flex flex-col gap-3 px-4 py-4"
              style={{
                background:
                  'linear-gradient(180deg, rgba(251,191,36,0.05) 0%, rgba(251,191,36,0.01) 100%)',
                border: '1px dashed rgba(251,191,36,0.22)',
                borderRadius: '14px',
              }}
            >
              <div className="flex items-center gap-2 text-[13px] font-medium" style={{ color: '#fcd34d' }}>
                <AlertTriangle className="w-4 h-4" />
                Missing from current signal set ({missingSymbols.length})
              </div>
              {missingSymbols.length === 0 ? (
                <div className="text-[11px] text-[var(--text-secondary)]">
                  All watchlist tickers have a live signal row.
                </div>
              ) : (
                <div className="flex flex-wrap gap-1.5">
                  {missingSymbols.map((sym) => (
                    <span
                      key={sym}
                      className="inline-flex items-center gap-1 px-2 py-1 rounded-md text-[11px] font-medium tabular-nums"
                      style={{
                        background: 'rgba(251,191,36,0.08)',
                        color: '#fcd34d',
                        border: '1px solid rgba(251,191,36,0.22)',
                      }}
                      title={`${sym} — not tuned or not in latest signal snapshot`}
                    >
                      <AlertTriangle className="w-3 h-3" />
                      {sym}
                      <button
                        type="button"
                        onClick={() => remove.mutate(sym)}
                        disabled={remove.isPending}
                        className="ml-0.5 opacity-70 hover:opacity-100 disabled:opacity-30"
                        title={`Remove ${sym}`}
                      >
                        <X className="w-3 h-3" />
                      </button>
                    </span>
                  ))}
                </div>
              )}
              <div className="text-[11px] text-[var(--text-secondary)]">
                These tickers exist in your watchlist but no row was emitted for
                them. That usually means they haven't been tuned yet or the
                latest snapshot is still refreshing.
              </div>
            </div>
          ) : filteredWatchlistRows.length === 0 ? (
            <div
              className="flex flex-col items-center justify-center text-center py-10 px-6"
              style={{
                background:
                  'linear-gradient(180deg, rgba(255,255,255,0.02) 0%, rgba(255,255,255,0.005) 100%)',
                border: '1px dashed rgba(255,255,255,0.08)',
                borderRadius: '14px',
              }}
            >
              <Filter className="w-6 h-6 mb-2" style={{ color: 'var(--text-secondary)' }} />
              <div className="text-sm font-medium text-[#e2e8f0]">No matches</div>
              <div className="text-[11px] text-[var(--text-secondary)] mt-1">
                No watchlist entries match your current filters.
              </div>
              <button
                type="button"
                onClick={clearFilters}
                className="mt-3 inline-flex items-center gap-1 px-2.5 py-1 rounded-[10px] text-[11px] font-medium"
                style={{
                  background: 'var(--violet-15)',
                  color: '#c4b5fd',
                  border: '1px solid rgba(167,139,250,0.25)',
                }}
              >
                <X className="w-3 h-3" />
                Clear filters
              </button>
            </div>
          ) : (
            <AllAssetsTable
              rows={filteredWatchlistRows}
              horizons={horizons}
              updatedAsset={updatedAsset}
              sortLevels={activeWatchlistSortLevels}
              onSort={handleWatchlistSort}
              onRemoveSort={removeWatchlistSort}
              expandedRow={expanded}
              onExpandRow={setExpanded}
              qualityScores={qualityScores}
              onNavigateChart={onNavigateChart}
              disablePagination
              detailDefaultChartType="area"
            />
          )}
        </>
      )}
    </div>
  );
}
