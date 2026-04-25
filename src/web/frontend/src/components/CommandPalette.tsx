import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api, type SignalSummaryData } from '../api';
import {
  Search,
  LayoutDashboard,
  Signal,
  ShieldAlert,
  LineChart,
  Settings,
  Database,
  Swords,
  Stethoscope,
  HeartPulse,
  RefreshCw,
  Zap,
  Clock,
  BarChart3,
} from 'lucide-react';

/* ─── Types ─────────────────────────────────────────────────────── */
interface PaletteItem {
  id: string;
  label: string;
  sublabel?: string;
  category: 'recent' | 'page' | 'asset' | 'action';
  icon?: typeof LayoutDashboard;
  badge?: { text: string; variant: 'bull' | 'bear' | 'neutral' };
  onSelect: () => void;
}

/* ─── Recent items persistence ──────────────────────────────────── */
const RECENT_KEY = 'command-palette-recent';
const MAX_RECENT = 8;

function getRecent(): { id: string; label: string; sublabel?: string; category: string }[] {
  try {
    return JSON.parse(localStorage.getItem(RECENT_KEY) || '[]');
  } catch {
    return [];
  }
}

function pushRecent(item: { id: string; label: string; sublabel?: string; category: string }) {
  try {
    const prev = getRecent().filter(r => r.id !== item.id);
    prev.unshift(item);
    localStorage.setItem(RECENT_KEY, JSON.stringify(prev.slice(0, MAX_RECENT)));
  } catch { /* noop */ }
}

/* ─── Pages definition ──────────────────────────────────────────── */
const PAGES = [
  { id: 'page:/', label: 'Overview', icon: LayoutDashboard, to: '/' },
  { id: 'page:/signals', label: 'Signals', icon: Signal, to: '/signals' },
  { id: 'page:/risk', label: 'Risk Dashboard', icon: ShieldAlert, to: '/risk' },
  { id: 'page:/charts', label: 'Charts', icon: LineChart, to: '/charts' },
  { id: 'page:/tuning', label: 'Tuning', icon: Settings, to: '/tuning' },
  { id: 'page:/data', label: 'Data', icon: Database, to: '/data' },
  { id: 'page:/arena', label: 'Arena', icon: Swords, to: '/arena' },
  { id: 'page:/diagnostics', label: 'Diagnostics', icon: Stethoscope, to: '/diagnostics' },
  { id: 'page:/services', label: 'Services', icon: HeartPulse, to: '/services' },
  { id: 'page:/diagnostics/profitability', label: 'Profitability', icon: BarChart3, to: '/diagnostics/profitability' },
];

/* ─── Actions definition ────────────────────────────────────────── */
const ACTIONS = [
  { id: 'action:refresh-data', label: 'Refresh Data', icon: RefreshCw, fn: 'refreshData' as const },
  { id: 'action:retune', label: 'Retune Models', icon: Settings, fn: 'retune' as const },
  { id: 'action:compute-signals', label: 'Compute Signals', icon: Zap, fn: 'computeSignals' as const },
  { id: 'action:compute-risk', label: 'Compute Risk', icon: ShieldAlert, fn: 'computeRisk' as const },
];

/* ─── Component ─────────────────────────────────────────────────── */
export default function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const navigate = useNavigate();
  const inputRef = useRef<HTMLInputElement>(null);
  const listRef = useRef<HTMLDivElement>(null);
  const [query, setQuery] = useState('');
  const [activeIdx, setActiveIdx] = useState(0);

  /* Fetch signal summary for asset search */
  const signalSummaryQ = useQuery({
    queryKey: ['signalSummary'],
    queryFn: api.signalSummary,
    staleTime: 60_000,
    enabled: open,
  });

  /* Focus input when opened */
  useEffect(() => {
    if (open) {
      setQuery('');
      setActiveIdx(0);
      requestAnimationFrame(() => inputRef.current?.focus());
    }
  }, [open]);

  /* Record selection + navigation */
  const selectItem = useCallback((item: PaletteItem) => {
    pushRecent({ id: item.id, label: item.label, sublabel: item.sublabel, category: item.category });
    item.onSelect();
    onClose();
  }, [onClose]);

  /* Action handlers */
  const runAction = useCallback(async (fn: string) => {
    try {
      switch (fn) {
        case 'refreshData': await api.triggerDataRefresh(); break;
        case 'retune': await api.triggerTuning(); break;
        case 'computeSignals': await api.triggerSignals(); break;
        case 'computeRisk': await api.triggerRisk(); break;
      }
    } catch { /* toast will handle */ }
  }, []);

  /* Build results list */
  const items = useMemo<PaletteItem[]>(() => {
    const q = query.toLowerCase().trim();
    const results: PaletteItem[] = [];

    /* Recent (shown when empty) */
    if (!q) {
      const recents = getRecent();
      for (const r of recents) {
        const page = PAGES.find(p => p.id === r.id);
        if (page) {
          results.push({
            id: r.id,
            label: r.label,
            sublabel: r.sublabel,
            category: 'recent',
            icon: page.icon,
            onSelect: () => navigate(page.to),
          });
        } else if (r.id.startsWith('asset:')) {
          results.push({
            id: r.id,
            label: r.label,
            sublabel: r.sublabel,
            category: 'recent',
            icon: LineChart,
            onSelect: () => navigate(`/charts/${r.label}`),
          });
        }
      }
    }

    /* Pages */
    const matchedPages = q
      ? PAGES.filter(p => p.label.toLowerCase().includes(q))
      : PAGES;
    for (const p of matchedPages) {
      results.push({
        id: p.id,
        label: p.label,
        category: 'page',
        icon: p.icon,
        onSelect: () => navigate(p.to),
      });
    }

    /* Assets */
    if (q && signalSummaryQ.data) {
      const rows = (signalSummaryQ.data as SignalSummaryData).summary_rows || [];
      const matched = rows.filter(r =>
        r.asset_label.toLowerCase().includes(q) ||
        (r.sector || '').toLowerCase().includes(q)
      ).slice(0, 12);
      for (const r of matched) {
        const direction = (r.nearest_label || '').toLowerCase();
        const badge = direction.includes('buy')
          ? { text: r.nearest_label || 'Buy', variant: 'bull' as const }
          : direction.includes('sell')
            ? { text: r.nearest_label || 'Sell', variant: 'bear' as const }
            : { text: r.nearest_label || 'Hold', variant: 'neutral' as const };
        results.push({
          id: `asset:${r.asset_label}`,
          label: r.asset_label,
          sublabel: r.sector || undefined,
          category: 'asset',
          icon: LineChart,
          badge,
          onSelect: () => navigate(`/charts/${r.asset_label}`),
        });
      }
    }

    /* Actions */
    const matchedActions = q
      ? ACTIONS.filter(a => a.label.toLowerCase().includes(q))
      : ACTIONS;
    for (const a of matchedActions) {
      results.push({
        id: a.id,
        label: a.label,
        category: 'action',
        icon: a.icon,
        onSelect: () => runAction(a.fn),
      });
    }

    return results;
  }, [query, signalSummaryQ.data, navigate, runAction]);

  /* Keyboard nav */
  useEffect(() => {
    setActiveIdx(0);
  }, [query]);

  const onKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'ArrowDown') {
      e.preventDefault();
      setActiveIdx(i => Math.min(i + 1, items.length - 1));
    } else if (e.key === 'ArrowUp') {
      e.preventDefault();
      setActiveIdx(i => Math.max(i - 1, 0));
    } else if (e.key === 'Enter' && items[activeIdx]) {
      e.preventDefault();
      selectItem(items[activeIdx]);
    } else if (e.key === 'Escape') {
      e.preventDefault();
      onClose();
    }
  }, [items, activeIdx, selectItem, onClose]);

  /* Scroll active into view */
  useEffect(() => {
    if (!listRef.current) return;
    const el = listRef.current.querySelector(`[data-idx="${activeIdx}"]`);
    if (el) el.scrollIntoView({ block: 'nearest' });
  }, [activeIdx]);

  if (!open) return null;

  /* Group items by category for display */
  const grouped = new Map<string, PaletteItem[]>();
  let runningIdx = 0;
  const indexMap = new Map<PaletteItem, number>();
  for (const item of items) {
    indexMap.set(item, runningIdx++);
    const cat = item.category;
    if (!grouped.has(cat)) grouped.set(cat, []);
    grouped.get(cat)!.push(item);
  }

  const categoryLabels: Record<string, string> = {
    recent: 'Recent',
    page: 'Pages',
    asset: 'Assets',
    action: 'Actions',
  };

  const badgeStyle = (variant: string) => {
    switch (variant) {
      case 'bull': return 'bg-[var(--emerald-12)] text-[var(--accent-emerald)]';
      case 'bear': return 'bg-[var(--rose-12)] text-[var(--accent-rose)]';
      default: return 'bg-[var(--violet-8)] text-[#C4B5FD]';
    }
  };

  return (
    <div
      className="fixed inset-0 z-[100] flex items-start justify-center pt-[15vh]"
      style={{
        background: 'rgba(3,0,20,0.85)',
        backdropFilter: 'blur(40px) saturate(1.8)',
        WebkitBackdropFilter: 'blur(40px) saturate(1.8)',
        animation: 'palette-bg-in 300ms cubic-bezier(0.2,0,0,1)',
      }}
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div
        className="w-full max-w-[560px] rounded-2xl border overflow-hidden"
        style={{
          background: 'linear-gradient(160deg, rgba(45,27,105,0.35) 0%, rgba(10,10,35,0.9) 50%, rgba(12,20,69,0.5) 100%)',
          borderColor: 'var(--border-glow)',
          boxShadow: '0 0 0 1px var(--border-glow), 0 24px 80px var(--violet-15), 0 0 120px var(--violet-6)',
          animation: 'palette-in 300ms cubic-bezier(0.2,0,0,1)',
        }}
      >
        {/* Search input */}
        <div className="relative px-4 pt-4 pb-3">
          <div className="flex items-center gap-3">
            <Search className="w-5 h-5 flex-shrink-0" style={{ color: 'var(--accent-violet)', opacity: 0.6 }} />
            <input
              ref={inputRef}
              type="text"
              value={query}
              onChange={e => setQuery(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Search pages, assets, actions..."
              className="flex-1 bg-transparent text-[15px] font-medium outline-none placeholder:text-[#6b7a90]"
              style={{ color: 'var(--text-luminous)' }}
              autoComplete="off"
              spellCheck={false}
            />
          </div>
          {/* Violet gradient underline */}
          <div
            className="mt-3 h-[2px] rounded-full"
            style={{
              background: 'linear-gradient(90deg, transparent 0%, var(--accent-violet) 50%, transparent 100%)',
              opacity: query ? 1 : undefined,
              animation: query ? 'none' : 'palette-pulse 2s ease-in-out infinite',
            }}
          />
        </div>

        {/* Results */}
        <div ref={listRef} className="max-h-[360px] overflow-y-auto px-2 pb-2">
          {items.length === 0 && query && (
            <div className="px-4 py-8 text-center text-[13px]" style={{ color: 'var(--text-muted)' }}>
              No results for "{query}"
            </div>
          )}

          {Array.from(grouped.entries()).map(([cat, catItems]) => (
            <div key={cat} className="mb-1">
              <div className="px-3 py-1.5 text-[10px] font-semibold uppercase tracking-widest flex items-center gap-2"
                   style={{ color: 'var(--text-violet)' }}>
                {cat === 'recent' && <Clock className="w-3 h-3" style={{ color: 'var(--text-muted)' }} />}
                {categoryLabels[cat] || cat}
              </div>
              {catItems.map((item) => {
                const idx = indexMap.get(item) ?? 0;
                const isActive = idx === activeIdx;
                const Icon = item.icon;

                return (
                  <button
                    key={item.id}
                    data-idx={idx}
                    onClick={() => selectItem(item)}
                    onMouseEnter={() => setActiveIdx(idx)}
                    className={`w-full flex items-center gap-3 px-3 py-2 rounded-xl text-left text-[13px] transition-all duration-120 ${
                      isActive
                        ? 'border-l-2'
                        : 'border-l-2 border-transparent'
                    }`}
                    style={{
                      background: isActive ? 'var(--void-hover)' : 'transparent',
                      borderLeftColor: isActive ? 'var(--accent-violet)' : 'transparent',
                    }}
                  >
                    {Icon && (
                      <Icon
                        className="w-4 h-4 flex-shrink-0"
                        style={{ color: cat === 'action' ? 'var(--accent-cyan)' : 'var(--accent-violet)' }}
                      />
                    )}
                    <div className="flex-1 min-w-0">
                      <span style={{ color: isActive ? 'var(--text-luminous)' : 'var(--text-primary)' }}>
                        {item.label}
                      </span>
                      {item.sublabel && (
                        <span className="ml-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                          {item.sublabel}
                        </span>
                      )}
                    </div>
                    {item.badge && (
                      <span className={`micro-badge ${badgeStyle(item.badge.variant)}`}>
                        {item.badge.text}
                      </span>
                    )}
                    {cat === 'recent' && (
                      <Clock className="w-3 h-3 flex-shrink-0" style={{ color: 'var(--text-muted)' }} />
                    )}
                  </button>
                );
              })}
            </div>
          ))}
        </div>

        {/* Keyboard hints bar */}
        <div
          className="px-4 py-2 flex items-center gap-4 text-[10px] border-t"
          style={{
            background: 'var(--void-active)',
            borderTopColor: 'var(--border-void)',
            color: 'var(--text-muted)',
          }}
        >
          <span><kbd className="kbd-hint">Enter</kbd> to select</span>
          <span><kbd className="kbd-hint">Tab</kbd> to autocomplete</span>
          <span><kbd className="kbd-hint">Esc</kbd> to close</span>
        </div>
      </div>
    </div>
  );
}
