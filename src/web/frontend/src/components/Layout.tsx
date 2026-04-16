import { Outlet, NavLink, useLocation } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { api, type SignalStats, type TuneStats, type DataSummary, type ArenaStatus, type ServicesHealth, type DiagCalibrationFailures } from '../api';
import {
  LayoutDashboard,
  Signal,
  ShieldAlert,
  LineChart,
  Settings,
  Database,
  Swords,
  Activity,
  HeartPulse,
  Stethoscope,
  ChevronsLeft,
  ChevronsRight,
  Grid3X3,
} from 'lucide-react';
import { useState, useEffect, useCallback, useRef } from 'react';
import CommandPalette from './CommandPalette';
import BreadcrumbBar from './BreadcrumbBar';
import StatusStrip from './StatusStrip';
import AmbientOrbs from './AmbientOrbs';
import { KeyboardShortcutOverlay } from './KeyboardShortcuts';
import { initTilt } from '../utils/tilt';

/* ─── Types ─────────────────────────────────────────────────────── */
interface NavItem {
  to: string;
  label: string;
  icon: typeof LayoutDashboard;
  badgeFn?: () => { text: string; color: string } | null;
  tooltipFn?: () => string[];
}

/* ─── Sidebar collapse persistence ──────────────────────────────── */
const COLLAPSE_KEY = 'sidebar-collapsed';
function getSavedCollapsed(): boolean {
  try { return localStorage.getItem(COLLAPSE_KEY) === '1'; } catch { return false; }
}

/* ─── Badge color helpers ───────────────────────────────────────── */
const badgeColors = {
  emerald: 'bg-[var(--emerald-12)] text-[var(--accent-emerald)]',
  rose: 'bg-[var(--rose-12)] text-[var(--accent-rose)]',
  amber: 'bg-[var(--amber-12)] text-[var(--accent-amber)]',
  violet: 'bg-[var(--violet-12)] text-[#C4B5FD]',
  fuchsia: 'bg-[rgba(226,122,245,0.12)] text-[var(--accent-fuchsia)]',
  cyan: 'bg-[rgba(56,217,245,0.12)] text-[var(--accent-cyan)]',
};

/* ─── Layout component ──────────────────────────────────────────── */
export default function Layout() {
  const location = useLocation();
  const [collapsed, setCollapsed] = useState(getSavedCollapsed);
  const [tooltip, setTooltip] = useState<{ item: NavItem; rect: DOMRect } | null>(null);
  const [paletteOpen, setPaletteOpen] = useState(false);
  const tooltipTimer = useRef<ReturnType<typeof setTimeout>>();

  /* Persist collapse */
  useEffect(() => {
    try { localStorage.setItem(COLLAPSE_KEY, collapsed ? '1' : '0'); } catch { /* noop */ }
  }, [collapsed]);

  /* 3D tilt for hover-lift cards */
  useEffect(() => {
    const cleanup = initTilt();
    return cleanup;
  }, []);

  /* Cmd+B toggle, Cmd+K palette */
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === 'b') {
        e.preventDefault();
        setCollapsed(c => !c);
      }
      if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
        e.preventDefault();
        setPaletteOpen(o => !o);
      }
    }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, []);

  /* ── Data queries (reuse existing caches) ──────────────────────── */
  const signalStatsQ = useQuery({
    queryKey: ['signalStats'],
    queryFn: api.signalStats,
    staleTime: 60_000,
    retry: false,
  });
  const tuneStatsQ = useQuery({
    queryKey: ['tuneStats'],
    queryFn: api.tuneStats,
    staleTime: 120_000,
    retry: false,
  });
  const dataQ = useQuery({
    queryKey: ['dataStatus'],
    queryFn: api.dataStatus,
    staleTime: 120_000,
    retry: false,
  });
  const arenaQ = useQuery({
    queryKey: ['arenaStatus'],
    queryFn: api.arenaStatus,
    staleTime: 120_000,
    retry: false,
  });
  const healthQ = useQuery({
    queryKey: ['servicesHealth'],
    queryFn: api.servicesHealth,
    refetchInterval: 30_000,
    retry: false,
  });
  const diagFailQ = useQuery({
    queryKey: ['diagCalibrationFailures'],
    queryFn: api.diagCalibrationFailures,
    staleTime: 120_000,
    retry: false,
  });
  const riskQ = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 60_000,
    retry: false,
  });

  /* Derived micro-indicator data */
  const ss = signalStatsQ.data as SignalStats | undefined;
  const ts = tuneStatsQ.data as TuneStats | undefined;
  const ds = dataQ.data as DataSummary | undefined;
  const as2 = arenaQ.data as ArenaStatus | undefined;
  const hs = healthQ.data as ServicesHealth | undefined;
  const df = diagFailQ.data as DiagCalibrationFailures | undefined;
  const rs = riskQ.data;

  const allServicesOk = hs
    ? hs.api.status === 'ok' && hs.signal_cache.status !== 'missing' && hs.price_data.status === 'ok'
    : true;

  /* ── Navigation items with badge + tooltip functions ───────────── */
  const navItems: NavItem[] = [
    {
      to: '/', label: 'Overview', icon: LayoutDashboard,
      tooltipFn: () => {
        if (!ss) return ['Loading...'];
        return [
          `${ss.total_assets} assets tracked`,
          `${ss.strong_buy_signals + ss.strong_sell_signals} strong signals`,
          `${ss.failed} failed assets`,
        ];
      },
    },
    {
      to: '/heatmap', label: 'Heatmap', icon: Grid3X3,
      badgeFn: () => {
        if (!ss) return null;
        const ct = ss.strong_buy_signals + ss.strong_sell_signals;
        if (ct === 0) return null;
        return { text: String(ct), color: ss.strong_buy_signals >= ss.strong_sell_signals ? 'emerald' : 'rose' };
      },
      tooltipFn: () => {
        if (!ss) return ['Loading...'];
        return [
          `${ss.total_assets} assets across all sectors`,
          'Signal star-map with filtering',
        ];
      },
    },
    {
      to: '/signals', label: 'Signals', icon: Signal,
      badgeFn: () => {
        if (!ss) return null;
        const ct = ss.strong_buy_signals + ss.strong_sell_signals;
        if (ct === 0) return null;
        return { text: String(ct), color: ss.strong_buy_signals >= ss.strong_sell_signals ? 'emerald' : 'rose' };
      },
      tooltipFn: () => {
        if (!ss) return ['Loading...'];
        return [
          `${ss.buy_signals + ss.strong_buy_signals} buy / ${ss.sell_signals + ss.strong_sell_signals} sell`,
          `${ss.hold_signals} hold, ${ss.exit_signals} exit`,
          ss.cache_age_seconds != null ? `Updated ${Math.round(ss.cache_age_seconds / 60)}m ago` : 'No cache',
        ];
      },
    },
    {
      to: '/risk', label: 'Risk', icon: ShieldAlert,
      badgeFn: () => {
        if (!rs) return null;
        const t = rs.combined_temperature;
        const color = t < 0.3 ? 'emerald' : t < 0.7 ? 'amber' : 'rose';
        return { text: t.toFixed(1), color };
      },
      tooltipFn: () => {
        if (!rs) return ['Loading...'];
        return [
          `Temperature: ${rs.combined_temperature.toFixed(2)} (${rs.status})`,
          `Risk: ${rs.risk_temperature.toFixed(2)} | Metals: ${rs.metals_temperature.toFixed(2)}`,
          `Market: ${rs.market_temperature.toFixed(2)}`,
        ];
      },
    },
    {
      to: '/charts', label: 'Charts', icon: LineChart,
      tooltipFn: () => ['Interactive charting with forecasts', 'Select any tracked asset'],
    },
    {
      to: '/tuning', label: 'Tuning', icon: Settings,
      badgeFn: () => {
        if (!ts || ts.total === 0) return null;
        const rate = Math.round((ts.pit_pass / ts.total) * 100);
        return { text: `${rate}%`, color: rate >= 80 ? 'emerald' : rate >= 60 ? 'amber' : 'rose' };
      },
      tooltipFn: () => {
        if (!ts) return ['Loading...'];
        return [
          `${ts.total} assets tuned`,
          `PIT: ${ts.pit_pass} pass / ${ts.pit_fail} fail / ${ts.pit_unknown} unknown`,
          `${Object.keys(ts.models_distribution).length} model types`,
        ];
      },
    },
    {
      to: '/data', label: 'Data', icon: Database,
      badgeFn: () => {
        if (!ds) return null;
        if (ds.stale_files > 0) return { text: String(ds.stale_files), color: 'rose' };
        return null;
      },
      tooltipFn: () => {
        if (!ds) return ['Loading...'];
        return [
          `${ds.total_files} files, ${ds.total_size_mb.toFixed(1)} MB`,
          `${ds.fresh_files} fresh, ${ds.stale_files} stale`,
          ds.oldest_hours != null ? `Oldest: ${Math.round(ds.oldest_hours)}h ago` : '',
        ].filter(Boolean);
      },
    },
    {
      to: '/arena', label: 'Arena', icon: Swords,
      badgeFn: () => {
        if (!as2) return null;
        if (as2.safe_storage_count > 0) return { text: String(as2.safe_storage_count), color: 'fuchsia' };
        return null;
      },
      tooltipFn: () => {
        if (!as2) return ['Loading...'];
        return [
          `${as2.safe_storage_count} models in safe storage`,
          `${as2.experimental_count} experimental`,
          `Benchmark: ${as2.benchmark_symbols.length} symbols`,
        ];
      },
    },
    {
      to: '/diagnostics', label: 'Diagnostics', icon: Stethoscope,
      badgeFn: () => {
        if (!df) return null;
        if (df.count > 0) return { text: String(df.count), color: 'amber' };
        return null;
      },
      tooltipFn: () => {
        if (!df) return ['Loading...'];
        return [
          `${df.count} calibration failures`,
          df.file_exists ? 'Failure file exists' : 'No failure file',
        ];
      },
    },
    {
      to: '/services', label: 'Services', icon: HeartPulse,
      badgeFn: () => null, // pulse dot handled separately
      tooltipFn: () => {
        if (!hs) return ['Loading...'];
        return [
          `API: ${hs.api.status} (${hs.api.uptime_human})`,
          `Memory: ${hs.api.memory_mb.toFixed(0)} MB`,
          `Errors: ${hs.recent_errors.length}`,
        ];
      },
    },
  ];

  /* ── Tooltip handlers ──────────────────────────────────────────── */
  const showTooltip = useCallback((item: NavItem, el: HTMLElement) => {
    clearTimeout(tooltipTimer.current);
    tooltipTimer.current = setTimeout(() => {
      setTooltip({ item, rect: el.getBoundingClientRect() });
    }, 400);
  }, []);

  const hideTooltip = useCallback(() => {
    clearTimeout(tooltipTimer.current);
    setTooltip(null);
  }, []);

  /* ── Render ────────────────────────────────────────────────────── */
  const sidebarWidth = collapsed ? 56 : 220;

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--void)' }}>
      {/* Living background */}
      <AmbientOrbs />

      {/* Ambient Status Strip */}
      <StatusStrip />

      {/* Sidebar */}
      <aside
        className="cosmic-sidebar flex-shrink-0 flex flex-col overflow-hidden relative z-10"
        style={{ width: sidebarWidth }}
        aria-label="Main navigation"
      >
        {/* Logo */}
        <div className={`px-4 py-6 ${collapsed ? 'flex items-center justify-center' : ''}`}>
          {collapsed ? (
            <div className="w-9 h-9 rounded-xl flex items-center justify-center"
                 style={{
                   background: 'linear-gradient(135deg, rgba(139,92,246,0.18) 0%, rgba(56,217,245,0.08) 100%)',
                   boxShadow: '0 0 20px var(--violet-12)',
                 }}>
              <Activity className="w-4.5 h-4.5" style={{ color: 'var(--accent-violet-bright)' }} />
            </div>
          ) : (
            <div className="flex items-center gap-3.5">
              <div className="relative">
                <div
                  className="w-10 h-10 rounded-xl flex items-center justify-center"
                  style={{
                    background: 'linear-gradient(135deg, rgba(139,92,246,0.18) 0%, rgba(56,217,245,0.08) 100%)',
                    boxShadow: '0 0 24px var(--violet-12)',
                    border: '1px solid var(--violet-12)',
                  }}
                >
                  <Activity className="w-5 h-5" style={{ color: 'var(--accent-violet-bright)' }} />
                </div>
                <span
                  className={`absolute -top-0.5 -right-0.5 w-2.5 h-2.5 rounded-full ring-2 pulse-dot`}
                  style={{
                    background: allServicesOk ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                    ringColor: 'var(--void)',
                    boxShadow: `0 0 8px ${allServicesOk ? 'rgba(62,232,165,0.6)' : 'rgba(255,107,138,0.6)'}`,
                  }}
                />
              </div>
              <div>
                <h1 className="text-[14px] font-semibold gradient-text tracking-tight leading-tight">
                  Signal Engine
                </h1>
                <p className="text-[10px] font-medium tracking-widest uppercase mt-0.5" style={{ color: 'var(--text-muted)', letterSpacing: '0.12em' }}>
                  BMA + Kalman v5.30
                </p>
              </div>
            </div>
          )}
        </div>

        <div className="divider-fade mx-4" />

        {/* Navigation */}
        <nav className="flex-1 py-3 px-2.5 space-y-0.5 overflow-y-auto" aria-label="Pages">
          {navItems.map((item) => {
            const badge = item.badgeFn?.();
            const isActive = item.to === '/'
              ? location.pathname === '/'
              : location.pathname.startsWith(item.to);
            const Icon = item.icon;

            return (
              <div
                key={item.to}
                className="relative"
                onMouseEnter={(e) => showTooltip(item, e.currentTarget as HTMLElement)}
                onMouseLeave={hideTooltip}
              >
                <NavLink
                  to={item.to}
                  end={item.to === '/'}
                  className={`flex items-center gap-3 rounded-xl text-[13px] font-medium transition-all duration-200 ${
                    collapsed ? 'justify-center px-2 py-2.5' : 'px-3 py-2.5'
                  } ${
                    isActive
                      ? 'nav-active-indicator nav-cosmic-active'
                      : 'hover:bg-[var(--violet-5)]'
                  }`}
                  style={{
                    color: isActive ? 'var(--accent-violet-bright)' : 'var(--text-muted)',
                  }}
                >
                  <Icon className="w-[18px] h-[18px] flex-shrink-0" style={{ opacity: isActive ? 1 : 0.7 }} />

                  {!collapsed && (
                    <>
                      <span
                        className="flex-1 truncate"
                        style={{
                          color: isActive ? 'var(--accent-violet-bright)' : 'var(--text-secondary)',
                        }}
                      >
                        {item.label}
                      </span>

                      {/* Micro-indicators */}
                      {item.to === '/services' ? (
                        <span
                          className="w-2 h-2 rounded-full ml-auto pulse-dot"
                          style={{
                            background: allServicesOk ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                            boxShadow: `0 0 8px ${allServicesOk ? 'var(--emerald-50)' : 'var(--rose-50)'}`,
                          }}
                        />
                      ) : badge ? (
                        <span
                          className={`micro-badge ml-auto ${badgeColors[badge.color as keyof typeof badgeColors] || badgeColors.violet}`}
                        >
                          {badge.text}
                        </span>
                      ) : null}
                    </>
                  )}
                </NavLink>
              </div>
            );
          })}
        </nav>

        {/* Collapse toggle */}
        <div className="px-3 py-4">
          <div className="divider-fade mb-3" />
          <button
            onClick={() => setCollapsed(c => !c)}
            className="flex items-center justify-center w-full rounded-xl py-2 transition-all duration-200 hover:bg-[rgba(139,92,246,0.07)]"
            style={{ color: 'var(--text-muted)' }}
            title={collapsed ? 'Expand sidebar (Cmd+B)' : 'Collapse sidebar (Cmd+B)'}
          >
            {collapsed ? (
              <ChevronsRight className="w-4 h-4" />
            ) : (
              <div className="flex items-center gap-2 text-[10px]">
                <ChevronsLeft className="w-3.5 h-3.5" />
                <span className="tracking-widest uppercase font-medium" style={{ color: 'var(--text-muted)', letterSpacing: '0.1em' }}>Collapse</span>
              </div>
            )}
          </button>
          {!collapsed && (
            <p className="text-[9px] font-medium tracking-widest text-center mt-2.5 uppercase" style={{ color: 'var(--text-muted)', opacity: 0.4, letterSpacing: '0.14em' }}>
              Bayesian Model Averaging
            </p>
          )}
        </div>
      </aside>

      {/* Command Palette (Cmd+K) */}
      <CommandPalette open={paletteOpen} onClose={() => setPaletteOpen(false)} />

      {/* Keyboard Shortcut Overlay (?) */}
      <KeyboardShortcutOverlay />

      {/* Tooltip portal (rendered outside sidebar to avoid clipping) */}
      {tooltip && (
        <SidebarTooltip
          item={tooltip.item}
          rect={tooltip.rect}
          sidebarWidth={sidebarWidth}
        />
      )}

      {/* Main content */}
      <main
        className="flex-1 overflow-auto"
        style={{
          background: 'linear-gradient(180deg, var(--void) 0%, var(--void-surface) 15%, var(--void-surface) 100%)',
        }}
        role="main"
        aria-label="Page content"
      >
        <div className="px-10 py-9 max-w-[1600px] mx-auto">
          <BreadcrumbBar />
          <Outlet />
        </div>
      </main>
    </div>
  );
}

/* ─── Sidebar Tooltip ───────────────────────────────────────────── */
function SidebarTooltip({ item, rect, sidebarWidth }: { item: NavItem; rect: DOMRect; sidebarWidth: number }) {
  const lines = item.tooltipFn?.() || [];
  if (lines.length === 0) return null;

  return (
    <div
      className="sidebar-tooltip fixed z-50 px-4 py-3 min-w-[200px] max-w-[280px] pointer-events-none"
      style={{
        left: sidebarWidth + 8,
        top: rect.top + rect.height / 2,
        transform: 'translateY(-50%)',
      }}
    >
      <div className="text-[12px] font-semibold mb-1.5" style={{ color: 'var(--text-luminous)' }}>
        {item.label}
      </div>
      {lines.map((line, i) => (
        <div key={i} className="text-[11px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
          {line}
        </div>
      ))}
      <div className="text-[9px] mt-1.5" style={{ color: 'var(--text-muted)' }}>
        {new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
      </div>
    </div>
  );
}
