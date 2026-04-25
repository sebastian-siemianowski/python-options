import { useQuery } from '@tanstack/react-query';
import { useState, useMemo, useEffect, useRef } from 'react';
import { api } from '../api';
import type {
  RiskDashboardFull, RiskStressCategory, RiskStressIndicator,
  SectorMetrics, CurrencyMetrics,
  MarketBreadth, CorrelationStress,
} from '../api';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { RiskSkeleton } from '../components/CosmicSkeleton';
import { CosmicErrorCard } from '../components/CosmicErrorState';
import { RiskEmpty } from '../components/CosmicEmptyState';
import {
  ShieldAlert, Thermometer, Activity, RefreshCw,
  ChevronDown, ChevronRight, ArrowUp, ArrowDown, Minus,
  Gem, Globe, BarChart3, Link2,
  CircleDot, CircleAlert, Flame, Skull,
  DollarSign, TrendingUp, Landmark, Droplets, Cog,
  Award, Medal, Trophy,
} from 'lucide-react';

type RiskTab = 'overview' | 'cross_asset' | 'metals' | 'market' | 'sectors' | 'currencies';

/* ══════════════════════════════════════════════════════════════════
   SVG Gauge Helpers (Story 5.1)
   ══════════════════════════════════════════════════════════════════ */

const GAUGE_CX = 100;
const GAUGE_CY = 100;
const GAUGE_R = 86;
const GAUGE_START_DEG = 225; // bottom-left (0° = top, clockwise)
const GAUGE_SWEEP_DEG = 270;

function gaugePoint(angleDeg: number, r = GAUGE_R) {
  const rad = ((angleDeg - 90) * Math.PI) / 180;
  return { x: GAUGE_CX + r * Math.cos(rad), y: GAUGE_CY + r * Math.sin(rad) };
}

function gaugeArcPath(startDeg: number, endDeg: number, r = GAUGE_R) {
  const s = gaugePoint(startDeg, r);
  const e = gaugePoint(endDeg, r);
  const sweep = endDeg - startDeg;
  const large = sweep > 180 ? 1 : 0;
  return `M ${s.x} ${s.y} A ${r} ${r} 0 ${large} 1 ${e.x} ${e.y}`;
}

function regimeColor(status: string) {
  switch (status) {
    case 'Calm': return 'var(--accent-emerald)';
    case 'Elevated': return 'var(--accent-amber)';
    case 'Stressed': return 'var(--accent-rose)';
    case 'Crisis': return 'var(--accent-rose)';
    default: return '#7a8ba4';
  }
}

function regimeGlow(status: string) {
  switch (status) {
    case 'Calm': return 'var(--emerald-8)';
    case 'Elevated': return 'rgba(245,197,66,0.08)';
    case 'Stressed': return 'var(--rose-8)';
    case 'Crisis': return 'var(--rose-12)';
    default: return 'var(--violet-6)';
  }
}

/* ── Temperature History (localStorage) ───────────────────────── */

const TEMP_HISTORY_KEY = 'risk-temp-history';
const MAX_HISTORY = 168; // 7 days * 24 hours

interface TempPoint { t: number; v: number; status: string }

function loadTempHistory(): TempPoint[] {
  try {
    const raw = localStorage.getItem(TEMP_HISTORY_KEY);
    return raw ? JSON.parse(raw) : [];
  } catch { return []; }
}

function saveTempSnapshot(temp: number, status: string) {
  const history = loadTempHistory();
  const now = Date.now();
  // Only save one per hour
  if (history.length > 0 && now - history[history.length - 1].t < 3600_000) return;
  history.push({ t: now, v: temp, status });
  if (history.length > MAX_HISTORY) history.splice(0, history.length - MAX_HISTORY);
  try { localStorage.setItem(TEMP_HISTORY_KEY, JSON.stringify(history)); } catch { /* noop */ }
}

/* ══════════════════════════════════════════════════════════════════
   Main Page Component
   ══════════════════════════════════════════════════════════════════ */

export default function RiskPage() {
  const [tab, setTab] = useState<RiskTab>('overview');
  const [refreshing, setRefreshing] = useState(false);

  const summaryQ = useQuery({
    queryKey: ['riskSummary'],
    queryFn: api.riskSummary,
    staleTime: 5 * 60_000,
  });

  const dashQ = useQuery({
    queryKey: ['riskDashboard'],
    queryFn: () => api.riskDashboard() as Promise<RiskDashboardFull>,
    staleTime: 5 * 60_000,
  });

  const data = summaryQ.data;
  const dash = dashQ.data;

  // Persist temperature history
  useEffect(() => {
    if (data) saveTempSnapshot(data.combined_temperature, data.status);
  }, [data]);

  const handleRefresh = async () => {
    setRefreshing(true);
    try {
      await api.riskRefresh();
      summaryQ.refetch();
      dashQ.refetch();
    } finally {
      setRefreshing(false);
    }
  };

  if (summaryQ.isLoading && !data) return <RiskSkeleton />;
  if (summaryQ.error) return <CosmicErrorCard title="Unable to compute risk dashboard" error={summaryQ.error as Error} onRetry={() => summaryQ.refetch()} />;
  if (!data) return <RiskEmpty />;

  const tabs: { id: RiskTab; label: string; icon: typeof ShieldAlert }[] = [
    { id: 'overview', label: 'Overview', icon: Thermometer },
    { id: 'cross_asset', label: 'Cross-Asset Stress', icon: ShieldAlert },
    { id: 'metals', label: 'Metals', icon: Gem },
    { id: 'market', label: 'Market', icon: Globe },
    { id: 'sectors', label: 'Sectors', icon: BarChart3 },
    { id: 'currencies', label: 'Currencies', icon: Link2 },
  ];

  return (
    <>
      <PageHeader
        title="Risk Dashboard"
        action={
          <div className="flex items-center gap-3">
            {dash?._cached && (
              <span className="px-2.5 py-1 rounded-lg text-[10px] font-medium"
                style={{ background: 'var(--violet-6)', color: 'var(--text-muted)', border: '1px solid var(--violet-8)' }}>
                ⏱ cached {dash._cache_age_seconds ? `${Math.round(dash._cache_age_seconds / 60)}m ago` : ''}
              </span>
            )}
            <button
              onClick={handleRefresh}
              disabled={refreshing || summaryQ.isFetching}
              className="flex items-center gap-2 px-4 py-2 rounded-xl text-[13px] font-medium transition-all duration-300 disabled:opacity-50"
              style={{
                background: 'linear-gradient(135deg, var(--violet-12), var(--violet-8))',
                color: '#b49aff',
                border: '1px solid var(--violet-15)',
                boxShadow: '0 2px 8px var(--violet-8)',
              }}
            >
              <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
              {refreshing ? 'Refreshing...' : 'Refresh'}
            </button>
          </div>
        }
      >
        Unified cross-asset risk assessment
      </PageHeader>

      {/* ── Cosmic Speedometer Gauge ──────────────────────────────── */}
      <TemperatureGauge temperature={data.combined_temperature} status={data.status} computedAt={data.computed_at} />

      {/* ── Per-module summary cards ──────────────────────────────── */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8 fade-up-delay-1">
        {[
          { title: 'Cross-Asset Stress', val: data.risk_temperature, sub: 'FX · Equities · Duration · Commodities', icon: ShieldAlert, tab: 'cross_asset' as RiskTab },
          { title: 'Metals Risk', val: data.metals_temperature, sub: 'Gold · Silver · Copper · Palladium', icon: Gem, tab: 'metals' as RiskTab },
          { title: 'Market Temperature', val: data.market_temperature, sub: 'Equity Universe · Sectors · Currencies', icon: Globe, tab: 'market' as RiskTab },
        ].map(({ title, val, sub, icon: Icon, tab: cardTab }) => {
          const cardColor = val < 0.3 ? 'var(--accent-emerald)' : val < 0.7 ? 'var(--accent-amber)' : 'var(--accent-rose)';
          const cardGlow = val < 0.3 ? 'var(--emerald-8)' : val < 0.7 ? 'rgba(245,197,66,0.06)' : 'var(--rose-8)';
          return (
            <button key={title} onClick={() => setTab(cardTab)}
              className="glass-card p-5 text-left hover-lift transition-all duration-300 group relative overflow-hidden"
              style={{ borderTop: `2px solid ${cardColor}` }}>
              <div style={{
                position: 'absolute', top: 0, right: 0, width: 120, height: 120, borderRadius: '50%',
                background: `radial-gradient(circle, ${cardGlow} 0%, transparent 70%)`,
                transform: 'translate(30%, -30%)', pointerEvents: 'none',
              }} />
              <div className="relative z-10">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <Icon className="w-4 h-4" style={{ color: cardColor }} />
                    <span className="text-xs font-medium" style={{ color: 'var(--text-secondary)' }}>{title}</span>
                  </div>
                  <ChevronRight className="w-3.5 h-3.5 opacity-0 group-hover:opacity-100 transition-opacity" style={{ color: cardColor }} />
                </div>
                <div className="text-2xl font-bold mb-1" style={{ color: cardColor, fontVariantNumeric: 'tabular-nums' }}>
                  {val.toFixed(3)}
                </div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{sub}</div>
              </div>
            </button>
          );
        })}
      </div>

      {/* ── Tab nav ───────────────────────────────────────────────── */}
      <div className="mb-8 fade-up-delay-2">
        <div className="flex items-center gap-1 p-1 rounded-xl overflow-x-auto"
          style={{ background: 'var(--violet-4)', border: '1px solid var(--violet-8)' }}>
          {tabs.map(({ id, label, icon: Icon }) => {
            const isActive = tab === id;
            return (
              <button
                key={id}
                onClick={() => setTab(id)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg text-[12px] font-medium transition-all duration-200 whitespace-nowrap"
                style={{
                  background: isActive ? 'linear-gradient(135deg, var(--violet-15), var(--violet-10))' : 'transparent',
                  color: isActive ? '#b49aff' : '#7a8ba4',
                  boxShadow: isActive ? '0 2px 8px var(--violet-10)' : 'none',
                  border: isActive ? '1px solid var(--violet-20)' : '1px solid transparent',
                }}
              >
                <Icon className="w-3.5 h-3.5" />
                {label}
              </button>
            );
          })}
        </div>
      </div>

      {/* ── Tab content ───────────────────────────────────────────── */}
      <div className="fade-up">
        {tab === 'overview' && <OverviewTab data={data} />}
        {tab === 'cross_asset' && dash && <CrossAssetTab categories={dash.risk_temperature?.categories} />}
        {tab === 'metals' && dash && <MetalsTab metals={dash.metals_risk_temperature} />}
        {tab === 'market' && dash && <MarketTab market={dash.market_temperature} />}
        {tab === 'sectors' && dash && <SectorsTab sectors={dash.market_temperature?.sectors} />}
        {tab === 'currencies' && dash && <CurrenciesTab currencies={dash.market_temperature?.currencies} />}
        {dashQ.isLoading && tab !== 'overview' && <LoadingSpinner text="Loading full dashboard data..." />}
      </div>
    </>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 5.1: Cosmic Speedometer Temperature Gauge
   ══════════════════════════════════════════════════════════════════ */

function TemperatureGauge({ temperature, status, computedAt }: {
  temperature: number; status: string; computedAt?: string;
}) {
  const [animatedFrac, setAnimatedFrac] = useState(0);
  const targetFrac = Math.min(temperature / 2, 1);
  const color = regimeColor(status);
  const glow = regimeGlow(status);

  // Spring animation on mount
  useEffect(() => {
    let frame: number;
    const start = performance.now();
    const duration = 800;
    const overshoot = 1.15;

    function tick(now: number) {
      const t = Math.min((now - start) / duration, 1);
      // Spring function: overshoot then settle
      const spring = t < 0.7
        ? (t / 0.7) * overshoot * targetFrac
        : targetFrac + (overshoot * targetFrac - targetFrac) * (1 - (t - 0.7) / 0.3);
      setAnimatedFrac(Math.max(0, spring));
      if (t < 1) frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [targetFrac]);

  // Arc paths
  const trackPath = gaugeArcPath(GAUGE_START_DEG, GAUGE_START_DEG + GAUGE_SWEEP_DEG);
  const fillEnd = GAUGE_START_DEG + GAUGE_SWEEP_DEG * animatedFrac;
  const fillPath = animatedFrac > 0.01 ? gaugeArcPath(GAUGE_START_DEG, fillEnd) : '';

  // Needle
  const needleAngle = GAUGE_START_DEG + GAUGE_SWEEP_DEG * animatedFrac;
  const needleTip = gaugePoint(needleAngle);
  const needleBase = gaugePoint(needleAngle, 20);

  // Trend direction from history
  const history = useMemo(() => loadTempHistory(), []);
  const trend = useMemo(() => {
    if (history.length < 3) return 'stable';
    const last3 = history.slice(-3).map(p => p.v);
    const rising = last3[2] > last3[0] + 0.02;
    const falling = last3[2] < last3[0] - 0.02;
    return rising ? 'rising' : falling ? 'falling' : 'stable';
  }, [history]);

  return (
    <div className="glass-card p-8 mb-8 fade-up" style={{ position: 'relative', overflow: 'hidden' }}>
      {/* Radial glow behind gauge */}
      <div style={{
        position: 'absolute', top: '20%', left: '50%', transform: 'translate(-50%, -30%)',
        width: 280, height: 280, borderRadius: '50%',
        background: `radial-gradient(circle, ${glow} 0%, transparent 60%)`,
        pointerEvents: 'none',
      }} />

      <div className="flex flex-col items-center relative z-10">
        {/* SVG Gauge */}
        <div className="w-48 h-48 md:w-52 md:h-52 lg:w-56 lg:h-56 relative"
          style={{ animation: 'gauge-entrance 0.6s cubic-bezier(0.16, 1, 0.3, 1) both' }}>
          <svg viewBox="0 0 200 200" className="w-full h-full"
            style={{ filter: `drop-shadow(0 0 12px ${glow})` }}>
            <defs>
              <linearGradient id="gauge-gradient" x1="0%" y1="0%" x2="100%" y2="0%">
                <stop offset="0%" stopColor="var(--accent-violet)" />
                <stop offset="50%" stopColor="var(--accent-amber)" />
                <stop offset="100%" stopColor="var(--accent-rose)" />
              </linearGradient>
              <filter id="gauge-glow">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
              <filter id="needle-glow">
                <feGaussianBlur stdDeviation="2" result="blur" />
                <feMerge>
                  <feMergeNode in="blur" />
                  <feMergeNode in="SourceGraphic" />
                </feMerge>
              </filter>
            </defs>

            {/* Track (unfilled) */}
            <path d={trackPath} fill="none" stroke="var(--violet-8)" strokeWidth={6}
              strokeLinecap="round" />

            {/* Fill (gradient) */}
            {fillPath && (
              <path d={fillPath} fill="none" stroke="url(#gauge-gradient)" strokeWidth={6}
                strokeLinecap="round" filter="url(#gauge-glow)" />
            )}

            {/* Needle */}
            <line x1={needleBase.x} y1={needleBase.y} x2={needleTip.x} y2={needleTip.y}
              stroke={color} strokeWidth={2} strokeLinecap="round" filter="url(#needle-glow)" />
            <circle cx={needleTip.x} cy={needleTip.y} r={4} fill={color}
              filter="url(#needle-glow)" />
            <circle cx={GAUGE_CX} cy={GAUGE_CY} r={6} fill="var(--violet-15)"
              stroke="var(--violet-30)" strokeWidth={1} />
          </svg>

          {/* Center text overlay */}
          <div className="absolute top-1/2 left-1/2 text-center"
            style={{ transform: 'translate(-50%, -40%)' }}>
            <div className="text-stat-value"
              style={{
                background: `linear-gradient(135deg, ${color}, ${color}88)`,
                WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
              }}>
              {temperature.toFixed(2)}
            </div>
            <div className="text-caption mt-1" style={{ color, fontWeight: 600 }}>
              {status}
            </div>
          </div>
        </div>

        {/* Trend arrow + sparkline row */}
        <div className="flex items-center gap-3 mt-2">
          {/* Trend arrow */}
          {trend === 'rising' && (
            <div style={{ color: 'var(--accent-rose)', filter: 'drop-shadow(0 0 4px var(--rose-50))' }}>
              <ArrowUp className="w-3 h-3" />
            </div>
          )}
          {trend === 'falling' && (
            <div style={{ color: 'var(--accent-emerald)', filter: 'drop-shadow(0 0 4px var(--emerald-50))' }}>
              <ArrowDown className="w-3 h-3" />
            </div>
          )}
          {trend === 'stable' && (
            <Minus className="w-3 h-3" style={{ color: '#7a8ba4' }} />
          )}

          {/* 7-day sparkline */}
          <TemperatureSparkline history={history} />
        </div>

        {/* Timestamp */}
        <p className="text-[11px] mt-3" style={{ color: 'var(--text-muted)' }}>
          Computed at {computedAt ? new Date(computedAt).toLocaleString() : 'N/A'}
        </p>
      </div>
    </div>
  );
}

/* ── Temperature Sparkline ──────────────────────────────────────── */

function TemperatureSparkline({ history }: { history: TempPoint[] }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const W = 160, H = 32;
  const [drawProgress, setDrawProgress] = useState(0);

  // Animate draw from left-to-right
  useEffect(() => {
    if (history.length < 2) return;
    let frame: number;
    const start = performance.now();
    const duration = 800;
    function tick(now: number) {
      const t = Math.min((now - start) / duration, 1);
      const eased = 1 - Math.pow(1 - t, 3);
      setDrawProgress(eased);
      if (t < 1) frame = requestAnimationFrame(tick);
    }
    frame = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(frame);
  }, [history.length]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || history.length < 2) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const dpr = window.devicePixelRatio || 1;
    canvas.width = W * dpr;
    canvas.height = H * dpr;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, W, H);

    const vals = history.map(p => p.v);
    const min = Math.min(...vals) * 0.9;
    const max = Math.max(...vals) * 1.1 || 1;
    const range = max - min || 1;

    // Only draw up to current progress
    const visibleCount = Math.max(2, Math.ceil(vals.length * drawProgress));
    const visibleVals = vals.slice(0, visibleCount);

    // Fill below curve
    ctx.beginPath();
    ctx.moveTo(0, H);
    visibleVals.forEach((v, i) => {
      const x = (i / (vals.length - 1)) * W;
      const y = H - ((v - min) / range) * (H - 4);
      ctx.lineTo(x, y);
    });
    const lastVisibleX = ((visibleCount - 1) / (vals.length - 1)) * W;
    ctx.lineTo(lastVisibleX, H);
    ctx.closePath();
    const fillGrad = ctx.createLinearGradient(0, 0, 0, H);
    fillGrad.addColorStop(0, 'rgba(139,92,246,0.12)');
    fillGrad.addColorStop(1, 'rgba(15,15,35,0)');
    ctx.fillStyle = fillGrad;
    ctx.fill();

    // Line - 2px with rounded caps
    ctx.beginPath();
    visibleVals.forEach((v, i) => {
      const x = (i / (vals.length - 1)) * W;
      const y = H - ((v - min) / range) * (H - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    // Use regime accent for the last point's status
    const lastStatus = history[visibleCount - 1]?.status || '';
    ctx.strokeStyle = regimeColor(lastStatus) || '#8B5CF6';
    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowColor = 'rgba(139,92,246,0.4)';
    ctx.shadowBlur = 4;
    ctx.stroke();
    ctx.shadowBlur = 0;

    // Pulsing dot at the most recent point
    if (drawProgress >= 0.95) {
      const lastVal = visibleVals[visibleVals.length - 1];
      const dotX = ((visibleCount - 1) / (vals.length - 1)) * W;
      const dotY = H - ((lastVal - min) / range) * (H - 4);
      ctx.beginPath();
      ctx.arc(dotX, dotY, 3, 0, Math.PI * 2);
      ctx.fillStyle = regimeColor(lastStatus) || '#8B5CF6';
      ctx.fill();
      // Outer glow ring
      ctx.beginPath();
      ctx.arc(dotX, dotY, 5, 0, Math.PI * 2);
      ctx.strokeStyle = regimeColor(lastStatus) || '#8B5CF6';
      ctx.lineWidth = 1;
      ctx.globalAlpha = 0.3;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    // Regime transition markers
    for (let i = 1; i < visibleCount && i < history.length; i++) {
      if (history[i].status !== history[i - 1].status) {
        const x = (i / (vals.length - 1)) * W;
        ctx.setLineDash([2, 2]);
        ctx.strokeStyle = 'rgba(139,92,246,0.3)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, H);
        ctx.stroke();
        ctx.setLineDash([]);
        // Dot at top
        ctx.beginPath();
        ctx.arc(x, 3, 2, 0, Math.PI * 2);
        ctx.fillStyle = regimeColor(history[i].status);
        ctx.fill();
      }
    }
  }, [history, drawProgress]);

  return (
    <canvas
      ref={canvasRef}
      style={{ width: W, height: H, opacity: history.length < 2 ? 0.3 : 1 }}
    />
  );
}

/* ══════════════════════════════════════════════════════════════════
   Overview Tab
   ══════════════════════════════════════════════════════════════════ */

function OverviewTab({ data: _data }: { data: { combined_temperature: number; status: string } }) {
  const tiers = [
    { range: '< 0.3', label: 'Calm', desc: 'Full exposure permitted', color: 'var(--accent-emerald)', glow: 'var(--emerald-8)', Icon: CircleDot },
    { range: '0.3 – 0.7', label: 'Elevated', desc: 'Monitor positions closely', color: 'var(--accent-amber)', glow: 'rgba(245,197,66,0.06)', Icon: CircleAlert },
    { range: '0.7 – 1.2', label: 'Stressed', desc: 'Reduce risk exposure', color: 'var(--accent-rose)', glow: 'var(--rose-8)', Icon: Flame },
    { range: '> 1.2', label: 'Crisis', desc: 'Capital preservation mode', color: 'var(--accent-rose)', glow: 'var(--rose-12)', Icon: Skull },
  ];
  const activeTier = _data.combined_temperature < 0.3 ? 0 : _data.combined_temperature < 0.7 ? 1 : _data.combined_temperature < 1.2 ? 2 : 3;

  return (
    <div className="space-y-6">
      {/* Risk regime cards */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {tiers.map((tier, idx) => {
          const isActive = idx === activeTier;
          return (
            <div key={tier.label} className="glass-card p-4 relative overflow-hidden transition-all duration-300"
              style={{
                borderLeft: `3px solid ${isActive ? tier.color : 'transparent'}`,
                boxShadow: isActive ? `0 0 20px ${tier.glow}, inset 0 0 20px ${tier.glow}` : 'none',
                transform: isActive ? 'scale(1.02)' : 'scale(1)',
              }}>
              {isActive && <div style={{
                position: 'absolute', top: 0, right: 0, width: 80, height: 80, borderRadius: '50%',
                background: `radial-gradient(circle, ${tier.glow} 0%, transparent 70%)`,
                transform: 'translate(20%, -20%)', pointerEvents: 'none',
              }} />}
              <div className="relative z-10">
                <div className="mb-1"><tier.Icon className="w-5 h-5" style={{ color: isActive ? tier.color : 'var(--text-muted)' }} /></div>
                <div className="text-sm font-semibold mb-0.5" style={{ color: isActive ? tier.color : 'var(--text-primary)' }}>
                  {tier.label}
                </div>
                <div className="text-[10px] font-mono mb-1.5" style={{ color: isActive ? tier.color : 'var(--text-muted)' }}>
                  {tier.range}
                </div>
                <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{tier.desc}</div>
                {isActive && (
                  <div className="mt-2 px-2 py-0.5 rounded-full text-[9px] font-bold inline-block"
                    style={{ background: tier.glow, color: tier.color, border: `1px solid ${tier.color}30` }}>
                    CURRENT
                  </div>
                )}
              </div>
            </div>
          );
        })}
      </div>

      {/* Methodology info */}
      <div className="glass-card p-5">
        <h3 className="text-sm font-medium mb-3 flex items-center gap-2" style={{ color: 'var(--text-secondary)' }}>
          <Activity className="w-4 h-4" style={{ color: 'var(--accent-violet)' }} />
          How It Works
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="space-y-1">
            <div className="font-medium" style={{ color: 'var(--accent-violet)' }}>Cross-Asset Stress (40%)</div>
            <div style={{ color: 'var(--text-muted)' }}>FX carry, equity futures, rates, and commodity signals aggregated via z-scores</div>
          </div>
          <div className="space-y-1">
            <div className="font-medium" style={{ color: 'var(--accent-violet)' }}>Metals Risk (30%)</div>
            <div style={{ color: 'var(--text-muted)' }}>Gold, silver, copper, palladium stress with vol inversion and crash detection</div>
          </div>
          <div className="space-y-1">
            <div className="font-medium" style={{ color: 'var(--accent-violet)' }}>Market Temperature (30%)</div>
            <div style={{ color: 'var(--text-muted)' }}>Equity breadth, correlation stress, sector rotation, and currency positioning</div>
          </div>
        </div>
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 5.2: Cross-Asset Stress Tab with Constellation
   ══════════════════════════════════════════════════════════════════ */

const RADAR_CATEGORIES = [
  { id: 'FX_Stress', label: 'FX Carry', Icon: DollarSign },
  { id: 'Futures_Stress', label: 'Equities', Icon: TrendingUp },
  { id: 'Rates_Stress', label: 'Duration', Icon: Landmark },
  { id: 'Commodity_Stress', label: 'Commodities', Icon: Droplets },
  { id: 'Metals_Stress', label: 'Metals', Icon: Cog },
];

function stressNodeColor(stress: number) {
  if (stress > 1.5) return { main: '#ff6b8a', glow: 'rgba(255,107,138,0.35)', bg: 'rgba(255,107,138,0.12)' };
  if (stress > 0.7) return { main: '#f5c542', glow: 'rgba(245,197,66,0.30)', bg: 'rgba(245,197,66,0.10)' };
  return { main: '#3ee8a5', glow: 'rgba(62,232,165,0.30)', bg: 'rgba(62,232,165,0.10)' };
}

function CrossAssetTab({ categories }: { categories?: Record<string, RiskStressCategory> }) {
  const [expanded, setExpanded] = useState<Set<string>>(new Set());
  if (!categories) return <div className="text-sm" style={{ color: 'var(--text-muted)' }}>No cross-asset data available</div>;

  const toggle = (name: string) =>
    setExpanded(prev => {
      const n = new Set(prev);
      if (n.has(name)) n.delete(name);
      else n.add(name);
      return n;
    });

  const sorted = Object.values(categories).sort((a, b) => b.weighted_contribution - a.weighted_contribution);
  // Index by dict key and name for flexible lookup
  const catByKey: Record<string, RiskStressCategory> = {};
  Object.entries(categories).forEach(([k, v]) => { catByKey[k] = v; catByKey[v.name] = v; });

  // Build radar data
  const radarData = RADAR_CATEGORIES.map(rc => {
    const cat = catByKey[rc.id];
    return { ...rc, stress: cat?.stress_level ?? 0, weight: cat?.weight ?? 0.2, contribution: cat?.weighted_contribution ?? 0 };
  });
  const avgStress = radarData.reduce((s, d) => s + d.stress, 0) / (radarData.length || 1);
  const maxStress = Math.max(...radarData.map(d => d.stress), 0.01);
  const avgColors = stressNodeColor(avgStress);

  // Radar pentagon math
  const CX = 200, CY = 190, R = 110;
  const radarPt = (i: number, frac: number) => {
    const angle = (Math.PI * 2 * i) / radarData.length - Math.PI / 2;
    return { x: CX + R * frac * Math.cos(angle), y: CY + R * frac * Math.sin(angle) };
  };
  const radarScale = 3; // max stress on radar
  const radarPath = radarData.map((d, i) => {
    const p = radarPt(i, Math.min(d.stress / radarScale, 1));
    return `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`;
  }).join(' ') + ' Z';

  // Per-vertex label placement to avoid overlaps
  const labelLayout = (i: number) => {
    const n = radarData.length;
    const angleDeg = (360 * i) / n - 90;
    // Normalized angle 0-360
    const norm = ((angleDeg % 360) + 360) % 360;
    // Determine anchor and offset based on position around the pentagon
    let anchor: 'middle' | 'start' | 'end' = 'middle';
    let dx = 0, dy = 0;
    if (norm >= 350 || norm <= 10) { // top
      dy = -18; anchor = 'middle';
    } else if (norm > 10 && norm < 90) { // top-right
      dx = 18; dy = -6; anchor = 'start';
    } else if (norm >= 90 && norm < 170) { // bottom-right
      dx = 18; dy = 10; anchor = 'start';
    } else if (norm >= 170 && norm <= 210) { // bottom
      dy = 22; anchor = 'middle';
    } else if (norm > 210 && norm < 270) { // bottom-left
      dx = -18; dy = 10; anchor = 'end';
    } else { // top-left
      dx = -18; dy = -6; anchor = 'end';
    }
    return { anchor, dx, dy };
  };

  return (
    <div className="space-y-6">
      {/* ── Radial Stress Dashboard ── */}
      <div className="glass-card" style={{
        position: 'relative', overflow: 'hidden', padding: '32px 24px 24px',
        background: 'linear-gradient(160deg, rgba(10,14,40,0.85) 0%, rgba(20,10,50,0.7) 50%, rgba(8,18,48,0.85) 100%)',
        borderTop: `1px solid ${avgColors.glow}`,
      }}>
        {/* Ambient glow */}
        <div style={{
          position: 'absolute', top: '40%', left: '50%', transform: 'translate(-50%, -50%)',
          width: 420, height: 420, borderRadius: '50%', pointerEvents: 'none',
          background: `radial-gradient(circle, ${avgColors.glow} 0%, transparent 70%)`,
        }} />

        {/* Title row */}
        <div style={{ position: 'relative', zIndex: 2, display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 20 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <div style={{
              width: 10, height: 10, borderRadius: '50%', background: avgColors.main,
              boxShadow: `0 0 12px ${avgColors.glow}`,
            }} />
            <span style={{ color: 'var(--text-luminous)', fontSize: 15, fontWeight: 700, letterSpacing: '0.02em' }}>
              Cross-Asset Stress Radar
            </span>
          </div>
          <div style={{
            padding: '4px 14px', borderRadius: 20,
            background: avgColors.bg, border: `1px solid ${avgColors.glow}`,
            color: avgColors.main, fontSize: 13, fontWeight: 700,
            fontVariantNumeric: 'tabular-nums',
          }}>
            Avg {avgStress.toFixed(2)}
          </div>
        </div>

        {/* Radar Chart + Category Cards Layout */}
        <div style={{ position: 'relative', zIndex: 2, display: 'flex', gap: 28, alignItems: 'stretch', flexWrap: 'wrap' }}>
          {/* SVG Radar */}
          <div style={{ flex: '0 0 400px', display: 'flex', justifyContent: 'center', alignItems: 'center' }}>
            <svg viewBox="0 0 400 400" width={400} height={400}>
              <defs>
                <radialGradient id="radar-center-glow">
                  <stop offset="0%" stopColor={avgColors.glow} />
                  <stop offset="100%" stopColor="transparent" />
                </radialGradient>
                <linearGradient id="radar-fill-grad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={avgColors.main} stopOpacity={0.25} />
                  <stop offset="100%" stopColor={avgColors.main} stopOpacity={0.05} />
                </linearGradient>
                <filter id="radar-glow">
                  <feGaussianBlur stdDeviation="4" result="blur" />
                  <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
                </filter>
              </defs>

              {/* Background glow */}
              <circle cx={CX} cy={CY} r={R + 20} fill="url(#radar-center-glow)" opacity={0.5} />

              {/* Grid rings */}
              {[0.33, 0.66, 1.0].map((frac, i) => (
                <polygon key={i}
                  points={radarData.map((_, j) => { const p = radarPt(j, frac); return `${p.x},${p.y}`; }).join(' ')}
                  fill="none" stroke="rgba(140,160,200,0.08)" strokeWidth={1}
                  strokeDasharray={i < 2 ? '3,6' : undefined}
                />
              ))}

              {/* Axis lines */}
              {radarData.map((_, i) => {
                const p = radarPt(i, 1);
                return <line key={i} x1={CX} y1={CY} x2={p.x} y2={p.y} stroke="rgba(140,160,200,0.06)" strokeWidth={1} />;
              })}

              {/* Grid labels */}
              {[1, 2, 3].map(v => {
                const p = radarPt(0, v / radarScale);
                return (
                  <text key={v} x={p.x + 6} y={p.y - 4} fill="rgba(140,160,200,0.25)" fontSize={9} fontWeight={500}>
                    {v}
                  </text>
                );
              })}

              {/* Filled radar area */}
              <path d={radarPath} fill="url(#radar-fill-grad)" stroke={avgColors.main}
                strokeWidth={2} strokeLinejoin="round" filter="url(#radar-glow)" opacity={0.9} />

              {/* Data points on radar vertices */}
              {radarData.map((d, i) => {
                const p = radarPt(i, Math.min(d.stress / radarScale, 1));
                const colors = stressNodeColor(d.stress);
                return (
                  <g key={d.id}>
                    <circle cx={p.x} cy={p.y} r={7} fill={colors.bg} stroke={colors.main} strokeWidth={2} />
                    <circle cx={p.x} cy={p.y} r={3} fill={colors.main} />
                  </g>
                );
              })}

              {/* Category labels at the tips — per-vertex positioned */}
              {radarData.map((d, i) => {
                const lp = radarPt(i, 1.18);
                const colors = stressNodeColor(d.stress);
                const layout = labelLayout(i);
                const bx = lp.x + layout.dx;
                const by = lp.y + layout.dy;
                return (
                  <g key={`label-${d.id}`}>
                    {/* Label name */}
                    <text x={bx} y={by} textAnchor={layout.anchor}
                      fill="rgba(180,195,220,0.7)" fontSize={10} fontWeight={500}>
                      {d.label}
                    </text>
                    {/* Stress value below label */}
                    <text x={bx} y={by + 14} textAnchor={layout.anchor}
                      fill={colors.main} fontSize={13} fontWeight={800}
                      fontFamily="var(--font-mono, monospace)">
                      {d.stress.toFixed(2)}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>

          {/* Category mini cards */}
          <div style={{ flex: 1, minWidth: 240, display: 'flex', flexDirection: 'column', gap: 10, justifyContent: 'center' }}>
            {radarData.sort((a, b) => b.stress - a.stress).map((d) => {
              const colors = stressNodeColor(d.stress);
              const barPct = Math.min((d.stress / maxStress) * 100, 100);
              return (
                <div key={d.id} style={{
                  display: 'flex', alignItems: 'center', gap: 12, padding: '10px 14px',
                  borderRadius: 10, background: 'rgba(15,20,55,0.5)',
                  border: `1px solid rgba(140,160,200,0.08)`,
                  transition: 'all 0.2s',
                }}>
                  <span style={{ width: 28, display: 'flex', alignItems: 'center', justifyContent: 'center' }}><d.Icon style={{ width: 18, height: 18, color: stressNodeColor(d.stress).main }} /></span>
                  <div style={{ flex: 1, minWidth: 0 }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 4 }}>
                      <span style={{ color: 'var(--text-luminous)', fontSize: 12, fontWeight: 600 }}>{d.label}</span>
                      <span style={{
                        color: colors.main, fontSize: 14, fontWeight: 800,
                        fontVariantNumeric: 'tabular-nums',
                      }}>{d.stress.toFixed(2)}</span>
                    </div>
                    {/* Stress bar */}
                    <div style={{
                      height: 4, borderRadius: 2, background: 'rgba(140,160,200,0.08)', overflow: 'hidden',
                    }}>
                      <div style={{
                        height: '100%', borderRadius: 2, width: `${barPct}%`,
                        background: `linear-gradient(90deg, ${colors.main}88, ${colors.main})`,
                        transition: 'width 0.6s ease',
                      }} />
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 3 }}>
                      <span style={{ color: 'rgba(140,160,200,0.4)', fontSize: 9 }}>
                        Weight {(d.weight * 100).toFixed(0)}%
                      </span>
                      <span style={{ color: 'rgba(140,160,200,0.4)', fontSize: 9 }}>
                        Contribution {d.contribution.toFixed(2)}
                      </span>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      {/* Category Detail Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {sorted.map(cat => {
          const catStressColor = cat.stress_level > 0.7 ? 'var(--accent-rose)' : cat.stress_level > 0.3 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
          return (
            <div key={cat.name} className="glass-card overflow-hidden hover-lift"
              style={{ borderLeft: `3px solid ${catStressColor}` }}>
              <button onClick={() => toggle(cat.name)}
                className="w-full px-4 py-3.5 flex items-center justify-between transition-colors duration-150"
                style={{ background: expanded.has(cat.name) ? 'var(--violet-4)' : undefined }}
              >
                <div className="flex items-center gap-3">
                  <StressPip level={cat.stress_level} size={10} />
                  <div className="text-left">
                    <span className="text-sm font-semibold" style={{ color: 'var(--text-luminous)' }}>
                      {cat.name.replace(/_/g, ' ')}
                    </span>
                    <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      Weight: {(cat.weight * 100).toFixed(0)}%
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-3 text-xs">
                  <span className="text-base font-bold" style={{ color: catStressColor, fontVariantNumeric: 'tabular-nums' }}>
                    {cat.stress_level.toFixed(2)}
                  </span>
                  {expanded.has(cat.name)
                    ? <ChevronDown className="w-4 h-4" style={{ color: '#7a8ba4' }} />
                    : <ChevronRight className="w-4 h-4" style={{ color: '#7a8ba4' }} />}
                </div>
              </button>
              {expanded.has(cat.name) && cat.indicators && (
                <div className="px-4 pb-4">
                  {/* Contribution bar */}
                  <div className="flex gap-0.5 mb-3 h-2 rounded-full overflow-hidden" style={{ width: '100%', maxWidth: 200 }}>
                    {cat.indicators.map((ind, i) => (
                      <div key={i} className="transition-all duration-300" style={{
                        flex: Math.max(ind.contribution, 0.01),
                        background: ['var(--accent-violet)', '#b49aff', 'var(--text-violet)', '#818cf8', 'var(--accent-indigo)'][i % 5],
                        borderRadius: 4,
                      }} />
                    ))}
                  </div>
                  <IndicatorsTable indicators={cat.indicators} />
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 5.3: Metals Tab with Forecast Spectrum Strips
   ══════════════════════════════════════════════════════════════════ */

function MetalsTab({ metals: data }: { metals: RiskDashboardFull['metals_risk_temperature'] }) {
  const [comparison, setComparison] = useState(false);
  if (!data?.metals) return <div className="text-sm" style={{ color: 'var(--text-muted)' }}>No metals data available</div>;

  const metalEntries = Object.entries(data.metals);

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="glass-card p-4 flex items-center justify-between hover-lift">
        <div className="flex items-center gap-3">
          <Gem className="w-5 h-5" style={{ color: regimeColor(data.status ?? 'Calm') }} />
          <div>
            <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{data.status}</span>
            <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>{data.action_text}</span>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <div className="text-xs" style={{ color: 'var(--text-muted)' }}>
            Regime: {data.regime_state} | Crash: {data.crash_risk_level} ({(data.crash_risk_pct * 100).toFixed(1)}%)
          </div>
          <button
            onClick={() => setComparison(!comparison)}
            className="px-3 py-1 rounded-full text-[11px] font-medium transition-all duration-200"
            style={{
              background: comparison ? 'var(--violet-15)' : 'var(--violet-6)',
              color: comparison ? '#b49aff' : 'var(--accent-violet)',
              border: `1px solid ${comparison ? 'var(--violet-30)' : 'var(--violet-10)'}`,
            }}
          >
            {comparison ? 'Cards' : 'Compare'}
          </button>
        </div>
      </div>

      {/* Stress indicators */}
      {data.indicators && data.indicators.length > 0 && (
        <div className="glass-card p-4">
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Stress Indicators</h3>
          <IndicatorsTable indicators={data.indicators} />
        </div>
      )}

      {comparison ? (
        /* Comparison matrix mode */
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="premium-table">
              <thead className="premium-thead">
                <tr>
                  <th className="text-left">Metal</th>
                  <th className="text-center">Mom</th>
                  <th className="text-center">Stress</th>
                  {['7D', '30D', '90D', '180D', '365D'].map(h => (
                    <th key={h} className="text-center">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {metalEntries.map(([name, m]) => (
                  <tr key={name}>
                    <td className="font-medium" style={{ color: 'var(--text-luminous)' }}>{name}</td>
                    <td className="text-center"><MomentumBadge signal={m.momentum_signal} /></td>
                    <td className="text-center"><span className={stressColor(m.stress_level)}>{m.stress_level.toFixed(2)}</span></td>
                    {[m.forecast_7d, m.forecast_30d, m.forecast_90d, m.forecast_180d, m.forecast_365d].map((f, i) => (
                      <td key={i}>
                        <SpectrumCell value={f} />
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        /* Card grid mode */
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4" style={{ gridAutoRows: '1fr' }}>
          {metalEntries.map(([name, m], idx) => {
            const forecasts = [m.forecast_7d, m.forecast_30d, m.forecast_90d, m.forecast_180d, m.forecast_365d];
            const posCount = forecasts.filter(f => f != null && f > 0).length;
            const negCount = forecasts.filter(f => f != null && f < 0).length;
            const borderColor = posCount > negCount ? 'var(--accent-emerald)' : negCount > posCount ? 'var(--accent-rose)' : 'var(--accent-violet)';
            return (
              <div key={name} className="glass-card p-4 hover-lift fade-up"
                style={{
                  borderLeft: `3px solid ${borderColor}`,
                  animationDelay: `${idx * 100}ms`,
                }}>
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <Gem className="w-5 h-5" style={{ color: 'var(--accent-violet)' }} />
                    <span className="text-base font-semibold" style={{ color: 'var(--text-luminous)' }}>{name}</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="text-lg font-bold" style={{ color: 'var(--text-luminous)', fontVariantNumeric: 'tabular-nums' }}>
                      {m.price != null ? m.price.toFixed(2) : '--'}
                    </span>
                    {m.return_1d != null && (
                      <span className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                        style={{
                          background: m.return_1d > 0 ? 'var(--emerald-12)' : 'var(--rose-12)',
                          color: m.return_1d > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                        }}>
                        {m.return_1d > 0 ? '+' : ''}{(m.return_1d * 100).toFixed(2)}%
                      </span>
                    )}
                  </div>
                </div>

                {/* Forecast spectrum strip */}
                <div className="flex gap-0.5 mb-3">
                  {['7D', '30D', '90D', '180D', '365D'].map((label, i) => (
                    <SpectrumCell key={label} value={forecasts[i]} label={label} showLabel />
                  ))}
                </div>

                {/* Momentum + Stress row */}
                <div className="flex items-center justify-between text-xs">
                  <div className="flex items-center gap-2">
                    <span style={{ color: 'var(--text-muted)' }}>Mom:</span>
                    <MomentumBadge signal={m.momentum_signal} />
                  </div>
                  <div className="flex items-center gap-2">
                    <StressBar level={m.stress_level} />
                    <span style={{ color: 'var(--text-muted)' }}>{m.forecast_confidence}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 5.4: Market Tab with Dual Arc Breadth + Correlation
   ══════════════════════════════════════════════════════════════════ */

function MarketTab({ market }: { market: RiskDashboardFull['market_temperature'] }) {
  if (!market) return <div className="text-sm" style={{ color: 'var(--text-muted)' }}>No market data available</div>;

  return (
    <div className="space-y-6">
      {/* Status bar */}
      <div className="glass-card p-4 flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-3">
          <Globe className="w-5 h-5" style={{ color: regimeColor(market.status ?? 'Calm') }} />
          <div>
            <span className="text-sm font-medium" style={{ color: 'var(--text-primary)' }}>{market.status}</span>
            <span className="text-xs ml-2" style={{ color: 'var(--text-muted)' }}>{market.action_text}</span>
          </div>
        </div>
        <div className="flex items-center gap-4 text-xs" style={{ color: 'var(--text-muted)' }}>
          <span>Mom: {market.overall_momentum}</span>
          <span>Crash: {market.crash_risk_level} ({((market.crash_risk_pct ?? 0) * 100).toFixed(1)}%)</span>
          {market.exit_signal && <span className="font-bold" style={{ color: 'var(--accent-rose)' }}>EXIT: {market.exit_reason}</span>}
        </div>
      </div>

      {/* Breadth + Correlation side by side */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {market.breadth && <BreadthGauge breadth={market.breadth} />}
        {market.correlation && <CorrelationCard corr={market.correlation} />}
      </div>

      {/* Universe instrument pills */}
      {market.universes && (
        <div>
          <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Market Instruments</h3>
          <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))' }}>
            {Object.entries(market.universes).map(([name, u]) => (
              <div key={name} className="glass-card p-3 hover-lift cursor-pointer transition-all duration-150"
                style={{ height: 48, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <div>
                  <div className="text-xs font-semibold" style={{ color: 'var(--text-luminous)' }}>{name}</div>
                  <div className="flex items-center gap-1 mt-0.5">
                    <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>{u.current_level?.toFixed(1) ?? '--'}</span>
                    {u.return_1d != null && (
                      <span className="text-[10px]" style={{ color: u.return_1d > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)' }}>
                        {u.return_1d > 0 ? '+' : ''}{(u.return_1d * 100).toFixed(1)}%
                      </span>
                    )}
                  </div>
                </div>
                {/* Micro forecast fingerprint */}
                <div className="flex gap-1">
                  {[u.forecast_7d, u.forecast_30d, u.forecast_90d].map((f, i) => (
                    <span key={i} className="w-1.5 h-1.5 rounded-full" style={{
                      background: f == null ? '#333' : f > 0 ? 'var(--accent-emerald)' : f < 0 ? 'var(--accent-rose)' : '#7a8ba4',
                    }} />
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Dual Arc Breadth Gauge (Story 5.4) ───────────────────────── */

function BreadthGauge({ breadth }: { breadth: MarketBreadth }) {
  const upPct = breadth.pct_above_50ma;
  const downPct = 1 - upPct;
  const upCount = breadth.new_highs;
  const downCount = breadth.new_lows;

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Market Breadth</h3>
      <div className="flex items-center justify-center gap-4 mb-4">
        {/* Left arc (emerald - UP) */}
        <div className="flex flex-col items-center">
          <svg viewBox="0 0 60 32" width={60} height={32}>
            <path d={`M 4 30 A 26 26 0 0 1 56 30`} fill="none" stroke="var(--emerald-15)" strokeWidth={5} strokeLinecap="round" />
            <path d={`M 4 30 A 26 26 0 0 1 ${4 + 52 * upPct} ${30 - Math.sin(Math.PI * upPct) * 26}`}
              fill="none" stroke="var(--accent-emerald)" strokeWidth={5} strokeLinecap="round"
              style={{ filter: 'drop-shadow(0 0 4px var(--emerald-30))' }} />
          </svg>
          <span className="text-[10px] mt-0.5" style={{ color: 'var(--accent-emerald)' }}>{(upPct * 100).toFixed(0)}%</span>
        </div>

        {/* Center ratio */}
        <div className="text-center">
          <div className="text-lg font-bold" style={{ color: 'var(--text-luminous)', fontVariantNumeric: 'tabular-nums' }}>
            {upCount} / {downCount}
          </div>
          <div className="text-[10px]" style={{ color: 'var(--text-muted)' }}>Highs / Lows</div>
        </div>

        {/* Right arc (rose - DOWN) */}
        <div className="flex flex-col items-center">
          <svg viewBox="0 0 60 32" width={60} height={32} style={{ transform: 'scaleX(-1)' }}>
            <path d={`M 4 30 A 26 26 0 0 1 56 30`} fill="none" stroke="var(--rose-15)" strokeWidth={5} strokeLinecap="round" />
            <path d={`M 4 30 A 26 26 0 0 1 ${4 + 52 * downPct} ${30 - Math.sin(Math.PI * downPct) * 26}`}
              fill="none" stroke="var(--accent-rose)" strokeWidth={5} strokeLinecap="round"
              style={{ filter: 'drop-shadow(0 0 4px var(--rose-30))' }} />
          </svg>
          <span className="text-[10px] mt-0.5" style={{ color: 'var(--accent-rose)' }}>{(downPct * 100).toFixed(0)}%</span>
        </div>
      </div>

      {/* Additional breadth stats */}
      <div className="space-y-1.5 text-xs">
        <BreadthRow label="A/D Ratio" value={breadth.advance_decline_ratio.toFixed(2)} />
        <BreadthRow label="Above 200 MA" value={`${(breadth.pct_above_200ma * 100).toFixed(1)}%`} />
        {breadth.breadth_thrust && <div className="font-medium" style={{ color: 'var(--accent-emerald)' }}>Breadth Thrust Active</div>}
        {breadth.breadth_warning && <div className="font-medium" style={{ color: 'var(--accent-rose)' }}>Breadth Warning</div>}
        <p className="mt-1.5" style={{ color: 'var(--text-muted)' }}>{breadth.interpretation}</p>
      </div>
    </div>
  );
}

/* ── Correlation Card ─────────────────────────────────────────── */

function CorrelationCard({ corr }: { corr: CorrelationStress }) {
  const isHigh = corr.correlation_percentile > 0.8;
  const isElevated = corr.correlation_percentile > 0.5;
  const gradColor = isHigh ? 'var(--accent-rose)' : isElevated ? 'var(--accent-amber)' : 'var(--accent-violet)';

  return (
    <div className="glass-card p-5">
      <h3 className="text-sm font-medium mb-3" style={{ color: 'var(--text-secondary)' }}>Correlation Stress</h3>
      <div className="text-center mb-4">
        <div className="text-3xl font-bold mb-1" style={{
          background: `linear-gradient(135deg, ${gradColor}, ${gradColor}88)`,
          WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
          textShadow: isHigh ? `0 0 20px var(--rose-30)` : undefined,
          fontVariantNumeric: 'tabular-nums',
        }}>
          {corr.avg_correlation.toFixed(3)}
        </div>
        <div className="text-xs" style={{
          color: isHigh ? 'var(--accent-rose)' : 'var(--accent-emerald)',
          animation: isHigh ? 'pulse 2s ease-in-out infinite' : undefined,
        }}>
          {corr.interpretation}
        </div>
      </div>
      <div className="space-y-1.5 text-xs">
        <BreadthRow label="Max Correlation" value={corr.max_correlation.toFixed(3)} />
        <BreadthRow label="Percentile" value={`${(corr.correlation_percentile * 100).toFixed(0)}%`} />
        {corr.systemic_risk_elevated && (
          <div className="font-medium" style={{ color: 'var(--accent-rose)' }}>Systemic Risk Elevated</div>
        )}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 5.5: Currencies Tab with Glass Cards + Heatmap
   ══════════════════════════════════════════════════════════════════ */

function CurrenciesTab({ currencies }: { currencies?: Record<string, CurrencyMetrics> }) {
  const [heatmap, setHeatmap] = useState(false);
  if (!currencies) return <div className="text-sm" style={{ color: 'var(--text-muted)' }}>No currency data available</div>;

  const entries = Object.entries(currencies).sort(([, a], [, b]) => b.risk_score - a.risk_score);

  // Find JPY pair for callout
  const jpyEntry = entries.find(([name]) => name.includes('JPY'));

  // Aggregate stats
  const avgRisk = entries.reduce((acc, [, c]) => acc + c.risk_score, 0) / entries.length;
  const bullishCount = entries.filter(([, c]) => c.forecast_30d != null && c.forecast_30d > 0).length;
  const bearishCount = entries.length - bullishCount;

  return (
    <div className="space-y-6">
      {/* JPY Strength Callout (Story 5.5 AC-3) */}
      {jpyEntry && (
        <div className="glass-card p-5" style={{
          background: 'linear-gradient(135deg, #0c1445 0%, #1a0533 50%, #110f2e 100%)',
          position: 'relative', overflow: 'hidden',
        }}>
          <div style={{
            position: 'absolute', left: '20%', top: '50%', transform: 'translate(-50%, -50%)',
            width: 200, height: 200, borderRadius: '50%',
            background: 'radial-gradient(ellipse, rgba(34,211,238,0.08) 0%, transparent 60%)',
            pointerEvents: 'none',
          }} />
          <div className="relative z-10">
            <h3 className="text-sm font-semibold mb-2" style={{ color: 'var(--accent-cyan)' }}>Yen Strength View</h3>
            <div className="flex items-center gap-4">
              <div>
                <div className="text-xl font-bold" style={{
                  background: 'linear-gradient(135deg, var(--accent-cyan), var(--accent-violet))',
                  WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent',
                }}>
                  {jpyEntry[0]}: {jpyEntry[1].rate?.toFixed(2) ?? '--'}
                </div>
                <div className="flex items-center gap-2 mt-1">
                  <MomentumBadge signal={jpyEntry[1].momentum_signal} />
                  <span className="text-xs" style={{ color: 'var(--text-muted)' }}>
                    Risk: <RiskScoreCell score={jpyEntry[1].risk_score} />
                  </span>
                </div>
              </div>
              {/* Spectrum strip */}
              <div className="flex gap-0.5">
                {[jpyEntry[1].forecast_7d, jpyEntry[1].forecast_30d, jpyEntry[1].forecast_90d, jpyEntry[1].forecast_180d, jpyEntry[1].forecast_365d].map((f, i) => (
                  <SpectrumCell key={i} value={f} label={['7D','30D','90D','180D','365D'][i]} showLabel />
                ))}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Summary + Heatmap toggle */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4 text-xs">
          <span style={{ color: 'var(--text-muted)' }}>{entries.length} pairs</span>
          <span style={{ color: 'var(--text-muted)' }}>
            Avg Risk: <span style={{ color: avgRisk > 50 ? 'var(--accent-rose)' : 'var(--accent-emerald)', fontWeight: 600 }}>{avgRisk.toFixed(0)}</span>
          </span>
          <span style={{ color: 'var(--text-muted)' }}>
            <span style={{ color: 'var(--accent-emerald)' }}>▲{bullishCount}</span>
            {' / '}
            <span style={{ color: 'var(--accent-rose)' }}>▼{bearishCount}</span>
          </span>
        </div>
        <div className="flex gap-1 p-0.5 rounded-lg" style={{ background: 'var(--violet-4)', border: '1px solid var(--violet-8)' }}>
          <button
            onClick={() => setHeatmap(false)}
            className="px-3 py-1 rounded-md text-[11px] font-medium transition-all duration-200"
            style={{
              background: !heatmap ? 'var(--violet-15)' : 'transparent',
              color: !heatmap ? '#b49aff' : '#7a8ba4',
            }}
          >
            Cards
          </button>
          <button
            onClick={() => setHeatmap(true)}
            className="px-3 py-1 rounded-md text-[11px] font-medium transition-all duration-200"
            style={{
              background: heatmap ? 'var(--violet-15)' : 'transparent',
              color: heatmap ? '#b49aff' : '#7a8ba4',
            }}
          >
            Table
          </button>
        </div>
      </div>

      {heatmap ? (
        /* Heatmap mode */
        <div className="glass-card overflow-hidden">
          <div className="overflow-x-auto">
            <table className="premium-table">
              <thead className="premium-thead">
                <tr>
                  <th className="text-left">Pair</th>
                  <th className="text-center">Mom</th>
                  <th className="text-center">Risk</th>
                  {['7D', '30D', '90D', '180D', '365D'].map(h => (
                    <th key={h} className="text-center">{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {entries.map(([name, c]) => (
                  <tr key={name}>
                    <td className="font-medium" style={{ color: 'var(--text-luminous)' }}>
                      {name}
                      {c.is_inverse && <span className="ml-1 text-[10px]" style={{ color: 'var(--accent-amber)' }}>inv</span>}
                    </td>
                    <td className="text-center"><MomentumBadge signal={c.momentum_signal} /></td>
                    <td className="text-center"><RiskScoreCell score={c.risk_score} /></td>
                    {[c.forecast_7d, c.forecast_30d, c.forecast_90d, c.forecast_180d, c.forecast_365d].map((f, i) => (
                      <td key={i}><SpectrumCell value={f} /></td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      ) : (
        /* Glass card grid */
        <div className="grid gap-3" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))', gridAutoRows: '1fr' }}>
          {entries.map(([name, c], idx) => {
            const pairColor = c.risk_score > 60 ? 'var(--accent-rose)' : c.risk_score > 30 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
            return (
              <div key={name} className="glass-card p-4 hover-lift transition-all duration-200 fade-up relative overflow-hidden"
                style={{ animationDelay: `${idx * 30}ms`, borderTop: `2px solid ${pairColor}` }}>
                <div style={{
                  position: 'absolute', bottom: 0, right: 0, width: 100, height: 100, borderRadius: '50%',
                  background: `radial-gradient(circle, ${c.risk_score > 60 ? 'var(--rose-6)' : 'var(--emerald-6)'} 0%, transparent 70%)`,
                  transform: 'translate(30%, 30%)', pointerEvents: 'none',
                }} />
                <div className="relative z-10">
                  <div className="flex items-center justify-between mb-2">
                    <div>
                      <div className="text-sm font-semibold" style={{ color: 'var(--text-luminous)' }}>
                        {name}
                        {c.is_inverse && <span className="ml-1.5 px-1 py-0.5 rounded text-[8px] font-bold" style={{ background: 'rgba(245,197,66,0.15)', color: 'var(--accent-amber)' }}>INV</span>}
                      </div>
                      <div className="flex items-center gap-2 mt-0.5">
                        <span className="text-base font-bold" style={{ color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>
                          {c.rate?.toFixed(4) ?? '--'}
                        </span>
                        {c.return_1d != null && (
                          <span className="px-1.5 py-0.5 rounded text-[10px] font-medium"
                            style={{
                              background: c.return_1d > 0 ? 'rgba(62,232,165,0.12)' : 'rgba(255,107,138,0.12)',
                              color: c.return_1d > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)',
                            }}>
                            {c.return_1d > 0 ? '+' : ''}{(c.return_1d * 100).toFixed(2)}%
                          </span>
                        )}
                      </div>
                    </div>
                    <div className="flex flex-col items-end gap-1">
                      <MomentumBadge signal={c.momentum_signal} />
                      <RiskScoreCell score={c.risk_score} />
                    </div>
                  </div>
                  {/* Forecast spectrum */}
                  <div className="flex gap-0.5 mt-1">
                    {[c.forecast_7d, c.forecast_30d, c.forecast_90d, c.forecast_180d, c.forecast_365d].map((f, i) => (
                      <SpectrumCell key={i} value={f} label={['7D','30D','90D','180D','365D'][i]} showLabel />
                    ))}
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Story 5.6: Sectors Tab with Ranked Cards
   ══════════════════════════════════════════════════════════════════ */

type SectorSort = 'performance' | 'momentum' | 'risk' | 'alpha';

function SectorsTab({ sectors }: { sectors?: Record<string, SectorMetrics> }) {
  const [sortBy, setSortBy] = useState<SectorSort>('performance');
  const entries = useMemo(() => {
    const arr = Object.entries(sectors ?? {});
    switch (sortBy) {
      case 'performance': return arr.sort(([, a], [, b]) => (b.return_21d ?? 0) - (a.return_21d ?? 0));
      case 'momentum': return arr.sort(([, a], [, b]) => {
        const mScore = (s: string) => s?.includes('Strong') ? 3 : s?.includes('Rising') ? 2 : s?.includes('Falling') ? -1 : s?.includes('Weak') ? -2 : 0;
        return mScore(b.momentum_signal) - mScore(a.momentum_signal);
      });
      case 'risk': return arr.sort(([, a], [, b]) => b.risk_score - a.risk_score);
      default: return arr.sort(([a], [b]) => a.localeCompare(b));
    }
  }, [sectors, sortBy]);

  if (!sectors) return <div className="text-sm" style={{ color: 'var(--text-muted)' }}>No sector data available</div>;

  const MedalIcons = [Trophy, Award, Medal];

  return (
    <div className="space-y-5">
      {/* Sort control bar */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium" style={{ color: 'var(--text-secondary)' }}>
          {entries.length} Sectors · Sorted by {sortBy}
        </h3>
        <div className="flex gap-1 p-0.5 rounded-lg" style={{ background: 'var(--violet-4)', border: '1px solid var(--violet-8)' }}>
          {(['performance', 'momentum', 'risk', 'alpha'] as SectorSort[]).map(s => (
            <button key={s} onClick={() => setSortBy(s)}
              className="px-3 py-1 rounded-md text-[11px] font-medium transition-all duration-200"
              style={{
                background: sortBy === s ? 'var(--violet-15)' : 'transparent',
                color: sortBy === s ? '#b49aff' : '#7a8ba4',
                boxShadow: sortBy === s ? '0 1px 4px var(--violet-8)' : 'none',
              }}>
              {s.charAt(0).toUpperCase() + s.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Ranked sector cards */}
      <div className="space-y-2">
        {entries.map(([name, s], idx) => {
          const sectorColor = s.risk_score > 70 ? 'var(--accent-rose)' : s.risk_score > 40 ? 'var(--accent-amber)' : 'var(--accent-emerald)';
          return (
            <div key={name} className="glass-card px-5 py-3.5 hover-lift flex items-center gap-4 fade-up"
              style={{ minHeight: 60, animationDelay: `${idx * 40}ms`, borderLeft: `3px solid ${sectorColor}` }}>
              {/* Rank badge */}
              <div className="w-8 text-center flex-shrink-0">
                {idx < 3 ? (
                  <>{(() => { const MIcon = MedalIcons[idx]; return <MIcon className="w-5 h-5" style={{ color: idx === 0 ? '#fbbf24' : idx === 1 ? '#94a3b8' : '#d97706' }} />; })()}</>
                ) : (
                  <span className="text-xs font-mono" style={{ color: 'var(--text-muted)' }}>#{idx + 1}</span>
                )}
              </div>

              {/* Name + ticker */}
              <div className="flex-shrink-0" style={{ width: 130 }}>
                <div className="text-sm font-semibold" style={{ color: 'var(--text-luminous)' }}>{name}</div>
                <div className="text-[10px] font-mono" style={{ color: 'var(--accent-violet)' }}>{s.ticker}</div>
              </div>

              {/* Multi-period returns */}
              <div className="flex gap-1 flex-shrink-0">
                {[
                  { label: '1D', v: s.return_1d },
                  { label: '5D', v: s.return_5d },
                  { label: '21D', v: s.return_21d },
                ].map(({ label, v }) => (
                  <div key={label} className="flex flex-col items-center px-1.5 py-0.5 rounded"
                    style={{
                      background: v != null && v > 0 ? 'rgba(62,232,165,0.08)' : v != null && v < 0 ? 'rgba(255,107,138,0.08)' : 'rgba(100,116,139,0.05)',
                      minWidth: 48,
                    }}>
                    <span className="text-[8px]" style={{ color: 'var(--text-muted)' }}>{label}</span>
                    <span className="text-[10px] font-semibold" style={{
                      color: v != null && v > 0 ? 'var(--accent-emerald)' : v != null && v < 0 ? 'var(--accent-rose)' : '#7a8ba4',
                    }}>
                      {v != null ? `${v > 0 ? '+' : ''}${(v * 100).toFixed(1)}%` : '--'}
                    </span>
                  </div>
                ))}
              </div>

              {/* Momentum */}
              <div className="flex items-center gap-1 flex-shrink-0" style={{ width: 80 }}>
                <MomentumBadge signal={s.momentum_signal} />
              </div>

              {/* Risk score with gradient bar */}
              <div className="flex-1 flex items-center gap-2">
                <div className="flex-1 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
                  <div className="h-full rounded-full transition-all duration-500" style={{
                    width: `${Math.min(s.risk_score, 100)}%`,
                    background: `linear-gradient(90deg, var(--accent-emerald), var(--accent-amber), var(--accent-rose))`,
                  }} />
                </div>
                <RiskScoreCell score={s.risk_score} />
              </div>

              {/* Forecast spectrum */}
              <div className="flex gap-0.5 flex-shrink-0">
                {[
                  { v: s.forecast_7d, l: '7D' },
                  { v: s.forecast_30d, l: '30D' },
                  { v: s.forecast_90d, l: '90D' },
                ].map(({ v, l }) => (
                  <SpectrumCell key={l} value={v} label={l} showLabel compact />
                ))}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

/* ══════════════════════════════════════════════════════════════════
   Shared Helper Components (Cosmified)
   ══════════════════════════════════════════════════════════════════ */

function SpectrumCell({ value, label, showLabel, compact }: {
  value: number | null | undefined; label?: string; showLabel?: boolean; compact?: boolean;
}) {
  if (value == null) {
    return (
      <div className="flex flex-col items-center" style={{
        width: compact ? 24 : 40, height: compact ? 20 : 28,
        borderRadius: 4, background: 'rgba(100,116,139,0.06)',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
      }}>
        <span className="text-[8px]" style={{ color: '#6b7a90' }}>--</span>
      </div>
    );
  }
  // Forecast values are already in percent (e.g., 5.97 = 5.97%)
  const pct = value;
  const absMax = 20; // +-20% max for color mapping
  const intensity = Math.min(Math.abs(pct) / absMax, 1);
  const bg = pct > 0
    ? `rgba(62,232,165,${0.05 + intensity * 0.2})`
    : `rgba(255,107,138,${0.05 + intensity * 0.2})`;
  const fg = pct > 0 ? 'var(--accent-emerald)' : 'var(--accent-rose)';

  return (
    <div className="flex flex-col items-center transition-transform duration-150 hover:scale-110" style={{
      width: compact ? 24 : 40, minHeight: compact ? 20 : 28,
      borderRadius: 4, background: bg,
      display: 'flex', alignItems: 'center', justifyContent: 'center',
    }}>
      <span style={{ fontSize: compact ? 8 : 10, fontFamily: 'monospace', color: fg, fontWeight: 600 }}>
        {pct > 0 ? '+' : ''}{pct.toFixed(compact ? 0 : 1)}%
      </span>
      {showLabel && label && (
        <span style={{ fontSize: 7, color: '#7a8ba4', marginTop: -1 }}>{label}</span>
      )}
    </div>
  );
}

function IndicatorsTable({ indicators }: { indicators: RiskStressIndicator[] }) {
  return (
    <table className="premium-table">
      <thead className="premium-thead">
        <tr>
          <th className="text-left">Indicator</th>
          <th className="text-right">Value</th>
          <th className="text-right">Z-Score</th>
          <th className="text-right">Contrib</th>
          <th className="text-center">Data</th>
        </tr>
      </thead>
      <tbody>
        {indicators.map((ind, i) => (
          <tr key={i}>
            <td style={{ color: 'var(--text-secondary)' }}>
              <div className="flex items-center gap-1.5">
                <StressPip level={ind.contribution * 4} size={6} />
                {ind.name}
              </div>
            </td>
            <td className="text-right" style={{ color: 'var(--text-primary)' }}>
              {ind.value != null ? ind.value.toFixed(4) : '--'}
            </td>
            <td className="text-right">
              <span style={{
                color: ind.zscore != null && Math.abs(ind.zscore) > 2 ? 'var(--accent-rose)'
                  : ind.zscore != null && Math.abs(ind.zscore) > 1 ? 'var(--accent-amber)'
                  : 'var(--text-secondary)',
              }}>
                {ind.zscore != null ? ind.zscore.toFixed(2) : '--'}
              </span>
            </td>
            <td className="text-right" style={{ color: 'var(--text-secondary)' }}>
              {ind.contribution.toFixed(4)}
            </td>
            <td className="text-center">
              {ind.data_available
                ? <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-emerald)' }} />
                : <span className="inline-block w-1.5 h-1.5 rounded-full" style={{ background: 'var(--accent-rose)' }} />}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function StressBar({ level }: { level: number }) {
  const w = Math.min(level / 2 * 100, 100);
  const color = level < 0.3 ? 'var(--accent-emerald)' : level < 0.7 ? 'var(--accent-amber)' : 'var(--accent-rose)';
  return (
    <div className="w-12 h-1.5 rounded-full overflow-hidden" style={{ background: 'var(--violet-6)' }}>
      <div className="h-full rounded-full transition-all duration-500" style={{ width: `${w}%`, background: color }} />
    </div>
  );
}

function StressPip({ level, size = 8 }: { level: number; size?: number }) {
  const color = level < 0.3 ? 'var(--accent-emerald)' : level < 0.7 ? 'var(--accent-amber)' : 'var(--accent-rose)';
  const pulse = level >= 0.7 ? '1s' : level >= 0.3 ? '2s' : undefined;
  return (
    <span style={{
      width: size, height: size, borderRadius: '50%', background: color,
      display: 'inline-block', flexShrink: 0,
      animation: pulse ? `pulse ${pulse} ease-in-out infinite` : undefined,
    }} />
  );
}

function stressColor(level: number): string {
  return level < 0.3 ? 'text-[var(--accent-emerald)]' : level < 0.7 ? 'text-[var(--accent-amber)]' : 'text-[var(--accent-rose)]';
}

function MomentumBadge({ signal }: { signal: string }) {
  const color = signal?.includes('Strong') || signal?.includes('\u2191') ? 'var(--accent-emerald)'
    : signal?.includes('Rising') || signal?.includes('\u2197') ? '#6ff0c0'
    : signal?.includes('Falling') || signal?.includes('\u2198') ? '#fb923c'
    : signal?.includes('Weak') || signal?.includes('\u2193') ? 'var(--accent-rose)'
    : '#94a3b8';
  return <span className="text-[10px] font-medium" style={{ color }}>{signal || '--'}</span>;
}

function RiskScoreCell({ score }: { score: number }) {
  const color = score < 30 ? 'var(--accent-emerald)' : score < 60 ? 'var(--accent-amber)' : 'var(--accent-rose)';
  return <span className="font-medium text-xs" style={{ color, fontVariantNumeric: 'tabular-nums' }}>{score}</span>;
}

function BreadthRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between">
      <span style={{ color: 'var(--text-muted)' }}>{label}</span>
      <span className="font-medium" style={{ color: 'var(--text-primary)', fontVariantNumeric: 'tabular-nums' }}>{value}</span>
    </div>
  );
}
