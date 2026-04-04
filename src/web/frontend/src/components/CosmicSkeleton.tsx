/**
 * Cosmic Skeleton System (Story 10.1)
 *
 * Skeleton screens that match page layouts exactly, with coordinated
 * cosmic shimmer animation. The shimmer is global -- one sweep across
 * all elements simultaneously.
 */

interface SkeletonProps {
  className?: string;
  style?: React.CSSProperties;
}

function SkeletonBlock({ className = '', style }: SkeletonProps) {
  return (
    <div
      className={`cosmic-shimmer rounded-lg ${className}`}
      style={{ background: 'var(--void-surface)', ...style }}
    />
  );
}

/* ── Dashboard Skeleton ──────────────────────────────────────── */
export function DashboardSkeleton() {
  return (
    <div className="space-y-6 p-6 skeleton-enter">
      {/* Header */}
      <div className="flex items-center gap-4">
        <SkeletonBlock className="h-8" style={{ width: '60%' }} />
        <div className="flex-1" />
        <SkeletonBlock className="h-8 w-24" />
      </div>
      {/* 2x2 card grid */}
      <div className="grid grid-cols-2 gap-4">
        {[0, 1, 2, 3].map((i) => (
          <div key={i} className="glass-card p-5 space-y-3" style={{ animationDelay: `${i * 40}ms` }}>
            <SkeletonBlock className="h-3" style={{ width: '40%' }} />
            <SkeletonBlock className="h-7 w-20" />
            <SkeletonBlock className="h-2" style={{ width: '70%' }} />
          </div>
        ))}
      </div>
      {/* Chart area */}
      <div className="glass-card p-5 space-y-3">
        <SkeletonBlock className="h-3 w-32" />
        <SkeletonBlock className="h-48" />
      </div>
    </div>
  );
}

/* ── Signal Table Skeleton ───────────────────────────────────── */
export function SignalTableSkeleton() {
  return (
    <div className="space-y-4 p-6 skeleton-enter">
      <div className="flex items-center gap-4">
        <SkeletonBlock className="h-8" style={{ width: '50%' }} />
        <div className="flex-1" />
        <SkeletonBlock className="h-8 w-32" />
      </div>
      {/* Filter bar */}
      <div className="flex gap-2">
        {[80, 60, 70, 50, 90].map((w, i) => (
          <SkeletonBlock key={i} className="h-7 rounded-full" style={{ width: `${w}px` }} />
        ))}
      </div>
      {/* Table header */}
      <div className="glass-card overflow-hidden">
        <div className="flex gap-3 px-4 py-3 border-b" style={{ borderColor: 'var(--border-void)' }}>
          {[60, 80, 50, 70, 50, 40].map((w, i) => (
            <SkeletonBlock key={i} className="h-3" style={{ width: `${w}px` }} />
          ))}
        </div>
        {/* 10 rows */}
        {Array.from({ length: 10 }, (_, i) => (
          <div
            key={i}
            className="flex gap-3 px-4 py-3 border-b"
            style={{ borderColor: 'var(--border-void)', animationDelay: `${i * 30}ms` }}
          >
            <SkeletonBlock className="h-3 w-14" />
            <SkeletonBlock className="h-3" style={{ width: '80%' }} />
            <SkeletonBlock className="h-3 w-10" />
            <SkeletonBlock className="h-3 w-12" />
            <SkeletonBlock className="h-3 w-10" />
            <SkeletonBlock className="h-3 w-8" />
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Charts Skeleton ─────────────────────────────────────────── */
export function ChartsSkeleton() {
  return (
    <div className="flex gap-4 p-6 skeleton-enter" style={{ height: 'calc(100vh - 120px)' }}>
      {/* Sidebar */}
      <div className="w-52 flex-shrink-0 space-y-2">
        <SkeletonBlock className="h-8 w-full rounded-lg" />
        {Array.from({ length: 12 }, (_, i) => (
          <SkeletonBlock key={i} className="h-7 w-full" style={{ animationDelay: `${i * 20}ms` }} />
        ))}
      </div>
      {/* Chart area */}
      <div className="flex-1 space-y-3">
        <div className="flex items-center gap-3">
          <SkeletonBlock className="h-6 w-20" />
          <SkeletonBlock className="h-6 w-32" />
          <div className="flex-1" />
          <SkeletonBlock className="h-6 w-24" />
        </div>
        <SkeletonBlock className="h-[400px] rounded-xl" />
        <SkeletonBlock className="h-24 rounded-xl" />
      </div>
    </div>
  );
}

/* ── Risk Dashboard Skeleton ─────────────────────────────────── */
export function RiskSkeleton() {
  return (
    <div className="space-y-6 p-6 skeleton-enter">
      <SkeletonBlock className="h-8" style={{ width: '40%' }} />
      {/* Gauge */}
      <div className="flex justify-center py-8">
        <SkeletonBlock className="w-32 h-32 rounded-full" />
      </div>
      {/* Category cards */}
      <div className="grid grid-cols-3 gap-4">
        {[0, 1, 2].map((i) => (
          <div key={i} className="glass-card p-5 space-y-3" style={{ animationDelay: `${i * 40}ms` }}>
            <SkeletonBlock className="h-3 w-24" />
            <SkeletonBlock className="h-6 w-16" />
            <SkeletonBlock className="h-2" style={{ width: '80%' }} />
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Tuning Skeleton ─────────────────────────────────────────── */
export function TuningSkeleton() {
  return (
    <div className="space-y-6 p-6 skeleton-enter">
      <div className="flex items-center gap-4">
        <SkeletonBlock className="h-8" style={{ width: '50%' }} />
        <div className="flex-1" />
        <SkeletonBlock className="h-8 w-28" />
      </div>
      {/* Treemap area */}
      <SkeletonBlock className="h-64 rounded-xl" />
      {/* Health grid */}
      <div className="grid grid-cols-4 gap-3">
        {Array.from({ length: 8 }, (_, i) => (
          <div key={i} className="glass-card p-4 space-y-2" style={{ animationDelay: `${i * 30}ms` }}>
            <SkeletonBlock className="h-3 w-16" />
            <SkeletonBlock className="h-5 w-12" />
          </div>
        ))}
      </div>
    </div>
  );
}

/* ── Diagnostics Skeleton ────────────────────────────────────── */
export function DiagnosticsSkeleton() {
  return (
    <div className="space-y-6 p-6 skeleton-enter">
      <SkeletonBlock className="h-8" style={{ width: '45%' }} />
      {/* Tab bar */}
      <div className="flex gap-2">
        {[70, 60, 80, 50, 60].map((w, i) => (
          <SkeletonBlock key={i} className="h-8 rounded-lg" style={{ width: `${w}px` }} />
        ))}
      </div>
      {/* Chart */}
      <SkeletonBlock className="h-56 rounded-xl" />
      {/* Summary table */}
      <div className="glass-card overflow-hidden">
        {Array.from({ length: 6 }, (_, i) => (
          <div
            key={i}
            className="flex gap-3 px-4 py-3 border-b"
            style={{ borderColor: 'var(--border-void)' }}
          >
            <SkeletonBlock className="h-3 w-20" />
            <SkeletonBlock className="h-3 w-12" />
            <SkeletonBlock className="h-3 w-16" />
            <SkeletonBlock className="h-3 w-10" />
          </div>
        ))}
      </div>
    </div>
  );
}
