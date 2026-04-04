import { Loader2 } from 'lucide-react';

interface Props {
  text?: string;
  variant?: 'spinner' | 'cards' | 'table' | 'stats';
}

function SkeletonCards() {
  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-5 px-6 py-8 skeleton-crossfade-enter">
      {Array.from({ length: 6 }).map((_, i) => (
        <div key={i} className="skeleton skeleton-card" style={{ animationDelay: `${i * 80}ms` }} />
      ))}
    </div>
  );
}

function SkeletonTable() {
  return (
    <div className="flex flex-col gap-2 px-6 py-8 skeleton-crossfade-enter">
      {Array.from({ length: 8 }).map((_, i) => (
        <div key={i} className="skeleton skeleton-row" style={{ animationDelay: `${i * 60}ms` }} />
      ))}
    </div>
  );
}

function SkeletonStats() {
  return (
    <div className="grid grid-cols-2 lg:grid-cols-4 gap-5 px-6 py-8 skeleton-crossfade-enter">
      {Array.from({ length: 4 }).map((_, i) => (
        <div key={i} className="skeleton" style={{ height: 100, borderRadius: 16, animationDelay: `${i * 80}ms` }} />
      ))}
      <div className="col-span-full flex flex-col gap-3 mt-4">
        {Array.from({ length: 3 }).map((_, i) => (
          <div key={i} className="skeleton skeleton-text" style={{ width: `${60 + i * 10}%`, animationDelay: `${(i + 4) * 60}ms` }} />
        ))}
      </div>
    </div>
  );
}

export default function LoadingSpinner({ text = 'Loading...', variant = 'spinner' }: Props) {
  if (variant === 'cards') return <SkeletonCards />;
  if (variant === 'table') return <SkeletonTable />;
  if (variant === 'stats') return <SkeletonStats />;

  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 fade-up">
      <div className="relative">
        <div
          className="w-14 h-14 rounded-full"
          style={{
            background: 'linear-gradient(135deg, var(--violet-8) 0%, rgba(56,217,245,0.04) 100%)',
            boxShadow: '0 0 30px var(--violet-8)',
          }}
        />
        <Loader2 className="w-6 h-6 animate-spin absolute inset-0 m-auto" style={{ color: 'var(--accent-violet-bright)' }} />
      </div>
      <p className="text-[13px] font-medium" style={{ color: 'var(--text-muted)' }}>{text}</p>
    </div>
  );
}
