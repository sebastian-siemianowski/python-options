import { Loader2 } from 'lucide-react';

export default function LoadingSpinner({ text = 'Loading...' }: { text?: string }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 gap-4 fade-up">
      <div className="relative">
        <div
          className="w-14 h-14 rounded-full"
          style={{
            background: 'linear-gradient(135deg, rgba(139,92,246,0.08) 0%, rgba(56,217,245,0.04) 100%)',
            boxShadow: '0 0 30px rgba(139,92,246,0.08)',
          }}
        />
        <Loader2 className="w-6 h-6 animate-spin absolute inset-0 m-auto" style={{ color: 'var(--accent-violet-bright)' }} />
      </div>
      <p className="text-[13px] font-medium" style={{ color: 'var(--text-muted)' }}>{text}</p>
    </div>
  );
}
