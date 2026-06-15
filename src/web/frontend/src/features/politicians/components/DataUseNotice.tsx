import { ShieldAlert } from 'lucide-react';
import { POLITICIANS_DATA_USE_BULLETS, POLITICIANS_DATA_USE_NOTICE } from '../disclaimers';

export default function DataUseNotice() {
  return (
    <section
      className="glass-card p-5 border-l-2 border-amber-500/20"
      aria-label="Politician disclosure data use notice"
    >
      <div className="flex items-start gap-3">
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: 'var(--amber-12)' }}
        >
          <ShieldAlert className="w-4.5 h-4.5" style={{ color: 'var(--accent-amber)' }} />
        </div>
        <div className="min-w-0">
          <h2 className="text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
            Public Disclosure Data Use
          </h2>
          <p className="mt-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            {POLITICIANS_DATA_USE_NOTICE}
          </p>
          <ul className="mt-3 grid gap-2 text-[11px] leading-relaxed" style={{ color: 'var(--text-muted)' }}>
            {POLITICIANS_DATA_USE_BULLETS.map((item) => (
              <li key={item} className="flex gap-2">
                <span className="mt-[0.55em] h-1 w-1 rounded-full flex-shrink-0" style={{ background: 'var(--accent-amber)' }} />
                <span>{item}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </section>
  );
}
