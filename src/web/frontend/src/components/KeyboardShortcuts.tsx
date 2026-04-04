/**
 * Keyboard Shortcut Cheat Sheet (Story 10.5)
 *
 * Full-screen cosmic glass overlay showing all shortcuts, triggered by `?`
 */
import { useState, useEffect, useMemo, useRef } from 'react';
import { X, Search } from 'lucide-react';

interface Shortcut {
  keys: string[];
  description: string;
}

interface ShortcutSection {
  title: string;
  shortcuts: Shortcut[];
}

const SECTIONS: ShortcutSection[] = [
  {
    title: 'Navigation',
    shortcuts: [
      { keys: ['1'], description: 'Go to Dashboard' },
      { keys: ['2'], description: 'Go to Signals' },
      { keys: ['3'], description: 'Go to Charts' },
      { keys: ['4'], description: 'Go to Risk' },
      { keys: ['5'], description: 'Go to Tuning' },
      { keys: ['6'], description: 'Go to Diagnostics' },
    ],
  },
  {
    title: 'Signals',
    shortcuts: [
      { keys: ['/'], description: 'Focus search' },
      { keys: ['F'], description: 'Toggle filters' },
      { keys: ['R'], description: 'Refresh data' },
    ],
  },
  {
    title: 'Charts',
    shortcuts: [
      { keys: ['['], description: 'Previous symbol' },
      { keys: [']'], description: 'Next symbol' },
      { keys: ['L'], description: 'Toggle layers panel' },
    ],
  },
  {
    title: 'General',
    shortcuts: [
      { keys: ['Cmd', 'K'], description: 'Command Palette' },
      { keys: ['?'], description: 'Show keyboard shortcuts' },
      { keys: ['Esc'], description: 'Close overlay / Clear search' },
    ],
  },
];

function KeyBadge({ label }: { label: string }) {
  return (
    <kbd
      className="inline-flex items-center justify-center text-[11px] font-mono font-medium"
      style={{
        minWidth: 24,
        height: 24,
        padding: '0 6px',
        borderRadius: 6,
        background: 'var(--void-active)',
        border: '1px solid var(--border-void)',
        color: 'var(--text-secondary)',
        boxShadow: 'inset 0 -1px 0 rgba(0,0,0,0.2)',
      }}
    >
      {label}
    </kbd>
  );
}

export function KeyboardShortcutOverlay() {
  const [open, setOpen] = useState(false);
  const [search, setSearch] = useState('');
  const searchRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement;
      const isInput = target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable;

      if (e.key === '?' && !isInput) {
        e.preventDefault();
        setOpen(true);
      }
      if (e.key === 'Escape' && open) {
        setOpen(false);
        setSearch('');
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [open]);

  useEffect(() => {
    if (open) {
      setTimeout(() => searchRef.current?.focus(), 200);
    }
  }, [open]);

  const filtered = useMemo(() => {
    if (!search) return SECTIONS;
    const q = search.toLowerCase();
    return SECTIONS.map((sec) => ({
      ...sec,
      shortcuts: sec.shortcuts.filter(
        (s) =>
          s.description.toLowerCase().includes(q) ||
          s.keys.some((k) => k.toLowerCase().includes(q)),
      ),
    })).filter((sec) => sec.shortcuts.length > 0);
  }, [search]);

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-center justify-center"
      style={{ background: 'rgba(3,0,20,0.8)', backdropFilter: 'blur(8px)' }}
      onClick={(e) => { if (e.target === e.currentTarget) { setOpen(false); setSearch(''); } }}
    >
      <div
        className="glass-card w-full max-w-[680px] max-h-[80vh] overflow-y-auto p-6"
        style={{ animation: 'fade-up 200ms cubic-bezier(0.2,0,0,1) both' }}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold" style={{ color: 'var(--text-luminous)' }}>
            Keyboard Shortcuts
          </h2>
          <button
            onClick={() => { setOpen(false); setSearch(''); }}
            className="p-1.5 rounded-lg transition-colors duration-150 hover:bg-[rgba(139,92,246,0.08)]"
            style={{ color: 'var(--text-muted)' }}
          >
            <X size={16} />
          </button>
        </div>

        {/* Search */}
        <div className="relative mb-5">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />
          <input
            ref={searchRef}
            type="text"
            placeholder="Search shortcuts..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full h-9 pl-9 pr-3 text-sm rounded-lg outline-none transition-shadow duration-200
              focus:ring-2 focus:ring-[rgba(139,92,246,0.3)]"
            style={{
              background: 'rgba(10,10,26,0.6)',
              border: '1px solid var(--border-void)',
              color: 'var(--text-primary)',
            }}
          />
        </div>

        {/* Sections */}
        <div className="space-y-5">
          {filtered.map((sec) => (
            <div key={sec.title}>
              <h3 className="text-xs font-semibold uppercase tracking-wider mb-2" style={{ color: 'var(--text-violet)' }}>
                {sec.title}
              </h3>
              <div className="space-y-1">
                {sec.shortcuts.map((s, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between px-3 py-2 rounded-lg transition-colors duration-150 hover:bg-[rgba(139,92,246,0.04)]"
                  >
                    <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                      {s.description}
                    </span>
                    <div className="flex items-center gap-1">
                      {s.keys.map((k, ki) => (
                        <span key={ki} className="flex items-center gap-1">
                          {ki > 0 && <span className="text-[10px]" style={{ color: 'var(--text-muted)' }}>+</span>}
                          <KeyBadge label={k} />
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
          {filtered.length === 0 && (
            <p className="text-sm text-center py-6" style={{ color: 'var(--text-muted)' }}>
              No shortcuts match "{search}"
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
