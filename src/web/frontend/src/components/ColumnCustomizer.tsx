import { useEffect, useRef, useState } from 'react';

export interface ColumnDef {
  key: string;
  label: string;
  /** Locked columns can't be toggled off (e.g., primary identifier). */
  locked?: boolean;
  /** Optional hint shown next to the label. */
  hint?: string;
}

interface ColumnCustomizerProps {
  columns: ColumnDef[];
  visible: Set<string>;
  onToggle: (key: string) => void;
  onReset: () => void;
  /** Optional label on the button. */
  label?: string;
}

/**
 * Compact popover that lets users toggle column visibility.
 *
 * UX principles:
 *  - Sortable ≠ visible. Clicking a column header still sorts; this popover
 *    independently controls which columns are rendered at all.
 *  - Locked columns (e.g., Asset, Signal) always remain visible.
 *  - Selection persists to localStorage by the parent (this is a pure component).
 *  - Keyboard: Esc closes, outside-click closes, focus ring on all interactive elements.
 *  - No emojis — pure SVG iconography.
 */
export function ColumnCustomizer({
  columns,
  visible,
  onToggle,
  onReset,
  label = 'Columns',
}: ColumnCustomizerProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement | null>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(e.target as Node)) setOpen(false);
    };
    const esc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setOpen(false);
    };
    window.addEventListener('mousedown', handler);
    window.addEventListener('keydown', esc);
    return () => {
      window.removeEventListener('mousedown', handler);
      window.removeEventListener('keydown', esc);
    };
  }, [open]);

  const totalToggleable = columns.filter((c) => !c.locked).length;
  const visibleToggleable = columns.filter((c) => !c.locked && visible.has(c.key)).length;

  return (
    <div ref={rootRef} className="relative inline-block">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="inline-flex items-center gap-1.5 px-2.5 h-7 rounded-md text-[11px] font-medium transition-colors focus:outline-none focus:ring-1 focus:ring-[var(--accent-violet)]"
        style={{
          color: open ? 'var(--accent-violet)' : 'var(--text-secondary)',
          background: open ? 'var(--void-active)' : 'transparent',
          border: '1px solid var(--border-void)',
        }}
        aria-expanded={open}
        aria-haspopup="true"
        title="Show / hide columns"
      >
        {/* Columns grid SVG icon */}
        <svg
          width="12"
          height="12"
          viewBox="0 0 16 16"
          fill="none"
          stroke="currentColor"
          strokeWidth="1.5"
          strokeLinecap="round"
          strokeLinejoin="round"
          aria-hidden="true"
        >
          <rect x="1.5" y="2.5" width="3.5" height="11" rx="0.6" />
          <rect x="6.25" y="2.5" width="3.5" height="11" rx="0.6" />
          <rect x="11" y="2.5" width="3.5" height="11" rx="0.6" />
        </svg>
        <span>{label}</span>
        <span
          className="tabular-nums text-[10px]"
          style={{ color: 'var(--text-muted)' }}
        >
          {visibleToggleable}/{totalToggleable}
        </span>
      </button>

      {open && (
        <div
          role="menu"
          className="absolute z-50 mt-1.5 right-0 rounded-md overflow-hidden"
          style={{
            minWidth: 220,
            background: 'var(--void-surface, #0d0b16)',
            border: '1px solid var(--border-void)',
            boxShadow:
              '0 12px 32px rgba(0,0,0,0.55), 0 0 0 1px rgba(139,92,246,0.08), 0 0 24px rgba(139,92,246,0.08)',
          }}
        >
          <div
            className="px-3 py-2 text-[10px] font-semibold uppercase tracking-[0.08em]"
            style={{
              color: 'var(--text-muted)',
              borderBottom: '1px solid var(--border-void)',
              background: 'var(--void-hover)',
            }}
          >
            Show columns
          </div>

          <ul className="py-1 max-h-[360px] overflow-auto">
            {columns.map((col) => {
              const isVisible = visible.has(col.key);
              const disabled = !!col.locked;
              return (
                <li key={col.key}>
                  <button
                    type="button"
                    disabled={disabled}
                    onClick={() => {
                      if (!disabled) onToggle(col.key);
                    }}
                    className="w-full flex items-center gap-2.5 px-3 py-1.5 text-left text-[12px] transition-colors focus:outline-none"
                    style={{
                      color: disabled
                        ? 'var(--text-muted)'
                        : isVisible
                          ? 'var(--text-primary)'
                          : 'var(--text-secondary)',
                      cursor: disabled ? 'not-allowed' : 'pointer',
                      opacity: disabled ? 0.6 : 1,
                    }}
                    onMouseEnter={(e) => {
                      if (!disabled) e.currentTarget.style.background = 'var(--void-hover)';
                    }}
                    onMouseLeave={(e) => {
                      e.currentTarget.style.background = 'transparent';
                    }}
                  >
                    {/* Check box */}
                    <span
                      className="inline-flex items-center justify-center flex-shrink-0 rounded"
                      style={{
                        width: 14,
                        height: 14,
                        background: isVisible ? 'var(--accent-violet)' : 'transparent',
                        border: `1px solid ${isVisible ? 'var(--accent-violet)' : 'var(--border-void)'}`,
                        boxShadow: isVisible ? '0 0 6px rgba(139,92,246,0.35)' : 'none',
                        transition: 'all 120ms ease',
                      }}
                    >
                      {isVisible && (
                        <svg
                          width="10"
                          height="10"
                          viewBox="0 0 16 16"
                          fill="none"
                          stroke="#ffffff"
                          strokeWidth="2.2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          aria-hidden="true"
                        >
                          <path d="M3 8.5l3.2 3.2L13 4.8" />
                        </svg>
                      )}
                    </span>

                    <span className="flex-1 truncate">{col.label}</span>

                    {col.hint && (
                      <span
                        className="text-[9px] uppercase tracking-wider"
                        style={{ color: 'var(--text-muted)' }}
                      >
                        {col.hint}
                      </span>
                    )}
                    {disabled && (
                      <span
                        className="text-[9px] uppercase tracking-wider"
                        style={{ color: 'var(--text-muted)' }}
                        title="Always visible"
                      >
                        locked
                      </span>
                    )}
                  </button>
                </li>
              );
            })}
          </ul>

          <div
            className="flex items-center justify-between px-3 py-1.5"
            style={{
              borderTop: '1px solid var(--border-void)',
              background: 'var(--void-hover)',
            }}
          >
            <span
              className="text-[10px]"
              style={{ color: 'var(--text-muted)' }}
            >
              Saved automatically
            </span>
            <button
              type="button"
              onClick={onReset}
              className="text-[10px] font-medium transition-colors focus:outline-none focus:ring-1 focus:ring-[var(--accent-violet)] rounded px-1.5 py-0.5"
              style={{ color: 'var(--accent-violet)' }}
            >
              Reset
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
