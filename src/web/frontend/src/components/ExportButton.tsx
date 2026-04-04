/**
 * Export Popover (Story 12.3)
 *
 * Three format tiles: CSV, JSON, Clipboard. Shows loading/success/error states.
 */
import { useState, useRef, useEffect } from 'react';
import { Download, FileText, FileJson, Clipboard, Check, Loader2, X } from 'lucide-react';
import { exportData, type ExportFormat } from '../utils/exportData';

interface ExportButtonProps {
  filename: string;
  columns: { key: string; label: string }[];
  data: Record<string, unknown>[];
  /** Number of visible (filtered) rows */
  filteredCount?: number;
  totalCount?: number;
}

type ExportState = 'idle' | 'loading' | 'success' | 'error';

export function ExportButton({ filename, columns, data, filteredCount, totalCount }: ExportButtonProps) {
  const [open, setOpen] = useState(false);
  const [state, setState] = useState<ExportState>('idle');
  const [lastFormat, setLastFormat] = useState<ExportFormat | null>(null);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, [open]);

  // Auto-reset after success/error
  useEffect(() => {
    if (state === 'success' || state === 'error') {
      const t = setTimeout(() => setState('idle'), 2000);
      return () => clearTimeout(t);
    }
  }, [state]);

  const doExport = async (format: ExportFormat) => {
    setState('loading');
    setLastFormat(format);
    const result = await exportData({ filename, columns, data, format });
    setState(result);
  };

  const formats: { key: ExportFormat; label: string; desc: string; icon: typeof FileText }[] = [
    { key: 'csv', label: 'CSV', desc: 'Spreadsheet format', icon: FileText },
    { key: 'json', label: 'JSON', desc: 'Structured data', icon: FileJson },
    { key: 'clipboard', label: 'Copy', desc: 'Tab-delimited', icon: Clipboard },
  ];

  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1.5 h-8 px-3 rounded-lg text-xs font-medium transition-all duration-200
          hover:bg-[rgba(139,92,246,0.08)]"
        style={{ color: 'var(--text-secondary)', border: '1px solid var(--border-void)' }}
        aria-label="Export data"
      >
        <Download size={13} />
        Export
      </button>

      {open && (
        <div
          className="absolute right-0 top-full mt-2 w-56 glass-card p-3 z-50"
          style={{ animation: 'fade-up 150ms cubic-bezier(0.2,0,0,1) both' }}
        >
          {/* Header */}
          <div className="flex items-center justify-between mb-2">
            <span className="text-[10px] font-semibold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>
              Export {filteredCount && totalCount && filteredCount < totalCount
                ? `${filteredCount} of ${totalCount} rows`
                : `${data.length} rows`}
            </span>
          </div>

          {/* Format tiles */}
          <div className="space-y-1.5">
            {formats.map((f) => {
              const isActive = state === 'loading' && lastFormat === f.key;
              const isSuccess = state === 'success' && lastFormat === f.key;
              const isError = state === 'error' && lastFormat === f.key;
              const Icon = f.icon;

              return (
                <button
                  key={f.key}
                  onClick={() => doExport(f.key)}
                  disabled={state === 'loading'}
                  className="w-full flex items-center gap-3 px-3 py-2.5 rounded-lg text-left transition-all duration-200
                    hover:bg-[rgba(139,92,246,0.06)] disabled:opacity-50"
                  style={{ border: '1px solid var(--border-void)' }}
                >
                  <div
                    className="w-7 h-7 rounded-md flex items-center justify-center flex-shrink-0"
                    style={{ background: 'rgba(139,92,246,0.08)' }}
                  >
                    {isActive ? (
                      <Loader2 size={14} className="animate-spin" style={{ color: '#8b5cf6' }} />
                    ) : isSuccess ? (
                      <Check size={14} style={{ color: '#34d399' }} />
                    ) : isError ? (
                      <X size={14} style={{ color: '#fb7185' }} />
                    ) : (
                      <Icon size={14} style={{ color: '#a78bfa' }} />
                    )}
                  </div>
                  <div>
                    <p className="text-xs font-medium" style={{ color: 'var(--text-primary)' }}>{f.label}</p>
                    <p className="text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {isSuccess ? 'Done!' : isError ? 'Failed' : f.desc}
                    </p>
                  </div>
                </button>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
