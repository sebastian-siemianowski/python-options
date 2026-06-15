import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api, type PoliticiansSourceEntry } from '../../../api';
import { AlertTriangle, CheckCircle2, ChevronDown, ChevronUp, XCircle } from 'lucide-react';

export default function SourceHealthStrip() {
  const [open, setOpen] = useState(false);
  const healthQ = useQuery({
    queryKey: ['politiciansSourceHealth'],
    queryFn: api.politiciansSourceHealth,
    staleTime: 60_000,
    retry: false,
  });

  const sources = healthQ.data?.sources || {};
  const house = sources.house;
  const senate = sources.senate;
  const lowConfidence = Object.values(sources).reduce((total, source) => total + (source.low_confidence_rows || 0), 0);
  const newestSync = latestSync(Object.values(sources));

  return (
    <section className="glass-card overflow-hidden">
      <button
        type="button"
        onClick={() => setOpen((value) => !value)}
        className="flex w-full flex-col gap-3 p-4 text-left md:flex-row md:items-center md:justify-between"
      >
        <div className="flex flex-wrap items-center gap-2">
          <SourcePill label="House" source={house} loading={healthQ.isLoading} />
          <SourcePill label="Senate" source={senate} loading={healthQ.isLoading} />
        </div>
        <div className="flex flex-wrap items-center gap-3 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <span>Last sync: {newestSync || '—'}</span>
          <span>Review rows: {lowConfidence.toLocaleString()}</span>
          <span className="inline-flex items-center gap-1" style={{ color: hasSourceWarning(sources) ? 'var(--accent-amber)' : 'var(--text-secondary)' }}>
            {hasSourceWarning(sources) ? 'Source warnings visible' : 'No source warnings'}
            {open ? <ChevronUp className="h-3.5 w-3.5" /> : <ChevronDown className="h-3.5 w-3.5" />}
          </span>
        </div>
      </button>

      {open && (
        <div className="grid gap-3 border-t border-[var(--violet-8)] p-4 md:grid-cols-2">
          {Object.entries(sources).map(([source, entry]) => (
            <div key={source} className="rounded-[8px] border border-[var(--violet-8)] p-3">
              <div className="flex items-center justify-between gap-3">
                <h3 className="text-[12px] font-semibold capitalize" style={{ color: 'var(--text-luminous)' }}>{source}</h3>
                <span className="text-[11px]" style={{ color: statusColor(entry.status) }}>{entry.status}</span>
              </div>
              <div className="mt-2 grid grid-cols-2 gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
                <div>Parse success: {entry.parse_success_rate == null ? '—' : `${Math.round(entry.parse_success_rate * 100)}%`}</div>
                <div>Low confidence: {entry.low_confidence_rows}</div>
                <div className="col-span-2">Last sync: {entry.last_sync_time || '—'}</div>
              </div>
              <p className="mt-2 text-[11px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>{entry.remediation}</p>
              {entry.recent_errors.length > 0 && (
                <div className="mt-2 space-y-1">
                  {entry.recent_errors.slice(0, 3).map((error, idx) => (
                    <pre key={idx} className="max-h-20 overflow-auto rounded-[6px] bg-black/20 p-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                      {formatError(error)}
                    </pre>
                  ))}
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function SourcePill({ label, source, loading }: { label: string; source?: PoliticiansSourceEntry; loading: boolean }) {
  const status = loading ? 'loading' : source?.status || 'offline';
  const Icon = status === 'ok' ? CheckCircle2 : status === 'offline' ? XCircle : AlertTriangle;
  return (
    <span className="inline-flex min-h-8 items-center gap-2 rounded-[8px] border px-3 text-[12px]" style={{ borderColor: statusColor(status), color: statusColor(status), background: statusBg(status) }}>
      <Icon className="h-3.5 w-3.5" />
      {label}: {status}
      {source?.parse_success_rate != null && <span className="tabular-nums">{Math.round(source.parse_success_rate * 100)}%</span>}
    </span>
  );
}

function hasSourceWarning(sources: Record<string, PoliticiansSourceEntry>): boolean {
  return Object.values(sources).some((source) => source.status !== 'ok' || source.recent_errors.length > 0);
}

function latestSync(sources: PoliticiansSourceEntry[]): string | null {
  const values = sources.map((source) => source.last_sync_time).filter(Boolean) as string[];
  values.sort();
  return values.length ? values[values.length - 1] : null;
}

function statusColor(status: string): string {
  if (status === 'ok') return 'var(--accent-emerald)';
  if (status === 'offline') return 'var(--accent-rose)';
  if (status === 'loading') return 'var(--text-muted)';
  return 'var(--accent-amber)';
}

function statusBg(status: string): string {
  if (status === 'ok') return 'var(--emerald-12)';
  if (status === 'offline') return 'var(--rose-12)';
  if (status === 'loading') return 'rgba(255,255,255,0.025)';
  return 'var(--amber-12)';
}

function formatError(error: unknown): string {
  if (typeof error === 'string') return error;
  try {
    return JSON.stringify(error, null, 2);
  } catch {
    return String(error);
  }
}
