import { useEffect, useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import LoadingSpinner from '../../../components/LoadingSpinner';
import { api, type PoliticiansFilerResponse } from '../../../api';
import { ExternalLink, Landmark, ShieldCheck, X } from 'lucide-react';

interface Props {
  filerId: string | null;
  onClose: () => void;
}

export default function FilerDrawer({ filerId, onClose }: Props) {
  const closeRef = useRef<HTMLButtonElement | null>(null);
  const filerQ = useQuery({
    queryKey: ['politiciansFiler', filerId],
    queryFn: () => api.politiciansFiler(filerId || ''),
    enabled: Boolean(filerId),
    staleTime: 60_000,
    retry: false,
  });

  useEffect(() => {
    if (!filerId) return;
    const previous = document.activeElement as HTMLElement | null;
    closeRef.current?.focus();
    function onKey(event: KeyboardEvent) {
      if (event.key === 'Escape') onClose();
    }
    window.addEventListener('keydown', onKey);
    return () => {
      window.removeEventListener('keydown', onKey);
      previous?.focus?.();
    };
  }, [filerId, onClose]);

  if (!filerId) return null;

  const data = filerQ.data as PoliticiansFilerResponse | undefined;

  return (
    <div className="fixed inset-0 z-50 flex justify-end bg-black/40" role="presentation" onMouseDown={onClose}>
      <aside
        role="dialog"
        aria-modal="true"
        aria-label="Politician disclosure detail"
        className="h-full w-full max-w-[560px] overflow-y-auto border-l border-[var(--violet-8)] p-5 shadow-2xl"
        style={{ background: 'var(--void-surface)' }}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Landmark className="h-4 w-4" style={{ color: 'var(--accent-cyan)' }} />
              <h2 className="truncate text-[15px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
                {data?.metadata?.filer_name || filerId}
              </h2>
            </div>
            <p className="mt-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
              Delayed public disclosure records
            </p>
          </div>
          <button
            ref={closeRef}
            type="button"
            onClick={onClose}
            className="flex h-9 w-9 items-center justify-center rounded-[8px] border hover:brightness-110"
            style={{ borderColor: 'var(--violet-8)', color: 'var(--text-secondary)' }}
            aria-label="Close politician detail"
          >
            <X className="h-4 w-4" />
          </button>
        </div>

        {filerQ.isLoading && <LoadingSpinner text="Loading politician detail..." variant="cards" />}
        {data?.status === 'ok' && (
          <div className="mt-5 space-y-4">
            <Metadata data={data} />
            <CommitteeEnrichment data={data} />
            <Rollups data={data} />
            <RecentTrades data={data} />
            <SourceDocuments data={data} />
          </div>
        )}
      </aside>
    </div>
  );
}

function Metadata({ data }: { data: PoliticiansFilerResponse }) {
  const meta = data.metadata;
  return (
    <section className="rounded-[8px] border border-[var(--violet-8)] p-3">
      <div className="grid grid-cols-2 gap-3 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
        <Field label="Chamber" value={meta?.chamber} />
        <Field label="Party/State" value={[meta?.party, meta?.state].filter(Boolean).join('/')} />
        <Field label="Source" value={meta?.source} />
        <Field label="Metadata" value={meta?.metadata_complete ? 'complete' : 'incomplete'} />
      </div>
    </section>
  );
}

function CommitteeEnrichment({ data }: { data: PoliticiansFilerResponse }) {
  const committees = data.metadata?.committee_enrichment || [];
  if (committees.length === 0) return null;
  return (
    <section className="rounded-[8px] border border-[var(--violet-8)] p-3">
      <div className="flex items-center gap-2 text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
        <ShieldCheck className="h-4 w-4" style={{ color: 'var(--accent-amber)' }} />
        Committee Enrichment
      </div>
      <p className="mt-1 text-[11px]" style={{ color: 'var(--text-muted)' }}>
        Enrichment from member metadata where available, not a disclosure source field.
      </p>
      <div className="mt-2 flex flex-wrap gap-1.5">
        {committees.map((committee) => (
          <span key={committee} className="rounded-[6px] border border-[var(--violet-8)] px-2 py-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            {committee}
          </span>
        ))}
      </div>
    </section>
  );
}

function Rollups({ data }: { data: PoliticiansFilerResponse }) {
  return (
    <section className="grid gap-3 md:grid-cols-2">
      <RollupList title="Top Tickers" rows={data.top_tickers || []} labelKey="ticker" />
      <RollupList title="Top Sectors (Enrichment)" rows={data.top_sectors || []} labelKey="sector" />
      <div className="rounded-[8px] border border-[var(--violet-8)] p-3">
        <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>Ownership</h3>
        <div className="mt-2 grid grid-cols-2 gap-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          {Object.entries(data.ownership_breakdown || {}).map(([owner, count]) => (
            <div key={owner} className="flex justify-between gap-2"><span>{owner}</span><span>{count}</span></div>
          ))}
        </div>
      </div>
      <div className="rounded-[8px] border border-[var(--violet-8)] p-3">
        <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>Delay Stats</h3>
        <div className="mt-2 space-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
          <div>Average: {data.delay_stats?.average_days ?? '—'} days</div>
          <div>Median: {data.delay_stats?.median_days ?? '—'} days</div>
          <div>Late filings: {data.delay_stats?.late_filing_count ?? 0}</div>
        </div>
      </div>
    </section>
  );
}

function RecentTrades({ data }: { data: PoliticiansFilerResponse }) {
  return (
    <section className="rounded-[8px] border border-[var(--violet-8)] p-3">
      <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>Recent Trades</h3>
      <div className="mt-2 space-y-2">
        {(data.recent_trades || []).slice(0, 10).map((row, idx) => (
          <div key={String(row.trade_id || idx)} className="grid grid-cols-[1fr_auto] gap-3 rounded-[8px] bg-[rgba(255,255,255,0.025)] px-3 py-2 text-[11px]">
            <div className="min-w-0">
              <div className="truncate font-semibold" style={{ color: 'var(--text-luminous)' }}>{String(row.ticker || '—')} · {String(row.transaction_type || 'unknown')}</div>
              <div className="truncate" style={{ color: 'var(--text-secondary)' }}>{String(row.disclosure_date || '—')} · {String(row.owner || 'unknown')}</div>
            </div>
            <div className="text-right" style={{ color: 'var(--text-secondary)' }}>{formatUsd(Number(row.amount_mid_usd || 0))}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function SourceDocuments({ data }: { data: PoliticiansFilerResponse }) {
  return (
    <section className="rounded-[8px] border border-[var(--violet-8)] p-3">
      <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>Source Documents</h3>
      <div className="mt-2 space-y-2">
        {(data.source_documents || []).map((doc, idx) => (
          <a
            key={String(doc.official_source_url || idx)}
            href={String(doc.official_source_url)}
            target="_blank"
            rel="noopener noreferrer"
            className="flex items-center justify-between gap-3 rounded-[8px] border border-[var(--violet-8)] px-3 py-2 text-[11px] hover:brightness-110"
            style={{ color: 'var(--accent-cyan)' }}
          >
            <span className="min-w-0 truncate">{String(doc.disclosure_date || doc.report_id || 'Official source')}</span>
            <ExternalLink className="h-3 w-3 shrink-0" />
          </a>
        ))}
      </div>
    </section>
  );
}

function RollupList({ title, rows, labelKey }: { title: string; rows: Array<Record<string, number | string>>; labelKey: string }) {
  return (
    <div className="rounded-[8px] border border-[var(--violet-8)] p-3">
      <h3 className="text-[12px] font-semibold" style={{ color: 'var(--text-luminous)' }}>{title}</h3>
      <div className="mt-2 space-y-1 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        {rows.slice(0, 5).map((row) => (
          <div key={String(row[labelKey])} className="flex justify-between gap-2">
            <span className="truncate">{String(row[labelKey])}</span>
            <span>{formatUsd(Number(row.amount_mid_usd || 0))}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function Field({ label, value }: { label: string; value?: string | null }) {
  return (
    <div className="min-w-0">
      <div className="text-[10px] font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>{label}</div>
      <div className="mt-1 truncate">{value || '—'}</div>
    </div>
  );
}

function formatUsd(value: number): string {
  return Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', notation: 'compact', maximumFractionDigits: 1 }).format(value);
}
