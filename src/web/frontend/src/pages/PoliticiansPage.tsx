import { useMutation, useQuery, useQueryClient, type UseMutationResult } from '@tanstack/react-query';
import { useState } from 'react';
import PageHeader from '../components/PageHeader';
import LoadingSpinner from '../components/LoadingSpinner';
import { api } from '../api';
import DataUseNotice from '../features/politicians/components/DataUseNotice';
import PoliticianInsightBar, { insightFilterCount, type PoliticianInsightFilter } from '../features/politicians/components/PoliticianInsightBar';
import TradeFeedTable from '../features/politicians/components/TradeFeedTable';
import AssetActivityPanel from '../features/politicians/components/AssetActivityPanel';
import FilerDrawer from '../features/politicians/components/FilerDrawer';
import SourceHealthStrip from '../features/politicians/components/SourceHealthStrip';
import { getFrontendPoliticiansComplianceMode } from '../features/politicians/disclaimers';
import { CheckCircle2, DatabaseZap, Lock, RefreshCw, ShieldAlert } from 'lucide-react';

export default function PoliticiansPage() {
  const queryClient = useQueryClient();
  const [activeFilter, setActiveFilter] = useState<PoliticianInsightFilter>('all');
  const [selectedSymbol, setSelectedSymbol] = useState<string | null>(null);
  const [selectedFiler, setSelectedFiler] = useState<string | null>(null);
  const noticeQ = useQuery({
    queryKey: ['politiciansNotice'],
    queryFn: api.politiciansNotice,
    staleTime: 60_000,
    retry: false,
  });
  const summaryQ = useQuery({
    queryKey: ['politiciansSummary'],
    queryFn: api.politiciansSummary,
    staleTime: 60_000,
    retry: false,
    enabled: noticeQ.data?.enabled === true,
  });
  const syncMutation = useMutation({
    mutationFn: api.politiciansSync,
    onSuccess: async () => {
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ['politiciansSummary'] }),
        queryClient.invalidateQueries({ queryKey: ['politiciansSourceHealth'] }),
        queryClient.invalidateQueries({ queryKey: ['politiciansTrades'] }),
        queryClient.invalidateQueries({ queryKey: ['politiciansAsset'] }),
        queryClient.invalidateQueries({ queryKey: ['politiciansFiler'] }),
      ]);
    },
  });

  if (noticeQ.isLoading) return <LoadingSpinner text="Checking politician disclosure availability..." />;

  const notice = noticeQ.data;
  const frontendMode = getFrontendPoliticiansComplianceMode();
  const disabled = !notice || notice.status === 'disabled' || notice.enabled === false;
  const matchedCount = insightFilterCount(summaryQ.data, activeFilter);

  return (
    <>
      <PageHeader title="Politicians">
        Delayed congressional disclosures parsed into ticker-level market context.
      </PageHeader>

      <div className="max-w-[1500px] space-y-4">
        {disabled ? (
          <DisabledPanel
            backendMode={notice?.compliance_mode || 'research_only'}
            frontendMode={frontendMode}
            reason={notice?.disabled_reason || 'Backend notice endpoint unavailable'}
          />
        ) : (
          <>
            <section className="glass-card overflow-hidden">
              <div className="grid gap-5 p-5 lg:grid-cols-[minmax(0,1fr)_auto] lg:items-center">
                <div className="flex min-w-0 items-start gap-4">
                  <div
                    className="flex h-11 w-11 flex-shrink-0 items-center justify-center rounded-[8px]"
                    style={{ background: 'linear-gradient(135deg, rgba(62,232,165,0.18), rgba(56,217,245,0.10))' }}
                  >
                    <ShieldAlert className="h-5 w-5" style={{ color: 'var(--accent-emerald)' }} />
                  </div>
                  <div className="min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <h2 className="text-[15px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
                        Public Disclosure Monitor
                      </h2>
                      <span
                        className="inline-flex h-6 items-center rounded-[6px] border px-2 text-[10px] font-semibold uppercase"
                        style={{ borderColor: 'rgba(62,232,165,0.28)', color: 'var(--accent-emerald)', background: 'var(--emerald-12)' }}
                      >
                        Research mode
                      </span>
                    </div>
                    <div className="mt-3 grid gap-2 text-[11px] sm:grid-cols-3" style={{ color: 'var(--text-secondary)' }}>
                      <StatusDatum label="Compliance" value={`${notice.compliance_mode} / ${frontendMode}`} />
                      <StatusDatum label="Newest disclosure" value={summaryQ.data?.newest_disclosure_date || 'pending'} />
                      <StatusDatum label="Parsed records" value={summaryQ.data?.total_trades?.toLocaleString() || '0'} />
                    </div>
                    <SyncStatus mutation={syncMutation} />
                  </div>
                </div>
                <div className="flex flex-col gap-2 sm:flex-row lg:flex-col lg:items-end">
                  <button
                    type="button"
                    onClick={() => syncMutation.mutate()}
                    disabled={syncMutation.isPending}
                    className="inline-flex h-11 items-center justify-center gap-2 rounded-[8px] border px-4 text-[12px] font-semibold transition disabled:cursor-wait disabled:opacity-70"
                    style={{
                      borderColor: 'rgba(56,217,245,0.55)',
                      color: 'var(--text-luminous)',
                      background: 'linear-gradient(135deg, rgba(56,217,245,0.16), rgba(139,92,246,0.10))',
                      boxShadow: '0 0 22px rgba(56,217,245,0.10)',
                    }}
                    title="Sync and parse public disclosure records"
                  >
                    <RefreshCw className={`h-4 w-4 ${syncMutation.isPending ? 'animate-spin' : ''}`} />
                    {syncMutation.isPending ? 'Syncing...' : 'Sync & Parse'}
                  </button>
                  <div className="inline-flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-muted)' }}>
                    <DatabaseZap className="h-3.5 w-3.5" />
                    {summaryQ.data?.data_age_seconds == null ? 'cache pending' : `cache age ${formatAge(summaryQ.data.data_age_seconds)}`}
                  </div>
                </div>
              </div>
            </section>
            <PoliticianInsightBar
              summary={summaryQ.data}
              isLoading={summaryQ.isLoading}
              activeFilter={activeFilter}
              onFilterChange={setActiveFilter}
            />
            <SourceHealthStrip />
            {selectedSymbol && <AssetActivityPanel symbol={selectedSymbol} />}
            {!summaryQ.isLoading && summaryQ.data?.status === 'ok' && matchedCount === 0 && (
              <section className="glass-card p-5">
                <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                  <div>
                    <h2 className="text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
                      No Disclosures Matched Current Filters
                    </h2>
                    <p className="mt-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
                      This does not mean no politicians traded. It means no delayed public disclosure records matched the selected filter and current data window.
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => syncMutation.mutate()}
                    disabled={syncMutation.isPending}
                    className="inline-flex h-9 shrink-0 items-center justify-center gap-2 rounded-[8px] border px-3 text-[12px] font-semibold disabled:cursor-wait disabled:opacity-70"
                    style={{ borderColor: 'var(--violet-8)', color: 'var(--accent-cyan)' }}
                    title="Sync and parse public disclosure records"
                  >
                    <RefreshCw className={`h-3.5 w-3.5 ${syncMutation.isPending ? 'animate-spin' : ''}`} />
                    {syncMutation.isPending ? 'Syncing...' : 'Sync & Parse'}
                  </button>
                </div>
              </section>
            )}
            <TradeFeedTable
              insightFilter={activeFilter}
              onSelectTicker={setSelectedSymbol}
              onSelectFiler={setSelectedFiler}
            />
            <FilerDrawer filerId={selectedFiler} onClose={() => setSelectedFiler(null)} />
          </>
        )}
        <DataUseNotice />
      </div>
    </>
  );
}

function StatusDatum({ label, value }: { label: string; value: string }) {
  return (
    <div className="min-w-0 rounded-[8px] border border-[var(--violet-8)] px-3 py-2" style={{ background: 'rgba(255,255,255,0.025)' }}>
      <div className="text-[9px] font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
        {label}
      </div>
      <div className="mt-1 truncate text-[12px] font-medium" style={{ color: 'var(--text-luminous)' }}>
        {value}
      </div>
    </div>
  );
}

function SyncStatus({
  mutation,
}: {
  mutation: UseMutationResult<Awaited<ReturnType<typeof api.politiciansSync>>, Error, void, unknown>;
}) {
  if (mutation.isIdle) return null;
  if (mutation.isPending) {
    return (
      <p className="mt-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
        Fetching official records, parsing transactions, and refreshing the feed...
      </p>
    );
  }
  if (mutation.isError) {
    return (
      <p className="mt-2 text-[11px]" style={{ color: 'var(--accent-rose)' }}>
        Sync failed: {mutation.error instanceof Error ? mutation.error.message : 'unknown error'}
      </p>
    );
  }
  const counts = mutation.data?.counts || {};
  const parsedTrades = Number(counts.valid_count || counts.trade_count || 0);
  return (
    <p className="mt-2 inline-flex items-center gap-1 text-[11px]" style={{ color: 'var(--accent-emerald)' }}>
      <CheckCircle2 className="h-3.5 w-3.5" />
      Sync {mutation.data?.status || 'complete'} · {parsedTrades.toLocaleString()} parsed trades
    </p>
  );
}

function formatAge(seconds: number): string {
  if (seconds < 60) return `${Math.max(0, Math.round(seconds))}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  return `${Math.round(seconds / 3600)}h`;
}

function DisabledPanel({
  backendMode,
  frontendMode,
  reason,
}: {
  backendMode: string;
  frontendMode: string;
  reason: string;
}) {
  return (
    <section className="glass-card p-5 border-l-2 border-amber-500/20">
      <div className="flex items-start gap-3">
        <div
          className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
          style={{ background: 'var(--amber-12)' }}
        >
          <Lock className="w-4.5 h-4.5" style={{ color: 'var(--accent-amber)' }} />
        </div>
        <div>
          <h2 className="text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
            Politician Monitoring Disabled
          </h2>
          <p className="mt-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            {reason}. Backend mode: {backendMode}. Frontend mode: {frontendMode}.
          </p>
        </div>
      </div>
    </section>
  );
}
