import { useMemo, useState, type ReactNode } from 'react';
import { useQuery } from '@tanstack/react-query';
import LoadingSpinner from '../../../components/LoadingSpinner';
import { api, type PoliticiansSuccessfulTrader, type PoliticiansTradeRow } from '../../../api';
import type { PoliticianInsightFilter } from './PoliticianInsightBar';
import {
  ArrowDownUp,
  BadgeDollarSign,
  CalendarClock,
  ExternalLink,
  Landmark,
  Search,
  ShieldCheck,
  SlidersHorizontal,
  Trophy,
  UserRound,
  WalletCards,
  X,
} from 'lucide-react';

type SortKey = 'disclosure_date' | 'transaction_date' | 'amount_mid_usd' | 'delay_days' | 'filer_name' | 'ticker' | 'successful_trader_rank';
type SortDir = 'asc' | 'desc';
type QuickTransactionSide = '' | 'purchase' | 'sale';

interface Props {
  insightFilter: PoliticianInsightFilter;
  onSelectTicker?: (symbol: string) => void;
  onSelectFiler?: (filerId: string) => void;
}

const SORT_OPTIONS: Array<{ key: SortKey; label: string }> = [
  { key: 'disclosure_date', label: 'Disclosure' },
  { key: 'transaction_date', label: 'Transaction' },
  { key: 'amount_mid_usd', label: 'Amount' },
  { key: 'delay_days', label: 'Delay' },
  { key: 'filer_name', label: 'Politician' },
  { key: 'ticker', label: 'Ticker' },
  { key: 'successful_trader_rank', label: 'Rank' },
];

export default function TradeFeedTable({ insightFilter, onSelectTicker, onSelectFiler }: Props) {
  const [search, setSearch] = useState('');
  const [symbol, setSymbol] = useState('');
  const [chamber, setChamber] = useState('');
  const [party, setParty] = useState('');
  const [state, setState] = useState('');
  const [owner, setOwner] = useState('');
  const [transactionType, setTransactionType] = useState('');
  const [quickTransactionSide, setQuickTransactionSide] = useState<QuickTransactionSide>('');
  const [flag, setFlag] = useState('');
  const [trackedOnly, setTrackedOnly] = useState(false);
  const [watchlistOnly, setWatchlistOnly] = useState(false);
  const [topTradersOnly, setTopTradersOnly] = useState(false);
  const [stockLinkedOnly, setStockLinkedOnly] = useState(false);
  const [selectedTrader, setSelectedTrader] = useState<PoliticiansSuccessfulTrader | null>(null);
  const [sortKey, setSortKey] = useState<SortKey>('disclosure_date');
  const [sortDir, setSortDir] = useState<SortDir>('desc');
  const [mobileFiltersOpen, setMobileFiltersOpen] = useState(false);
  const [advancedFiltersOpen, setAdvancedFiltersOpen] = useState(false);

  const tradesQ = useQuery({
    queryKey: ['politiciansTrades', 'feed', { topTradersOnly, stockLinkedOnly, quickTransactionSide, selectedTraderKey: selectedTrader?.filer_key || null }],
    queryFn: () => api.politiciansTrades({
      limit: 500,
      top_traders_only: topTradersOnly,
      stock_linked_only: stockLinkedOnly,
      transaction_side: quickTransactionSide || undefined,
      filer: selectedTrader?.filer_name,
    }),
    staleTime: 60_000,
    retry: false,
  });

  const rows = useMemo(() => {
    const raw = tradesQ.data?.trades || [];
    const filtered = raw.filter((row) => {
      if (insightFilter === 'tracked' && !row.is_tracked_asset) return false;
      if (insightFilter === 'watchlist' && !row.is_watchlist_asset) return false;
      if (insightFilter === 'late' && !rowFlags(row).includes('late_disclosure')) return false;
      if (trackedOnly && !row.is_tracked_asset) return false;
      if (watchlistOnly && !row.is_watchlist_asset) return false;
      if (stockLinkedOnly && !isStockLinkedRow(row)) return false;
      if (symbol && text(row.ticker).toLowerCase() !== symbol.trim().toLowerCase()) return false;
      if (chamber && text(row.chamber).toLowerCase() !== chamber) return false;
      if (party && text(row.party).toLowerCase() !== party.toLowerCase()) return false;
      if (state && text(row.state).toLowerCase() !== state.toLowerCase()) return false;
      if (owner && text(row.owner).toLowerCase() !== owner) return false;
      if (transactionType && text(row.transaction_type).toLowerCase() !== transactionType) return false;
      if (quickTransactionSide && !matchesTransactionSide(row, quickTransactionSide)) return false;
      if (flag && !rowFlags(row).includes(flag)) return false;
      if (topTradersOnly && !row.successful_trader_rank) return false;
      if (search) {
        const haystack = [
          row.filer_name,
          row.ticker,
          row.asset_name_raw,
          row.asset_name,
          row.transaction_type,
          row.chamber,
          row.party,
          row.state,
        ].map(text).join(' ').toLowerCase();
        if (!haystack.includes(search.trim().toLowerCase())) return false;
      }
      return true;
    });
    return [...filtered].sort((a, b) => compareRows(a, b, sortKey, sortDir));
  }, [tradesQ.data, insightFilter, trackedOnly, watchlistOnly, topTradersOnly, stockLinkedOnly, symbol, chamber, party, state, owner, transactionType, quickTransactionSide, flag, search, sortKey, sortDir]);
  const advancedFilterCount = [party, state, owner, flag].filter(Boolean).length;
  const topTraders = tradesQ.data?.successful_traders?.leaderboard || [];
  const matchedCount = tradesQ.data?.total ?? rows.length;

  if (tradesQ.isLoading) return <LoadingSpinner text="Loading disclosure feed..." variant="table" />;

  return (
    <section className="glass-card overflow-hidden">
      <div className="border-b border-[var(--violet-8)] p-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
          <div className="min-w-0">
            <div className="flex items-center gap-2">
              <Landmark className="h-4 w-4" style={{ color: 'var(--accent-cyan)' }} />
              <h2 className="text-[15px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
                Disclosure Feed
              </h2>
            </div>
            <p className="mt-1 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
              {matchedCount.toLocaleString()} matched records
              {selectedTrader && <span> · {shortName(selectedTrader.filer_name)}</span>}
            </p>
          </div>
          <button
            type="button"
            onClick={() => setMobileFiltersOpen(true)}
            className="inline-flex h-9 items-center justify-center gap-2 rounded-[8px] border px-3 text-[12px] md:hidden"
            style={{ borderColor: 'var(--violet-8)', color: 'var(--accent-cyan)' }}
            title="Open disclosure filters"
          >
            <SlidersHorizontal className="h-3.5 w-3.5" /> Filters
          </button>
          <div className="hidden min-w-0 flex-1 md:block xl:max-w-[1040px]">
            <div className="grid grid-cols-2 gap-2 lg:grid-cols-[minmax(260px,1.7fr)_120px_140px_140px_auto]">
              <FilterInput icon value={search} onChange={setSearch} placeholder="Search" className="col-span-2 lg:col-span-1" />
              <FilterInput value={symbol} onChange={setSymbol} placeholder="Symbol" />
              <FilterSelect value={chamber} onChange={setChamber} options={['house', 'senate']} placeholder="Chamber" />
              <FilterSelect value={transactionType} onChange={handleTransactionTypeChange} options={['purchase', 'sale', 'sale_partial', 'exchange', 'received', 'other', 'unknown']} placeholder="Type" />
              <button
                type="button"
                onClick={() => setAdvancedFiltersOpen((value) => !value)}
                className="inline-flex h-9 items-center justify-center gap-2 rounded-[8px] border px-3 text-[12px] font-medium"
                style={{
                  borderColor: advancedFiltersOpen || advancedFilterCount > 0 ? 'rgba(56,217,245,0.45)' : 'var(--violet-8)',
                  color: advancedFiltersOpen || advancedFilterCount > 0 ? 'var(--accent-cyan)' : 'var(--text-secondary)',
                  background: advancedFiltersOpen ? 'rgba(56,217,245,0.08)' : 'rgba(255,255,255,0.02)',
                }}
              >
                <SlidersHorizontal className="h-3.5 w-3.5" />
                More{advancedFilterCount > 0 ? ` ${advancedFilterCount}` : ''}
              </button>
            </div>
            {advancedFiltersOpen && (
              <div className="mt-2 grid grid-cols-2 gap-2 lg:grid-cols-4">
                <FilterInput value={party} onChange={setParty} placeholder="Party" />
                <FilterInput value={state} onChange={setState} placeholder="State" />
                <FilterSelect value={owner} onChange={setOwner} options={['self', 'spouse', 'dependent_child', 'joint', 'unknown']} placeholder="Owner" />
                <FilterSelect value={flag} onChange={setFlag} options={['late_disclosure', 'large_trade_bucket', 'ticker_ambiguous', 'amended']} placeholder="Flag" />
              </div>
            )}
            <div className="mt-3 flex flex-wrap items-center gap-2">
              <span className="text-[10px] font-semibold uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>
                Sort
              </span>
              {SORT_OPTIONS.map((option) => (
                <SortChip
                  key={option.key}
                  label={option.label}
                  active={sortKey === option.key}
                  dir={sortDir}
                  onClick={() => handleSort(option.key)}
                />
              ))}
              <div className="ml-0 flex gap-2 xl:ml-auto">
                <Toggle checked={topTradersOnly} onChange={handleTopTradersToggle} label="Top 10" />
                <Toggle checked={stockLinkedOnly} onChange={setStockLinkedOnly} label="Stocks" />
                <Toggle checked={quickTransactionSide === 'purchase'} onChange={(checked) => handleQuickTransactionSide(checked ? 'purchase' : '')} label="Purchase" />
                <Toggle checked={quickTransactionSide === 'sale'} onChange={(checked) => handleQuickTransactionSide(checked ? 'sale' : '')} label="Sale" />
                <Toggle checked={trackedOnly} onChange={setTrackedOnly} label="Tracked" />
                <Toggle checked={watchlistOnly} onChange={setWatchlistOnly} label="Watch" />
              </div>
            </div>
            {topTraders.length > 0 && (
              <TopTraderStrip
                traders={topTraders}
                active={topTradersOnly}
                selectedKey={selectedTrader?.filer_key || null}
                onActivate={() => handleTopTradersToggle(true)}
                onSelectTrader={handleSelectTrader}
                onClearSelected={handleClearSelectedTrader}
              />
            )}
          </div>
        </div>
      </div>

      {mobileFiltersOpen && (
        <div className="fixed inset-0 z-50 bg-black/45 p-4 md:hidden" role="dialog" aria-modal="true" aria-label="Disclosure filters">
          <div className="ml-auto flex max-h-full w-full max-w-[360px] flex-col overflow-y-auto rounded-[8px] border border-[var(--violet-8)] p-4" style={{ background: 'var(--void-surface)' }}>
            <div className="mb-3 flex items-center justify-between gap-3">
              <h3 className="text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>Filters</h3>
              <button
                type="button"
                onClick={() => setMobileFiltersOpen(false)}
                className="flex h-8 w-8 items-center justify-center rounded-[8px] border"
                style={{ borderColor: 'var(--violet-8)', color: 'var(--text-secondary)' }}
                aria-label="Close filters"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <div className="grid grid-cols-1 gap-2">
              <FilterInput icon value={search} onChange={setSearch} placeholder="Search" />
              <FilterInput value={symbol} onChange={setSymbol} placeholder="Symbol" />
              <FilterSelect value={chamber} onChange={setChamber} options={['house', 'senate']} placeholder="Chamber" />
              <FilterInput value={party} onChange={setParty} placeholder="Party" />
              <FilterInput value={state} onChange={setState} placeholder="State" />
              <FilterSelect value={owner} onChange={setOwner} options={['self', 'spouse', 'dependent_child', 'joint', 'unknown']} placeholder="Owner" />
              <FilterSelect value={transactionType} onChange={handleTransactionTypeChange} options={['purchase', 'sale', 'sale_partial', 'exchange', 'received', 'other', 'unknown']} placeholder="Type" />
              <FilterSelect value={flag} onChange={setFlag} options={['late_disclosure', 'large_trade_bucket', 'ticker_ambiguous', 'amended']} placeholder="Flag" />
              <div className="flex gap-2">
                <Toggle checked={topTradersOnly} onChange={handleTopTradersToggle} label="Top 10" />
                <Toggle checked={stockLinkedOnly} onChange={setStockLinkedOnly} label="Stocks" />
                <Toggle checked={quickTransactionSide === 'purchase'} onChange={(checked) => handleQuickTransactionSide(checked ? 'purchase' : '')} label="Purchase" />
                <Toggle checked={quickTransactionSide === 'sale'} onChange={(checked) => handleQuickTransactionSide(checked ? 'sale' : '')} label="Sale" />
                <Toggle checked={trackedOnly} onChange={setTrackedOnly} label="Tracked" />
                <Toggle checked={watchlistOnly} onChange={setWatchlistOnly} label="Watch" />
              </div>
              {topTraders.length > 0 && (
                <TopTraderStrip
                  traders={topTraders}
                  active={topTradersOnly}
                  selectedKey={selectedTrader?.filer_key || null}
                  onActivate={() => handleTopTradersToggle(true)}
                  onSelectTrader={handleSelectTrader}
                  onClearSelected={handleClearSelectedTrader}
                  compact
                />
              )}
            </div>
          </div>
        </div>
      )}

      {rows.length === 0 ? (
        <div className="p-5">
          <h3 className="text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
            No Disclosures Matched Current Filters
          </h3>
          <p className="mt-2 text-[12px] leading-relaxed" style={{ color: 'var(--text-secondary)' }}>
            No delayed public disclosure records matched the current filter set.
          </p>
        </div>
      ) : (
        <div className="divide-y divide-[rgba(255,255,255,0.055)]">
          {rows.slice(0, 240).map((row, idx) => (
            <DisclosureRow
              key={text(row.trade_id) || `${text(row.report_id)}-${idx}`}
              row={row}
              onSelectTicker={onSelectTicker}
              onSelectFiler={onSelectFiler}
            />
          ))}
        </div>
      )}
    </section>
  );

  function handleSort(next: SortKey) {
    if (next === sortKey) {
      setSortDir((dir) => (dir === 'asc' ? 'desc' : 'asc'));
    } else {
      setSortKey(next);
      setSortDir(next === 'filer_name' || next === 'ticker' || next === 'successful_trader_rank' ? 'asc' : 'desc');
    }
  }

  function handleTopTradersToggle(next: boolean) {
    setTopTradersOnly(next);
    if (!next) {
      setSelectedTrader(null);
    }
    if (next) {
      sortByNewestDisclosure();
    }
  }

  function handleSelectTrader(trader: PoliticiansSuccessfulTrader) {
    setTopTradersOnly(true);
    sortByNewestDisclosure();
    setSelectedTrader((current) => (current?.filer_key === trader.filer_key ? null : trader));
  }

  function handleClearSelectedTrader() {
    setSelectedTrader(null);
    setTopTradersOnly(true);
    sortByNewestDisclosure();
  }

  function handleQuickTransactionSide(side: QuickTransactionSide) {
    setQuickTransactionSide(side);
    if (side) {
      setTransactionType('');
    }
  }

  function handleTransactionTypeChange(next: string) {
    setTransactionType(next);
    if (next) {
      setQuickTransactionSide('');
    }
  }

  function sortByNewestDisclosure() {
    setSortKey('disclosure_date');
    setSortDir('desc');
  }
}

function DisclosureRow({ row, onSelectTicker, onSelectFiler }: { row: PoliticiansTradeRow; onSelectTicker?: (symbol: string) => void; onSelectFiler?: (filerId: string) => void }) {
  const lowConfidence = isLowConfidence(row);
  const tone = transactionTone(row);
  return (
    <article
      className="grid gap-4 px-4 py-4 transition-colors hover:bg-[rgba(255,255,255,0.025)] lg:grid-cols-[minmax(190px,0.8fr)_minmax(0,2.1fr)_minmax(230px,0.9fr)]"
      style={{ background: lowConfidence ? 'rgba(245,158,11,0.045)' : undefined }}
    >
      <div className="min-w-0 space-y-2">
        <RowMeta icon={<CalendarClock className="h-3.5 w-3.5" />} label="Disclosure" value={text(row.disclosure_date) || 'unknown'} />
        <RowMeta icon={<ArrowDownUp className="h-3.5 w-3.5" />} label="Transaction" value={text(row.transaction_date) || 'unknown'} />
        <span
          className="inline-flex h-6 items-center rounded-[6px] border px-2 text-[11px] font-medium"
          style={{ borderColor: delayColor(row), color: delayColor(row), background: 'rgba(255,255,255,0.025)' }}
        >
          {value(row.delay_days)} day delay
        </span>
      </div>

      <div className="min-w-0">
        <div className="flex flex-wrap items-center gap-2">
          <TickerButton ticker={text(row.ticker)} onSelectTicker={onSelectTicker} />
          <span
            className="inline-flex h-7 items-center rounded-[6px] border px-2 text-[11px] font-semibold capitalize"
            style={{ borderColor: tone.border, color: tone.color, background: tone.bg }}
          >
            {transactionLabel(row)}
          </span>
          <span className="inline-flex h-7 items-center gap-1 rounded-[6px] border border-[var(--violet-8)] px-2 text-[11px] font-semibold" style={{ color: 'var(--text-luminous)', background: 'rgba(255,255,255,0.025)' }}>
            <BadgeDollarSign className="h-3.5 w-3.5" style={{ color: 'var(--accent-amber)' }} />
            {amount(row)}
          </span>
        </div>
        <div className="mt-2 text-[13px] leading-relaxed" style={{ color: 'var(--text-luminous)' }}>
          {assetLabel(row)}
        </div>
        <div className="mt-2 flex flex-wrap items-center gap-2 text-[12px]" style={{ color: 'var(--text-secondary)' }}>
          <UserRound className="h-3.5 w-3.5" />
          <FilerButton row={row} onSelectFiler={onSelectFiler} />
          {row.successful_trader_rank && (
            <span
              className="inline-flex h-6 items-center gap-1 rounded-[6px] border px-2 text-[10px] font-semibold"
              style={{ borderColor: 'rgba(245,197,66,0.30)', color: 'var(--accent-amber)', background: 'var(--amber-12)' }}
            >
              <Trophy className="h-3 w-3" />
              Top {text(row.successful_trader_rank)}
            </span>
          )}
          <span>{[row.party, row.state].map(text).filter(Boolean).join('/') || text(row.chamber) || 'unknown chamber'}</span>
        </div>
      </div>

      <div className="min-w-0 space-y-2 lg:text-right">
        <div className="flex flex-wrap gap-2 lg:justify-end">
          <InfoPill icon={<WalletCards className="h-3.5 w-3.5" />} value={text(row.owner) || 'unknown owner'} />
          <InfoPill icon={<ShieldCheck className="h-3.5 w-3.5" />} value={confidence(row)} tone={lowConfidence ? 'warn' : 'ok'} />
        </div>
        <div className="flex flex-wrap items-center gap-2 lg:justify-end">
          <span className="text-[11px] capitalize" style={{ color: 'var(--text-secondary)' }}>
            {text(row.chamber) || 'unknown'}
          </span>
          {sourceLink(row)}
        </div>
        {rowFlags(row).length > 0 && (
          <div className="flex flex-wrap gap-1 lg:justify-end">
            {rowFlags(row).map((flag) => (
              <span key={flag} className="rounded-[6px] border border-[rgba(245,197,66,0.25)] px-2 py-1 text-[10px]" style={{ color: 'var(--accent-amber)', background: 'var(--amber-12)' }}>
                {flag.replace(/_/g, ' ')}
              </span>
            ))}
          </div>
        )}
      </div>
    </article>
  );
}

function TopTraderStrip({
  traders,
  active,
  selectedKey,
  onActivate,
  onSelectTrader,
  onClearSelected,
  compact = false,
}: {
  traders: PoliticiansSuccessfulTrader[];
  active: boolean;
  selectedKey: string | null;
  onActivate: () => void;
  onSelectTrader: (trader: PoliticiansSuccessfulTrader) => void;
  onClearSelected: () => void;
  compact?: boolean;
}) {
  const selectedTrader = traders.find((trader) => trader.filer_key === selectedKey) || null;
  const displayedTraders = traders.slice(0, 10);
  return (
    <div
      className="mt-3 rounded-[8px] border p-3"
      style={{
        borderColor: active ? 'rgba(245,197,66,0.38)' : 'var(--violet-8)',
        background: active
          ? 'linear-gradient(135deg, rgba(245,197,66,0.075), rgba(56,217,245,0.035))'
          : 'rgba(255,255,255,0.02)',
      }}
    >
      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <button
            type="button"
            onClick={onActivate}
            className="flex min-w-0 items-center gap-3 text-left"
            aria-label="Show top congressional trader cohort"
          >
            <span
              className="flex h-10 w-10 shrink-0 items-center justify-center rounded-[8px] border"
              style={{ borderColor: 'rgba(245,197,66,0.24)', color: 'var(--accent-amber)', background: 'rgba(245,197,66,0.08)' }}
            >
              <Trophy className="h-4 w-4" />
            </span>
            <span className="min-w-0">
              <span className="block text-[13px] font-semibold" style={{ color: 'var(--text-luminous)' }}>
                {selectedTrader ? shortName(selectedTrader.filer_name) : 'Top trader filter'}
              </span>
              <span className="block text-[11px] leading-snug" style={{ color: 'var(--text-secondary)' }}>
                {selectedTrader
                  ? `Rank #${selectedTrader.rank} · ${formatScore(selectedTrader.success_score)} score · ${selectedTrader.scored_trades} scored trades`
                  : 'Top 10 congressional traders · 90D post-disclosure signed return'}
              </span>
            </span>
          </button>
          <button
            type="button"
            onClick={selectedTrader ? onClearSelected : onActivate}
            className="inline-flex h-8 shrink-0 items-center justify-center rounded-[8px] border px-3 text-[11px] font-medium transition hover:brightness-110"
            style={{
              borderColor: selectedTrader ? 'rgba(56,217,245,0.40)' : 'rgba(245,197,66,0.26)',
              color: selectedTrader ? 'var(--accent-cyan)' : 'var(--accent-amber)',
              background: selectedTrader ? 'rgba(56,217,245,0.08)' : 'rgba(245,197,66,0.07)',
            }}
          >
            All top traders
          </button>
        </div>

        <div className={`scrollbar-thin flex gap-2 overflow-x-auto pb-1 ${compact ? 'pr-1' : ''}`} aria-label="Top congressional trader filters">
          {displayedTraders.map((trader) => {
            const selected = selectedKey === trader.filer_key;
            return (
              <button
                type="button"
                key={text(trader.filer_key)}
                onClick={() => onSelectTrader(trader)}
                aria-pressed={selected}
                aria-label={`Filter disclosures to ${shortName(trader.filer_name)}`}
                className="min-h-[86px] w-[188px] shrink-0 rounded-[8px] border p-3 text-left transition hover:brightness-110"
                style={{
                  borderColor: selected ? 'rgba(245,197,66,0.62)' : 'rgba(255,255,255,0.10)',
                  color: selected ? 'var(--text-luminous)' : 'var(--text-secondary)',
                  background: selected ? 'rgba(245,197,66,0.11)' : 'rgba(255,255,255,0.028)',
                  boxShadow: selected ? '0 0 0 1px rgba(245,197,66,0.18) inset' : 'none',
                }}
              >
                <span className="flex items-center justify-between gap-2">
                  <span
                    className="rounded-[6px] border px-1.5 py-0.5 text-[10px] font-semibold"
                    style={{ borderColor: 'rgba(245,197,66,0.26)', color: 'var(--accent-amber)', background: 'rgba(245,197,66,0.07)' }}
                  >
                    #{trader.rank}
                  </span>
                  <span className="text-[10px] tabular-nums" style={{ color: scoreColor(trader.success_score) }}>
                    {formatScore(trader.success_score)}
                  </span>
                </span>
                <span className="mt-2 block min-h-[30px] whitespace-normal text-[12px] font-semibold leading-snug" style={{ color: selected ? 'var(--text-luminous)' : 'var(--text-secondary)' }}>
                  {shortName(trader.filer_name)}
                </span>
                <span className="mt-2 flex items-center justify-between gap-2 text-[10px]" style={{ color: 'var(--text-muted)' }}>
                  <span>{trader.scored_trades} scored</span>
                  <span>{Math.round(trader.win_rate * 100)}% win</span>
                </span>
              </button>
            );
          })}
        </div>
        {selectedTrader && (
          <div className="text-[11px]" style={{ color: 'var(--text-secondary)' }}>
            Showing disclosures for <span style={{ color: 'var(--accent-amber)' }}>{shortName(selectedTrader.filer_name)}</span>.
          </div>
        )}
      </div>
    </div>
  );
}

function FilerButton({ row, onSelectFiler }: { row: PoliticiansTradeRow; onSelectFiler?: (filerId: string) => void }) {
  const label = text(row.filer_name) || 'Unknown';
  const filerId = text(row.filer_id) || label;
  if (!onSelectFiler || label === 'Unknown') return <span>{label}</span>;
  return (
    <button
      type="button"
      onClick={() => onSelectFiler(filerId)}
      className="max-w-full text-left font-medium hover:underline"
      style={{ color: 'var(--accent-cyan)' }}
    >
      {label}
    </button>
  );
}

function TickerButton({ ticker, onSelectTicker }: { ticker: string; onSelectTicker?: (symbol: string) => void }) {
  if (!ticker) return <span style={{ color: 'var(--text-muted)' }}>Unmapped</span>;
  return (
    <button
      type="button"
      onClick={() => onSelectTicker?.(ticker)}
      className="inline-flex h-8 items-center rounded-[8px] border px-3 text-[14px] font-semibold tracking-normal transition hover:brightness-110"
      style={{
        borderColor: 'rgba(56,217,245,0.40)',
        color: 'var(--text-luminous)',
        background: 'linear-gradient(135deg, rgba(56,217,245,0.16), rgba(139,92,246,0.10))',
      }}
    >
      {ticker}
    </button>
  );
}

function SortChip({ label, active, dir, onClick }: { label: string; active: boolean; dir: SortDir; onClick: () => void }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="inline-flex h-7 items-center gap-1 rounded-[7px] border px-2 text-[11px] font-medium transition hover:brightness-110"
      style={{
        borderColor: active ? 'rgba(56,217,245,0.45)' : 'var(--violet-8)',
        color: active ? 'var(--accent-cyan)' : 'var(--text-secondary)',
        background: active ? 'rgba(56,217,245,0.08)' : 'rgba(255,255,255,0.02)',
      }}
    >
      {label}
      {active && <span className="text-[10px] uppercase">{dir}</span>}
    </button>
  );
}

function RowMeta({ icon, label, value: displayValue }: { icon: ReactNode; label: string; value: string }) {
  return (
    <div className="flex items-center gap-2 text-[11px]" style={{ color: 'var(--text-secondary)' }}>
      <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-[6px]" style={{ color: 'var(--accent-cyan)', background: 'rgba(56,217,245,0.08)' }}>
        {icon}
      </span>
      <div className="min-w-0">
        <div className="text-[9px] uppercase" style={{ color: 'var(--text-muted)', letterSpacing: '0.08em' }}>{label}</div>
        <div className="truncate font-medium" style={{ color: 'var(--text-luminous)' }}>{displayValue}</div>
      </div>
    </div>
  );
}

function InfoPill({ icon, value: displayValue, tone = 'neutral' }: { icon: ReactNode; value: string; tone?: 'neutral' | 'ok' | 'warn' }) {
  const color = tone === 'ok' ? 'var(--accent-emerald)' : tone === 'warn' ? 'var(--accent-amber)' : 'var(--text-secondary)';
  return (
    <span className="inline-flex h-7 items-center gap-1 rounded-[6px] border border-[var(--violet-8)] px-2 text-[11px]" style={{ color, background: 'rgba(255,255,255,0.025)' }}>
      {icon}
      {displayValue}
    </span>
  );
}

function FilterInput({ value, onChange, placeholder, icon = false, className = '' }: { value: string; onChange: (value: string) => void; placeholder: string; icon?: boolean; className?: string }) {
  return (
    <label className={`relative min-w-0 ${className}`}>
      {icon && <Search className="pointer-events-none absolute left-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2" style={{ color: 'var(--text-muted)' }} />}
      <input
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
        className={`h-9 w-full rounded-[8px] border bg-transparent text-[12px] outline-none focus:border-[var(--accent-cyan)] ${icon ? 'pl-7' : 'pl-3'} pr-3`}
        style={{ borderColor: 'var(--violet-8)', color: 'var(--text-luminous)' }}
      />
    </label>
  );
}

function FilterSelect({ value, onChange, options, placeholder }: { value: string; onChange: (value: string) => void; options: string[]; placeholder: string }) {
  return (
    <select
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className="h-9 min-w-0 rounded-[8px] border bg-[var(--void-surface)] px-3 text-[12px] outline-none focus:border-[var(--accent-cyan)]"
      style={{ borderColor: 'var(--violet-8)', color: 'var(--text-luminous)' }}
    >
      <option value="">{placeholder}</option>
      {options.map((option) => <option key={option} value={option}>{option}</option>)}
    </select>
  );
}

function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (value: boolean) => void; label: string }) {
  return (
    <label className="flex h-8 min-w-[78px] items-center justify-center gap-1 rounded-[8px] border px-2 text-[11px]" style={{ borderColor: checked ? 'var(--accent-cyan)' : 'var(--violet-8)', color: checked ? 'var(--accent-cyan)' : 'var(--text-secondary)' }}>
      <input className="sr-only" type="checkbox" checked={checked} onChange={(event) => onChange(event.target.checked)} />
      <span className="h-2 w-2 rounded-full" style={{ background: checked ? 'var(--accent-cyan)' : 'var(--text-muted)' }} />
      {label}
    </label>
  );
}

function compareRows(a: PoliticiansTradeRow, b: PoliticiansTradeRow, key: SortKey, dir: SortDir): number {
  const multiplier = dir === 'asc' ? 1 : -1;
  if (key === 'successful_trader_rank') {
    const left = numberValue(a[key]) || 9999;
    const right = numberValue(b[key]) || 9999;
    return (left - right) * multiplier;
  }
  if (key === 'amount_mid_usd' || key === 'delay_days') {
    return (numberValue(a[key]) - numberValue(b[key])) * multiplier;
  }
  return text(a[key]).localeCompare(text(b[key])) * multiplier;
}

function sourceLink(row: PoliticiansTradeRow) {
  const url = text(row.official_source_url);
  if (!url) return <span style={{ color: 'var(--text-muted)' }}>No source</span>;
  return (
    <a href={url} target="_blank" rel="noopener noreferrer" className="inline-flex h-7 items-center gap-1 rounded-[6px] border border-[rgba(56,217,245,0.30)] px-2 text-[11px] font-medium hover:brightness-110" style={{ color: 'var(--accent-cyan)', background: 'rgba(56,217,245,0.06)' }}>
      Source <ExternalLink className="h-3 w-3" />
    </a>
  );
}

function amount(row: PoliticiansTradeRow): string {
  const min = numberValue(row.amount_min_usd);
  const max = numberValue(row.amount_max_usd);
  const mid = numberValue(row.amount_mid_usd);
  if (min && max) return `$${compact(min)}-$${compact(max)}`;
  if (mid) return `$${compact(mid)}`;
  return 'unknown';
}

function confidence(row: PoliticiansTradeRow): string {
  const score = numberValue(row.parser_confidence);
  if (!score) return text(row.confidence_status) || 'unknown confidence';
  return `${Math.round(score * 100)}%`;
}

function isLowConfidence(row: PoliticiansTradeRow): boolean {
  const score = numberValue(row.parser_confidence);
  return Boolean(score && score < 0.8) || text(row.confidence_status).includes('quarantine');
}

function isStockLinkedRow(row: PoliticiansTradeRow): boolean {
  const ticker = text(row.ticker).trim();
  const assetType = text(row.asset_type).toLowerCase();
  return Boolean(ticker) && ['stock', 'etf', 'option'].includes(assetType);
}

function matchesTransactionSide(row: PoliticiansTradeRow, side: QuickTransactionSide): boolean {
  const transaction = text(row.transaction_type).toLowerCase();
  if (side === 'purchase') return transaction === 'purchase' || transaction === 'received';
  if (side === 'sale') return transaction === 'sale' || transaction === 'sale_partial';
  return true;
}

function rowFlags(row: PoliticiansTradeRow): string[] {
  return Array.isArray(row.flags) ? row.flags.map(String) : [];
}

function assetLabel(row: PoliticiansTradeRow): string {
  return text(row.asset_name_raw) || text(row.asset_name) || text(row.asset_type) || 'Unknown asset';
}

function transactionLabel(row: PoliticiansTradeRow): string {
  return text(row.transaction_type).replace(/_/g, ' ') || 'unknown';
}

function transactionTone(row: PoliticiansTradeRow): { color: string; border: string; bg: string } {
  const type = text(row.transaction_type).toLowerCase();
  if (type === 'purchase' || type === 'received') {
    return { color: 'var(--accent-emerald)', border: 'rgba(62,232,165,0.28)', bg: 'var(--emerald-12)' };
  }
  if (type === 'sale' || type === 'sale_partial') {
    return { color: 'var(--accent-rose)', border: 'rgba(255,107,138,0.28)', bg: 'var(--rose-12)' };
  }
  return { color: 'var(--accent-amber)', border: 'rgba(245,197,66,0.28)', bg: 'var(--amber-12)' };
}

function delayColor(row: PoliticiansTradeRow): string {
  const delay = numberValue(row.delay_days);
  if (delay >= 45) return 'var(--accent-rose)';
  if (delay >= 30) return 'var(--accent-amber)';
  return 'var(--text-secondary)';
}

function text(value: unknown): string {
  return value == null ? '' : String(value);
}

function value(value: unknown): string {
  const numeric = numberValue(value);
  return numeric || numeric === 0 ? String(numeric) : 'unknown';
}

function numberValue(value: unknown): number {
  if (typeof value === 'number') return value;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : 0;
}

function compact(value: number): string {
  return Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 1 }).format(value);
}

function shortName(value: string): string {
  return value.replace(/^Hon\.\s+/i, '').replace(/^The Honorable\s+/i, '');
}

function formatScore(value: number): string {
  const prefix = value > 0 ? '+' : '';
  return `${prefix}${value.toFixed(2)}`;
}

function scoreColor(value: number): string {
  if (value > 0) return 'var(--accent-emerald)';
  if (value < 0) return 'var(--accent-rose)';
  return 'var(--text-muted)';
}
