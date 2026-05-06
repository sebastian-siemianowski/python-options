import type { ReversalFlipEntry, ReversalFlipsData, SectorGroup, SummaryRow } from '../../api';

export type HorizonTone = 'greens' | 'reds' | 'mixed';
export type SignalFilter =
  | 'all'
  | 'bullish'
  | 'bearish'
  | 'greens'
  | 'reds'
  | 'strong_buy'
  | 'buy'
  | 'hold'
  | 'sell'
  | 'strong_sell'
  | 'reversal_buy'
  | 'reversal_sell';
export type ReversalQuickFilter = 'reversal_buy' | 'reversal_sell';
export type SortColumn = 'asset' | 'sector' | 'signal' | 'momentum' | 'quality' | 'crash_risk' | `horizon_${number}`;
export type SortDir = 'asc' | 'desc';

const ISO_CURRENCY_CODES = new Set([
  'AUD', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP', 'HKD',
  'HUF', 'INR', 'JPY', 'KRW', 'MXN', 'NOK', 'NZD', 'PLN', 'SEK', 'SGD',
  'TRY', 'USD', 'ZAR',
]);

export const extractTicker = (label: string): string => {
  if (label.includes('(')) return label.split('(').pop()!.replace(')', '').trim();
  return label;
};

export const rowHorizonColor = (row: SummaryRow): HorizonTone => {
  const sigs = Object.values(row.horizon_signals || {});
  if (!sigs.length) return 'mixed';
  let pos = 0;
  let neg = 0;
  for (const s of sigs) {
    const r = s?.exp_ret;
    if (typeof r !== 'number' || Number.isNaN(r)) continue;
    if (r > 0) pos++;
    else if (r < 0) neg++;
  }
  const total = pos + neg;
  if (total === 0) return 'mixed';
  if (pos / total > 0.5) return 'greens';
  if (neg / total > 0.5) return 'reds';
  return 'mixed';
};

export const isFiatCurrencyTicker = (symbol: string | undefined | null): boolean => {
  let sym = String(symbol || '').trim().toUpperCase();
  if (sym.endsWith('_X')) sym = `${sym.slice(0, -2)}=X`;
  if (!sym.endsWith('=X')) return false;
  const pair = sym.slice(0, -2).replace(/[\/_-]/g, '');
  if (pair.length !== 6) return false;
  return ISO_CURRENCY_CODES.has(pair.slice(0, 3)) && ISO_CURRENCY_CODES.has(pair.slice(3));
};

export const isCurrencyAsset = (assetLabelOrTicker: string | undefined | null): boolean =>
  isFiatCurrencyTicker(extractTicker(String(assetLabelOrTicker || '').trim()));

export const rebuildSectorFromAssets = (sector: SectorGroup, assets: SummaryRow[]): SectorGroup => {
  const counts = { strong_buy: 0, buy: 0, hold: 0, sell: 0, strong_sell: 0, exit: 0 };
  for (const asset of assets) {
    const label = (asset.nearest_label || 'HOLD').toUpperCase().replace(/\s+/g, '_');
    if (label === 'STRONG_BUY') counts.strong_buy += 1;
    else if (label === 'BUY') counts.buy += 1;
    else if (label === 'SELL') counts.sell += 1;
    else if (label === 'STRONG_SELL') counts.strong_sell += 1;
    else if (label === 'EXIT') counts.exit += 1;
    else counts.hold += 1;
  }
  const avg = (field: 'momentum_score' | 'crash_risk_score') => (
    assets.length > 0 ? assets.reduce((sum, asset) => sum + (asset[field] ?? 0), 0) / assets.length : 0
  );
  return {
    ...sector,
    assets,
    asset_count: assets.length,
    strong_buy: counts.strong_buy,
    buy: counts.buy,
    hold: counts.hold,
    sell: counts.sell,
    strong_sell: counts.strong_sell,
    exit: counts.exit,
    avg_momentum: avg('momentum_score'),
    avg_crash_risk: avg('crash_risk_score'),
  };
};

export const isReversalQuickFilter = (value: SignalFilter | string): value is ReversalQuickFilter =>
  value === 'reversal_buy' || value === 'reversal_sell';

export const reversalFlipForAsset = (
  flips: ReversalFlipsData | undefined,
  assetLabel: string | undefined | null,
): ReversalFlipEntry | undefined => {
  if (!flips || !assetLabel) return undefined;
  const raw = extractTicker(assetLabel).trim();
  if (!raw) return undefined;
  const variants = [
    raw,
    raw.toUpperCase(),
    raw.replace(/=/g, '_'),
    raw.replace(/_/g, '='),
    raw.replace(/-/g, '_'),
    raw.replace(/_/g, '-'),
  ];
  for (const key of variants) {
    const entry = flips.signals[key];
    if (entry) return entry;
  }
  return undefined;
};

export const rowMatchesReversalFilter = (
  row: SummaryRow,
  filter: ReversalQuickFilter,
  flips: ReversalFlipsData | undefined,
): boolean => {
  const entry = reversalFlipForAsset(flips, row.asset_label);
  return entry?.signal === (filter === 'reversal_buy' ? 'buy' : 'sell');
};
