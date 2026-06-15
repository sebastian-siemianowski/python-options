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
export type SortColumn = 'asset' | 'sector' | 'signal' | 'momentum' | 'quality' | 'crash_risk' | 'pct30d' | 'politician' | `horizon_${number}`;
export type SortDir = 'asc' | 'desc';
export type SignalSortLevel = { col: SortColumn; dir: SortDir };
export const DEFAULT_SIGNAL_SORT: SignalSortLevel = { col: 'quality', dir: 'desc' };

const ISO_CURRENCY_CODES = new Set([
  'AUD', 'BRL', 'CAD', 'CHF', 'CNY', 'CZK', 'DKK', 'EUR', 'GBP', 'HKD',
  'HUF', 'INR', 'JPY', 'KRW', 'MXN', 'NOK', 'NZD', 'PLN', 'SEK', 'SGD',
  'TRY', 'USD', 'ZAR',
]);

const AI_STOCK_SECTORS = new Set([
  'AI Utility / Infrastructure',
  'AI Software / Data Platforms',
  'Semiconductor Equipment',
  'AI Power Semiconductors',
  'AI Hardware / Edge Compute',
  'Cloud & Cybersecurity',
  'Asian Tech & Manufacturing',
  'Industrial Infrastructure',
  'TechBio / AI Drug Discovery',
  'Quantum Computing',
]);

const AI_STOCK_TICKERS = new Set([
  'AAPL', 'ACN', 'ADBE', 'AMD', 'AMZN', 'ANET', 'ARM', 'AVGO', 'CRM', 'GOOGL',
  'GOOG', 'META', 'MSFT', 'NVDA', 'ORCL', 'PLTR', 'QCOM', 'TSM',
  'ASML', 'AMAT', 'LRCX', 'KLAC', 'TER', 'ONTO', 'MKSI', 'FORM', 'ACLS',
  'CAMT', 'NVMI', 'SNPS', 'CDNS', '8035.T', 'ASM.AS', 'BESI.AS', '6146.T',
  '6861.T',
  'MU', 'MRVL', 'MPWR', 'RMBS', 'MTSI', 'LSCC', 'SITM', 'CRDO', 'ALAB',
  'NVTS', 'WOLF', 'AEHR', 'IFX.DE', 'STM', 'AOSL', 'POWI', 'VSH',
  '005930.KS', '000660.KS', '2382.TW', '3711.TW', '6669.TW', '3443.TW',
  '3661.TW', '6723.T',
  'SMCI', 'DELL', 'PSTG', 'CIEN', 'COHR', 'CLS', 'JBL', 'FLEX', 'VRT',
  'ETN', 'PWR', 'TT', 'FIX', 'EME', '2308.TW',
  'CFLT', 'CRWD', 'DDOG', 'ESTC', 'MDB', 'NET', 'PATH', 'SNOW', 'ZS',
  'NBIS', 'IREN', 'CIFR', 'CRWV', 'GLXY',
  'RXRX', 'SDGR', 'ABCL', 'TEM', 'IONQ', 'QBTS', 'ARQQ', 'RGTI', 'QUBT',
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

export const defaultSortDirFor = (col: SortColumn): SortDir =>
  col === 'asset' || col === 'sector' || col === 'crash_risk' ? 'asc' : 'desc';

export const nextSortLevels = (
  previous: SignalSortLevel[],
  col: SortColumn,
  shiftKey: boolean,
): SignalSortLevel[] => {
  const idx = previous.findIndex((s) => s.col === col);
  if (idx >= 0) {
    const existing = previous[idx];
    const defaultDir = defaultSortDirFor(col);
    const oppositeDir: SortDir = defaultDir === 'asc' ? 'desc' : 'asc';
    if (existing.dir === oppositeDir) {
      const next = previous.filter((_, i) => i !== idx);
      return next.length > 0 ? next : [DEFAULT_SIGNAL_SORT];
    }
    return previous.map((s, i) => (i === idx ? { ...s, dir: oppositeDir } : s));
  }
  if (shiftKey && previous.length < 3) {
    return [...previous, { col, dir: defaultSortDirFor(col) }];
  }
  return [{ col, dir: defaultSortDirFor(col) }];
};

export const removeSortLevel = (previous: SignalSortLevel[], col: SortColumn): SignalSortLevel[] => {
  const next = previous.filter((s) => s.col !== col);
  return next.length > 0 ? next : [DEFAULT_SIGNAL_SORT];
};

export const horizonSignalFor = (row: SummaryRow, horizon: number) => {
  const sigs = row.horizon_signals || {};
  return sigs[horizon] || sigs[String(horizon)];
};

export const horizonExpReturn = (row: SummaryRow, horizon: number): number => {
  const value = horizonSignalFor(row, horizon)?.exp_ret;
  return typeof value === 'number' && Number.isFinite(value) ? value : 0;
};

const signalSortRank = (label: string | undefined | null): number => {
  const normalized = (label || 'HOLD').toUpperCase();
  const ranks: Record<string, number> = {
    'STRONG BUY': 5,
    BUY: 4,
    HOLD: 3,
    SELL: 2,
    'STRONG SELL': 1,
    EXIT: 0,
  };
  return ranks[normalized] ?? 3;
};

const sortValueForColumn = (
  row: SummaryRow,
  col: SortColumn,
  qualityScores: Record<string, number>,
): number | string => {
  if (col === 'asset') return row.asset_label || '';
  if (col === 'sector') return row.sector || '';
  if (col === 'signal') return signalSortRank(row.nearest_label);
  if (col === 'momentum') return row.momentum_score ?? 0;
  if (col === 'quality') return qualityScores[extractTicker(row.asset_label)] ?? 50;
  if (col === 'crash_risk') return row.crash_risk_score ?? 0;
  if (col === 'pct30d') return row.pct_30d ?? 0;
  if (col === 'politician') return Number((row as SummaryRow & { politician_activity_score?: number }).politician_activity_score ?? 0);
  if (col.startsWith('horizon_')) {
    const horizon = Number.parseInt(col.slice('horizon_'.length), 10);
    return Number.isFinite(horizon) ? horizonExpReturn(row, horizon) : 0;
  }
  return 0;
};

export const sortSummaryRows = (
  rows: SummaryRow[],
  sortLevels: SignalSortLevel[],
  qualityScores: Record<string, number>,
): SummaryRow[] => {
  if (sortLevels.length === 0 || rows.length < 2) return rows.slice();
  const decorated = rows.map((row, index) => ({
    row,
    index,
    values: sortLevels.map((level) => sortValueForColumn(row, level.col, qualityScores)),
  }));
  decorated.sort((a, b) => {
    for (let i = 0; i < sortLevels.length; i += 1) {
      const av = a.values[i];
      const bv = b.values[i];
      let cmp = 0;
      if (typeof av === 'string' && typeof bv === 'string') cmp = av.localeCompare(bv);
      else cmp = ((av as number) - (bv as number)) || 0;
      if (cmp !== 0) return sortLevels[i].dir === 'desc' ? -cmp : cmp;
    }
    return a.index - b.index;
  });
  return decorated.map((entry) => entry.row);
};

export const isFiatCurrencyTicker = (symbol: string | undefined | null): boolean => {
  let sym = String(symbol || '').trim().toUpperCase();
  if (sym.endsWith('_X')) sym = `${sym.slice(0, -2)}=X`;
  if (!sym.endsWith('=X')) return false;
  const pair = sym.slice(0, -2).replace(/[/_-]/g, '');
  if (pair.length !== 6) return false;
  return ISO_CURRENCY_CODES.has(pair.slice(0, 3)) && ISO_CURRENCY_CODES.has(pair.slice(3));
};

export const isCurrencyAsset = (assetLabelOrTicker: string | undefined | null): boolean =>
  isFiatCurrencyTicker(extractTicker(String(assetLabelOrTicker || '').trim()));

export const isAiStockAsset = (
  assetLabelOrTicker: string | undefined | null,
  sector?: string | null,
): boolean => {
  const ticker = extractTicker(String(assetLabelOrTicker || '').trim()).toUpperCase();
  if (ticker && AI_STOCK_TICKERS.has(ticker)) return true;
  return !!sector && AI_STOCK_SECTORS.has(sector);
};

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
