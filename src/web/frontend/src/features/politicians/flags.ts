export const POLITICIANS_FLAG_OPTIONS = [
  { value: 'late_disclosure', label: 'Late disclosure' },
  { value: 'large_trade_bucket', label: 'Large bucket' },
  { value: 'ticker_ambiguous', label: 'Ambiguous ticker' },
  { value: 'amended', label: 'Amended filing' },
] as const;

export type PoliticiansFlag = typeof POLITICIANS_FLAG_OPTIONS[number]['value'];
