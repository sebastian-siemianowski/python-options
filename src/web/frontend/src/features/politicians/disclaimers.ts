export const POLITICIANS_DATA_USE_NOTICE =
  'Politician trade records are delayed, range-based public disclosures from official sources. They are research context only, not real-time trade confirmations, investment advice, copy-trade instructions, credit-rating inputs, solicitation material, or an insider-trading signal.';

export const POLITICIANS_DATA_USE_BULLETS = [
  'Public filings are delayed disclosures, not real-time trade confirmations or execution feeds.',
  'Amounts are reported in broad ranges and must be displayed as estimates or buckets, never as exact position sizes.',
  'The data must not be used for credit rating, unlawful purposes, solicitation, or any deployment prohibited by official source terms.',
  "The product must not label these records as insider-trading signals or invite users to copy a politician's trades.",
  'Research and backtests must use disclosure dates or filed dates as the knowable-time anchor to avoid lookahead leakage.',
  'Every displayed record must preserve source attribution, official document links, parser confidence, and delayed-data context.',
];

const VALID_COMPLIANCE_MODES = ['research_only', 'internal', 'public'] as const;

export type PoliticiansComplianceMode = typeof VALID_COMPLIANCE_MODES[number];

export function getFrontendPoliticiansComplianceMode(): PoliticiansComplianceMode {
  const env = import.meta.env as Record<string, string | undefined>;
  const requested = (env.POLITICIANS_COMPLIANCE_MODE || env.VITE_POLITICIANS_COMPLIANCE_MODE || 'research_only').toLowerCase();
  if ((VALID_COMPLIANCE_MODES as readonly string[]).includes(requested)) {
    return requested as PoliticiansComplianceMode;
  }
  return 'research_only';
}
