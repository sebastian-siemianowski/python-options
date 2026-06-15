import { chromium } from 'playwright';
import { mkdir } from 'node:fs/promises';
import { join } from 'node:path';

const baseUrl = process.env.POLITICIANS_QA_URL || 'http://127.0.0.1:4173/politicians';
const outputDir = process.env.POLITICIANS_QA_OUT || '/tmp/python-options-politicians-qa';

const viewports = [
  { name: 'desktop', width: 1440, height: 1000 },
  { name: 'tablet', width: 900, height: 1100 },
  { name: 'mobile', width: 390, height: 900 },
];

const notice = {
  feature: 'politicians',
  status: 'notice_only',
  enabled: true,
  compliance_mode: 'research_only',
  requested_compliance_mode: 'research_only',
  compliance_mode_valid: true,
  valid_compliance_modes: ['research_only', 'internal', 'public'],
  generated_at: '2026-05-29T10:00:00Z',
  data_age_seconds: 60,
  data_use_notice: {
    title: 'Public Disclosure Notice',
    summary: 'Delayed public congressional disclosure records.',
    bullets: ['Research context only.'],
    official_sources: [],
    reviewed_at: '2026-05-29',
  },
};

const summary = {
  ...notice,
  status: 'ok',
  total_trades: 3,
  new_disclosures_7d: 2,
  new_disclosures_last_7_days: 2,
  new_tracked_asset_disclosures_7d: 1,
  new_watchlist_disclosures_7d: 1,
  tracked_asset_trades: 2,
  watchlist_trades: 1,
  late_filings: 1,
  newest_disclosure_date: '2026-05-28',
  source_health: {
    updated_at: '2026-05-29T10:00:00Z',
    sources: {
      house: { status: 'ok' },
      senate: { status: 'degraded' },
    },
  },
  by_chamber: {},
};
summary.summary = { ...summary };
delete summary.summary.summary;

const healthySummary = {
  ...summary,
  total_trades: 1,
  new_disclosures_7d: 1,
  new_disclosures_last_7_days: 1,
  new_tracked_asset_disclosures_7d: 1,
  new_watchlist_disclosures_7d: 1,
  tracked_asset_trades: 1,
  watchlist_trades: 1,
  late_filings: 0,
  source_health: {
    updated_at: '2026-05-29T10:00:00Z',
    sources: {
      house: { status: 'ok' },
      senate: { status: 'ok' },
    },
  },
};
healthySummary.summary = { ...healthySummary };
delete healthySummary.summary.summary;

const emptySummary = {
  ...healthySummary,
  total_trades: 0,
  new_disclosures_7d: 0,
  new_disclosures_last_7_days: 0,
  new_tracked_asset_disclosures_7d: 0,
  new_watchlist_disclosures_7d: 0,
  tracked_asset_trades: 0,
  watchlist_trades: 0,
  late_filings: 0,
  newest_disclosure_date: null,
};
emptySummary.summary = { ...emptySummary };
delete emptySummary.summary.summary;

const disabledNotice = {
  ...notice,
  status: 'disabled',
  enabled: false,
  disabled_reason: 'Compliance review has not enabled politician disclosure monitoring',
};

const sourceHealth = {
  ...notice,
  status: 'ok',
  overall_status: 'degraded',
  sources: {
    house: {
      status: 'ok',
      last_sync_time: '2026-05-29T10:00:00Z',
      newest_filing: '2026-05-28',
      parse_success_rate: 1,
      trade_count: 2,
      parse_error_count: 0,
      low_confidence_rows: 0,
      recent_errors: [],
      remediation: 'house source healthy; continue normal monitoring.',
    },
    senate: {
      status: 'degraded',
      last_sync_time: '2026-05-29T09:00:00Z',
      newest_filing: '2026-05-27',
      parse_success_rate: 0.86,
      trade_count: 1,
      parse_error_count: 1,
      low_confidence_rows: 1,
      recent_errors: [{ report_id: 'S1', errors: ['missing_required:ticker'] }],
      remediation: 'Review recent senate parser errors and low-confidence rows; source layout may have changed.',
    },
  },
  source_health: summary.source_health,
  confidence_buckets: { high: 2, warning: 1, quarantined: 0 },
  parse_error_count: 1,
};

const healthySourceHealth = {
  ...sourceHealth,
  overall_status: 'ok',
  sources: {
    house: { ...sourceHealth.sources.house, status: 'ok', recent_errors: [], low_confidence_rows: 0 },
    senate: { ...sourceHealth.sources.senate, status: 'ok', parse_success_rate: 1, parse_error_count: 0, low_confidence_rows: 0, recent_errors: [], remediation: 'senate source healthy; continue normal monitoring.' },
  },
  source_health: healthySummary.source_health,
  confidence_buckets: { high: 1, warning: 0, quarantined: 0 },
  parse_error_count: 0,
};

const trades = [
  {
    trade_id: '1',
    filer_id: 'jane-doe',
    filer_name: 'Jane Doe',
    chamber: 'house',
    party: 'D',
    state: 'CA',
    owner: 'self',
    ticker: 'NVDA',
    asset_name: 'NVIDIA Corporation',
    asset_type: 'stock',
    transaction_type: 'purchase',
    transaction_date: '2026-05-18',
    disclosure_date: '2026-05-28',
    delay_days: 10,
    amount_min_usd: 1001,
    amount_max_usd: 15000,
    amount_mid_usd: 8000,
    parser_confidence: 0.97,
    flags: [],
    is_tracked_asset: true,
    is_watchlist_asset: true,
    official_source_url: 'https://example.test/nvda.pdf',
  },
  {
    trade_id: '2',
    filer_id: 'john-smith',
    filer_name: 'John Smith',
    chamber: 'senate',
    party: 'R',
    state: 'TX',
    owner: 'spouse',
    ticker: 'AAPL',
    asset_name: 'Apple Inc.',
    asset_type: 'stock',
    transaction_type: 'sale',
    transaction_date: '2026-05-02',
    disclosure_date: '2026-05-27',
    delay_days: 25,
    amount_mid_usd: 32000,
    parser_confidence: 0.72,
    confidence_status: 'valid_with_warnings',
    flags: ['late_disclosure'],
    is_tracked_asset: true,
    is_watchlist_asset: false,
    official_source_url: 'https://example.test/aapl.pdf',
  },
];

const asset = {
  ...notice,
  status: 'ok',
  symbol: 'NVDA',
  window_days: 180,
  total: 1,
  total_symbol_trades: 1,
  recent_trades: trades.slice(0, 1),
  trades: trades.slice(0, 1),
  unique_filers: ['Jane Doe'],
  unique_filer_count: 1,
  buy_sell_imbalance: {
    buy_count: 1,
    sell_count: 0,
    net_count: 1,
    buy_amount_mid_usd: 8000,
    sell_amount_mid_usd: 0,
    net_amount_mid_usd: 8000,
  },
  amount_estimates: { amount_mid_usd: 8000, amount_min_usd: 1001, amount_max_usd: 15000 },
  activity: { politician_activity_score: 0.42, confidence: 0.8 },
  disclosure_timeline: [{ date: '2026-05-28', trade_count: 1, buy_count: 1, sell_count: 0, net_amount_mid_usd: 8000 }],
  known_limitations: [],
};

const filer = {
  ...notice,
  status: 'ok',
  filer_id: 'jane-doe',
  window_days: 180,
  metadata: {
    filer_id: 'jane-doe',
    filer_name: 'Jane Doe',
    chamber: 'house',
    party: 'D',
    state: 'CA',
    source: 'house',
    committee_enrichment: ['Financial Services'],
    committee_data_source: 'enrichment_not_source_disclosure',
    metadata_complete: true,
  },
  total: 1,
  total_filer_trades: 1,
  recent_trades: trades.slice(0, 1),
  top_tickers: [{ ticker: 'NVDA', trade_count: 1, amount_mid_usd: 8000 }],
  top_sectors: [{ sector: 'Technology', trade_count: 1, amount_mid_usd: 8000 }],
  delay_stats: { count: 1, average_days: 10, median_days: 10, max_days: 10, late_filing_count: 0 },
  ownership_breakdown: { self: 1, spouse: 0, dependent_child: 0, joint: 0, unknown: 0 },
  source_documents: [{ official_source_url: 'https://example.test/nvda.pdf', disclosure_date: '2026-05-28', report_id: 'R1' }],
};

const scenarios = [
  {
    name: 'degraded-low-confidence',
    viewports,
    notice,
    summary,
    sourceHealth,
    trades,
    waitFor: 'Disclosure Feed',
    required: ['Delayed Public Disclosures', 'Source health', 'Disclosure Feed', 'Source warnings visible'],
    legacyViewportNames: true,
  },
  {
    name: 'healthy',
    viewports: [viewports[0]],
    notice,
    summary: healthySummary,
    sourceHealth: healthySourceHealth,
    trades: trades.slice(0, 1),
    waitFor: 'Disclosure Feed',
    required: ['Delayed Public Disclosures', 'House: ok', 'Disclosure Feed'],
  },
  {
    name: 'empty',
    viewports: [viewports[0]],
    notice,
    summary: emptySummary,
    sourceHealth: healthySourceHealth,
    trades: [],
    waitFor: 'No Disclosures Matched Current Filters',
    required: ['No Disclosures Matched Current Filters', 'Disclosure Feed'],
  },
  {
    name: 'disabled',
    viewports: [viewports[0]],
    notice: disabledNotice,
    disabled: true,
    waitFor: 'Politician Monitoring Disabled',
    required: ['Politician Monitoring Disabled', 'Compliance review'],
  },
  {
    name: 'loading',
    viewports: [viewports[0]],
    loading: true,
    waitFor: 'Checking politician disclosure availability...',
    required: ['Checking politician disclosure availability...'],
  },
];

await mkdir(outputDir, { recursive: true });
const browser = await chromium.launch();
try {
  for (const scenario of scenarios) {
    for (const viewport of scenario.viewports) {
      const page = await browser.newPage({ viewport });
      await installRoutes(page, scenario);
      await page.goto(baseUrl, { waitUntil: 'domcontentloaded' });
      await page.getByText(scenario.waitFor).first().waitFor({ timeout: 10_000 });
      for (const label of scenario.required) {
        if (!(await page.getByText(label).first().isVisible())) {
          throw new Error(`${scenario.name}/${viewport.name}: missing ${label}`);
        }
      }
      if (!scenario.loading) {
        await assertNoBrokenLayout(page, `${scenario.name}/${viewport.name}`);
      }
      if (viewport.name === 'mobile' && !scenario.disabled && !scenario.loading) {
        await page.getByText('Disclosure Feed').scrollIntoViewIfNeeded();
      }
      const path = join(outputDir, `politicians-${scenario.name}-${viewport.name}.png`);
      await page.screenshot({ path, fullPage: true });
      if (scenario.legacyViewportNames) {
        await page.screenshot({ path: join(outputDir, `politicians-${viewport.name}.png`), fullPage: true });
      }
      await page.close();
    }
  }
  console.log(`politicians visual QA screenshots: ${outputDir}`);
} finally {
  await browser.close();
}

async function installRoutes(page, scenario) {
  if (scenario.loading) {
    await page.route('**/api/politicians/notice', async (route) => {
      await new Promise((resolve) => setTimeout(resolve, 1500));
      try {
        await route.fulfill({ json: notice });
      } catch {
        // Page may have closed after the loading screenshot.
      }
    });
    return;
  }
  await page.route('**/api/politicians/notice', (route) => route.fulfill({ json: scenario.notice }));
  if (scenario.disabled) return;
  await page.route('**/api/politicians/summary', (route) => route.fulfill({ json: scenario.summary }));
  await page.route('**/api/politicians/source-health', (route) => route.fulfill({ json: scenario.sourceHealth }));
  await page.route('**/api/politicians/trades**', (route) => route.fulfill({
    json: {
      ...notice,
      status: 'ok',
      total: scenario.trades.length,
      page: { limit: 500, offset: 0, returned: scenario.trades.length, total: scenario.trades.length, has_next: false },
      trades: scenario.trades,
    },
  }));
  await page.route('**/api/politicians/assets/**', (route) => route.fulfill({ json: asset }));
  await page.route('**/api/politicians/filers/**', (route) => route.fulfill({ json: filer }));
}

async function assertNoBrokenLayout(page, label) {
  const overflow = await page.evaluate(() => Math.max(0, document.documentElement.scrollWidth - window.innerWidth));
  if (overflow > 4) {
    throw new Error(`${label}: horizontal overflow ${overflow}px`);
  }
  const brokenRows = await page.locator('tbody tr').evaluateAll((rows) => (
    rows.filter((row) => {
      const rect = row.getBoundingClientRect();
      return rect.width > 0 && rect.height > 0 && rect.height < 24;
    }).length
  ));
  if (brokenRows > 0) {
    throw new Error(`${label}: ${brokenRows} broken table rows`);
  }
}
