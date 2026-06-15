# Politicians.md -- Congressional Trade Monitoring Product Requirements

**Author**: Product Owner for Quant Signal Engine
**Date**: May 2026
**Scope**: Automated ingestion, parsing, storage, enrichment, and web display of politician securities disclosures
**MVP Jurisdiction**: United States Congress, using official House and Senate public disclosure sources
**Philosophy**: Politician trades are delayed, range-based public disclosures. They are context, not an oracle.

---

## Product Thesis

The signal engine already treats market behavior as probabilistic: every model is a hypothesis, uncertainty is explicit, and no single signal source should dominate. Politician trade monitoring must follow the same standard.

Congressional disclosures can help answer:

- Which assets in our universe have recent public political trading activity?
- Which politicians, chambers, committees, and parties are associated with that activity?
- Was the disclosure timely or late?
- Does activity cluster around sectors, committees, macro events, or high-conviction signals?
- Is this useful as research context without leaking future information into models?

This feature must never present disclosed trades as real-time insider activity. Public filings are lagged, reported in amount buckets, sometimes amended, and often messy. The system should surface them with source links, confidence levels, parse quality, and disclosure-delay context.

---

## Problem Statement

Users currently see quantitative signals, watchlists, market risk, diagnostics, and model competition results, but they do not see public political trading disclosures that may create contextual interest around individual assets or sectors.

The project needs a new Politicians web section that:

1. Automatically retrieves public congressional financial disclosure filings.
2. Parses Periodic Transaction Reports and related amendments.
3. Normalizes raw filings into a stable internal trade schema.
4. Stores immutable raw source artifacts and deduplicated normalized trades.
5. Links trades to tickers already monitored by the signal engine.
6. Exposes backend APIs for summary, feed, member, asset, filing, and source-health views.
7. Shows the data in a new frontend section designed for scanning, filtering, source verification, and watchlist relevance.

---

## Non-Negotiable Constraints

- **Official-source first**: House Clerk and Senate eFD public disclosure portals are the source of truth for MVP. Third-party datasets may be used only for validation if legally allowed and clearly marked as non-authoritative.
- **No lookahead leakage**: Quant research and signal overlays must use `disclosure_date` or `filed_date`, never `transaction_date`, when simulating what was knowable at the time.
- **No copy-trade language**: UI must not say "follow this trade", "copy this politician", "guaranteed edge", or equivalent.
- **No silent parse failures**: Every failed row, ambiguous ticker, unsupported asset type, and low-confidence parse must be recorded.
- **Raw source preservation**: Every parsed trade must link back to an immutable raw source artifact and official document URL.
- **Delayed-data disclosure**: UI must show filing delay and explain that disclosures are not real-time.
- **Compliance gate**: Before production use, confirm allowed use of official reports for the intended deployment context. Official pages warn against unlawful, credit-rating, solicitation, and certain commercial uses.

---

## Source Register

| Source | Scope | MVP Use | Notes |
|--------|-------|---------|-------|
| House Clerk Financial Disclosure Reports | House Member and candidate reports, including downloadable yearly disclosure archives and PTR PDFs | Primary House source | Official yearly downloads and search pages should seed the manifest. |
| House Clerk Search | Member and candidate search | Manual verification, source fallback | Must preserve official document URLs. |
| Senate eFD Search | Senate Member and candidate financial disclosures | Primary Senate source | No guaranteed bulk export; implement behind a source adapter with source-health monitoring. |
| Senate Select Committee on Ethics | Rules, instructions, reporting requirements | Compliance and documentation | Confirms public access and PTR obligations. |
| Senate Public Disclosure page | Public disclosure and PTR instructions | Compliance and source verification | Useful for current reporting threshold and disclosure rules. |
| House/Senate member metadata | Names, chamber, state, district, party, committee mappings | Enrichment | Use official Clerk, Senate, Congress.gov, or Biographical Directory sources where possible. |

**Source URLs verified on 2026-05-28**:

- House Financial Disclosure Reports: https://disclosures-clerk.house.gov/FinancialDisclosure/ViewReport
- House Financial Disclosure Search: https://disclosures-clerk.house.gov/FinancialDisclosure/ViewSearch
- Senate Financial Disclosure overview: https://www.ethics.senate.gov/public/index.cfm/financialdisclosure
- Senate eFD public search: https://efdsearch.senate.gov/search/home/
- Senate Public Disclosure: https://www.senate.gov/legislative/lobbyingdisc.htm

---

## Target Architecture

```
OFFICIAL SOURCES
  House Clerk yearly archives + PTR PDFs
  Senate eFD search + PTR pages
        |
        v
INGESTION
  src/ingestion/politicians/
    sources/house.py
    sources/senate.py
    source_health.py
    download.py
    cli.py
        |
        v
RAW ARCHIVE
  src/data/politicians/raw/{source}/{year}/...
  src/data/politicians/manifests/{source}_{year}.json
        |
        v
PARSING + NORMALIZATION
  parsers/house_pdf.py
  parsers/senate_html.py
  parsers/senate_pdf.py
  normalize.py
  ticker_resolution.py
  validation.py
        |
        v
NORMALIZED STORAGE
  src/data/politicians/trades.jsonl
  src/data/politicians/filings.jsonl
  src/data/politicians/members.json
  src/data/politicians/issuer_aliases.json
  src/data/politicians/parse_errors.jsonl
  src/data/politicians/sync_state.json
        |
        v
BACKEND API
  src/web/backend/routers/politicians.py
  src/web/backend/services/politicians_service.py
        |
        v
WEB SECTION
  src/web/frontend/src/pages/PoliticiansPage.tsx
  src/web/frontend/src/features/politicians/
```

---

## Canonical Trade Schema

All normalized records must conform to this shape before storage:

```json
{
  "trade_id": "house:2026:20031020:row:004",
  "source": "house",
  "chamber": "house",
  "report_id": "20031020",
  "report_type": "ptr",
  "is_amendment": false,
  "amends_report_id": null,
  "filer_id": "bioguide-or-source-id",
  "filer_name": "Jane Doe",
  "party": "D",
  "state": "CA",
  "district": "11",
  "office": "Representative",
  "committees": ["Financial Services"],
  "owner": "spouse",
  "asset_name_raw": "NVIDIA Corporation Common Stock",
  "asset_name_normalized": "NVIDIA Corp",
  "ticker": "NVDA",
  "asset_type": "stock",
  "transaction_type": "purchase",
  "amount_bucket_raw": "$15,001 - $50,000",
  "amount_min_usd": 15001,
  "amount_max_usd": 50000,
  "amount_mid_usd": 32500.5,
  "transaction_date": "2026-04-12",
  "notification_date": "2026-04-18",
  "filed_date": "2026-04-28",
  "disclosure_date": "2026-04-28",
  "delay_days": 16,
  "document_url": "https://...",
  "raw_artifact_path": "src/data/politicians/raw/house/2026/20031020.pdf",
  "source_hash": "sha256:...",
  "row_hash": "sha256:...",
  "parser_version": "politicians-parser-v1",
  "parser_confidence": 0.97,
  "validation_status": "valid",
  "warnings": [],
  "created_at": "2026-05-28T12:00:00Z",
  "updated_at": "2026-05-28T12:00:00Z"
}
```

### Required Enumerations

| Field | Values |
|-------|--------|
| `source` | `house`, `senate`, `manual_fixture`, `third_party_validation` |
| `chamber` | `house`, `senate`, `candidate`, `unknown` |
| `report_type` | `ptr`, `annual`, `new_filer`, `termination`, `candidate`, `amendment`, `unknown` |
| `owner` | `self`, `spouse`, `dependent_child`, `joint`, `unknown` |
| `transaction_type` | `purchase`, `sale`, `sale_partial`, `exchange`, `received`, `other`, `unknown` |
| `asset_type` | `stock`, `etf`, `option`, `bond`, `fund`, `crypto`, `commodity`, `private_asset`, `unknown` |
| `validation_status` | `valid`, `valid_with_warnings`, `quarantined`, `parse_failed` |

---

## Storage Contract

The project mostly uses CSV and JSON artifacts under `src/data`. The politician module should follow that pattern before introducing a database.

| Artifact | Purpose | Mutability |
|----------|---------|------------|
| `raw/{source}/{year}/...` | Immutable downloaded PDFs, HTML pages, ZIP files, and text indexes | Append-only |
| `manifests/{source}_{year}.json` | Known filings, official URLs, source IDs, last-seen hashes | Upsert by source ID |
| `trades.jsonl` | Deduplicated normalized transaction rows | Rebuilt or upserted from manifests |
| `filings.jsonl` | Filing-level metadata and parse status | Rebuilt or upserted from manifests |
| `members.json` | Enriched politician metadata | Versioned upsert |
| `issuer_aliases.json` | Asset-name to ticker resolution overrides | Human-reviewable |
| `parse_errors.jsonl` | Failed rows and low-confidence extraction details | Append-only |
| `sync_state.json` | Last successful incremental sync by source | Upsert |

SQLite may be added later for query performance, but JSONL is the MVP source of truth so the system remains inspectable and easy to diff.

---

## Epic Roadmap

| Epic | Name | Priority | Depends On |
|------|------|----------|------------|
| 0 | Compliance and Product Guardrails | Critical | None |
| 1 | Official Source Ingestion | Critical | Epic 0 |
| 2 | Parsing and Normalization | Critical | Epic 1 |
| 3 | Data Quality, Deduplication, and Auditability | Critical | Epic 2 |
| 4 | Ticker Resolution and Signal-Engine Integration | High | Epic 2 |
| 5 | Backend API and Service Layer | High | Epic 3 |
| 6 | New Web Section UX | High | Epic 5 |
| 7 | Watchlist Alerts and Contextual Surfaces | Medium | Epic 4, Epic 6 |
| 8 | Research Analytics and Event Studies | Medium | Epic 4 |
| 9 | Operations, Scheduling, and Observability | High | Epic 1, Epic 3 |
| 10 | Testing, Fixtures, and Release Gates | Critical | All epics |

---

# Epic 0: Compliance and Product Guardrails

**Goal**: Ensure the feature is safe, accurate, source-linked, and legally reviewable before any data is shown as part of a trading product.

**Target Files**:

- `Politicians.md`
- `src/ingestion/politicians/compliance.py`
- `src/web/frontend/src/features/politicians/disclaimers.ts`
- `src/web/backend/services/politicians_service.py`

## Story 0.1: Define Data Use Policy

**As a** product owner,
**I need** a written data-use policy for congressional financial disclosures,
**So that** the engineering team does not accidentally build a prohibited or misleading workflow.

**Acceptance Criteria**:

- [x] A `DATA_USE_POLICY` constant exists in the politician ingestion package.
- [x] Policy states that filings are public disclosures, not real-time trade confirmations.
- [x] Policy states that data must not be used for credit rating, solicitation, unlawful purposes, or any deployment prohibited by official source terms.
- [x] Policy states that the app must not call this an insider-trading signal.
- [x] Policy is surfaced in the web section footer or source-info drawer.
- [x] Policy can be linked from API responses via `data_use_notice`.

## Story 0.2: Add Compliance Mode

**As a** maintainer,
**I need** a feature flag that can disable ingestion or UI display until data-use review is complete,
**So that** development can continue without accidentally publishing the section.

**Acceptance Criteria**:

- [x] Environment variable `POLITICIANS_ENABLED=0|1` controls backend API availability.
- [x] Environment variable `POLITICIANS_COMPLIANCE_MODE=research_only|internal|public` is read by backend and frontend.
- [x] If disabled, `/api/politicians/*` returns a structured disabled response, not a stack trace.
- [x] Frontend route `/politicians` shows a disabled-state panel when the backend marks the feature unavailable.
- [x] Unit tests cover enabled, disabled, and unknown compliance modes.

## Story 0.3: Prevent Misleading Signal Integration

**As a** quantitative researcher,
**I need** politician-trade data to be clearly separated from predictive signal generation until it is validated,
**So that** the BMA engine is not polluted by untested contextual features.

**Acceptance Criteria**:

- [x] Politician activity is initially classified as `contextual_research`, not `model_feature`.
- [x] Signal pages may show contextual badges, but `latest_signals()` output is unchanged.
- [x] No BMA weights, PIT calibration, Kelly sizing, or high-conviction labels change when politician data is present.
- [x] Any future model integration must pass a separate no-leakage research gate using `disclosure_date`.

---

# Epic 1: Official Source Ingestion

**Goal**: Build robust, repeatable source adapters for official House and Senate disclosure sources.

**Target Files**:

- `src/ingestion/politicians/sources/house.py`
- `src/ingestion/politicians/sources/senate.py`
- `src/ingestion/politicians/download.py`
- `src/ingestion/politicians/source_health.py`
- `src/ingestion/politicians/cli.py`
- `Makefile`

## Story 1.1: House Yearly Archive Downloader

**As a** data engineer,
**I need** to download House financial disclosure archives by year,
**So that** the system can backfill and incrementally parse House PTR filings.

**Acceptance Criteria**:

- [x] CLI supports `python -m ingestion.politicians.cli house backfill --year 2026`.
- [x] Downloader discovers or accepts official yearly archive URLs from the House Financial Disclosure Reports page.
- [x] Downloaded artifacts are stored under `src/data/politicians/raw/house/{year}/`.
- [x] Each artifact is hashed with SHA-256 and recorded in `manifests/house_{year}.json`.
- [x] Re-running the command is idempotent and does not duplicate files.
- [x] Downloader respects `OFFLINE_MODE=1` by using existing raw files only.
- [x] Source errors are written to `source_health.json` with status `degraded`, not swallowed.

## Story 1.2: House PTR PDF Fetcher

**As a** parser,
**I need** every House PTR to have a local PDF artifact,
**So that** parsing can be deterministic and auditable.

**Acceptance Criteria**:

- [x] Manifest rows identify report IDs, filing year, document type, filer name, and official PDF URL where available.
- [x] PDFs are downloaded only when missing or when the remote hash changed.
- [x] File paths follow `raw/house/{year}/{report_id}.pdf`.
- [x] HTTP retries use exponential backoff and a conservative source-specific rate limit.
- [x] The fetcher records `downloaded_at`, `source_url`, `content_type`, `content_length`, and `sha256`.
- [x] If a PDF is unavailable, the filing is marked `missing_artifact` and visible in source health.

## Story 1.3: Senate eFD Search Adapter

**As a** data engineer,
**I need** an adapter for Senate eFD public search,
**So that** Senate PTR disclosures can be collected from the official public source.

**Acceptance Criteria**:

- [x] Adapter can search latest Senate PTR filings by filing date window.
- [x] Adapter stores raw HTML or PDF pages under `raw/senate/{year}/`.
- [x] Adapter does not bypass access controls, CAPTCHA, authentication, or public-use acknowledgements.
- [x] Adapter records the exact request parameters and response metadata in the Senate manifest.
- [x] If the public search endpoint changes, source health reports `degraded` with actionable diagnostics.
- [x] Senate rows include official `document_url` back to eFD search wherever possible.

## Story 1.4: Incremental Sync State

**As an** operator,
**I need** the ingestion process to know what it has already collected,
**So that** daily syncs are fast and safe.

**Acceptance Criteria**:

- [x] `sync_state.json` stores `last_successful_sync_at`, source, date window, counts, and error summary.
- [x] Incremental sync defaults to the last 14 calendar days to catch late updates and amendments.
- [x] Backfill mode can process a full year without overwriting unrelated years.
- [x] Sync emits Rich console tables consistent with `tune_ux.py` and `signals_ux.py`.
- [x] A failed source does not block parsing of already-downloaded artifacts from the other source.

## Story 1.5: Make Targets

**As a** developer,
**I need** simple Make commands for the politician data pipeline,
**So that** the feature is operable like the rest of the project.

**Acceptance Criteria**:

- [x] `make politicians-sync` runs incremental House and Senate sync.
- [x] `make politicians-backfill YEAR=2026` runs source backfill for the requested year.
- [x] `make politicians-parse` parses local raw artifacts into normalized JSONL.
- [x] `make politicians-status` shows source health, filing counts, parse errors, and newest disclosure.
- [x] `make politicians-test` runs politician ingestion, parser, API, and frontend unit tests.

---

# Epic 2: Parsing and Normalization

**Goal**: Convert messy public filings into a stable, deduplicated, source-linked trade dataset.

**Target Files**:

- `src/ingestion/politicians/parsers/house_pdf.py`
- `src/ingestion/politicians/parsers/senate_html.py`
- `src/ingestion/politicians/parsers/senate_pdf.py`
- `src/ingestion/politicians/normalize.py`
- `src/ingestion/politicians/schema.py`
- `src/ingestion/politicians/ticker_resolution.py`

## Story 2.1: House PDF Table Parser

**As a** data engineer,
**I need** to parse House PTR PDF tables,
**So that** transaction rows can be extracted from official filings.

**Acceptance Criteria**:

- [x] Parser extracts filer name, filing date, report ID, owner, asset name, transaction type, transaction date, notification date, and amount bucket.
- [x] Parser supports both modern table-like PDFs and common older text-layout PDFs.
- [x] Parser handles multi-line asset names without merging adjacent rows.
- [x] Parser maps House transaction codes such as `P`, `S`, `S-P`, and `E` to canonical transaction types.
- [x] Parser assigns `parser_confidence` per row.
- [x] Rows below confidence `0.80` are quarantined unless manually approved.
- [x] Golden fixture tests cover at least 25 House PDFs across 2024, 2025, and 2026.

## Story 2.2: Senate eFD Parser

**As a** data engineer,
**I need** to parse Senate eFD PTR pages and printable reports,
**So that** Senate disclosures are normalized to the same schema as House rows.

**Acceptance Criteria**:

- [x] Parser extracts owner, ticker if present, asset name, asset type, transaction type, amount bucket, and comments.
- [x] Parser supports Senate HTML table layouts and PDF/printable layouts if present.
- [x] Parser preserves Senate-specific fields that do not exist in House data in a `source_extra` object.
- [x] Parser maps Senate owner labels to `self`, `spouse`, `dependent_child`, `joint`, or `unknown`.
- [x] Parser records missing ticker fields without failing the row.
- [x] Golden fixture tests cover at least 15 Senate filings with stocks, ETFs, options, and ambiguous assets.

## Story 2.3: Amount Bucket Normalization

**As a** researcher,
**I need** amount buckets converted to numeric ranges,
**So that** activity can be aggregated without pretending exact trade sizes are known.

**Acceptance Criteria**:

- [x] `$1,001 - $15,000` maps to `amount_min_usd=1001`, `amount_max_usd=15000`.
- [x] `Over $50,000,000` maps to `amount_min_usd=50000001`, `amount_max_usd=null`.
- [x] Spouse/dependent-child special buckets are preserved in `amount_bucket_raw`.
- [x] UI and API label midpoint-derived values as estimates.
- [x] Tests cover every official amount bucket observed in House and Senate fixtures.

## Story 2.4: Date Normalization

**As a** quantitative researcher,
**I need** transaction, notification, filed, and disclosure dates normalized separately,
**So that** event studies can avoid lookahead bias.

**Acceptance Criteria**:

- [x] Parser stores `transaction_date`, `notification_date`, `filed_date`, and `disclosure_date` independently.
- [x] Unknown dates are stored as `null`, not inferred silently.
- [x] `delay_days = disclosure_date - transaction_date` when both dates are present.
- [x] Records with impossible dates, such as disclosure before transaction, are `valid_with_warnings` or `quarantined`.
- [x] Event-study helpers use `disclosure_date` by default.

## Story 2.5: Transaction Deduplication and Amendments

**As a** user,
**I need** amended filings to correct or supersede prior records without double-counting,
**So that** aggregate views are trustworthy.

**Acceptance Criteria**:

- [x] Every row has deterministic `trade_id` and `row_hash`.
- [x] Exact duplicate rows are stored once.
- [x] Amendments are linked through `is_amendment` and `amends_report_id` where source data allows.
- [x] Aggregations default to latest effective row, but filing-detail view can show amendment history.
- [x] Tests cover duplicate PDFs, amended reports, and repeated rows within a single filing.

---

# Epic 3: Data Quality, Validation, and Auditability

**Goal**: Make the dataset inspectable, reproducible, and honest about uncertainty.

**Target Files**:

- `src/ingestion/politicians/validation.py`
- `src/ingestion/politicians/storage.py`
- `src/ingestion/politicians/quality.py`
- `src/tests/test_politicians_validation.py`

## Story 3.1: Validation Contract

**As a** maintainer,
**I need** explicit validation rules for normalized trades,
**So that** bad rows do not silently enter the web section.

**Acceptance Criteria**:

- [x] Required fields are validated before writing to `trades.jsonl`.
- [x] Invalid enum values fail validation.
- [x] Missing `document_url` fails validation unless the record source is `manual_fixture`.
- [x] Missing ticker is allowed only when `asset_type` is not confidently public equity or ETF.
- [x] Validation failures are written to `parse_errors.jsonl` with filing ID and row context.
- [x] Validation summary is available through `make politicians-status`.

## Story 3.2: Parser Confidence Scoring

**As a** user,
**I need** to know which parsed rows are reliable,
**So that** low-confidence extracted data is not over-trusted.

**Acceptance Criteria**:

- [x] Parser confidence combines field completeness, table alignment, date validity, amount-bucket recognition, and ticker-resolution confidence.
- [x] `parser_confidence >= 0.95` is considered high confidence.
- [x] `0.80 <= parser_confidence < 0.95` is shown with a warning in filing detail.
- [x] `parser_confidence < 0.80` is quarantined by default.
- [x] Source-health API reports counts by confidence bucket.

## Story 3.3: Immutable Source Audit Trail

**As a** compliance reviewer,
**I need** every normalized row traceable to raw source content,
**So that** the system can explain how a displayed trade was produced.

**Acceptance Criteria**:

- [x] Every trade has `source_hash`, `row_hash`, `parser_version`, `document_url`, and `raw_artifact_path`.
- [x] Re-parsing the same raw artifact with the same parser version produces identical row hashes.
- [x] Parser-version changes are recorded in `filings.jsonl`.
- [x] UI filing detail includes source document link and parse metadata.
- [x] Tests verify deterministic parse output for golden fixtures.

## Story 3.4: Late Filing and Anomaly Flags

**As a** user,
**I need** to see late or unusual filings,
**So that** the feed can highlight records that deserve closer review.

**Acceptance Criteria**:

- [x] Records with `delay_days > 45` are flagged `late_disclosure`.
- [x] Records with unusually large amount buckets are flagged `large_trade_bucket`.
- [x] Records with ambiguous ticker resolution are flagged `ticker_ambiguous`.
- [x] Records with amendments are flagged `amended`.
- [x] Flags are filterable in backend API and frontend.
- [x] Flag logic is covered by unit tests.

---

# Epic 4: Ticker Resolution and Signal-Engine Integration

**Goal**: Connect disclosures to the existing asset universe without overstating certainty.

**Target Files**:

- `src/ingestion/politicians/ticker_resolution.py`
- `src/ingestion/politicians/issuer_aliases.json`
- `src/web/backend/services/politicians_service.py`
- `src/decision/politician_context.py`

## Story 4.1: Asset Name to Ticker Resolution

**As a** researcher,
**I need** raw asset names mapped to tradable symbols,
**So that** disclosures can be linked to existing signals and watchlists.

**Acceptance Criteria**:

- [x] Resolver first trusts explicit ticker fields from the source when present.
- [x] Resolver uses `issuer_aliases.json` for known aliases and manual corrections.
- [x] Resolver can mark `ticker=null` with reason `private_asset`, `fund_unmapped`, `bond_unmapped`, or `ambiguous`.
- [x] Resolver never guesses a ticker when multiple public issuers match.
- [x] All manual aliases include `added_by`, `added_at`, `reason`, and optional source note.
- [x] Tests cover common company suffixes, share classes, ETFs, ADRs, and ambiguous names.

## Story 4.2: Existing Universe Linkage

**As a** signal-engine user,
**I need** politician activity linked to assets already tracked by the system,
**So that** context appears where it is most useful.

**Acceptance Criteria**:

- [x] Service computes `is_tracked_asset` by matching normalized ticker to current signal universe.
- [x] Service computes `is_watchlist_asset` by matching normalized ticker to watchlist symbols.
- [x] API can filter `tracked_only=true` and `watchlist_only=true`.
- [x] Charts and signal pages can request politician activity for a single symbol.
- [x] No existing signal output changes when this context is missing or disabled.

## Story 4.3: Disclosure-Time Event Context

**As a** quantitative researcher,
**I need** event windows anchored on public disclosure dates,
**So that** analysis uses only information that would have been available.

**Acceptance Criteria**:

- [x] Helper returns event windows keyed by `disclosure_date`, not `transaction_date`.
- [x] Event windows include prior 5 trading days and forward 1, 7, 30, and 90 trading days.
- [x] Helper can join to existing price cache in `src/data/prices`.
- [x] Missing price data returns structured warnings, not exceptions.
- [x] Documentation explicitly labels transaction-date analysis as retrospective only.

## Story 4.4: Context Score

**As a** user,
**I need** a compact score that summarizes politician activity around an asset,
**So that** the web section can rank assets without pretending certainty.

**Acceptance Criteria**:

- [x] `politician_activity_score` is computed from disclosed buy/sell imbalance, amount-bucket midpoint, recency, unique filers, and parser confidence.
- [x] Score is bounded `[-1, 1]`, where positive means net disclosed purchases and negative means net disclosed sales.
- [x] Score includes a `confidence` field separate from direction.
- [x] Score uses exponential recency decay based on `disclosure_date`.
- [x] Score is never fed into BMA or Kelly sizing in MVP.
- [x] Score explanation is returned by API for UI tooltips.

---

# Epic 5: Backend API and Service Layer

**Goal**: Provide stable API endpoints for the frontend and future integrations.

**Target Files**:

- `src/web/backend/routers/politicians.py`
- `src/web/backend/services/politicians_service.py`
- `src/web/backend/models.py`
- `src/web/backend/main.py`
- `src/web/frontend/src/api.ts`

## Story 5.1: Register Politicians Router

**As a** frontend developer,
**I need** `/api/politicians` endpoints,
**So that** the new web section can load politician-trade data consistently.

**Acceptance Criteria**:

- [x] `src/web/backend/main.py` includes `politicians.router` with prefix `/api/politicians`.
- [x] Router returns structured errors when the feature is disabled or data files are missing.
- [x] Router responses include `generated_at`, `data_age_seconds`, and `data_use_notice`.
- [x] API tests cover healthy, missing-data, and disabled states.

## Story 5.2: Summary Endpoint

**As a** dashboard user,
**I need** a summary endpoint,
**So that** the page can render headline metrics quickly.

**Acceptance Criteria**:

- [x] `GET /api/politicians/summary` returns total trades, new disclosures in last 7 days, tracked-asset trades, watchlist trades, late filings, source health, and newest disclosure date.
- [x] Endpoint computes buy/sell counts and amount-bucket midpoint totals by chamber.
- [x] Endpoint responds in under 250 ms for 100,000 normalized rows on a warm process.
- [x] Endpoint handles empty dataset with zero counts and no exception.

## Story 5.3: Trade Feed Endpoint

**As a** user,
**I need** a paginated and filterable trade feed,
**So that** I can inspect disclosures by asset, politician, source, date, amount, and flags.

**Acceptance Criteria**:

- [x] `GET /api/politicians/trades` supports `limit`, `offset`, `symbol`, `filer`, `chamber`, `party`, `state`, `transaction_type`, `owner`, `flag`, `tracked_only`, `watchlist_only`, `from`, and `to`.
- [x] Default sorting is newest `disclosure_date` first.
- [x] Response includes total count and page metadata.
- [x] Response includes official source URL for every row.
- [x] Invalid filters return HTTP 422 with helpful detail.

## Story 5.4: Asset Detail Endpoint

**As a** signal user,
**I need** politician activity for a specific ticker,
**So that** chart and signal views can show relevant disclosure context.

**Acceptance Criteria**:

- [x] `GET /api/politicians/assets/{symbol}` returns recent trades, unique filers, buy/sell imbalance, amount estimates, activity score, and disclosure timeline.
- [x] Endpoint accepts `window_days` with default `180`.
- [x] Endpoint includes `known_limitations` when ticker resolution is ambiguous.
- [x] Endpoint works for symbols containing dots, dashes, or equals signs.

## Story 5.5: Filer Detail Endpoint

**As a** user,
**I need** a member-level view,
**So that** I can understand a politician's disclosed activity over time.

**Acceptance Criteria**:

- [x] `GET /api/politicians/filers/{filer_id}` returns filer metadata, recent trades, top tickers, top sectors, delay stats, and source documents.
- [x] Endpoint does not expose non-public personal data.
- [x] Endpoint distinguishes self, spouse, dependent child, joint, and unknown ownership.
- [x] Endpoint works when member metadata is incomplete.

## Story 5.6: Source Health Endpoint

**As an** operator,
**I need** source health visibility,
**So that** broken scrapers or changed source layouts are caught quickly.

**Acceptance Criteria**:

- [x] `GET /api/politicians/source-health` returns source status, last sync time, newest filing, parse success rate, low-confidence rows, and recent errors.
- [x] Status values are `ok`, `degraded`, `offline`, and `disabled`.
- [x] Health includes source-specific remediation messages.
- [x] Services page can consume this endpoint later without additional backend changes.

---

# Epic 6: New Web Section UX

**Goal**: Add a first-class Politicians section to the web dashboard for scanning, filtering, verification, and ticker context.

**Target Files**:

- `src/web/frontend/src/pages/PoliticiansPage.tsx`
- `src/web/frontend/src/features/politicians/components/PoliticianInsightBar.tsx`
- `src/web/frontend/src/features/politicians/components/TradeFeedTable.tsx`
- `src/web/frontend/src/features/politicians/components/AssetActivityPanel.tsx`
- `src/web/frontend/src/features/politicians/components/FilerDrawer.tsx`
- `src/web/frontend/src/features/politicians/components/SourceHealthStrip.tsx`
- `src/web/frontend/src/components/Layout.tsx`
- `src/web/frontend/src/App.tsx`
- `src/web/frontend/src/api.ts`

## Story 6.1: Route and Navigation

**As a** user,
**I need** a Politicians nav item,
**So that** political disclosure activity is discoverable from the main dashboard.

**Acceptance Criteria**:

- [x] `/politicians` route renders `PoliticiansPage`.
- [x] Sidebar includes `Politicians` with a lucide icon such as `Landmark`, `Scale`, or `BadgeDollarSign`.
- [x] Nav badge shows number of new tracked-asset disclosures in the last 7 days.
- [x] Tooltip shows newest disclosure date, source health, and watchlist match count.
- [x] Command palette can navigate to Politicians.

## Story 6.2: Insight Bar

**As a** user,
**I need** a compact headline summary,
**So that** I can understand the latest activity in one glance.

**Acceptance Criteria**:

- [x] Insight bar shows total recent disclosures, tracked-asset matches, watchlist matches, late filings, and source health.
- [x] Count chips are clickable filters.
- [x] The page clearly labels data as delayed public disclosures.
- [x] Empty state explains that no disclosures matched current filters, not that no politicians traded.
- [x] Loading state uses existing skeleton/loading conventions.

## Story 6.3: Trade Feed Table

**As a** user,
**I need** a dense, sortable trade feed,
**So that** I can inspect many disclosures efficiently.

**Acceptance Criteria**:

- [x] Table columns include disclosure date, transaction date, delay, politician, chamber, party/state, owner, ticker, asset, transaction type, amount bucket, confidence, and source link.
- [x] Sort supports disclosure date, transaction date, amount midpoint, delay days, politician, and ticker.
- [x] Filters include search, symbol, chamber, party, state, owner, transaction type, flags, tracked-only, and watchlist-only.
- [x] Source link opens official document in a new tab.
- [x] Low-confidence rows are visually marked but still inspectable.
- [x] Text does not overflow or overlap at desktop and mobile widths.

## Story 6.4: Asset Activity Panel

**As a** signal user,
**I need** an asset-level panel,
**So that** I can see politician activity around a ticker without leaving the section.

**Acceptance Criteria**:

- [x] Selecting a ticker shows recent trades, buy/sell imbalance, amount estimates, unique filers, and activity score.
- [x] Panel links to existing `/charts/{symbol}` route when the asset is tracked.
- [x] Panel distinguishes disclosed purchases from disclosed sales using semantic green/red accents.
- [x] Panel uses disclosure dates for timelines by default.
- [x] Panel includes "Retrospective transaction dates" only as a clearly labeled secondary view.

## Story 6.5: Filer Drawer

**As a** user,
**I need** a politician detail drawer,
**So that** I can inspect one filer's activity and source documents.

**Acceptance Criteria**:

- [x] Clicking a politician opens a drawer with metadata, recent trades, top tickers, top sectors, ownership breakdown, and delay statistics.
- [x] Drawer includes source links for each filing.
- [x] Drawer displays committee data when available and marks it as enrichment, not source disclosure.
- [x] Drawer avoids non-public personal information.
- [x] Drawer is keyboard accessible and closable with Escape.

## Story 6.6: Source Health Strip

**As a** user,
**I need** to know whether the data pipeline is healthy,
**So that** stale or partial data is not mistaken for quiet political activity.

**Acceptance Criteria**:

- [x] Source health strip shows House status, Senate status, last sync, parse success, and low-confidence count.
- [x] Degraded sources are amber, offline sources are red, healthy sources are green.
- [x] Clicking the strip reveals latest source errors and remediation hints.
- [x] UI never hides source-health warnings behind a settings-only flow.

## Story 6.7: Responsive and Visual QA

**As a** frontend maintainer,
**I need** the page to work across viewports,
**So that** the new section feels native to the existing dashboard.

**Acceptance Criteria**:

- [x] Desktop layout supports table plus side panel without nested cards.
- [x] Mobile layout collapses to filter drawer plus stacked trade rows.
- [x] Buttons use icons where appropriate and tooltips for unfamiliar actions.
- [x] No text overlaps inside filters, badges, tables, drawers, or nav.
- [x] Playwright screenshots pass for desktop, tablet, and mobile viewports.

---

# Epic 7: Watchlist Alerts and Contextual Surfaces

**Goal**: Surface relevant politician activity where users already look, without turning it into a trade directive.

**Target Files**:

- `src/web/frontend/src/features/signals/components/WatchlistView.tsx`
- `src/web/frontend/src/features/signals/components/AllAssetsTable.tsx`
- `src/web/frontend/src/pages/ChartsPage.tsx`
- `src/web/backend/services/politicians_service.py`
- `src/web/frontend/src/stores/toastStore.tsx`

## Story 7.1: Watchlist Disclosure Matches

**As a** watchlist user,
**I need** to see when a watched ticker has recent public politician activity,
**So that** I can inspect context without searching manually.

**Acceptance Criteria**:

- [x] Watchlist rows can display a small Politicians badge when the symbol has recent disclosures.
- [x] Badge count uses disclosure date window, default 30 days.
- [x] Clicking the badge navigates to `/politicians?symbol={symbol}&watchlist_only=true`.
- [x] Badge never changes bullish/bearish signal color or ranking.
- [x] If politician data is disabled or stale, watchlist behaves exactly as it does today.

## Story 7.2: Signal Table Context Column

**As a** signal user,
**I need** optional politician context in the all-assets table,
**So that** I can scan where public disclosures intersect with model signals.

**Acceptance Criteria**:

- [x] Column is hidden by default unless enabled in column customizer.
- [x] Column shows net purchases/sales and disclosure count for the selected window.
- [x] Column tooltip explains amount estimates and delay.
- [x] Sorting by the column uses `politician_activity_score`.
- [x] No existing signal computations change.

## Story 7.3: Chart Timeline Overlay

**As a** chart user,
**I need** disclosure markers on price charts,
**So that** I can visually compare public disclosures to price movement and forecasts.

**Acceptance Criteria**:

- [x] Chart page can request `/api/politicians/assets/{symbol}`.
- [x] Markers are plotted by `disclosure_date` by default.
- [x] Purchases and sales have distinct marker shapes or colors.
- [x] Multiple same-day disclosures aggregate into one marker with a tooltip.
- [x] Overlay can be toggled off and persists per browser.

---

# Epic 8: Research Analytics and Event Studies

**Goal**: Provide honest analytics that quantify historical behavior without optimizing production signals prematurely.

**Target Files**:

- `src/research/politicians/event_study.py`
- `src/research/politicians/report.py`
- `src/decision/politician_context.py`
- `src/tests/test_politicians_event_study.py`

## Story 8.1: Disclosure-Date Event Study

**As a** researcher,
**I need** an event-study report anchored on disclosure dates,
**So that** I can measure post-disclosure behavior without lookahead bias.

**Acceptance Criteria**:

- [x] Event study supports forward returns at 1, 7, 30, and 90 trading days.
- [x] Events are grouped by transaction type, chamber, asset type, amount bucket, and parser-confidence bucket.
- [x] Report includes sample counts and confidence intervals.
- [x] Report excludes events with missing or stale price data.
- [x] Command prints clear warnings when sample size is too small.

## Story 8.2: Transaction-Date Retrospective Analysis

**As a** researcher,
**I need** a retrospective transaction-date analysis,
**So that** I can compare actual transaction timing versus public disclosure timing.

**Acceptance Criteria**:

- [x] Report is clearly labeled `RETROSPECTIVE_ONLY`.
- [x] No production feature consumes transaction-date returns.
- [x] Output includes median filing delay by chamber and transaction type.
- [x] Analysis quantifies how much apparent edge disappears when switching from transaction date to disclosure date.
- [x] Tests ensure production helpers default to disclosure date.

## Story 8.3: Committee and Sector Clustering

**As a** risk analyst,
**I need** to identify clusters by committee and sector,
**So that** public trading activity can be reviewed in policy-relevant context.

**Acceptance Criteria**:

- [x] Enrichment can map tickers to sectors using existing sector mappings where available.
- [x] Enrichment can map members to committees when official metadata is available.
- [x] Report shows committee-sector heatmap with counts and amount estimates.
- [x] Missing committee data is displayed as unknown, not inferred.
- [x] UI labels committee data as enrichment separate from disclosure source fields.

---

# Epic 9: Operations, Scheduling, and Observability

**Goal**: Make the politician pipeline reliable enough for daily use.

**Target Files**:

- `src/ingestion/politicians/cli.py`
- `src/web/backend/tasks.py`
- `src/web/backend/routers/tasks.py`
- `src/web/backend/services/health_service.py`
- `Makefile`

## Story 9.1: Daily Sync Task

**As an** operator,
**I need** a daily background task,
**So that** new disclosures appear without manual intervention.

**Acceptance Criteria**:

- [x] Task can run via CLI and backend task router.
- [x] Task syncs sources, parses new artifacts, validates output, and refreshes API cache.
- [x] Task writes a structured run summary with counts and errors.
- [x] Task is safe to run multiple times per day.
- [x] Task respects `OFFLINE_MODE=1`.

## Story 9.2: Services Health Integration

**As a** dashboard operator,
**I need** politician ingestion health included in service monitoring,
**So that** data freshness problems are visible.

**Acceptance Criteria**:

- [x] Services health includes politician source status and data age.
- [x] Degraded politician ingestion does not mark the entire app down.
- [x] Health panel links to `/politicians` source-health details.
- [x] Error logs include source, filing ID, parser version, and exception class.

## Story 9.3: Cache Strategy

**As a** backend maintainer,
**I need** politician API responses cached safely,
**So that** the web section is fast without serving stale data indefinitely.

**Acceptance Criteria**:

- [x] Service caches parsed JSONL in memory with file mtime invalidation.
- [x] Cache includes a version tied to response schema.
- [x] Manual refresh endpoint invalidates politician cache.
- [x] API returns `data_age_seconds`.
- [x] Tests cover cache invalidation after file updates.

---

# Epic 10: Testing, Fixtures, and Release Gates

**Goal**: Release only when parsing, storage, API, and UI are verified end to end.

**Target Files**:

- `src/tests/test_politicians_*.py`
- `src/web/frontend/src/features/politicians/*.test.tsx`
- `src/web/frontend/src/pages/PoliticiansPage.test.tsx`
- `src/data/politicians/fixtures/`

## Story 10.1: Golden Filing Fixtures

**As a** maintainer,
**I need** stable filing fixtures,
**So that** source layout changes and parser regressions are caught.

**Acceptance Criteria**:

- [x] Fixture set includes at least 25 House and 15 Senate filings.
- [x] Fixtures cover typed PDFs, multi-line assets, amendments, spouse trades, dependent-child trades, ETFs, options, bonds, and unknown assets.
- [x] Expected normalized JSON exists for every fixture.
- [x] Tests compare exact normalized output excluding timestamps.
- [x] Fixtures are documented with source URL and retrieval date.

## Story 10.2: End-to-End Pipeline Test

**As a** release owner,
**I need** an offline end-to-end test,
**So that** the pipeline can be verified without network access.

**Acceptance Criteria**:

- [x] Test runs from raw fixture artifacts to `trades.jsonl`.
- [x] Test validates row counts, hashes, schema, and parse-error counts.
- [x] Test verifies idempotent rerun.
- [x] Test verifies `OFFLINE_MODE=1` uses fixture artifacts only.
- [x] Test runs in under 30 seconds on local development hardware.

## Story 10.3: API Contract Tests

**As a** frontend developer,
**I need** backend API contract tests,
**So that** UI code can rely on response shapes.

**Acceptance Criteria**:

- [x] Tests cover summary, trades, asset detail, filer detail, and source health.
- [x] Tests cover pagination and filters.
- [x] Tests cover missing data and disabled feature states.
- [x] Tests assert every trade includes official source URL and parser confidence.
- [x] Tests verify invalid filter errors are structured and user-readable.

## Story 10.4: Frontend Interaction Tests

**As a** frontend maintainer,
**I need** component and page tests,
**So that** filtering, sorting, drawers, and source links work reliably.

**Acceptance Criteria**:

- [x] Page renders summary, table, filters, and source-health strip from mocked API data.
- [x] Clicking count chips applies filters.
- [x] Sorting updates table order.
- [x] Filer drawer opens and closes via click and Escape.
- [x] Source links have `target="_blank"` and safe `rel` attributes.
- [x] Disabled state renders when backend marks feature unavailable.

## Story 10.5: Visual QA Gate

**As a** product owner,
**I need** visual verification before release,
**So that** the new section feels like part of the existing dashboard.

**Acceptance Criteria**:

- [x] Playwright screenshots captured for desktop, tablet, and mobile.
- [x] No overlapping text, clipped buttons, or broken table rows.
- [x] Empty, loading, healthy, degraded, and low-confidence states are captured.
- [x] Politicians page uses existing dashboard spacing, typography, and semantic colors.
- [x] Page avoids a marketing-style landing screen; the usable monitoring surface is first.

---

## MVP Release Plan

### Phase 0: Source and Compliance Spike

- Validate House yearly archive structure and Senate eFD access pattern.
- Confirm data-use constraints for intended deployment.
- Build 5 House and 5 Senate fixtures manually.
- Deliver source-health prototype.

**Exit Criteria**:

- [ ] Official source paths verified.
- [ ] Compliance mode implemented.
- [ ] Fixture parser proof-of-concept reaches 90%+ row extraction on sample filings.

### Phase 1: Data Pipeline MVP

- Build House downloader, Senate adapter, raw archive, parser, schema validation, and JSONL storage.
- Add Make targets and offline fixtures.

**Exit Criteria**:

- [ ] 2025 and 2026 House backfill runs locally.
- [ ] Senate latest-window sync runs or reports degraded with actionable diagnostics.
- [ ] `trades.jsonl` generated with deterministic IDs and source links.

### Phase 2: Backend API MVP

- Add `/api/politicians` router and service.
- Implement summary, trades, asset detail, filer detail, and source health.

**Exit Criteria**:

- [ ] API contract tests pass.
- [ ] Warm summary response under 250 ms on target dataset.
- [ ] Disabled and missing-data states are clean.

### Phase 3: Web Section MVP

- Add sidebar route, summary bar, filters, table, asset panel, filer drawer, and source-health strip.

**Exit Criteria**:

- [ ] Page is usable as first screen.
- [ ] Watchlist-only and tracked-only filters work.
- [ ] Source links and confidence warnings are visible.
- [ ] Playwright visual QA passes.

### Phase 4: Contextual Integrations

- Add optional badges to Watchlist, Signal table, and Charts.
- Add disclosure-date event study report.

**Exit Criteria**:

- [ ] Existing signal outputs remain unchanged.
- [ ] Context badges link to Politicians section.
- [ ] Event study proves no lookahead leakage.

---

## Success Metrics

| Metric | MVP Target |
|--------|------------|
| House parse success rate | >= 95% high-confidence rows on fixture set |
| Senate parse success rate | >= 90% high-confidence rows on fixture set |
| Row source-link coverage | 100% |
| API summary latency | < 250 ms warm |
| Trade feed latency | < 500 ms for first page |
| Unknown ticker rate for public equities/ETFs | < 10% after alias review |
| Duplicate effective-trade rate | < 1% after deduplication |
| UI visual QA | 0 critical overlap/clipping issues |
| Signal regression | 0 changes to existing signal outputs |

---

## Open Questions

1. Is the intended use strictly internal research, or will the web section be exposed in any commercial/user-facing deployment?
2. Should MVP include candidates and senior staff, or only sitting Members of Congress?
3. Should third-party APIs be allowed for validation, or should MVP remain official-source only?
4. Should options trades receive a separate options-aware parser and display model in MVP?
5. Should committee/sector conflict analysis be shown immediately, or delayed until member metadata quality is proven?
6. Should the app store raw PDFs in gitignored local data only, or support a configurable external artifact directory?
7. Should alerts be browser-only initially, or should they integrate with email/desktop notifications later?

---

## Definition of Done

The Politicians section is done when:

- [ ] Official source ingestion works for House and Senate or reports source degradation transparently.
- [ ] Raw source artifacts and normalized trades are stored under `src/data/politicians`.
- [ ] Every displayed row links to an official source document.
- [ ] Parser confidence, source health, and delayed-disclosure warnings are visible.
- [ ] `/api/politicians` endpoints are tested and documented through types in `api.ts`.
- [ ] `/politicians` route is navigable from the sidebar and command palette.
- [ ] Watchlist and chart integrations are optional and non-invasive.
- [ ] Existing signal, tuning, risk, and arena behavior is unchanged.
- [ ] Offline fixture tests, API tests, and frontend visual QA pass.
- [ ] Compliance mode and data-use policy are active.
