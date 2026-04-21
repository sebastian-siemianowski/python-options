# signals.md — Premium Apple-Grade UX for `/signals`

**Status:** Draft v1 — ready for implementation
**Owner:** UI / UX engineering
**Date:** 21 April 2026
**Reference design language:** `/tuning` v3/v4 (see AGENTS.md and `memories/session/tuning-ux-v2.md`)

---

## Philosophy

The `/signals` page is the **command center** of the platform. Traders should feel the same clarity, density, and quiet authority as Apple's native trading apps and the best desktop terminals. No decorative chrome. No redundant information. Every pixel earns its keep.

**Design principles (carried from `/tuning` v3/v4):**
- Apple Vision Pro / Notion Calendar era: matte glass, hairline borders, semantic color only.
- One hero numeral per surface. 64 px `num-hero`, `-0.04em` tracking, `tabular-nums`, `lh 0.92`.
- Single violet→cyan accent; semantic emerald / amber / rose for direction; everything else is greyscale.
- Radius scale `8 / 14 / 20 / 28`. Hairline borders `rgba(255,255,255,0.035–0.05)`.
- Hover = brightness 1.08. **No scale / transform.** 150 ms ease-out for everything.
- Row height 40 px, `tabular-nums`, dot + white label replaces every colored pill.
- Selected row = 2 px violet rail + 20 px inset violet glow + 5 % violet tint.

**Shared CSS tokens already available** (`src/web/frontend/src/index.css`):
`.hero-surface`, `.num-hero`, `.num-display`, `.label-micro`, `.stat-col-divider`, `.glass-card`, `.fade-up`, `.focus-ring`, `.premium-table`, `.signals-row-selected`.

---

## Story 0 — Foundation Audit (completed in v1)

**Already shipped in the v1 pass and verified green (`tsc --noEmit` → EXIT=0):**
- Default view `sectors` → `all`, persisted in `localStorage` (fixes "only seeing a couple of positions").
- Sector view auto-expands all sectors on first load.
- `SignalsHero` band replaces `PageHeader` + 6-card `StatCard` grid: 64 px conviction numeral, live WS dot, bullish/bearish split bar, 5-column stats strip.
- `CosmicSignalRow` and `SectorSignalRow`: whole row clickable, rotating `ChevronRight` indicator, 40 px height, hairline separators, violet rail + glow when selected.
- `ExportButton` relocated to its own lightweight row above the toolbar.

**Files touched:** [src/web/frontend/src/pages/SignalsPage.tsx](src/web/frontend/src/pages/SignalsPage.tsx).

---

## Story 1 — Hero Command Center polish

**Outcome:** The hero band becomes a genuine at-a-glance command center — one glance tells you market tilt, data freshness, and total conviction.

**Acceptance criteria:**
- [ ] `SignalsHero` gets three live micro-metrics under the numeral: `Universe median momentum` (tabular %), `Median crash risk` (tabular %), `Top sector` (dot + name).
- [ ] The split bar grows a **third lane** for `Exit` signals (amber) between neutral and bearish.
- [ ] Hovering a split lane shows a tooltip with exact count + percentage (tabular-nums) using the `.glass-card` radius-14 popover.
- [ ] The numeral pulses (2 s ease-in-out) exactly once when the WebSocket pushes a new batch that changes `strong_buy + strong_sell`.
- [ ] `label-micro` caption includes data-age: `LIVE · 12s ago` or `STALE · 4m ago` (amber if > 120 s).
- [ ] Hero surface height is locked at 152 px on desktop, 240 px on mobile (stacked).

**Out of scope:** any color other than the established palette.

**Files:** `SignalsPage.tsx` (SignalsHero component).

---

## Story 2 — Unified Command Toolbar

**Outcome:** One single 56 px rounded-full bar replaces the current three-row jumble of `view-toggle / filter / search / ticker-tape / horizon-pills`.

**Acceptance criteria:**
- [ ] Layout: `[search 36px]  [view segment: All · Sectors · Strong]  [filter segment: All · Buy · Hold · Sell]  ———  [horizon pills]  [count 490 of 490]  [⌘K]`
- [ ] Search is a rounded-full 36 px input, `rgba(13,13,24,0.55)` with hairline border; `/` keyboard shortcut focuses it; clears with `Esc`.
- [ ] Segmented controls use the "liquid glass" pattern: the inactive segment is transparent, the active segment is a violet→cyan gradient pill that slides with a 220 ms cubic-bezier transition using `transform: translateX(...)` on a single absolutely-positioned indicator div (one DOM node, GPU-accelerated).
- [ ] Horizon pills remain but shrink to 28 px height, use `label-micro` for labels, and collapse into a `+N` overflow chip on widths < 1100 px.
- [ ] Count readout uses `num-display` (22 px) on the right, with `tabular-nums` so numbers never jitter.
- [ ] `⌘K` button opens Story 12 (command palette). Shows `⌘K` on macOS, `Ctrl K` on Windows/Linux using `navigator.platform`.
- [ ] Toolbar sticks to `top: 0` with `backdrop-filter: blur(28px) saturate(1.2)` and a hairline bottom border on scroll.

**Out of scope:** modifying filter / view semantics themselves.

**Files:** `SignalsPage.tsx`. Possibly extract `SignalsToolbar.tsx`.

---

## Story 3 — TradingView Subpanel

**Outcome:** Clicking a row expands a world-class chart powered by TradingView's own `lightweight-charts` (already at v5.1 in `package.json`). This replaces the existing canvas `MiniChartPanel` entirely.

**Acceptance criteria:**
- [ ] New component `SignalDetailPanel` replaces `MiniChartPanel`. Height animates from 0 → 360 px with `cubic-bezier(0.22, 1, 0.36, 1)` over 280 ms.
- [ ] Uses `createChart()` from `lightweight-charts` in dark mode, violet/cyan palette for series, emerald/rose for candles, grid `rgba(255,255,255,0.035)`.
- [ ] Data source: `api.chartOhlcv(symbol, tail=365)`. Cached with react-query for 5 min per symbol.
- [ ] Controls in a 40 px header strip above the chart:
  - Chart type toggle: `Candles · Line · Area` (segmented, same liquid-glass pattern).
  - Range toggle: `1M · 3M · 6M · 1Y · Max` (uses `setVisibleLogicalRange`).
  - Right-aligned: `Open in Full View` → navigates to `/charts/:symbol` (opens existing `ChartsPage`).
- [ ] Left 72 % = main chart, right 28 % = stats strip with:
  - `Signal` (dot + label, e.g. `STRONG BUY`),
  - `Momentum %` (`num-display` 22 px),
  - `Crash risk %` (heatmap block),
  - `Horizon forecasts` (7d / 30d / 90d as hairline rows with direction arrow),
  - `Last price` and `Δ 1d` % with tabular-nums.
- [ ] Crosshair readout: floating pill bottom-right with `O / H / L / C / V` formatted values, tabular, hairline, violet text.
- [ ] Resize: uses `ResizeObserver` on the container; chart `resize()` debounced at 60 ms.
- [ ] Disposal: `chart.remove()` in the `useEffect` cleanup — no leaks.
- [ ] **Accessibility:** subpanel has `role="region"`, `aria-label="${ticker} price chart"`, and receives focus on expand; `Esc` collapses it.
- [ ] **Empty / error state:** if `chartOhlcv` returns 0 bars → `CosmicEmptyState`-styled block with `LineChart` icon and CTA `Run data refresh` (links into Story 6 flow for this single symbol).

**Symbol-swap behavior:** When a user expands a different ticker while one subpanel is already open, the existing panel collapses in 180 ms and the new one expands in 280 ms. Data swap is seamless — `setData([])` is **never** called; the chart instance is fully replaced. Price data fades (opacity 1→0.4→1) during the swap so the switch is visible.

**TradingView theme tokens** (added to `index.css` as a single frozen object `TV_THEME`):
```
layout.background: { type: 'solid', color: 'transparent' }
layout.textColor: '#a8b2c8'
grid.vertLines.color: 'rgba(255,255,255,0.025)'
grid.horzLines.color: 'rgba(255,255,255,0.035)'
crosshair.mode: CrosshairMode.Magnet
crosshair.vertLine.color: 'rgba(139,92,246,0.45)'
crosshair.horzLine.color: 'rgba(139,92,246,0.45)'
candles.upColor: '#10b981' / downColor: '#f43f5e'
candles.borderVisible: false, wickUpColor: '#10b981', wickDownColor: '#f43f5e'
areaSeries.topColor: 'rgba(139,92,246,0.24)' / bottomColor: 'rgba(139,92,246,0)'
lineSeries.color: '#8b5cf6', lineWidth: 2
```
No literal hex in the component body — all live in `TV_THEME`.

**Definition of done:** Expanding 10 rows in a row never leaks a chart instance (verified with devtools memory tab).

**Files:** new `src/web/frontend/src/components/SignalDetailPanel.tsx`; delete or retire `MiniChartPanel`.

---

## Story 4 — Signal annotations on the chart

**Outcome:** Past model signals become visible on the chart as buy/sell arrows, so a trader can audit the model at a glance.

**Acceptance criteria:**
- [ ] `SignalDetailPanel` calls a new endpoint (or reuses an existing one) to fetch per-symbol historical signal transitions with `{ date, signal, score }[]`.
  - If no endpoint yet: derive from the existing summary row's `last_signal_change_date` + today's `signal` as a minimum (two markers).
- [ ] Uses `seriesMarkers` API: `▲` emerald `below-bar` for buy transitions, `▼` rose `above-bar` for sell, `●` amber `in-bar` for exit.
- [ ] Marker density capped at 24 visible at once; more are clustered into a single `· N` pill.
- [ ] Hovering a marker shows a floating hairline card: `date · signal · momentum · crash risk` — all tabular-nums.
- [ ] A toggle `Show signals` in the subpanel header lets users hide the markers (persisted in `localStorage`).

**Files:** `SignalDetailPanel.tsx`; possibly new backend route if signal history not yet exposed.

---

## Story 5 — Horizon projection overlays

**Outcome:** The chart visualizes the model's forward-looking forecasts as translucent fan bands, echoing what `/charts` already does but inside the subpanel.

**Acceptance criteria:**
- [ ] `api.chartForecast(symbol)` is consumed and horizons 7 / 30 / 90 / 180 / 365 are rendered as right-extending `lineSeries` with `lineStyle: LineStyle.Dashed` and a translucent `areaSeries` fan between lower and upper bounds.
- [ ] Colors: mean line uses violet at 0.9 opacity; fan fills use violet at 0.08 opacity.
- [ ] A right-axis label shows the terminal price of each horizon, tabular-nums, 9.5 px `label-micro`.
- [ ] Toggle `Show forecasts` in the header (on by default); persisted in `localStorage` per-user.
- [ ] If `chartForecast` returns nothing: silently hide toggles, do not show an error.

**Files:** `SignalDetailPanel.tsx`.

---

## Story 6 — "Run Stocks" command rail

**Outcome:** Users can trigger the equivalent of `make stocks` directly from `/signals` with a gorgeous, reassuring component that shows every phase of the job.

**Acceptance criteria:**
- [ ] Floating **SignalsCommandBar** appears in the top-right of the hero band, aligned with the `ExportButton`. Primary button label: `Refresh Stocks` with a subtle violet→cyan sheen. 32 px rounded-full, paired with a kebab menu for advanced options.
- [ ] Advanced options (in a radius-14 glass-card popover): `Universe: All · Favorites · Failing PIT`, `Recompute signals: yes / no`, `Clear cache after: yes / no`.
- [ ] Clicking `Refresh Stocks`:
  1. Calls `api.triggerDataRefresh(symbols?)` → receives `{ task_id }`.
  2. Polls `api.taskStatus(task_id)` every 800 ms (react-query with `refetchInterval`).
  3. On success → calls `api.triggerSignals(args?)` → polls again.
  4. On success → calls `api.refreshSignalCache()` → invalidates `summary / sectors / strong / stats` queries via `queryClient.invalidateQueries`.
- [ ] While running, the button transforms into a progress pill: `Refreshing · Fetching prices · 38 / 490`. Uses an `svg` circular progress ring at 16 px for the spinner (no emoji, no lucide spinner).
- [ ] A collapsible `JobLogDrawer` appears at the bottom-right (like macOS activity monitor), 360 × 220 px, with a scrolling monospace feed of the task status events. Dismissible with `Esc` or × icon. Persisted collapsed/expanded state in `localStorage`.
- [ ] On error: the button turns rose; an inline hairline strip under the hero shows `Failed: <reason>  ·  Retry`. Clears on retry or after 10 s.
- [ ] On success: a one-shot 1.4 s emerald shimmer sweeps the hero band, the WS dot pulses, and a toast appears top-right: `N new signals · K changed since last run`.
- [ ] **Concurrency guard:** if a job is already running, the button is disabled and tooltip reads `Refresh in progress · opened <timeAgo>`.
- [ ] **Cancel:** while the job is running, the progress pill exposes a hairline × that calls `api.cancelTask(task_id)` (if the endpoint exists; otherwise the button is hidden). Confirmation is inline: `Cancel refresh?  · Yes  · Keep going` replacing the pill for 3 s before auto-reverting.
- [ ] **Accessibility:** drawer has `role="log"` `aria-live="polite"`; button has `aria-busy` while running. The cancel × has `aria-label="Cancel refresh"`.

**Files:** new `src/web/frontend/src/components/SignalsCommandBar.tsx` + `JobLogDrawer.tsx`; wire into `SignalsPage.tsx`.

---

## Story 7 — Keyboard-first navigation

**Outcome:** A power-user trader never needs the mouse.

**Acceptance criteria:**
- [ ] `/` focuses the search input.
- [ ] `j` / `k` move the row selection down / up (adds a focus ring — not the "expanded" state).
- [ ] `Enter` or `Space` expands / collapses the focused row's detail subpanel.
- [ ] `Shift+J` / `Shift+K` jumps by 10 rows.
- [ ] `g g` jumps to top, `G` jumps to bottom (vim-style).
- [ ] `1 / 2 / 3` switches view to All / Sectors / Strong.
- [ ] `f` cycles through filters (`All → Buy → Hold → Sell → All`).
- [ ] `?` opens a keyboard shortcut cheatsheet in a glass-card modal (radius 20, hairline border).
- [ ] Shortcuts must not fire when `input`, `textarea`, or `contenteditable` has focus.
- [ ] Focus ring uses the existing `.focus-ring` utility (violet 2 px outset, 2 px offset).
- [ ] When a row is expanded with the subpanel open, `j / k` still navigate; moving focus to another row keeps the previous panel closed and opens the new one (single-expand policy). Auto-scrolls the focused row into view with `scrollIntoView({ block: 'nearest', behavior: 'smooth' })`.

**Files:** `SignalsPage.tsx`, possibly new `useSignalsKeymap.ts` hook.

---

## Story 8 — Column density & customization

**Outcome:** Traders can tune the table to their screen and workflow.

**Acceptance criteria:**
- [ ] A gear icon (lucide `SlidersHorizontal`) in the toolbar opens a glass-card popover `Table Settings`.
- [ ] Toggles (persisted in `localStorage`):
  - Density: `Comfortable (40 px) · Compact (32 px) · Ultra (26 px)`.
  - Show columns: `Sector · Sparkline · Momentum · Crash · Each horizon`.
  - Show `Δ 1d %` column.
  - Number format: `Absolute · Δ since last refresh`.
- [ ] Changes animate (height transition only — no layout thrash).
- [ ] `Reset defaults` link at the bottom of the popover.

**Files:** new `src/web/frontend/src/components/SignalsTableSettings.tsx`.

---

## Story 9 — Responsive & mobile pass

**Outcome:** The page is usable on a 375 px-wide iPhone without a hamburger or horizontal scroll.

**Acceptance criteria:**
- [ ] Breakpoints: desktop ≥ 1280, tablet 768 – 1279, mobile < 768.
- [ ] Mobile:
  - Hero stacks (numeral on top, split bar below, stats become horizontal scroll snaps).
  - Toolbar collapses: search remains, view/filter collapse into a single `Filters` sheet (bottom sheet pattern, drags up with iOS-style 280 ms spring).
  - Table collapses to cards: `{ticker · signal dot · momentum · chevron}`. Tapping a card expands the chart subpanel inline, full-bleed, 320 px tall.
  - Command bar collapses into a floating action button bottom-right.
- [ ] All hover affordances have tap equivalents (no hover-only info).
- [ ] Touch targets ≥ 44 × 44 px.

**Files:** `SignalsPage.tsx` + CSS in `index.css`.

---

## Story 10 — Watchlist / pinned rows

**Outcome:** Traders pin the 5–15 tickers they actually care about to the top.

**Acceptance criteria:**
- [ ] A subtle `Star` icon appears on row hover (lucide, 14 px, `text-muted` → `amber-400` on click).
- [ ] Pinned symbols are persisted in `localStorage` as a string array.
- [ ] When any pins exist, a new segment `Watchlist` appears in the view toggle, to the left of `All`. It's the new default if `watchlist.length > 0`.
- [ ] Pinned rows in All / Sectors views are sticky to the top of their group with a hairline divider between pinned and unpinned.
- [ ] A `★ N` pill in the hero hero shows watchlist count. Clicking it switches to watchlist view.
- [ ] Cap at 30 items; show hairline warning at 25.

**Files:** `SignalsPage.tsx`, new `useWatchlist.ts` hook.

---

## Story 11 — Ticker tape polish

**Outcome:** The live ticker tape (if present) feels like a Bloomberg terminal, not a marquee scroller.

**Acceptance criteria:**
- [ ] Single-line, 32 px height, full-bleed below the toolbar.
- [ ] CSS-only marquee using `@keyframes` (no JS `setInterval`). `prefers-reduced-motion: reduce` freezes it.
- [ ] Each ticker: `{ticker · price tabular-nums · Δ% color dot · sparkline 40×12}`.
- [ ] Click a ticker → scrolls the table to that row and expands its subpanel.
- [ ] Pauses on hover.
- [ ] Hairline separators `rgba(255,255,255,0.035)` between ticker cells.

**Files:** `SignalsPage.tsx` or new `TickerTape.tsx`.

---

## Story 12 — Command Palette (⌘K)

**Outcome:** A single keyboard shortcut surfaces every jump point on `/signals`.

**Acceptance criteria:**
- [ ] `⌘K` (or `Ctrl K`) opens a centered glass-card modal, 520 × auto, radius 20, violet→cyan rim.
- [ ] Sections: `Jump to ticker`, `Change view`, `Change filter`, `Run command` (`Refresh stocks`, `Clear cache`, `Open tuning`, `Open charts`).
- [ ] Fuzzy search input at the top, auto-focused; results render in hairline rows with `label-micro` section headers.
- [ ] `↑ / ↓` navigates, `Enter` executes, `Esc` closes.
- [ ] Max 8 results per section, scrollable.
- [ ] Ticker results show dot + ticker + sector + current signal label.
- [ ] Executing `Jump to ticker` scrolls the table and auto-expands the subpanel.

**Files:** new `src/web/frontend/src/components/CommandPalette.tsx`; integrate in `SignalsPage.tsx`.

---

## Story 13 — Empty / loading / error state polish

**Outcome:** Every non-happy-path state looks intentional and beautiful.

**Acceptance criteria:**
- [ ] Loading: 6 skeleton rows using the existing `SignalTableSkeleton` but with the new 40 px row metrics.
- [ ] Empty (0 rows after filter): a single card with an icon glyph, title `No signals match`, body `Try clearing filters or running a refresh`, CTA `Clear filters`.
- [ ] API error: a hairline card with the error message in `text-secondary` and a `Retry` button. No stack traces visible by default; a `Show details` disclosure reveals them for devs.
- [ ] No WebSocket: a discreet hairline banner at the top of the hero `Live feed offline — showing last cached data from <time>` in amber; dismissible.

**Files:** `SignalsPage.tsx`, `CosmicEmptyState.tsx`, `CosmicErrorCard.tsx`.

---

## Story 14 — Perceived-performance & micro-interactions

**Outcome:** The page feels instant and alive.

**Acceptance criteria:**
- [ ] Row expand uses `content-visibility: auto` on collapsed rows (big gain on long tables).
- [ ] Sparklines render via `requestIdleCallback` in batches of 32, not all at once.
- [ ] All numeric cells use `tabular-nums`; no layout shift when WS pushes updates.
- [ ] A 1.4 s **emerald shimmer** sweeps any row whose signal changed since last refresh (uses the existing `aurora-upgrade` animation).
- [ ] Skeleton rows dissolve (opacity 1 → 0 over 220 ms) rather than pop.
- [ ] Respect `prefers-reduced-motion`: disable shimmer, marquee, and splitbar transitions.

**Files:** `SignalsPage.tsx`, `index.css`.

---

## Story 15 — Row virtualization

**Outcome:** 490 rows scroll at 60 fps even on low-end laptops.

**Acceptance criteria:**
- [ ] Introduce `@tanstack/react-virtual` (or `react-window`) behind a feature flag, default ON for `view === 'all'` and `rows.length > 150`.
- [ ] Overscan 8 rows above and below the viewport.
- [ ] Keyboard nav (Story 7) correctly scrolls the virtualized range into view.
- [ ] Expanded subpanel row is tall; the virtualizer handles variable-height rows (use `measureElement`).
- [ ] No visual change vs. the non-virtualized version.
- [ ] Disable virtualization when a screen reader is detected (`navigator.userAgent` / `matchMedia('(prefers-reduced-motion)')` is a poor proxy — use `@tanstack/react-virtual` + `aria-rowcount` so that row count is correctly exposed).

**Files:** `SignalsPage.tsx`; add dep in `package.json` if needed.

---

## Implementation order

1. **Story 3** — TradingView subpanel (biggest user win, already partly asked).
2. **Story 6** — Run-stocks command rail (explicit user ask).
3. **Story 2** — Unified toolbar.
4. **Story 12** — Command palette (rides on toolbar).
5. **Story 4** — Signal annotations.
6. **Story 5** — Horizon projections.
7. **Story 7** — Keyboard nav.
8. **Story 10** — Watchlist.
9. **Story 1** — Hero polish.
10. **Story 8** — Column density.
11. **Story 11** — Ticker tape.
12. **Story 13** — States.
13. **Story 14** — Micro-interactions.
14. **Story 15** — Virtualization.
15. **Story 9** — Responsive pass (final).

Each story lands as its own focused commit. Every commit must leave the dev build at `npx tsc --noEmit` → EXIT=0 and the production build (`npm run build`) green.

---

## Global definition of done

- `tsc --noEmit` clean.
- `npm run build` clean.
- No emojis anywhere in code, UI, or strings. Use lucide-react or inline SVG.
- No hard-coded HEX colors outside `index.css`; use CSS variables (`--accent-violet`, `--accent-cyan`, `--text-primary`, etc.).
- All new components have a `role` and `aria-label` where appropriate.
- `prefers-reduced-motion` honored on all animations.
- No new third-party deps beyond `lightweight-charts` (already in) and — only if needed — `@tanstack/react-virtual`.
