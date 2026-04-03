# UX.md -- Complete UX Redesign: Bayesian Signal Engine Dashboard

> Written by: Product Owner (Craftsman)
> Date: 3 April 2026
> Philosophy: Every pixel earns its place. Every interaction teaches.
> Guiding Principle: "Would a Bloomberg terminal user, a Figma designer,
> and a hedge fund PM all independently say 'this is the best dashboard
> I have ever used'? If not, iterate until they do."

---

## Design North Star

This is not a dashboard. This is a **decision cockpit** for professional
quantitative traders managing 100+ assets with Bayesian Model Averaging.

Every surface must satisfy three simultaneous demands:

1. **Information Density** -- A single glance communicates more than
   competitors convey in three clicks.
2. **Emotional Clarity** -- The user never wonders "is this good or bad?"
   Color, shape, and motion answer before the conscious mind asks.
3. **Muscle Memory** -- After one week of daily use, the user navigates
   entirely by keyboard and spatial memory. The mouse becomes optional.

---

## Design Language Specification

### Color System

| Token              | Value       | Usage                                      |
|--------------------|-------------|---------------------------------------------|
| `--bg-void`        | `#050510`   | Deepest background, page canvas             |
| `--bg-surface`     | `#0a0a1a`   | Card/panel surfaces                         |
| `--bg-raised`      | `#12122a`   | Elevated elements, modals, popovers         |
| `--border-subtle`  | `#ffffff08` | Card edges (barely visible)                 |
| `--border-focus`   | `#42A5F540` | Focus rings, active states                  |
| `--text-primary`   | `#f1f5f9`   | Headings, values                            |
| `--text-secondary` | `#94a3b8`   | Labels, descriptions                        |
| `--text-muted`     | `#475569`   | Timestamps, footnotes                       |
| `--accent-bull`    | `#00E676`   | Bullish signals, passing metrics            |
| `--accent-bear`    | `#FF1744`   | Bearish signals, failing metrics            |
| `--accent-warn`    | `#FFB300`   | Caution, stale data, elevated risk          |
| `--accent-info`    | `#42A5F5`   | Interactive elements, links, focus          |
| `--accent-purple`  | `#AB47BC`   | Rare highlights, premium indicators         |

### Typography Scale

| Token       | Size   | Weight | Tracking | Usage                      |
|-------------|--------|--------|----------|----------------------------|
| `display`   | 36px   | 700    | -0.02em  | Page-level hero numbers    |
| `heading-1` | 28px   | 600    | -0.01em  | Page titles                |
| `heading-2` | 20px   | 600    | -0.005em | Section headers            |
| `heading-3` | 15px   | 500    | 0        | Card titles                |
| `body`      | 13px   | 400    | 0.01em   | Primary readable text      |
| `caption`   | 11px   | 500    | 0.04em   | Labels, badges, metadata   |
| `mono`      | 12px   | 400    | 0.02em   | Numbers, code, tickers     |

### Spacing Scale

4px baseline grid: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80.

### Motion Principles

| Type         | Duration | Easing                         | Usage                    |
|--------------|----------|--------------------------------|--------------------------|
| `micro`      | 120ms    | `cubic-bezier(0.2, 0, 0, 1)`  | Hover, focus, toggle     |
| `standard`   | 250ms    | `cubic-bezier(0.2, 0, 0, 1)`  | Panel open, tab switch   |
| `expressive` | 400ms    | `cubic-bezier(0.16, 1, 0.3, 1)` | Page enter, modal reveal |
| `spring`     | 500ms    | `cubic-bezier(0.34, 1.56, 0.64, 1)` | Celebratory feedback |

### Elevation System

| Level | Shadow                                            | Usage              |
|-------|---------------------------------------------------|--------------------|
| 0     | none                                              | Flat inline content|
| 1     | `0 1px 2px rgba(0,0,0,0.3)`                      | Cards at rest      |
| 2     | `0 4px 12px rgba(0,0,0,0.4)`                     | Cards on hover     |
| 3     | `0 8px 32px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.04)` | Modals, drawers |

---

## Keyboard-First Navigation Specification

| Key             | Global Context               | Page Context                     |
|-----------------|------------------------------|----------------------------------|
| `1` - `9`       | Navigate to page by number   | Page-specific shortcuts          |
| `Cmd+K`         | Open Command Palette         | Focus search if visible          |
| `Cmd+Shift+S`   | Jump to Signals              | --                               |
| `Cmd+Shift+R`   | Jump to Risk                 | --                               |
| `Cmd+Shift+C`   | Jump to Charts               | --                               |
| `Esc`           | Close modal / deselect       | Collapse expanded rows           |
| `j` / `k`       | --                           | Navigate rows (Vim-style)        |
| `Enter`         | --                           | Expand selected row              |
| `Tab`           | Next focusable element       | Next column in table             |
| `/`             | Focus global search          | Focus page search                |
| `?`             | Open shortcut reference      | --                               |

---

# EPIC 1: Navigation Shell and Spatial Orientation

> **Vision**: The application shell is so spatially intuitive that a new user
> understands where they are, where they can go, and what changed since their
> last visit -- all within 2 seconds of any page load.

---

## Story 1.1: Intelligent Sidebar with Contextual Density

**As a** quantitative trader navigating between 10+ pages,
**I want** a sidebar that communicates page health at a glance without clicking,
**so that** I never waste a click visiting a page only to discover nothing changed.

### Acceptance Criteria

- [ ] AC-1: Each navigation item displays a **live micro-indicator** to the right
  of the label showing the page's current state:
  - Signals: count of Strong Buy + Strong Sell signals (e.g., "12")
  - Risk: temperature value with color dot (green/amber/red)
  - Charts: nothing (navigation-only page)
  - Tuning: PIT pass rate as percentage
  - Data: count of stale files (red if > 0)
  - Arena: count of models in safe storage
  - Services: green pulse if all OK, red pulse if any service down
  - Diagnostics: calibration failure count
- [ ] AC-2: Micro-indicators update via the same React Query cache as the pages
  themselves -- **zero additional API calls**.
- [ ] AC-3: Hovering a nav item reveals a **tooltip card** (elevation 3) showing
  a 3-line summary of that page's current state, with the last-updated timestamp.
  Tooltip appears after 400ms delay and animates with `expressive` motion.
- [ ] AC-4: The active page indicator is a subtle vertical gradient bar (4px wide)
  on the left edge of the nav item, using `--accent-info` with 60% opacity at top
  fading to 0% at bottom.
- [ ] AC-5: Sidebar collapses to icon-only mode (48px wide) on screens < 1280px.
  In collapsed mode, hovering an icon expands only that item with label + tooltip.
- [ ] AC-6: Collapsed/expanded preference persists in `localStorage`.
- [ ] AC-7: Keyboard shortcut `Cmd+B` toggles sidebar collapse.
- [ ] AC-8: The UX of contextual micro-indicators gives users enough ambient
  information that they would absolutely fall in love with the sidebar
  and drool over how much context is available without a single click.

---

## Story 1.2: Command Palette (Global Quick Navigation)

**As a** power user who values speed above all else,
**I want** a command palette accessible from any page via `Cmd+K`,
**so that** I can jump to any asset, page, or action in under 2 keystrokes.

### Acceptance Criteria

- [ ] AC-1: `Cmd+K` opens a centered modal overlay (max-width 560px) with a search
  input auto-focused. The overlay background dims to `rgba(0,0,0,0.7)` with
  `backdrop-filter: blur(8px)`. Animation: scale from 95% to 100% + fade, 250ms.
- [ ] AC-2: Search results appear as the user types (debounced 80ms) in categorized
  groups:
  - **Pages** (always visible): "Signals", "Risk Dashboard", "Charts", etc.
  - **Assets** (from signalSummary cache): ticker + sector + current signal badge
  - **Actions**: "Refresh Data", "Retune Models", "Compute Signals"
- [ ] AC-3: Each result row shows: icon (page/asset/action), label, right-aligned
  keyboard shortcut hint where applicable.
- [ ] AC-4: Arrow keys navigate results. `Enter` activates. `Esc` closes.
- [ ] AC-5: Selecting an asset navigates to `/charts/{SYMBOL}`.
- [ ] AC-6: Selecting an action triggers the corresponding API call and shows
  a toast notification with progress.
- [ ] AC-7: Most recently used items appear at the top of the list before typing.
  Recent items persist in `localStorage` (max 8).
- [ ] AC-8: The command palette feels so instantaneous and intelligent that users
  would absolutely fall in love with its responsiveness and drool over the
  speed of navigating the entire 100+ asset universe in milliseconds.

---

## Story 1.3: Breadcrumb Trail with Temporal Context

**As a** user drilling into a specific asset from the Signals page,
**I want** a breadcrumb trail that shows my navigation path and how fresh the data is,
**so that** I always know where I am and whether the data I'm looking at is current.

### Acceptance Criteria

- [ ] AC-1: A breadcrumb bar appears below the page header when the user is deeper
  than the top-level page (e.g., Charts > AAPL, Diagnostics > PIT Calibration).
- [ ] AC-2: Each breadcrumb segment is clickable and navigates back to that level.
- [ ] AC-3: The final breadcrumb segment (current context) displays in
  `--text-primary` with font-weight 600. Parent segments display in
  `--text-secondary`.
- [ ] AC-4: A **data freshness badge** appears at the right end of the breadcrumb
  bar showing the oldest data dependency on the current view:
  - Green badge "Live" if < 60 seconds old
  - Amber badge "2m ago" / "5m ago" etc. if stale
  - Red badge "Stale: 2h" with subtle pulse if critically stale
- [ ] AC-5: Clicking the freshness badge opens a dropdown showing all data
  dependencies for the current view with their individual ages and a
  "Refresh All" button.
- [ ] AC-6: Breadcrumbs animate in with a subtle left-to-right stagger (40ms per
  segment) using `standard` motion.

---

## Story 1.4: Toast Notification System

**As a** user who triggers background actions (refresh data, retune models),
**I want** non-blocking toast notifications that track progress,
**so that** I can continue working while background tasks complete.

### Acceptance Criteria

- [ ] AC-1: Toasts appear in the bottom-right corner, stacked vertically with
  8px gap between them. Maximum 4 visible simultaneously.
- [ ] AC-2: Four toast variants exist:
  - **Info**: Blue left-border, informational icon
  - **Success**: Green left-border, checkmark icon, auto-dismiss after 4s
  - **Warning**: Amber left-border, alert icon, auto-dismiss after 6s
  - **Error**: Red left-border, X icon, persists until manually dismissed
- [ ] AC-3: **Progress toasts** (for long-running tasks) show:
  - Title + description
  - Animated progress bar (indeterminate or percentage-based)
  - Elapsed time counter
  - Cancel button (if the backend supports cancellation)
- [ ] AC-4: Toasts animate in from the right (translate-x: 120% to 0) with
  `expressive` motion and animate out by fading + sliding down.
- [ ] AC-5: Each toast has an `aria-live="polite"` attribute for screen readers.
- [ ] AC-6: Clicking a toast with an associated page (e.g., "Tuning complete")
  navigates to that page.
- [ ] AC-7: Toast history is accessible via a bell icon in the sidebar footer.
  Clicking it opens a slide-out panel showing the last 20 notifications with
  timestamps.

---

## Story 1.5: Global Ambient Status Strip

**As a** trader who needs to feel the market's pulse without looking away from my work,
**I want** a thin ambient status strip along the top of the viewport,
**so that** system health and market regime are always peripherally visible.

### Acceptance Criteria

- [ ] AC-1: A 3px-tall strip spans the full width of the viewport at the very top,
  above the sidebar. Its color reflects the combined risk temperature:
  - Calm (< 0.3): `--accent-bull` at 40% opacity
  - Elevated (0.3-0.7): `--accent-warn` at 50% opacity
  - Stressed (0.7-1.2): `--accent-bear` at 40% opacity
  - Crisis (> 1.2): `--accent-bear` at 70% with a slow pulse animation (2s)
- [ ] AC-2: The strip transitions color smoothly over 1000ms when the regime changes.
- [ ] AC-3: Hovering the strip expands it to 32px height (250ms spring animation)
  showing: "Risk: 0.42 Elevated | 147 Assets | 12 Strong Buys | Services: OK"
- [ ] AC-4: The expanded strip text uses `caption` typography in `--text-secondary`.
- [ ] AC-5: This ambient strip is so subtle yet informative that users would
  absolutely fall in love with the peripheral awareness and drool over always
  knowing the market regime without a single interaction.

---

# EPIC 2: Dashboard (Overview Page) Reimagination

> **Vision**: The Overview page is not a landing page -- it is a **morning briefing**.
> A trader opens this at 9:25 AM and within 8 seconds knows: what changed overnight,
> what demands attention, and what the models are most confident about today.

---

## Story 2.1: Morning Briefing Hero Card

**As a** trader starting my day,
**I want** a hero section that immediately communicates what changed since my last visit,
**so that** I can prioritize my attention on new developments rather than re-scanning
familiar information.

### Acceptance Criteria

- [ ] AC-1: The top of the Overview page displays a **Briefing Card** spanning
  full width with three columns:
  - **Left**: "Since Last Visit" -- count of signal changes (upgrades, downgrades,
    new entries) with directional arrows. Shows the most impactful change first
    (e.g., "NVDA upgraded to Strong Buy"). "Last visit" tracked via `localStorage`
    timestamp comparison against `signals.computed_at`.
  - **Center**: "Today's Conviction" -- the single highest-conviction signal across
    all assets with: ticker, signal direction, expected return, probability. Large
    typography (`display` size for the ticker, `heading-2` for the return).
  - **Right**: "System Pulse" -- 4 micro-gauges in a 2x2 grid:
    - Risk Temperature (circular gauge, 0-2 scale)
    - PIT Pass Rate (circular gauge, 0-100%)
    - Data Freshness (circular gauge, green/amber/red)
    - Asset Coverage (circular gauge, tuned/total)
- [ ] AC-2: The briefing card background uses a subtle gradient that shifts based on
  overall market sentiment: slightly green tint if majority bullish, slightly red
  if majority bearish, neutral gray if balanced.
- [ ] AC-3: Each micro-gauge is a 48px SVG circle with a colored arc stroke.
  The arc animates from 0 to the target value over 600ms on page load using
  `expressive` motion.
- [ ] AC-4: The "Since Last Visit" section shows "Welcome back" on first visit
  (no `localStorage` timestamp) and "All caught up" if nothing changed.
- [ ] AC-5: The entire briefing card animates in with `fade-up` at page load.
- [ ] AC-6: The morning briefing hero must be so information-rich yet clean that
  users would absolutely fall in love with starting their day here and drool
  over how much they learn in under 8 seconds.

---

## Story 2.2: Signal Distribution Radar

**As a** portfolio manager monitoring signal balance,
**I want** a rich signal distribution visualization that shows both counts
and directional momentum,
**so that** I can see not just what the distribution is, but how it is shifting.

### Acceptance Criteria

- [ ] AC-1: Replace the current donut chart with a **stacked horizontal bar**
  spanning the full card width, showing Strong Sell | Sell | Hold | Buy | Strong Buy
  in a continuous gradient strip. Each segment width is proportional to count.
- [ ] AC-2: Below the bar, a **sparkline row** shows how the distribution has shifted
  over the last 7 days (one mini data point per day). This requires storing
  historical snapshot data in `localStorage` or an API endpoint.
- [ ] AC-3: Hovering any segment of the bar highlights it with a subtle glow and
  shows a tooltip: count, percentage, and list of top 3 tickers in that category.
- [ ] AC-4: Below the visualization, a single-line sentence summarizes the shift:
  "Distribution shifted +4% bullish over 7 days" or "Stable this week" in
  `body` typography with appropriate directional color.
- [ ] AC-5: The stacked bar segments animate in from center-out on page load,
  expanding to their natural widths over 400ms with `expressive` motion.
- [ ] AC-6: Clicking any segment filters the Signal Heatmap below to show only
  assets in that signal category.

---

## Story 2.3: Living Signal Heatmap with Drill-Through

**As a** trader scanning 100+ assets simultaneously,
**I want** the signal heatmap to be an interactive exploration surface,
**so that** I can spot anomalies, drill into sectors, and navigate to charts
without leaving the Overview page.

### Acceptance Criteria

- [ ] AC-1: The heatmap renders a matrix of **asset rows x horizon columns** with
  cells colored using a perceptually uniform diverging colormap:
  - Bearish: `--accent-bear` (opacity mapped to magnitude)
  - Neutral: transparent
  - Bullish: `--accent-bull` (opacity mapped to magnitude)
- [ ] AC-2: Sectors are collapsible groups. Each sector header row shows:
  - Sector name
  - Asset count
  - A **sector sentiment bar** (mini stacked bar: bull/neutral/bear proportions)
  - Expand/collapse chevron
- [ ] AC-3: Hovering a cell shows a **rich tooltip** (elevation 3) containing:
  - Ticker + Horizon label
  - Expected return (% with sign)
  - Probability (p_up)
  - Kelly fraction
  - Signal label with colored badge
  - A 30-day mini sparkline of that asset's price
- [ ] AC-4: Clicking a cell navigates to `/charts/{SYMBOL}` with the chart
  auto-scrolled to the relevant time range.
- [ ] AC-5: Keyboard navigation: `j/k` moves between rows, `h/l` moves between
  columns, `Enter` clicks the cell, `Esc` deselects.
- [ ] AC-6: A **color scale legend** appears in the top-right of the heatmap
  showing the gradient from -10% to +10% with labeled ticks.
- [ ] AC-7: Sector groups remember their collapsed/expanded state in `localStorage`.
- [ ] AC-8: The heatmap renders smoothly with 150+ assets via virtualized rows.
  Only visible rows are in the DOM; scrolling lazy-loads additional rows.
- [ ] AC-9: The heatmap experience is so fluid, information-dense, and satisfying
  to explore that users would absolutely fall in love with scanning their
  entire portfolio and drool over the instant visual pattern recognition.

---

## Story 2.4: Model Confidence Leaderboard

**As a** quant researcher monitoring model performance,
**I want** the model distribution chart to show not just counts but confidence
and calibration quality,
**so that** I understand which models the system trusts most and why.

### Acceptance Criteria

- [ ] AC-1: Replace the current horizontal bar chart with a **leaderboard table**
  showing top 10 models, ranked by BMA selection frequency:
  - Rank number (1-10) with medal icons for top 3 (gold/silver/bronze SVG)
  - Model name (truncated intelligently, full name on hover)
  - Selection count (with bar visualization behind the number)
  - Average BMA weight (as percentage)
  - Average PIT pass rate (colored badge: green >= 80%, amber >= 60%, red < 60%)
- [ ] AC-2: Each row has a subtle hover state (bg shift + elevation 1).
- [ ] AC-3: Clicking a model row opens an inline expansion showing:
  - Which assets selected this model (list of ticker badges)
  - Average BIC, CRPS, Hyvarinen scores for this model
- [ ] AC-4: A "View All Models" link navigates to Diagnostics > Model Comparison.
- [ ] AC-5: The leaderboard animates in row-by-row with 50ms stagger using
  `standard` motion, giving a satisfying cascade effect.

---

## Story 2.5: Top Movers and Conviction Spotlight

**As a** trader looking for actionable opportunities,
**I want** a prominently featured section showing the highest-conviction signals
with enough context to act immediately,
**so that** I can identify and act on the best opportunities in seconds.

### Acceptance Criteria

- [ ] AC-1: A "Conviction Spotlight" section appears after the stat cards,
  showing two side-by-side panels: **Strongest Buys** (left, green accent)
  and **Strongest Sells** (right, red accent).
- [ ] AC-2: Each panel shows up to 5 assets as **rich cards** (not just list rows):
  - Ticker (large, `heading-3` size) + Sector badge
  - 60-day mini price sparkline (48px tall, using cached OHLCV data)
  - Expected return for best horizon (large, colored)
  - Probability (p_up / p_down)
  - Kelly fraction with a mini gauge bar
  - Signal age ("2h ago", "today")
- [ ] AC-3: Each asset card is clickable and navigates to `/charts/{SYMBOL}`.
- [ ] AC-4: The two panels have a subtle ambient glow matching their accent color
  (`--accent-bull` and `--accent-bear` respectively, at 5% opacity background).
- [ ] AC-5: If no high-conviction signals exist, the panel shows an elegant empty
  state: "No strong signals today. Markets in equilibrium." with a balanced
  scales icon.
- [ ] AC-6: The conviction spotlight must be so visually striking and actionable
  that users would absolutely fall in love with the zero-click path to their
  best trades and drool over the immediate actionability.

---

# EPIC 3: Signals Page -- The Decision Table

> **Vision**: The Signals page is the trader's primary workspace. It must feel like
> a Bloomberg terminal reimagined by Apple: dense with data, yet every element has
> breathing room. Sorting, filtering, and scanning 100+ assets must feel effortless.

---

## Story 3.1: Signal Table with Inline Micro-Charts

**As a** trader scanning signal recommendations across my asset universe,
**I want** each asset row to include a micro price chart and visual signal strength,
**so that** I can make faster visual comparisons without clicking into individual charts.

### Acceptance Criteria

- [ ] AC-1: Each asset row in the All Assets table includes a **60px-wide sparkline**
  column showing the 30-day price movement. The sparkline uses a single-color
  line: green if the asset is above its 20-day SMA, red if below.
- [ ] AC-2: The signal column displays a **gradient strength bar** (40px wide, 8px tall)
  next to the signal text label. The bar's fill percentage represents the
  model's confidence (composite of p_up and Kelly). Color: green for buy signals,
  red for sell signals, gray for hold.
- [ ] AC-3: Momentum score displays as a **colored numeric badge** with background
  opacity proportional to magnitude: stronger momentum = deeper background color.
  Positive momentum = green tones, negative = red tones.
- [ ] AC-4: Crash risk score displays as a **heat indicator**: a 4-segment bar
  (like a battery) where more filled segments = higher risk. Colors transition
  from green (1 segment) through amber (2-3) to red (4).
- [ ] AC-5: Horizon columns show expected return with a **directional micro-arrow**
  (SVG, 8px) and conditional coloring. Returns > +5% get bold weight.
  Returns < -5% get bold weight.
- [ ] AC-6: The table header row is sticky on scroll with a subtle bottom shadow
  that appears only when scrolled (elevation 1, fading in over 100ms).
- [ ] AC-7: Row hover state: entire row gets a subtle `--bg-raised` background
  with 120ms transition. The sparkline in the hovered row gains a tooltip
  showing: current price, 30-day change %, and volume trend.
- [ ] AC-8: The table is so information-dense yet readable that users would
  absolutely fall in love with the ability to scan 100+ assets and drool
  over how much decision-support data fits cleanly in each row.

---

## Story 3.2: Multi-Axis Sort with Visual Priority Indicators

**As a** trader who sorts by different criteria depending on my objective,
**I want** the ability to sort by multiple columns simultaneously with clear
visual indicators of sort priority,
**so that** I can create compound rankings like "highest momentum among strong buys."

### Acceptance Criteria

- [ ] AC-1: Clicking a column header sets it as the **primary sort**. A directional
  arrow appears next to the header text (up for ascending, down for descending).
- [ ] AC-2: Holding `Shift` and clicking a second column adds it as a **secondary
  sort**. A smaller "2" badge appears on the secondary sort column header.
  Up to 3 sort levels supported.
- [ ] AC-3: Sort indicators show: a numbered badge (1, 2, 3) and directional
  arrow for each active sort column. Active sort columns have `--accent-info`
  colored text.
- [ ] AC-4: Clicking an already-sorted column toggles direction (asc/desc).
  Clicking it a third time removes that sort level.
- [ ] AC-5: A "Sort indicator bar" appears above the table showing the active
  sort chain in plain language: "Sorted by Signal (desc), then Momentum (desc)"
  with small click-to-remove X buttons on each criterion.
- [ ] AC-6: Sort state persists in `localStorage` per view mode.
- [ ] AC-7: `Shift+Click` sorting reorders rows with a subtle 200ms animation
  where rows slide to their new positions rather than instantly jumping.

---

## Story 3.3: Sector Panel Redesign with Aggregate Intelligence

**As a** trader who thinks in sector allocations,
**I want** sector panels that show aggregate statistics and visual portfolio weight,
**so that** sector-level patterns and concentrations are immediately visible.

### Acceptance Criteria

- [ ] AC-1: Each sector panel header displays in a single row:
  - Sector name (font-weight 600)
  - Asset count badge
  - **Sector sentiment bar**: a thin horizontal stacked bar (120px wide, 4px tall)
    showing the proportion of Strong Buy / Buy / Hold / Sell / Strong Sell in
    the sector using the 5-color scale. This makes sector conviction instantly
    visual.
  - Average momentum score (colored number)
  - Average expected return (colored number)
  - Expand/collapse chevron (animated rotation, 200ms)
- [ ] AC-2: When a sector panel is collapsed, a **peek row** shows the top
  performing asset as a single teaser line: "Best: NVDA +8.2% (Strong Buy)"
  in `caption` typography. This gives value even without expanding.
- [ ] AC-3: Sector panels sort by aggregate momentum by default. A dropdown
  in the section header allows sorting sectors by: Momentum, Expected Return,
  Signal Strength, Asset Count, or Alphabetical.
- [ ] AC-4: The expand/collapse animation is a smooth height transition (250ms,
  `standard` motion) with content fading in after the height settles.
- [ ] AC-5: Sector panels include a subtle left border (2px) colored by the
  sector's dominant signal: green if majority bullish, red if majority bearish,
  gray if mixed.
- [ ] AC-6: The sector panel design summarizes so much information in the
  collapsed state that users would absolutely fall in love with the zero-click
  sector overview and drool over the aggregate intelligence embedded in
  each header row.

---

## Story 3.4: Real-Time Signal Flash with Change Context

**As a** trader monitoring signals via WebSocket,
**I want** signal changes to be visually prominent with before/after context,
**so that** I notice every upgrade and downgrade as it happens and understand
the magnitude of the change.

### Acceptance Criteria

- [ ] AC-1: When a signal changes via WebSocket update, the affected row flashes
  with a colored border animation:
  - Upgrade (e.g., Hold -> Buy): green border pulse (2 cycles, 600ms each)
  - Downgrade (e.g., Buy -> Hold): red border pulse (2 cycles, 600ms each)
  - New entry: blue border pulse (single cycle)
- [ ] AC-2: The changed cell shows a **transition badge** for 10 seconds after
  the change: a small tag showing "was Hold" or "was 4.2%" in `--text-muted`
  with strikethrough styling, positioned above the new value.
- [ ] AC-3: A **change counter badge** appears in the page header showing the
  count of changes since the page was opened: "3 changes" in a subtle pill.
  Clicking it scrolls to the most recent change.
- [ ] AC-4: An optional "Live Feed" toggle in the toolbar enables a **ticker tape**
  at the top of the signal table: a horizontally scrolling strip showing the
  last 10 signal changes as compact items: "NVDA: Hold->Buy | TSLA: Sell->Hold".
- [ ] AC-5: The flash animation uses a CSS keyframe that transitions
  `box-shadow` from `0 0 0 2px ${color}` to transparent, rather than
  background color changes, to avoid readability issues during the flash.
- [ ] AC-6: Flash animations are disabled when the browser tab is not visible
  (`document.hidden`) to avoid a jarring burst of animations when returning.

---

## Story 3.5: Smart Search with Fuzzy Matching and Filters

**As a** trader searching for specific assets in a universe of 100+,
**I want** search to be fast, fuzzy, and combinable with active filters,
**so that** I can find any asset configuration in under 2 seconds.

### Acceptance Criteria

- [ ] AC-1: The search input supports fuzzy matching: "nvd" matches "NVDA",
  "apl" matches "AAPL", "cro" matches "CRWD" and "CRM" and "Crowdstrike".
- [ ] AC-2: Search matches highlight with `--accent-info` background (20% opacity)
  on the matching characters in the result rows.
- [ ] AC-3: Search works simultaneously with signal filter and view mode.
  For example: searching "gold" while filter is "Strong Buy" shows only
  gold-related assets with strong buy signals.
- [ ] AC-4: The search input shows a live result count: "12 of 147 assets"
  updating as the user types.
- [ ] AC-5: `Cmd+K` or `/` focuses the search input from anywhere on the page.
  `Esc` clears the search and blurs the input. These shortcuts are shown as
  faded hint text inside the input when empty.
- [ ] AC-6: The search is debounced at 100ms for responsiveness.
- [ ] AC-7: An "X" clear button appears when the input has text, positioned
  inside the input on the right side.

---

## Story 3.6: Horizon Column Smart Density

**As a** trader viewing signals on different screen sizes,
**I want** horizon columns to intelligently adapt to available width,
**so that** I get maximum data density without horizontal scrolling.

### Acceptance Criteria

- [ ] AC-1: The table detects available viewport width and displays the maximum
  number of horizon columns that fit without scrolling:
  - >= 1600px: All horizons (1D, 3D, 7D, 30D, 90D, 180D, 365D)
  - >= 1280px: 5 horizons (7D, 30D, 90D, 180D, 365D)
  - >= 1024px: 3 horizons (7D, 30D, 365D)
  - < 1024px: 1 horizon (30D) with a "..." button to expand others
- [ ] AC-2: A **horizon selector** appears above the table allowing the user
  to choose which horizons to display. Selected horizons save to `localStorage`.
  This overrides the auto-fit behavior.
- [ ] AC-3: Each horizon column cell shows the expected return as the primary
  number and probability as a subtle sub-line (8px, `--text-muted`).
- [ ] AC-4: Hovering a horizon cell shows a rich tooltip with:
  - Expected return (large, colored)
  - Probability (with visual bar)
  - Kelly fraction
  - Upper/Lower uncertainty envelope
  - Signal classification for that specific horizon
- [ ] AC-5: The horizon density adaptation is so seamless that users would
  absolutely fall in love with always seeing the perfect amount of data and
  drool over never needing to horizontally scroll.

---

# EPIC 4: Charts Page -- The Analysis Theater

> **Vision**: The Charts page is where decisions crystallize. It must combine the
> technical analysis power of TradingView with the Bayesian model overlay that is
> our unique advantage. The chart is not just price history -- it is price history
> annotated with probability.

---

## Story 4.1: Chart Area with Probability Overlay Layer

**As a** trader analyzing a specific asset,
**I want** the chart to overlay forecast confidence intervals directly on the
price chart as shaded probability regions,
**so that** I can visually see where the model expects price to go and how
uncertain it is.

### Acceptance Criteria

- [ ] AC-1: When forecast data is loaded, a **probability cone** renders on the
  chart extending from the last price candle into the future:
  - Median forecast: solid line (2px, `--accent-info`)
  - 50% confidence interval: shaded region (20% opacity, `--accent-info`)
  - 90% confidence interval: shaded region (8% opacity, `--accent-info`)
- [ ] AC-2: The probability cone extends to the farthest available forecast
  horizon (365 days) but only shows detail for visible horizons based on zoom.
- [ ] AC-3: Hovering a point within the forecast cone shows a tooltip with:
  - Date
  - Median expected price
  - 50% CI range (low-high)
  - 90% CI range (low-high)
  - Probability of being above current price at that date
- [ ] AC-4: The forecast cone animates in when first loaded: the shaded regions
  grow from left (today) to right (future) over 600ms with `expressive` motion.
- [ ] AC-5: Forecast overlay can be toggled independently of other overlays.
  Its toggle button shows a small probability icon.
- [ ] AC-6: If the current price is outside the 90% CI at any past forecast point
  (a "surprise"), that region highlights with a subtle red/green tint to indicate
  the model was surprised.
- [ ] AC-7: The probability overlay gives the chart such a unique analytical edge
  that users would absolutely fall in love with seeing the future probability
  landscape and drool over the visual integration of Bayesian uncertainty.

---

## Story 4.2: Asset Detail Sidebar with Signal Summary

**As a** trader viewing a chart,
**I want** a detail sidebar that shows the full Bayesian signal intelligence
for the charted asset without navigating away,
**so that** chart analysis and signal analysis happen in a unified context.

### Acceptance Criteria

- [ ] AC-1: When a symbol is selected for charting, a **detail sidebar** appears
  to the right of the chart (320px wide, resizable), containing:
  - **Header**: Ticker (large), sector badge, current price, daily change
  - **Signal Badge**: Current signal (Strong Buy / Buy / Hold / Sell / Strong Sell)
    with background matching the signal color, full-width across the sidebar
  - **Horizon Table**: All available horizons with columns:
    Expected Return, Probability, Kelly, Signal per horizon
  - **Model Info**: Best model name, BMA weight, PIT status badge
  - **Risk Metrics**: Momentum score (with trend arrow), Crash risk (heat bar),
    Regime classification
- [ ] AC-2: The detail sidebar is collapsible via a handle/button. When collapsed,
  a thin strip (32px) remains with the ticker name and signal badge.
  Collapse preference persists in `localStorage`.
- [ ] AC-3: Horizon rows in the detail sidebar are clickable. Clicking a horizon
  row draws a horizontal reference line on the chart at the expected price
  level for that horizon.
- [ ] AC-4: The sidebar scrolls independently of the chart.
- [ ] AC-5: On screens narrower than 1280px, the sidebar starts collapsed and
  opens as an overlay on top of the chart with a semi-transparent backdrop.
- [ ] AC-6: The sidebar animates in from the right (translate-x from 100% to 0)
  with `standard` motion when a new symbol is selected.
- [ ] AC-7: A "View All Signals" link at the bottom of the sidebar navigates to
  the Signals page with the current asset pre-searched.

---

## Story 4.3: Chart Toolbar with Grouped Overlay Controls

**As a** technical analyst toggling multiple chart overlays,
**I want** overlay controls grouped by category with visual state indicators,
**so that** I can see at a glance which overlays are active and quickly toggle
combinations.

### Acceptance Criteria

- [ ] AC-1: The chart toolbar displays overlay toggles in grouped segments:
  - **Trend**: SMA 20, SMA 50, SMA 200 (each with its color indicator dot)
  - **Volatility**: Bollinger Bands, RSI 14
  - **Forecast**: Median, CI Upper, CI Lower, Probability Cone
  - **Misc**: Price Line
- [ ] AC-2: Each toggle button shows:
  - A 6px colored dot on the left matching the overlay color on the chart
  - Label text
  - Keyboard shortcut hint on the right in `--text-muted`
  - Active state: filled background at 10% of the overlay color
  - Inactive state: transparent background, muted text
- [ ] AC-3: Group labels appear as tiny section headers (`caption` typography)
  above each group.
- [ ] AC-4: A **"Presets"** dropdown allows saving and loading overlay combinations:
  - "Technical" preset: SMA 20 + SMA 50 + SMA 200 + Bollinger
  - "Forecast" preset: Forecast Median + CI + Probability Cone
  - "Clean" preset: Price only (all overlays off)
  - Users can save custom presets (stored in `localStorage`, max 5)
- [ ] AC-5: The toolbar adapts to width: on narrow screens, groups collapse into
  a single dropdown menu.
- [ ] AC-6: Toggle transitions animate the overlay in/out on the chart (fade over
  200ms) rather than appearing/disappearing instantly.

---

## Story 4.4: Symbol Picker with Rich Preview Cards

**As a** trader browsing assets to chart,
**I want** the symbol picker sidebar to show enough information per asset to
help me choose which one to analyze next,
**so that** I don't waste time charting assets that aren't interesting right now.

### Acceptance Criteria

- [ ] AC-1: In the All view, each asset in the sidebar list shows a **mini card**
  with:
  - Ticker (bold) + Sector (small, muted)
  - Signal badge (colored pill: SB/B/H/S/SS)
  - 30-day sparkline (48px wide, colored by trend)
  - Daily change percentage (colored green/red/gray)
- [ ] AC-2: In the Sectors view, sector headers show:
  - Sector name + count badge
  - Mini sentiment bar (proportional bull/bear split)
  - Expand/collapse with smooth height animation
- [ ] AC-3: In the Ranked views (Momentum, Edge, Return, etc.), each asset card
  additionally shows:
  - The ranked metric's value (prominently)
  - A horizontal bar behind the metric showing relative magnitude (normalized
    against the top-ranked asset = 100%)
  - Rank number (#1, #2, #3... with colored badges for top 3)
- [ ] AC-4: The symbol picker search supports the same fuzzy matching as Story 3.5.
- [ ] AC-5: Clicking an asset in the picker transitions the chart with a brief
  crossfade (150ms), rather than a jarring full reload. The chart container
  shows a shimmer loading state during data fetch.
- [ ] AC-6: The currently selected asset in the picker has a left accent bar
  (4px, `--accent-info`) and a subtle raised background.
- [ ] AC-7: The symbol picker is so content-rich that users would absolutely fall
  in love with browsing assets and drool over making informed chart selections
  without ever needing to leave the sidebar.

---

## Story 4.5: Time Range Selector with Zoom Memory

**As a** trader who analyzes different horizons for different assets,
**I want** time range selection to be fluid with a visual timeline scrubber,
**so that** I can zoom into any period quickly and the chart remembers my
preferred zoom per asset.

### Acceptance Criteria

- [ ] AC-1: The time range selector displays as a row of pill buttons:
  1W, 1M, 3M, 6M, 1Y, 2Y, ALL. Active pill has filled background with
  `--accent-info` at 15% opacity and colored text.
- [ ] AC-2: Below the pill row, a **mini overview chart** (32px tall, full width)
  shows the complete price history with a **draggable range selector handle**.
  The selected range is highlighted with a semi-transparent overlay. Dragging
  the edges resizes the view. Dragging the center pans.
- [ ] AC-3: Pinch-to-zoom on trackpad / scroll-wheel zoom is supported on the
  main chart. Zooming in the main chart updates the range selector handle.
- [ ] AC-4: The chart stores the last-used time range per symbol in `localStorage`.
  When returning to a previously viewed symbol, the chart restores that range.
- [ ] AC-5: Double-clicking the range selector resets to the "6M" default.
- [ ] AC-6: Time range transitions animate smoothly: the chart x-axis rescales
  with 300ms ease rather than jumping.

---

## Story 4.6: Chart Annotations and Personal Notes

**As a** trader who marks support/resistance levels and makes chart notes,
**I want** the ability to draw horizontal lines and attach notes to price levels,
**so that** my analysis persists across sessions and helps me make better decisions.

### Acceptance Criteria

- [ ] AC-1: A "Draw" mode toggle in the toolbar enables annotation tools:
  - **Horizontal Line**: Click a price level to place a horizontal line.
    Color picker (5 preset colors) and line style (solid/dashed) options.
  - **Note**: `Shift+Click` a candle to place a note marker. A text input
    appears anchored to that point. Max 140 characters.
- [ ] AC-2: Annotations persist in `localStorage` per symbol. Loading a chart
  restores all saved annotations.
- [ ] AC-3: Annotations are editable: double-click to edit text/color.
  Right-click (context menu) to delete.
- [ ] AC-4: Each annotation shows author timestamp ("Added 2 days ago") on hover.
- [ ] AC-5: An "Export Annotations" button in the toolbar saves all annotations
  for the current symbol as JSON (for backup/sharing).
- [ ] AC-6: Maximum 20 annotations per symbol to prevent clutter.
- [ ] AC-7: Annotations do not interfere with chart interactions (hover, zoom, pan)
  -- they are on a separate interaction layer.

---

# EPIC 5: Risk Dashboard -- The Risk Nervous System

> **Vision**: The Risk page is the body's nervous system. It processes signals from
> every corner of the portfolio and synthesizes them into a unified sensation: calm,
> alert, or pain. A trader should feel the risk regime in their gut after 3 seconds
> on this page.

---

## Story 5.1: Temperature Gauge with Regime History

**As a** risk-conscious trader,
**I want** the temperature gauge to show not just the current level but the recent
trajectory and regime transitions,
**so that** I understand whether risk is increasing, decreasing, or oscillating.

### Acceptance Criteria

- [ ] AC-1: Replace the current simple progress bar with a **circular gauge**
  (160px diameter) rendered as an SVG arc:
  - 270-degree arc from bottom-left to bottom-right
  - Gradient fill: green at 0 degrees -> amber at 135 -> red at 270
  - A needle indicator pointing to the current temperature value
  - The current value displayed in `display` typography in the center
  - The status label ("Calm" / "Elevated" / "Stressed" / "Crisis") below the value
- [ ] AC-2: Below the gauge, a **7-day sparkline** (120px wide, 24px tall) shows
  the temperature history. This requires either a new API endpoint or `localStorage`
  snapshots. The sparkline is colored using the same gradient as the gauge.
- [ ] AC-3: **Regime transition markers** appear as vertical dotted lines on the
  sparkline where the regime changed (e.g., from Calm to Elevated). Hovering
  a marker shows the transition: "Calm -> Elevated at 2:30 PM Mar 31".
- [ ] AC-4: An arrow icon next to the gauge needle indicates trend direction:
  - Rising risk (temperature increasing over last 3 data points): red up arrow
  - Falling risk: green down arrow
  - Stable: no arrow
- [ ] AC-5: The gauge needle animates from 0 to the current value on page load
  over 800ms with `spring` motion, creating a satisfying mechanical feel.
- [ ] AC-6: The gauge experience must be so visceral that users would absolutely
  fall in love with feeling the market's risk temperature and drool over the
  combination of current state, trend, and history in one compact instrument.

---

## Story 5.2: Cross-Asset Stress Matrix with Contagion Visualization

**As a** portfolio manager monitoring cross-asset risk,
**I want** to see which asset classes are under stress and how stress is
propagating between them,
**so that** I can identify systemic risk before it cascades across my portfolio.

### Acceptance Criteria

- [ ] AC-1: The Cross-Asset Stress tab displays a **4x4 correlation matrix**
  showing stress contagion between: FX Carry, Equities, Duration, Commodities.
  Cell color intensity represents correlation strength. Diagonal cells show
  the individual stress level.
- [ ] AC-2: Each stress category has an **expandable detail card** containing:
  - Category stress score (large, colored number)
  - Individual indicator table (current value, threshold, status badge)
  - A **contribution bar**: a stacked horizontal bar showing each indicator's
    contribution to the category stress as a percentage
- [ ] AC-3: An animated **stress flow diagram** (optional, toggle-able) shows
  directional arrows between the 4 categories where the arrow thickness
  represents correlation strength and color represents whether the correlation
  is stress-amplifying (red) or diversifying (green).
- [ ] AC-4: Category cards arrange in a responsive grid: 2x2 on desktop,
  single column on mobile.
- [ ] AC-5: Each indicator row in the expanded card has a colored status pip:
  green (below threshold), amber (near threshold), red (above threshold).
  Hovering the pip shows the exact threshold value.
- [ ] AC-6: The stress matrix is so visually intuitive that portfolio contagion
  effects are obvious at a glance, causing users to absolutely fall in love
  with systemic risk visualization and drool over the clarity.

---

## Story 5.3: Metals Risk Console with Multi-Horizon Forecast Strip

**As a** metals trader monitoring gold, silver, copper, and palladium,
**I want** a dedicated metals console that visualizes forecasts across all horizons
with momentum and stress context,
**so that** I can compare metals on a unified visual surface.

### Acceptance Criteria

- [ ] AC-1: Each metal displays as a **rich card** (not a table row) containing:
  - Metal name + icon (symbolic SVG: gold bar, silver coin, etc.)
  - Current price (large, `heading-2` typography) + daily change (colored)
  - **Forecast strip**: a horizontal row of 5 horizon blocks (7D, 30D, 90D, 180D,
    365D), each showing the forecast return as a colored cell (green positive,
    red negative, intensity proportional to magnitude). This creates a visual
    "forecast fingerprint" unique to each metal.
  - Momentum indicator: horizontal bar with directional arrow
  - Stress score: heat bar (4-segment battery style)
  - Confidence: circular mini-gauge (24px)
- [ ] AC-2: Cards arrange in a 2x2 grid on desktop (4+ metals) with consistent
  card height via CSS grid `auto-rows: 1fr`.
- [ ] AC-3: Hovering a forecast cell in the strip expands it (scale 1.2x, 150ms)
  and shows a tooltip with: exact forecast %, confidence interval, and a sentence
  like "Gold expected +2.3% over 30 days (68% CI: -0.5% to +5.1%)".
- [ ] AC-4: A **comparison mode** toggle switches the view to a single table where
  each row is a metal and each column is a horizon, enabling direct cross-metal
  comparison. The table cells are heat-colored.
- [ ] AC-5: Each metal card border-left color indicates the dominant forecast
  direction across all horizons: green if majority positive, red if majority
  negative, amber if mixed.
- [ ] AC-6: Metal cards animate in with a 100ms stagger cascade on tab load.

---

## Story 5.4: Market Breadth and Correlation Stress Dashboard

**As a** market observer monitoring broad health,
**I want** a combined breadth + correlation view that shows both dispersion
and clustering in one visual surface,
**so that** I can identify whether the market is healthy-dispersed or
dangerously-correlated.

### Acceptance Criteria

- [ ] AC-1: A **breadth gauge** displays as two semi-circle arcs side by side:
  - Left arc (green): count of UP assets, sized proportionally
  - Right arc (red): count of DOWN assets, sized proportionally
  - Center: ratio text "87 / 60" in `heading-2` typography
  - Below: percentage bars for each side
- [ ] AC-2: A **correlation stress card** shows:
  - Correlation stress score (large number, colored)
  - A statement: "Markets are loosely correlated" (green) or "Dangerous
    correlation clustering detected" (red pulsing text)
  - Average cross-asset correlation as a number
- [ ] AC-3: The universe indicators table shows tracked instruments (SPY, DXY,
  VIX, TLT, etc.) as **pill cards** in a flowing grid, each showing:
  - Instrument name
  - Price with directional arrow
  - Daily change (colored)
  - Multi-horizon forecast preview (3 colored dots for 7D/30D/90D direction)
- [ ] AC-4: VIX specifically gets special treatment: if VIX > 25, its card gains
  a pulsing red border and an elevated z-index to draw attention.
- [ ] AC-5: Clicking any universe instrument card navigates to its chart page.
- [ ] AC-6: The breadth and correlation view creates such an immediate sense of
  market health that users would absolutely fall in love with the visceral
  understanding and drool over knowing the market's shape in one glance.

---

## Story 5.5: Currency Risk Panel with Carry and Momentum Heatmap

**As a** trader with currency exposure,
**I want** a dedicated currency panel that shows carry, momentum, and risk for
all tracked pairs with forward-looking forecasts,
**so that** I can manage FX risk alongside equity positions.

### Acceptance Criteria

- [ ] AC-1: Currency pairs display as a **grid of cards** (similar to metals),
  each showing:
  - Pair name (e.g., "USD/JPY") in `heading-3` typography
  - Current rate + daily change (colored)
  - Momentum score with directional arrow and colored background
  - Risk score (heat bar)
  - Multi-horizon forecast strip (same visual pattern as metals Story 5.3)
- [ ] AC-2: A **currency heatmap** mode (toggle) shows all pairs in a single
  matrix visualization with:
  - Rows: currency pairs
  - Columns: metrics (Momentum, Risk, 7D Forecast, 30D Forecast)
  - Cell color: diverging green-red colormap
- [ ] AC-3: The JPY section gets a special **"Yen Strength View"** callout card
  at the top of the currencies tab (based on the existing JPY Forecasts feature)
  showing: current yen strength assessment, multi-horizon directional forecast,
  and a recommendation sentence in natural language.
- [ ] AC-4: Currency cards support click-through to the Charts page for the
  corresponding currency pair's analysis.
- [ ] AC-5: Forecast confidence is indicated by cell opacity in the forecast strip:
  high confidence = full opacity, low confidence = semi-transparent.

---

## Story 5.6: Sector Risk Breakdown with Relative Strength

**As a** sector-focused portfolio manager,
**I want** sectors displayed with relative strength ranking and risk attribution,
**so that** I can see which sectors are leading and which are lagging, along with
their individual risk contributions.

### Acceptance Criteria

- [ ] AC-1: Sector ETFs (XLK, XLV, XLI, XLE, etc.) display as **ranked cards**
  in a single column, sorted by performance with rank numbers (#1, #2, ...).
  Top 3 get medal-style badges.
- [ ] AC-2: Each card shows:
  - Sector name + ETF ticker
  - Multi-period returns (1D, 5D, 21D) as colored badges in a row
  - Momentum score with trend arrow
  - Signal summary (what % of assets in this sector are buy/sell/hold)
  - Risk contribution: a tiny stacked bar showing this sector's contribution
    to overall portfolio risk
- [ ] AC-3: A **relative strength chart** renders above the sector list:
  a normalized line chart showing all sector ETFs overlaid (each with its own
  color) for the last 30 days, rebased to 100. This instantly shows which
  sectors are outperforming.
- [ ] AC-4: The sector list supports reordering by: Performance (default),
  Momentum, Risk, Alphabetical via a sort dropdown.
- [ ] AC-5: Clicking a sector card expands it to show all individual assets
  within that sector with their signal status.

---

# EPIC 6: Tuning Page -- The Model Workshop

> **Vision**: The Tuning page is the engine room. A quant engineer comes here to
> understand what is happening inside the model factory: which models are being
> selected, which assets are failing calibration, and whether a retune is needed.

---

## Story 6.1: Retune Control Panel with Live Progress

**As a** quant engineer initiating a model retune,
**I want** a professional control panel that shows real-time progress with
granular visibility into what's happening,
**so that** I can monitor the retune and intervene if something goes wrong.

### Acceptance Criteria

- [ ] AC-1: The retune control area displays as a **control console card**
  (full width, prominent) with:
  - Mode selector: dropdown with modes (Full Retune, Tune Only, Calibrate Failed)
    styled as segmented pill buttons rather than a raw dropdown
  - Start/Stop button: large (48px tall), green when ready ("Start Retune"),
    red when running ("Stop") with a pulsing ring animation during execution
  - Status indicator: text + colored badge (Idle / Running / Completed / Failed)
  - Elapsed time counter (visible when running)
- [ ] AC-2: When a retune is running, a **progress dashboard** expands below the
  control console showing:
  - Progress bar (percentage with asset count: "42 / 147 assets")
  - Current asset being processed: ticker name with a spinning indicator
  - Phase indicator: "Phase 2/3: Kalman Fitting"
  - Estimated time remaining (calculated from average time per asset)
  - A scrollable log terminal (dark background, monospace font) showing the
    last 50 lines of log output with color coding:
    - Green: progress messages
    - Blue: phase transitions
    - Red bold: errors
    - Gray: debug/verbose
- [ ] AC-3: The log terminal auto-scrolls to bottom. A "Scroll to bottom" button
  appears if the user scrolls up. A "Copy log" button copies the full log to
  clipboard.
- [ ] AC-4: The progress dashboard animates in when a retune starts (height expand
  + fade-in, 300ms) and collapses when complete (after a 2s delay showing the
  completion summary).
- [ ] AC-5: Completion shows a summary card: total duration, assets processed,
  pass/fail count, PIT improvement (before vs. after if available).
- [ ] AC-6: The retune control panel gives such confident command over the engine
  that users would absolutely fall in love with the operational control and
  drool over feeling like a mission control engineer.

---

## Story 6.2: Model Distribution Treemap

**As a** quant researcher understanding model selection patterns,
**I want** to see model distribution as a treemap rather than a bar chart,
**so that** I can see both frequency and grouping patterns simultaneously.

### Acceptance Criteria

- [ ] AC-1: The model distribution visualization renders as a **treemap** where:
  - Each rectangle represents a model
  - Rectangle area is proportional to selection count
  - Rectangle color represents the model family:
    - Kalman Gaussian: blue tones
    - Phi Student-t: green tones
    - Momentum-augmented: amber tones
    - NIG/GMM/other: purple tones
  - Rectangle label shows model name (truncated if necessary) and count
- [ ] AC-2: Hovering a treemap cell shows a tooltip with:
  - Full model name
  - Selection count and percentage of total
  - Average BMA weight
  - Average PIT pass rate
  - Top 3 assets using this model
- [ ] AC-3: Clicking a treemap cell filters the asset table below to show only
  assets using that model.
- [ ] AC-4: A toggle switches between Treemap and the traditional Horizontal Bar
  chart for users who prefer the linear view.
- [ ] AC-5: The treemap cells animate in with a coordinated cascade (50ms stagger)
  from largest to smallest on initial load.

---

## Story 6.3: Asset Health Grid with Traffic-Light Status

**As a** quant engineer monitoring the calibration health of 100+ assets,
**I want** a compact visual grid that shows the health status of every asset
at a glance with drill-down capability,
**so that** I can immediately spot which assets need attention.

### Acceptance Criteria

- [ ] AC-1: A **health grid** displays all tuned assets as small square tiles
  (24px x 24px) in a flowing grid, colored by PIT status:
  - Green: PIT pass
  - Red: PIT fail
  - Gray: Unknown / not tested
  Tiles are grouped by sector with sector labels above each group.
- [ ] AC-2: Hovering a tile shows a tooltip with: ticker, best model, PIT status,
  BMA weight, last tuned timestamp.
- [ ] AC-3: Clicking a tile selects it, loading the full detail panel below.
- [ ] AC-4: A **summary bar** above the grid shows pass/fail/unknown counts with
  a stacked progress bar and percentage labels.
- [ ] AC-5: The grid supports filtering: a "Show only failures" toggle dims
  all passing assets to 20% opacity, making failures pop visually.
- [ ] AC-6: Failed asset tiles have a subtle continuous pulse animation (opacity
  oscillation 60-100% over 2s) to draw the eye.
- [ ] AC-7: The health grid makes the calibration state of 100+ assets
  comprehensible in a single glance, causing users to absolutely fall in
  love with the visual density and drool over seeing every asset's status
  simultaneously in 24x24 pixel tiles.

---

## Story 6.4: Model Detail Deep-Dive Panel

**As a** quant researcher investigating why a specific asset chose a particular model,
**I want** a formatted detail panel that presents model parameters, diagnostics,
and calibration metrics in a structured, readable layout,
**so that** I can diagnose model behavior without reading raw JSON.

### Acceptance Criteria

- [ ] AC-1: Selecting an asset in the health grid or table opens a **detail panel**
  (right side, 400px or bottom panel, depending on screen) with structured sections:
  - **Header**: Ticker + Sector badge + PIT status badge (large)
  - **Best Model**: Name, BMA weight (with bar visualization), selection reason
  - **All Competing Models**: Sorted table showing every model that competed:
    Model Name, BIC, CRPS, Hyvarinen, PIT p-value, BMA Weight %, Nu parameter.
    Winner row is highlighted with subtle green background.
  - **Kalman State**: Current mu estimate, sigma estimate, phi parameter,
    process noise q. Each displayed with a label and monospace-formatted value.
  - **Regime**: Current regime classification with color badge, volatility state
  - **Calibration History**: If stored, a mini timeline showing PIT status over
    the last 5 retunes (green/red dots on a horizontal line)
- [ ] AC-2: Numeric values in the competing models table use conditional formatting:
  - CRPS: green if < 0.02, amber 0.02-0.03, red > 0.03
  - Hyvarinen: green if < 500, amber 500-1000, red > 1000
  - PIT p-value: green if >= 0.05, red if < 0.05
  - BMA Weight: intensity-mapped blue background (darker = higher weight)
- [ ] AC-3: The panel includes a "View in Diagnostics" link that navigates to the
  Diagnostics page with the asset pre-selected in the PIT Calibration tab.
- [ ] AC-4: A "Compare" button allows selecting a second asset. When two assets are
  selected, the panel shows them side-by-side for direct parameter comparison.
- [ ] AC-5: The detail panel animates in with a slide-from-right (250ms, `standard`
  motion) and slides out when deselected.

---

# EPIC 7: Diagnostics Page -- The Calibration Laboratory

> **Vision**: The Diagnostics page is a laboratory for model calibration. A quant
> engineer comes here to verify that the system's probabilistic predictions match
> reality. Every metric, every chart, every table answers one question: "Can we
> trust the model's outputs?"

---

## Story 7.1: PIT Calibration Dashboard with Reliability Diagram

**As a** quant engineer validating model calibration,
**I want** a reliability diagram alongside the PIT summary table,
**so that** I can visually assess calibration quality and identify
where the model is overconfident or underconfident.

### Acceptance Criteria

- [ ] AC-1: The PIT Calibration tab shows a **reliability diagram** card
  at the top spanning half-width:
  - X-axis: Predicted probability (0% to 100%, 10 bins)
  - Y-axis: Observed frequency (0% to 100%)
  - Perfect calibration line: diagonal dashed line (gray)
  - Actual calibration: colored connected dots (green if close to diagonal,
    red if deviating). Each dot is sized by the count of observations in that bin.
  - Shaded confidence band around the diagonal (95% CI for a well-calibrated
    model) -- points inside the band are "acceptable" calibration.
- [ ] AC-2: Hovering a dot on the reliability diagram shows: predicted probability
  range, observed frequency, count of observations, deviation from ideal.
- [ ] AC-3: A **summary metric card** beside the reliability diagram shows:
  - ECE (Expected Calibration Error) with colored badge
  - MCE (Maximum Calibration Error) with colored badge
  - Overall PIT assessment sentence: "Well calibrated" (green) or
    "Overconfident in the 40-60% range" (amber/red with specific diagnosis)
- [ ] AC-4: The PIT summary table below includes all current columns plus:
  - A mini reliability dot indicator (3 dots: green/amber/red) representing
    calibration quality at low/mid/high probability ranges
- [ ] AC-5: Expandable asset rows show the per-model detail table with the same
  conditional formatting as Story 6.4 AC-2.
- [ ] AC-6: The reliability diagram is interactive: clicking a bin segment
  filters the asset table below to show only assets whose predictions fell
  in that probability range.
- [ ] AC-7: The PIT dashboard combines statistical rigor with visual clarity
  so effectively that users would absolutely fall in love with understanding
  model trustworthiness and drool over the reliability diagram's insight.

---

## Story 7.2: Cross-Asset Calibration Matrix with Filtering

**As a** quant engineer comparing calibration across all assets and models,
**I want** a fully interactive cross-asset matrix with rich filtering,
**so that** I can identify systematic calibration patterns across the universe.

### Acceptance Criteria

- [ ] AC-1: The cross-asset matrix displays a **zoomable heatmap** with:
  - Rows: Assets (sorted by sector, then alphabetical)
  - Columns: Models (sorted by BMA selection frequency)
  - Cell value: the selected metric (CRPS, PIT KS p-value, or BMA Weight)
  - Cell color: diverging colormap appropriate to the metric
    - CRPS: green (< 0.015) -> amber (0.02) -> red (> 0.03)
    - PIT p: green (> 0.1) -> amber (0.05) -> red (< 0.01)
    - Weight: white (0%) -> deep blue (100%)
- [ ] AC-2: Row and column headers are **sticky** during scroll (both horizontal
  and vertical scrolling keeps headers visible).
- [ ] AC-3: A **metric selector** (segmented pill buttons) switches between CRPS,
  PIT p-value, and Weight with a smooth crossfade animation on the heatmap cells.
- [ ] AC-4: Clicking a cell shows a popup detail card with: asset name, model name,
  all metrics for that combination (BIC, CRPS, Hyv, PIT p, Weight, Nu).
- [ ] AC-5: A "Highlight Outliers" toggle dims all cells within normal range
  and makes outlier cells glow with a pulsing ring.
- [ ] AC-6: The matrix supports **column sorting** by clicking a model header:
  all asset rows reorder by that model's metric value.
- [ ] AC-7: For large matrices (150+ assets x 15+ models), virtualized rendering
  ensures smooth 60fps scrolling.

---

## Story 7.3: Model Comparison with Statistical Horse Race

**As a** quant researcher evaluating which models are performing best,
**I want** a visual "horse race" comparison showing win rates, strengths, and
weaknesses per model,
**so that** I can make informed decisions about model roster changes.

### Acceptance Criteria

- [ ] AC-1: The Model Comparison tab opens with a **podium visualization** showing
  the top 3 models:
  - Gold (#1): center position, largest card, medal icon
  - Silver (#2): left position, medium card
  - Bronze (#3): right position, medium card
  Each podium card shows: model name, win count, win rate, average BMA weight.
- [ ] AC-2: Below the podium, a **comparison radar chart** (spider/radar) allows
  selecting 2-5 models for multi-axis comparison. Axes include:
  - Win Rate, Avg Weight, Avg CRPS, Avg PIT p-value, Asset Count, Avg BIC
  - Each model is a differently colored polygon on the radar.
  - Model selection via checkboxes next to the full model table.
- [ ] AC-3: The full model statistics table includes all current columns plus:
  - A **trend indicator**: arrow showing whether each model's win count has
    increased or decreased compared to the previous tune run
  - A **consistency score**: standard deviation of BMA weights across assets
    (lower = more consistent performer)
- [ ] AC-4: Hovering a model row highlights all assets using that model in a
  connected detail view below the table.
- [ ] AC-5: The podium cards animate in with a stagger effect: bronze first
  (slide up from bottom), then silver, then gold, each with `spring` motion.

---

## Story 7.4: Regime Distribution Sunburst Chart

**As a** quant engineer understanding regime classification patterns,
**I want** to see how assets distribute across regimes in a hierarchical
visualization,
**so that** I can identify regime concentrations and anomalies.

### Acceptance Criteria

- [ ] AC-1: The Regime Distribution tab displays a **sunburst chart** with:
  - Inner ring: 5 regime segments (LOW_VOL_TREND, HIGH_VOL_TREND,
    LOW_VOL_RANGE, HIGH_VOL_RANGE, CRISIS_JUMP)
  - Outer ring: individual assets within each regime, sized by confidence
  - Colors match regime semantics:
    - LOW_VOL_TREND: calm blue
    - HIGH_VOL_TREND: warm amber
    - LOW_VOL_RANGE: cool gray
    - HIGH_VOL_RANGE: orange
    - CRISIS_JUMP: bright red
- [ ] AC-2: Clicking an inner ring segment zooms into that regime, enlarging
  it to fill the chart and showing all individual assets with detail labels.
  A "Back" button or clicking center returns to the full sunburst view.
- [ ] AC-3: Hovering an outer ring segment (asset) shows: ticker, sector,
  regime confidence score, volatility metrics.
- [ ] AC-4: A **regime summary table** beside the chart shows: regime name,
  asset count, percentage, average volatility, average momentum, top asset.
- [ ] AC-5: The sunburst animates in by growing from center outward (inner ring
  first, then outer ring) over 600ms with `expressive` motion.

---

## Story 7.5: Profitability Tracking with Target Achievement Bands

**As a** quant engineer monitoring system profitability metrics over time,
**I want** charts that clearly show whether metrics are within acceptable bands,
**so that** I can quickly identify degradation trends before they become critical.

### Acceptance Criteria

- [ ] AC-1: Each profitability metric chart renders with:
  - Line chart of the metric over time (primary color per metric)
  - **Target band**: a shaded horizontal region showing the acceptable range
    (e.g., Hit Rate target: 50-65%). Band color: green at 10% opacity.
  - **Danger zone**: regions outside the acceptable band shaded red at 5% opacity
  - **Target line**: dashed horizontal line at the exact target value
  - Points where the metric crosses a band boundary get a **marker dot** (8px)
    with color indicating the direction (green = improving, red = degrading)
- [ ] AC-2: The current metric value displays as a large number above each chart
  with a colored badge: green if within target band, amber if close to boundary,
  red if outside band.
- [ ] AC-3: Hovering any point on the line chart shows the value, date, and
  distance from target as a percentage.
- [ ] AC-4: Charts display in a responsive 2x3 grid (3 columns on desktop,
  2 on tablet, 1 on mobile).
- [ ] AC-5: A "Summary" card at the top shows how many metrics are: On Target,
  Near Boundary, Outside Target, with respective colored counts.
- [ ] AC-6: The profitability tracking makes trend health so visually obvious
  that users would absolutely fall in love with monitoring system performance
  and drool over the instant visual assessment of all metrics simultaneously.

---

# EPIC 8: Data Management and System Health

> **Vision**: Data and Services pages are operational infrastructure pages. They
> must feel like professional sysadmin dashboards: dense, accurate, and actionable.
> A user checks these pages to confirm the system is healthy and data is fresh.

---

## Story 8.1: Data Freshness Dashboard with Aging Visualization

**As a** system operator ensuring data pipeline health,
**I want** a visual freshness dashboard that shows data age as a physical
metaphor (degrading over time),
**so that** I can identify stale data immediately and take action.

### Acceptance Criteria

- [ ] AC-1: The top section shows 4 stat cards (existing) plus a new
  **"Freshness Timeline"** spanning full width below the stats:
  - A horizontal timeline representing 0 hours to 72 hours
  - Each data file appears as a dot on the timeline positioned at its age
  - Dots cluster: fresh data (< 6h) bunches on the left in green,
    aging data (6-24h) spreads toward center in amber,
    stale data (> 24h) scatters right in red
  - This creates an immediate visual "weight" -- healthy systems have dots
    clustered left, unhealthy systems have dots spread right.
- [ ] AC-2: Hovering a dot on the timeline shows the file name, exact age,
  size, and last update timestamp.
- [ ] AC-3: The timeline has labeled tick marks at: 1h, 6h, 12h, 24h, 48h, 72h.
  A threshold marker at 24h is prominently indicated.
- [ ] AC-4: Below the timeline, a sentence summary: "147 of 152 files are fresh.
  5 files need refresh (oldest: IONQ, 38h ago)." in `body` typography.
- [ ] AC-5: An "Auto-Refresh" toggle enables automatic daily data refresh
  scheduling. The toggle shows next scheduled refresh time when enabled.
- [ ] AC-6: The dots on the timeline animate in from the right (gathering toward
  their position) on page load, like data arriving at its age position.

---

## Story 8.2: Directory Health Tree with Capacity Indicators

**As a** system operator managing disk space and data organization,
**I want** a visual directory tree that shows sizes, file counts, and health,
**so that** I can identify bloated or missing directories immediately.

### Acceptance Criteria

- [ ] AC-1: The directories section renders as a **visual tree** rather than a
  flat list. Each directory shows:
  - Directory name with folder icon (SVG)
  - File count badge
  - Disk size (formatted: KB, MB, GB)
  - A **capacity bar** (60px wide) showing relative size compared to the
    largest directory, with the bar fill progressing from green to amber
    to red as size increases
  - Status badge: "OK" (green) or "Missing" (red with alert icon)
- [ ] AC-2: Directories with zero files display with a faded appearance
  (40% opacity) and an "Empty" badge in amber.
- [ ] AC-3: The tree supports expand/collapse for nested directories
  (if applicable in the data structure).
- [ ] AC-4: A **disk usage summary card** at the top shows:
  - Total disk usage (large number)
  - Pie chart (mini, 48px) showing distribution across directory categories
  - Last cleanup timestamp
  - A "Clean Cache" button that triggers cache cleanup and shows a confirmation
    dialog with what will be deleted and estimated space recovered.
- [ ] AC-5: Hovering a directory shows a tooltip with: full path, exact size,
  file count, and oldest file age.

---

## Story 8.3: Services Health with Dependency Graph

**As a** system operator monitoring service health,
**I want** a dependency graph showing how services relate and which failures
cascade to other services,
**so that** I can diagnose root causes rather than just symptoms.

### Acceptance Criteria

- [ ] AC-1: The hero section shows an **animated status ring** instead of a
  simple text message:
  - A circle of 5 status dots (API, Cache, Prices, Workers, Redis), evenly
    spaced around a central status icon
  - Green dots for healthy services, red for failed, amber for degraded
  - Connecting lines between dependent services (e.g., API -> Cache, Workers -> Redis)
  - Line color: green if both endpoints healthy, red if either endpoint failed
  - Central icon: large checkmark (all green), warning triangle (some amber),
    X (any red)
- [ ] AC-2: Each service card expands to show a **dependency chain**:
  "API Server -> Signal Cache -> Price Data" with health status at each step.
  This makes it clear that a stale cache might be caused by stale price data.
- [ ] AC-3: The **error log** section shows a styled log viewer with:
  - Color-coded severity (error=red, warning=amber, info=blue)
  - Collapsible stack traces (if present in error messages)
  - Timestamp on the left in monospace font
  - Source service as a colored badge
  - Search/filter by source or keyword
  - Auto-refresh indicator showing countdown to next refresh
- [ ] AC-4: A "Health History" toggle shows a 24-hour timeline with colored
  bars showing uptime/downtime per service over the last day.
- [ ] AC-5: The service dependency visualization creates such immediate clarity
  about system health that users would absolutely fall in love with the
  operational awareness and drool over diagnosing issues in seconds
  rather than minutes.

---

# EPIC 9: Arena Page -- The Competitive Research Lab

> **Vision**: The Arena page is a competitive simulation lab where experimental
> models fight for survival against established baselines. It must feel like a
> sports analytics dashboard: rankings, head-to-head comparisons, and promotion
> ceremonies when a model graduates.

---

## Story 9.1: Model Leaderboard with Performance Sparklines

**As a** quant researcher evaluating experimental models,
**I want** a leaderboard that shows model rankings with inline performance
indicators and trend history,
**so that** I can quickly assess which models are competitive and improving.

### Acceptance Criteria

- [ ] AC-1: The safe storage models display as a **leaderboard table** with:
  - Rank number (#1-#N) with colored badges for top 3 (gold/silver/bronze SVG)
  - Model name (truncated, full on hover)
  - **Performance sparkline** (60px wide): showing the Final score for this model
    across its last 5 evaluations (requires historical data). Single dots if
    only one evaluation exists.
  - Final Score (large, bold, colored: green > 70, amber 60-70, red < 60)
  - Key metric pills: BIC, CRPS, Hyv, CSS, FEC (each as a tiny colored badge)
  - vs STD delta ("+10.7" in green or "-2.1" in red)
  - Evaluation time and file size (muted)
- [ ] AC-2: Each metric pill uses semantic coloring:
  - BIC: green if < -29000, red otherwise
  - CRPS: green if < 0.020, red otherwise
  - Hyv: green if < 1000, amber 1000-1500, red > 1500
  - CSS: green if >= 0.65, red otherwise
  - FEC: green if >= 0.75, red otherwise
- [ ] AC-3: Clicking a model row expands it to show:
  - Full metrics table (all scoring dimensions)
  - Hard gates checklist with pass/fail indicators (SVG check/X icons)
  - Model description / notes (if stored in the model file)
  - "View Source" link that opens the model's .py file details
- [ ] AC-4: The leaderboard supports sorting by any column (click header).
- [ ] AC-5: A "Compare Models" mode allows selecting 2-3 models and viewing
  their metrics side-by-side in a comparison card with highlighted differences.

---

## Story 9.2: Hard Gates Visualization as a Gauntlet

**As a** quant researcher understanding what prevents a model from graduating,
**I want** hard gates displayed as a visual gauntlet that a model must pass through,
**so that** I can see exactly which gates pass, which fail, and by how much.

### Acceptance Criteria

- [ ] AC-1: The hard gates section renders as a **horizontal gauntlet** (pipeline):
  8 gates arranged left-to-right, each represented as a vertical bar:
  - Green fill if the model passes that gate
  - Red fill if the model fails
  - Yellow fill if within 10% of the threshold
  - Bar height proportional to the margin of pass/fail
    (taller = further above threshold, shorter = barely passing)
- [ ] AC-2: Each gate column shows:
  - Gate name (caption text above)
  - Threshold value (small text below the bar)
  - Model's actual value (inside the bar)
  - Pass/fail icon at the top (check or X)
- [ ] AC-3: When all 8 gates are green, a **"PROMOTED"** celebration state triggers:
  - A brief golden shimmer animation across the gauntlet
  - A confetti-style micro-animation (subtle, professional -- not playful)
  - A "Promoted to Production" badge appears
- [ ] AC-4: The gauntlet is filterable: selecting different models from a dropdown
  updates the gauntlet visualization for that model with smooth value transitions
  (bar heights animate, 300ms).
- [ ] AC-5: A "All Models Gauntlet" view shows multiple models stacked vertically,
  each as a single row of 8 colored dots (pass/fail/partial), creating a
  matrix that instantly shows which models are closest to graduation.
- [ ] AC-6: The gauntlet visualization makes model promotion criteria so tangible
  that users would absolutely fall in love with understanding exactly what
  separates a good model from a promotable one and drool over the visual
  clarity of the competitive pipeline.

---

## Story 9.3: Benchmark Universe Interactive Performance Map

**As a** researcher analyzing model performance across the benchmark universe,
**I want** an interactive grid showing per-symbol performance for each model,
**so that** I can identify which models excel on which types of assets.

### Acceptance Criteria

- [ ] AC-1: The 12 benchmark symbols display as **large card tiles** in a 4x3 grid,
  each containing:
  - Symbol ticker (large) + asset class label (Small Cap / Mid Cap / Large / Index)
  - 90-day price sparkline (64px tall)
  - Current regime classification badge
  - The winning model for this symbol (colored badge)
  - Final score achieved on this symbol (colored number)
- [ ] AC-2: Hovering a symbol card reveals an overlay showing all models' scores
  for that symbol as a mini leaderboard (top 5 models with scores), without
  leaving the grid view.
- [ ] AC-3: A **model filter** allows selecting a specific model to highlight.
  When a model is selected, each symbol card shows that model's score,
  with green/red coloring indicating whether it beat the best standard model
  for that symbol.
- [ ] AC-4: Symbols in the grid sort by: Default order, Score (best first),
  Regime, or Asset Class via a sort dropdown.
- [ ] AC-5: Each symbol card is clickable and navigates to `/charts/{symbol}`.

---

# EPIC 10: Micro-Interactions and Delightful Details

> **Vision**: The difference between "good" and "falls in love and drools" lives
> in the micro-interactions. These are the small touches that reward attention
> and create emotional engagement.

---

## Story 10.1: Loading States as Content Previews

**As a** user waiting for data to load,
**I want** loading states that preview the content layout with animated placeholders,
**so that** the page feels fast even before data arrives and the transition from
loading to loaded is seamless.

### Acceptance Criteria

- [ ] AC-1: Every page implements **skeleton loading** matching its content layout:
  - Cards: gray rounded rectangles matching card dimensions with shimmer animation
  - Tables: rows of gray bars matching column widths
  - Charts: a gray rectangle matching chart dimensions with a slow shimmer
  - Stat values: short gray bars where numbers will appear
- [ ] AC-2: The skeleton shimmer animation uses a diagonal gradient sweep
  (left-to-right, 2s duration, infinite repeat) with `linear-gradient(110deg,
  transparent 30%, rgba(255,255,255,0.04) 50%, transparent 70%)`.
- [ ] AC-3: When data arrives, the skeleton fades out (150ms) and content fades in
  (200ms), with a 50ms overlap creating a seamless crossfade.
- [ ] AC-4: Skeleton elements match the exact dimensions and positions of their
  data counterparts, so there is zero layout shift when data loads.
- [ ] AC-5: If data loading takes > 3 seconds, a subtle text hint appears below
  the skeleton: "Computing... this may take a moment" with a soft fade-in.
- [ ] AC-6: The loading experience is so polished that users would absolutely
  fall in love with the perceived performance and drool over the seamless
  transition from skeleton to content.

---

## Story 10.2: Empty States with Personality and Guidance

**As a** user encountering a page or section with no data,
**I want** empty states that explain why there's no data and what I can do,
**so that** I never feel confused or abandoned by the interface.

### Acceptance Criteria

- [ ] AC-1: Every section that can be empty has a **designed empty state** with:
  - An illustrative SVG icon (64px, single-color line art style matching the
    section's theme)
  - A concise headline explaining the state (e.g., "No strong signals today")
  - A sub-text with guidance (e.g., "The models are in agreement -- hold positions
    are dominant during low-volatility regimes")
  - An optional action button (e.g., "View All Signals" or "Adjust Filters")
- [ ] AC-2: Empty state icons are unique per section:
  - No signals: balanced scales icon
  - No chart data: empty graph with question mark
  - No errors: rocket ship (everything working!)
  - No arena models: laboratory flask
  - Search returns nothing: magnifying glass with X
- [ ] AC-3: Empty states use `--text-secondary` for the headline and
  `--text-muted` for the sub-text, centered within the container.
- [ ] AC-4: The empty state animates in with `fade-up` + slight scale
  (from 95% to 100%) over 300ms.

---

## Story 10.3: Contextual Tooltips with Rich Content

**As a** user hovering over metrics and indicators throughout the application,
**I want** tooltips that provide context, definitions, and visual aids,
**so that** every number and indicator is self-documenting.

### Acceptance Criteria

- [ ] AC-1: Every abbreviated metric in the application has a **rich tooltip** on hover:
  - BIC: "Bayesian Information Criterion. Lower is better. Penalizes model complexity.
    Good: < -29,000."
  - CRPS: "Continuous Ranked Probability Score. Measures calibration + sharpness.
    Good: < 0.020."
  - Hyvarinen: "Hyvarinen Score. Detects variance collapse. Good: < 1,000,
    Elite: < 500."
  - PIT: "Probability Integral Transform. Tests if P(r > 0) = predicted probability.
    Pass: p-value >= 0.05."
  - CSS: "Calibration Stability Under Stress. Must hold during market stress.
    Required: >= 0.65."
  - FEC: "Forecast Entropy Consistency. Uncertainty must track market uncertainty.
    Required: >= 0.75."
  - Kelly: "Kelly Criterion fraction. Optimal bet sizing based on edge and
    probability distribution."
- [ ] AC-2: Tooltips use `tooltip-dark` styling: dark background (rgba(10,10,26,0.95)),
  `backdrop-filter: blur(12px)`, 1px border `--border-subtle`, max-width 280px.
- [ ] AC-3: Tooltips appear after 300ms hover delay and fade in over 150ms.
  They are positioned automatically to avoid viewport overflow (flip above/below
  as needed).
- [ ] AC-4: Where applicable, tooltips include a **mini visualization**:
  - Temperature tooltip: mini colored scale bar
  - PIT pass/fail: colored status dot
  - Momentum: directional arrow
- [ ] AC-5: Tooltips are keyboard-accessible: focusing an element with Tab
  shows the same tooltip.

---

## Story 10.4: Number Transitions with Counting Animation

**As a** user watching values update (signal counts, temperatures, percentages),
**I want** numbers to animate when they change rather than snapping to new values,
**so that** changes are noticeable and the interface feels alive.

### Acceptance Criteria

- [ ] AC-1: All primary numeric displays (stat card values, temperature numbers,
  percentages, counts) use **counting animation** when their value changes:
  - The number counts from the old value to the new value over 400ms
  - Uses `requestAnimationFrame` for smooth 60fps counting
  - Easing: `cubic-bezier(0.2, 0, 0, 1)` (starts fast, decelerates)
- [ ] AC-2: When a value increases, the number briefly flashes with a subtle
  green pulse. When it decreases, a subtle red pulse. Neutral changes (no
  semantic direction) use a blue pulse.
- [ ] AC-3: Decimal precision is preserved during counting (no jarring precision
  jumps mid-animation).
- [ ] AC-4: The counting animation is disabled when:
  - The page is loading for the first time (values should appear, not count up from 0)
  - The browser tab is not visible (avoid animation burst on tab return)
  - The user has `prefers-reduced-motion` enabled in their OS settings
- [ ] AC-5: Percentage values show a micro progress bar below the number that
  adjusts width to match the percentage (8px tall, colored to match value).

---

## Story 10.5: Responsive Breakpoint System

**As a** user accessing the dashboard on different screen sizes,
**I want** the layout to adapt gracefully at defined breakpoints,
**so that** usability is maintained from 1024px to 2560px ultrawide monitors.

### Acceptance Criteria

- [ ] AC-1: The application supports 4 breakpoint tiers:
  - **Compact** (1024-1279px): Sidebar collapsed by default, single-column
    card layouts, condensed tables (fewer columns), chart sidebar hidden
  - **Standard** (1280-1599px): Sidebar expanded, 2-column grids, standard
    table columns, chart sidebar collapsed by default
  - **Comfortable** (1600-1919px): Full layouts, all table columns, chart
    sidebar expanded
  - **Ultrawide** (1920px+): Extra spacing, wider cards, side-by-side panels
    where vertically stacked in other tiers
- [ ] AC-2: Main content max-width scales per breakpoint:
  - Compact: 100% (no max)
  - Standard: 1280px
  - Comfortable: 1600px
  - Ultrawide: 2000px
- [ ] AC-3: Tables switch to horizontally scrollable mode at compact breakpoint
  rather than dropping columns. A subtle horizontal scroll indicator appears.
- [ ] AC-4: Card grids use CSS Grid `auto-fit` with `minmax(320px, 1fr)` for
  natural responsive flow without explicit breakpoint logic.
- [ ] AC-5: Charts resize to fill their container width with a `ResizeObserver`
  that triggers a smooth re-render (not a jump).
- [ ] AC-6: Users on ultrawide monitors (>= 2560px) get a **split view** option:
  two pages side-by-side (e.g., Signals on the left, Charts on the right),
  activated via `Cmd+Shift+Split`.

---

## Story 10.6: Keyboard Shortcut Reference Overlay

**As a** power user learning the keyboard shortcuts,
**I want** a visual shortcut reference overlay accessible via `?`,
**so that** I can discover and memorize shortcuts without leaving the app.

### Acceptance Criteria

- [ ] AC-1: Pressing `?` (when no input is focused) opens a **shortcut overlay**
  centered on screen (480px wide, max-height 70vh, scrollable) with:
  - Grouped sections: Global, Signals Page, Charts Page, Tables
  - Each shortcut shows: key combination (styled as keyboard key caps) + description
  - Key caps use a subtle raised appearance with 1px border and slight shadow
    to look like physical keys
- [ ] AC-2: The overlay background dims to `rgba(0,0,0,0.6)` with backdrop blur.
  `Esc` or clicking outside closes it.
- [ ] AC-3: A search bar at the top of the overlay filters shortcuts by keyword.
- [ ] AC-4: The overlay includes a "Print" version (clean layout for printing
  as a desk reference card).
- [ ] AC-5: At the bottom, a toggle: "Show shortcut hints in UI" -- when enabled,
  keyboard shortcut badges appear next to interactive elements throughout the app.

---

# EPIC 11: Cross-Cutting Quality and Performance

> **Vision**: A beautiful interface that lags or flickers is worse than an ugly one
> that is fast. Performance is not a feature -- it is the foundation on which every
> feature stands.

---

## Story 11.1: React Query Cache Orchestration

**As a** user navigating between pages without re-fetching unchanged data,
**I want** an intelligent cache strategy that shares data across pages and
minimizes redundant API calls,
**so that** page transitions are instant when data is fresh.

### Acceptance Criteria

- [ ] AC-1: **Shared query keys** are established so that data fetched on one page
  is available on another without re-fetching:
  - `['signalSummary']`: shared between Overview, Signals, Charts, Risk (sidebar indicators)
  - `['strongSignals']`: shared between Overview (spotlight), Signals (cards), Charts (filter)
  - `['riskSummary']`: shared between Overview (stat cards) and Risk (hero)
  - `['servicesHealth']`: shared between Layout (sidebar pulse) and Services page
- [ ] AC-2: **Stale times** are calibrated per data type:
  - Signal data: 60 seconds (near real-time with WebSocket updates)
  - Risk data: 5 minutes (computationally expensive, changes slowly)
  - Tuning data: 30 minutes (changes only after retune)
  - Data pipeline: 5 minutes (file ages change slowly)
  - Services health: 10 seconds (operational monitoring)
- [ ] AC-3: **Background refetch** is enabled for critical queries (signals, risk,
  services) so data stays fresh without user-initiated refresh.
- [ ] AC-4: A **cache hit indicator** (development mode only) appears as a tiny
  green flash in the corner when a page renders using cached data, helping
  developers verify cache behavior.

---

## Story 11.2: Virtual Scrolling for Large Data Sets

**As a** user viewing tables with 100+ rows,
**I want** the table to render only visible rows while maintaining smooth scrolling,
**so that** large datasets do not cause performance degradation.

### Acceptance Criteria

- [ ] AC-1: Tables with more than 50 rows implement **virtual scrolling** using
  a windowing library (e.g., @tanstack/react-virtual or react-window).
- [ ] AC-2: Virtual scrolling maintains:
  - Correct scroll height (scrollbar reflects total row count)
  - Sticky header position
  - Smooth 60fps scrolling (no jank or white flashes)
  - Correct row striping (alternating backgrounds stay consistent)
- [ ] AC-3: Expanded rows (with sub-tables) work correctly within virtual scrolling
  by dynamically adjusting row heights.
- [ ] AC-4: Search/filter updates re-virtualize the list without scroll position
  loss when possible.
- [ ] AC-5: The Signal Heatmap (200+ cells) uses a 2D virtualization approach:
  only visible rows AND columns are rendered. Off-screen cells are placeholder
  rectangles.

---

## Story 11.3: WebSocket Connection Resilience

**As a** user relying on real-time signal updates,
**I want** the WebSocket connection to automatically reconnect, indicate status,
and queue missed updates,
**so that** I have confidence that the data I see is always current.

### Acceptance Criteria

- [ ] AC-1: WebSocket connection status is visible as a subtle indicator in the
  page header of any page that uses WebSocket (currently Signals):
  - Connected: green dot with "Live" label
  - Reconnecting: amber dot with pulse, "Reconnecting..." label, attempt count
  - Disconnected: red dot with "Offline" label, time since disconnect
- [ ] AC-2: Automatic reconnection uses exponential backoff:
  1s, 2s, 4s, 8s, 16s, max 30s between attempts.
- [ ] AC-3: On successful reconnection, a full data refresh is triggered
  (re-query all signal data) to ensure no updates were missed.
- [ ] AC-4: If the connection is lost for > 60 seconds, a non-blocking banner
  appears at the top of the page: "Real-time updates paused. Data may be stale.
  [Reconnect Now]" with a manual reconnect button.
- [ ] AC-5: Connection state is available globally (React context) so any page
  can show real-time status awareness even if it doesn't directly use WebSocket.
- [ ] AC-6: The connection resilience is so invisible during normal operation
  and so helpful during disruptions that users would absolutely fall in love
  with never worrying about stale real-time data and drool over the
  confidence that what they see is always current.

---

## Story 11.4: Accessibility Foundation (WCAG 2.1 AA)

**As a** user who may use assistive technology or have visual preferences,
**I want** the application to meet WCAG 2.1 AA accessibility standards,
**so that** the dashboard is usable by everyone regardless of ability.

### Acceptance Criteria

- [ ] AC-1: All interactive elements are keyboard-focusable with visible focus
  rings (2px solid `--border-focus`, 2px offset) on `:focus-visible`.
- [ ] AC-2: Color is never the sole indicator of meaning. Every colored element
  (signal badges, status dots, heatmap cells) has an accompanying text label,
  icon, or pattern.
- [ ] AC-3: All images and icons have appropriate `aria-label` or `aria-hidden`
  attributes. Decorative icons use `aria-hidden="true"`.
- [ ] AC-4: Tables use proper `<th>` headers with `scope` attributes.
  Screen readers can navigate table cells with column/row context.
- [ ] AC-5: The application respects `prefers-reduced-motion`: all animations
  reduce to simple opacity fades or are disabled entirely.
- [ ] AC-6: Color contrast ratios meet WCAG AA (4.5:1 for normal text, 3:1 for
  large text) across the dark theme. Critical: `--text-primary` on `--bg-surface`
  must be >= 4.5:1.
- [ ] AC-7: ARIA live regions announce dynamic content updates (toast notifications,
  signal changes, refresh completions) to screen readers.
- [ ] AC-8: A `prefers-color-scheme: light` media query supports a light mode
  foundation (optional future enhancement -- create CSS custom property structure
  that enables light mode without restructuring components).

---

## Story 11.5: Error Boundaries with Graceful Recovery

**As a** user whose session encounters a JavaScript error,
**I want** the error to be contained to the affected component with a recovery path,
**so that** a single failing chart or table does not crash the entire dashboard.

### Acceptance Criteria

- [ ] AC-1: Every page-level component is wrapped in an Error Boundary that catches
  render errors and shows:
  - An error card matching the glass-card style
  - A descriptive message: "This section encountered an error"
  - Technical detail (collapsed by default): error message and component stack
  - A "Retry" button that resets the error boundary and re-renders the component
  - A "Report Bug" button that copies error details to clipboard
- [ ] AC-2: Section-level error boundaries exist for independent data sections
  within a page (e.g., each tab on Diagnostics is independently error-bounded).
- [ ] AC-3: Error boundaries log errors to the browser console with structured
  format: timestamp, component name, error message, stack trace.
- [ ] AC-4: The error UI does NOT show raw stack traces by default (security).
  Only the component name and error type are visible. Full details require
  clicking "Show Details".
- [ ] AC-5: After error recovery (clicking Retry), if the same error recurs
  within 5 seconds, the error boundary shows "Persistent Error" and suggests
  page reload: "This issue requires a page refresh. [Refresh Now]".

---

# EPIC 12: Data Export and Sharing

> **Vision**: A dashboard that cannot export its insights is a silo. Professional
> users need to share analysis, create reports, and integrate with other tools.

---

## Story 12.1: One-Click Data Export for Tables

**As a** user who needs to share signal data with colleagues or use it in Excel,
**I want** every data table in the application to support one-click export,
**so that** data extraction never requires manual copying.

### Acceptance Criteria

- [ ] AC-1: Every data table in the application has an **export button** (download
  icon) in its header/toolbar area.
- [ ] AC-2: Clicking the export button offers format options:
  - **CSV**: comma-separated values, UTF-8 with BOM for Excel compatibility
  - **JSON**: structured with metadata (export timestamp, filter state, sort state)
  - **Clipboard**: copies tab-separated values for direct paste into spreadsheets
- [ ] AC-3: Exports respect the current sort and filter state -- the exported data
  matches exactly what the user sees on screen.
- [ ] AC-4: File names include context: `signals_strong_buy_2026-04-03.csv`,
  `pit_calibration_all_2026-04-03.json`.
- [ ] AC-5: For large exports (> 500 rows), a progress toast shows during generation.
- [ ] AC-6: Export buttons are universally available on: Signal summary table,
  All Assets table, PIT calibration table, Cross-asset matrix, Model comparison
  table, Price files table, Arena leaderboard, Metals table, Currencies table,
  Sectors table.

---

## Story 12.2: Chart Screenshot and Sharing

**As a** trader who wants to share a chart analysis with a team member,
**I want** to capture a chart with all its overlays, annotations, and signal
data as a high-quality image,
**so that** sharing analysis is a one-click operation.

### Acceptance Criteria

- [ ] AC-1: The chart toolbar includes a **"Capture"** button (camera icon).
- [ ] AC-2: Clicking Capture generates a high-resolution PNG (2x device pixel ratio)
  of the chart area including:
  - Price chart with all active overlays
  - Forecast probability cone (if visible)
  - Annotations (if any)
  - A subtle watermark in the bottom-right: "BMA Signal Engine | {SYMBOL} | {date}"
  - The current signal badge for the symbol
- [ ] AC-3: After generation, the image is offered via the browser's file save dialog
  with a suggested filename: `{SYMBOL}_chart_{date}.png`.
- [ ] AC-4: A "Copy to Clipboard" alternative copies the image to clipboard for
  direct paste into messaging apps or documents.
- [ ] AC-5: The captured image has a clean dark background matching the app theme,
  ensuring the screenshot looks professional when shared on any background.
- [ ] AC-6: Chart captures are so convenient and professional that users would
  absolutely fall in love with sharing analysis and drool over the one-click
  path from insight to shared artifact.

---

## Story 12.3: Custom Dashboard Report Generation

**As a** portfolio manager who prepares weekly reports for stakeholders,
**I want** to generate a comprehensive PDF/printable report combining data from
multiple pages,
**so that** stakeholder communication is automated and consistent.

### Acceptance Criteria

- [ ] AC-1: A "Generate Report" action is available from the command palette
  and from the Overview page header.
- [ ] AC-2: The report wizard allows selecting sections to include:
  - Executive Summary (risk temperature, signal distribution, top convictions)
  - Signal Table (full or top-20, sortable by the user's preference)
  - Risk Assessment (temperature, cross-asset stress summary)
  - Model Health (PIT pass rate, top models, calibration quality)
  - Sector Summary (performance, sentiment, forecasts)
- [ ] AC-3: Selecting sections and clicking "Generate" produces a printable HTML
  page (opens in new tab) with:
  - Clean white background, professional typography (print-optimized CSS)
  - Tables, metrics, and mini-charts formatted for A4 paper
  - Header with report title, date, and generation timestamp
  - Page break hints for proper multi-page printing
- [ ] AC-4: The report page includes a "Print" button that triggers `window.print()`
  and a "Save PDF" note instructing the user to use "Print > Save as PDF".
- [ ] AC-5: Report section preferences persist in `localStorage` so the next report
  generation pre-selects the same sections.
- [ ] AC-6: Report generation processes all data already cached in React Query --
  no additional API calls are made.

---

# Summary: Epic and Story Index

| Epic | Stories | Theme |
|------|---------|-------|
| 1: Navigation Shell | 1.1-1.5 | Sidebar, Command Palette, Breadcrumbs, Toasts, Status Strip |
| 2: Dashboard | 2.1-2.5 | Briefing Hero, Signal Radar, Heatmap, Model Leaderboard, Conviction |
| 3: Signals | 3.1-3.6 | Micro-Charts, Multi-Sort, Sectors, Flash, Search, Horizon Density |
| 4: Charts | 4.1-4.6 | Probability Overlay, Detail Sidebar, Toolbar, Picker, Zoom, Notes |
| 5: Risk | 5.1-5.6 | Gauge, Stress Matrix, Metals, Breadth, Currencies, Sectors |
| 6: Tuning | 6.1-6.4 | Control Panel, Treemap, Health Grid, Detail Panel |
| 7: Diagnostics | 7.1-7.5 | PIT Dashboard, Cross-Asset Matrix, Horse Race, Sunburst, Profitability |
| 8: Data/Services | 8.1-8.3 | Freshness Dashboard, Directory Tree, Dependency Graph |
| 9: Arena | 9.1-9.3 | Leaderboard, Hard Gates Gauntlet, Benchmark Map |
| 10: Micro-Interactions | 10.1-10.6 | Loading, Empty States, Tooltips, Number Anims, Responsive, Shortcuts |
| 11: Quality/Performance | 11.1-11.5 | Cache Strategy, Virtual Scroll, WebSocket, A11y, Error Boundaries |
| 12: Export/Sharing | 12.1-12.3 | Table Export, Chart Capture, Report Generation |

**Total: 12 Epics, 47 Stories, ~280 Acceptance Criteria**

---

> End of UX.md
> "Ship something so good, users involuntarily show it to the person sitting next to them."






