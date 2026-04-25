# Watchlist — Premium UX Design Notes

> "The Watchlist is not a data table. It is the user's portfolio attention
> span made visible. Every interaction must reward the glance."

This document is the single source of truth for the Watchlist panel's
interaction model and visual grammar. Any future change to the panel
should be reconciled with these principles. Design beats consensus —
when in doubt, defer to the principles below.

## 1. Philosophy

The Watchlist lives inside the Signals page. Users arrive here not to
hunt data but to **re-acquaint themselves with the names they care
about**. The UX must respect that:

- Content is the hero. Controls defer to tickers and their signals.
- The panel rewards a 1-second glance with a meaningful verdict
  ("how is my list doing right now?") before demanding any interaction.
- Every control earns its pixels. If a control cannot justify its
  presence in a one-line summary it belongs in a drawer, not the main
  view.
- Motion conveys state change. Motion is never decoration.

## 2. Apple-HIG principles applied

| Principle | Concrete expression in the Watchlist |
|-----------|---------------------------------------|
| Deference | Chrome is near-transparent. One accent per semantic state. |
| Clarity   | Typography scale is fixed. Tabular numerals everywhere. |
| Depth     | Subtle shadows and translucent backgrounds imply layering. |
| Directness| Clicking a count filters the list to the thing counted. |
| Feedback  | Active controls glow, hover lifts, press compresses. |
| Consistency| Same color means the same thing in every location. |

## 3. Hierarchy (top to bottom)

The panel is a four-tier stack. Each tier is *earned* — nothing
renders unless it carries information.

### Tier 1 — Insight Bar (always visible)
One horizontal line at the top. The user's headline verdict:

```
[ Star ]  Watchlist · 5 of 12 bullish · 2 all greens · 1 missing     [ + Add ]
```

- "Watchlist" is the anchor label.
- Each count phrase ("5 of 12 bullish", "2 all greens", "1 missing")
  is a clickable inline chip. Clicking filters the list to that
  bucket; clicking again clears.
- The "+ Add" button on the right is the *only* way to reveal the
  manage drawer. No always-open manage card.

### Tier 2 — Segmented Control (only when symbols > 0)
A single segmented control replaces the previous dual pill groups
plus a separate missing button:

```
[ All  |  Bullish  |  Bearish  |  Greens  |  Reds  |  Missing ]
```

- Six segments, maximum. Never more.
- Each segment glows in its own semantic color when active.
- `Neutral` and `Mixed` are *not* first-class — they belong in the
  Refine popover. Power users can find them; novices do not need
  them.

### Tier 3 — Refine strip (collapsed by default)
A subtle "Refine" trigger sits below the segmented control. When
expanded it reveals:

- Ticker/name search (focusable with `/`)
- Sector select (hidden if only one sector is present)
- Sort select (Signal / Momentum / Risk / Alphabetical)
- Clear button (visible only when refinements are active)

The Refine strip is **collapsed by default**. The main view never
shows more than two rows of controls at rest.

### Tier 4 — Ticker table
Full-width table (re-uses `AllAssetsTable` with `disablePagination`).
Zebra striping is intentionally absent — Apple tables are flat, with
hairline separators at 1px `rgba(255,255,255,0.04)`.

### Tier 5 — Manage drawer (hidden until "+ Add")
Click the "+ Add" button in the insight bar to slide up a drawer
containing:

- A large, focus-first text input for ticker entry.
- A grid of color-coded chips, one per tracked symbol.
- Each chip shows a colored dot (bull/bear) or amber warning (missing).
- Remove via the `×` button on the chip. Future: swipe-left.

When `symbols.length === 0` the drawer is auto-opened and the insight
bar collapses to a single "Add your first ticker" call to action.

## 4. Micro-interactions

Motion is the invisible UI. All transitions use the same easing:
`cubic-bezier(0.16, 1, 0.3, 1)` (iOS "standard curve"). Durations:

- Hover lift: 140 ms
- Press scale: 90 ms
- Pill switch: 180 ms
- Drawer open: 260 ms
- Filter change (row fade): 180 ms

Specific affordances:

- **Pill hover**: `transform: translateY(-1px)`, subtle glow.
- **Pill active**: `box-shadow: inset 0 0 0 1px <accent>`, color fill
  at 12% alpha of accent.
- **Chip remove**: row fades and height collapses; the list re-flows
  with a 220 ms transition.
- **Insight-bar chips**: numbers use `tabular-nums` so transitions
  don't cause character jitter.
- **Sticky header**: filter strip sticks to top of the scroll
  container with a `backdrop-filter: blur(14px)` plate.

## 5. Keyboard shortcuts (opt-in, surfaced in a tiny footer)

| Key | Action |
|-----|--------|
| `/` | Focus the ticker search |
| `A` | Toggle the Manage drawer |
| `Esc` | Close Manage drawer / clear search |
| `1..6` | Jump to segment (All / Bullish / Bearish / Greens / Reds / Missing) |

Shortcuts are rendered as a tiny `⌘` hint row at the bottom of the
panel, visible only when the panel has focus.

## 6. Empty state

First-time experience is critical. When `symbols.length === 0`:

- Hero star icon, gentle violet glow.
- Headline: "Track the tickers you care about."
- Sub: "Signals update live. Your list persists across restarts."
- Three **suggested-ticker chips**: `AAPL`, `NVDA`, `SPY`. Clicking a
  chip adds it and smoothly reveals the populated watchlist.
- The Manage drawer is auto-expanded with the ticker input focused.

## 7. Typography scale (locked)

| Role | Size | Weight | Treatment |
|------|------|--------|-----------|
| Insight headline | 13 px | 500 | secondary color, counts in accent |
| KPI value        | 22 px | 600 | tabular-nums, accent color |
| Label            | 10 px | 600 | uppercase, tracking-[0.1em] |
| Table row        | 13 px | 400 | tabular-nums |
| Chip             | 11 px | 500 | tabular-nums |
| Footer hint      | 10 px | 400 | secondary |

## 8. Color palette (locked)

| Role     | Accent     | Tint       | Border         |
|----------|------------|------------|----------------|
| Bullish  | `#34d399`  | `#6ee7b7`  | rgba(52,211,153,0.28) |
| Bearish  | `#f87171`  | `#fca5a5`  | rgba(248,113,113,0.28) |
| Greens   | `#10b981`  | `#6ee7b7`  | rgba(110,231,183,0.26) |
| Reds     | `#ef4444`  | `#fca5a5`  | rgba(252,165,165,0.26) |
| Neutral  | `#cbd5e1`  | `#cbd5e1`  | rgba(148,163,184,0.22) |
| Missing  | `#fbbf24`  | `#fcd34d`  | rgba(251,191,36,0.28) |
| Violet   | `#a78bfa`  | `#c4b5fd`  | rgba(167,139,250,0.25) |

Violet is reserved for *actions* (add button, primary CTAs). It must
never be used to signal market state.

## 9. Spacing grid

All paddings, margins, gaps must snap to the 4 px grid. Common values:

- Card padding: `16 px` (p-4)
- Inner row gap: `8 px` (gap-2)
- Pill padding: `10 px / 4 px`
- Section separation: `16 px` vertical

## 10. Sorting contract (important)

The watchlist has two sorting surfaces that historically conflicted:

1. The **Sort dropdown** in the Refine popover (Signal / Momentum /
   Risk / Alpha). This is a *preset* — a one-click way to apply a
   common order.
2. **Column header clicks** in the table. These mutate the parent's
   `sortLevels`. Historically these did not visibly re-order watchlist
   rows because the watchlist's filter memo ignored `sortLevels`.

**The new contract** resolves this:

- The Sort dropdown is the default source of truth.
- When the user clicks a column header, we mark an internal
  `wlSortOverride` flag and **preserve the incoming `allRows` order**
  (which the parent has already sorted by `sortLevels`). The
  watchlist's own dropdown becomes a no-op until override is cleared.
- Changing the Sort dropdown clears the override.

This means: *both surfaces work*, they do not fight, and the user's
most-recent intent always wins.

## 11. Non-goals

These are intentionally out of scope — do not add without a fresh
design review:

- Drag-to-reorder tickers.
- Per-ticker notes or tags.
- Multiple named watchlists.
- Custom alerts on threshold breach.

## 12. How to extend this panel

Before adding a new control ask:

1. Does it fit into the Insight Bar summary? If yes, add it there.
2. Can it be merged into the existing segmented control? If yes, do.
3. Is it a refinement? If yes, put it in the Refine popover.
4. Is it a destructive/administrative action? If yes, put it in the
   Manage drawer.
5. None of the above → it probably does not belong in the panel.

## Change log

- 2026-04-23 — Initial design doc. Introduces Insight Bar, unified
  segmented control, Refine popover, Manage drawer, sorting contract.
