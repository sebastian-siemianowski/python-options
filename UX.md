# UX.md -- Cosmic Edition: Bayesian Signal Engine Dashboard

> Written by: Product Owner (Craftsman)
> Date: 3 April 2026
> Aesthetic: Cosmic Apple Gradient -- deep void, violet nebula, aurora glass
> Philosophy: Every pixel radiates light from within. Every surface bends
> space-time around the data it carries. Every interaction leaves
> a luminous trail that whispers "this was built by someone who cares."
> North Star: "Would an Apple design engineer lean forward in their chair,
> pull out their phone to photograph this screen, and text their team
> 'we need to talk about this'? If not, we are not done."

---

## Design North Star

This is not a dashboard. This is a **celestial observatory** for professional
quantitative traders managing 100+ assets with Bayesian Model Averaging.

The visual language draws from deep space: void-black canvases punctuated by
violet nebula gradients, aurora-glass surfaces that refract light at their
edges, and data that glows with bioluminescent intensity against the cosmic
dark. Every surface exists in a z-space of light emission -- not shadow.
Elevation is communicated by how much light a surface radiates, not by how
much shadow it casts.

Every surface must satisfy four simultaneous demands:

1. **Information Density** -- A single glance communicates more than
   competitors convey in three clicks. Data is the starfield; the UI is
   the telescope that brings it into focus.
2. **Emotional Clarity** -- The user never wonders "is this good or bad?"
   Violet calm, emerald confidence, and crimson urgency speak before the
   conscious mind processes a single number.
3. **Muscle Memory** -- After one week of daily use, the user navigates
   entirely by keyboard and spatial memory. The mouse becomes optional.
   The interface becomes an extension of thought.
4. **Cosmic Beauty** -- Every screen is a photograph worth sharing. The
   gradients, the glass, the light -- they create an emotional response
   that transcends utility. Users do not just use this tool. They show it
   to people. They drool.

---

## Design Language Specification

### The Cosmic Palette

The palette is born from deep space: a void so dark it absorbs everything,
punctuated by nebula-violet light that bleeds from the edges of every
interactive surface. Think: the exact moment a star ignites inside a
molecular cloud.

#### Foundation Colors

| Token                 | Value                     | Usage                                        |
|-----------------------|---------------------------|----------------------------------------------|
| `--void`              | `#030014`                 | Deepest canvas. The cosmic void itself.      |
| `--void-surface`      | `#0a0a23`                 | Card/panel surfaces. Barely lighter than void.|
| `--void-raised`       | `#110f2e`                 | Elevated elements, modals, popovers.         |
| `--void-hover`        | `#16133a`                 | Hover state backgrounds. Violet emerges.     |
| `--void-active`       | `#1c1845`                 | Active/pressed states. Deep amethyst.        |

#### Nebula Gradients (The Signature)

These gradients are the soul of the interface. They flow across surfaces
like gas clouds illuminated by distant stars. Every gradient uses at least
three stops to prevent banding and create organic depth.

| Token                     | Value                                             | Usage                              |
|---------------------------|---------------------------------------------------|------------------------------------|
| `--gradient-nebula`       | `linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)` | Primary card backgrounds   |
| `--gradient-aurora`       | `linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%)` | Hero sections, feature cards |
| `--gradient-cosmic-glow`  | `radial-gradient(ellipse at 30% 50%, rgba(139,92,246,0.15) 0%, transparent 70%)` | Ambient background glow |
| `--gradient-violet-shift` | `linear-gradient(160deg, #2d1b69 0%, #11001c 50%, #0c1445 100%)` | Navigation surfaces |
| `--gradient-deep-space`   | `linear-gradient(180deg, #030014 0%, #0a0528 30%, #110f2e 100%)` | Page-level backgrounds |
| `--gradient-signal-bull`  | `linear-gradient(135deg, #064e3b 0%, #065f46 50%, #047857 100%)` | Bullish accent surfaces |
| `--gradient-signal-bear`  | `linear-gradient(135deg, #4c0519 0%, #6b0f2a 50%, #881337 100%)` | Bearish accent surfaces |
| `--gradient-glass-edge`   | `linear-gradient(135deg, rgba(139,92,246,0.2) 0%, rgba(59,130,246,0.1) 50%, rgba(139,92,246,0.05) 100%)` | Glass border shimmer |

#### Interactive Colors

| Token                  | Value       | Usage                                           |
|------------------------|-------------|--------------------------------------------------|
| `--accent-violet`      | `#8B5CF6`  | Primary interactive. Links, focus, active state. |
| `--accent-violet-soft` | `#7C3AED`  | Hover variant, slightly deeper.                  |
| `--accent-violet-glow` | `rgba(139,92,246,0.4)` | Focus rings, ambient glow halos.    |
| `--accent-indigo`      | `#6366F1`  | Secondary interactive. Tabs, toggles.            |
| `--accent-cyan`        | `#22D3EE`  | Informational accents, live indicators.          |
| `--accent-emerald`     | `#34D399`  | Bullish signals, success, passing metrics.       |
| `--accent-rose`        | `#FB7185`  | Bearish signals, errors, failing metrics.        |
| `--accent-amber`       | `#FBBF24`  | Caution, stale data, elevated risk.              |
| `--accent-fuchsia`     | `#E879F9`  | Rare highlights, premium indicators, celebration.|

#### Text Hierarchy

| Token              | Value       | Usage                                      |
|--------------------|-------------|---------------------------------------------|
| `--text-luminous`  | `#f8fafc`   | Hero numbers, primary headings. Almost white.|
| `--text-primary`   | `#e2e8f0`   | Body headings, values, labels.              |
| `--text-secondary` | `#94a3b8`   | Descriptions, secondary labels.             |
| `--text-muted`     | `#475569`   | Timestamps, footnotes, disabled.            |
| `--text-violet`    | `#C4B5FD`   | Violet-tinted text for special emphasis.    |

#### Border & Surface

| Token                  | Value                      | Usage                    |
|------------------------|----------------------------|--------------------------|
| `--border-void`        | `rgba(139,92,246,0.08)`    | Barely visible card edges|
| `--border-glow`        | `rgba(139,92,246,0.25)`    | Focus rings, active cards|
| `--border-aurora`      | `rgba(34,211,238,0.15)`    | Info-accent borders      |
| `--glass-surface`      | `rgba(10,10,35,0.80)`      | Glass morphism base      |
| `--glass-blur`         | `blur(20px) saturate(1.4)` | Glass backdrop filter    |

### Typography Scale

Font: **SF Pro Display** (Apple system), falling back to Inter, then system sans.

| Token       | Size   | Weight | Tracking | Line Height | Usage                      |
|-------------|--------|--------|----------|-------------|----------------------------|
| `display`   | 40px   | 700    | -0.025em | 1.1         | Hero numbers, page-level KPIs |
| `heading-1` | 28px   | 600    | -0.015em | 1.2         | Page titles                |
| `heading-2` | 20px   | 600    | -0.01em  | 1.3         | Section headers            |
| `heading-3` | 15px   | 500    | -0.005em | 1.4         | Card titles                |
| `body`      | 13px   | 400    | 0.01em   | 1.5         | Primary readable text      |
| `caption`   | 11px   | 500    | 0.04em   | 1.4         | Labels, badges, metadata   |
| `mono`      | 12px   | 400    | 0.02em   | 1.5         | Numbers, code, tickers     |

Numbers use **tabular figures** (`font-variant-numeric: tabular-nums`) so
columns of numbers align perfectly without monospace. This is non-negotiable.

### Spacing Scale

4px baseline grid: 4, 8, 12, 16, 20, 24, 32, 40, 48, 64, 80.

All spacing between elements uses this grid. No magic numbers. No 5px.
No 13px. Every gap is divisible by 4. This creates the invisible rhythm
that makes Apple interfaces feel "right" without the user knowing why.

### Motion Principles

Motion follows Apple's Human Interface Guidelines: physics-based, purposeful,
never decorative. Every animation has a job. Spring physics create organic
feel. Ease curves prevent mechanical feel.

| Type         | Duration | Easing                                    | Usage                              |
|--------------|----------|-------------------------------------------|------------------------------------|
| `micro`      | 120ms    | `cubic-bezier(0.2, 0, 0, 1)`             | Hover, focus, toggle, button press |
| `standard`   | 250ms    | `cubic-bezier(0.2, 0, 0, 1)`             | Panel open, tab switch, sidebar    |
| `expressive` | 400ms    | `cubic-bezier(0.16, 1, 0.3, 1)`          | Page enter, modal reveal, charts   |
| `spring`     | 500ms    | `cubic-bezier(0.34, 1.56, 0.64, 1)`      | Celebratory, gauge needle, spring  |
| `cosmic`     | 800ms    | `cubic-bezier(0.22, 0.68, 0, 1.71)`      | Ambient glow pulse, nebula drift   |

### Elevation System (Light-Based)

Unlike Material Design's shadow-based elevation, this system uses **light
emission** for depth. Higher elements glow more. They emit violet-tinged
light from their edges. This creates the illusion that cards are floating
screens projecting information into the void.

| Level | Effect                                                    | Usage              |
|-------|-----------------------------------------------------------|--------------------|
| 0     | none                                                      | Flat inline content|
| 1     | `0 0 0 1px var(--border-void), 0 1px 3px rgba(0,0,0,0.4)` | Cards at rest |
| 2     | `0 0 0 1px var(--border-glow), 0 4px 16px rgba(139,92,246,0.08), 0 1px 3px rgba(0,0,0,0.3)` | Cards on hover |
| 3     | `0 0 0 1px var(--border-glow), 0 8px 32px rgba(139,92,246,0.12), 0 0 80px rgba(139,92,246,0.05)` | Modals, drawers |
| 4     | `0 0 0 1px var(--border-glow), 0 16px 64px rgba(139,92,246,0.15), 0 0 120px rgba(139,92,246,0.08)` | Command palette, full-screen overlays |

### Glass Morphism Specification

Every card surface uses glass morphism to create depth without visual weight:

```css
.cosmic-glass {
  background: var(--glass-surface);
  backdrop-filter: var(--glass-blur);
  -webkit-backdrop-filter: var(--glass-blur);
  border: 1px solid var(--border-void);
  border-radius: 16px;
  position: relative;
  overflow: hidden;
  transition: all 250ms cubic-bezier(0.2, 0, 0, 1);
}

/* The violet edge shimmer -- a gradient border that catches light */
.cosmic-glass::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1px;
  background: var(--gradient-glass-edge);
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  opacity: 0;
  transition: opacity 250ms cubic-bezier(0.2, 0, 0, 1);
}

.cosmic-glass:hover::before {
  opacity: 1;
}

.cosmic-glass:hover {
  border-color: var(--border-glow);
  box-shadow: 0 0 0 1px var(--border-glow),
              0 4px 16px rgba(139,92,246,0.08),
              0 1px 3px rgba(0,0,0,0.3);
  transform: translateY(-1px);
}
```

### Ambient Glow Specification

Each page has a **cosmic ambient glow** -- a large, soft, slow-moving
radial gradient behind the content that gives the page a living, breathing
personality. The glow shifts position and color based on page context,
creating the feeling that the void behind the content is alive.

| Page        | Glow Position   | Glow Color                           | Meaning                |
|-------------|-----------------|--------------------------------------|------------------------|
| Overview    | top-center      | `rgba(139,92,246,0.08)`              | Neutral, observatory   |
| Signals     | top-right       | shifts emerald/rose based on bull/bear ratio | Market sentiment |
| Risk        | center          | shifts amber/rose based on temperature | Risk heat             |
| Charts      | top-left        | `rgba(99,102,241,0.06)`              | Analytical, cool       |
| Tuning      | bottom-center   | `rgba(139,92,246,0.05)`              | Engine room, subtle    |
| Arena       | center          | `rgba(232,121,249,0.06)`             | Competition, fuchsia   |
| Diagnostics | top-right       | `rgba(34,211,238,0.05)`              | Laboratory, cyan       |

The glow animates with `cosmic` motion (800ms) on page entry, expanding
from 0% to 100% opacity, and drifts slowly (20s cycle, `ease-in-out`)
in a figure-eight pattern, creating a living background that breathes.

```css
@keyframes cosmic-drift {
  0%, 100% { background-position: 30% 50%; }
  25%      { background-position: 40% 30%; }
  50%      { background-position: 60% 50%; }
  75%      { background-position: 35% 65%; }
}

.page-canvas {
  background: var(--void);
  position: relative;
}

.page-canvas::before {
  content: '';
  position: absolute;
  inset: 0;
  background: var(--gradient-cosmic-glow);
  animation: cosmic-drift 20s ease-in-out infinite;
  pointer-events: none;
}
```

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

> **Vision**: The application shell is a cosmic vessel. The sidebar is not a menu --
> it is a **navigation constellation** where each page is a star, and the active
> page burns brighter than the rest. A new user grasps the entire spatial map
> within 2 seconds. The shell whispers ambient intelligence from every page
> without requiring a single click.

---

## Story 1.1: Intelligent Sidebar with Violet Constellation Navigation

**As a** quantitative trader navigating between 10+ pages,
**I want** a sidebar that communicates page health at a glance without clicking,
**so that** I never waste a click visiting a page only to discover nothing changed.

### Visual Design

The sidebar is a vertical strip of `--gradient-violet-shift` glass, floating
against the void canvas. Each navigation item is a glass pill that responds
to proximity and hover with violet luminance. The active page does not merely
highlight -- it **ignites**: a radial violet glow bleeds from behind the
active item into the void, like a star seen through a telescope.

The sidebar's top section features the application wordmark rendered in
`--text-luminous` with a subtle violet gradient shimmer on the text itself,
animated on a 6-second cycle with `cosmic` motion.

### Acceptance Criteria

- [ ] AC-1: Each navigation item displays a **live micro-indicator** to the right
  of the label showing the page's current state. The indicator is a tiny pill
  badge (20px wide, 14px tall) with `caption` typography, rendered in a color
  matching the page's health:
  - Signals: count of Strong Buy + Strong Sell signals (e.g., "12") in
    `--accent-emerald` if majority bullish, `--accent-rose` if bearish
  - Risk: temperature value with colored dot (emerald/amber/rose)
  - Charts: nothing (navigation-only page -- clean and minimal)
  - Tuning: PIT pass rate as percentage, emerald if >= 80%, amber if >= 60%
  - Data: count of stale files (rose pill if > 0, hidden if 0)
  - Arena: count of models in safe storage (`--accent-fuchsia` badge)
  - Services: animated pulse dot -- emerald if all OK, rose if any down
  - Diagnostics: calibration failure count (`--accent-amber` if > 0)
- [ ] AC-2: Micro-indicators update via the same React Query cache as the pages
  themselves -- **zero additional API calls**. The sidebar subscribes to
  existing query keys (signalSummary, riskSummary, servicesHealth, etc.)
  and derives its indicators from cached data.
- [ ] AC-3: Hovering a nav item reveals a **tooltip card** (elevation 3, cosmic
  glass) showing a 3-line summary of that page's current state, with the
  last-updated timestamp in `--text-muted`. The tooltip appears after 400ms
  delay and animates with `expressive` motion (scale 0.96 -> 1.0, fade in).
  The tooltip's background uses `--gradient-nebula` for a rich depth feel.
- [ ] AC-4: The active page indicator is a **violet aurora bar** (4px wide) on
  the left edge of the nav item: a vertical gradient from `--accent-violet`
  at full opacity in the center, fading to transparent at top and bottom.
  Behind the active item, a `radial-gradient(circle at -10% 50%,
  rgba(139,92,246,0.12) 0%, transparent 60%)` creates a soft glow that
  bleeds into the page canvas -- the star-through-telescope effect.
- [ ] AC-5: Sidebar collapses to icon-only mode (48px wide) on screens < 1280px.
  In collapsed mode, hovering an icon expands only that item with label +
  indicator as a floating glass pill (elevation 3) that appears to slide out
  from the sidebar. The expansion uses `spring` motion for a delightful bounce.
- [ ] AC-6: Collapsed/expanded preference persists in `localStorage`. The
  sidebar remembers the user's choice across sessions.
- [ ] AC-7: Keyboard shortcut `Cmd+B` toggles sidebar collapse with a smooth
  200ms width transition. Icons crossfade with labels during the transition.
- [ ] AC-8: The sidebar's glass surface uses `--gradient-violet-shift` as its
  base, with `backdrop-filter: blur(24px) saturate(1.6)`. The right edge
  has a 1px border of `--border-void` that subtly catches the ambient glow
  from the page content.
- [ ] AC-9: The UX of this sidebar -- the aurora active indicator, the live
  micro-badges, the glass tooltips, the violet glow bleeding into the void --
  gives users so much ambient intelligence without a single click that they
  would absolutely fall in love with the sidebar and drool over how much
  context radiates from this single navigation strip. Apple engineers would
  study this sidebar and ask "how did they make a nav feel this alive?"

---

## Story 1.2: Command Palette (Cosmic Quick Navigation)

**As a** power user who values speed above all else,
**I want** a command palette accessible from any page via `Cmd+K`,
**so that** I can jump to any asset, page, or action in under 2 keystrokes.

### Visual Design

The command palette is a floating glass rectangle against a `rgba(3,0,20,0.85)`
backdrop with `backdrop-filter: blur(40px)`. It appears to emerge from the
void itself: the blur is heavier than any other element in the system,
creating the sense of a portal opening in the cosmic fabric. The search
input has a violet gradient underline that pulses gently while waiting for
input. Results appear as glass rows with violet-glow hover states.

### Acceptance Criteria

- [ ] AC-1: `Cmd+K` opens a centered modal overlay (max-width 560px) with a search
  input auto-focused. The overlay background dims to `rgba(3,0,20,0.85)` with
  `backdrop-filter: blur(40px) saturate(1.8)`. Animation: scale from 0.94 to
  1.0 + opacity from 0 to 1, 300ms with `expressive` easing. The palette
  rises from the cosmic void like a luminous portal.
- [ ] AC-2: The search input sits atop a violet gradient underline (2px tall):
  `linear-gradient(90deg, transparent 0%, var(--accent-violet) 50%, transparent 100%)`.
  The gradient pulses with 40% to 80% opacity on a 2s cycle while idle,
  stopping at 100% once the user types. An SVG search icon on the left
  renders in `--accent-violet` at 60% opacity.
- [ ] AC-3: Search results appear as the user types (debounced 80ms) in categorized
  groups with `--text-violet` section headers:
  - **Pages**: "Overview", "Signals", "Risk Dashboard", etc. Each with its
    page icon in `--accent-violet`
  - **Assets** (from signalSummary cache): ticker + sector + current signal
    badge. The signal badge pill uses `--gradient-signal-bull` or
    `--gradient-signal-bear` based on direction.
  - **Actions**: "Refresh Data", "Retune Models", "Compute Signals" with
    action icons in `--accent-cyan`
- [ ] AC-4: Each result row has a `--void-hover` background on hover with a
  1px left-border in `--accent-violet` that fades in over 120ms. Arrow keys
  navigate results. `Enter` activates. `Esc` closes with reverse animation.
- [ ] AC-5: Selecting an asset navigates to `/charts/{SYMBOL}` with a seamless
  page transition. The palette fades out (150ms) before the page navigates.
- [ ] AC-6: Selecting an action triggers the corresponding API call and shows
  a toast notification with progress (see Story 1.4).
- [ ] AC-7: Most recently used items appear at the top of the list before typing,
  under a "Recent" section header. Recent items persist in `localStorage`
  (max 8). Each recent item has a subtle clock icon in `--text-muted`.
- [ ] AC-8: The palette's bottom edge shows a keyboard shortcut hint bar:
  "Enter to select | Tab to autocomplete | Esc to close" in `caption`
  typography, `--text-muted`, against a `--void-active` background strip.
- [ ] AC-9: The command palette feels so instantaneous, so visually polished,
  and so intelligent that users would absolutely fall in love with its
  responsiveness and drool over the cosmic portal that lets them navigate
  the entire 100+ asset universe in milliseconds. Apple engineers would
  recognize the palette as the spiritual successor to Spotlight.

---

## Story 1.3: Breadcrumb Trail with Temporal Aurora

**As a** user drilling into a specific asset from the Signals page,
**I want** a breadcrumb trail that shows my navigation path and how fresh the data is,
**so that** I always know where I am and whether the data I'm looking at is current.

### Visual Design

The breadcrumb bar is a thin glass strip (40px tall) that sits below the page
header. Breadcrumb segments are connected by subtle violet chevron icons that
feel like starlight path markers. The data freshness badge on the right edge
pulses with aurora colors -- emerald for fresh, amber for aging, rose for stale --
creating an ambient vitality indicator visible in peripheral vision.

### Acceptance Criteria

- [ ] AC-1: A breadcrumb bar appears below the page header when the user is deeper
  than the top-level page (e.g., Charts > AAPL, Diagnostics > PIT Calibration).
  The bar uses `--glass-surface` background with `--glass-blur` and a 1px
  bottom border of `--border-void`.
- [ ] AC-2: Each breadcrumb segment is clickable. Segments use `--text-secondary`
  with a violet gradient underline on hover (appearing left-to-right over 200ms).
  The separator between segments is a small (8px) chevron SVG in `--text-muted`.
- [ ] AC-3: The final breadcrumb segment (current context) displays in
  `--text-luminous` with font-weight 600 and a subtle text-shadow of
  `0 0 12px rgba(139,92,246,0.3)` -- it literally glows against the void.
  Parent segments display in `--text-secondary`.
- [ ] AC-4: A **data freshness badge** appears at the right end of the breadcrumb
  bar showing the oldest data dependency on the current view:
  - Emerald badge "Live" if < 60 seconds old, with a gentle pulse animation
  - Amber badge "2m ago" / "5m ago" etc. if stale, with a slow pulse
  - Rose badge "Stale: 2h" with a faster pulse if critically stale
  The badge background uses the corresponding `--gradient-signal-*` for
  depth, not a flat color.
- [ ] AC-5: Clicking the freshness badge opens a dropdown (elevation 3, cosmic
  glass, `--gradient-nebula` background) showing all data dependencies for
  the current view with their individual ages, colored status dots, and a
  "Refresh All" button styled with `--accent-violet` gradient background.
- [ ] AC-6: Breadcrumbs animate in with a subtle left-to-right stagger (40ms per
  segment) using `standard` motion. Each segment fades in + slides 8px right.

---

## Story 1.4: Toast Notification System with Cosmic Accents

**As a** user who triggers background actions (refresh data, retune models),
**I want** non-blocking toast notifications that track progress,
**so that** I can continue working while background tasks complete.

### Visual Design

Toasts are floating cosmic glass cards that slide in from the right edge of
the viewport. Each variant has a colored left accent -- a 3px vertical
gradient strip that matches the toast severity. The glass morphism ensures
toasts feel layered above the content without fully obscuring it. The
violet-tinted blur creates visual continuity with the overall cosmic theme.

### Acceptance Criteria

- [ ] AC-1: Toasts appear in the bottom-right corner, stacked vertically with
  8px gap between them. Maximum 4 visible simultaneously. Each toast is a
  cosmic glass card (elevation 3) with `border-radius: 12px`.
- [ ] AC-2: Four toast variants exist, each with a 3px left-border gradient:
  - **Info**: Left border `linear-gradient(180deg, var(--accent-violet) 0%,
    var(--accent-indigo) 100%)`, violet tint icon
  - **Success**: Left border `linear-gradient(180deg, var(--accent-emerald) 0%,
    #059669 100%)`, emerald checkmark icon, auto-dismiss after 4s with a
    shrinking progress bar at the bottom showing remaining time
  - **Warning**: Left border `linear-gradient(180deg, var(--accent-amber) 0%,
    #D97706 100%)`, amber alert icon, auto-dismiss after 6s
  - **Error**: Left border `linear-gradient(180deg, var(--accent-rose) 0%,
    #E11D48 100%)`, rose X icon, persists until manually dismissed, with a
    subtle rose glow: `box-shadow: 0 0 20px rgba(251,113,133,0.08)`
- [ ] AC-3: **Progress toasts** (for long-running tasks) show:
  - Title + description in `--text-primary` / `--text-secondary`
  - Animated progress bar: a thin (3px) gradient strip
    `linear-gradient(90deg, var(--accent-violet) 0%, var(--accent-cyan) 100%)`
    that fills left-to-right. If indeterminate, the gradient slides continuously.
  - Elapsed time counter in `mono` typography, `--text-muted`
  - Cancel button: ghost-style, `--text-secondary`, violet on hover
- [ ] AC-4: Toasts animate in from the right (translate-x: 120% to 0) with
  `expressive` motion and animate out by fading + sliding down (150ms).
  Stacking animation: existing toasts slide up smoothly to make room for new ones.
- [ ] AC-5: Each toast has an `aria-live="polite"` attribute for screen readers.
- [ ] AC-6: Clicking a toast with an associated page (e.g., "Tuning complete for
  NVDA") navigates to that page. The toast shows a subtle arrow icon on the
  right indicating it is clickable.
- [ ] AC-7: Toast history is accessible via a bell icon in the sidebar footer.
  The bell shows an unread count badge (rose/violet pill). Clicking it opens
  a slide-out panel (elevation 3, `--gradient-nebula` background) showing the
  last 20 notifications with timestamps, grouped by "Today" / "Earlier".
- [ ] AC-8: The toast notification system is so visually refined -- the glass
  surfaces, the gradient accents, the smooth animations -- that users would
  absolutely fall in love with receiving updates and drool over how even
  background task feedback feels like a luxury experience.

---

## Story 1.5: Global Ambient Status Strip (The Cosmic Horizon)

**As a** trader who needs to feel the market's pulse without looking away from my work,
**I want** a thin ambient status strip along the top of the viewport,
**so that** system health and market regime are always peripherally visible.

### Visual Design

The status strip is a 3px-tall horizon line at the very top of the viewport --
the edge where the cosmic void meets the viewport boundary. Its color is a
flowing gradient that shifts based on market regime, creating the impression
that the entire application bathes in the light of the current market state.
When the market is calm, the horizon glows soft violet. When stressed, it
burns amber-to-rose. In crisis, it pulses crimson like a distant supernova.

### Acceptance Criteria

- [ ] AC-1: A 3px-tall strip spans the full width of the viewport at the very top,
  above the sidebar. The strip renders as a horizontal gradient:
  - Calm (< 0.3): `linear-gradient(90deg, transparent 0%, var(--accent-violet) 20%,
    var(--accent-indigo) 50%, var(--accent-violet) 80%, transparent 100%)` at 50% opacity
  - Elevated (0.3-0.7): `linear-gradient(90deg, transparent 0%, var(--accent-amber) 20%,
    var(--accent-violet) 50%, var(--accent-amber) 80%, transparent 100%)` at 60% opacity
  - Stressed (0.7-1.2): `linear-gradient(90deg, transparent 0%, var(--accent-rose) 20%,
    var(--accent-amber) 50%, var(--accent-rose) 80%, transparent 100%)` at 50% opacity
  - Crisis (> 1.2): `linear-gradient(90deg, transparent 0%, var(--accent-rose) 30%,
    #E11D48 50%, var(--accent-rose) 70%, transparent 100%)` at 80% with a slow
    pulse animation (2s cycle, opacity oscillating 60-100%). The pulse creates
    an unmistakable "something is wrong" ambient signal.
- [ ] AC-2: The strip gradient transitions smoothly over 1000ms when the regime
  changes, using color interpolation in OKLCH color space for perceptually
  smooth transitions. The gradient position also shifts (sliding left-to-right
  over 1000ms) to make the transition visible even in peripheral vision.
- [ ] AC-3: Hovering the strip expands it to 36px height (300ms `spring` animation)
  revealing a cosmic glass bar showing: "Risk: 0.42 -- Elevated | 147 Assets |
  12 Strong Buys | Services: OK" in `caption` typography. Each metric has its
  own colored status dot. The expanded bar uses `--gradient-nebula` background
  with glass blur.
- [ ] AC-4: The expanded strip text uses `caption` typography in `--text-secondary`
  with metric values in `--text-primary`. Status dots pulse gently in their
  semantic color (emerald/amber/rose).
- [ ] AC-5: This ambient horizon strip is so subtle yet so alive -- the flowing
  gradient, the regime-responsive color, the expandable context bar -- that
  users would absolutely fall in love with the peripheral awareness it provides
  and drool over always knowing the market regime without a single interaction.
  Apple engineers would recognize this as the kind of detail that separates
  "well-designed" from "obsessed over."

---

# EPIC 2: Dashboard (Overview Page) -- The Celestial Briefing

> **Vision**: The Overview page is not a landing page -- it is a **morning briefing
> transmitted from orbit**. A trader opens this at 9:25 AM and within 8 seconds
> absorbs: what changed overnight, what demands attention, and what the models are
> most confident about today. The page is a single, unified visual composition --
> not a collection of cards. Every element flows into the next, connected by the
> cosmic gradient that bathes the entire canvas.

---

## Story 2.1: Morning Briefing Hero Card with Nebula Background

**As a** trader starting my day,
**I want** a hero section that immediately communicates what changed since my last visit,
**so that** I can prioritize my attention on new developments rather than re-scanning
familiar information.

### Visual Design

The Briefing Card is the first thing the eye touches. It spans the full content
width, using `--gradient-aurora` as its background with a subtle animated
`--gradient-cosmic-glow` overlay that drifts slowly (20s cycle). The card has
no visible border -- instead, a 1px gradient border (`--gradient-glass-edge`)
that only becomes visible on hover. The card's content is arranged in three
columns separated by thin vertical dividers that fade at both ends
(`linear-gradient(180deg, transparent 0%, rgba(139,92,246,0.15) 50%, transparent 100%)`).

The hero number (today's highest-conviction ticker) uses `display` typography
with a subtle `background: linear-gradient(135deg, var(--text-luminous) 0%,
var(--accent-violet) 100%); -webkit-background-clip: text; -webkit-text-fill-color:
transparent;` -- gradient text that makes the ticker shimmer like a star.

### Acceptance Criteria

- [ ] AC-1: The top of the Overview page displays a **Briefing Card** spanning
  full width (cosmic glass, elevation 2, `--gradient-aurora` background) with
  three columns divided by fading violet separator lines:
  - **Left**: "Since Last Visit" -- count of signal changes with directional
    arrows. Shows the most impactful change first (e.g., "NVDA upgraded to
    Strong Buy") with the ticker in `--accent-violet` and the direction change
    rendered as a gradient badge (bull/bear gradient). "Last visit" tracked via
    `localStorage` timestamp comparison against `signals.computed_at`.
  - **Center**: "Today's Conviction" -- the single highest-conviction signal
    across all assets. Ticker in gradient text (`display` typography), signal
    direction as a large badge (`--gradient-signal-bull/bear`), expected return
    in `heading-2` size with bright color, probability as a thin arc gauge
    (emerald fill on `--void-surface` track).
  - **Right**: "System Pulse" -- 4 micro-gauges in a 2x2 grid:
    - Risk Temperature: 40px SVG circle, gradient arc (violet -> amber -> rose)
    - PIT Pass Rate: 40px SVG circle, gradient arc (rose -> amber -> emerald)
    - Data Freshness: 40px SVG circle, emerald/amber/rose fill
    - Asset Coverage: 40px SVG circle, violet fill
    Each gauge has a center label in `mono` typography and a label below in
    `caption` typography.
- [ ] AC-2: The briefing card background uses `--gradient-aurora` with a
  `--gradient-cosmic-glow` overlay that shifts based on overall market sentiment:
  - Majority bullish: glow shifts toward emerald tint
    `radial-gradient(ellipse at 50% 50%, rgba(52,211,153,0.06) 0%, transparent 70%)`
  - Majority bearish: glow shifts toward rose tint
    `radial-gradient(ellipse at 50% 50%, rgba(251,113,133,0.06) 0%, transparent 70%)`
  - Balanced: neutral violet glow (default `--gradient-cosmic-glow`)
- [ ] AC-3: Each micro-gauge is a 40px SVG circle with a colored arc stroke (3px).
  The arc animates from 0 to target value over 800ms on page load using `spring`
  motion, creating a satisfying mechanical snap at the end. The arc uses
  gradient stroke via SVG `<linearGradient>` definition.
- [ ] AC-4: The "Since Last Visit" section shows "Welcome back" on first visit
  (no `localStorage` timestamp) with a gentle wave icon. Shows "All caught up"
  with an emerald checkmark if nothing changed. Both states use `--text-violet`
  typography with subtle aurora glow.
- [ ] AC-5: The entire briefing card animates in with `fade-up` (translate-y: 12px -> 0,
  opacity: 0 -> 1) at page load, 400ms with `expressive` easing. The three columns
  stagger by 80ms each (left first, center second, right third).
- [ ] AC-6: The morning briefing hero is so information-rich yet so visually
  stunning -- the aurora gradient, the shimmer text, the micro-gauges spinning
  to their values -- that users would absolutely fall in love with starting
  their day here and drool over how much intelligence radiates from this one
  card. Apple engineers would study the gradient text treatment and the gauge
  spring animation and wonder why their own dashboards feel static by comparison.

---

## Story 2.2: Signal Distribution Radar with Flowing Gradient Bar

**As a** portfolio manager monitoring signal balance,
**I want** a rich signal distribution visualization that shows both counts
and directional momentum,
**so that** I can see not just what the distribution is, but how it is shifting.

### Visual Design

The signal distribution replaces a static donut chart with a living gradient
bar that flows across the full card width. Each signal category occupies a
proportional segment, and the segments bleed into each other through gradient
transitions -- no hard edges. Below the bar, a sparkline row shows 7-day
history as a series of stacked mini-bars, creating a temporal depth that
makes distribution shifts visually obvious.

### Acceptance Criteria

- [ ] AC-1: Replace the current donut chart with a **flowing gradient bar**
  spanning the full card width (cosmic glass card, elevation 1). Five segments
  render as a continuous gradient strip:
  - Strong Sell: `--gradient-signal-bear` (100% opacity)
  - Sell: rose at 50% opacity
  - Hold: `--void-active` (neutral violet-gray)
  - Buy: emerald at 50% opacity
  - Strong Buy: `--gradient-signal-bull` (100% opacity)
  Segments blend at boundaries through 4px gradient transitions rather than
  hard color stops, creating a flowing organic feel. Each segment width is
  proportional to its count. The bar has `border-radius: 8px` and a height
  of 12px.
- [ ] AC-2: Below the bar, a **sparkline row** shows how the distribution has
  shifted over the last 7 days. Each day renders as a thin (2px) version of
  the same gradient bar, stacked vertically with 2px gap. This creates a
  "geological stratum" effect showing distribution drift over time. Historical
  snapshot data persists in `localStorage` (saved once per day on first visit).
- [ ] AC-3: Hovering any segment of the bar highlights it with a violet-tinted
  glow (`box-shadow: 0 0 12px rgba(139,92,246,0.15)`) and shows a cosmic
  glass tooltip (elevation 3): count, percentage, and list of top 3 tickers
  in that category as clickable badges.
- [ ] AC-4: Below the visualization, a single-line sentence summarizes the shift:
  "Distribution shifted +4% bullish over 7 days" in `body` typography with
  the directional word ("bullish"/"bearish") colored in context. Stable
  distributions show "Stable this week" in `--text-muted`.
- [ ] AC-5: The gradient bar segments animate in from center-out on page load:
  Hold segment appears first, then Buy/Sell grow outward, then Strong Buy/Sell
  reach their positions. 400ms total with `expressive` motion. The effect is
  like a gradient rainbow expanding from the center.
- [ ] AC-6: Clicking any segment filters the Signal Heatmap below to show only
  assets in that signal category. The clicked segment pulses once with violet
  glow to confirm the interaction.

---

## Story 2.3: Living Signal Heatmap with Drill-Through Cosmos

**As a** trader scanning 100+ assets simultaneously,
**I want** the signal heatmap to be an interactive exploration surface,
**so that** I can spot anomalies, drill into sectors, and navigate to charts
without leaving the Overview page.

### Visual Design

The heatmap is a matrix of colored cells against the void, where each cell
glows with the intensity of its signal. Strong signals burn bright (emerald
or rose), while holds are nearly invisible dimples in the void. The effect
is a star map where each asset is a star, and its brightness tells you its
signal strength. Hovering a cell creates a radial glow that illuminates
nearby cells, guiding the eye through the constellation.

### Acceptance Criteria

- [ ] AC-1: The heatmap renders a matrix of **asset rows x horizon columns** with
  cells colored using a perceptually uniform diverging colormap centered on void:
  - Bearish: rose-to-void gradient (`--accent-rose` at opacity mapped to magnitude)
  - Neutral: `--void-surface` (nearly invisible against the card background)
  - Bullish: emerald-to-void gradient (`--accent-emerald` at opacity mapped to magnitude)
  Cell size: 32px x 24px with 1px gap. Corner radius: 3px. Each cell has a
  1px border of `--border-void` that transitions to `--border-glow` on hover.
- [ ] AC-2: Sectors are collapsible groups. Each sector header row uses a glass
  surface (`--void-hover` background) showing:
  - Sector name in `heading-3` typography, `--text-violet` color
  - Asset count in `caption` badge
  - A **sector sentiment bar**: 80px wide, 4px tall, same flowing gradient
    as Story 2.2 but miniaturized
  - Average momentum (emerald/rose colored number)
  - Expand/collapse chevron with 200ms rotation animation
- [ ] AC-3: Hovering a cell creates a **radial glow effect** around the hovered
  cell (`box-shadow: 0 0 16px rgba(139,92,246,0.2)`) and shows a cosmic glass
  tooltip (elevation 3, `--gradient-nebula` background) containing:
  - Ticker + Horizon label in `heading-3` typography
  - Expected return (large, colored with gradient text for strong signals)
  - Probability (p_up) with a mini arc gauge (20px)
  - Kelly fraction with a tiny gradient bar
  - Signal label with colored badge (`--gradient-signal-*`)
  - A 30-day mini sparkline (40px wide) showing price trend
- [ ] AC-4: Clicking a cell navigates to `/charts/{SYMBOL}`. The cell flashes
  with violet glow (200ms) before navigation as haptic-style feedback.
- [ ] AC-5: Keyboard navigation: `j/k` moves between rows, `h/l` moves between
  columns, `Enter` clicks the cell, `Esc` deselects. The focused cell has a
  `--border-glow` ring (2px) with violet outer glow.
- [ ] AC-6: A **color scale legend** appears in the top-right of the heatmap
  as a thin gradient strip (120px wide, 8px tall) from rose through void to
  emerald, with labeled ticks at -10%, 0%, +10%.
- [ ] AC-7: Sector groups remember collapsed/expanded state in `localStorage`.
- [ ] AC-8: The heatmap renders smoothly with 150+ assets via virtualized rows.
  Only visible rows are in the DOM; scrolling lazy-loads additional rows.
  Scrolling is buttery at 60fps -- no jank, no white flashes.
- [ ] AC-9: The heatmap experience -- the star-map aesthetic, the glowing cells,
  the rich tooltips, the sector intelligence -- is so fluid and satisfying
  that users would absolutely fall in love with scanning their entire portfolio
  and drool over the instant visual pattern recognition. Apple engineers would
  study the cell interaction model and the perceptual colormap and recognize
  a level of craft they rarely see outside Cupertino.

---

## Story 2.4: Model Confidence Leaderboard with Cosmic Rankings

**As a** quant researcher monitoring model performance,
**I want** the model distribution chart to show not just counts but confidence
and calibration quality,
**so that** I understand which models the system trusts most and why.

### Visual Design

The leaderboard uses numbered glass rows with the top 3 positions distinguished
by gradient badges: gold (warm amber gradient), silver (cool gray gradient),
bronze (copper-rose gradient). Behind each rank number, a subtle radial glow
matches the medal color. The entire leaderboard sits in a cosmic glass card
with the `--gradient-nebula` background.

### Acceptance Criteria

- [ ] AC-1: Replace the current horizontal bar chart with a **leaderboard table**
  (cosmic glass card, elevation 1) showing top 10 models, ranked by BMA
  selection frequency:
  - Rank number (1-10): top 3 use gradient badge circles (28px diameter):
    #1: `linear-gradient(135deg, #F59E0B 0%, #D97706 100%)` (gold)
    #2: `linear-gradient(135deg, #94A3B8 0%, #64748B 100%)` (silver)
    #3: `linear-gradient(135deg, #FB923C 0%, #C2410C 100%)` (bronze)
    #4-10: plain `--text-muted` numbers
  - Model name in `body` typography, truncated with ellipsis, full on hover
  - Selection count with a gradient fill bar behind the number: the bar uses
    `linear-gradient(90deg, rgba(139,92,246,0.15) 0%, rgba(139,92,246,0.03) 100%)`
    width proportional to count vs maximum
  - Average BMA weight as percentage in `mono` typography
  - Average PIT pass rate as colored badge: emerald >= 80%, amber >= 60%, rose < 60%
- [ ] AC-2: Each row uses `--void-surface` background with `--void-hover` on hover.
  Hover also triggers elevation 2 glow. Transition: 120ms with `micro` easing.
- [ ] AC-3: Clicking a model row expands it (smooth height transition, 250ms) to
  show: list of assets using this model as small `--accent-violet` ticker badges,
  and average BIC/CRPS/Hyvarinen scores in `mono` type with semantic coloring.
- [ ] AC-4: A "View All Models" link at the bottom navigates to
  Diagnostics > Model Comparison. The link uses `--accent-violet` color with
  an arrow icon, and a gradient underline on hover.
- [ ] AC-5: The leaderboard animates in row-by-row with 50ms stagger using
  `standard` motion: each row fades in + slides from left (8px), creating a
  satisfying cascade effect that builds anticipation as the rankings unfold.

---

## Story 2.5: Top Movers Conviction Spotlight with Dual Nebula Glow

**As a** trader looking for actionable opportunities,
**I want** a prominently featured section showing the highest-conviction signals
with enough context to act immediately,
**so that** I can identify and act on the best opportunities in seconds.

### Visual Design

Two side-by-side panels -- Strongest Buys and Strongest Sells -- each bathed
in their own nebula glow. The Buy panel has a soft emerald aurora bleeding
from its top-left corner. The Sell panel has a soft rose aurora from its
top-right corner. Between them, a thin void gap creates visual separation
while the ambient glows just barely overlap in the center, creating a
beautiful emerald-rose gradient blending point.

### Acceptance Criteria

- [ ] AC-1: A "Conviction Spotlight" section appears after the stat cards,
  showing two side-by-side panels (cosmic glass cards, elevation 1) with
  16px gap between them:
  - **Strongest Buys** (left): emerald accent. Background: `--gradient-nebula`
    with an emerald glow overlay: `radial-gradient(ellipse at 10% 10%,
    rgba(52,211,153,0.08) 0%, transparent 60%)`
  - **Strongest Sells** (right): rose accent. Background: `--gradient-nebula`
    with a rose glow overlay: `radial-gradient(ellipse at 90% 10%,
    rgba(251,113,133,0.08) 0%, transparent 60%)`
- [ ] AC-2: Each panel shows up to 5 assets as **rich mini-cards** (no visible
  border, `--void-hover` background on hover):
  - Ticker (large, `heading-3` size, gradient text matching the panel's accent)
  - Sector badge (pill, `--void-active` background, `caption` typography)
  - 60-day mini price sparkline (48px tall, colored emerald/rose)
  - Expected return for best horizon (large, `heading-2` size, fully colored)
  - Probability as a micro arc gauge (16px, matching accent color)
  - Kelly fraction with a thin gradient bar (40px wide, 3px tall)
  - Signal age ("2h ago", "today") in `--text-muted`, `caption` typography
- [ ] AC-3: Each asset card is clickable (navigates to `/charts/{SYMBOL}`).
  On hover, the card lifts 1px (`transform: translateY(-1px)`) and gains
  the panel's accent glow: `box-shadow: 0 0 20px rgba(accent, 0.06)`.
- [ ] AC-4: The panel headers use `heading-2` typography with an SVG icon:
  "Strongest Buys" with an upward arrow icon in `--accent-emerald`,
  "Strongest Sells" with a downward arrow icon in `--accent-rose`.
- [ ] AC-5: If no high-conviction signals exist, the panel shows an elegant
  empty state: a balanced scales SVG icon (40px, `--text-muted`, line art)
  with "No strong signals today" in `--text-secondary` and "Markets in
  equilibrium" in `--text-muted`. The icon has a subtle floating animation
  (2px up/down, 3s cycle).
- [ ] AC-6: The conviction spotlight -- with its dual nebula glows, the emerald
  and rose auroras meeting in the void between panels, the rich asset cards
  with gradient text and micro-gauges -- must be so visually striking and
  actionable that users would absolutely fall in love with the zero-click
  path to their best trades and drool over the immediate actionability. Apple
  engineers would photograph this section and send it to their team as an
  example of how data visualization can be simultaneously beautiful and
  brutally functional.

---

# EPIC 3: Signals Page -- The Cosmic Decision Table

> **Vision**: The Signals page is the trader's primary workspace. It must feel like
> a Bloomberg terminal reimagined by Apple's design team after a weekend retreat
> studying deep-space photography: dense with data, yet every element has breathing
> room. The void-black canvas makes every number pop. Emerald and rose signals
> glow against the dark like bioluminescent organisms in the deep ocean. Sorting,
> filtering, and scanning 100+ assets must feel effortless -- like scrolling through
> a thoughtfully curated infinite canvas.

---

## Story 3.1: Signal Table with Inline Micro-Charts and Violet Row Glow

**As a** trader scanning signal recommendations across my asset universe,
**I want** each asset row to include a micro price chart and visual signal strength,
**so that** I can make faster visual comparisons without clicking into individual charts.

### Visual Design

The table lives inside a cosmic glass card. Row backgrounds alternate between
`--void` and `--void-surface` -- so subtle you feel the rhythm more than see
it. Hover brings the row to life: a radial violet glow appears behind the
row from its left edge, the row lifts by 1px, and the border-radius of the
row creates a floating pill effect. The sticky header has a `--gradient-nebula`
background with glass blur, making it feel like a control bar floating above
the data stream.

The sparklines are thin luminous threads -- emerald or rose against the void,
creating miniature star trails next to each ticker. Signal strength bars glow
with their accent color, creating a column of light that the eye follows
naturally.

### Acceptance Criteria

- [ ] AC-1: Each asset row in the All Assets table includes a **60px-wide sparkline**
  column showing the 30-day price movement. The sparkline is a 1.5px line rendered
  on a transparent background:
  - Above 20-day SMA: emerald line with a subtle emerald glow
    (`filter: drop-shadow(0 0 2px rgba(52,211,153,0.4))`)
  - Below 20-day SMA: rose line with a subtle rose glow
    (`filter: drop-shadow(0 0 2px rgba(251,113,133,0.4))`)
  The sparkline appears as a luminous thread in the void.
- [ ] AC-2: The signal column displays a **gradient strength bar** (40px wide, 6px
  tall, border-radius: 3px) next to the signal text label. The bar sits on a
  `--void-active` track. Fill uses:
  - Buy signals: `linear-gradient(90deg, rgba(52,211,153,0.3) 0%, var(--accent-emerald) 100%)`
  - Sell signals: `linear-gradient(90deg, rgba(251,113,133,0.3) 0%, var(--accent-rose) 100%)`
  - Hold: `linear-gradient(90deg, rgba(139,92,246,0.2) 0%, rgba(139,92,246,0.4) 100%)`
  Fill percentage represents composite confidence (p_up + Kelly). The filled
  portion has a subtle glow matching its color.
- [ ] AC-3: Momentum score displays as a **colored numeric badge** with a background
  that uses gradient opacity proportional to magnitude:
  - Positive momentum: text in `--accent-emerald`, background
    `rgba(52,211,153, momentum * 0.15)`, border-radius 6px
  - Negative momentum: text in `--accent-rose`, background
    `rgba(251,113,133, abs(momentum) * 0.15)`
  Strong momentum values (abs > 0.7) get a subtle outer glow matching their
  color: `box-shadow: 0 0 8px rgba(color, 0.12)`.
- [ ] AC-4: Crash risk score displays as a **heat indicator**: four small rectangle
  segments (6px wide, 12px tall each, 2px gap, border-radius 2px). Segments
  fill from left to right based on risk level. Colors transition through a
  gradient: segment 1 = `--accent-emerald`, segment 2 = `--accent-amber`,
  segment 3 = `#F97316` (orange), segment 4 = `--accent-rose`. Filled segments
  glow; empty segments use `--void-active` (nearly invisible).
- [ ] AC-5: Horizon columns show expected return with a **directional micro-arrow**
  (SVG, 8px, emerald up-arrow for positive, rose down-arrow for negative).
  Returns > +5% use `font-weight: 600` and `--accent-emerald` with subtle
  text-shadow glow. Returns < -5% use `font-weight: 600` and `--accent-rose`.
  Returns between -1% and +1% use `--text-muted`.
- [ ] AC-6: The table header row is sticky on scroll with a `--gradient-nebula`
  background and `backdrop-filter: blur(12px)`. A bottom shadow appears only when
  scrolled: `box-shadow: 0 4px 12px rgba(0,0,0,0.3)`, fading in over 150ms.
  Header text uses `caption` typography, `--text-violet` color, `text-transform:
  uppercase`, `letter-spacing: 0.06em`.
- [ ] AC-7: Row hover state: the entire row gains `--void-hover` background with
  a violet radial glow from the left: `background: linear-gradient(90deg,
  rgba(139,92,246,0.06) 0%, transparent 40%), var(--void-hover)`. The row lifts
  1px (`transform: translateY(-0.5px)`) and gains elevation 2 glow. Transition:
  120ms with `micro` easing. The sparkline in the hovered row gains a tooltip
  (cosmic glass, elevation 3) showing: current price, 30-day change %, volume
  trend as colored arrow.
- [ ] AC-8: The table is so information-dense yet so visually refined -- the
  luminous sparklines, the gradient strength bars, the glowing momentum badges,
  the heat indicators, all against the cosmic void -- that users would absolutely
  fall in love with scanning 100+ assets and drool over how much decision-support
  data fits cleanly in each row. Apple engineers would marvel at how data density
  and visual beauty coexist without compromise.

---

## Story 3.2: Multi-Axis Sort with Violet Priority Indicators

**As a** trader who sorts by different criteria depending on my objective,
**I want** the ability to sort by multiple columns simultaneously with clear
visual indicators of sort priority,
**so that** I can create compound rankings like "highest momentum among strong buys."

### Visual Design

Sort indicators are refined glass pills with violet numbering. When a sort is
active, the column header text shifts from `--text-violet` to `--accent-violet`
with a subtle glow, and a numbered badge appears -- a tiny circle filled with
`--accent-violet` containing the sort priority number in white. The sort
direction arrow uses a smooth 200ms rotation animation when toggled.

### Acceptance Criteria

- [ ] AC-1: Clicking a column header sets it as the **primary sort**. A directional
  arrow SVG (10px, `--accent-violet`) appears next to the header text, pointing
  up for ascending, down for descending. The arrow fades in over 120ms.
- [ ] AC-2: Holding `Shift` and clicking a second column adds it as a **secondary
  sort**. A numbered badge (14px circle, `--accent-violet` background, white
  text in `caption` typography) appears on each active sort column header.
  Up to 3 sort levels supported.
- [ ] AC-3: Active sort columns have their header text in `--accent-violet` with
  a subtle text-shadow: `0 0 8px rgba(139,92,246,0.3)`. Inactive sort columns
  remain `--text-violet`. This creates a "light up" effect on sort activation.
- [ ] AC-4: Clicking an already-sorted column toggles direction (asc/desc) with
  the arrow rotating 180 degrees over 200ms. Clicking it a third time removes
  that sort level with a 120ms fade-out.
- [ ] AC-5: A **sort indicator bar** appears above the table as a thin glass strip
  (28px tall, `--void-hover` background) showing the active sort chain in
  plain language: "Sorted by Signal (desc), then Momentum (desc)" in `caption`
  typography, `--text-secondary`. Each criterion has a small X button
  (`--text-muted`, rose on hover) for removal.
- [ ] AC-6: Sort state persists in `localStorage` per view mode (all/sectors/ranked).
- [ ] AC-7: `Shift+Click` sorting reorders rows with a subtle 200ms animation
  where rows slide to their new positions using `transform: translateY()` transitions
  rather than DOM reorder, creating a smooth reshuffling effect that feels like
  cards being dealt into a new order.

---

## Story 3.3: Sector Panel Redesign with Aggregate Nebula Intelligence

**As a** trader who thinks in sector allocations,
**I want** sector panels that show aggregate statistics and visual portfolio weight,
**so that** sector-level patterns and concentrations are immediately visible.

### Visual Design

Each sector panel is a collapsible cosmic glass container. The collapsed state
packs an incredible density of information into a single 48px-tall header row.
The sector sentiment bar is a miniature version of the flowing gradient bar
from Story 2.2 -- a tiny flowing rainbow of emerald through void to rose that
tells the sector's story in 80 pixels. The expand/collapse animation reveals
the asset rows beneath like a constellation unfurling.

### Acceptance Criteria

- [ ] AC-1: Each sector panel header displays as a 48px-tall glass row
  (`--void-hover` background) in a single row with generous spacing:
  - Sector name in `heading-3` typography, `--text-luminous`
  - Asset count badge: pill shaped, `--void-active` background, `caption` text
  - **Sector sentiment bar**: 80px wide, 4px tall, border-radius 2px. Same
    flowing gradient pattern as Story 2.2 (Strong Sell rose -> Hold void ->
    Strong Buy emerald) proportional to asset signal distribution. The bar
    has a 1px `--border-void` border and a subtle glow on hover.
  - Average momentum score: `mono` typography, colored (emerald +, rose -),
    with a tiny directional arrow (6px)
  - Average expected return: `mono` typography, colored
  - Expand/collapse chevron (12px SVG, `--accent-violet`) with smooth 200ms
    rotation animation
- [ ] AC-2: When a sector panel is collapsed, a **peek row** appears inside the
  header (right-aligned, before the chevron): the top performing asset as
  "Best: NVDA +8.2% (Strong Buy)" in `caption` typography. The ticker uses
  `--accent-violet`, the return uses `--accent-emerald`. This gives value
  even without expanding -- a zero-click insight embedded in the header.
- [ ] AC-3: Sector panels sort by aggregate momentum by default. A dropdown
  (cosmic glass, elevation 3) in the section header allows sorting sectors by:
  Momentum, Expected Return, Signal Strength, Asset Count, or Alphabetical.
  The dropdown trigger is a subtle "Sort" label with a small chevron.
- [ ] AC-4: The expand/collapse animation is a smooth height transition (250ms,
  `standard` motion) with content rows fading in after the height settles
  (100ms stagger between rows). The effect is like a constellation of stars
  appearing one by one as the sector unfurls.
- [ ] AC-5: Each sector panel has a subtle left border (2px) that is a gradient:
  - Majority bullish sectors: `linear-gradient(180deg, var(--accent-emerald) 0%,
    rgba(52,211,153,0.3) 100%)`
  - Majority bearish: `linear-gradient(180deg, var(--accent-rose) 0%,
    rgba(251,113,133,0.3) 100%)`
  - Mixed: `linear-gradient(180deg, var(--accent-violet) 0%,
    rgba(139,92,246,0.3) 100%)`
  This creates a colored light strip on the left edge that tells the sector
  story at the most peripheral glance.
- [ ] AC-6: The sector panel design -- with its flowing sentiment bars, gradient
  left borders, peek rows, and constellation expand animation -- summarizes
  so much information in the collapsed state that users would absolutely fall
  in love with the zero-click sector overview and drool over the aggregate
  intelligence embedded in each 48-pixel header. Apple engineers would study
  the information density of the collapsed state and recognize that every
  pixel was deliberated over.

---

## Story 3.4: Real-Time Signal Flash with Aurora Change Trails

**As a** trader monitoring signals via WebSocket,
**I want** signal changes to be visually prominent with before/after context,
**so that** I notice every upgrade and downgrade as it happens and understand
the magnitude of the change.

### Visual Design

When a signal changes, the affected row does not merely blink -- it creates
an **aurora trail**. A brief wave of color sweeps across the row from left to
right: emerald for upgrades, rose for downgrades. The wave leaves behind a
fading glow that persists for 10 seconds, a ghost of the change that keeps
the user aware of what just happened. The transition badge above the new value
uses a glass pill style with strikethrough on the old value.

### Acceptance Criteria

- [ ] AC-1: When a signal changes via WebSocket update, the affected row triggers
  an **aurora sweep** animation:
  - Upgrade (e.g., Hold -> Buy): an emerald gradient wave sweeps left-to-right
    across the row over 600ms:
    `background: linear-gradient(90deg, transparent 0%, rgba(52,211,153,0.12) 40%,
    rgba(52,211,153,0.2) 50%, rgba(52,211,153,0.12) 60%, transparent 100%);
    background-size: 200% 100%; animation: aurora-sweep 600ms ease-out;`
    The sweep repeats twice then fades to a residual glow.
  - Downgrade (e.g., Buy -> Hold): rose aurora sweep with same pattern but
    using `rgba(251,113,133,...)` values.
  - New entry: violet aurora sweep using `rgba(139,92,246,...)`.
- [ ] AC-2: The changed cell shows a **transition badge** for 10 seconds: a small
  glass pill (elevation 1, `--void-active` background) positioned above the
  new value showing "was Hold" or "was 4.2%" in `caption` typography,
  `--text-muted`, with the old value in strikethrough. The badge fades in
  (150ms) and fades out after 10 seconds (300ms). It has a tiny colored dot
  matching the change direction (emerald for improvement, rose for regression).
- [ ] AC-3: A **change counter badge** appears in the page header: a violet pill
  showing "3 changes" in `caption` typography. The badge pulses gently
  (opacity 70-100%, 2s cycle) to maintain awareness. Clicking it scrolls to
  the most recent change and highlights the row with a violet glow.
- [ ] AC-4: An optional "Live Feed" toggle in the toolbar enables a **ticker tape**
  at the top of the signal table: a thin glass strip (28px tall) with
  horizontally scrolling change items: "NVDA: Hold -> Buy | TSLA: Sell -> Hold".
  Each item uses colored arrows (emerald up, rose down) and `mono` typography.
  The tape scrolls with constant velocity (60px/s) and pauses on hover.
- [ ] AC-5: The aurora sweep uses CSS `@keyframes` that transitions the gradient
  `background-position`, not `background-color`, ensuring text remains
  readable during the animation. The glow is additive over the row's existing
  background.
- [ ] AC-6: Animations are disabled when the browser tab is not visible
  (`document.hidden`). On tab return, any changes that occurred while hidden
  are shown as a batch: "4 signals changed while away" with a "Review" button.

---

## Story 3.5: Smart Search with Violet Focus and Fuzzy Matching

**As a** trader searching for specific assets in a universe of 100+,
**I want** search to be fast, fuzzy, and combinable with active filters,
**so that** I can find any asset configuration in under 2 seconds.

### Visual Design

The search input is a refined glass pill that expands on focus. When idle,
it shows a muted search icon and placeholder. On focus, the input's border
transitions from `--border-void` to a luminous `--accent-violet` glow ring,
and the input background darkens slightly to `--void-active`, creating the
feeling of the input "activating" -- like a console powering on in a
spacecraft cockpit.

### Acceptance Criteria

- [ ] AC-1: The search input supports fuzzy matching: "nvd" matches "NVDA",
  "apl" matches "AAPL", "cro" matches "CRWD" and "CRM". Fuzzy scoring ranks
  exact prefix matches highest, then substring matches, then character-skip
  matches. The algorithm is case-insensitive and handles both ticker symbols
  and company names.
- [ ] AC-2: Search matches highlight with `--accent-violet` background at 20%
  opacity on the matching characters in result rows. The highlight uses
  `border-radius: 2px` and a subtle violet glow for individual characters:
  `box-shadow: 0 0 4px rgba(139,92,246,0.2)`.
- [ ] AC-3: Search works simultaneously with signal filter and view mode.
  Example: searching "gold" while filter is "Strong Buy" shows only
  gold-related assets with strong buy signals. Filters are composable --
  each narrows the others.
- [ ] AC-4: The search input shows a live result count on the right side (inside
  the input, before the clear button): "12 of 147" in `caption` typography,
  `--text-muted`, updating as the user types. The count uses `tabular-nums`
  for stable width.
- [ ] AC-5: `Cmd+K` or `/` focuses the search input from anywhere on the page
  with a brief violet flash of the input border (200ms). `Esc` clears the
  search and blurs the input. These shortcuts are shown as faded key-cap hint
  text inside the input when empty: a tiny "/" in a rounded square.
- [ ] AC-6: The search is debounced at 100ms for responsiveness. During the
  debounce wait, a subtle activity indicator (tiny violet dot pulsing at
  the right edge of the input) shows that processing is happening.
- [ ] AC-7: An "X" clear button appears when the input has text, positioned
  inside the input on the right side. The X uses `--text-muted` at rest,
  `--accent-rose` on hover, with a 120ms transition.
- [ ] AC-8: The search input's focus state -- the violet glow ring, the
  expanding animation, the live count, the keyboard shortcut hints -- creates
  such a refined interaction that users would absolutely fall in love with
  the search experience and drool over how even finding an asset feels like
  using a precision instrument crafted by artisans.

---

## Story 3.6: Horizon Column Smart Density with Violet Compact Indicators

**As a** trader viewing signals on different screen sizes,
**I want** horizon columns to intelligently adapt to available width,
**so that** I get maximum data density without horizontal scrolling.

### Visual Design

Visible horizons display as full columns with expected return as the primary
number and probability as a subordinate line below. Hidden horizons (at narrow
viewports) collapse into a compact "..." indicator that expands on hover into
a floating glass card showing all horizons. The horizon selector above the
table uses violet pill buttons that glow when active.

### Acceptance Criteria

- [ ] AC-1: The table detects available viewport width and displays the maximum
  number of horizon columns that fit without scrolling:
  - >= 1600px: All horizons (1D, 3D, 7D, 30D, 90D, 180D, 365D)
  - >= 1280px: 5 horizons (7D, 30D, 90D, 180D, 365D)
  - >= 1024px: 3 horizons (7D, 30D, 365D)
  - < 1024px: 1 horizon (30D) with a "..." pill button (violet border, `caption`
    text) that expands into a floating glass card showing all horizons on click
- [ ] AC-2: A **horizon selector** appears above the table as a row of pill
  toggles. Active pills have `--accent-violet` background at 15% opacity,
  `--accent-violet` text, and a 1px `--border-glow` border. Inactive pills
  have `--void-active` background and `--text-secondary` text. Selected
  horizons save to `localStorage`. This overrides the auto-fit behavior.
  Toggle transition: 120ms with `micro` easing.
- [ ] AC-3: Each horizon column cell shows expected return as the primary
  number in `mono` typography (colored emerald/rose) and probability as a
  subtle sub-line (11px, `--text-muted`) below. The two-line layout uses
  `line-height: 1.2` for tight vertical rhythm.
- [ ] AC-4: Hovering a horizon cell shows a cosmic glass tooltip (elevation 3,
  `--gradient-nebula` background) with:
  - Expected return (large, `heading-3` size, gradient text for strong signals)
  - Probability with a mini arc gauge (16px, violet track, emerald/rose fill)
  - Kelly fraction with gradient bar (30px wide)
  - Upper/Lower uncertainty envelope in `mono` typography, `--text-muted`
  - Signal classification for that specific horizon as a colored badge
- [ ] AC-5: The horizon density adaptation -- automatic column fitting,
  collapsible overflow, violet pill selectors, rich hover tooltips -- is so
  seamless that users would absolutely fall in love with always seeing the
  perfect amount of data and drool over never needing to horizontally scroll
  or squint at cramped columns.

---

# EPIC 4: Charts Page -- The Cosmic Analysis Theater

> **Vision**: The Charts page is where decisions crystallize. It must combine the
> technical analysis power of TradingView with the Bayesian model overlay that is
> our unique advantage -- and wrap it all in a visual experience that makes the
> act of chart analysis feel like peering through a telescope into the market's
> probability space.
>
> The chart canvas is a window into the void. Price candles glow against the
> black. The forecast probability cone extends into the future like a nebula --
> violet-tinged, translucent, beautiful. The detail sidebar is a cosmic glass
> panel that floats alongside the chart, providing signal context without
> obscuring the price action.

---

## Story 4.1: Chart Area with Probability Nebula Overlay

**As a** trader analyzing a specific asset,
**I want** the chart to overlay forecast confidence intervals directly on the
price chart as shaded probability regions,
**so that** I can visually see where the model expects price to go and how
uncertain it is.

### Visual Design

The forecast overlay is not a generic shaded area -- it is a **probability
nebula**. The median line glows with `--accent-violet` light against the void.
The 50% confidence interval is a violet-tinted translucent region that
creates depth, and the 90% confidence interval is an even more ethereal
violet mist at the edges. The entire cone appears to emanate from the last
price candle, spreading into the future like starlight through a prism.

The candle chart itself uses the cosmic palette: green candles use
`--accent-emerald` with a subtle glow, red candles use `--accent-rose`.
The chart background is pure `--void`. Grid lines are nearly invisible
threads of `rgba(139,92,246,0.04)`.

### Acceptance Criteria

- [ ] AC-1: When forecast data is loaded, a **probability nebula** renders on the
  chart extending from the last price candle into the future:
  - Median forecast: solid line (2px, `--accent-violet`) with a subtle glow
    (`filter: drop-shadow(0 0 3px rgba(139,92,246,0.5))`)
  - 50% confidence interval: filled region using `rgba(139,92,246,0.12)` with
    a gradient that fades to `rgba(139,92,246,0.06)` at the edges, creating
    translucent depth
  - 90% confidence interval: filled region using `rgba(139,92,246,0.04)` --
    the faintest violet mist, barely visible but giving the cone its nebula
    character. The outer edge has a thin (0.5px) dashed line in
    `rgba(139,92,246,0.15)`.
- [ ] AC-2: The probability nebula extends to the farthest available forecast
  horizon (365 days) but only shows detail for visible horizons based on zoom.
  At long zoom-out, the nebula compresses. At zoom-in, it expands to show
  granular day-by-day confidence.
- [ ] AC-3: Hovering a point within the forecast cone shows a cosmic glass tooltip:
  - Date in `caption` typography
  - Median expected price in `heading-3` with gradient text
  - 50% CI range (low-high) in `mono` typography, `--text-secondary`
  - 90% CI range (low-high) in `mono` typography, `--text-muted`
  - Probability of being above current price at that date: shown as a mini
    arc gauge (16px) with emerald fill
  - A thin dashed crosshair line extends from the hover point to both axes
- [ ] AC-4: The forecast nebula animates in when first loaded: the shaded regions
  grow from left (today) to right (future) over 800ms with `expressive` motion.
  The median line draws itself like a laser beam extending into the future.
  The confidence intervals bloom around it like gas expanding from a star.
- [ ] AC-5: Forecast overlay can be toggled independently of technical overlays.
  Its toggle button uses a nebula icon (SVG, `--accent-violet`) and sits in
  the Forecast group of the chart toolbar.
- [ ] AC-6: If the current price is outside the 90% CI at any past forecast point
  (a "surprise"), that region on the chart gets a subtle colored tint:
  - Positive surprise (price above upper CI): faint emerald patch
  - Negative surprise (price below lower CI): faint rose patch
  This creates a visual "scar" on the chart showing where the model was wrong.
- [ ] AC-7: The probability nebula overlay -- the glowing median line, the
  violet-tinged confidence regions blooming into the future, the surprise
  scars on past predictions -- gives the chart such a unique analytical edge
  that users would absolutely fall in love with seeing the future probability
  landscape and drool over the visual integration of Bayesian uncertainty
  into the price chart. Apple engineers would study the nebula render technique
  and the animate-in sequence and recognize a new standard for financial
  data visualization.

---

## Story 4.2: Asset Detail Sidebar with Cosmic Glass Signal Panel

**As a** trader viewing a chart,
**I want** a detail sidebar that shows the full Bayesian signal intelligence
for the charted asset without navigating away,
**so that** chart analysis and signal analysis happen in a unified context.

### Visual Design

The detail sidebar is a cosmic glass panel that slides in from the right edge.
Its background is `--gradient-nebula` with heavy glass blur (`blur(24px)`),
creating a frosted-glass cockpit panel that floats over the chart edge. The
signal badge at the top is a full-width gradient strip (emerald or rose)
that saturates the top of the sidebar with the asset's directional sentiment.
Below, data organized in discrete glass sections separated by fading violet
divider lines.

### Acceptance Criteria

- [ ] AC-1: When a symbol is selected, a **detail sidebar** (320px wide, resizable
  via drag handle) slides in from the right, cosmic glass with elevation 3:
  - **Header**: Ticker in `heading-1` with gradient text (`--text-luminous` to
    `--accent-violet`), sector badge (violet pill), current price in `display`
    typography, daily change with colored badge (emerald/rose gradient background)
  - **Signal Badge**: Full-width (minus padding) gradient strip (32px tall,
    border-radius 8px):
    - Strong Buy: `--gradient-signal-bull` with `--accent-emerald` text
    - Buy: emerald at 50% opacity
    - Hold: `--void-active` with `--text-secondary`
    - Sell: rose at 50% opacity
    - Strong Sell: `--gradient-signal-bear` with white text
  - **Horizon Table**: All available horizons with columns: Expected Return
    (colored), Probability (mini arc gauge), Kelly (gradient bar), Signal
    badge. Rows alternate `--void` / `--void-surface`. Best horizon row has
    a subtle violet left border.
  - **Model Info**: Best model name in `--text-violet`, BMA weight with
    gradient bar, PIT status badge (emerald/rose)
  - **Risk Metrics**: Momentum (colored with arrow), Crash risk (heat bar),
    Regime classification (colored pill badge)
- [ ] AC-2: The sidebar is collapsible via a handle (8px wide strip on the left
  edge, `--void-hover` background, a small chevron icon). Collapsed state shows
  a thin 36px strip with: ticker in vertical text, signal badge as a colored
  dot, and the expand handle. Collapse preference persists in `localStorage`.
- [ ] AC-3: Horizon rows are clickable. Clicking a horizon draws a horizontal
  reference line on the chart at the expected price level for that horizon,
  using a dashed line in `--accent-violet` with a label pill showing the
  horizon name and expected price. Multiple horizons can be active simultaneously.
- [ ] AC-4: The sidebar scrolls independently of the chart with a custom thin
  scrollbar (4px wide, `--accent-violet` thumb at 30% opacity, `--void` track).
- [ ] AC-5: On screens narrower than 1280px, the sidebar starts collapsed and
  opens as an overlay with a `rgba(3,0,20,0.6)` backdrop + blur behind it.
- [ ] AC-6: The sidebar animates in from the right (translate-x: 100% -> 0) with
  `standard` motion (250ms). Content sections stagger in by 60ms each.
- [ ] AC-7: A "View All Signals" link at the bottom uses `--accent-violet`
  with an arrow icon and gradient underline on hover.

---

## Story 4.3: Chart Toolbar with Cosmic Grouped Overlay Controls

**As a** technical analyst toggling multiple chart overlays,
**I want** overlay controls grouped by category with visual state indicators,
**so that** I can see at a glance which overlays are active and quickly toggle
combinations.

### Visual Design

The toolbar is a thin cosmic glass bar (48px tall) that sits above the chart.
Overlay toggles are grouped in segments separated by fading vertical dividers.
Each toggle is a micro glass pill. Active toggles glow with their overlay's
color -- creating a colorful row of lit indicators above the chart, like
cockpit controls.

### Acceptance Criteria

- [ ] AC-1: The chart toolbar (cosmic glass, elevation 1, `--gradient-nebula`
  background) displays overlay toggles in grouped segments separated by
  fading violet dividers (`linear-gradient(180deg, transparent 20%,
  rgba(139,92,246,0.12) 50%, transparent 80%)`):
  - **Trend**: SMA 20 (blue dot), SMA 50 (violet dot), SMA 200 (amber dot)
  - **Volatility**: Bollinger Bands (cyan dot), RSI 14 (fuchsia dot)
  - **Forecast**: Median (violet dot), CI Bands (violet at 50%), Probability Cone
  - **Misc**: Price Line (white dot)
- [ ] AC-2: Each toggle button (28px tall, border-radius 6px) shows:
  - A 6px colored dot on the left matching the overlay color on the chart
  - Label text in `caption` typography
  - Keyboard shortcut hint on the right in `--text-muted` (e.g., "B" for Bollinger)
  - Active state: `--void-active` background, colored text matching the dot,
    the dot gains a glow (`box-shadow: 0 0 6px rgba(color, 0.4)`)
  - Inactive state: transparent background, `--text-muted` text, dim dot
  - Transition: 120ms with `micro` easing
- [ ] AC-3: Group labels appear as tiny section headers (`caption` typography,
  `--text-muted`, `text-transform: uppercase`, `letter-spacing: 0.08em`)
  above each group.
- [ ] AC-4: A **"Presets"** dropdown (cosmic glass, elevation 3) allows saving
  and loading overlay combinations:
  - "Technical": SMA 20 + SMA 50 + SMA 200 + Bollinger
  - "Forecast": Forecast Median + CI + Probability Cone
  - "Clean": Price only
  - Custom presets (max 5) saved in `localStorage`, each with a user-defined
    name and a color-coded icon showing which overlay types are included
- [ ] AC-5: The toolbar adapts to width: on narrow screens, groups collapse into
  a single glass dropdown menu with sections matching the group structure.
- [ ] AC-6: Toggle transitions animate the overlay in/out on the chart (fade over
  200ms) rather than appearing/disappearing instantly. SMA lines draw
  themselves left-to-right (300ms) when activated.

---

## Story 4.4: Symbol Picker with Rich Aurora Preview Cards

**As a** trader browsing assets to chart,
**I want** the symbol picker sidebar to show enough information per asset to
help me choose which one to analyze next,
**so that** I don't waste time charting assets that aren't interesting right now.

### Visual Design

The symbol picker is a scrollable list of mini-cards, each one a tiny cosmic
glass container showing a micro-universe of data about that asset. The
currently selected asset's card has a luminous violet left border and a
subtle aurora glow. Hovering any card creates a brief violet pulse at its
border -- a "breathing" effect that makes the list feel alive.

### Acceptance Criteria

- [ ] AC-1: In the All view, each asset in the sidebar list shows a **mini card**
  (cosmic glass, 64px tall, no visible border, `--void-surface` background):
  - Ticker in `heading-3` typography, `--text-luminous` + Sector in `caption`,
    `--text-muted`
  - Signal badge: tiny pill (20px x 14px, colored gradient background) showing
    SB/B/H/S/SS
  - 30-day sparkline (48px wide, 20px tall): emerald/rose luminous thread
  - Daily change % in `mono` typography, colored (emerald/rose/gray)
- [ ] AC-2: In the Sectors view, sector headers use `heading-3` typography in
  `--text-violet` with count badge and mini sentiment bar (same pattern as
  Story 3.3). Expand/collapse with smooth height animation (200ms).
- [ ] AC-3: In the Ranked views (Momentum, Edge, Return, etc.), each card
  additionally shows:
  - The ranked metric prominently in `heading-2` size, fully colored
  - A horizontal gradient bar behind the metric (40px wide): fill uses
    `linear-gradient(90deg, rgba(139,92,246,0.1) 0%, rgba(139,92,246,0.3) 100%)`
    width proportional to value (normalized against #1 = 100%)
  - Rank number (#1, #2, #3...) as gradient badges for top 3:
    #1: gold gradient, #2: silver, #3: bronze (same as Story 2.4)
- [ ] AC-4: The symbol picker search uses the same fuzzy matching as Story 3.5.
  The search input uses the same violet focus glow treatment.
- [ ] AC-5: Clicking an asset transitions the chart with a brief crossfade
  (150ms fade-out old, 150ms fade-in new). During data fetch, the chart
  container shows a cosmic shimmer loading state (see Story 10.1).
- [ ] AC-6: The currently selected asset card has:
  - 4px left border: `linear-gradient(180deg, var(--accent-violet) 0%,
    var(--accent-indigo) 100%)`
  - Background: `--void-hover` with glow: `radial-gradient(ellipse at 0% 50%,
    rgba(139,92,246,0.08) 0%, transparent 60%)`
  Unselected cards hover with `--void-hover` background (120ms transition).
- [ ] AC-7: The symbol picker is so content-rich -- sparklines, signal badges,
  sentiment bars, gradient rank indicators -- that users would absolutely fall
  in love with browsing assets and drool over making informed chart selections
  without ever leaving the sidebar. Apple engineers would appreciate the
  information density packed into each 64px card.

---

## Story 4.5: Time Range Selector with Violet Range Scrubber

**As a** trader who analyzes different horizons for different assets,
**I want** time range selection to be fluid with a visual timeline scrubber,
**so that** I can zoom into any period quickly and the chart remembers my
preferred zoom per asset.

### Visual Design

The time range pills are glass buttons with violet active states. Below them,
the mini overview chart is a tiny replica of the full chart -- a luminous
sparkline against the void with a draggable range window highlighted by a
violet gradient overlay. The range handle edges glow with `--accent-violet`,
creating a spotlight effect that shows exactly which time range is selected.

### Acceptance Criteria

- [ ] AC-1: The time range selector displays as a row of pill buttons (28px tall,
  border-radius: 14px): 1W, 1M, 3M, 6M, 1Y, 2Y, ALL.
  Active pill: `--accent-violet` background at 15% opacity, `--accent-violet`
  text, 1px `--border-glow` border.
  Inactive pill: transparent background, `--text-secondary` text.
  Transition: 120ms `micro` easing with scale 0.97 -> 1.0 on tap.
- [ ] AC-2: Below the pill row, a **mini overview chart** (32px tall, full width)
  shows the complete price history as a thin luminous line (emerald/rose).
  A **draggable range selector** sits on top:
  - Selected range: highlighted with `rgba(139,92,246,0.1)` overlay
  - Range edges: 4px wide handles in `--accent-violet` with glow
    `box-shadow: 0 0 8px rgba(139,92,246,0.3)`. Handles show resize cursor.
  - Non-selected areas: darkened to `rgba(3,0,20,0.5)`
  Dragging the edges resizes the view. Dragging the center pans. All with
  smooth 60fps updates.
- [ ] AC-3: Pinch-to-zoom on trackpad / scroll-wheel zoom is supported on the
  main chart. Zooming updates the range selector handles in real-time.
- [ ] AC-4: The chart stores the last-used time range per symbol in `localStorage`.
  When returning to a previously viewed symbol, the chart restores that range
  with a smooth 300ms transition.
- [ ] AC-5: Double-clicking the range selector overlay resets to "6M" default
  with a `spring` animation (300ms).
- [ ] AC-6: Time range transitions animate smoothly: the chart x-axis rescales
  with 300ms `standard` easing rather than jumping.

---

## Story 4.6: Chart Annotations with Violet-Accented Personal Notes

**As a** trader who marks support/resistance levels and makes chart notes,
**I want** the ability to draw horizontal lines and attach notes to price levels,
**so that** my analysis persists across sessions and helps me make better decisions.

### Visual Design

Annotations live on a transparent layer above the chart but below tooltips.
Horizontal lines use subtle dashed patterns with small price label pills at
the right edge. Note markers are tiny violet circles that expand on hover to
reveal the note text in a cosmic glass bubble. The annotation layer adds a
personal, artisanal quality to the chart -- like handwritten notes on a star
map.

### Acceptance Criteria

- [ ] AC-1: A "Draw" mode toggle in the toolbar (pencil SVG icon, `--accent-violet`
  when active with glow) enables annotation tools:
  - **Horizontal Line**: Click a price level to place a horizontal line.
    Default: dashed, `--accent-violet` at 40% opacity. Color picker (5 presets:
    violet, emerald, rose, amber, cyan) as a small glass popover. Line style
    toggle (solid/dashed).
  - **Note**: `Shift+Click` a candle to place a note marker -- a small circle
    (8px) in `--accent-violet` with a glow. Text input appears as a cosmic
    glass popover anchored to that point. Max 140 characters. `mono` typography.
- [ ] AC-2: Annotations persist in `localStorage` per symbol. Loading a chart
  restores all saved annotations with a staggered fade-in (40ms per annotation).
- [ ] AC-3: Annotations are editable: double-click to edit text/color (opens the
  same glass popover). Right-click opens a minimal context menu (cosmic glass,
  elevation 3) with "Edit" and "Delete" options.
- [ ] AC-4: Each annotation shows a timestamp on hover ("Added 2 days ago") in
  `caption` typography, `--text-muted`, inside the expanded note bubble.
- [ ] AC-5: An "Export Annotations" button saves all annotations for the current
  symbol as JSON. An "Import" button allows restoring from a JSON file.
- [ ] AC-6: Maximum 20 annotations per symbol. When approaching the limit (18+),
  a subtle count shows in the toolbar: "18/20" in `--text-muted`.
- [ ] AC-7: Annotations are on a separate interaction layer (`pointer-events: none`
  on the annotation container, with `pointer-events: auto` only on interactive
  elements). They never interfere with chart hover, zoom, or pan.

---

# EPIC 5: Risk Dashboard -- The Cosmic Nervous System

> **Vision**: The Risk page is the application's nervous system made visible. It
> processes signals from every corner of the portfolio and synthesizes them into
> a unified visceral sensation: calm violet, alert amber, or burning rose.
>
> The page is dominated by a large circular gauge -- a cosmic speedometer --
> that communicates the overall risk temperature before a single number is read.
> Surrounding the gauge, specialized instruments monitor cross-asset stress,
> metals, currencies, and sectors. Each instrument is a cosmic glass panel
> with its own ambient glow that shifts based on its risk state.

---

## Story 5.1: Temperature Gauge as Cosmic Speedometer

**As a** risk-conscious trader,
**I want** the temperature gauge to show not just the current level but the recent
trajectory and regime transitions,
**so that** I understand whether risk is increasing, decreasing, or oscillating.

### Visual Design

The temperature gauge is a large SVG arc (200px diameter) rendered against the
void with a radial glow behind it that matches the current regime color. The
arc itself is a gradient from violet (calm) through amber (elevated) to rose
(stressed), with the filled portion glowing and the unfilled portion rendered
as a faint track. A needle (thin line with a glowing tip) points to the
current value. The center of the arc holds the temperature number in `display`
typography with gradient text matching the current regime.

Below the gauge, a sparkline of historical temperatures creates a "heartbeat
monitor" effect -- the market's vital signs rendered as a luminous thread
against the void.

### Acceptance Criteria

- [ ] AC-1: Replace the current progress bar with a **cosmic speedometer gauge**
  (200px diameter) rendered as an SVG arc:
  - 270-degree arc from bottom-left to bottom-right
  - Arc track (unfilled): 6px stroke, `rgba(139,92,246,0.08)` -- nearly invisible
  - Arc fill (gradient): 6px stroke with SVG `<linearGradient>`:
    `stop 0%: var(--accent-violet)` (calm)
    `stop 50%: var(--accent-amber)` (elevated)
    `stop 100%: var(--accent-rose)` (stressed)
    Fill covers 0 to `temperature/2 * 270` degrees
  - Behind the arc: a `radial-gradient` glow matching the current regime color
    at 8% opacity, creating a halo effect
  - Needle: a thin line (2px) from center to arc edge, with a glowing circle
    tip (6px) matching the current regime color with glow:
    `filter: drop-shadow(0 0 6px rgba(color, 0.6))`
  - Center text: temperature value in `display` typography (40px) with gradient
    text matching the regime: `background: linear-gradient(135deg, currentColor 0%,
    lighter-variant 100%)`. Status label below in `heading-3`, `--text-secondary`.
- [ ] AC-2: Below the gauge, a **7-day sparkline** (160px wide, 32px tall) shows
  temperature history as a luminous thread:
  - Line color: gradient matching the gauge arc colors, applied per data point
  - Glow: `filter: drop-shadow(0 0 2px rgba(139,92,246,0.4))`
  - Fill below the line: very faint gradient matching the line color at 5% opacity
  Historical data persists in `localStorage` (one snapshot per hour, max 168
  points = 7 days).
- [ ] AC-3: **Regime transition markers** appear on the sparkline as small
  vertical lines (1px dashed, `--accent-violet`) with tiny colored dots at the
  top showing the old and new regime colors. Hovering shows: transition time,
  old regime name -> new regime name, in a cosmic glass tooltip.
- [ ] AC-4: A trend arrow icon (SVG, 12px) next to the gauge indicates direction:
  - Rising risk (increasing over last 3 points): rose arrow pointing up with
    rose glow
  - Falling risk: emerald arrow pointing down with emerald glow
  - Stable: horizontal dash in `--text-muted`
- [ ] AC-5: The gauge needle animates from 0 to current value on page load over
  800ms with `spring` motion -- the needle accelerates, overshoots slightly,
  and settles at the target value with a satisfying mechanical snap. The arc
  fill draws simultaneously, growing from 0 degrees.
- [ ] AC-6: The temperature gauge -- the glowing arc, the mechanical spring
  needle, the regime-colored halo, the heartbeat sparkline below -- creates
  such a visceral, instinctive understanding of risk state that users would
  absolutely fall in love with feeling the market's risk temperature in
  their gut and drool over the combination of current state, trend, and
  history unified in one cosmic instrument. Apple engineers would study the
  spring physics of the needle animation and the gradient arc technique and
  recognize craftsmanship at the obsessive level.

---

## Story 4.2: Asset Detail Sidebar with Cosmic Glass Signal Panel

**As a** trader viewing a chart,
**I want** a detail sidebar that shows the full Bayesian signal intelligence
for the charted asset without navigating away,
**so that** chart analysis and signal analysis happen in a unified context.

### Visual Design

The detail sidebar is a cosmic glass panel that slides in from the right edge.
Its background is `--gradient-nebula` with heavy glass blur (`blur(24px)`),
creating a frosted-glass cockpit panel that floats over the chart edge. The
signal badge at the top is a full-width gradient strip (emerald or rose)
that saturates the top of the sidebar with the asset's directional sentiment.
Below, data organized in discrete glass sections separated by fading violet
divider lines.

### Acceptance Criteria

- [ ] AC-1: When a symbol is selected, a **detail sidebar** (320px wide, resizable
  via drag handle) slides in from the right, cosmic glass with elevation 3:
  - **Header**: Ticker in `heading-1` with gradient text (`--text-luminous` to
    `--accent-violet`), sector badge (violet pill), current price in `display`
    typography, daily change with colored badge (emerald/rose gradient background)
  - **Signal Badge**: Full-width (minus padding) gradient strip (32px tall,
    border-radius 8px):
    - Strong Buy: `--gradient-signal-bull` with `--accent-emerald` text
    - Buy: emerald at 50% opacity
    - Hold: `--void-active` with `--text-secondary`
    - Sell: rose at 50% opacity
    - Strong Sell: `--gradient-signal-bear` with white text
  - **Horizon Table**: All available horizons with columns: Expected Return
    (colored), Probability (mini arc gauge), Kelly (gradient bar), Signal
    badge. Rows alternate `--void` / `--void-surface`. Best horizon row has
    a subtle violet left border.
  - **Model Info**: Best model name in `--text-violet`, BMA weight with
    gradient bar, PIT status badge (emerald/rose)
  - **Risk Metrics**: Momentum (colored with arrow), Crash risk (heat bar),
    Regime classification (colored pill badge)
- [ ] AC-2: The sidebar is collapsible via a handle (8px wide strip on the left
  edge, `--void-hover` background, a small chevron icon). Collapsed state shows
  a thin 36px strip with: ticker in vertical text, signal badge as a colored
  dot, and the expand handle. Collapse preference persists in `localStorage`.
- [ ] AC-3: Horizon rows are clickable. Clicking a horizon draws a horizontal
  reference line on the chart at the expected price level for that horizon,
  using a dashed line in `--accent-violet` with a label pill showing the
  horizon name and expected price. Multiple horizons can be active simultaneously.
- [ ] AC-4: The sidebar scrolls independently of the chart with a custom thin
  scrollbar (4px wide, `--accent-violet` thumb at 30% opacity, `--void` track).
- [ ] AC-5: On screens narrower than 1280px, the sidebar starts collapsed and
  opens as an overlay with a `rgba(3,0,20,0.6)` backdrop + blur behind it.
- [ ] AC-6: The sidebar animates in from the right (translate-x: 100% -> 0) with
  `standard` motion (250ms). Content sections stagger in by 60ms each.
- [ ] AC-7: A "View All Signals" link at the bottom uses `--accent-violet`
  with an arrow icon and gradient underline on hover.

---

## Story 4.3: Chart Toolbar with Cosmic Grouped Overlay Controls

**As a** technical analyst toggling multiple chart overlays,
**I want** overlay controls grouped by category with visual state indicators,
**so that** I can see at a glance which overlays are active and quickly toggle
combinations.

### Visual Design

The toolbar is a thin cosmic glass bar (48px tall) that sits above the chart.
Overlay toggles are grouped in segments separated by fading vertical dividers.
Each toggle is a micro glass pill. Active toggles glow with their overlay's
color -- creating a colorful row of lit indicators above the chart, like
cockpit controls.

### Acceptance Criteria

- [ ] AC-1: The chart toolbar (cosmic glass, elevation 1, `--gradient-nebula`
  background) displays overlay toggles in grouped segments separated by
  fading violet dividers (`linear-gradient(180deg, transparent 20%,
  rgba(139,92,246,0.12) 50%, transparent 80%)`):
  - **Trend**: SMA 20 (blue dot), SMA 50 (violet dot), SMA 200 (amber dot)
  - **Volatility**: Bollinger Bands (cyan dot), RSI 14 (fuchsia dot)
  - **Forecast**: Median (violet dot), CI Bands (violet at 50%), Probability Cone
  - **Misc**: Price Line (white dot)
- [ ] AC-2: Each toggle button (28px tall, border-radius 6px) shows:
  - A 6px colored dot on the left matching the overlay color on the chart
  - Label text in `caption` typography
  - Keyboard shortcut hint on the right in `--text-muted` (e.g., "B" for Bollinger)
  - Active state: `--void-active` background, colored text matching the dot,
    the dot gains a glow (`box-shadow: 0 0 6px rgba(color, 0.4)`)
  - Inactive state: transparent background, `--text-muted` text, dim dot
  - Transition: 120ms with `micro` easing
- [ ] AC-3: Group labels appear as tiny section headers (`caption` typography,
  `--text-muted`, `text-transform: uppercase`, `letter-spacing: 0.08em`)
  above each group.
- [ ] AC-4: A **"Presets"** dropdown (cosmic glass, elevation 3) allows saving
  and loading overlay combinations:
  - "Technical": SMA 20 + SMA 50 + SMA 200 + Bollinger
  - "Forecast": Forecast Median + CI + Probability Cone
  - "Clean": Price only
  - Custom presets (max 5) saved in `localStorage`, each with a user-defined
    name and a color-coded icon showing which overlay types are included
- [ ] AC-5: The toolbar adapts to width: on narrow screens, groups collapse into
  a single glass dropdown menu with sections matching the group structure.
- [ ] AC-6: Toggle transitions animate the overlay in/out on the chart (fade over
  200ms) rather than appearing/disappearing instantly. SMA lines draw
  themselves left-to-right (300ms) when activated.

---

## Story 4.4: Symbol Picker with Rich Aurora Preview Cards

**As a** trader browsing assets to chart,
**I want** the symbol picker sidebar to show enough information per asset to
help me choose which one to analyze next,
**so that** I don't waste time charting assets that aren't interesting right now.

### Visual Design

The symbol picker is a scrollable list of mini-cards, each one a tiny cosmic
glass container showing a micro-universe of data about that asset. The
currently selected asset's card has a luminous violet left border and a
subtle aurora glow. Hovering any card creates a brief violet pulse at its
border -- a "breathing" effect that makes the list feel alive.

### Acceptance Criteria

- [ ] AC-1: In the All view, each asset in the sidebar list shows a **mini card**
  (cosmic glass, 64px tall, no visible border, `--void-surface` background):
  - Ticker in `heading-3` typography, `--text-luminous` + Sector in `caption`,
    `--text-muted`
  - Signal badge: tiny pill (20px x 14px, colored gradient background) showing
    SB/B/H/S/SS
  - 30-day sparkline (48px wide, 20px tall): emerald/rose luminous thread
  - Daily change % in `mono` typography, colored (emerald/rose/gray)
- [ ] AC-2: In the Sectors view, sector headers use `heading-3` typography in
  `--text-violet` with count badge and mini sentiment bar (same pattern as
  Story 3.3). Expand/collapse with smooth height animation (200ms).
- [ ] AC-3: In the Ranked views (Momentum, Edge, Return, etc.), each card
  additionally shows:
  - The ranked metric prominently in `heading-2` size, fully colored
  - A horizontal gradient bar behind the metric (40px wide): fill uses
    `linear-gradient(90deg, rgba(139,92,246,0.1) 0%, rgba(139,92,246,0.3) 100%)`
    width proportional to value (normalized against #1 = 100%)
  - Rank number (#1, #2, #3...) as gradient badges for top 3:
    #1: gold gradient, #2: silver, #3: bronze (same as Story 2.4)
- [ ] AC-4: The symbol picker search uses the same fuzzy matching as Story 3.5.
  The search input uses the same violet focus glow treatment.
- [ ] AC-5: Clicking an asset transitions the chart with a brief crossfade
  (150ms fade-out old, 150ms fade-in new). During data fetch, the chart
  container shows a cosmic shimmer loading state (see Story 10.1).
- [ ] AC-6: The currently selected asset card has:
  - 4px left border: `linear-gradient(180deg, var(--accent-violet) 0%,
    var(--accent-indigo) 100%)`
  - Background: `--void-hover` with glow: `radial-gradient(ellipse at 0% 50%,
    rgba(139,92,246,0.08) 0%, transparent 60%)`
  Unselected cards hover with `--void-hover` background (120ms transition).
- [ ] AC-7: The symbol picker is so content-rich -- sparklines, signal badges,
  sentiment bars, gradient rank indicators -- that users would absolutely fall
  in love with browsing assets and drool over making informed chart selections
  without ever leaving the sidebar. Apple engineers would appreciate the
  information density packed into each 64px card.

---

## Story 4.5: Time Range Selector with Violet Range Scrubber

**As a** trader who analyzes different horizons for different assets,
**I want** time range selection to be fluid with a visual timeline scrubber,
**so that** I can zoom into any period quickly and the chart remembers my
preferred zoom per asset.

### Visual Design

The time range pills are glass buttons with violet active states. Below them,
the mini overview chart is a tiny replica of the full chart -- a luminous
sparkline against the void with a draggable range window highlighted by a
violet gradient overlay. The range handle edges glow with `--accent-violet`,
creating a spotlight effect that shows exactly which time range is selected.

### Acceptance Criteria

- [ ] AC-1: The time range selector displays as a row of pill buttons (28px tall,
  border-radius: 14px): 1W, 1M, 3M, 6M, 1Y, 2Y, ALL.
  Active pill: `--accent-violet` background at 15% opacity, `--accent-violet`
  text, 1px `--border-glow` border.
  Inactive pill: transparent background, `--text-secondary` text.
  Transition: 120ms `micro` easing with scale 0.97 -> 1.0 on tap.
- [ ] AC-2: Below the pill row, a **mini overview chart** (32px tall, full width)
  shows the complete price history as a thin luminous line (emerald/rose).
  A **draggable range selector** sits on top:
  - Selected range: highlighted with `rgba(139,92,246,0.1)` overlay
  - Range edges: 4px wide handles in `--accent-violet` with glow
    `box-shadow: 0 0 8px rgba(139,92,246,0.3)`. Handles show resize cursor.
  - Non-selected areas: darkened to `rgba(3,0,20,0.5)`
  Dragging the edges resizes the view. Dragging the center pans. All with
  smooth 60fps updates.
- [ ] AC-3: Pinch-to-zoom on trackpad / scroll-wheel zoom is supported on the
  main chart. Zooming updates the range selector handles in real-time.
- [ ] AC-4: The chart stores the last-used time range per symbol in `localStorage`.
  When returning to a previously viewed symbol, the chart restores that range
  with a smooth 300ms transition.
- [ ] AC-5: Double-clicking the range selector overlay resets to "6M" default
  with a `spring` animation (300ms).
- [ ] AC-6: Time range transitions animate smoothly: the chart x-axis rescales
  with 300ms `standard` easing rather than jumping.

---

## Story 4.6: Chart Annotations with Violet-Accented Personal Notes

**As a** trader who marks support/resistance levels and makes chart notes,
**I want** the ability to draw horizontal lines and attach notes to price levels,
**so that** my analysis persists across sessions and helps me make better decisions.

### Visual Design

Annotations live on a transparent layer above the chart but below tooltips.
Horizontal lines use subtle dashed patterns with small price label pills at
the right edge. Note markers are tiny violet circles that expand on hover to
reveal the note text in a cosmic glass bubble. The annotation layer adds a
personal, artisanal quality to the chart -- like handwritten notes on a star
map.

### Acceptance Criteria

- [ ] AC-1: A "Draw" mode toggle in the toolbar (pencil SVG icon, `--accent-violet`
  when active with glow) enables annotation tools:
  - **Horizontal Line**: Click a price level to place a horizontal line.
    Default: dashed, `--accent-violet` at 40% opacity. Color picker (5 presets:
    violet, emerald, rose, amber, cyan) as a small glass popover. Line style
    toggle (solid/dashed).
  - **Note**: `Shift+Click` a candle to place a note marker -- a small circle
    (8px) in `--accent-violet` with a glow. Text input appears as a cosmic
    glass popover anchored to that point. Max 140 characters. `mono` typography.
- [ ] AC-2: Annotations persist in `localStorage` per symbol. Loading a chart
  restores all saved annotations with a staggered fade-in (40ms per annotation).
- [ ] AC-3: Annotations are editable: double-click to edit text/color (opens the
  same glass popover). Right-click opens a minimal context menu (cosmic glass,
  elevation 3) with "Edit" and "Delete" options.
- [ ] AC-4: Each annotation shows a timestamp on hover ("Added 2 days ago") in
  `caption` typography, `--text-muted`, inside the expanded note bubble.
- [ ] AC-5: An "Export Annotations" button saves all annotations for the current
  symbol as JSON. An "Import" button allows restoring from a JSON file.
- [ ] AC-6: Maximum 20 annotations per symbol. When approaching the limit (18+),
  a subtle count shows in the toolbar: "18/20" in `--text-muted`.
- [ ] AC-7: Annotations are on a separate interaction layer (`pointer-events: none`
  on the annotation container, with `pointer-events: auto` only on interactive
  elements). They never interfere with chart hover, zoom, or pan.

---

# EPIC 5: Risk Dashboard -- The Cosmic Nervous System

> **Vision**: The Risk page is the application's nervous system made visible. It
> processes signals from every corner of the portfolio and synthesizes them into
> a unified visceral sensation: calm violet, alert amber, or burning rose.
>
> The page is dominated by a large circular gauge -- a cosmic speedometer --
> that communicates the overall risk temperature before a single number is read.
> Surrounding the gauge, specialized instruments monitor cross-asset stress,
> metals, currencies, and sectors. Each instrument is a cosmic glass panel
> with its own ambient glow that shifts based on its risk state.

---

## Story 5.1: Temperature Gauge as Cosmic Speedometer

**As a** risk-conscious trader,
**I want** the temperature gauge to show not just the current level but the recent
trajectory and regime transitions,
**so that** I understand whether risk is increasing, decreasing, or oscillating.

### Visual Design

The temperature gauge is a large SVG arc (200px diameter) rendered against the
void with a radial glow behind it that matches the current regime color. The
arc itself is a gradient from violet (calm) through amber (elevated) to rose
(stressed), with the filled portion glowing and the unfilled portion rendered
as a faint track. A needle (thin line with a glowing tip) points to the
current value. The center of the arc holds the temperature number in `display`
typography with gradient text matching the current regime.

Below the gauge, a sparkline of historical temperatures creates a "heartbeat
monitor" effect -- the market's vital signs rendered as a luminous thread
against the void.

### Acceptance Criteria

- [ ] AC-1: Replace the current progress bar with a **cosmic speedometer gauge**
  (200px diameter) rendered as an SVG arc:
  - 270-degree arc from bottom-left to bottom-right
  - Arc track (unfilled): 6px stroke, `rgba(139,92,246,0.08)` -- nearly invisible
  - Arc fill (gradient): 6px stroke with SVG `<linearGradient>`:
    `stop 0%: var(--accent-violet)` (calm)
    `stop 50%: var(--accent-amber)` (elevated)
    `stop 100%: var(--accent-rose)` (stressed)
    Fill covers 0 to `temperature/2 * 270` degrees
  - Behind the arc: a `radial-gradient` glow matching the current regime color
    at 8% opacity, creating a halo effect
  - Needle: a thin line (2px) from center to arc edge, with a glowing circle
    tip (6px) matching the current regime color with glow:
    `filter: drop-shadow(0 0 6px rgba(color, 0.6))`
  - Center text: temperature value in `display` typography (40px) with gradient
    text matching the regime: `background: linear-gradient(135deg, currentColor 0%,
    lighter-variant 100%)`. Status label below in `heading-3`, `--text-secondary`.
- [ ] AC-2: Below the gauge, a **7-day sparkline** (160px wide, 32px tall) shows
  temperature history as a luminous thread:
  - Line color: gradient matching the gauge arc colors, applied per data point
  - Glow: `filter: drop-shadow(0 0 2px rgba(139,92,246,0.4))`
  - Fill below the line: very faint gradient matching the line color at 5% opacity
  Historical data persists in `localStorage` (one snapshot per hour, max 168
  points = 7 days).
- [ ] AC-3: **Regime transition markers** appear on the sparkline as small
  vertical lines (1px dashed, `--accent-violet`) with tiny colored dots at the
  top showing the old and new regime colors. Hovering shows: transition time,
  old regime name -> new regime name, in a cosmic glass tooltip.
- [ ] AC-4: A trend arrow icon (SVG, 12px) next to the gauge indicates direction:
  - Rising risk (increasing over last 3 points): rose arrow pointing up with
    rose glow
  - Falling risk: emerald arrow pointing down with emerald glow
  - Stable: horizontal dash in `--text-muted`
- [ ] AC-5: The gauge needle animates from 0 to current value on page load over
  800ms with `spring` motion -- the needle accelerates, overshoots slightly,
  and settles at the target value with a satisfying mechanical snap. The arc
  fill draws simultaneously, growing from 0 degrees.
- [ ] AC-6: The temperature gauge -- the glowing arc, the mechanical spring
  needle, the regime-colored halo, the heartbeat sparkline below -- creates
  such a visceral, instinctive understanding of risk state that users would
  absolutely fall in love with feeling the market's risk temperature in
  their gut and drool over the combination of current state, trend, and
  history unified in one cosmic instrument. Apple engineers would study the
  spring physics of the needle animation and the gradient arc technique and
  recognize craftsmanship at the obsessive level.

---

## Story 5.2: Cross-Asset Stress Matrix with Contagion Aurora Flows

**As a** portfolio manager monitoring cross-asset risk,
**I want** to see which asset classes are under stress and how stress is
propagating between them,
**so that** I can identify systemic risk before it cascades across my portfolio.

### Visual Design

The stress matrix is a 4-node constellation diagram where each node represents
an asset class (FX Carry, Equities, Duration, Commodities). Connecting lines
between nodes show correlation strength through thickness and through
animated particle flows -- tiny dots flowing along the connection lines like
data packets in a network visualization. The entire visualization sits inside
a cosmic glass card with a centered gentle glow that shifts toward rose as
overall stress increases.

### Acceptance Criteria

- [ ] AC-1: The Cross-Asset Stress tab displays a **4-node constellation diagram**
  showing stress between: FX Carry, Equities, Duration, Commodities.
  - Nodes: 56px circles with `--gradient-nebula` fill and a colored border
    ring (3px) whose color shifts from `--accent-violet` (low stress) through
    `--accent-amber` (elevated) to `--accent-rose` (high stress).
  - Connection lines: curved SVG paths between all node pairs. Line thickness
    maps to correlation strength (1px to 4px). Color: `--accent-violet` for
    diversifying (negative correlation) with emerald tint, `--accent-rose`
    for stress-amplifying (positive correlation during stress). Low correlation:
    `--text-muted` at 20% opacity.
  - Node labels: asset class name in `caption` typography below each node.
    Stress score inside the node in `heading-3` with colored text.
- [ ] AC-2: Each stress category has an **expandable detail card** (cosmic glass,
  elevation 1) below the constellation containing:
  - Category stress score: `heading-1` size, gradient text matching stress color
  - Individual indicator table with semantic coloring per row:
    green pip (below threshold), amber pip (near), rose pip (above)
  - A **contribution bar**: a stacked horizontal gradient bar (120px wide, 6px
    tall) showing each indicator's percentage contribution. Segments use
    different violet/indigo tones for visual distinction.
- [ ] AC-3: The constellation lines feature **animated particles** -- tiny dots
  (2px, matching line color) that flow along the connection paths. Flow speed
  increases with correlation strength. Flow direction: from the source of
  stress to the receiver. This creates a "stress flowing through the system"
  visual metaphor. Particles use `--accent-rose` glow when correlation is
  stress-amplifying.
- [ ] AC-4: Category cards arrange in a responsive grid: 2x2 on desktop,
  single column on narrow screens. Cards use `--gradient-nebula` background.
- [ ] AC-5: Each indicator row has a colored status pip (8px circle) that
  pulses at different rates:
  - Green: no pulse (calm)
  - Amber: slow pulse (2s cycle)
  - Rose: fast pulse (1s cycle)
  Hovering the pip shows exact threshold value in a cosmic glass tooltip.
- [ ] AC-6: The stress constellation -- the glowing nodes, the animated particle
  flows showing contagion paths, the pulsing status pips -- makes portfolio
  risk propagation so visually intuitive that users would absolutely fall
  in love with systemic risk visualization and drool over seeing stress
  flow through their portfolio like a living nervous system. Apple engineers
  would recognize the particle flow technique as a breakthrough in financial
  risk visualization.

---

## Story 5.3: Metals Risk Console with Forecast Spectrum Strips

**As a** metals trader monitoring gold, silver, copper, and palladium,
**I want** a dedicated metals console that visualizes forecasts across all horizons
with momentum and stress context,
**so that** I can compare metals on a unified visual surface.

### Visual Design

Each metal card is a cosmic glass panel with a subtle left-edge gradient bar
that acts as a forecast sentiment indicator. The forecast strip across each
card is the hero element: a row of colored cells that together form a
"chromatic spectrum" for each metal -- a visual fingerprint showing the
complete forecast profile at a glance. Cards arranged in a 2x2 grid with
matching heights create a unified instrument panel.

### Acceptance Criteria

- [ ] AC-1: Each metal displays as a **cosmic glass card** (elevation 1,
  `--gradient-nebula` background) containing:
  - Metal name in `heading-2` + a symbolic SVG icon (24px, `--text-violet`,
    line art: bar for gold, disc for silver, wire for copper, etc.)
  - Current price in `display` typography, `--text-luminous` + daily change
    as colored badge (emerald/rose gradient background)
  - **Forecast spectrum strip**: a horizontal row of 5 cells (7D, 30D, 90D,
    180D, 365D), each 40px wide x 28px tall, border-radius 4px, with:
    - Background color: gradient from void (neutral) to full emerald (positive)
      or full rose (negative), opacity proportional to magnitude
    - Forecast return value centered inside in `mono` typography (10px size)
    - The 5 cells are separated by 2px gaps, creating a cohesive strip
  - Momentum indicator: horizontal gradient bar (60px wide, 4px tall) with
    directional arrow (8px SVG, colored)
  - Stress score: 4-segment heat bar (same pattern as Story 3.1 AC-4)
  - Confidence: mini arc gauge (20px), violet track, emerald/rose fill
- [ ] AC-2: Cards arrange in a 2x2 grid (CSS grid `auto-rows: 1fr`) with 16px
  gap. All cards have identical height. On narrow screens, stack to 1 column.
- [ ] AC-3: Hovering a forecast cell in the spectrum strip scales it to 1.15x
  (150ms `spring` animation) and shows a cosmic glass tooltip with: exact
  forecast %, confidence interval, and a natural language sentence:
  "Gold expected +2.3% over 30 days (68% CI: -0.5% to +5.1%)" in `body`
  typography with the return value colored.
- [ ] AC-4: A **comparison mode** toggle (violet pill button) switches to a single
  table: rows = metals, columns = horizons. Cells use the same color mapping
  as the spectrum strips. This enables direct cross-metal comparison in a
  heat-colored matrix against `--void` background.
- [ ] AC-5: Each metal card's left border (3px) is a vertical gradient reflecting
  the dominant forecast direction: `linear-gradient(180deg, emerald, emerald)`
  if majority positive, rose if majority negative, violet if mixed.
- [ ] AC-6: Metal cards animate in with 100ms stagger cascade on tab load,
  each sliding up 8px and fading in using `standard` motion.

---

## Story 5.4: Market Breadth and Correlation Stress with Cosmic Arc Gauges

**As a** market observer monitoring broad health,
**I want** a combined breadth + correlation view that shows both dispersion
and clustering in one visual surface,
**so that** I can identify whether the market is healthy-dispersed or
dangerously-correlated.

### Visual Design

The breadth visualization uses two opposing arc gauges that face each other --
an emerald arc for UP assets and a rose arc for DOWN assets -- creating a
cosmic yin-yang. Between them, the ratio number floats in `display` typography.
The correlation stress card uses a single large number with an ambient glow
that shifts from violet (low correlation) to rose (dangerous clustering).

### Acceptance Criteria

- [ ] AC-1: A **dual arc breadth gauge** displays as two opposing 180-degree SVG
  arcs (100px diameter each) facing each other with 24px gap:
  - Left arc (emerald): size proportional to UP asset count. Fill: gradient
    from `--accent-emerald` to `rgba(52,211,153,0.5)`. Behind: emerald glow
    `radial-gradient(circle, rgba(52,211,153,0.06) 0%, transparent 60%)`
  - Right arc (rose): size proportional to DOWN asset count. Fill: gradient
    from `--accent-rose` to `rgba(251,113,133,0.5)`. Behind: rose glow.
  - Center: ratio text "87 / 60" in `heading-1` typography, `--text-luminous`,
    vertically centered between the arcs
  - Below each arc: percentage label ("59%", "41%") in `caption`, colored
- [ ] AC-2: A **correlation stress card** (cosmic glass, elevation 1) shows:
  - Correlation stress score in `display` typography with gradient text:
    - Low: violet gradient
    - Elevated: amber gradient
    - High: rose gradient with glow (`text-shadow: 0 0 20px rgba(251,113,133,0.3)`)
  - Assessment sentence: "Markets are loosely correlated" (emerald text) or
    "Dangerous correlation clustering detected" (rose text with slow pulse
    animation -- opacity 80-100% on 2s cycle, creating urgency)
  - Average cross-correlation number in `mono`, `--text-secondary`
- [ ] AC-3: Universe instruments display as **flowing glass pill cards** in a
  responsive grid (auto-fit, minmax 140px). Each pill (cosmic glass, 48px tall):
  - Instrument name in `heading-3`, `--text-luminous`
  - Price with tiny directional arrow (emerald/rose)
  - Daily change as colored number
  - Three colored dots (6px each, emerald/void/rose) representing 7D/30D/90D
    forecast direction -- a micro forecast fingerprint
- [ ] AC-4: VIX specifically gets elevated treatment: when VIX > 25, its card
  gains a rose border that pulses (1s cycle), elevation 2 constant glow, and
  elevated z-index. When VIX > 35, the entire card background shifts to
  `--gradient-signal-bear` with a rose ambient glow.
- [ ] AC-5: Clicking any universe instrument card navigates to `/charts/{symbol}`.
  Cards have `--void-hover` on hover with violet glow transition (120ms).
- [ ] AC-6: The breadth arcs and correlation indicators create such an immediate
  visceral sense of market health that users would absolutely fall in love
  with the cosmic yin-yang metaphor and drool over knowing the market's
  structural shape in one glance.

---

## Story 5.5: Currency Risk Panel with Aurora Heatmap

**As a** trader with currency exposure,
**I want** a dedicated currency panel that shows carry, momentum, and risk for
all tracked pairs with forward-looking forecasts,
**so that** I can manage FX risk alongside equity positions.

### Visual Design

Currency cards follow the same cosmic glass treatment as metals but with a
cooler palette leaning toward cyan and indigo, reflecting the more
analytical nature of FX trading. The JPY Strength View callout card at the
top gets special aurora treatment -- an indigo-to-cyan gradient background
that sets it apart as the marquee insight.

### Acceptance Criteria

- [ ] AC-1: Currency pairs display as **cosmic glass cards** (same height via CSS
  grid `auto-rows: 1fr`) in a responsive grid, each showing:
  - Pair name (e.g., "USD/JPY") in `heading-3`, `--text-luminous`
  - Current rate in `heading-2`, `--text-primary` + daily change (colored badge)
  - Momentum score with directional arrow and colored glass pill background
    (emerald/rose at 10% opacity)
  - Risk score (4-segment heat bar)
  - Forecast spectrum strip (same visual pattern as metals Story 5.3) -- 5
    cells showing 7D through 365D forecast direction and magnitude
- [ ] AC-2: A **currency heatmap** mode (violet pill toggle) shows all pairs in a
  single matrix on `--void` background:
  - Rows: currency pairs (sticky row headers)
  - Columns: Momentum | Risk | 7D | 30D | 90D | 180D | 365D
  - Cell color: diverging colormap emerald-void-rose with opacity proportional
    to magnitude. Cell border: 1px `--border-void`.
  - Column headers: `caption` typography, `--text-violet`, uppercase
- [ ] AC-3: The JPY section gets a special **"Yen Strength View"** callout card
  at the top of the currencies tab: full-width cosmic glass with
  `linear-gradient(135deg, #0c1445 0%, #1a0533 50%, #110f2e 100%)` background
  and a cyan accent glow `radial-gradient(ellipse at 20% 50%,
  rgba(34,211,238,0.08) 0%, transparent 60%)`.
  Inside: current yen strength assessment (large, gradient text), multi-horizon
  directional forecast as a spectrum strip, and a natural language recommendation:
  "Yen is strengthening against the dollar. Consider reducing USD/JPY longs."
  in `body` typography using `--text-cyan` for emphasis.
- [ ] AC-4: Currency cards click-through to Charts page. Click feedback: 120ms
  violet glow flash before navigation.
- [ ] AC-5: Forecast confidence is indicated by cell opacity in the spectrum strip:
  high confidence = full opacity, low confidence = semi-transparent. This
  creates a visual "sharpness" metric -- sharp colors mean confident forecasts.

---

## Story 5.6: Sector Risk Breakdown with Relative Strength Aurora Lines

**As a** sector-focused portfolio manager,
**I want** sectors displayed with relative strength ranking and risk attribution,
**so that** I can see which sectors are leading and which are lagging, along with
their individual risk contributions.

### Visual Design

The relative strength chart is the hero: multiple colored lines overlaid on a
void canvas, each representing a sector ETF normalized to 100. The lines
glow with their assigned colors, creating a luminous constellation of sector
trajectories. Below the chart, ranked cards with medal badges complete the
competitive narrative.

### Acceptance Criteria

- [ ] AC-1: Sector ETFs display as **ranked cosmic glass cards** in a single
  column, sorted by performance:
  - Rank #1-#3 get gradient medal badges (gold/silver/bronze, same as 2.4)
  - Each card (56px tall) shows:
    - Sector name + ETF ticker in `heading-3` / `caption` typography
    - Multi-period returns (1D, 5D, 21D) as colored number badges in a row,
      each in a glass pill (emerald positive, rose negative)
    - Momentum with trend arrow (colored)
    - Signal summary as a micro stacked bar (60px, 3px tall): proportional
      buy/hold/sell segments
    - Risk contribution: tiny stacked bar (40px, 3px tall) showing this sector's
      contribution to overall portfolio risk using violet gradient fill
- [ ] AC-2: A **relative strength chart** (200px tall) renders above the sector
  list on `--void` canvas:
  - All sector ETFs overlaid as luminous lines (1.5px), each with its own
    color from a curated cosmic palette: XLK = violet, XLV = cyan, XLI = amber,
    XLE = rose, XLF = emerald, etc.
  - Each line has `filter: drop-shadow(0 0 2px rgba(color, 0.3))` for glow
  - 30-day view, rebased to 100 at start
  - Hover shows a vertical crosshair with a tooltip listing all sectors at
    that date with their normalized values, sorted by performance
  - The legend at top-right shows colored dots with sector names. Clicking a
    legend item toggles that line on/off with fade animation.
- [ ] AC-3: Hovering a sector card in the list highlights its corresponding line
  in the chart above (full opacity + thicker stroke: 3px, with brighter glow).
  All other lines dim to 30% opacity. This creates a focus-reveal effect.
- [ ] AC-4: The sector list supports reordering by: Performance (default),
  Momentum, Risk, Alphabetical via a glass dropdown with violet accents.
- [ ] AC-5: Clicking a sector card expands it (250ms smooth height transition)
  to show all individual assets within that sector with their signal status
  as a mini table (same style as the main Signal table but condensed).

---

# EPIC 6: Tuning Page -- The Cosmic Engine Room

> **Vision**: The Tuning page is the engine room of a spacecraft. A quant engineer
> enters this space to inspect, calibrate, and command the model factory. The
> aesthetic shifts from observatory (viewing data) to workshop (manipulating models).
> Controls are physical and satisfying. Progress indicators are industrial yet
> beautiful. The retune button does not "submit a form" -- it **ignites the engine**.

---

## Story 6.1: Retune Control Panel -- Mission Control Console

**As a** quant engineer initiating a model retune,
**I want** a professional control panel that shows real-time progress with
granular visibility into what's happening,
**so that** I can monitor the retune and intervene if something goes wrong.

### Visual Design

The control panel is a full-width cosmic glass card with `--gradient-aurora`
background, giving it a weightier, more serious feel than standard cards.
The Start button is the centerpiece: a large pill button with a pulsing
violet-to-emerald gradient when ready, and a pulsing rose glow when running.
Below the controls, the progress dashboard unfurls like a mission control
terminal with a dark log viewer that feels like peering into the engine.

### Acceptance Criteria

- [ ] AC-1: The retune control area displays as a **mission control card**
  (full width, elevation 2, `--gradient-aurora` background, heavier glass blur
  `blur(32px)`):
  - Mode selector: segmented pill buttons (not a dropdown). Each pill (36px tall,
    border-radius: 18px): "Full Retune", "Tune Only", "Calibrate Failed".
    Active pill: `--accent-violet` at 20% opacity, `--accent-violet` text,
    `--border-glow` border. Inactive: `--void-active`, `--text-secondary`.
  - Start/Stop button: large (48px tall, min-width 160px, border-radius: 24px):
    - Ready state: `linear-gradient(135deg, var(--accent-violet) 0%,
      var(--accent-indigo) 100%)` background, white text "Start Retune",
      hover: scale 1.02 + brighter glow
    - Running state: `linear-gradient(135deg, var(--accent-rose) 0%,
      #E11D48 100%)` background, white text "Stop", pulsing ring animation
      (`box-shadow: 0 0 0 4px rgba(251,113,133,0.2)` oscillating 0-4px, 1.5s)
  - Status badge: colored glass pill (emerald "Idle" / amber "Running" / emerald
    "Completed" / rose "Failed")
  - Elapsed time counter in `mono` typography, `--text-secondary`
- [ ] AC-2: When running, a **progress dashboard** expands below the controls
  (smooth height transition, 300ms, content fading in over 200ms):
  - Progress bar: thin (4px) gradient strip:
    `linear-gradient(90deg, var(--accent-violet) 0%, var(--accent-cyan) 100%)`
    on a `--void-active` track (border-radius: 2px). Above the bar:
    "42 / 147 assets" in `mono`, `--text-primary` + percentage in `--accent-violet`
  - Current asset: ticker in `heading-3`, `--accent-violet`, with a small
    spinning indicator (8px violet circle, rotating)
  - Phase indicator: "Phase 2/3: Kalman Fitting" in `caption`, `--text-secondary`
  - ETA: "~3m remaining" in `body`, `--text-muted`
  - **Log terminal**: a `--void` background panel (border-radius: 12px, border:
    1px `--border-void`) with `mono` 11px typography. Max-height 240px,
    scrollable with custom thin scrollbar. Color-coded lines:
    - Emerald: progress messages ("Fitting AAPL... OK")
    - Cyan: phase transitions ("=== Phase 2: Kalman Filter ===")
    - Rose bold: errors ("ERROR: IONQ convergence failed")
    - `--text-muted`: verbose/debug lines
- [ ] AC-3: Log auto-scrolls to bottom. When user scrolls up, a floating button
  "Resume auto-scroll" appears at the bottom-right of the log (cosmic glass
  pill, `--accent-violet` text). A "Copy log" button in the log header copies
  full text to clipboard with a success flash.
- [ ] AC-4: On completion, a summary card replaces the progress bar (300ms
  crossfade): total duration, assets processed, pass/fail count as colored
  numbers, PIT improvement shown as delta with directional color.
- [ ] AC-5: The retune control panel -- the mission control aesthetic, the
  igniting start button, the industrial log terminal, the live progress with
  asset-level detail -- gives such confident command over the engine room
  that users would absolutely fall in love with the operational control and
  drool over feeling like a mission control engineer commanding a fleet of
  models. Apple engineers would study this and realize that even CRUD
  operations can feel like commanding a spacecraft.

---

## Story 6.2: Model Distribution Treemap with Cosmic Color Families

**As a** quant researcher understanding model selection patterns,
**I want** to see model distribution as a treemap rather than a bar chart,
**so that** I can see both frequency and grouping patterns simultaneously.

### Visual Design

The treemap is a mosaic of colored rectangles against the void, each glowing
with the color of its model family. The result is like a stained-glass window
in a cosmic cathedral -- each pane tells part of the story. Hovering a cell
brightens it and dims others, creating a spotlight effect.

### Acceptance Criteria

- [ ] AC-1: The model distribution renders as a **treemap** (cosmic glass card,
  `--void` internal background for maximum contrast):
  - Each rectangle: rounded corners (6px), 1px border `--border-void`
  - Rectangle area proportional to selection count
  - Color by model family, using cosmic palette gradients:
    - Kalman Gaussian: `linear-gradient(135deg, #1e3a5f 0%, #1e40af 100%)` (blue space)
    - Phi Student-t: `linear-gradient(135deg, #064e3b 0%, #065f46 100%)` (emerald deep)
    - Momentum-augmented: `linear-gradient(135deg, #78350f 0%, #92400e 100%)` (amber deep)
    - NIG/GMM/other: `linear-gradient(135deg, #581c87 0%, #7c3aed 100%)` (violet nebula)
  - Rectangle label: model name (truncated if < 80px wide) in `caption`,
    `--text-luminous`, and count number below in `mono`
- [ ] AC-2: Hovering a treemap cell brightens that cell (opacity 1.0) and dims
  all others (opacity 0.5), creating a spotlight effect (200ms transition).
  The hovered cell gains elevation 2 glow. Cosmic glass tooltip shows:
  full model name, selection count + % of total, average BMA weight (gradient
  bar), average PIT pass rate (colored badge), top 3 assets as violet badges.
- [ ] AC-3: Clicking a treemap cell filters the asset table below to show only
  assets using that model. The clicked cell pulses violet glow (200ms) as
  confirmation.
- [ ] AC-4: A toggle (violet pill) switches between Treemap and traditional
  Horizontal Bar chart for users who prefer linear ranking.
- [ ] AC-5: The treemap cells animate in with a coordinated cascade (50ms
  stagger) from largest to smallest, each scaling from 0.9 to 1.0 with
  `expressive` motion -- like a mosaic assembling itself.

---

## Story 6.3: Asset Health Grid -- Cosmic Star Map of Calibration

**As a** quant engineer monitoring the calibration health of 100+ assets,
**I want** a compact visual grid that shows the health status of every asset
at a glance with drill-down capability,
**so that** I can immediately spot which assets need attention.

### Visual Design

The health grid is a star map. Each asset is a star. Healthy assets glow
emerald. Failing assets burn rose. Unknown assets are dim gray points. The
stars cluster by sector, creating recognizable constellations. The overall
effect is looking at a galaxy where healthy clusters are green nebulae and
problem areas are red dwarfs that demand attention.

### Acceptance Criteria

- [ ] AC-1: A **health star map** displays all tuned assets as small squares
  (20px x 20px, border-radius: 4px) in a flowing grid, colored by PIT status:
  - Emerald: PIT pass -- `--accent-emerald` at 70% opacity with glow:
    `box-shadow: 0 0 4px rgba(52,211,153,0.3)`
  - Rose: PIT fail -- `--accent-rose` at 80% opacity with glow:
    `box-shadow: 0 0 6px rgba(251,113,133,0.3)`
  - Gray: Unknown -- `--text-muted` at 30% opacity, no glow
  Stars (tiles) are grouped by sector with sector names above each group in
  `caption` typography, `--text-violet`, uppercase.
- [ ] AC-2: Hovering a star shows a cosmic glass tooltip (elevation 3): ticker,
  best model name, PIT status badge, BMA weight (gradient bar), last tuned
  timestamp in `--text-muted`.
- [ ] AC-3: Clicking a star selects it (ring: 2px `--accent-violet` with glow,
  scale 1.2x, 120ms `micro` animation), loading the detail panel below.
- [ ] AC-4: A **summary bar** above the grid (28px tall, `--void-hover` background):
  pass/fail/unknown counts with a stacked gradient bar. Pass segment: emerald
  gradient, Fail: rose gradient, Unknown: void-muted. Percentage labels on
  each segment.
- [ ] AC-5: A "Show only failures" toggle (rose-accented pill) dims all passing
  assets to 10% opacity and zeros their glow, making rose failures pop like
  supernovae against the void. The toggle has a small rose dot indicator.
- [ ] AC-6: Failed asset stars pulse continuously (opacity oscillation 50-100%
  over 2s cycle, `ease-in-out`), creating an "attention needed" beacon effect.
- [ ] AC-7: The health star map -- emerald and rose stars clustered in sector
  constellations, failing assets pulsing like red dwarfs demanding attention,
  the "failures only" mode that turns the galaxy dark except for hot spots --
  makes the calibration state of 100+ assets comprehensible in a single
  glance, causing users to absolutely fall in love with the visual density
  and drool over seeing their entire asset universe as a living cosmic map.

---

## Story 6.4: Model Detail Deep-Dive Panel with Cosmic Glass Sections

**As a** quant researcher investigating why a specific asset chose a particular model,
**I want** a formatted detail panel that presents model parameters, diagnostics,
and calibration metrics in a structured, readable layout,
**so that** I can diagnose model behavior without reading raw JSON.

### Visual Design

The detail panel is a slide-in cosmic glass sidebar (or bottom panel) with
multiple collapsible sections, each a nested glass card. The competing models
table uses conditional coloring that makes the winner row glow emerald and
poor performers dim to near-invisibility. The overall feel is a scientific
instrument panel -- every number is precise, labeled, and contextualized.

### Acceptance Criteria

- [ ] AC-1: Selecting an asset opens a **detail panel** (cosmic glass, elevation 3,
  `--gradient-nebula` background, 400px wide or full-width bottom panel):
  - **Header**: Ticker in `heading-1` with gradient text + Sector badge (violet
    pill) + PIT status badge (emerald/rose, large: 24px tall)
  - **Best Model**: Name in `heading-3` `--text-violet`, BMA weight with
    gradient bar (80px, showing relative magnitude), selection reason in
    `--text-secondary`
  - **Competing Models Table**: sorted by BMA weight desc. Columns: Model Name,
    BIC, CRPS, Hyv, PIT p-value, BMA Weight %, Nu.
    - Winner row: `rgba(52,211,153,0.06)` background + emerald left border (2px)
    - All cells use `mono` typography with conditional formatting:
      - CRPS: emerald < 0.02, amber 0.02-0.03, rose > 0.03
      - Hyv: emerald < 500, amber 500-1000, rose > 1000
      - PIT p: emerald >= 0.05, rose < 0.05
      - BMA Weight: violet gradient bar behind the number (width proportional)
  - **Kalman State**: 2x2 grid of labeled values. Labels in `caption`,
    `--text-secondary`. Values in `mono`, `--text-luminous`. Subtle glass
    dividers between items.
  - **Regime**: regime name as colored badge (blue/amber/gray/orange/rose
    matching the 5 regime types), volatility state text.
  - **Calibration History** (if available): 5 dots on a horizontal line
    (timeline), each emerald or rose, showing PIT status over last 5 retunes
- [ ] AC-2: Numeric formatting uses semantic color with subtle background:
  numbers that are "good" have a faint emerald background pill,
  "bad" numbers have a faint rose background pill. This creates an instant
  visual health assessment across the entire table without reading values.
- [ ] AC-3: "View in Diagnostics" link: `--accent-violet` with arrow, navigates
  to Diagnostics page with the asset pre-selected.
- [ ] AC-4: A "Compare" button (violet ghost button) allows selecting a second
  asset. When two are selected, the panel splits vertically (50/50) showing
  both side-by-side with differences highlighted (background pulse on values
  that differ by more than 20%).
- [ ] AC-5: The panel animates in with slide-from-right (250ms, `standard` motion).
  Sections stagger in: 60ms delay each. The entire panel feels like a cockpit
  instrument unfolding to reveal deeper data.

---

# EPIC 7: Diagnostics Page -- The Cosmic Calibration Laboratory

> **Vision**: The Diagnostics page is a laboratory. The aesthetic shifts from
> dark observatory to something with hints of cyan and violet -- the colors
> of precision instruments and analytical light. Every chart, every matrix,
> every table answers one question: "Can we trust the model's outputs?"
>
> The ambient glow on this page uses cyan tint rather than pure violet,
> creating a distinct "laboratory" atmosphere that the user associates with
> scientific rigor.

---

## Story 7.1: PIT Calibration Dashboard with Reliability Nebula Diagram

**As a** quant engineer validating model calibration,
**I want** a reliability diagram alongside the PIT summary table,
**so that** I can visually assess calibration quality and identify
where the model is overconfident or underconfident.

### Visual Design

The reliability diagram plots predicted vs. observed probability on a void
canvas. The diagonal "perfect calibration" line is a faint dashed violet
thread -- the ideal. Actual calibration dots glow with their quality:
emerald dots close to the diagonal, rose dots far from it. A translucent
cyan confidence band surrounds the diagonal, and dots inside the band
glow emerald while dots outside glow rose with warning halos.

### Acceptance Criteria

- [ ] AC-1: The PIT Calibration tab shows a **reliability diagram** (cosmic glass
  card, elevation 1, `--void` chart background) at the top spanning half-width:
  - X-axis: Predicted probability (0% to 100%, 10 bins). Axis labels in
    `caption`, `--text-muted`. Axis line: `rgba(139,92,246,0.08)`.
  - Y-axis: Observed frequency (same styling)
  - Perfect calibration line: diagonal dashed line (1px, `--accent-violet` at
    30% opacity) -- the "north star" of calibration
  - Actual calibration: connected dots (8px circles, stroke 2px, fill with glow).
    Color determined by distance from diagonal:
    - Close (< 5% deviation): `--accent-emerald` with glow
    - Medium (5-10%): `--accent-amber`
    - Far (> 10%): `--accent-rose` with warning glow
    (`box-shadow: 0 0 8px rgba(accent, 0.3)`)
    Dot size proportional to observation count in that bin (8px to 16px).
  - Confidence band: translucent `rgba(34,211,238,0.06)` shaded region around
    the diagonal (95% CI for well-calibrated model). Band bounds as thin
    dashed cyan lines.
- [ ] AC-2: Hovering a dot shows cosmic glass tooltip: predicted probability
  range, observed frequency, observation count, deviation from ideal, and a
  diagnostic sentence: "4.2% overconfident in the 40-50% range" in `body`
  typography.
- [ ] AC-3: A **summary metric card** (cosmic glass, elevation 1) beside the
  diagram shows:
  - ECE (Expected Calibration Error): `heading-2` size, gradient text.
    Badge: emerald (< 0.03), amber (0.03-0.05), rose (> 0.05).
  - MCE (Maximum Calibration Error): same format.
  - Overall assessment: "Well calibrated" (emerald text with subtle emerald glow)
    or "Overconfident in the 40-60% range" (amber/rose text).
- [ ] AC-4: The PIT summary table below includes a **mini reliability indicator**
  per asset: 3 dots (6px each) in a row representing calibration quality at
  low/mid/high probability ranges (emerald/amber/rose per dot).
- [ ] AC-5: Expandable asset rows use smooth height transition (200ms) to reveal
  per-model detail tables with the same conditional formatting as Story 6.4.
- [ ] AC-6: Clicking a bin dot on the reliability diagram filters the table to
  show only assets whose predictions fell in that probability range. The
  clicked dot pulses violet (200ms) as confirmation.
- [ ] AC-7: The PIT dashboard -- the reliability nebula with its emerald/rose
  calibration dots, the confidence band, the diagnostic tooltips -- combines
  statistical rigor with visual beauty so effectively that users would
  absolutely fall in love with understanding model trustworthiness and drool
  over the reliability diagram's fusion of science and aesthetics.

---

## Story 7.2: BMA Weight Distribution Matrix with Regime Phase Portrait

**As a** quant researcher understanding model selection across regimes,
**I want** a heatmap showing which models win in which regimes alongside
a visual diagram of regime transitions,
**so that** I can understand the engine's adaptive model selection behavior.

### Visual Design

The BMA matrix is a dense heatmap where rows are models and columns are
regimes. Each cell's intensity maps to average BMA weight -- creating a
cosmic heat signature that shows where each model thrives. The regime
transition diagram beside it uses a Sankey-style flow visualization
with cosmic-colored bands flowing between regimes.

### Acceptance Criteria

- [ ] AC-1: A **BMA heatmap** (cosmic glass card, elevation 1) renders with
  rows = model families (grouped by type), columns = 5 regimes:
  - Cell color: violet-to-cyan gradient intensity where opacity maps to
    average BMA weight for that model-regime pair:
    0% weight = `rgba(0,0,0,0)` (transparent), 100% weight =
    `rgba(139,92,246,0.6)` (deep violet with cyan tint)
  - Cell border: 1px `--border-void` (creates grid lines)
  - Each cell contains the average BMA weight as tiny `mono` text (8px)
  - Winner per column: bold border ring (2px `--accent-emerald`) + emerald
    tint background
  - Row headers: model names in `caption`, `--text-luminous`, left-aligned
    with 120px fixed width
  - Column headers: regime names in `caption`, `--text-violet`, center-aligned
- [ ] AC-2: Hovering a heatmap cell highlights the entire row + column
  (cross-highlight in `rgba(139,92,246,0.04)`) and shows cosmic glass tooltip:
  model name, regime name, average BMA weight (large, violet gradient text),
  sample count, best asset for this combination.
- [ ] AC-3: A **regime transition diagram** (200px wide, beside the heatmap)
  shows flows between regimes using a Sankey-style visualization:
  - Left column: "From" regimes as colored nodes (36px tall bars)
  - Right column: "To" regimes as colored nodes
  - Flowing bands between them with thickness proportional to transition frequency
  - Band color: blend of source and destination regime colors at 20% opacity
  Regime colors: LOW_VOL_TREND (emerald), HIGH_VOL_TREND (amber),
  LOW_VOL_RANGE (violet), HIGH_VOL_RANGE (orange), CRISIS_JUMP (rose)
- [ ] AC-4: Regime columns in the heatmap header are colored dots (8px) using
  the regime colors defined above.
- [ ] AC-5: The matrix and transition diagram animate in with a coordinated
  sequence: heatmap cells fill left-to-right in column waves (40ms per column),
  then the Sankey bands draw themselves (200ms, `standard` motion).

---

## Story 7.3: PIT Histogram Overlay with Cosmic Uniformity Band

**As a** quant calibration specialist inspecting distributional correctness,
**I want** to see PIT histograms with a clear visual reference for perfect
uniformity and a diagnostic for deviation patterns,
**so that** I can assess whether models produce well-calibrated probability
integral transforms.

### Visual Design

The PIT histogram shows bars that should all be the same height (uniform).
The "perfect" reference is a horizontal cyan line. Bars that tower above it
glow rose (overconfident), bars that fall short glow amber (underconfident).
The shape of the histogram tells the calibration story: U-shaped means
heavy tails underestimated, dome means tails overestimated.

### Acceptance Criteria

- [ ] AC-1: A **PIT histogram** (cosmic glass card, `--void` chart background)
  renders for the selected asset:
  - 10 bins (0-0.1, 0.1-0.2, ..., 0.9-1.0). Bars fill from bottom.
  - Uniform reference line: horizontal line at expected count, dashed 1px,
    `--accent-cyan` at 50% opacity, full width.
  - Bar color scheme (deviation from uniformity):
    - Within 1 std: `--accent-violet` at 60% opacity (acceptable)
    - Above 1 std (overrepresented): `linear-gradient(180deg,
      var(--accent-violet) 0%, var(--accent-rose) 100%)` -- violet at bottom
      fading to rose at top
    - Below 1 std (underrepresented): `--accent-amber` at 50%
    Bar border: 1px `--border-void`, `border-radius: 4px 4px 0 0`
  - Y-axis label: "Count" in `caption`, `--text-muted`
  - X-axis label: "PIT Value" in `caption`, `--text-muted`
- [ ] AC-2: A **diagnostic annotation** appears as text overlay on the chart:
  - U-shaped: "Heavy tails underestimated" (rose text, positioned top-center)
  - Dome-shaped: "Tails overestimated" (amber text)
  - Uniform: "Well calibrated" (emerald text with subtle glow)
  - Left-skewed: "Overconfident in upside" (amber text)
  - Right-skewed: "Overconfident in downside" (amber text)
- [ ] AC-3: An **overlay toggle** (violet pill) shows the KDE smooth curve over
  the histogram: 1.5px emerald line with glow, and the uniform reference as
  a flat cyan line. This makes shape deviation immediately visible.
- [ ] AC-4: **Multi-asset comparison**: up to 4 PIT histograms can display in a
  2x2 grid. Each has the asset ticker as a header label. An asset selector
  dropdown allows adding/removing comparison panels.
- [ ] AC-5: Hovering a bar shows cosmic glass tooltip: bin range, count, expected
  count, deviation (% above/below uniform), p-value for that bin.

---

## Story 7.4: Score Comparison Radar with Cosmic Polygon Webs

**As a** quant researcher comparing model performance across multiple metrics,
**I want** a radar (spider) chart that shows how models compare across BIC, CRPS,
Hyvarinen, PIT, and BMA weight simultaneously,
**so that** I can identify models that are strong across all dimensions versus
those that are spiky (strong in one, weak in others).

### Visual Design

The radar chart is a web of concentric cosmic pentagons against the void.
Each model's performance polygon uses a different cosmic color with fill
at low opacity, creating overlapping stained-glass shapes. The visual
metaphor is a jewel -- a perfectly cut gem has a balanced polygon while an
unbalanced model creates a spiky, asymmetric shape.

### Acceptance Criteria

- [ ] AC-1: A **radar chart** (cosmic glass card, `--void` chart background,
  300px minimum size) renders 5 axes: BIC, CRPS, Hyvarinen, PIT, BMA Weight.
  - Concentric reference pentagons (3 levels): 1px stroke, `--border-void`
    at 30% opacity, creating a web pattern
  - Axis lines: 1px, `--border-void` at 20% opacity, extending from center
    to edge
  - Axis labels: metric names in `caption`, `--text-violet`, positioned
    outside each vertex
- [ ] AC-2: Each model overlays as a colored polygon (1.5px stroke, filled at
  10% opacity). Color assignment from cosmic palette:
  - Model 1: `--accent-violet` (rgba(139,92,246,...))
  - Model 2: `--accent-cyan` (rgba(34,211,238,...))
  - Model 3: `--accent-emerald` (rgba(52,211,153,...))
  - Model 4: `--accent-rose` (rgba(251,113,133,...))
  - Model 5: `--accent-amber` (rgba(251,191,36,...))
  Vertices on each axis: filled dots (6px) with glow matching polygon color.
- [ ] AC-3: Hovering over any polygon highlights it (stroke: 2.5px, fill: 20%
  opacity) and dims all others (stroke: 1px, fill: 5%). A cosmic glass
  tooltip shows: model name, all 5 metric values, and an overall "balance
  score" (0-100) measuring how uniform the polygon is.
- [ ] AC-4: The currently "best" model (selected by the engine) has its polygon
  stroke rendered with a gradient and a subtle pulse animation (opacity
  80-100% on 3s cycle) -- the cosmic heartbeat of the winning model.
- [ ] AC-5: A legend below the chart shows colored dots + model names. Clicking
  a legend item toggles that polygon on/off with fade animation (200ms).
- [ ] AC-6: An asset selector dropdown (cosmic glass, violet accent) allows
  switching between assets. The radar polygons morph (each vertex animates
  from old position to new over 400ms with `standard` motion) rather than
  redrawing -- creating a mesmerizing shape-shifting effect.

---

## Story 7.5: Diagnostics Export with Cosmic Report Builder

**As a** quant engineer sharing diagnostic results with stakeholders,
**I want** export capabilities that produce beautiful, shareable reports,
**so that** calibration results can be reviewed outside the application.

### Visual Design

The export panel is a minimal cosmic glass popover with format options as
icon-labeled pills. The JSON preview uses syntax highlighting in cosmic
colors (keys in violet, values in cyan, strings in emerald). The CSV
preview shows a mini table with the same grid styling as the app.

### Acceptance Criteria

- [ ] AC-1: An "Export" button (SVG download icon, `--accent-violet` ghost button)
  in the diagnostics toolbar opens a **cosmic glass popover** (elevation 3,
  max-width 320px):
  - Format options as selectable cards (80px each):
    - **JSON**: `{ }` icon, "Full diagnostics data" description
    - **CSV**: table icon, "PIT summary table" description
    - **PNG**: image icon, "Chart snapshot" description
  - Active format: violet border ring with glow, `--accent-violet` text
  - Export button: gradient pill (`--accent-violet` to `--accent-indigo`)
- [ ] AC-2: JSON export produces a well-structured file with all diagnostic
  data for the current view. Download triggers via blob URL.
- [ ] AC-3: CSV export includes: Asset, Model, BIC, CRPS, Hyvarinen, PIT Status,
  BMA Weight, Regime, and all available score columns. Header row uses
  human-readable names.
- [ ] AC-4: PNG export captures the currently visible chart (reliability diagram,
  PIT histogram, or radar chart) as a high-resolution PNG with the cosmic
  void background preserved. The exported image includes the asset ticker
  and date as a watermark in the bottom-left (`caption`, `--text-muted`
  at 30% opacity).
- [ ] AC-5: While export is being prepared, the export button shows a small
  spinning indicator (8px, `--accent-violet`) replacing the download icon.
  On completion, a brief emerald checkmark flash (300ms) confirms success.

---

# EPIC 8: Data Management & Services -- The Cosmic Infrastructure

> **Vision**: Data and Services pages are the infrastructure layer -- less glamour,
> more reliability. The aesthetic is clean and utilitarian but still beautifully
> crafted: think a well-organized server room where every cable is perfect and
> every status LED is precisely placed. The cosmic theme is muted here: more
> void, less gradient, but still unmistakably part of the same universe.

---

## Story 8.1: Data Freshness Dashboard with Cache Topology Map

**As a** system operator ensuring data quality,
**I want** a visual map of data freshness across all symbols with stale
data highlighted prominently,
**so that** I can identify data gaps before they affect model quality.

### Visual Design

The freshness map is a dense grid of tiny status indicators -- green circles
for fresh, amber for aging, rose for stale, gray for missing. The grid is so
dense (100+ items) that it creates a visual texture where the health of the
data pipeline is immediately apparent: mostly green = healthy, clusters of
red = trouble.

### Acceptance Criteria

- [ ] AC-1: A **freshness grid** displays all cached symbols as small squares
  (16px x 16px, border-radius: 3px, 2px gap) in a flowing grid:
  - Fresh (< 24h): `--accent-emerald` at 50% opacity
  - Aging (24-72h): `--accent-amber` at 50% opacity
  - Stale (> 72h): `--accent-rose` at 70% with pulse (1.5s cycle)
  - Missing: `rgba(255,255,255,0.05)` with dashed 1px border
  Grid is sorted by staleness (most stale first). Hovering a square shows
  cosmic glass tooltip: symbol, last refreshed time (relative), row count,
  data range (earliest to latest date).
- [ ] AC-2: A **summary bar** (same as Story 6.3): stacked gradient bar showing
  fresh/aging/stale/missing proportions with count labels.
- [ ] AC-3: A "Refresh Stale" button (rose-accented ghost button) triggers a
  data refresh for all stale symbols. Progress shows as a thin gradient
  progress bar (violet-to-cyan) above the grid.
- [ ] AC-4: A **cache size indicator** (cosmic glass card, 120px): total cache
  size in MB, file count, and a mini arc gauge showing usage vs. limit.
- [ ] AC-5: Clicking a symbol square opens a detail popover with: last 5
  data points (date, close price) as a mini table in `mono`, a "Delete
  cache" button (rose, needs confirmation click), and a "Refresh" button
  (violet pill).

---

## Story 8.2: Service Health Tiles with Heartbeat Pulse Lines

**As a** system operator monitoring dependent services,
**I want** each service displayed as a health tile with a heartbeat indicator,
**so that** I can see at a glance whether the backend, cache, and data
pipelines are operational.

### Visual Design

Each service is a cosmic glass tile with a small heartbeat line (EKG-style
sparkline) that shows recent response times. Healthy services have calm,
regular heartbeats in emerald. Degraded services have erratic heartbeats in
amber. Dead services show a flatline in rose. The tiles create a miniature
hospital monitoring station aesthetic.

### Acceptance Criteria

- [ ] AC-1: Each service displays as a **cosmic glass tile** (cosmic glass,
  elevation 1, 160px x 120px):
  - Service name in `heading-3`, `--text-luminous`
  - Status badge: colored glass pill (emerald "Healthy" / amber "Degraded" /
    rose "Down")
  - **Heartbeat line**: 100px wide, 24px tall, showing last 10 response times
    as a connected line chart:
    - Healthy: emerald line, calm waves
    - Degraded: amber line, erratic spikes
    - Down: rose flatline with a small cross marker
    Line glow: `filter: drop-shadow(0 0 2px rgba(color, 0.4))`
  - Response time: last value in `mono`, colored
  - Uptime: "99.8%" in `caption`, `--text-secondary`
- [ ] AC-2: Services monitored: Backend API, Yahoo Finance API, Cache Layer,
  WebSocket Connection. Each tile auto-refreshes every 30 seconds via the
  existing health endpoint.
- [ ] AC-3: When a service status changes (e.g., healthy -> degraded), the tile's
  border flashes the status color (200ms) and a cosmic toast notification
  fires (Story 1.4) with the service name and new status.
- [ ] AC-4: Clicking a service tile expands it inline (250ms height animation)
  to show: last 10 health check results as a mini table (timestamp, status,
  response time), all in `mono` typography with colored rows.
- [ ] AC-5: Tiles arrange in a responsive row (flex-wrap). Dead services get
  elevated z-index and rose glow, ensuring they visually dominate.

---

## Story 8.3: Bulk Operations Panel with Cosmic Batch Controls

**As a** system operator performing bulk data operations,
**I want** batch controls with clear confirmation flows and progress tracking,
**so that** I can manage 100+ assets without repetitive individual actions.

### Visual Design

The bulk operations panel is a command bar with action pills and a target
selector. The confirmation flow uses a two-step pattern: first click reveals
the "Are you sure?" state with a visual countdown. This prevents accidents
while remaining fast for intentional bulk operations.

### Acceptance Criteria

- [ ] AC-1: A **bulk operations bar** (cosmic glass, 64px tall, full width) includes:
  - Target selector: "All symbols" / "Stale only" / "Failed only" / "Custom"
    as segmented violet pills (same pattern as Story 6.1)
  - Action buttons as colored ghost pills:
    - "Refresh Data" (violet)
    - "Clear Cache" (amber, destructive)
    - "Purge Failed" (rose, destructive)
  - Selected count badge: "147 symbols" in `mono`, violet glass pill
- [ ] AC-2: Destructive actions (Clear Cache, Purge Failed) use a **two-step
  confirmation**: first click transforms the button text to "Confirm? (3s)"
  with a countdown timer. The button background fills with the action color
  (amber/rose) over 3 seconds as a visual countdown bar. Clicking again within
  the countdown confirms. After 3 seconds without confirmation, it resets.
- [ ] AC-3: Running a bulk operation shows:
  - Progress bar: thin gradient strip (same pattern as Story 6.1)
  - Current item: ticker in `mono`, `--accent-violet`
  - Success/fail counter: emerald + rose numbers updating in real-time
  - A cancel button appears during operation (rose ghost pill)
- [ ] AC-4: On completion, a summary card (same pattern as Story 6.1 AC-4):
  total processed, succeeded (emerald), failed (rose, clickable to show list),
  duration.
- [ ] AC-5: Custom target selection opens a searchable checkbox list (cosmic glass
  popover, max-height 300px, scrollable): all symbols with checkboxes. "Select
  all" / "Deselect all" links at top. Search input with violet focus glow.

---

# EPIC 9: Arena Page -- The Cosmic Proving Ground

> **Vision**: The Arena is a gladiatorial proving ground for experimental models.
> The aesthetic is darker, grittier, more competitive -- like a cosmic colosseum
> where models compete for the right to be promoted. Score cards are battle
> results. The leaderboard is a ranking of champions. The mood is intense but
> beautiful: think a dark esports arena with neon violet accents.

---

## Story 9.1: Arena Leaderboard with Champion Glow Effects

**As a** quant researcher reviewing model competition results,
**I want** a leaderboard that ranks experimental models against standard baselines
with visual distinction between champions and underperformers,
**so that** I can identify which experimental approaches are genuinely superior.

### Visual Design

The leaderboard is a vertical ranking list where the #1 model's card is
visibly superior to all others: larger, glowing, with a cosmic gradient
background. As you scroll down the list, cards progressively dim and shrink,
creating a natural visual hierarchy from champion to contender to also-ran.

### Acceptance Criteria

- [ ] AC-1: The Arena leaderboard displays models in a **ranked card list**:
  - **Champion card** (#1): elevated (elevation 3), 80px tall, full width.
    Background: `linear-gradient(135deg, rgba(139,92,246,0.08) 0%,
    rgba(168,85,247,0.04) 50%, rgba(99,102,241,0.06) 100%)` -- subtle cosmic
    aurora. Gold gradient medal badge (32px). Model name in `heading-2`,
    `--text-luminous`. Final score in `display` typography with violet gradient
    text. Glow: `box-shadow: 0 0 30px rgba(139,92,246,0.06)`.
  - **Contender cards** (#2-#5): elevation 1, 64px tall. Silver/Bronze badges
    for #2/#3. Model name in `heading-3`. Score in `heading-2`.
  - **Also-ran cards** (#6+): elevation 0, 56px tall, subtly dimmed
    (`opacity: 0.85`). No medal. Score in `heading-3`, `--text-secondary`.
  - All cards: cosmic glass background, 1px `--border-void` border
- [ ] AC-2: Each leaderboard card shows:
  - Final score (large, colored: violet if beats standard, rose if below)
  - Score delta vs. best standard: "+10.7" in emerald or "-3.2" in rose,
    with arrow icon
  - Score breakdown as mini colored dots: BIC (blue), CRPS (violet), Hyv (cyan),
    PIT (emerald/rose), CSS (amber), FEC (indigo)
  - Hard gate status: a row of tiny gate badges (12px pills):
    CSS >= 0.65: emerald if pass, rose if fail
    FEC >= 0.75: same
    Hyv < 1000: same
    PIT >= 75%: same
    vs STD >= 3: same
- [ ] AC-3: Models are tagged as "Experimental" (violet pill) or "Standard"
  (gray pill). A toggle (violet pill button) filters between: All, Experimental
  Only, Standard Only.
- [ ] AC-4: Clicking a leaderboard card expands it (250ms, smooth height) to show
  a full score breakdown table: all scoring metrics in a formatted grid with
  conditional coloring (same rules as Story 6.4).
- [ ] AC-5: The champion card has a subtle continuous animation: the gradient
  background slowly shifts hue (30-degree rotation over 8 seconds, infinite
  loop) -- a living, breathing glow that signals "this is the best model."
- [ ] AC-6: Cards animate in with stagger cascade from #1 to last, 60ms delay
  each, sliding up 12px and fading in with `standard` motion.

---

## Story 9.2: Score Radar Comparison with Cosmic Polygon Duel

**As a** quant researcher comparing experimental vs. standard models visually,
**I want** a radar chart that overlays the best experimental model against the
best standard model,
**so that** I can see exactly which dimensions the experimental model wins or
loses on.

### Visual Design

Two overlapping polygons on the radar: one in violet (experimental) and one
in cyan (standard). Where the violet polygon exceeds cyan, the overlap region
fills with a "victory" emerald tint. Where cyan exceeds violet, the gap fills
with rose. This creates an instant visual battlefield showing where each
model wins.

### Acceptance Criteria

- [ ] AC-1: A **duel radar chart** (cosmic glass card, 300px, `--void` background)
  renders 7 axes: BIC, CRPS, Hyv, PIT, CSS, FEC, DIG.
  - Standard model polygon: `--accent-cyan` stroke (2px), fill at 8% opacity
  - Experimental model polygon: `--accent-violet` stroke (2px), fill at 8% opacity
  - Victory regions (where experimental exceeds standard): filled with
    `rgba(52,211,153,0.08)` (emerald tint)
  - Deficit regions (where standard exceeds experimental): filled with
    `rgba(251,113,133,0.08)` (rose tint)
  - Reference web: 3 concentric heptagons in `--border-void` at 20% opacity
- [ ] AC-2: Axis labels show the delta value: "+2.3" in emerald where experimental
  wins, "-1.1" in rose where it loses, positioned outside each vertex in
  `mono` typography.
- [ ] AC-3: Below the radar, a **head-to-head table** lists all 7 metrics with
  columns: Metric | Standard | Experimental | Delta | Winner. Winner column
  shows a colored arrow (emerald for experimental win, rose for standard win).
  Winner cell has faint colored background (emerald/rose at 5% opacity).
- [ ] AC-4: A model selector dropdown (cosmic glass, violet accent) allows
  choosing any experimental model to compare against the best standard.
  Changing model morphs the experimental polygon (vertices animate 400ms,
  `standard` motion -- the shape flows from one model to another).
- [ ] AC-5: The radar chart legend: two colored dots + labels at the bottom.
  Hovering a legend item brightens that polygon and dims the other.

---

## Story 9.3: Safe Storage Gallery with Cosmic Trophy Case

**As a** quant researcher reviewing historically graduated (promoted) models,
**I want** safe storage displayed as a trophy case with full performance profiles,
**so that** I can reference what has been proven to work.

### Visual Design

The safe storage gallery is a trophy case: cards arranged in a showcase grid,
each one a cosmic glass display case with the model's "championship stats"
prominently displayed. The best-ever model gets a special holographic frame
treatment -- a rainbow-iridescent border that shifts color on hover.

### Acceptance Criteria

- [ ] AC-1: Safe storage models display in a **trophy grid** (CSS grid, 3 columns
  desktop, 2 tablet, 1 mobile):
  - Each card (cosmic glass, elevation 2, 200px tall, border-radius: 16px):
    - Model name in `heading-2`, `--text-luminous`
    - Label: "Generation 18" (or appropriate gen) in `caption`, `--text-violet`
    - Final score: `display` typography, violet gradient text
    - vs STD delta: emerald badge "+10.7 vs standard"
    - Key metrics as a 2x3 grid of labeled values: CSS, FEC, BIC, CRPS, Hyv, PIT
      Each in a mini glass cell (48px, `mono` text, conditional emerald/rose color)
    - Tags: applicable technique tags as tiny gradient pills (e.g., "Q-shift",
      "DTCWT", "Hybrid") in `caption` typography. Each pill uses violet gradient
      background at 15% opacity.
  - Cards sorted by Final Score descending.
- [ ] AC-2: The **best-ever model** card (highest Final Score) gets special
  treatment: border ring uses `conic-gradient(from 0deg, var(--accent-violet),
  var(--accent-cyan), var(--accent-emerald), var(--accent-amber),
  var(--accent-rose), var(--accent-violet))` creating an iridescent holographic
  border (2px). On hover, the gradient rotates (animation: 3s linear infinite).
  A small "BEST" badge (gold gradient) appears top-right.
- [ ] AC-3: Hovering a card lifts it (translate-y: -4px, elevation increase,
  200ms `standard` motion) and shows a cosmic glass tooltip with the full
  description of the model's mathematical technique in `body` typography.
- [ ] AC-4: Clicking "Load into Arena" (violet pill button, visible on hover)
  copies the safe storage model into the experimental models directory for
  re-evaluation. Confirmation: toast notification with model name.
- [ ] AC-5: An "Archive" view toggle (violet pill) switches to a compact table
  list for users who prefer scanning many models quickly. Same columns as
  the Arena leaderboard with conditional coloring.
- [ ] AC-6: The trophy case gallery -- the holographic best-ever border, the
  showcase grid of champion models, the technique tags, the score profiles --
  makes safe storage feel like a hall of fame rather than a storage folder,
  causing users to absolutely fall in love with visiting the gallery and
  drool over the competitive legacy of their model research program.

---

# EPIC 10: Micro-Interactions & Loading States -- The Cosmic Polish

> **Vision**: This is the Cosmic Polish epic. Every loading state, every skeleton,
> every error, every empty state must feel like it belongs in the cosmic universe.
> Loading must never feel like waiting -- it must feel like the universe is
> materializing. Errors must not feel like failures -- they must feel like
> the system communicating gracefully. Empty states must not feel empty --
> they must feel like invitations.

---

## Story 10.1: Skeleton Screens with Cosmic Shimmer Animation

**As a** user waiting for data to load,
**I want** skeleton screens that match the exact layout of loaded content with
a beautiful shimmer that makes waiting feel premium,
**so that** loading never causes layout shift and the wait feels intentional.

### Visual Design

Skeleton screens use the exact same card shapes, grid layouts, and spacing
as loaded content. The shimmer is a diagonal cosmic sweep: a gradient
highlight band that sweeps across the skeleton from left to right, using
violet-to-transparent gradients against the void. The effect is like light
from a distant nebula slowly illuminating the interface as it materializes.

### Acceptance Criteria

- [ ] AC-1: Every data-driven section has a matching **skeleton variant** that
  renders on first load and data refetches (when `isLoading && !data`):
  - Dashboard: 2x2 card grid with skeleton cards matching exact card heights
  - Signal Table: skeleton rows (10) matching column widths, row heights
  - Charts: skeleton chart area (correct aspect ratio) + skeleton sidebar
  - Risk: skeleton gauge (circle) + skeleton category cards
  - Tuning: skeleton treemap + skeleton health grid
  - Diagnostics: skeleton chart + skeleton summary table
  Skeletons use `--void-surface` background (border-radius matching real cards).
- [ ] AC-2: The **cosmic shimmer**: a diagonal highlight band sweeps across all
  skeleton elements simultaneously (coordinated, not per-element):
  ```css
  background: linear-gradient(
    105deg,
    transparent 40%,
    rgba(139, 92, 246, 0.04) 45%,
    rgba(139, 92, 246, 0.08) 50%,
    rgba(139, 92, 246, 0.04) 55%,
    transparent 60%
  );
  background-size: 200% 100%;
  animation: cosmic-shimmer 2s ease-in-out infinite;
  ```
  `@keyframes cosmic-shimmer { from { background-position: 200% 0 } to { background-position: -200% 0 } }`
- [ ] AC-3: Skeleton to content transition: skeletons fade out while content fades
  in with a 200ms crossfade. Content sections stagger in by 40ms each. No
  layout shift: skeleton dimensions match content dimensions exactly.
- [ ] AC-4: Text skeletons use widths that approximate real content: headings are
  60% width, body text lines alternate 80% and 70%, numbers are 40px fixed.
- [ ] AC-5: Skeleton screens are so beautiful -- the coordinated cosmic shimmer
  sweep, the exact layout matching, the staggered materialization -- that users
  would prefer watching the loading animation to its completion and drool over
  the way the interface appears to materialize from the cosmic void.

---

## Story 10.2: Empty States with Cosmic Invitation Illustrations

**As a** new user encountering a page with no data,
**I want** empty states that guide me toward the action needed rather than
showing a blank page,
**so that** I know exactly what to do next and the experience feels inviting.

### Visual Design

Empty states are not sad error pages. They are invitations. Each empty state
features a cosmic SVG illustration (abstract, geometric, matching the page
context), a warm message, and a clear action button. The illustration uses
the cosmic gradient palette, creating a mini art piece that makes the empty
page almost more beautiful than the loaded page.

### Acceptance Criteria

- [ ] AC-1: Each page has a **custom empty state** (centered, max-width 400px):
  - **Dashboard**: cosmic circle SVG (concentric violet rings) + "No signals
    yet. Run your first tune to see the universe come alive." + "Start Tune"
    violet gradient button.
  - **Signals**: cosmic table SVG (column lines with floating dots) + "No signals
    generated. Start tuning to populate the signal table." + dual buttons:
    "Start Tune" (violet primary) + "Refresh Data" (ghost).
  - **Charts**: cosmic waveform SVG (sine wave with violet gradient fill) +
    "Select an asset to begin charting." + asset quick-picks as violet pills
    (SPY, AAPL, NVDA, TSLA).
  - **Risk**: cosmic gauge SVG (empty arc with faint gradient) + "No risk data
    available. Signals must be generated first." + "Generate Signals" button.
  - **Arena**: cosmic trophy SVG (empty pedestal with star sparkles) + "No arena
    results yet. Run an arena competition." + "Start Arena" button.
- [ ] AC-2: All illustrations use the cosmic palette: `--accent-violet`,
  `--accent-indigo` strokes and fills at 20-40% opacity. Lines are 1-2px.
  Total SVG size: 80px x 80px. SVGs animate subtly on loop (gentle float:
  translate-y 0 to -4px over 3s, ease-in-out, infinite).
- [ ] AC-3: Message text: `heading-3` typography for the main line, `body`
  typography for the description, `--text-secondary`. Center-aligned.
- [ ] AC-4: The action button uses the same gradient styling as primary buttons
  throughout the app (violet-to-indigo gradient, white text, 40px tall).
- [ ] AC-5: Empty states transition to loaded content with a coordinated sequence:
  the illustration scales down and fades (200ms), then content sections
  stagger in from below (40ms offset each).

---

## Story 10.3: Error States with Cosmic Recovery Guidance

**As a** user encountering an API error or data failure,
**I want** error messages that are clear, non-technical, and provide a recovery
path with retry capability,
**so that** errors feel temporary and recoverable rather than catastrophic.

### Visual Design

Error states use rose accents but remain calm -- not screaming red alerts.
A faint rose glow backgrounds the error card, the icon is a gentle warning
shape (not a harsh exclamation), and the retry button is prominent and
inviting. The error feels like the system saying "something went wrong, but
here's what you can do" rather than "FAILURE."

### Acceptance Criteria

- [ ] AC-1: API errors display as an **error card** (cosmic glass, elevation 1,
  with a subtle rose radial glow: `radial-gradient(ellipse at 50% 50%,
  rgba(251,113,133,0.04) 0%, transparent 60%)`):
  - Warning icon: SVG circle with line (24px, `--accent-rose`)
  - Error title: `heading-3`, `--text-luminous`: "Unable to load signals" (not
    "Error 500" or technical language)
  - Description: `body`, `--text-secondary`: "The backend may be restarting.
    This usually resolves within seconds."
  - Retry button: rose-accented gradient pill ("Try Again") + secondary ghost
    button ("View Status")
  - Technical detail (collapsible): `mono`, `caption`, `--text-muted`. Shows
    HTTP status, endpoint, timestamp. Hidden by default, expandable via
    "Show details" link.
- [ ] AC-2: Retry triggers a refetch with exponential backoff visual: the retry
  button shows a spinning indicator (8px rose circle) and "Retrying in 3s..."
  countdown text. Successive retries: 1s, 3s, 8s. After 3 retries, button
  changes to "Manual retry only" and stops auto-retrying.
- [ ] AC-3: Network errors (no connection) show a distinct state with a
  disconnected icon and message: "No connection to the server. Check that the
  backend is running." with the `make web-backend` hint in `mono` text.
- [ ] AC-4: Partial errors (some API calls succeed, others fail) render the
  successful data normally and show a thin error banner (40px tall, rose
  left border) at the top of the section: "Some data unavailable" with a
  retry link for just the failed requests.
- [ ] AC-5: All error states include a gentle pulse animation on the glow
  (rose glow opacity oscillating 2-4%, 3s cycle) -- just enough to draw
  attention without being alarming.

---

## Story 10.4: Inline Validation with Cosmic Micro-Feedback

**As a** user entering search queries, filter values, or settings,
**I want** immediate inline validation with beautiful micro-feedback,
**so that** I know my input is valid before submitting.

### Visual Design

Validation feedback appears as color changes and micro-animations on the
input field itself. Valid inputs gain a brief emerald glow pulse at the
border. Invalid inputs gain a rose glow with a gentle shake. The feedback
is subtle enough to not be distracting but clear enough to be unmistakable.

### Acceptance Criteria

- [ ] AC-1: **Search inputs** provide live feedback:
  - Typing: border subtly shifts to `--accent-violet` (focus state)
  - Valid query (matches exist): brief emerald pulse at border (`box-shadow:
    0 0 0 2px rgba(52,211,153,0.3)`, 300ms, then back to violet focus)
  - No matches: brief rose pulse at border (same timing) + "No matches"
    text in `caption`, `--accent-rose`, below the input, fading in 200ms
- [ ] AC-2: **Numeric inputs** (annotation confidence thresholds, export settings):
  - Out-of-range: rose border + "Value must be between X and Y" in rose caption
  - Gentle shake animation: `transform: translateX(-2px, 2px, -1px, 0)` over
    200ms with `micro` timing on invalid submission
- [ ] AC-3: **Filter selections**: selecting a filter that eliminates all results
  shows a warning: "This combination returns 0 results" in amber `caption`
  text below the filter bar. The "0 results" number uses `--accent-amber`.
- [ ] AC-4: All validation messages animate in with a 150ms slide-down + fade.
  Removing the error condition animates the message out (100ms fade).
- [ ] AC-5: Success states are brief and non-intrusive: a 300ms emerald flash
  that self-dismisses. Failure states persist until the user corrects the input.

---

## Story 10.5: Keyboard Shortcut Cheat Sheet with Cosmic Glass Overlay

**As a** power user learning the application's keyboard shortcuts,
**I want** a discoverable cheat sheet that lists all shortcuts in an organized,
beautiful overlay,
**so that** I can learn and reference shortcuts quickly.

### Visual Design

The cheat sheet is a full-screen cosmic glass overlay that dims the background
and presents shortcuts organized in a clean multi-column layout. Each shortcut
is a tiny card with a key badge (styled to look like a physical keycap) and a
description. The overlay is itself navigable by keyboard, creating a
dog-fooding moment for the shortcut system.

### Acceptance Criteria

- [ ] AC-1: Pressing `?` (when no input is focused) opens a **shortcut overlay**
  (full-screen, `rgba(3,0,20,0.8)` backdrop with `blur(8px)`, cosmic glass
  card centered, max-width 680px, max-height 80vh, scrollable):
  - Title: "Keyboard Shortcuts" in `heading-1`, `--text-luminous`
  - Sections organized by context: "Navigation", "Signals", "Charts", "Search",
    "General"
  - Each shortcut row: key badge(s) on the left + description on the right
  - Key badges: styled as physical keycaps (12px border-radius, `--void-active`
    background, 1px `--border-void` border, `mono` 11px text,
    `min-width: 24px`, center-aligned, slight inner shadow for depth).
    Multi-key combos show badges separated by "+" in `--text-muted`.
  - Close: `Esc` key or click outside. Close button top-right as "X" icon.
- [ ] AC-2: The overlay animates in: backdrop fades (200ms), then card scales
  from 0.95 to 1.0 + fades (200ms, `standard` motion). Animate out: reverse.
- [ ] AC-3: A search input at the top (violet focus glow) filters shortcuts
  in real-time (same fuzzy matching as Story 3.5). Non-matching shortcuts
  dim to 20% opacity.
- [ ] AC-4: Shortcut sections are collapsible (smooth height transition 200ms).
  Default: all expanded. Section headers in `heading-3`, `--text-violet`.
- [ ] AC-5: The cheat sheet is accessible from the Command Palette (Story 1.2)
  by typing "shortcuts" or "keys".

---

## Story 10.6: Scroll-Triggered Animations with Cosmic Materialization

**As a** user scrolling through long pages (Signals table, Risk dashboard),
**I want** content to animate into view as it becomes visible, creating a
living document feel,
**so that** the page feels responsive and alive rather than static.

### Visual Design

Elements below the fold start invisible and materialize as they enter the
viewport. The animation is subtle: a combination of fade-in and slight
upward drift. Sections animate in sequence with staggered timing, creating
a cascade effect that makes the page feel like it's unfurling from the void.

### Acceptance Criteria

- [ ] AC-1: The following elements animate on scroll-into-view using
  `IntersectionObserver` (threshold 0.1):
  - Dashboard cards: fade + slide up 12px, `standard` motion (250ms)
  - Signal table sections (when using sector groups): each sector group
    fades in as it enters viewport
  - Risk dashboard tabs: cards within each tab animate when the tab scrolls
  - Diagnostics charts: charts animate their data entrance when visible
- [ ] AC-2: Animation triggers only once per element (add `data-animated="true"`
  after animation). Re-entering the viewport does not re-trigger.
- [ ] AC-3: Elements above the fold (visible on page load) do NOT use scroll
  animation -- they render immediately. Only below-fold content uses this.
- [ ] AC-4: If the user has `prefers-reduced-motion: reduce`, scroll animations
  are disabled. Elements appear immediately without motion.
- [ ] AC-5: Stagger timing: when multiple elements enter the viewport
  simultaneously (e.g., a grid of cards), they stagger by 40ms each,
  creating a cascade rather than a simultaneous pop.
- [ ] AC-6: The scroll materialization must be so perfectly timed -- subtle enough
  to not be gimmicky, consistently applied to feel systematic -- that users
  would fall in love with the sense that the page is alive and responding to
  their attention, and Apple engineers would study the intersection observer
  implementation as a reference for scroll-driven progressive disclosure.

---

# EPIC 11: Accessibility & Visual Quality Assurance

> **Vision**: Inclusive design is not a checkbox -- it is a sign of engineering
> maturity. The cosmic aesthetic must not come at the cost of usability. Color
> contrast, keyboard access, screen reader support, and reduced-motion modes
> are non-negotiable. The best interfaces in the world (Apple, Linear, Vercel)
> are also the most accessible.

---

## Story 11.1: WCAG AA Contrast Compliance Across Cosmic Palette

**As a** user with visual impairments or in challenging lighting conditions,
**I want** all text and interactive elements to meet WCAG AA contrast ratios,
**so that** I can read and use the application comfortably.

### Acceptance Criteria

- [ ] AC-1: All text on the three background tiers meets WCAG AA:
  - `--text-luminous` on `--void` (#030014): ratio >= 7:1 (AAA)
  - `--text-primary` on `--void-surface`: ratio >= 4.5:1 (AA)
  - `--text-secondary` on `--void-surface`: ratio >= 4.5:1 (AA)
  - `--text-muted` on `--void-surface`: ratio >= 3:1 (AA for large text only,
    only used in `caption` / `mono` which are decorative supplementary text)
- [ ] AC-2: All colored data elements (emerald, rose, amber, cyan, violet text)
  meet 4.5:1 against their background. Verify with a contrast checker for
  each token value against `--void`, `--void-surface`, and `--void-hover`.
- [ ] AC-3: Interactive elements (buttons, links, toggles) have a visible focus
  indicator: 2px ring in `--accent-violet` with glow (`box-shadow: 0 0 0 2px
  rgba(139,92,246,0.5)`), visible on `:focus-visible` (not `:focus` to avoid
  click-triggered rings).
- [ ] AC-4: Gradient text must have a solid color fallback for screen readers
  and forced-colors mode: `@media (forced-colors: active) { .gradient-text
  { background: none; -webkit-text-fill-color: unset; color: CanvasText; } }`
- [ ] AC-5: No information is conveyed by color alone. All colored indicators
  also have: shape differences (circle vs. triangle), text labels, or
  pattern/icon alternatives. Specifically:
  - Signal badges: color + text (SB/B/H/S/SS)
  - PIT status: color + text ("Pass"/"Fail") + icon
  - Trend arrows: color + direction + aria-label

---

## Story 11.2: Screen Reader Annotations and ARIA Landmarks

**As a** screen reader user navigating the application,
**I want** proper ARIA landmarks, labels, and live regions,
**so that** I can navigate efficiently and receive updates about dynamic content.

### Acceptance Criteria

- [ ] AC-1: The app uses proper landmark roles:
  - `<nav>` for sidebar with `aria-label="Main navigation"`
  - `<main>` for page content
  - `<header>` for the breadcrumb bar
  - `role="complementary"` for detail sidebars
  - `role="search"` for search inputs
- [ ] AC-2: All charts include `aria-label` describing the chart type and data:
  "Temperature gauge showing risk level 0.82 out of 2.0, status elevated."
  Dynamic data values update via `aria-live="polite"` region.
- [ ] AC-3: Tables use `<th scope="col">` and `<th scope="row">` correctly.
  Sort state announced: `aria-sort="ascending"` / `"descending"`.
- [ ] AC-4: The toast notification container is an `aria-live="assertive"`
  region so screen readers announce new toasts immediately.
- [ ] AC-5: Modal overlays (command palette, keyboard shortcuts, popovers) use
  `aria-modal="true"`, `role="dialog"`, and trap focus within the modal.
  `Esc` closes and returns focus to the trigger element.
- [ ] AC-6: All SVG icons have `aria-hidden="true"` when decorative, or a
  descriptive `<title>` element when conveying information.

---

## Story 11.3: Reduced Motion Mode for Motion-Sensitive Users

**As a** user with vestibular disorders or motion sensitivity,
**I want** the application to respect my OS reduced-motion preference,
**so that** animations don't cause discomfort.

### Acceptance Criteria

- [ ] AC-1: When `prefers-reduced-motion: reduce` is active:
  - All CSS transitions reduce to 0ms (instant state changes)
  - The cosmic shimmer animation stops (static placeholder)
  - Spring animations use instant transitions
  - Scroll-triggered animations are disabled (elements visible immediately)
  - Particle effects and animated gradients are disabled
  - Pulsing indicators stop (static color only)
- [ ] AC-2: The needle spring animation on the temperature gauge falls back to
  an instant appearance at the final value (no overshoot).
- [ ] AC-3: Page transitions (route changes) are instant with no crossfade.
- [ ] AC-4: Interactive feedback (hover states, focus rings) still works but
  transitions are instant (0ms duration, not removed entirely).
- [ ] AC-5: A manual toggle in the settings (if/when a settings panel exists)
  allows force-reducing motion regardless of OS setting.

---

## Story 11.4: Responsive Layout Breakpoints with Cosmic Grid Adaptation

**As a** user on a tablet, narrow monitor, or ultra-wide display,
**I want** the layout to adapt intelligently across breakpoints,
**so that** the cosmic aesthetic works on any screen size.

### Acceptance Criteria

- [ ] AC-1: Breakpoint system:
  - **Mobile** (<768px): single column, sidebar collapses to bottom nav,
    charts full-width, tables horizontally scrollable
  - **Tablet** (768-1024px): two-column where appropriate, sidebar as overlay
  - **Desktop** (1024-1440px): full layout with sidebar + content + optional
    detail panels
  - **Ultra-wide** (>1440px): max-content-width 1440px centered, cosmic void
    background extends to screen edges
- [ ] AC-2: The signal table at mobile breakpoint: columns collapse to show
  only: Symbol, Signal Badge, Best Horizon Return. A "Show all columns"
  toggle expands to horizontal scroll mode.
- [ ] AC-3: Dashboard at mobile: cards stack in a single column. The hero card
  remains full-width. The heatmap (Story 2.3) rotates to a vertical list
  with horizontal scrolling for sectors.
- [ ] AC-4: Charts at mobile: full-screen mode removes all chrome (sidebar,
  toolbar) for maximum chart space. A floating "Exit" button (cosmic glass
  pill, top-right) returns to normal mode.
- [ ] AC-5: All cosmic glass cards maintain their visual treatment (gradient
  background, border, elevation) at every breakpoint. No degradation of
  the cosmic aesthetic on smaller screens.

---

## Story 11.5: Print and Share-Ready Page Snapshots

**As a** user sharing analysis results with colleagues,
**I want** print-optimized snapshots of key pages,
**so that** the cosmic beauty is preserved in static output.

### Acceptance Criteria

- [ ] AC-1: A `@media print` stylesheet converts the cosmic theme:
  - Backgrounds: white with light gray card backgrounds (inkjet-friendly)
  - Text: black/dark gray
  - Gradients: replaced with solid colors from the high-contrast end
  - Cosmic glow effects: removed (box-shadows set to `none`)
  - Charts and gauges: preserved with white canvas backgrounds
  - Layout: single column, sidebar hidden, full-width content
- [ ] AC-2: The Dashboard print includes: hero summary, signal distribution
  bar (simplified), top 5 conviction assets table. Fits on one A4 page.
- [ ] AC-3: The Signals print includes the full table (up to 50 rows per page)
  with column widths optimized for A4 landscape.
- [ ] AC-4: Each page has a "Print View" button (hidden in print, visible on
  screen as a ghost button with SVG printer icon). Clicking it opens the
  print dialog via `window.print()`.
- [ ] AC-5: Page header in print: "Quant Signal Engine" + date + page name.
  Page footer: "Generated at [timestamp]" in small gray text.

---

# EPIC 12: Performance & Export -- The Cosmic Engine Under the Hood

> **Vision**: The fastest interface in the world is useless if it feels sluggish.
> The most beautiful interface is useless if the data can't escape it. This
> epic ensures the cosmic experience runs at 60fps and the data flows freely
> to wherever the user needs it.

---

## Story 12.1: React Query Caching with Stale-While-Revalidate Strategy

**As a** user navigating between pages frequently,
**I want** data to be cached intelligently so that pages load instantly on
return visits while still refreshing in the background,
**so that** the application feels instantaneous.

### Acceptance Criteria

- [ ] AC-1: TanStack React Query configuration:
  - `staleTime: 2 * 60 * 1000` (2 minutes -- data is fresh for 2 min)
  - `gcTime: 10 * 60 * 1000` (10 minutes -- cache retained for 10 min)
  - `refetchOnWindowFocus: true` (refresh when tab regains focus)
  - `retry: 2` with exponential backoff
- [ ] AC-2: Page navigation between tabs shows cached data immediately (< 16ms
  render) while background refetch occurs. A subtle refresh indicator (thin
  violet line at the top of the content area, 2px, animating left-to-right)
  shows when a background refresh is in progress. It fades out (200ms) when
  the refetch completes.
- [ ] AC-3: Cache keys include all relevant parameters: endpoint + symbol +
  filters + sort configuration, so different views of the same data are
  cached independently.
- [ ] AC-4: The signals endpoint (heaviest response) is prefetched on hover
  of the Signals nav item (200ms hover delay before prefetch fires). This
  ensures the Signals page loads instantly when clicked.
- [ ] AC-5: Manual refresh button (SVG refresh icon, `--accent-violet` ghost
  button) in the page header forces invalidation of all queries for the
  current page. Icon rotates 360deg during refetch (300ms, `standard` easing).

---

## Story 12.2: Virtual Scrolling for Large Signal Tables

**As a** user with 100+ assets in the signal table,
**I want** the table to render smoothly without jank, even with complex row
content (sparklines, gradient bars, badge pills),
**so that** scrolling always feels as smooth as native.

### Acceptance Criteria

- [ ] AC-1: The signal table uses **virtual scrolling** (TanStack Virtual or
  equivalent) when row count exceeds 50.
  - Overscan: 5 rows above and below viewport
  - Row height: fixed 48px for consistent virtual scroll calculations
  - Visible rows render fully (sparklines, badges, all content)
  - Non-visible rows are unmounted (not hidden via CSS)
- [ ] AC-2: Sort changes re-render the virtual list from the top (scroll to 0
  over 200ms smooth scroll). Filter changes do the same.
- [ ] AC-3: Scroll performance: maintain 60fps during fast scroll as measured
  by Chrome DevTools performance panel. No dropped frames during normal
  scrolling (< 10 rows/sec scroll speed).
- [ ] AC-4: The scrollbar uses the same custom styling as elsewhere: 6px wide,
  `--accent-violet` thumb at 30% opacity on `--void` track, rounded.
- [ ] AC-5: A "Showing X of Y assets" counter in the table footer updates as
  the user scrolls, indicating position in the list.

---

## Story 12.3: Multi-Format Data Export with Cosmic Formatting

**As a** user who needs signal data in external tools (Excel, Python, reporting),
**I want** export to multiple formats with intelligent formatting,
**so that** exported data is immediately usable without cleanup.

### Visual Design

The export button opens a compact cosmic glass popover with format tiles.
Each format option is a small card showing the format icon, name, and a
one-line description of what's included. The popover is minimal and fast --
two clicks to export.

### Acceptance Criteria

- [ ] AC-1: An "Export" button (SVG download icon, `--accent-violet` ghost button)
  in the Signals toolbar opens a **cosmic glass popover** (elevation 3, 280px):
  - **CSV**: table icon + "All visible columns, filtered/sorted as displayed"
  - **JSON**: `{ }` icon + "Full signal data with model metadata"
  - **Clipboard**: copy icon + "Tab-delimited for pasting into spreadsheets"
  Format tiles: 56px tall, full width, cosmic glass hover (`--void-hover`).
  Active (hovered): violet left border (2px) + `--accent-violet` text.
- [ ] AC-2: CSV export:
  - Respects current sort order and filters
  - Numeric formatting: 4 decimal places for probabilities, 2 for returns/Kelly,
    0 for BIC
  - Header row: human-readable column names (not API field names)
  - Filename: `signals_{YYYY-MM-DD}_{HH-mm}.csv`
- [ ] AC-3: JSON export:
  - Full signal data including model metadata, regime, momentum
  - Pretty-printed (2-space indent)
  - Filename: `signals_{YYYY-MM-DD}_{HH-mm}.json`
- [ ] AC-4: Clipboard export: copies tab-delimited data (same as CSV but tabs
  instead of commas). Shows brief emerald toast: "Copied 147 rows to clipboard"
  with a duration of 2 seconds.
- [ ] AC-5: During export generation, the export button shows a spinning indicator
  (8px, `--accent-violet`). On success: brief emerald checkmark flash (300ms).
  On failure (unlikely but possible for very large exports): rose flash with
  retry option.
- [ ] AC-6: Export feature tracks the current filter/sort state so the exported
  data matches exactly what the user sees on screen.

---

# Summary

| # | Epic | Stories | Core Aesthetic |
|---|------|---------|----------------|
| 1 | Navigation Shell | 1.1 - 1.5 | Cosmic pulse sidebar, aurora status strip |
| 2 | Dashboard | 2.1 - 2.5 | Nebula hero canvas, constellation leaderboard |
| 3 | Signals | 3.1 - 3.6 | Micro-charts with aurora threads, cosmic fuzzy glow |
| 4 | Charts | 4.1 - 4.6 | Probability nebula overlay, violet annotations |
| 5 | Risk Dashboard | 5.1 - 5.6 | Speedometer gauge, contagion aurora flows |
| 6 | Tuning | 6.1 - 6.4 | Mission control, star map health grid |
| 7 | Diagnostics | 7.1 - 7.5 | Reliability nebula, polygon radar webs |
| 8 | Data & Services | 8.1 - 8.3 | Heartbeat pulses, batch controls |
| 9 | Arena | 9.1 - 9.3 | Champion glow, holographic trophy case |
| 10 | Micro-Interactions | 10.1 - 10.6 | Cosmic shimmer, materialization, invitation states |
| 11 | Accessibility & Quality | 11.1 - 11.5 | WCAG compliance, responsive cosmic grid |
| 12 | Performance & Export | 12.1 - 12.3 | Stale-while-revalidate, virtual scroll, multi-format |

**Total**: 12 epics, 47 stories, ~290 acceptance criteria.

**Cosmic Apple Gradient Design Thread**: Every story is unified by the cosmic
void-to-violet-to-fuchsia gradient vocabulary. Every surface is a glass
instrument in a cosmic cockpit. Every transition is a spring or ease that
Apple's HIG team would nod approvingly at. Every micro-interaction creates
a moment of delight. Together, they create an interface that is not just
usable -- it is **desirable**. Users don't just use it. They want to live
inside it.






