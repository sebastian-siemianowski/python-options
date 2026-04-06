# Python-Options Web -- Premium UX Redesign

> **Product vision**: Elevate python-options/web from a functional trading dashboard into a
> world-class, immersive command centre that matches the polished cosmic aesthetic of
> lumen-lingo-frontend -- delivering institutional-grade analytics wrapped in
> glass-morphic depth, spring-physics motion, and pixel-perfect typography.

---

## Table of Contents

| # | Epic | Lines |
|---|------|-------|
| 1 | [Design System Foundation](#epic-1--design-system-foundation) | Tokens, typography, colour, motion |
| 2 | [Core Component Library](#epic-2--core-component-library) | GlassCard, Button, motion primitives |
| 3 | [Layout and Navigation Chrome](#epic-3--layout-and-navigation-chrome) | Sidebar, breadcrumb, status strip |
| 4 | [Overview Dashboard](#epic-4--overview-dashboard) | Hero briefing, stat grid, leaderboard |
| 5 | [Signals Command Centre](#epic-5--signals-command-centre) | Table, filters, WebSocket, sparklines |
| 6 | [Risk Observatory](#epic-6--risk-observatory) | Gauge, tabs, cross-asset stress |
| 7 | [Heatmap Star Map](#epic-7--heatmap-star-map) | Grid, tooltips, zone charts |
| 8 | [Charts Terminal](#epic-8--charts-terminal) | Candlestick, overlays, sidebar picker |
| 9 | [Tuning Mission Control](#epic-9--tuning-mission-control) | Retune panel, star map grid, logs |
| 10 | [Arena and Diagnostics](#epic-10--arena-and-diagnostics) | Gate scoring, PIT, model comparison |
| 11 | [Data and Services Operations](#epic-11--data-and-services-operations) | Health, file stats, error logs |
| 12 | [Responsive and Accessibility](#epic-12--responsive-and-accessibility) | Mobile-first, a11y, reduced motion |

---

## Reference: lumen-lingo-frontend Design DNA

The following patterns are the **gold standard** that every story must target:

| Dimension | Pattern |
|-----------|---------|
| **Background** | Deep void `#0a0a0f` with animated mesh gradients and film-grain noise overlay |
| **Glass surfaces** | `rgba(255,255,255,0.04)` bg, `rgba(255,255,255,0.08)` border, `backdrop-filter: blur(12px)` |
| **Glass hover** | `rgba(255,255,255,0.06)` bg, border brightens to `rgba(255,255,255,0.12)`, lift `translateY(-4px)` |
| **Cursor sheen** | Radial gradient follows pointer via framer-motion `useMotionValue` + `useSpring` |
| **Animated border** | Conic gradient (violet-cyan-violet) rotating via `@property --gradient-angle` at 6 s |
| **Typography** | Inter (body, 400-600) + Space Grotesk (headings, 500-700), `tracking-tight` on display |
| **Colour** | Violet `#8b5cf6` primary, Cyan `#06b6d4` secondary, Amber `#f59e0b` premium, Emerald `#10b981` positive, Rose `#f43f5e` negative |
| **Motion** | Framer Motion springs: snappy (400/20), smooth (300/25), gentle (200/22), bouncy (250/15) |
| **FadeIn** | Directional (up/down/left/right), 24 px travel, viewport-triggered, stagger 0.08 s |
| **Buttons** | `whileHover: scale 1.02`, `whileTap: scale 0.97`, violet glow expands 20 px to 48 px |
| **Focus ring** | `ring-2 ring-violet-500 ring-offset-2 ring-offset-background` on `:focus-visible` |
| **Reduced motion** | All CSS animations to `0.01ms`, JS checks `useReducedMotion()`, static fallbacks |
| **Responsive** | Mobile-first Tailwind, blur 8 px mobile / 12 px desktop, parallax halved on touch |

---

<!-- ============================================================ -->
<!-- EPIC 1                                                        -->
<!-- ============================================================ -->

## Epic 1 -- Design System Foundation

> Establish the unified design-token layer, typography scale, colour palette, and motion
> primitives that every subsequent epic depends on. This is the bedrock -- nothing ships
> until these tokens are pixel-matched to lumen-lingo-frontend.

### Story 1.1 -- Migrate CSS Custom Properties to lumen-lingo Token Schema

**As a** front-end developer,
**I want** the CSS custom-property namespace and values in `index.css` to exactly mirror
the lumen-lingo-frontend token schema,
**so that** every downstream component inherits the correct visual language without
per-component overrides.

**Acceptance Criteria**

1. `:root` block defines the following background tokens with exact hex values:
   - `--background: #0a0a0f`
   - `--surface: #111118`
   - `--surface-card: #1a1a24`
   - `--surface-elevated: #22222e`
2. Foreground tokens match:
   - `--foreground: #f4f4f5`
   - `--foreground-secondary: #a1a1aa`
   - `--foreground-muted: #8b8b94`
3. Accent tokens define default, hover, active, and muted (15 % alpha) stops for
   violet, cyan, amber, emerald, and rose.
4. Glass tokens define `--glass`, `--glass-border`, `--glass-hover` at the
   exact `rgba` values used by lumen-lingo-frontend.
5. All existing void-scale tokens (`--void`, `--void-surface`, etc.) are aliased
   to the canonical names above so that no component references break.
6. A `_tokens.css` partial is extracted and imported first in `index.css` to
   separate tokens from component styles.
7. Visual regression: every existing page renders identically -- no colour shifts,
   no missing borders, no broken gradients.
8. Lighthouse colour-contrast audit passes AA for all text tokens against their
   intended background tokens.

---

### Story 1.2 -- Adopt Dual-Font Typography System (Inter + Space Grotesk)

**As a** designer,
**I want** headings rendered in Space Grotesk and body text in Inter with a strict
typographic scale,
**so that** the dashboard matches the premium editorial quality of lumen-lingo.

**Acceptance Criteria**

1. `@font-face` declarations load Space Grotesk (500, 700) and Inter (400, 500, 600)
   from self-hosted WOFF2 files with `font-display: swap`.
2. CSS custom properties define:
   - `--font-sans: 'Inter', system-ui, sans-serif`
   - `--font-display: 'Space Grotesk', 'Inter', system-ui, sans-serif`
3. An 8-stop type scale is defined:
   - `--text-xs: 0.625rem` (10 px)
   - `--text-sm: 0.75rem` (12 px)
   - `--text-base: 0.875rem` (14 px)
   - `--text-md: 1rem` (16 px)
   - `--text-lg: 1.125rem` (18 px)
   - `--text-xl: 1.25rem` (20 px)
   - `--text-2xl: 1.5rem` (24 px)
   - `--text-3xl: 2rem` (32 px)
4. A `<Heading>` utility component accepts `level` (1-4) and maps to responsive
   sizes matching lumen-lingo:
   - h1: `text-2xl sm:text-3xl lg:text-4xl xl:text-5xl`, weight 700
   - h2: `text-xl sm:text-2xl lg:text-3xl`, weight 700
   - h3: `text-lg sm:text-xl lg:text-2xl`, weight 600
   - h4: `text-base sm:text-lg lg:text-xl`, weight 600
5. All headings use `letter-spacing: -0.025em` (`tracking-tight`).
6. Body text defaults to `leading-relaxed` (1.625 line-height).
7. Existing page headings (PageHeader, BriefingCard, section titles) adopt the
   new `<Heading>` component -- no raw `<h1>`-`<h4>` tags remain outside it.
8. Cumulative Layout Shift from font loading is < 0.05 for every page.

---

### Story 1.3 -- Implement Spring-Physics Motion Token Layer

**As a** front-end developer,
**I want** a shared `motion.ts` module exporting spring presets, duration constants,
and variant factories matching lumen-lingo,
**so that** every animated component uses consistent, physically plausible motion.

**Acceptance Criteria**

1. `src/utils/motion.ts` exports four spring configs:
   - `springSnappy: { type: 'spring', stiffness: 400, damping: 20 }`
   - `springSmooth: { type: 'spring', stiffness: 300, damping: 25 }`
   - `springGentle: { type: 'spring', stiffness: 200, damping: 22 }`
   - `springBouncy: { type: 'spring', stiffness: 250, damping: 15 }`
2. Duration constants are exported:
   - `durationFast: 150`, `durationBase: 250`, `durationSlow: 400`,
     `durationContent: 600`, `durationDramatic: 1000`
3. Framer Motion is added as a dependency (`framer-motion ^12`).
4. A `fadeInVariants(direction, distance)` factory returns `{ hidden, visible }`
   variants matching lumen-lingo FadeIn (24 px travel desktop, 20 px mobile).
5. A `staggerContainerVariants(stagger)` factory returns `{ hidden, visible }`
   with `staggerChildren: stagger` (default 0.08).
6. A `useReducedMotion()` hook (re-exported from framer-motion) is used by
   every motion component to disable animation when the OS preference is set.
7. All existing `useScrollReveal` usages are migrated to the new `FadeIn`
   wrapper within this story.
8. Motion tokens are documented in a Storybook-style demo page at `/dev/motion`
   (dev-only route, excluded from production build).

---

### Story 1.4 -- Animated Mesh-Gradient Background with Noise Texture

**As a** user,
**I want** the page background to feature slow-drifting cosmic gradient orbs and a
subtle film-grain noise overlay,
**so that** the app feels alive and immersive, matching lumen-lingo's visual depth.

**Acceptance Criteria**

1. A `<CosmicBackground>` component renders behind all page content at `z-0`
   with `position: fixed; inset: 0; pointer-events: none`.
2. Three radial-gradient orbs are positioned:
   - Top-right: violet `rgba(139,92,246,0.15)`, 600 px diameter, blur 120 px
   - Bottom-left: cyan `rgba(6,182,212,0.12)`, 500 px diameter, blur 100 px
   - Centre: deep-blue `rgba(30,58,95,0.15)`, 700 px diameter, blur 140 px
3. Each orb drifts on a unique CSS `@keyframes` path (40-55 s period),
   `will-change: transform`, hardware-accelerated.
4. An inline SVG `<feTurbulence>` noise overlay is rendered at `opacity: 0.018`
   covering the viewport.
5. A vignette gradient (`radial-gradient(ellipse at center, transparent 0%,
   var(--background) 95%)`) is layered above the orbs.
6. When `prefers-reduced-motion` is active, orb drift animations are paused and
   orbs render at their midpoint positions.
7. GPU memory: the background consumes < 15 MB VRAM on a standard 1080 p display
   (verified via Chrome DevTools Layers panel).
8. The existing `<AmbientOrbs>` component is replaced by `<CosmicBackground>`,
   and the old component file is deleted.

---

### Story 1.5 -- Global Focus Ring and `:focus-visible` System

**As a** keyboard user,
**I want** every interactive element to show a consistent violet focus ring on
`:focus-visible`,
**so that** I can navigate the entire dashboard without a mouse.

**Acceptance Criteria**

1. A global CSS rule sets:
   ```css
   :focus-visible {
     outline: 2px solid #8b5cf6;
     outline-offset: 2px;
   }
   ```
2. `:focus:not(:focus-visible)` removes the outline for mouse clicks.
3. Glass-card components use `ring-2 ring-violet-500 ring-offset-2
   ring-offset-[var(--background)]` as their focus style (Tailwind utilities).
4. The sidebar nav links show the focus ring inside the pill shape, not outside.
5. The command palette (Cmd+K) traps focus within the modal when open and
   returns focus to the trigger element on close.
6. Tab order follows visual reading order on every page (verified by
   tabbing through each route).
7. No element uses `outline: none` or `outline: 0` without providing an
   equivalent visible alternative.
8. Automated axe-core scan of every route reports zero focus-related violations.


---

<!-- ============================================================ -->
<!-- EPIC 2                                                        -->
<!-- ============================================================ -->

## Epic 2 -- Core Component Library

> Build the reusable primitive components that form the visual vocabulary of the
> redesigned dashboard. Every component must match lumen-lingo quality: glass depth,
> spring motion, cursor interaction, and accessibility baked in from day one.

### Story 2.1 -- GlassCard Component with Cursor-Tracking Sheen

**As a** developer,
**I want** a `<GlassCard>` component with glass morphism, pointer-tracking gradient
sheen, hover lift, and optional tint variants,
**so that** every card surface in the app delivers the same premium depth as
lumen-lingo pricing cards.

**Acceptance Criteria**

1. `<GlassCard>` accepts props: `tint` (`'violet' | 'cyan' | 'amber' | 'emerald'
   | 'rose' | 'none'`), `hover` (boolean, default true), `className`, `children`.
2. Default styles:
   - `background: var(--glass)` (`rgba(255,255,255,0.04)`)
   - `border: 1px solid var(--glass-border)` (`rgba(255,255,255,0.08)`)
   - `backdrop-filter: blur(12px)` (8 px on `< sm` breakpoint)
   - `border-radius: 16px` (1rem)
3. When `hover` is true:
   - On mouse-enter, background transitions to `var(--glass-hover)` and border
     brightens to `rgba(255,255,255,0.12)`.
   - The card lifts `translateY(-4px)` and scales to `1.01` via `springSnappy`.
   - A radial gradient (white at 8 % opacity, 300 px diameter) follows the cursor
     position using `framer-motion` `useMotionValue` and `useSpring` (stiffness
     150, damping 15), fading in over 200 ms on enter and out over 400 ms on leave.
   - Box-shadow expands: `0 8px 32px rgba(0,0,0,0.5), 0 0 0 1px
     rgba(255,255,255,0.1)`.
4. When `tint` is set, the border colour shifts to the tint's muted value (e.g.
   `rgba(139,92,246,0.2)` for violet) and a 4 % tint overlay is applied.
5. The component renders a `<div>` with `role="group"` and forwards `ref`.
6. When `prefers-reduced-motion` is active, hover lift and cursor sheen are
   disabled; only background/border colour transitions remain (200 ms ease).
7. The existing `StatCard` component is refactored to wrap `<GlassCard>` instead
   of applying its own glass styles.
8. Storybook-style dev page shows all tint variants, hover states, and reduced-
   motion fallback side by side.

---

### Story 2.2 -- Animated Gradient Border (Conic Shimmer)

**As a** designer,
**I want** a composable `<ShimmerBorder>` wrapper that renders an animated conic-gradient
border rotating around any child component,
**so that** I can apply the signature lumen-lingo shimmer to highlight cards, tier badges,
and CTAs without duplicating CSS.

**Acceptance Criteria**

1. `<ShimmerBorder>` accepts `speed` (seconds, default 6), `colors` (array of 3+
   CSS colours, default `['#8b5cf6', '#06b6d4', '#8b5cf6']`), `borderWidth`
   (px, default 1), `borderRadius` (px, default 16), `children`.
2. Implementation uses `@property --gradient-angle` (registered via
   `CSS.registerProperty` or `@property` at-rule) animated from `0deg` to
   `360deg` over `speed` seconds, `linear`, `infinite`.
3. The shimmer is rendered as a pseudo-element (`::before`) with `conic-gradient(
   from var(--gradient-angle), ...colors)` and masked via `mask-composite:
   exclude` so only the border is visible.
4. Fallback: if `@property` is unsupported (Firefox < 128), the border renders
   as a static gradient at 45 deg -- no animation.
5. When `prefers-reduced-motion` is active, the rotation pauses at 0 deg.
6. `<ShimmerBorder>` is applied to the "champion" model row in ArenaPage and the
   active retune mode button in TuningPage as initial consumers.
7. Performance: the animation runs entirely on the compositor thread; no layout
   or paint is triggered (verified via Chrome Performance recording).

---

### Story 2.3 -- Premium Button Component with Glow and Spring Physics

**As a** user,
**I want** buttons with spring-physics hover/tap feedback and a subtle violet glow
that expands on hover,
**so that** every interaction feels responsive and luxurious.

**Acceptance Criteria**

1. `<Button>` accepts `variant` (`'primary' | 'secondary' | 'ghost' | 'danger'`),
   `size` (`'sm' | 'md' | 'lg'`), `loading` (boolean), `icon` (ReactNode),
   `children`, standard button HTML attributes.
2. **Primary** variant:
   - Background: violet-600 (`#7c3aed`)
   - Text: white
   - Box-shadow glow: `0 0 20px rgba(139,92,246,0.4)`, expanding to
     `0 0 48px rgba(139,92,246,0.25)` on hover
   - `whileHover: { scale: 1.02, filter: 'brightness(1.02)' }` transition `springSnappy`
   - `whileTap: { scale: 0.97, filter: 'brightness(1.04)' }` transition `springSnappy`
3. **Secondary** variant: glass-card background, white text, no glow, same spring scales.
4. **Ghost** variant: transparent background, `foreground-secondary` text, underline on hover.
5. **Danger** variant: rose-600 background, white text, rose glow on hover.
6. **Loading** state: text replaced by a 16 px spinner (SVG circle with
   `stroke-dasharray` animation), button disabled, opacity 0.7.
7. **Icon** renders left of text with 8 px gap; icon-only buttons are square.
8. Focus: `ring-2 ring-violet-500 ring-offset-2 ring-offset-[var(--background)]`.
9. Disabled: `opacity-50 cursor-not-allowed`, no hover/tap animations.
10. All existing `<button>` elements across all pages are migrated to `<Button>`.
11. Every button is accessible: `aria-busy` when loading, `aria-disabled` when
    disabled, minimum 44 x 44 px touch target on mobile.

---

### Story 2.4 -- FadeIn and StaggerChildren Motion Wrappers

**As a** developer,
**I want** `<FadeIn>` and `<StaggerChildren>` components that viewport-trigger
directional reveal animations with configurable stagger,
**so that** page content cascades in gracefully like lumen-lingo sections.

**Acceptance Criteria**

1. `<FadeIn>` accepts `direction` (`'up' | 'down' | 'left' | 'right'`, default
   `'up'`), `distance` (px, default 24, 20 on mobile), `delay` (ms, default 0),
   `duration` (spring config, default `springGentle`), `once` (boolean, default
   true), `className`, `children`.
2. Uses `framer-motion` `useInView` with `margin: '-40px'` to trigger when
   the element is 40 px inside the viewport.
3. Hidden state: `opacity: 0`, translated by `distance` in the given direction.
4. Visible state: `opacity: 1`, `translate: 0` with the specified spring.
5. When `once` is true, animation plays only on first intersection.
6. `<StaggerChildren>` wraps children and applies `staggerChildren: 0.08` (or
   custom prop) to the container variants.
7. `<StaggerItem>` is used as the direct child wrapper, applying `fadeInVariants`.
8. When `useReducedMotion()` returns true, both components render children
   immediately at full opacity with no translate.
9. All existing `useScrollReveal` hooks and manual IntersectionObserver
   animations are replaced by `<FadeIn>` / `<StaggerChildren>`.
10. Page load performance: Time to Interactive is not increased by more than 50 ms
    on any route (measured via Lighthouse).

---

### Story 2.5 -- CountUp Animated Number with Spring Physics

**As a** user,
**I want** stat numbers to count up from zero with spring easing when they scroll into
view,
**so that** the dashboard feels dynamic and data-rich.

**Acceptance Criteria**

1. `<CountUp>` accepts `value` (number), `decimals` (number, default 0),
   `prefix` (string), `suffix` (string), `duration` (spring config, default
   `springSmooth`), `className`.
2. Animation triggers on `useInView` (once, margin `-40px`).
3. The number interpolates from 0 to `value` using `framer-motion`
   `useSpring` + `useTransform`, rounded to `decimals` places.
4. When `value` changes after initial mount, the counter re-animates from the
   previous value to the new value.
5. `prefix` and `suffix` render outside the animated span (e.g. "$" prefix,
   "%" suffix).
6. When reduced motion is preferred, the final value renders immediately.
7. The existing `useCountUp` hook is replaced; all `<StatCard>` instances
   use `<CountUp>` internally.
8. Numbers above 999 are formatted with locale-appropriate thousand separators
   (using `Intl.NumberFormat('pl-PL')` for PLN context).

---

### Story 2.6 -- Cosmic Empty, Error, and Skeleton States

**As a** user,
**I want** loading skeletons, empty states, and error states to use the same
glass-morphic visual language as the rest of the app,
**so that** transitional UI feels intentional rather than broken.

**Acceptance Criteria**

1. `<CosmicSkeleton>` renders pulsing glass-card rectangles with a shimmer
   gradient sweep (1.8 s period, `background-size: 200%`).
   - Accepts `lines` (number), `height` (per line), `avatar` (boolean).
   - Shimmer gradient: `linear-gradient(90deg, transparent 0%, rgba(255,255,
     255,0.04) 50%, transparent 100%)`.
2. `<CosmicEmptyState>` renders centred content:
   - A 64 px Lucide icon in `foreground-muted` colour.
   - A `<Heading level={3}>` title.
   - A `<Text variant="secondary">` description.
   - An optional `<Button variant="secondary">` action.
   - The icon floats on a `6s ease-in-out infinite` keyframe (disabled for
     reduced motion).
3. `<CosmicErrorState>` renders:
   - A rose-tinted glass card with `tint="rose"`.
   - Error icon (AlertTriangle), title, message, and a "Retry" `<Button
     variant="danger">`.
   - The retry button calls the provided `onRetry` callback.
4. All three components wrap themselves in `<FadeIn direction="up">`.
5. Every `isLoading` branch across all pages uses `<CosmicSkeleton>` with
   appropriate `lines` / `height` matching the content it replaces.
6. Every `isError` branch uses `<CosmicErrorState>` with `onRetry` wired to
   the React Query `refetch` function.
7. Every empty-data branch uses `<CosmicEmptyState>` with a contextual message.
8. Skeleton shimmer pauses when the browser tab is hidden (`document.hidden`).


---

<!-- ============================================================ -->
<!-- EPIC 3                                                        -->
<!-- ============================================================ -->

## Epic 3 -- Layout and Navigation Chrome

> The application shell -- sidebar, breadcrumb, status strip, command palette -- is the
> user's constant companion. It must be flawless: responsive, animated, keyboard-driven,
> and visually aligned with lumen-lingo's navigation patterns.

### Story 3.1 -- Glass Sidebar with Collapse Animation and Keyboard Toggle

**As a** power user,
**I want** the sidebar to collapse and expand with a smooth spring animation, toggled
by Cmd+B or a dedicated button,
**so that** I can maximise screen space for charts and tables when needed.

**Acceptance Criteria**

1. The sidebar renders as a `<nav>` element with `aria-label="Main navigation"`.
2. Expanded width: 256 px. Collapsed width: 72 px (icon-only mode).
3. Transition uses `springSnappy` via `framer-motion` `animate` on `width`.
4. In collapsed mode:
   - Nav labels fade out (`opacity: 0`, `overflow: hidden`) over 150 ms.
   - Icons remain centred within the 72 px rail.
   - Hovering a collapsed nav item shows a tooltip (glass card, 400 ms delay,
     springSnappy enter, position right-of-icon).
5. In expanded mode:
   - Labels fade in after width reaches 200 px (sequenced via `AnimatePresence`).
   - Badge counts render to the right of labels.
6. Collapse state persists in `localStorage` key `sidebar-collapsed`.
7. `Cmd+B` (Mac) / `Ctrl+B` (Win) toggles collapse; focus remains on the
   previously focused element.
8. The sidebar background uses `var(--glass)` with `backdrop-filter: blur(16px)`,
   a right border of `var(--glass-border)`, and a subtle inner shadow.
9. On screens below 768 px, the sidebar renders as a slide-over drawer triggered
   by a hamburger button, with a backdrop overlay at `rgba(0,0,0,0.6)`.
10. Active route indicator: a 3 px violet bar on the left edge of the active
    link, animated in with `scaleY(0 to 1)` origin bottom.
11. Focus ring on nav links respects the pill shape (rounded, inset).

---

### Story 3.2 -- Breadcrumb Bar with Route Hierarchy and Page Transitions

**As a** user,
**I want** a breadcrumb bar above the content area showing my location in the app
hierarchy, with smooth page transition animations,
**so that** navigation feels spatial and I always know where I am.

**Acceptance Criteria**

1. `<BreadcrumbBar>` renders a `<nav aria-label="Breadcrumb">` with an `<ol>`
   of `<li>` items.
2. Breadcrumb items are derived from the current route path:
   - `/` renders "Overview" only (no breadcrumb trail).
   - `/charts/AAPL` renders "Charts / AAPL".
   - `/diagnostics/profitability` renders "Diagnostics / Profitability".
3. Each segment is a clickable link except the current (last) segment, which
   renders as `aria-current="page"`.
4. Separator: a Lucide `ChevronRight` icon at 14 px, `foreground-muted` colour.
5. Breadcrumb text uses `text-sm font-medium` in `foreground-secondary`,
   with the current segment in `foreground`.
6. Page content transitions: when navigating between routes, outgoing content
   fades out (opacity 1 to 0, y 0 to -8, 150 ms) and incoming content fades
   in (opacity 0 to 1, y 8 to 0, 200 ms, springGentle) via `AnimatePresence`
   wrapping the `<Outlet>`.
7. If routes share the same parent (e.g. `/diagnostics` to
   `/diagnostics/profitability`), only the child content transitions -- the
   header persists.
8. Reduced motion: page transitions are instant (no fade/translate).

---

### Story 3.3 -- Status Strip with Regime Awareness and Live Indicators

**As a** user,
**I want** the status strip to show real-time regime status, WebSocket health, and
last-updated timestamps with appropriate colour coding,
**so that** I have constant ambient awareness of system state.

**Acceptance Criteria**

1. `<StatusStrip>` renders as a fixed bar at the bottom of the viewport, 36 px
   tall, glass background with 20 px blur.
2. Left section: Regime badge showing current market regime from risk data:
   - Calm: emerald dot + "Calm" label
   - Elevated: amber dot + "Elevated"
   - Stressed: orange dot + "Stressed"
   - Crisis: rose dot + pulsing "Crisis" label
3. Centre section: key metric pills (Total Signals, PIT Pass %, Temperature)
   updating live.
4. Right section:
   - WebSocket status dot (green = connected, amber = reconnecting with pulse,
     red = disconnected)
   - Last data refresh timestamp as relative time ("2 min ago")
5. On mobile (< 768 px), the strip collapses to regime dot + WS dot only,
   with a tap-to-expand drawer.
6. Status strip data comes from existing React Query caches (no additional
   API calls).
7. Regime colour transitions smoothly over 400 ms when the regime changes.
8. The strip has a subtle top border of `var(--glass-border)`.

---

### Story 3.4 -- Command Palette Redesign with Asset Search and Quick Actions

**As a** power user,
**I want** the command palette to feel like a premium search experience with
fuzzy matching, categorised results, and keyboard-driven interaction,
**so that** I can navigate anywhere in the app within 2 keystrokes.

**Acceptance Criteria**

1. `Cmd+K` opens the palette; `Escape` closes it. Focus traps inside.
2. The palette renders as a centred modal (480 px wide) with glass background,
   20 px blur, and a shimmer border.
3. Input: full-width search field with `text-lg`, placeholder "Search assets,
   pages, actions...", auto-focused on open.
4. Results are categorised with section headers:
   - **Assets**: fuzzy-matched against all tracked symbols, showing signal badge
     and sector.
   - **Pages**: all 11 routes with icons.
   - **Actions**: Refresh Data, Start Retune, Toggle Sidebar, Export Signals.
5. Each result item shows: icon, label, description, and a keyboard shortcut
   hint (if applicable).
6. Up/Down arrows navigate results; Enter selects; hover highlights.
7. Selected asset navigates to `/charts/{symbol}`.
8. Selected page navigates to the route.
9. Selected action executes immediately (e.g. triggers `api.triggerDataRefresh()`).
10. Results animate in with staggered fade-up (0.03 s per item).
11. The backdrop overlay is `rgba(0,0,0,0.6)` with a click-to-close handler.
12. Search is debounced at 100 ms. Empty input shows "Recent" items (last 5
    navigation targets, stored in localStorage).
13. Maximum 12 results visible; overflow scrolls with custom scrollbar.
14. Accessible: `role="combobox"`, `aria-expanded`, `aria-activedescendant`,
    `aria-owns` on the listbox.

---

### Story 3.5 -- Logo Block and System Identity Polish

**As a** user,
**I want** the sidebar logo block to feature a refined brand identity with the system
version number and a health indicator,
**so that** the app feels professional and trustworthy.

**Acceptance Criteria**

1. The logo block sits at the top of the sidebar, 64 px tall in expanded mode.
2. Expanded: a custom SVG icon (gradient fill from violet to cyan, 32 px) next
   to "Signal Engine" in Space Grotesk 700, with `v5.30` as a muted badge.
3. Collapsed: only the SVG icon, centred in the 72 px rail.
4. A 6 px dot below the icon pulses in the regime colour (calm = emerald,
   etc.), indicating system health.
5. Hover on the logo block in expanded mode shows a glass tooltip with:
   - "BMA + Kalman Signal Engine"
   - Number of tracked assets
   - Cache age
   - API uptime
6. The tooltip has a 400 ms hover delay and enters with `springSnappy`.
7. The logo icon uses `will-change: transform` and subtly rotates 2 deg on
   sidebar hover (returning to 0 on leave) via `springGentle`.
8. The version badge links to a changelog anchor (scroll target, no navigation).


---

<!-- ============================================================ -->
<!-- EPIC 4                                                        -->
<!-- ============================================================ -->

## Epic 4 -- Overview Dashboard

> The overview page is the first thing users see. It must deliver a "mission-briefing"
> experience -- data-dense yet visually serene, with every metric card, chart, and panel
> wrapped in the premium glass language.

### Story 4.1 -- Morning Briefing Hero Card Redesign

**As a** user,
**I want** the morning briefing card to feel like a premium mission-briefing panel with
cosmic ambient glow, a three-column layout, and staggered reveal,
**so that** opening the dashboard feels like entering a command centre.

**Acceptance Criteria**

1. `<BriefingCard>` renders inside a `<GlassCard tint="violet">` with an animated
   shimmer border.
2. Header: "Morning Briefing" in `<Heading level={2}>` with a cosmic gradient
   text fill (`background-clip: text`, violet-to-cyan).
3. Subtitle: today's date in `text-sm foreground-secondary`, formatted as
   "Friday, 4 April 2026".
4. Three-column layout (stacked on mobile):
   - **Signals**: signal count, strong buy/sell counts, distribution mini-bar
   - **Models**: tuned count, PIT pass rate (colour-coded), freshness badge
   - **Market**: temperature gauge (mini), regime label, risk level
5. Each column has a gradient divider between them (vertical on desktop,
   horizontal on mobile): `from-transparent via-glass-border to-transparent`.
6. Data fields use `<CountUp>` for numeric values.
7. The entire card enters with `<FadeIn direction="up" delay={200}>`.
8. On error (partial data), affected columns show a muted "Unavailable" label
   instead of broken numbers.
9. The briefing card spans the full content width on all breakpoints.
10. A refresh button (secondary, icon-only) in the top-right corner triggers
    all overview queries to refetch.

---

### Story 4.2 -- Stat Card Grid with Staggered Reveal and Responsive Layout

**As a** user,
**I want** the stat cards to cascade in with staggered animation, each card
a mini glass surface with count-up numbers and contextual colour coding,
**so that** scanning key metrics is instant and delightful.

**Acceptance Criteria**

1. Stat cards are wrapped in `<StaggerChildren stagger={0.06}>`.
2. Each `<StatCard>` wraps a `<GlassCard hover>` containing:
   - A Lucide icon (24 px) in the card's tint colour.
   - A label in `text-xs foreground-muted uppercase tracking-wider`.
   - A value in `text-2xl font-display font-bold`.
   - An optional trend indicator (arrow up/down + percentage, colour-coded).
3. The top row (5 cards: Total Assets, Strong Buy, Hold, Sell, System Health)
   renders in a responsive grid: `grid-cols-2 sm:grid-cols-3 lg:grid-cols-5`.
4. The System Health card shows a live pulse dot (emerald when all services up,
   amber with 1+ degraded, rose when critical).
5. The second row (4 cards: Tuned Models, PIT Pass, Price Files, Cache Age)
   uses `grid-cols-2 lg:grid-cols-4`.
6. All numeric values use `<CountUp>` with appropriate `decimals` and `suffix`.
7. Cards that represent negative states (Sell, old cache) use `tint="rose"` on
   the `<GlassCard>`.
8. Cards that represent positive states (Strong Buy, fresh cache) use
   `tint="emerald"`.
9. On hover, the card's icon subtly scales to 1.1 via `springBouncy`.
10. Each card has a minimum height of 120 px to prevent layout shift as data loads.

---

### Story 4.3 -- Signal Distribution Bar Redesign

**As a** user,
**I want** the signal distribution bar to render as a flowing gradient strip with
labelled segments and hover detail,
**so that** I can instantly see the balance of buy vs sell signals.

**Acceptance Criteria**

1. `<SignalDistributionBar>` renders a horizontal bar, 48 px tall, fully rounded
   corners (24 px radius), inside a `<GlassCard>`.
2. Segments represent signal categories (strong buy, buy, hold, sell, strong sell,
   exit) with widths proportional to count.
3. Colours: strong buy = `#10b981`, buy = `#34d399`, hold = `#a1a1aa`,
   sell = `#fb7185`, strong sell = `#f43f5e`, exit = `#8b8b94`.
4. Segment transitions: width changes animate over `springSmooth`.
5. On hover over a segment:
   - The segment lifts 2 px (`translateY(-2px)`).
   - A glass tooltip appears above showing: label, count, percentage.
   - Adjacent segments dim to 60 % opacity.
6. Below the bar, a legend renders horizontally with coloured dots and labels.
7. On mobile, the bar stacks vertically as horizontal mini-bars per category.
8. When data is loading, the bar shows a shimmer skeleton at the same height.
9. The entire component enters with `<FadeIn direction="up">`.

---

### Story 4.4 -- Model Leaderboard with Rank Animation

**As a** user,
**I want** the model leaderboard to show ranked models in a premium list with
position indicators and confidence bars,
**so that** I can see which models are performing best at a glance.

**Acceptance Criteria**

1. `<ModelLeaderboard>` renders inside a `<GlassCard>` with a `<Heading
   level={3}>` title "Model Leaderboard".
2. Each row shows: rank number, model name, win count bar, average weight bar,
   and a confidence percentage.
3. Rank 1 (champion) has a `<ShimmerBorder>` around its row and the rank number
   renders in amber with a crown icon (Lucide `Crown`).
4. Win-count and weight bars are horizontal fills inside glass-surface
   containers, coloured by model family (violet for Kalman, cyan for Phi,
   emerald for Momentum).
5. Bars animate from 0 % to final width using `springSmooth` on viewport entry.
6. Rows enter with `<StaggerChildren stagger={0.05}>`.
7. On hover, a row's background shifts to `var(--glass-hover)` and the bars
   brighten by 10 %.
8. Maximum 8 models shown; if more exist, a "Show all" button reveals the rest
   with a slide-down animation.
9. Empty state: `<CosmicEmptyState>` with message "No models tuned yet".
10. Data refreshes every 120 s in the background.

---

### Story 4.5 -- Conviction Spotlight Panels

**As a** user,
**I want** the conviction spotlight to render strong buy and sell assets in dual
glass panels with signal strength indicators and quick-chart links,
**so that** high-conviction opportunities are immediately actionable.

**Acceptance Criteria**

1. `<ConvictionSpotlight>` renders two `<GlassCard>` panels side by side
   (stacked on mobile):
   - **Strong Buys**: `tint="emerald"`, heading "Strongest Buys"
   - **Strong Sells**: `tint="rose"`, heading "Strongest Sells"
2. Each panel lists up to 5 assets, each showing:
   - Ticker symbol in `font-display font-bold`.
   - Sector pill badge (glass background, 8 px radius).
   - Signal strength bar (0-100 %, colour-matched to panel tint).
   - Expected return as a `<CountUp>` value with "%" suffix.
   - A link icon that navigates to `/charts/{symbol}` on click.
3. The panels enter with `<FadeIn direction="left">` and
   `<FadeIn direction="right">` respectively.
4. Asset rows enter with `<StaggerChildren stagger={0.06}>`.
5. On hover, the row lifts 2 px and the link icon becomes fully opaque.
6. If no strong signals exist for a direction, the panel shows
   `<CosmicEmptyState>` with "No strong buys detected" / "No strong sells
   detected".
7. Signal strength bar animates from 0 to value on viewport entry.
8. Panel heading uses gradient text matching the tint colour.


---

<!-- ============================================================ -->
<!-- EPIC 5                                                        -->
<!-- ============================================================ -->

## Epic 5 -- Signals Command Centre

> The signals page is the operational heart of the app. Every interaction -- filtering,
> sorting, searching, expanding rows, receiving WebSocket updates -- must feel
> instantaneous, visually rich, and keyboard-accessible.

### Story 5.1 -- Signals Table Glass Redesign with Row Hover and Expand

**As a** trader,
**I want** the signals table to render inside glass surfaces with premium row hover
states, smooth row expansion, and column alignment,
**so that** scanning 100+ assets feels effortless.

**Acceptance Criteria**

1. The table renders inside a `<GlassCard>` with `overflow-x: auto` and custom
   thin scrollbar (4 px, violet thumb on hover).
2. Table header: sticky, `background: var(--surface)` with bottom border of
   `var(--glass-border)`, text in `text-xs uppercase tracking-wider font-medium
   foreground-muted`.
3. Table rows alternate between `transparent` and `rgba(255,255,255,0.015)`.
4. Row hover: entire row transitions to `var(--glass-hover)` with a left-edge
   accent glow (2 px, violet for buy, rose for sell, muted for hold).
5. Row click expands a detail panel below the row with `AnimatePresence`:
   - Height animates from 0 to auto via `springSmooth`.
   - Content fades in with 100 ms delay.
   - Detail panel has a glass background with 4 px left border matching signal
     colour.
6. Expanded panel shows: all-horizon signals, momentum chart (sparkline),
   expected returns per horizon, model confidence, and a "View Chart" button
   linking to `/charts/{symbol}`.
7. Only one row can be expanded at a time; expanding another collapses the
   previous with a crossfade.
8. Column widths are defined via `grid-template-columns` for consistent
   alignment (not flexbox).
9. The table supports horizontal scroll on mobile with a fade-edge indicator
   (gradient mask on the right edge).
10. Minimum column widths prevent content wrapping on any standard data.

---

### Story 5.2 -- Sparkline, Momentum Badge, and Signal Strength Bar Components

**As a** trader,
**I want** inline sparklines, momentum badges, and signal strength bars in each
table row,
**so that** I can assess direction, momentum, and strength without expanding the row.

**Acceptance Criteria**

1. `<Sparkline>` renders an inline SVG (80 x 24 px) showing the last 30 data
   points as a smooth line.
   - Positive trend: emerald stroke.
   - Negative trend: rose stroke.
   - Flat: `foreground-muted` stroke.
   - A filled area below the line at 10 % opacity of the stroke colour.
   - Optional: animated draw-in from left to right on viewport entry (600 ms).
2. `<MomentumBadge>` renders a pill showing momentum direction:
   - Accelerating: upward arrow + "Accel" in emerald on emerald-muted background.
   - Decelerating: downward arrow + "Decel" in rose on rose-muted background.
   - Stable: horizontal arrows + "Stable" in muted on glass background.
3. `<SignalStrengthBar>` renders a horizontal fill bar (60 x 8 px, rounded):
   - Fill colour: gradient from the signal category colour to its lighter variant.
   - Background: `var(--glass)`.
   - Fill width animates from 0 % to value on mount.
4. All three components accept a `compact` prop for tighter spacing in dense
   table layouts.
5. All three respect `prefers-reduced-motion`: sparkline renders without
   draw-in, bars render at full width immediately.
6. Each component has a `title` attribute for accessibility (e.g. "Sparkline:
   30-day trend, positive 3.2%").

---

### Story 5.3 -- Real-Time WebSocket Signal Flash and Change Log

**As a** trader,
**I want** live WebSocket signal updates to flash-highlight affected rows and
maintain a change log badge,
**so that** I never miss a signal change.

**Acceptance Criteria**

1. When a WebSocket message arrives for an asset:
   - The table row briefly flashes with a translucent colour wash:
     green for upgrade, rose for downgrade (300 ms fade-in, 1 s hold, 500 ms
     fade-out).
   - A subtle "aurora trail" gradient sweeps left-to-right across the row
     over 800 ms.
2. A `<ChangeCounter>` badge renders in the table header showing the count of
   changes since the user last acknowledged. Background pulses slowly.
3. Clicking the change counter opens a `<ChangeLog>` drawer (glass panel,
   slides in from the right, 320 px wide):
   - Lists changes chronologically: timestamp, symbol, old signal, new signal,
     direction arrow.
   - Each entry has a colour-coded left border (emerald for upgrade, rose for
     downgrade).
   - Maximum 50 entries stored in memory; older entries are pruned.
4. Closing the drawer resets the counter to 0.
5. WebSocket status indicator: a 6 px dot in the page header:
   - Connected: emerald, tooltip "Live -- real-time updates active"
   - Reconnecting: amber, pulsing, tooltip "Reconnecting..."
   - Disconnected: rose, tooltip "Disconnected -- data may be stale"
6. All flash animations respect `prefers-reduced-motion` (no motion, only
   colour highlight for 1 s).
7. WebSocket reconnection uses exponential backoff (1 s, 2 s, 4 s, ... max 30 s)
   as already implemented; this story ensures the UI reflects state accurately.

---

### Story 5.4 -- Signal Filter Pills and Horizon Selector Redesign

**As a** trader,
**I want** filter pills and horizon selectors styled as premium glass capsules with
spring-animated active states,
**so that** filtering feels tactile and the active state is unmistakable.

**Acceptance Criteria**

1. Filter pills render as a horizontally scrollable row (snap-scroll on mobile)
   of glass capsules (`px-4 py-2 rounded-full`).
2. Inactive pill: `var(--glass)` background, `foreground-secondary` text.
3. Active pill: filled with the signal's colour (e.g. emerald for "Strong Buy"),
   white text, subtle glow matching the colour. Transition via `layoutId`
   (framer-motion shared layout) for a sliding indicator animation.
4. Pill click triggers `springSnappy` scale tap animation (0.95 on press, 1.0
   on release).
5. Each pill shows a count badge (right-aligned, smaller font) if `showCounts`
   is true.
6. Horizon selector renders as a similar pill row with time labels (1D, 5D,
   21D, 63D, 126D, 252D).
7. Active horizon pill has a violet bottom bar (3 px) animated with `layoutId`.
8. A "responsive auto" option is the default, automatically selecting the
   horizon based on viewport width (fewer horizons on narrow screens).
9. Both pill rows are keyboard-navigable: left/right arrows move between pills,
   Enter/Space activates.
10. When the filter changes, the table content crossfades (150 ms out, 200 ms in)
    via `AnimatePresence` with `mode="wait"`.

---

### Story 5.5 -- Multi-Level Sort with Visual Indicators

**As a** power user,
**I want** to sort the signals table by up to 3 columns simultaneously with clear
visual indicators of sort priority and direction,
**so that** I can create custom rankings without leaving the page.

**Acceptance Criteria**

1. Clicking a column header sorts by that column ascending; clicking again
   reverses to descending; clicking a third time removes the sort.
2. Shift+clicking a column header adds it as a secondary (or tertiary) sort
   level without removing existing sorts.
3. Active sort columns show:
   - A directional arrow (up/down) in violet.
   - A small circled number (1/2/3) indicating priority, positioned to the
     right of the arrow.
   - The header text shifts to `foreground` (full brightness) from
     `foreground-muted`.
4. Sort state persists in `localStorage` per page.
5. When sort changes, affected rows reorder with `layoutId` animation
   (rows slide to their new positions over `springSmooth`).
6. A "Clear sort" button appears in the table toolbar when any sort is active.
7. The sort indicator elements animate in with `springBouncy` (scale from 0 to 1).
8. Column headers have a hover state: text brightens, a subtle underline fades in.
9. Keyboard: pressing Enter on a focused column header triggers the sort cycle.

---

### Story 5.6 -- Sector Grouping with Collapsible Glass Sections

**As a** trader,
**I want** the "By Sector" view to show collapsible sector groups with glass headers,
signal distribution bars, and staggered row reveals,
**so that** I can focus on sectors of interest and hide the rest.

**Acceptance Criteria**

1. Each sector group renders with a glass header bar containing:
   - Chevron icon (rotates 90 deg on expand/collapse via `springSnappy`).
   - Sector name in `font-display font-semibold`.
   - Asset count badge (`text-xs foreground-muted`).
   - Inline `<SignalDistributionBar>` (compact variant, 24 px tall) showing
     the sector's signal breakdown.
   - "Expand all" / "Collapse all" toggle button (ghost variant) in the
     toolbar.
2. Sector collapse/expand animates content height from 0 to auto via
   `springSmooth` with `overflow: hidden` during transition.
3. When a sector expands, its rows enter with `<StaggerChildren stagger={0.03}>`.
4. Collapsed state per sector persists in `localStorage`.
5. The total number of assets across all visible (expanded) sectors is shown
   in the toolbar.
6. Sector headers remain sticky at the top of the scroll container when
   scrolling within that sector's rows.
7. Keyboard: Enter on a focused sector header toggles expand/collapse.
8. When all sectors are collapsed, a `<CosmicEmptyState>` does NOT appear --
   the collapsed headers remain visible with their distribution bars.


---

<!-- ============================================================ -->
<!-- EPIC 6                                                        -->
<!-- ============================================================ -->

## Epic 6 -- Risk Observatory

> The risk page translates abstract market stress into visceral visual language. The
> centrepiece gauge, temperature sparkline, and stress category panels must feel like
> instruments in a premium observatory.

### Story 6.1 -- Cosmic Speedometer Gauge Redesign

**As a** user,
**I want** the risk gauge to render as a polished SVG arc with spring-animated needle,
regime-coloured glow, and a sparkling particle effect at the needle tip,
**so that** the risk level is immediately visceral.

**Acceptance Criteria**

1. The gauge renders as a 280 x 200 px SVG (scales proportionally on mobile).
2. Arc spans 180 deg (semicircle), divided into 4 colour zones:
   - 0-25: emerald (Calm)
   - 25-50: amber (Elevated)
   - 50-75: orange (Stressed)
   - 75-100: rose (Crisis)
3. Zone boundaries are soft gradients (no hard edges).
4. The needle animates from the previous temperature to the new value using
   `springSmooth` (via `framer-motion` `useSpring` on the angle).
5. A glow effect surrounds the arc at the needle position:
   - Radius: 30 px
   - Colour: matches the current zone
   - Opacity: `0.3 + (temperature / 100) * 0.4` (brighter at higher risk)
6. At the needle tip, 3-5 tiny particles (2 px circles) drift outward and fade
   over 1 s, creating a "spark" effect. Disabled for reduced motion.
7. Below the arc: temperature number in `text-3xl font-display font-bold`,
   colour-matched to the zone, with `<CountUp>`.
8. Below the number: regime label pill (glass background, zone-coloured text).
9. The gauge is wrapped in a `<GlassCard>` with the zone's tint.
10. On hover, the glow intensifies by 20 %.

---

### Story 6.2 -- Temperature Sparkline with Canvas Rendering

**As a** user,
**I want** a temperature history sparkline below the gauge showing the last 7 days
of hourly snapshots with animated reveal,
**so that** I can see risk trends at a glance.

**Acceptance Criteria**

1. The sparkline renders on an HTML `<canvas>` element (full width, 60 px height)
   inside a `<GlassCard>`.
2. Data source: localStorage-stored temperature snapshots (hourly, 7 days max).
3. Line colour: gradient from the oldest zone colour to the current zone colour.
4. Fill below the line at 8 % opacity of the line gradient.
5. On mount, the line draws left-to-right over 1.2 s using `requestAnimationFrame`.
6. A dashed horizontal line at the "Stressed" threshold (50) renders in
   `foreground-muted` at 30 % opacity.
7. On hover, a vertical crosshair follows the cursor with a tooltip showing:
   timestamp, temperature value, zone label.
8. Tooltip renders as a glass pill positioned above the crosshair point.
9. The latest data point renders as a filled dot (6 px) pulsing at the current
   zone colour.
10. When reduced motion is preferred, the sparkline draws immediately (no
    left-to-right animation).
11. If fewer than 2 data points exist, the sparkline shows a centred message:
    "Collecting data..." in `foreground-muted`.

---

### Story 6.3 -- Risk Tab Navigation with Glass Pill Selector

**As a** user,
**I want** the risk page tabs to use the same premium glass-pill styling as lumen-lingo
navigation, with a sliding indicator and spring animation,
**so that** tab switching feels spatial and intentional.

**Acceptance Criteria**

1. Tab bar renders 6 tabs (Overview, Cross-Asset, Metals, Market, Sectors,
   Currencies) as glass pills in a horizontal row.
2. Active tab: filled with `var(--glass-hover)`, `foreground` text, and a
   bottom 3 px violet bar.
3. The bottom bar slides between tabs using `framer-motion` `layoutId`
   shared-layout animation (`springSnappy`).
4. Inactive tabs: `foreground-secondary` text, `var(--glass)` background.
5. Each tab has an icon (Lucide) to the left of the label.
6. Tab content transitions: outgoing content fades out (opacity, 100 ms),
   incoming content fades in (opacity, 150 ms), direction matches tab
   position (left tabs slide content right, right tabs slide left).
7. On mobile (< 768 px), tabs render as horizontally scrollable pills with
   snap-scroll and the active tab auto-scrolls into view.
8. Keyboard: left/right arrows navigate between tabs, Enter/Space activates.
9. ARIA: `role="tablist"`, `role="tab"`, `role="tabpanel"`,
   `aria-selected`, `aria-controls`.

---

### Story 6.4 -- Cross-Asset Stress Categories with Glass Cards

**As a** user,
**I want** the cross-asset stress view to render each stress category as a glass card
with a coloured severity indicator, metric details, and hover expansion,
**so that** I can quickly identify which asset classes are under stress.

**Acceptance Criteria**

1. Each stress category renders as a `<GlassCard>` with tint matching severity:
   - Low: `tint="emerald"`
   - Moderate: `tint="amber"`
   - High: `tint="rose"`
2. Card header: category name + severity pill badge.
3. Card body: key metrics as label-value pairs (`text-xs` labels,
   `text-lg font-display` values).
4. Values use `<CountUp>` with appropriate decimal formatting.
5. Cards are arranged in a responsive grid: `grid-cols-1 sm:grid-cols-2
   lg:grid-cols-3`.
6. Cards enter with `<StaggerChildren stagger={0.06}>`.
7. On hover, the card lifts and the severity indicator glow intensifies.
8. A "Last updated" timestamp renders below the grid in `text-xs
   foreground-muted`.
9. On refresh, cards crossfade to updated values (not full unmount/remount).
10. Each card has a `role="group"` with an `aria-label` describing the category
    and its severity.


---

<!-- ============================================================ -->
<!-- EPIC 7                                                        -->
<!-- ============================================================ -->

## Epic 7 -- Heatmap Star Map

> The heatmap is the app's most visually distinctive view -- a sector-by-horizon grid
> of colour-coded cells. The redesign elevates it from a functional table into an
> immersive star map with rich tooltips, keyboard navigation, and inline zone charts.

### Story 7.1 -- Heatmap Grid with Premium Cell Rendering

**As a** trader,
**I want** each heatmap cell to render with smooth colour gradients, rounded corners,
subtle inner shadows, and hover glow,
**so that** the heatmap feels like a high-end data visualisation, not a spreadsheet.

**Acceptance Criteria**

1. Each cell renders as a 48 x 36 px rounded rectangle (`border-radius: 6px`)
   with a background colour derived from the expected return value:
   - Strong positive (> 5 %): emerald-500
   - Positive (0-5 %): emerald-400 at 60 % opacity
   - Neutral (close to 0): `var(--surface-card)`
   - Negative (0 to -5 %): rose-400 at 60 % opacity
   - Strong negative (< -5 %): rose-500
2. Colour transitions smoothly across the range (not discrete buckets) using
   a linear interpolation between emerald and rose through neutral.
3. Each cell displays the expected return value as centred text (`text-xs
   font-medium`), with white text on strong colours and `foreground-muted` on
   neutral.
4. On hover:
   - The cell scales to 1.15 via `springSnappy`.
   - A glow matching the cell colour appears (8 px radius, 40 % opacity).
   - Adjacent cells dim slightly (to 70 % opacity) creating a spotlight effect.
5. On click, the cell triggers row expansion (see Story 7.3).
6. Cells animate in with a wave effect: stagger by column index * row index *
   10 ms on initial load.
7. When data updates, cells crossfade colour/value over 400 ms (not remount).
8. Cell borders: `1px solid rgba(255,255,255,0.04)`.

---

### Story 7.2 -- Heatmap Tooltip with Rich Data Card

**As a** trader,
**I want** hovering a heatmap cell to show a premium glass tooltip with comprehensive
signal data,
**so that** I can assess an asset without expanding the row.

**Acceptance Criteria**

1. Tooltip appears after 200 ms hover delay, positioned above or below the
   cell (auto-flip to stay in viewport).
2. Tooltip renders as a `<GlassCard>` (200 px wide) with:
   - Asset name in `font-display font-semibold`.
   - Signal label pill (colour-coded).
   - Expected return as a large number with `<CountUp>`.
   - P(up) as a radial progress indicator (36 px SVG circle, filled arc).
   - Kelly criterion value.
   - Momentum badge (`<MomentumBadge>` component from Story 5.2).
   - Horizon label.
3. Tooltip enter animation: `opacity 0 to 1`, `scale 0.95 to 1`, `y +4 to 0`,
   `springSnappy`.
4. Tooltip exit: `opacity 1 to 0`, `scale 1 to 0.95`, 100 ms.
5. Tooltip follows cell position (not cursor) -- centred above the hovered cell.
6. On touch devices, tooltip appears on tap and dismisses on tap-outside.
7. Tooltip content is accessible: `role="tooltip"`, cell has
   `aria-describedby` linking to the tooltip ID.
8. When the user navigates cells via keyboard (arrows), the tooltip updates
   to reflect the focused cell.

---

### Story 7.3 -- Inline Asset Expansion with Zone Charts

**As a** trader,
**I want** clicking a heatmap cell to expand a detail row below the sector showing
buy/sell zone charts and forecast cards for that asset,
**so that** I can drill into signal detail without leaving the heatmap.

**Acceptance Criteria**

1. Clicking a cell (or pressing Enter on a focused cell) expands an inline
   panel below the sector row.
2. The panel slides open with height 0 to auto via `springSmooth` and content
   fades in with 100 ms delay.
3. Panel content:
   - `<BuySellZoneCharts>` showing the current zone distribution per horizon.
   - Forecast summary cards (one per horizon): expected return, P(up),
     signal label, Kelly -- each in a mini `<GlassCard>`.
   - A "View Full Chart" `<Button variant="secondary">` linking to
     `/charts/{symbol}`.
4. Zone charts use the application's colour tokens (emerald for buy zones,
   rose for sell zones, glass background).
5. If OHLCV data is loading, the panel shows `<CosmicSkeleton lines={3}>`.
6. Only one asset can be expanded at a time; expanding another collapses the
   previous.
7. Pressing Escape collapses the expanded panel and returns focus to the
   triggering cell.
8. The expanded panel has a left border (3 px) coloured by the asset's primary
   signal (emerald for buy, rose for sell, muted for hold).

---

### Story 7.4 -- Keyboard Navigation for Heatmap Grid

**As a** power user,
**I want** full keyboard navigation across the heatmap grid with arrow keys, Enter
to expand, and Escape to collapse,
**so that** I can operate the heatmap without touching the mouse.

**Acceptance Criteria**

1. Arrow keys (up/down/left/right) move the focus cursor across the grid.
2. The focused cell has a violet ring (`ring-2 ring-violet-500 ring-offset-1
   ring-offset-[var(--background)]`) visible as a bright outline.
3. `j` / `k` also work as down/up navigation (vim-style).
4. `Enter` or `Space` on a focused cell expands the inline detail panel.
5. `Escape` collapses an expanded panel and returns focus to the triggering cell.
6. `Home` moves focus to the first cell in the current row; `End` to the last.
7. `Ctrl+Home` moves to the top-left cell; `Ctrl+End` to the bottom-right.
8. Focus wraps: moving right from the last column moves to the first column
   of the next row.
9. The grid element has `role="grid"`, rows have `role="row"`, cells have
   `role="gridcell"`.
10. A screen reader announcement is made when focus enters a new sector group.

---

### Story 7.5 -- Heatmap Search, Filter, and Fullscreen Mode

**As a** trader,
**I want** to filter the heatmap by signal type and search for specific assets, and
toggle fullscreen mode for maximum data density,
**so that** I can focus on exactly the signals I need.

**Acceptance Criteria**

1. A toolbar above the heatmap contains:
   - Search input (glass background, Lucide Search icon, placeholder "Filter
     assets...", `Cmd+F` shortcut activates).
   - Signal filter dropdown (glass select, options: All, Strong Buy, Buy, Hold,
     Sell, Strong Sell).
   - Fullscreen toggle button (Lucide Maximize2 / Minimize2 icon).
2. Search filters the grid in real-time (100 ms debounce), hiding non-matching
   rows with a fade-out animation (150 ms).
3. Signal filter hides non-matching cells (cells fade to 10 % opacity, not
   removed) so grid structure is preserved.
4. Fullscreen mode:
   - Content expands to fill the viewport (sidebar and breadcrumb hidden).
   - `F` keyboard shortcut toggles fullscreen.
   - Transition: content scales from the current bounds to fullscreen via
     `springSmooth`.
   - Escape exits fullscreen.
5. Filter/search state persists in `localStorage`.
6. A summary strip above the grid shows: "Showing X of Y assets across Z
   sectors" with counts updating via `<CountUp>`.
7. Colour scale legend renders as a horizontal gradient bar (green to red) with
   labels at -10 %, 0 %, +10 %.
8. On mobile, the filter dropdown and search open in a bottom sheet instead
   of inline toolbar.


---

<!-- ============================================================ -->
<!-- EPIC 8                                                        -->
<!-- ============================================================ -->

## Epic 8 -- Charts Terminal

> The charts page is where traders spend the most time. The candlestick chart, overlay
> controls, and asset picker must feel like a professional trading terminal wrapped in
> the premium glass aesthetic.

### Story 8.1 -- Chart Sidebar Picker with Glass Sections and Signal Badges

**As a** trader,
**I want** the chart sidebar to show assets organised by sector with signal-coloured
badges, filtered views, and a premium glass aesthetic,
**so that** finding the right asset to chart is fast and visually clear.

**Acceptance Criteria**

1. The sidebar renders as a 280 px panel (collapsible to 0 px) with glass
   background and left border.
2. Top: a search input (glass, Lucide Search icon, `Cmd+K` focuses it).
3. Views toggle: horizontal glass pills (All, By Sector, Strong Buy, Strong
   Sell, Ranked).
4. **All view**: flat list of symbols, each with a 6 px signal-coloured dot
   (emerald for buy, rose for sell, amber for hold).
5. **Sector view**: collapsible sector groups (chevron + sector name + count
   badge). Expand/collapse animates with `springSnappy`.
6. **Strong Buy/Sell views**: filtered lists with `<SignalStrengthBar>` next to
   each symbol.
7. **Ranked view**: 8 ranking modes (momentum, edge, expected return, low risk,
   Kelly, P(up), forecast up, forecast down) as a dropdown selector. Each item
   shows a horizontal bar chart visualising the ranking metric.
8. Active (selected) symbol has a violet left bar (3 px) and `var(--glass-hover)`
   background.
9. Sidebar collapse animates width via `springSnappy`; `Cmd+B` toggles.
10. On mobile (< 768 px), the sidebar renders as a bottom sheet triggered by a
    floating pill button showing the current symbol.
11. URL synchronisation: selecting a symbol updates the URL to `/charts/{symbol}`
    without a full page reload.

---

### Story 8.2 -- Candlestick Chart Panel with Glass Chrome

**As a** trader,
**I want** the candlestick chart to render with a dark glass frame, premium grid lines,
and smooth data transitions,
**so that** the charting experience feels institutional-grade.

**Acceptance Criteria**

1. The chart panel fills the remaining width next to the sidebar, inside a
   `<GlassCard>` with subtle inner padding (16 px).
2. Chart header: symbol name in `<Heading level={2}>` + sector pill + signal
   badge + current price with `<CountUp>`.
3. Chart library: Lightweight Charts (TradingView) with custom theme:
   - Background: transparent (glass card shows through)
   - Grid lines: `rgba(255,255,255,0.03)` horizontal, `rgba(255,255,255,0.02)`
     vertical
   - Crosshair: `foreground-muted` with glass-background tooltip
   - Up candle: emerald (`#10b981`)
   - Down candle: rose (`#f43f5e`)
   - Volume bars: `rgba(139,92,246,0.2)` (violet tint)
4. Time range selector: glass pill bar (1M, 3M, 6M, 1Y, All) with sliding
   violet indicator via `layoutId`.
5. When the symbol changes, the chart crossfades (old data fades out, new data
   fades in) rather than abruptly replacing.
6. The chart height is responsive: minimum 400 px, grows to fill available
   viewport height.
7. Zoom: scroll-wheel zooms smoothly; pinch-to-zoom on touch devices.
8. Touch: drag to pan, double-tap to reset zoom.

---

### Story 8.3 -- Chart Overlay Toggle Panel with Keyboard Shortcuts

**As a** trader,
**I want** toggle buttons for chart overlays (SMA, Bollinger, RSI, Forecast) with
keyboard shortcuts and visual on/off states,
**so that** I can layer technical indicators without cluttering the UI.

**Acceptance Criteria**

1. Overlay toggles render as a horizontal row of glass pill buttons below
   the chart header.
2. Available overlays:
   - SMA 20 (shortcut `1`): moving average line in cyan
   - SMA 50 (shortcut `2`): moving average line in violet
   - SMA 200 (shortcut `3`): moving average line in amber
   - Bollinger Bands (shortcut `4`): upper/lower bands in `foreground-muted`
     at 40 % opacity, fill between at 4 %
   - RSI (shortcut `5`): sub-chart panel, 120 px tall, emerald/rose zones
   - Forecast Median (shortcut `6`): dashed violet line
   - Forecast CI (shortcut `7`): shaded violet band at 15 % opacity
   - Price Line (shortcut `p`): horizontal line at current price
3. Active toggle: filled with the overlay's colour at 15 % opacity, text in
   the overlay's colour, border brightens.
4. Inactive toggle: glass background, `foreground-secondary` text.
5. Toggle transition: `springSnappy` scale tap + colour crossfade.
6. The overlay data series animates in: line draws left-to-right over 600 ms;
   area fills fade in over 400 ms.
7. A "Reset" button clears all overlays.
8. Overlay state persists in `localStorage` per symbol.
9. On mobile, overlays render as a compact dropdown menu (glass select) to
   save horizontal space.
10. Keyboard shortcuts are shown as tiny badges on each pill (e.g. "1" in
    a 16 px circle, `foreground-muted`).

---

### Story 8.4 -- Buy/Sell Zone Charts below Main Chart

**As a** trader,
**I want** zone distribution charts to render below the main candlestick chart,
showing the signal engine's buy/sell zone boundaries per horizon,
**so that** I can see where the model places its conviction.

**Acceptance Criteria**

1. `<BuySellZoneCharts>` renders below the main chart panel, inside a `<GlassCard>`.
2. Each horizon (5D, 21D, 63D, 126D, 252D) shows a horizontal stacked bar:
   - Green zone (buy region) on the left.
   - Red zone (sell region) on the right.
   - Current price marker as a vertical white line with a diamond indicator.
3. Bars are labelled with the zone boundary values.
4. The bars animate in left-to-right with `springSmooth`, staggered by horizon
   (0.05 s each).
5. On hover over a zone, a tooltip shows: zone boundary, expected return, signal.
6. Horizon labels use `text-xs font-display uppercase tracking-wider`.
7. The component has a "View Full Analysis" link that scrolls to the forecasts
   section (if present).
8. When data is loading, a `<CosmicSkeleton lines={5} height={24}>` renders.
9. On mobile, zone charts stack vertically with full-width bars.
10. Each bar has a subtle glass border and rounded ends.


---

<!-- ============================================================ -->
<!-- EPIC 9                                                        -->
<!-- ============================================================ -->

## Epic 9 -- Tuning Mission Control

> The tuning page is Mission Control -- operators monitor retune processes, inspect model
> diagnostics per asset, and visualise the model ecosystem. Every element must reinforce
> the command-centre metaphor with terminal aesthetics and real-time feedback.

### Story 9.1 -- Retune Mission Control Panel Redesign

**As an** operator,
**I want** the retune control panel to look and feel like a premium mission control
console with mode selection, status indicators, and a streaming log terminal,
**so that** operating the retune process feels purposeful and professional.

**Acceptance Criteria**

1. The mission control panel renders at the top of the tuning page inside a
   `<GlassCard tint="violet">` with a `<ShimmerBorder>` when a retune is active.
2. Header: "Mission Control" in `<Heading level={2}>` with gradient text
   (violet-to-cyan).
3. Mode selector: three radio-style glass cards (Full Retune, Tune Only,
   Calibrate Failed), each with:
   - Icon (Lucide), label, description text.
   - Active state: violet border, tinted background, scale 1.02.
   - Inactive state: glass border, muted text.
   - Selection animates with `springSnappy`.
4. Action button:
   - Idle: "Start {Mode}" with `<Button variant="primary" size="lg">`.
   - Running: "Stop Retune" with `<Button variant="danger" size="lg">`,
     pulsing glow.
   - Completed: "Retune Complete" badge in emerald.
5. Status indicators: mode badge, elapsed time timer (counting up during
   retune), estimated progress percentage (if available from SSE).
6. Console toggle: a glass pill button "Show Logs" that expands the retune
   panel (see Story 9.2).
7. When retune is inactive, the shimmer border is hidden and the panel uses
   a standard glass border.

---

### Story 9.2 -- Streaming Log Terminal with Glass Chrome

**As an** operator,
**I want** the retune log terminal to render as a premium glass console with
colour-coded entries, auto-scroll, and copy functionality,
**so that** monitoring retune progress feels like using a high-end terminal.

**Acceptance Criteria**

1. The log terminal renders inside a `<GlassCard>` with `tint="none"` and a
   darker surface (`rgba(0,0,0,0.3)` overlay).
2. Terminal uses monospace font (`'JetBrains Mono', 'Fira Code', monospace`)
   at `text-xs`, `leading-relaxed`.
3. Log entries are colour-coded:
   - `[INFO]`: cyan text
   - `[SUCCESS]`: emerald text
   - `[WARNING]`: amber text
   - `[ERROR]`: rose text
   - `[PROGRESS]`: violet text
   - Timestamps: `foreground-muted`
4. New entries animate in: slide up from the bottom, 150 ms fade-in.
5. Auto-scroll: the terminal scrolls to the bottom on each new entry. A
   "scrolled away" indicator appears if the user scrolls up, with a "Jump to
   latest" button.
6. A progress bar (2 px, violet fill on glass track) renders at the top of
   the terminal, updating from SSE progress events.
7. Copy button (top-right, ghost variant) copies all log text to clipboard
   with a "Copied" toast confirmation.
8. Terminal height: 300 px default, resizable by dragging the bottom edge
   (glass resize handle, 8 px height).
9. Maximum 500 log entries in memory; older entries are pruned silently.
10. When no retune is active, the terminal shows the last retune's final
    status message or "No retune history" in `foreground-muted`.

---

### Story 9.3 -- Star Map Grid with Glass Cells and Status Colours

**As an** operator,
**I want** the tuning star map grid to render asset cells as premium glass tiles with
colour-coded PIT status and interactive selection,
**so that** I can scan 200+ assets' tuning status at a glance.

**Acceptance Criteria**

1. Star map grid renders as a responsive `flex-wrap` container of 32 x 32 px
   glass tiles.
2. Tile colours:
   - PIT pass: emerald background at 30 % opacity, emerald border.
   - PIT fail: rose background at 30 % opacity, rose border, subtle pulse
     animation (scale 1.0 to 1.05, 2 s period).
   - Unknown: glass background, muted border.
3. On hover: tile scales to 1.2 via `springSnappy`, a glass tooltip shows
   symbol name + PIT status + best model.
4. Selected tile: violet ring (2 px), scale 1.15, triggers the detail panel.
5. Tiles enter with a wave animation on page load: each tile fades in with
   stagger based on position (row * 5 ms + col * 5 ms).
6. Search filter: typing in the search input highlights matching tiles (full
   colour) and dims non-matching tiles (30 % opacity).
7. "Failures only" toggle dims passing tiles to 20 % opacity.
8. Grid / Table view toggle switches to the table view (Story 9.4) with a
   crossfade transition.
9. A summary bar below the grid shows pass/fail/unknown counts as a stacked
   bar chart similar to `<SignalDistributionBar>`.
10. On mobile, grid tiles shrink to 24 x 24 px to maintain density.

---

### Story 9.4 -- Model Distribution Chart Redesign

**As an** operator,
**I want** the model distribution chart to render as a premium Recharts visualisation
with glass frame and colour-coded model families,
**so that** I can see which model families dominate the ecosystem.

**Acceptance Criteria**

1. Chart wraps in a `<GlassCard>` with `<Heading level={3}>` "Model Distribution".
2. Horizontal bar chart (Recharts `<BarChart>` with `layout="vertical"`):
   - Each model family has a bar coloured by family: Kalman = violet,
     Phi = cyan, Momentum = emerald.
   - Bar corners are rounded (4 px on the right end).
   - Background: `var(--glass)` for bars' track.
3. Chart theme inherits design tokens:
   - Grid lines: `rgba(255,255,255,0.04)`
   - Axis text: `foreground-muted`, `text-xs`
   - Tooltip: glass card with 12 px blur, violet border
4. A treemap sub-chart shows asset-level model allocation as coloured
   rectangles grouped by family.
5. Both charts animate in: bars grow from 0 width on viewport entry;
   treemap tiles fade in with stagger.
6. On hover over a bar, the corresponding treemap cells highlight.
7. Model family legend: horizontal pill row above the chart.
8. Data source: `api.tuneStats` and `api.tuneList`.
9. On mobile, the treemap hides and only the bar chart renders.
10. Empty state: `<CosmicEmptyState>` with "Run a tuning cycle to see model
    distribution".


---

<!-- ============================================================ -->
<!-- EPIC 10                                                       -->
<!-- ============================================================ -->

## Epic 10 -- Arena and Diagnostics

> Arena is where models compete and diagnostics reveal calibration health. These pages
> serve analytical operators who need dense, precise data presented in a glass-morphic,
> navigable interface.

### Story 10.1 -- Arena Gate Scoring Table with Glass Rows and Badge System

**As an** analyst,
**I want** the arena safe-storage table to render with glass rows, gate-status badges,
and expandable detail panels,
**so that** I can evaluate model quality and promotion readiness at a glance.

**Acceptance Criteria**

1. Table renders inside a `<GlassCard>` with header columns: Rank, Model Name,
   Final Score, BIC, CRPS, Hyvarinen, PIT, CSS, FEC, Time, Size.
2. Row styles:
   - Rank 1 (champion): `<ShimmerBorder>` around the row, amber crown icon,
     `tint="amber"` on the row's glass background.
   - Top 3: subtle violet left border.
   - Others: default glass alternation.
3. Gate badges (`<GateBadge>`):
   - Pass: emerald circle with checkmark icon.
   - Borderline: amber circle with minus icon.
   - Fail: rose circle with X icon.
   - Each badge has a tooltip showing the threshold: e.g. "BIC < 100: Pass".
4. Row hover: glass-hover background, row lifts 2 px.
5. Row click expands a detail panel:
   - All gate metrics as labelled stat tiles in a grid.
   - Pass/fail status per gate.
   - Model parameters summary.
   - Animated height 0 to auto via `springSmooth`.
6. Only one row expanded at a time.
7. A "Hard Gates" reference card renders in the sidebar (or below the table on
   mobile) listing all 8 promotion criteria with their thresholds.
8. Stat cards above the table: Safe Storage count, Experimental count,
   Benchmark count -- using `<GlassCard>` + `<CountUp>`.
9. Benchmark universe renders as a pill cloud (glass capsules, `flex-wrap`).
10. All numeric values are right-aligned with monospace font (`font-variant-
    numeric: tabular-nums`).

---

### Story 10.2 -- Diagnostics Page Tab System and PIT Table

**As an** analyst,
**I want** the diagnostics page to use a premium tabbed interface with the PIT
calibration table as the primary view,
**so that** I can navigate between diagnostic dimensions without losing context.

**Acceptance Criteria**

1. Tab bar uses the same glass-pill design from Story 6.3 with 5 tabs:
   PIT Calibration, Model Comparison, Regimes, Calibration Failures, Cross-Asset
   Matrix.
2. PIT tab (default):
   - Stat cards: Total Assets, PIT Pass, PIT Fail, Calibration Failures.
   - Searchable table with columns: Symbol, Best Model, PIT Status, AD Stat,
     Grade, Models, Regime.
   - PIT status renders as a `<GateBadge>` (pass/fail).
   - Grade column uses letter grades (A-F) with colour coding.
   - Regime column uses a coloured pill (calm/elevated/stressed/crisis).
3. Row expansion shows a `<ModelMetricsTable>` sub-table with BIC, CRPS,
   Hyvarinen, PIT p-value, AD p-value, MAD, Weight, nu for each model.
4. PIT filter toggle (All / Pass Only / Fail Only) uses the glass-pill pattern.
5. Search input with live filtering (100 ms debounce).
6. Table header is sticky with `var(--surface)` background.
7. Row alternation with subtle glass tinting.
8. All numeric columns are right-aligned with `tabular-nums`.
9. Rows enter with `<StaggerChildren stagger={0.02}>` on filter/search change.
10. Empty filtered result: `<CosmicEmptyState>` with contextual message.

---

### Story 10.3 -- Model Comparison Charts and Cross-Asset Matrix

**As an** analyst,
**I want** the model comparison and cross-asset matrix tabs to render with premium
Recharts visualisations and interactive metric selection,
**so that** I can compare model performance across every dimension.

**Acceptance Criteria**

1. **Model Comparison tab**:
   - Two horizontal bar charts side by side (stacked on mobile):
     a) Win Count by model (sorted descending)
     b) Average BMA Weight by model (sorted descending)
   - Charts use Recharts with design-token theming (see Story 9.4 criteria 3).
   - Bars colour-coded by model family.
   - Full model comparison table below with all metrics.
2. **Cross-Asset Matrix tab**:
   - Model averages summary table at the top.
   - Metric selector: glass-pill bar with CRPS, PIT p-value, Weight.
   - Matrix grid: assets as rows, models as columns, cells colour-coded by
     metric value (green for good, red for bad).
   - Cell hover shows tooltip with exact value.
   - Matrix renders inside a `<GlassCard>` with horizontal scroll and custom
     scrollbar.
3. **Regimes tab**:
   - Recharts `<PieChart>` showing regime distribution with glass-card wrapper.
   - Legend with coloured dots and labels.
   - Regime duration metrics below the chart.
4. **Failures tab**:
   - List of calibration failures as glass cards.
   - Each failure card: asset name, failure reason, suggested action.
   - Cards enter with `<StaggerChildren>`.
5. All chart transitions: bars grow from 0 on viewport entry, pie slices
   expand from centre, matrix cells fade in with wave stagger.

---

### Story 10.4 -- Profitability Monitoring Page Redesign

**As an** analyst,
**I want** the profitability page to render metric cards with pass/fail glow indicators
and line charts with target overlays in premium Recharts styling,
**so that** I can monitor signal quality over time.

**Acceptance Criteria**

1. Page header: "Profitability Monitor" in `<Heading level={1}>` with gradient
   text.
2. Six metric cards in a responsive grid (`grid-cols-1 sm:grid-cols-2 lg:grid-cols-3`):
   - Each card is a `<GlassCard>` with:
     - Metric name label.
     - Current value as `<CountUp>` in `text-2xl font-display`.
     - Pass/fail indicator: emerald glow border if passing, rose glow if failing.
     - Target value shown in `text-xs foreground-muted` below the value.
3. Six line charts (one per metric) in a matching grid:
   - Each chart in a `<GlassCard>`.
   - Recharts `<LineChart>` with design-token theming.
   - Data line: violet stroke (2 px).
   - Target reference line: dashed amber horizontal line with label.
   - Area below target: subtle rose fill at 5 % opacity where below target;
     emerald fill where above.
   - X-axis: time labels in `foreground-muted`.
   - Y-axis: metric values with appropriate formatting.
4. Charts animate in: line draws left-to-right on viewport entry (800 ms).
5. Metric cards and charts enter with `<StaggerChildren stagger={0.06}>`.
6. On hover over a chart data point, a glass tooltip shows: date, value,
   delta from target.
7. A "Last updated" timestamp renders at the top-right.
8. On mobile, the grid collapses to single-column and charts render at
   full width.
9. If data is unavailable, show `<CosmicEmptyState>` with message
   "No profitability data available. Run a signals cycle first."


---

<!-- ============================================================ -->
<!-- EPIC 11                                                       -->
<!-- ============================================================ -->

## Epic 11 -- Data and Services Operations

> These pages serve operators monitoring system health -- data freshness, service
> uptime, and error logs. They must be clear, glanceable, and trustworthy with
> auto-refreshing indicators.

### Story 11.1 -- Data Management Page Redesign

**As an** operator,
**I want** the data page to show file statistics, directory health, and a price-file
table in a premium glass layout with freshness colour coding,
**so that** I can monitor data health at a glance and trigger refreshes confidently.

**Acceptance Criteria**

1. Stat cards (4 items: Total Files, Fresh, Stale, Disk Usage) in a responsive
   grid using `<GlassCard>` + `<CountUp>`.
   - Fresh card: `tint="emerald"`, Stale card: `tint="rose"`.
2. Data Directories section:
   - Each directory renders as a row in a `<GlassCard>`:
     - Directory path in `font-mono text-sm`.
     - Status badge: "Exists" (emerald) or "Missing" (rose) `<GateBadge>`.
     - Size and file count (if exists).
3. Price Files table inside a `<GlassCard>`:
   - Columns: Symbol, Rows, Size, Age, Last Modified.
   - Age column uses freshness badges:
     - Fresh (< 24 h): emerald pill.
     - Stale (24-72 h): amber pill.
     - Old (> 72 h): rose pill.
   - Table header sticky, sortable columns with sort indicators per Story 5.5.
   - Search input with live filtering.
4. Refresh button: `<Button variant="primary" icon={RefreshCw}>` "Refresh Data".
   - On click: button shows loading spinner; on completion, success toast.
   - If error: danger toast with error message.
5. Stat cards and table enter with `<StaggerChildren>`.
6. Auto-refresh indicator: "Updated X seconds ago" in `text-xs foreground-muted`,
   counting up live.
7. On mobile, stat cards render 2-column; table gains horizontal scroll.
8. Table row hover: glass-hover background, row lifts 1 px.

---

### Story 11.2 -- Services Health Dashboard Redesign

**As an** operator,
**I want** the services page to render as a premium health dashboard with service
cards, status indicators, and an error log panel,
**so that** system health is instantly readable and issues are surfaced prominently.

**Acceptance Criteria**

1. Hero status banner at the top:
   - "All Systems Operational": emerald gradient glow behind text, emerald dot.
   - "Issues Detected": rose gradient glow, pulsing rose dot.
   - Text in `<Heading level={2}>` with gradient text matching status colour.
2. Four service cards in a responsive grid (`grid-cols-1 sm:grid-cols-2`),
   each as a `<GlassCard>`:
   - **API Server**: uptime percentage (emerald if > 99.5%), memory usage bar,
     CPU % bar, PID.
   - **Signal Cache**: exists/missing badge, age (freshness badge), size, last
     modified timestamp.
   - **Price Data**: file count, stale file count (rose if > 0), freshest file
     timestamp, total size.
   - **Background Workers**: Redis status pill (emerald/rose), Redis memory bar,
     Celery status pill, worker count.
3. Each usage bar renders inside a glass track with a fill coloured by usage
   level (emerald < 60%, amber 60-80%, rose > 80%).
4. Cards auto-refresh every 10 s; a subtle pulse animation on the card border
   indicates refresh (200 ms, barely perceptible).
5. Error Log panel below the cards:
   - `<GlassCard>` with `<Heading level={3}>` "Recent Errors".
   - Entries: timestamp, source badge (coloured by service), message.
   - Empty state: emerald checkmark + "No recent errors" message.
   - Maximum 20 entries; auto-refreshes every 15 s.
6. All cards enter with `<StaggerChildren stagger={0.08}>`.
7. Manual refresh button in the header bar.
8. On mobile, service cards stack full-width.

---

### Story 11.3 -- Toast Notification System Redesign

**As a** user,
**I want** toast notifications to render as premium glass capsules with icons,
auto-dismiss, and stacked positioning,
**so that** feedback for actions (retune started, data refreshed) is noticeable
but non-intrusive.

**Acceptance Criteria**

1. Toasts render in a fixed container at the bottom-right (top-right on mobile),
   stacked vertically with 8 px gap.
2. Each toast is a glass capsule (320 px max-width, `backdrop-filter: blur(12px)`,
   `border-radius: 12px`):
   - Success: emerald left bar (3 px), Lucide CheckCircle icon.
   - Error: rose left bar, Lucide XCircle icon.
   - Warning: amber left bar, Lucide AlertTriangle icon.
   - Info: cyan left bar, Lucide Info icon.
3. Toast enter animation: slide-in from the right (24 px) + fade-in, `springSnappy`.
4. Toast exit: slide-out to the right + fade-out, 200 ms.
5. Auto-dismiss: 5 s default, configurable per toast. A progress bar (2 px,
   themed colour) at the bottom counts down.
6. Hover pauses the auto-dismiss timer and progress bar.
7. Close button (X icon, ghost) in the top-right corner.
8. Maximum 5 visible toasts; older toasts are dismissed to make room.
9. Stacking animation: when a toast is dismissed, remaining toasts slide down
   via `layoutId` animation.
10. Toasts are announced to screen readers via `role="alert"` and
    `aria-live="polite"`.

---

### Story 11.4 -- Export Button with Format Selection Overlay

**As a** user,
**I want** the export button to open a glass overlay with format options (CSV, JSON)
and export scope selection,
**so that** data export feels premium rather than a bare file download.

**Acceptance Criteria**

1. `<ExportButton>` renders as `<Button variant="secondary" icon={Download}>`.
2. Click opens a glass overlay positioned below the button (popover style):
   - Glass background with 16 px blur, 12 px radius, `<ShimmerBorder>`.
   - Format selection: two glass cards (CSV, JSON) with radio selection.
   - Scope selection (if applicable): "All Data", "Filtered Only", "Current
     View" as glass pills.
   - "Export" `<Button variant="primary">` at the bottom.
3. On export: button shows loading spinner, file downloads, success toast.
4. Overlay enter: scale 0.95 to 1, opacity 0 to 1, `springSnappy`.
5. Overlay exit: scale 1 to 0.95, opacity 1 to 0, 150 ms.
6. Escape closes the overlay; click-outside closes the overlay.
7. Keyboard navigation: tab between format cards, arrow between scope pills,
   Enter triggers export.
8. Overlay is positioned with collision detection (flips above the button
   if insufficient space below).


---

<!-- ============================================================ -->
<!-- EPIC 12                                                       -->
<!-- ============================================================ -->

## Epic 12 -- Responsive and Accessibility

> Premium UX means every user can access every feature, regardless of device, input
> method, or ability. This epic ensures the redesigned app works beautifully on mobile,
> responds to system accessibility preferences, and passes automated accessibility audits.

### Story 12.1 -- Mobile-First Responsive Layout Overhaul

**As a** mobile user,
**I want** every page to render beautifully on a 375 px viewport with optimised touch
targets, stacked layouts, and responsive typography,
**so that** I can monitor signals and risk on my phone.

**Acceptance Criteria**

1. All component grids implement mobile-first breakpoints:
   - `grid-cols-1` at base, expanding at `sm`, `md`, `lg`, `xl`.
2. Glass blur scales: 8 px on mobile (< 640 px), 12 px on desktop.
3. Sidebar:
   - On screens < 768 px, renders as a slide-over drawer (from left, 280 px).
   - Trigger: hamburger button in a fixed top bar (48 px height, glass
     background).
   - Backdrop overlay: `rgba(0,0,0,0.6)`, click-to-close.
4. Tables on screens < 768 px:
   - Priority columns visible; secondary columns hidden behind a horizontal
     scroll with right-edge fade gradient.
   - Or: switch to card-based layout where each row becomes a glass card.
5. Charts:
   - Full-width on mobile with 16 px horizontal padding.
   - Minimum height 280 px.
   - Touch: drag to pan, pinch to zoom, double-tap to reset.
6. Touch targets: all interactive elements have a minimum 44 x 44 px hit area.
7. Safe area support: content respects `env(safe-area-inset-*)` for notched
   devices.
8. Typography scales:
   - Page headings: `text-xl` on mobile, scaling up at breakpoints.
   - Body: `text-sm` on mobile, `text-base` on desktop.
9. Status strip: collapses to minimal (dots only) on mobile with tap-to-expand.
10. Command palette: full-screen overlay on mobile instead of centred modal.

---

### Story 12.2 -- Reduced Motion and Prefers-Contrast Support

**As a** user with motion sensitivity,
**I want** all animations to be disabled or minimised when I have
`prefers-reduced-motion` enabled, and contrast to increase when I have
`prefers-contrast: more` enabled,
**so that** the app is comfortable and usable for me.

**Acceptance Criteria**

1. Global CSS rule:
   ```css
   @media (prefers-reduced-motion: reduce) {
     *, *::before, *::after {
       animation-duration: 0.01ms !important;
       animation-iteration-count: 1 !important;
       transition-duration: 0.01ms !important;
       scroll-behavior: auto !important;
     }
   }
   ```
2. All Framer Motion components check `useReducedMotion()`:
   - `<FadeIn>`: renders children immediately at full opacity.
   - `<StaggerChildren>`: renders all children simultaneously.
   - `<CountUp>`: renders final value immediately.
   - `<CosmicBackground>`: orbs render at midpoint positions, no drift.
   - `<ShimmerBorder>`: gradient pauses at 0 deg.
   - Spring transitions on hover/tap use `duration: 0` instead of springs.
3. Sparkline draw-in animation is skipped; line renders fully drawn.
4. Gauge needle snaps to value without spring animation.
5. Heatmap wave entrance is skipped; all cells render simultaneously.
6. High-contrast mode (`prefers-contrast: more`):
   - Glass backgrounds increase to `rgba(255,255,255,0.08)`.
   - Glass borders increase to `rgba(255,255,255,0.16)`.
   - Text colours shift to pure white (`#ffffff`) for primary and `#d4d4d8`
     for secondary.
   - Focus rings widen to 3 px.
7. Both media queries are tested with Chrome DevTools emulation on every page.

---

### Story 12.3 -- ARIA Landmarks, Roles, and Screen Reader Announcements

**As a** screen reader user,
**I want** proper ARIA landmarks, roles, and live-region announcements throughout
the app,
**so that** I can navigate and understand the dashboard using assistive technology.

**Acceptance Criteria**

1. Page structure uses semantic landmarks:
   - Sidebar: `<nav aria-label="Main navigation">`.
   - Content: `<main>` element.
   - Status strip: `<footer>`or `<aside aria-label="System status">`.
   - Breadcrumb: `<nav aria-label="Breadcrumb">`.
2. Each page's primary heading is `<h1>` (one per page).
3. Data tables use:
   - `<table>` with `<caption>` describing the table content.
   - `<thead>`, `<tbody>` structure.
   - Sortable headers: `aria-sort="ascending" | "descending" | "none"`.
4. Live data updates use `aria-live="polite"`:
   - WebSocket signal changes announce "Signal updated: {symbol} changed to
     {signal}".
   - Retune progress announces "Retune progress: {percent}%".
   - Service status changes announce when a service goes down/up.
5. Modal dialogs (command palette, export overlay) use:
   - `role="dialog"`, `aria-modal="true"`, `aria-labelledby`.
   - Focus trap with tab/shift-tab cycling.
   - Escape to close with focus return.
6. Charts have `aria-label` descriptions: "Candlestick chart for {symbol}
   showing price data from {startDate} to {endDate}".
7. Colour-only indicators (heatmap cells, gate badges) have text alternatives:
   - Heatmap: `aria-label="{symbol}: expected return {value}%, signal {label}"`.
   - Gate badge: `aria-label="{gate}: {status}"`.
8. Icon-only buttons have `aria-label` text (e.g. "Collapse sidebar",
   "Refresh data", "Toggle fullscreen").
9. Automated axe-core scan of every route reports zero critical or serious
   violations.

---

### Story 12.4 -- Keyboard Shortcuts System with Discoverable Help Overlay

**As a** power user,
**I want** a comprehensive keyboard shortcut system with a discoverable help overlay,
**so that** I can navigate the entire app at keystroke speed.

**Acceptance Criteria**

1. Global shortcuts:
   - `Cmd+K` / `Ctrl+K`: open command palette.
   - `Cmd+B` / `Ctrl+B`: toggle sidebar.
   - `?`: open keyboard shortcuts help overlay.
   - `g` then `o`: go to Overview.
   - `g` then `s`: go to Signals.
   - `g` then `r`: go to Risk.
   - `g` then `h`: go to Heatmap.
   - `g` then `c`: go to Charts.
   - `g` then `t`: go to Tuning.
   - `g` then `d`: go to Data.
   - `g` then `a`: go to Arena.
   - `g` then `x`: go to Diagnostics.
   - `g` then `v`: go to Services.
2. Page-specific shortcuts are documented per page (chart overlays 1-8, p;
   heatmap arrows, j/k; signals Cmd+F, etc.).
3. Help overlay renders as a full-screen glass modal with categorised shortcut
   tables:
   - Navigation, Signals, Charts, Heatmap, Tuning sections.
   - Each shortcut shows: key combo in a `<kbd>` styled pill (glass background,
     monospace) and description.
4. Shortcuts are disabled when an input field is focused (except Escape and
   Cmd+K).
5. `g` then `{key}` combos use a 500 ms timeout: if the second key is not
   pressed within 500 ms, the combo resets.
6. A small "?" indicator in the status strip links to the shortcuts overlay.
7. Overlay enter: fade-in + scale 0.95 to 1, `springSnappy`.
8. Overlay exit: fade-out + scale 1 to 0.95, 150 ms.

---

### Story 12.5 -- Performance Budget and Lazy Loading Strategy

**As a** user on a slower connection,
**I want** pages to load fast with lazy-loaded heavy components and a strict
performance budget,
**so that** the premium visuals never come at the cost of responsiveness.

**Acceptance Criteria**

1. Performance budget (measured via Lighthouse on a simulated 4G connection):
   - First Contentful Paint: < 1.5 s
   - Largest Contentful Paint: < 2.5 s
   - Time to Interactive: < 3.5 s
   - Cumulative Layout Shift: < 0.05
   - Total Bundle Size (gzipped): < 350 KB (initial route)
2. Code splitting: each page route is `React.lazy()` with a `<Suspense>`
   fallback rendering `<CosmicSkeleton>`.
3. Heavy components are lazy-loaded:
   - Recharts (bar, line, pie charts): loaded on demand per page.
   - Lightweight Charts (TradingView): loaded only on `/charts`.
   - `<CommandPalette>`: loaded on first `Cmd+K` invocation.
   - `<KeyboardShortcutOverlay>`: loaded on first `?` press.
4. Font loading: Inter and Space Grotesk use `font-display: swap`; WOFF2
   format only; preloaded via `<link rel="preload">` for the two most
   common weights (400, 700).
5. Images: all icons are inline SVGs or Lucide components (no raster images
   in the core UI).
6. CSS: the design-system tokens layer (`_tokens.css`) is inlined in the HTML
   `<head>` to prevent FOUC.
7. React Query: stale-while-revalidate caching prevents unnecessary network
   requests on tab switches.
8. Skeleton placeholders match the dimensions of their loaded content to
   prevent layout shift.
9. Bundle analysis: `vite-plugin-bundle-analyzer` is added to the dev config;
   any chunk exceeding 100 KB triggers a console warning.
10. A CI performance check runs Lighthouse against the production build and
    fails the pipeline if any metric exceeds the budget by more than 10 %.

---

## Appendix A -- Component Inventory

| Component | Epic | Status |
|-----------|------|--------|
| `CosmicBackground` | 1 | New |
| `GlassCard` | 2 | Rewrite |
| `ShimmerBorder` | 2 | New |
| `Button` | 2 | Rewrite |
| `FadeIn` | 2 | New (replaces useScrollReveal) |
| `StaggerChildren` / `StaggerItem` | 2 | New |
| `CountUp` | 2 | Rewrite (replaces useCountUp) |
| `CosmicSkeleton` | 2 | Rewrite |
| `CosmicEmptyState` | 2 | Rewrite |
| `CosmicErrorState` | 2 | Rewrite |
| `Heading` | 1 | New |
| `Text` | 1 | New |
| `Layout` (sidebar) | 3 | Rewrite |
| `BreadcrumbBar` | 3 | Rewrite |
| `StatusStrip` | 3 | Rewrite |
| `CommandPalette` | 3 | Rewrite |
| `BriefingCard` | 4 | Rewrite |
| `StatCard` | 4 | Rewrite (wraps GlassCard) |
| `SignalDistributionBar` | 4 | Rewrite |
| `ModelLeaderboard` | 4 | Rewrite |
| `ConvictionSpotlight` | 4 | Rewrite |
| `Sparkline` | 5 | Rewrite |
| `MomentumBadge` | 5 | Rewrite |
| `SignalStrengthBar` | 5 | New |
| `HighConvictionCard` | 5 | Rewrite |
| `ChangeLog` | 5 | New |
| `CosmicGauge` | 6 | Rewrite |
| `BuySellZoneCharts` | 7/8 | Rewrite |
| `ChartPanel` | 8 | Rewrite |
| `RetunePanel` | 9 | Rewrite |
| `GateBadge` | 10 | Rewrite |
| `ModelMetricsTable` | 10 | Rewrite |
| `ToastContainer` | 11 | Rewrite |
| `ExportButton` | 11 | Rewrite |
| `KeyboardShortcutOverlay` | 12 | Rewrite |

## Appendix B -- Dependency Additions

| Package | Version | Purpose |
|---------|---------|---------|
| `framer-motion` | ^12 | Spring physics, AnimatePresence, layout animations |
| (existing) `recharts` | -- | Themed chart visualisations |
| (existing) `lightweight-charts` | -- | Candlestick terminal |
| (existing) `lucide-react` | -- | Icon library |

## Appendix C -- Design Token Migration Checklist

- [ ] `:root` tokens match lumen-lingo-frontend exactly
- [ ] Font files self-hosted (Inter WOFF2, Space Grotesk WOFF2)
- [ ] Motion tokens exported from `motion.ts`
- [ ] CSS utility classes for glass, text, and spacing documented
- [ ] Focus-ring system applied globally
- [ ] Reduced-motion system verified on every page
- [ ] Colour-contrast AA verified for all token pairings
