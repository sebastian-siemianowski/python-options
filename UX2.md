# UX2.md -- Premium Visual Overhaul Specification

**Target Benchmark**: LumenLingo.com -- "Every pixel considered"
**Date**: April 2026
**Status**: Draft

---

## Design Philosophy

LumenLingo achieves its premium feel through five pillars that this specification
adapts to a quantitative trading dashboard:

1. **Glass-morphic Depth** -- Every surface has real optical weight: backdrop-filter blur,
   inset shadows, luminous edge highlights, and layered translucency.
2. **Physics-Based Motion** -- Interactions use spring curves (cubic-bezier 0.16, 1, 0.3, 1),
   not linear easing. Elements overshoot and settle. Entrances blur-to-sharp.
3. **Multi-Sensory Feedback** -- Hover states combine colour shift + shadow lift + subtle
   scale transform. Active states compress. Focus states glow.
4. **Typographic Hierarchy** -- A strict 6-tier scale with defined letter-spacing, weight,
   and colour per tier. No ad-hoc font-size values.
5. **Colour as Information** -- Every colour carries meaning. Hardcoded hex values are
   eliminated; all colours reference CSS custom properties with semantic names.
6. **Celebration & Delight** -- Success moments are rewarded with visual payoff:
   confetti particles on tuning completion, emerald glow bursts on calibration pass,
   number counting reveals on data load. The app celebrates your wins.
7. **Textural Depth** -- Surfaces have physical presence: subtle noise grain on
   glass-cards, inner light edges that catch ambient light, layered shadows that
   create real optical distance between stacked elements.

---

## Terminology

- **AC** = Acceptance Criterion
- **Page Rating** = Current premium score 1--10, target is 9+
- **CSS var** = CSS custom property defined in `:root` or `@theme` in index.css

---

## Epic Index

| # | Epic | Scope | Stories |
|---|------|-------|---------|
| E1 | Global Design System | index.css, tokens, utilities | 11 stories |
| E2 | Overview Page | OverviewPage.tsx, BriefingCard.tsx, StatCard.tsx | 7 stories |
| E3 | Risk Dashboard | RiskPage.tsx | 5 stories |
| E4 | Signals Page | SignalsPage.tsx, SignalTableVisuals.tsx | 8 stories |
| E5 | Data Management | DataPage.tsx | 5 stories |
| E6 | Arena Competition | ArenaPage.tsx | 5 stories |
| E7 | Diagnostics & Calibration | DiagnosticsPage.tsx | 7 stories |
| E8 | Charts & Technical Analysis | ChartsPage.tsx | 5 stories |
| E9 | Heatmap & Sentiment | HeatmapPage.tsx | 6 stories |
| E10 | Tuning Mission Control | TuningPage.tsx | 6 stories |
| E11 | Services & Health | ServicesPage.tsx | 5 stories |
| E12 | Profitability Analytics | ProfitabilityPage.tsx | 7 stories |
| E13 | Shared Components | Layout, PageHeader, LoadingSpinner, etc. | 6 stories |

---

## E1: Global Design System

**Current Rating**: 7/10
**Target Rating**: 9.5/10
**Files**: `src/index.css`

The design system CSS has excellent foundations (animated mesh gradient, glass-card,
cosmic-row) but lacks a strict token contract. 50%+ of component styles bypass CSS
variables with hardcoded hex/rgba values. There is no typographic scale, no spacing
scale, and interactive states are inconsistently applied.

---

### S1.1: Typographic Scale System

**As a** developer,
**I want** a defined 6-tier typographic scale in CSS custom properties,
**so that** every text element references a scale token instead of ad-hoc pixel values.

**AC-1**: Set the global font stack to Inter (with system fallback) for its
excellent tabular figures, optical sizing, and variable weight support:
```css
--font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
body { font-family: var(--font-family); -webkit-font-smoothing: antialiased; }
```

**AC-2**: Set the global font stack to Inter (with system fallback) for its
excellent tabular figures, optical sizing, and variable weight support:
```css
--font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
body { font-family: var(--font-family); -webkit-font-smoothing: antialiased; }
```

**AC-2**: Define the following CSS custom properties in `:root`:
```
--font-2xs: 10px    (labels, badges, table headers)
--font-xs:  11px    (secondary text, captions)
--font-sm:  12px    (body small, table cells)
--fon3-base: 14px   (body text, inputs)
--font-lg:  16px    (section titles)
--font-xl:  20px    (card values, emphasis)
--font-2xl: 28px    (page stat values)
--font-3xl: 32px    (page headings)
```

**AC-3**: Define letter-spacing tokens:
```
--tra4king-tight:  -0.03em  (headings)
--tracking-normal:  0em     (body)
--tracking-wide:    0.06em  (small caps, nav)
--tracking-wider:   0.10em  (table headers, section labels)
--tracking-widest:  0.14em  (stat titles, badges)
```

**AC-4**: Define font-weight tokens:
```
--wei5ht-normal: 400
--weight-medium: 500
--weight-semibold: 600
--weight-bold: 700
--weight-extrabold: 800
```

**AC-5**: Create utility classes that compose these tokens:
```css
.text-heading    { font-size: var(--font-3xl); font-weight: var(--weight-bold); letter-spacing: var(--tracking-tight); }
.text6stat-value { font-size: var(--font-2xl); font-weight: var(--weight-bold); letter-spacing: var(--tracking-tight); font-variant-numeric: tabular-nums; }
.text-section    { font-size: var(--font-lg); font-weight: var(--weight-semibold); }
.text-body       { font-size: var(--font-base); font-weight: var(--weight-normal); }
.text-caption    { font-size: var(--font-xs); font-weight: var(--weight-medium); color: var(--text-secondary); }
.text-label      { font-size: var(--font-2xs); font-weight: var(--weight-semibold); text-transform: uppercase; letter-spacing: var(--tracking-widest); color: var(--text-muted); }
```

**AC-6**: No component in the codebase uses a raw pixel font-size value -- all sizes
reference a scale token or Tailwind equivalent mapped to the scale.

---

### S1.2: Spacing Scale Enforcement

**As a** developer,
**I want** a consistent spacing scale used across all padding, margin, and gap values,
**so that** the interface has a rhythmic vertical and horizontal flow.

**AC-1**: Adopt the 4px base grid:
```
--space-0:  0px
--space-1:  4px
--space-2:  8px
--space-3:  12px
--space-4:  16px
--space-5:  20px
--space-6:  24px
--space-7:  28px
--space-8:  32px
--space-10: 40px
--space-12: 48px
--space-16: 64px
```

**AC-2**: All inline `style={{ padding: 'Xpx' }}` replaced with Tailwind classes or
CSS variable references from this scale.

**AC-3**: Grid gaps normalised:
- Stat card grids: `gap-5` (20px)
- Section spacing (margin-bottom between sections): `mb-8` (32px)
- Card internal padding: `p-6` (24px) on all glass-card elements

**AC-4**: No raw pixel values (e.g. `padding: '20px'`, `gap: '1.5rem'`) remain outside
the index.css design system file.

---

### S1.3: Colour Variable Consolidation

**As a** developer,
**I want** every hardcoded hex, rgb, and rgba colour in component files replaced with
CSS custom property references,
**so that** theming is centralised and palette changes propagate everywhere.

**AC-1**: Audit and replace all hardcoded colour values in `.tsx` files:
- `#3ee8a5` replaced with `var(--accent-emerald)`
- `#ff6b8a` replaced with `var(--accent-rose)`
- `#f5c542` replaced with `var(--accent-amber)`
- `#8b5cf6` replaced with `var(--accent-violet)`
- `#b49aff` replaced with `var(--text-violet)`
- `#7a8ba4` replaced with `var(--text-muted)`
- `#e2e8f0` replaced with `var(--text-primary)`
- `#94a3b8` replaced with `var(--text-secondary)`
- `#f97316` replaced with `var(--accent-orange)` (add to palette)
- `#2a2a4a` replaced with `var(--border-void)`

**AC-2**: All `rgba(139,92,246,0.XX)` patterns use a single CSS property with opacity
modifiers, or are defined as named semantic tokens:
```
--glass-border:     rgba(139,92,246,0.08)
--glass-border-hover: rgba(139,92,246,0.15)
--glass-surface:    rgba(139,92,246,0.04)
--glass-glow:       rgba(139,92,246,0.06)
```

**AC-3**: Each colour has a semantic role documented in a comment block:
```css
/* Accent Palette -- semantic meaning */
--accent-emerald: #3ee8a5;  /* Pass, buy, positive, healthy */
--accent-rose:    #ff6b8a;  /* Fail, sell, negative, danger */
--accent-amber:   #f5c542;  /* Warning, caution, stale */
--accent-violet:  #8b5cf6;  /* Brand, interactive, focus */
--accent-orange:  #f97316;  /* Elevated, moderate risk */
```

**AC-4**: Zero hardcoded hex/rgb values remain in any `.tsx` file (verified via grep).

---

### S1.4: Interactive State System

**As a** developer,
**I want** a unified set of CSS classes for hover, active, focus, and disabled states,
**so that** every interactive element has consistent, premium feedback.

**AC-1**: Define the signature hover-lift with 3D perspective tilt and border
luminance -- the single most impactful drool-worthy effect. When hovering a card,
the surface tilts subtly toward the cursor and an edge light appears:
```css
.hover-lift {
  transition: transform 300ms cubic-bezier(0.16,1,0.3,1),
              box-shadow 300ms cubic-bezier(0.16,1,0.3,1),
              border-color 300ms cubic-bezier(0.16,1,0.3,1);
  transform-style: preserve-3d;
  perspective: 800px;
}
.hover-lift:hover {
  transform: translateY(-3px) rotateX(1deg) rotateY(-0.5deg);
  box-shadow: var(--shadow-card-hover), 0 20px 40px rgba(0,0,0,0.15);
  border-color: rgba(139,92,246,0.18);
}
```
A lightweight JS utility `tilt.ts` tracks cursor position within each `.hover-lift`
element and sets `--tilt-x` / `--tilt-y` CSS variables for per-card directional
tilt (max +/-2deg). The CSS reads:
```css
.hover-lift:hover {
  transform: translateY(-3px) rotateX(var(--tilt-x, 1deg)) rotateY(var(--tilt-y, -0.5deg));
}
```

**AC-2**: Define border luminance effect -- a moving gradient highlight that
follows the cursor along the card edge, mimicking real glass catching light:
```css
.hover-lift::before {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  padding: 1px;
  background: linear-gradient(
    var(--light-angle, 135deg),
    rgba(139,92,246,0.15),
    transparent 40%,
    transparent 60%,
    rgba(62,232,165,0.08)
  );
  -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
  -webkit-mask-composite: xor;
  mask-composite: exclude;
  pointer-events: none;
  opacity: 0;
  transition: opacity 300ms;
}
.hover-lift:hover::before { opacity: 1; }
```
The `--light-angle` CSS variable is updated by the `tilt.ts` utility based on
cursor position. This creates the LumenLingo-grade "glass that catches light" effect.

**AC-3**: Define focus-visible ring for all form inputs and buttons:
```css
.focus-ring:focus-visible {
  outline: none;
  box-shadow: 0 0 0 2px var(--void-surface), 0 0 0 4px var(--accent-violet);
  border-color: var(--accent-violet);
}
```

**AC-4**: Define active/pressed state with spring compress:
```css
.press-spring:active {
  transform: scale(0.97);
  transition: transform 80ms cubic-bezier(0.3, 0, 0.8, 0.15);
}
```

**AC-5**: Define disabled state with muted appearance:
```css
.state-disabled {
  opacity: 0.4;
  pointer-events: none;
  filter: grayscale(0.3);
}
```

**AC-6**: Every `<button>`, `<input>`, and clickable `<div>` in the app uses the
appropriate state classes. No interactive element exists without a defined hover state.

**AC-7**: Define a glass-morphic `.glass-input` class for all text inputs and selects:
```css
.glass-input {
  background: rgba(139,92,246,0.03);
  border: 1px solid rgba(139,92,246,0.10);
  border-radius: 10px;
  backdrop-filter: blur(8px);
  color: var(--text-primary);
  padding: 10px 14px;
  font-size: var(--font-base);
  transition: all 200ms cubic-bezier(0.16,1,0.3,1);
}
.glass-input:focus {
  border-color: rgba(139,92,246,0.3);
  box-shadow: 0 0 0 3px rgba(139,92,246,0.08), 0 0 20px rgba(139,92,246,0.06);
  background: rgba(139,92,246,0.05);
}
```

---

### S1.5: Premium Loading Skeletons

**As a** developer,
**I want** sophisticated skeleton placeholder patterns for every loading state,
**so that** the loading experience feels intentional rather than broken.

**AC-1**: Create a `.skeleton-pulse` CSS class with animated gradient shimmer:
```css
.skeleton-pulse {
  background: linear-gradient(
    110deg,
    rgba(139,92,246,0.03) 30%,
    rgba(139,92,246,0.08) 50%,
    rgba(139,92,246,0.03) 70%
  );
  background-size: 200% 100%;
  animation: skeleton-shimmer 1.8s ease-in-out infinite;
  border-radius: 8px;
}
```

**AC-2**: Create skeleton variants for common shapes:
- `.skeleton-text`: height 12px, 60-80% width, rounded-full
- `.skeleton-stat`: height 28px, 30% width (for stat card values)
- `.skeleton-row`: height 44px, 100% width (for table rows)
- `.skeleton-card`: height 200px, 100% width, rounded-2xl

**AC-3**: Every page that fetches data shows skeleton placeholders during loading,
not a centred spinner with text. The LoadingSpinner component is upgraded to render
contextual skeletons when given a `variant` prop.

**AC-4**: Skeleton elements fade out and real content fades in with a 200ms crossfade.

---

### S1.6: Premium Empty States

**As a** developer,
**I want** branded empty state designs with illustrations and helpful copy,
**so that** "no data" screens feel intentional and guide the user.

**AC-1**: Create a reusable `EmptyState` component with props:
- `icon`: Lucide icon component
- `title`: string (e.g. "No signals yet")
- `description`: string (e.g. "Run make stocks to generate trading signals")
- `action?`: { label: string, onClick: () => void }

**AC-2**: EmptyState visual design:
- Icon rendered at 48px with `var(--text-muted)` colour and subtle violet glow
- Title uses `.text-section` typography
- Description uses `.text-caption` typography
- Optional CTA button uses glass-card styling with hover-lift
- Container uses `glass-card` with `p-12` centre-aligned

**AC-3**: Every page that can show "No data" uses this component instead of inline
`<p>` tags with hardcoded colours.

**AC-4**: The icon uses a 1.5px stroke weight (matching Lucide defaults) and renders
at 48px inside a 72px circular glass-badge background:
```css
.empty-icon-badge {
  width: 72px; height: 72px;
  border-radius: 50%;
  background: rgba(139,92,246,0.04);
  border: 1px solid rgba(139,92,246,0.06);
  display: flex; align-items: center; justify-content: center;
}
```

**AC-5**: Around the empty state icon, 3-4 tiny floating particles (2-3px circles)
drift slowly upward, fading in and out at random intervals. This adds subtle life
to empty states, preventing them from feeling like dead ends. Particles use
`--accent-violet` at 0.15 opacity.

---

### S1.7: Premium Error States

**As a** developer,
**I want** elevated error state designs that communicate severity without panic,
**so that** errors look intentional and provide clear next steps.

**AC-1**: Create CSS classes for error elevation:
```css
.error-card {
  background: linear-gradient(135deg, rgba(255,107,138,0.04) 0%, rgba(10,10,26,0.95) 100%);
  border: 1px solid rgba(255,107,138,0.12);
  box-shadow: 0 0 24px rgba(255,107,138,0.06), var(--shadow-card);
}
```

**AC-2**: Create a reusable `ErrorState` component with:
- Rose-tinted glass-card background
- Icon with pulsing rose glow
- Error title, description, and optional retry button
- Retry button with `hover-lift` and `press-spring`

**AC-3**: All `try/catch` error boundaries and API error states use this component.

---

### S1.8: Recharts Theme Configuration

**As a** developer,
**I want** a centralised Recharts theme object that applies the design system colours
and typography to all charts,
**so that** charts look like they belong to the same product as the rest of the UI.

**AC-1**: Create a `chartTheme.ts` utility exporting:
```typescript
export const CHART_COLORS = {
  violet: 'var(--accent-violet)',
  emerald: 'var(--accent-emerald)',
  rose: 'var(--accent-rose)',
  amber: 'var(--accent-amber)',
  muted: 'var(--text-muted)',
  grid: 'rgba(139,92,246,0.06)',
};

export const CHART_TOOLTIP_STYLE = {
  background: 'rgba(15,15,35,0.95)',
  border: '1px solid rgba(139,92,246,0.15)',
  borderRadius: 12,
  color: 'var(--text-primary)',
  backdropFilter: 'blur(16px)',
  boxShadow: '0 8px 32px rgba(0,0,0,0.4)',
  padding: '12px 16px',
  fontSize: 12,
};

export const CHART_AXIS_STYLE = {
  fill: 'var(--text-muted)',
  fontSize: 11,
};
```

**AC-2**: All Recharts `<Tooltip>`, `<CartesianGrid>`, `<XAxis>`, `<YAxis>` components
across all pages reference this shared theme object.

**AC-3**: Cart grid uses `stroke={CHART_COLORS.grid}` and `strokeDasharray="3 3"`.

**AC-4**: Tooltip has fade-in entrance animation via CSS class.

**AC-5**: Create gradient area fill helper `<defs>` for reuse across all chart pages:
```typescript
export const CHART_GRADIENTS = {
  violetArea: { id: 'violet-area', from: 'rgba(139,92,246,0.15)', to: 'rgba(15,15,35,0)' },
  emeraldArea: { id: 'emerald-area', from: 'rgba(62,232,165,0.12)', to: 'rgba(15,15,35,0)' },
  roseArea: { id: 'rose-area', from: 'rgba(255,107,138,0.12)', to: 'rgba(15,15,35,0)' },
};
```
Every `<Area>` or `<AreaChart>` uses these gradient definitions. No chart area
should render as a flat colour fill.

**AC-6**: Create a `useAnimatedLine` hook that draws chart lines left-to-right
on mount using `stroke-dashoffset` animation over 1000ms. This gives every
chart the LumenLingo "reveal" feel.

---

### S1.9: Global Scrollbar Theming

**As a** developer,
**I want** all scrollbars to use subtle branded styling,
**so that** scrolling feels like part of the design, not an OS intrusion.

**AC-1**: Apply custom scrollbar to the entire app:
```css
* {
  scrollbar-width: thin;
  scrollbar-color: rgba(139,92,246,0.12) transparent;
}
*::-webkit-scrollbar { width: 6px; height: 6px; }
*::-webkit-scrollbar-track { background: transparent; }
*::-webkit-scrollbar-thumb {
  background: rgba(139,92,246,0.12);
  border-radius: 100px;
  transition: background 200ms;
}
*::-webkit-scrollbar-thumb:hover {
  background: rgba(139,92,246,0.25);
}
*::-webkit-scrollbar-corner { background: transparent; }
```

**AC-2**: Scrollbar thumb only appears when scrolling (auto-hide behaviour):
```css
*::-webkit-scrollbar-thumb {
  background: transparent;
}
*:hover::-webkit-scrollbar-thumb {
  background: rgba(139,92,246,0.12);
}
```

**AC-3**: The main content area scrollbar has a slightly wider width (8px) than
inner panels (4px).

---

### S1.10: Noise Texture & Depth Layer

**As a** developer,
**I want** a subtle noise/grain texture overlaid on glass-card surfaces,
**so that** they feel like real frosted glass with physical texture, not flat CSS.

**AC-1**: Create a 200x200 SVG noise texture as inline data URI:
```css
.noise-texture::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background-image: url("data:image/svg+xml,..."); /* Perlin noise at 3% opacity */
  opacity: 0.03;
  pointer-events: none;
  mix-blend-mode: overlay;
}
```

**AC-2**: Apply `.noise-texture` to all `.glass-card` elements. The noise should
be nearly invisible but add a tactile "grain" that distinguishes the surface from
a flat `rgba()` background.

**AC-3**: Create an inner light edge on glass-cards:
```css
.glass-card {
  box-shadow:
    inset 0 1px 0 0 rgba(255,255,255,0.04),  /* top edge light */
    inset 0 -1px 0 0 rgba(0,0,0,0.1),         /* bottom edge shadow */
    var(--shadow-card);
}
```
This inset highlight creates the illusion of a glass edge catching ambient light,
matching LumenLingo's "every pixel considered" standard.

**AC-4**: On hover, the top edge brightens:
```css
.glass-card:hover {
  box-shadow:
    inset 0 1px 0 0 rgba(255,255,255,0.08),
    inset 0 -1px 0 0 rgba(0,0,0,0.1),
    var(--shadow-card-hover);
}
```

---

### S1.11: Spring Physics & Motion Tokens

**As a** developer,
**I want** named motion timing tokens instead of ad-hoc cubic-bezier values,
**so that** every animation in the app uses consistent physics-based curves.

**AC-1**: Define named spring curves as CSS custom properties:
```css
--spring-bounce:  cubic-bezier(0.16, 1, 0.3, 1);   /* main spring: overshoot + settle */
--spring-snappy:  cubic-bezier(0.34, 1.56, 0.64, 1); /* playful overshoot */
--spring-gentle:  cubic-bezier(0.22, 1, 0.36, 1);    /* subtle ease-out */
--ease-out-expo:  cubic-bezier(0.16, 1, 0.3, 1);     /* fast start, slow end */
--ease-in-out:    cubic-bezier(0.4, 0, 0.2, 1);      /* symmetric motion */
```

**AC-2**: Define duration tokens:
```css
--duration-instant: 80ms;    /* active/press states */
--duration-fast:    150ms;   /* tooltip show, small state changes */
--duration-normal:  250ms;   /* tab slides, panel transitions */
--duration-slow:    400ms;   /* entrance animations, page reveals */
--duration-reveal:  600ms;   /* dramatic reveals, chart line draws */
--duration-epic:    1000ms;  /* full-page entrance choreography */
```

**AC-3**: All CSS transitions and animations reference these tokens:
```css
/* BEFORE: */
transition: all 200ms cubic-bezier(0.16,1,0.3,1);
/* AFTER: */
transition: all var(--duration-normal) var(--spring-bounce);
```

**AC-4**: Define a JS-side spring config for Framer Motion (if adopted) or CSS:
```typescript
export const SPRING = {
  bounce: { type: 'spring', stiffness: 300, damping: 20 },
  snappy: { type: 'spring', stiffness: 400, damping: 15 },
  gentle: { type: 'spring', stiffness: 200, damping: 25 },
};
```

---

## E2: Overview Page

**Current Rating**: 8/10
**Target Rating**: 9.5/10
**Files**: `src/pages/OverviewPage.tsx`, `src/components/BriefingCard.tsx`, `src/components/StatCard.tsx`

The Overview page is the strongest page with its BriefingCard hero and StatCard grid.
Gaps are in lower sections (ModelLeaderboard, TopSectors, ConvictionSpotlight) which
lack the same depth as the hero, and hardcoded colour values throughout.

---

### S2.1: StatCard Colour Variable Migration

**As a** user,
**I want** StatCard colours to come from the design system,
**so that** palette changes are instant and consistent.

**AC-1**: Replace the hardcoded `ACCENT` map in StatCard.tsx:
```typescript
// BEFORE (hardcoded)
const ACCENT: Record<string, { r: number; g: number; b: number }> = {
  green:  { r: 62,  g: 232, b: 165 },
  red:    { r: 255, g: 107, b: 138 },
  blue:   { r: 139, g: 92,  b: 246 },
};
// AFTER (CSS variable references)
const ACCENT_VARS: Record<string, string> = {
  green:  'var(--accent-emerald)',
  red:    'var(--accent-rose)',
  blue:   'var(--accent-violet)',
  amber:  'var(--accent-amber)',
};
```

**AC-2**: The glow gradient overlay uses CSS variable with opacity modifiers, not
inline `rgba()` with extracted r/g/b values.

**AC-3**: StatCard title uses `.text-label` typography class.
StatCard value uses `.text-stat-value` typography class.

**AC-4**: StatCard padding uses Tailwind `p-6` class instead of inline
`style={{ padding: '24px 24px 20px' }}`.

**AC-5**: StatCards respond to the 3D perspective tilt from `.hover-lift` (S1.4).
On hover, the card tilts toward the cursor with the border luminance edge-light
effect. This is the single highest-impact "drool" moment on the Overview page --
cards that feel like real glass surfaces floating above the void.

---

### S2.2: BriefingCard Sentiment Colour Extraction

**As a** user,
**I want** BriefingCard sentiment colours to use CSS classes instead of inline styles,
**so that** the aurora glow and badge gradients are maintainable.

**AC-1**: Create CSS classes for sentiment glow:
```css
.sentiment-glow-bull {
  background: radial-gradient(ellipse at 30% 50%, rgba(62,232,165,0.08) 0%, transparent 70%);
}
.sentiment-glow-bear {
  background: radial-gradient(ellipse at 30% 50%, rgba(255,107,138,0.08) 0%, transparent 70%);
}
.sentiment-glow-neutral {
  background: radial-gradient(ellipse at 30% 50%, rgba(139,92,246,0.06) 0%, transparent 70%);
}
```

**AC-2**: Badge direction indicators use `data-direction="buy|sell|hold"` attribute
with CSS styling instead of inline gradient strings.

**AC-3**: BriefingCard entrance animation extracted to a named CSS class
`.briefing-enter` instead of inline `style={{ transition, transitionDelay }}`.

**AC-4**: Section titles within BriefingCard use `.text-label` class (11px, uppercase,
wide tracking) instead of inline style objects.

---

### S2.3: ModelLeaderboard Premium Elevation

**As a** user,
**I want** the Model Leaderboard section to have the same visual depth as the
BriefingCard hero,
**so that** it feels like a premium data surface, not an afterthought.

**AC-1**: ModelLeaderboard container uses `glass-card hover-lift` classes with
consistent `p-6` padding.

**AC-2**: Leaderboard rows use `premium-row` class with alternating backgrounds
and hover glow.

**AC-3**: Top-3 model names have `rank-badge rank-gold`, `rank-silver`, `rank-bronze`
classes from the global design system.

**AC-4**: BMA weight values display as mini horizontal bars (40px wide) with gradient
fill matching the `premium-row` pattern, not just text percentages.

**AC-5**: Section header "Model Leaderboard" uses `.premium-section-label` class.

**AC-6**: BMA weight mini-bars animate from 0 to final width on mount using
`transition: width var(--duration-reveal) var(--spring-bounce)`, creating a
satisfying "growing bars" reveal when the section loads.

---

### S2.4: ConvictionSpotlight Card Enhancement

**As a** user,
**I want** ConvictionSpotlight buy/sell cards to have dramatic colour glow and hover
interaction,
**so that** high-conviction signals feel urgent and important.

**AC-1**: Buy cards have a subtle emerald glow aura:
```css
.conviction-buy {
  box-shadow: 0 0 40px rgba(62,232,165,0.06), var(--shadow-card);
  border: 1px solid rgba(62,232,165,0.10);
}
.conviction-buy:hover {
  box-shadow: 0 0 60px rgba(62,232,165,0.10), var(--shadow-card-hover);
  border-color: rgba(62,232,165,0.18);
  transform: translateY(-2px);
}
```

**AC-2**: Sell cards have the equivalent rose glow aura.

**AC-3**: The asset ticker name within each card uses `.text-section` typography
with `var(--text-luminous)` colour.

**AC-4**: Expected return percentage uses large `.text-stat-value` typography
with colour matching the card sentiment (emerald for positive, rose for negative).

**AC-5**: Cards have staggered entrance animation using `fade-up-delay-*` classes.

**AC-6**: Active high-conviction cards (> 8% expected return) have a pulsing border
glow that breathes at 3s intervals, creating urgency:
```css
.conviction-urgent {
  animation: conviction-breathe 3s ease-in-out infinite;
}
@keyframes conviction-breathe {
  0%, 100% { border-color: rgba(62,232,165,0.10); }
  50% { border-color: rgba(62,232,165,0.25); box-shadow: 0 0 30px rgba(62,232,165,0.08); }
}
```

---

### S2.5: SignalDistributionBar Redesign

**As a** user,
**I want** the signal distribution bar (buy/hold/sell breakdown) to be a premium
visualisation,
**so that** it communicates the portfolio sentiment at a glance.

**AC-1**: Bar segments use gradient fills instead of flat colours:
- Buy segment: `linear-gradient(90deg, var(--accent-emerald), rgba(62,232,165,0.6))`
- Sell segment: `linear-gradient(90deg, rgba(255,107,138,0.6), var(--accent-rose))`
- Hold segment: `var(--glass-surface)` (barely visible neutral)

**AC-2**: Bar has a subtle inset shadow: `inset 0 1px 2px rgba(0,0,0,0.3)` for depth.

**AC-3**: Segment widths animate from 0 to final value on mount using
`transition: width 600ms cubic-bezier(0.16,1,0.3,1)`.

**AC-4**: Percentage labels appear above each segment with `.text-caption` typography,
fading in 200ms after the bar animation completes.

**AC-5**: The entire bar component uses CSS custom property `--bar-width` set via
React style prop for each segment -- no hardcoded `width: 40` pixel values.

---

### S2.6: Overview Page Entrance Choreography

**As a** user,
**I want** the Overview page to feel like a "reveal" when I navigate to it,
**so that** the first impression is dramatic and premium.

**AC-1**: Elements appear in this staggered sequence:
1. PageHeader (0ms) -- fade-up
2. BriefingCard (80ms) -- fade-up with subtle scale from 0.98
3. StatCard grid (160ms--320ms) -- staggered per card (4 cards x 40ms delay)
4. ConvictionSpotlight (400ms) -- fade-up
5. Lower sections (480ms+) -- staggered fade-up

**AC-2**: Each `fade-up` animation runs 500ms with `cubic-bezier(0.16,1,0.3,1)` and
includes a subtle blur-to-sharp effect: `filter: blur(4px)` at start, `blur(0)` at end.

**AC-3**: On subsequent visits (SPA navigation), animations still play but at 60%
duration (300ms) so returning users are not slowed down.

**AC-4**: Animations respect `prefers-reduced-motion` -- if enabled, elements appear
instantly with no animation.

**AC-5**: Below-fold content (ModelLeaderboard, TopSectors, lower sections) uses
IntersectionObserver-triggered reveals: content stays at `opacity: 0; translateY(16px)`
until it enters the viewport, then animates in. This rewards scrolling with
continuous visual payoff -- the page keeps "revealing" as you explore.

---

### S2.7: Overview Scroll-Triggered Data Reveals

**As a** user,
**I want** numbers and bars in the Overview page to animate when I scroll to them,
**so that** every section feels alive, not pre-rendered.

**AC-1**: StatCard values use the `useCountUp` hook (S12.2) -- numbers count from
0 to their final value when the card enters the viewport.

**AC-2**: Signal distribution bar segments grow from 0 width when the bar enters
the viewport, not on page mount.

**AC-3**: BMA weight bars in ModelLeaderboard grow from 0 when the section scrolls
into view.

**AC-4**: Each data reveal respects `prefers-reduced-motion` by showing final values
instantly.

---

## E3: Risk Dashboard

**Current Rating**: 7/10
**Target Rating**: 9.5/10
**Files**: `src/pages/RiskPage.tsx`

The Risk page has an excellent temperature gauge with spring animation. Gaps are in
the tab navigation, sparkline styling, trend indicators, and the sub-tab tables for
cross-asset/metals/equity which are utilitarian.

---

### S3.1: Temperature Gauge Responsive Scaling

**As a** user,
**I want** the temperature gauge to scale with the viewport instead of being fixed-size,
**so that** it looks proportional on all screen sizes.

**AC-1**: Gauge SVG uses `viewBox="0 0 200 200"` with responsive container sizing:
`w-48 h-48 md:w-52 md:h-52 lg:w-56 lg:h-56` instead of fixed `200x200` pixels.

**AC-2**: Centre text uses `.text-stat-value` typography class instead of hardcoded
`fontSize: 40, fontWeight: 700`.

**AC-3**: Status label below the number uses `.text-caption` typography.

**AC-4**: Gauge has a premium entrance animation: scale from 0.85 to 1.0 with spring
overshoot during the needle sweep animation that already exists.

**AC-5**: The gauge ring has a subtle outer glow matching the regime colour:
```css
filter: drop-shadow(0 0 12px rgba(regime-color, 0.2));
```

**AC-6**: Gauge needle sweeps from 0 to the target value with spring overshoot
(overshoots by 5% then settles back) using a custom `@keyframes` animation.
The sweep leaves a faint trailing arc glow that fades over 500ms, like a comet
tail -- this is the hero moment of the Risk page.

---

### S3.2: Regime Colour CSS Class System

**As a** user,
**I want** regime colours to be CSS classes instead of inline functions,
**so that** regime indicators are consistent across all tabs and pages.

**AC-1**: Create CSS utility classes for each regime:
```css
.regime-calm      { --regime-color: var(--accent-emerald); color: var(--accent-emerald); }
.regime-elevated  { --regime-color: var(--accent-amber); color: var(--accent-amber); }
.regime-stressed  { --regime-color: var(--accent-orange); color: var(--accent-orange); }
.regime-crisis    { --regime-color: var(--accent-rose); color: var(--accent-rose); }
```

**AC-2**: Each regime class includes a glow variant:
```css
.regime-calm-glow      { box-shadow: 0 0 20px rgba(62,232,165,0.12); }
.regime-elevated-glow  { box-shadow: 0 0 20px rgba(245,197,66,0.12); }
.regime-stressed-glow  { box-shadow: 0 0 20px rgba(249,115,22,0.12); }
.regime-crisis-glow    { box-shadow: 0 0 20px rgba(255,107,138,0.12); }
```

**AC-3**: Replace `regimeColor()` and `regimeGlow()` inline functions with a lookup
to CSS class names. The functions return class strings, not colour values.

**AC-4**: All regime indicators across RiskPage, DiagnosticsPage, and SignalsPage
use these shared classes.

---

### S3.3: Tab Navigation Premium Upgrade

**As a** user,
**I want** the Risk page tab navigation to feel like a premium selector,
**so that** switching between views is a satisfying interaction.

**AC-1**: Tab buttons have increased padding: `px-5 py-3` instead of current tight
spacing.

**AC-2**: Active tab has a smooth animated underline indicator that slides to the
active position using `transform: translateX()` with 300ms spring transition,
instead of just changing border-bottom colour.

**AC-3**: Active tab text uses `var(--text-violet)` with `font-weight: 600`, inactive
uses `var(--text-muted)` with `font-weight: 400`.

**AC-4**: Tab background on active has a subtle fill:
`background: rgba(139,92,246,0.04); border-radius: 12px 12px 0 0`.

**AC-5**: Tab icons (if present) scale to 1.1x on active state.

**AC-6**: Behind the active tab text, a filled pill background slides horizontally
to the active position using `transform: translateX()` with absolute positioning.
This sliding pill (not just an underline) matches the LumenLingo premium selector
feel -- a solid surface that physically moves between options.

---

### S3.4: Sparkline Premium Styling

**As a** user,
**I want** the temperature sparkline to look like a premium data visualisation,
**so that** recent history is visually rich, not a bare line.

**AC-1**: Sparkline has a gradient fill below the curve:
```
linear-gradient(to bottom, rgba(139,92,246,0.12) 0%, rgba(15,15,35,0) 100%)
```

**AC-2**: The line uses a 2px stroke with rounded line-cap and the regime accent colour.

**AC-3**: Sparkline has an entrance animation: the line draws from left to right
over 800ms using `stroke-dashoffset` animation.

**AC-4**: The most recent data point has a small pulsing dot (4px diameter) at the
line end, coloured with the regime accent.

**AC-5**: On hover over the sparkline container, a tooltip shows the exact value
with the premium `CHART_TOOLTIP_STYLE` from the theme.

---

### S3.5: Risk Sub-Tab Table Styling

**As a** user,
**I want** the cross-asset, metals, equity, and currency tables in the Risk page
sub-tabs to use premium row styling,
**so that** data tables are visually consistent with the rest of the app.

**AC-1**: All risk sub-tab tables use `premium-thead` for the header row.

**AC-2**: All data rows use `premium-row` class with hover glow and alternating
backgrounds.

**AC-3**: Numeric values in forecast columns use `font-variant-numeric: tabular-nums`
for alignment.

**AC-4**: Positive forecast values display in `var(--accent-emerald)`, negative in
`var(--accent-rose)`, neutral in `var(--text-secondary)`.

**AC-5**: Risk score columns display as a mini horizontal bar (similar to
SignalStrengthBar) with gradient fill from emerald through amber to rose based on
the risk level, not just a coloured number.

---

## E4: Signals Page

**Current Rating**: 6/10
**Target Rating**: 9.5/10
**Files**: `src/pages/SignalsPage.tsx`, `src/components/SignalTableVisuals.tsx`

The Signals page is the most data-dense page. The main table already uses
`cosmic-row` and `cosmic-table-header` but many supporting elements are flat:
filter pills, search bar, stats bar, WebSocket indicator, and the
StrongSignalsView rows. SignalTableVisuals components use hardcoded colour maps.

---

### S4.1: Signal Filter Pills Premium Redesign

**As a** user,
**I want** signal filter pills (Buy/Sell/Hold/All) to feel like premium toggles,
**so that** selecting a view mode is satisfying and clear.

**AC-1**: Filter pills use glass-card styling with border:
```css
.filter-pill {
  padding: 6px 14px;
  border-radius: 100px;
  border: 1px solid var(--glass-border);
  background: transparent;
  color: var(--text-muted);
  font-size: var(--font-2xs);
  font-weight: var(--weight-semibold);
  text-transform: uppercase;
  letter-spacing: var(--tracking-wider);
  transition: all 200ms cubic-bezier(0.16,1,0.3,1);
}
```

**AC-2**: Active filter pill has filled background with accent glow:
```css
.filter-pill[data-active="true"] {
  background: rgba(139,92,246,0.12);
  border-color: rgba(139,92,246,0.25);
  color: var(--text-violet);
  box-shadow: 0 0 12px rgba(139,92,246,0.08);
}
```

**AC-3**: Buy-specific active pill uses emerald accent, Sell uses rose, Hold uses
amber. The `data-filter` attribute drives the colour.

**AC-4**: Pills have `press-spring` active state and `hover-lift` on hover.

**AC-5**: Filter pill container gap increased from `gap-0.5` to `gap-2`.

**AC-6**: An animated pill background (similar to S3.3 AC-6) slides behind the
active filter when switching. The background pill is a separate `<span>` absolutely
positioned, with `transform: translateX()` animated via `var(--spring-bounce)`,
creating the illusion that a glass surface is physically sliding between options
rather than toggling visibility.

---

### S4.2: Search Bar Focus Elevation

**As a** user,
**I want** the signal search bar to glow when focused,
**so that** it feels responsive and premium during interaction.

**AC-1**: Search input uses `focus-ring` class from the design system producing a
violet glow ring on focus.

**AC-2**: Search container has a subtle backdrop-filter blur: `backdrop-filter: blur(8px)`.

**AC-3**: The search icon transitions from `var(--text-muted)` to `var(--accent-violet)`
when the input has focus, using `group-focus-within` Tailwind modifier.

**AC-4**: Placeholder text uses `var(--text-muted)` and fades out when the user starts
typing (via `placeholder-shown` pseudo-class or CSS transition).

**AC-5**: Search input padding increased from `py-1.5` to `py-2.5` for a more
spacious touch target.

---

### S4.3: WebSocket Status Indicator

**As a** user,
**I want** the WebSocket connection indicator to pulse when connected and show a
clear disconnected state,
**so that** real-time data freshness is visible.

**AC-1**: Connected state: green dot with pulsing glow ring animation:
```css
.ws-connected {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--accent-emerald);
  box-shadow: 0 0 8px rgba(62,232,165,0.4);
  animation: ws-pulse 2s ease-in-out infinite;
}
@keyframes ws-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(62,232,165,0.4); }
  50% { box-shadow: 0 0 16px rgba(62,232,165,0.6), 0 0 4px rgba(62,232,165,0.8); }
}
```

**AC-2**: Disconnected state: rose dot with no animation, dim opacity 0.6.

**AC-3**: Reconnecting state: amber dot with fast pulse (0.8s period).

**AC-4**: Status label text uses `.text-caption` typography and shows
"Live", "Disconnected", or "Reconnecting" next to the dot.

**AC-5**: The connected indicator also functions as a data quality ring: a thin
(2px) circle around the dot that fills clockwise over 60s to show time since
last update. At full circle (60s), the ring colour transitions from emerald
through amber to rose, providing at-a-glance data staleness.

---

### S4.4: Signal Change Log Entrance Animation

**As a** user,
**I want** new signal changes in the aurora trail to slide in with a staggered
animation,
**so that** real-time updates feel alive and noticeable.

**AC-1**: New change log entries animate in from the right:
```css
@keyframes signal-entry {
  from { opacity: 0; transform: translateX(20px); filter: blur(4px); }
  to   { opacity: 1; transform: translateX(0); filter: blur(0); }
}
```

**AC-2**: Each entry has a `50ms * index` stagger delay (max 5 entries visible).

**AC-3**: The aurora sweep gradient (existing green/rose animation) triggers on
new entry insertion, not on page load.

**AC-4**: Old entries that leave the visible area fade out with `opacity: 0` and
`transform: translateX(-20px)` over 300ms.

---

### S4.5: SignalTableVisuals Colour Variable Migration

**As a** user,
**I want** all SignalTableVisuals components to use design system colours,
**so that** badges and bars are centrally themed.

**AC-1**: `SignalStrengthBar` gradient stops use CSS variables:
- Buy gradient: `var(--accent-emerald)` with opacity modifiers
- Sell gradient: `var(--accent-rose)` with opacity modifiers
- Hold: `var(--text-muted)` with low opacity

**AC-2**: `MomentumBadge` replaces inline colour calculation with
`data-sentiment="positive|negative|neutral"` attribute and CSS classes:
```css
.momentum-badge[data-sentiment="positive"] {
  background: rgba(62,232,165,0.12);
  color: var(--accent-emerald);
}
```

**AC-3**: `CrashRiskHeat` replaces hardcoded `#F97316` with `var(--accent-orange)`.

**AC-4**: `HorizonCell` colour ternary logic replaced with CSS class system:
`data-direction="up|down|flat"` driving colour via CSS.

**AC-5**: All badge widths use responsive units instead of hardcoded `width: 40` pixels.

---

### S4.6: StrongSignalsView Premium Rows

**As a** user,
**I want** Strong Buy and Strong Sell signal rows to have dramatic hover effects
and premium visual hierarchy,
**so that** the strongest signals command attention.

**AC-1**: Each signal row has a hover gradient glow matching its sentiment:
- Buy rows: `linear-gradient(90deg, rgba(62,232,165,0.06) 0%, transparent 60%)`
- Sell rows: `linear-gradient(90deg, rgba(255,107,138,0.06) 0%, transparent 60%)`

**AC-2**: Row hover includes `translateY(-0.5px)` lift and
`box-shadow: 0 2px 12px rgba(accent, 0.06)`.

**AC-3**: Asset ticker uses `var(--text-luminous)` with `font-weight: 600`.
Sector label uses `var(--text-muted)`.

**AC-4**: Expected return uses `.text-stat-value` size when the value exceeds
a threshold (e.g. |ret| > 5%), making standout signals visually larger.

**AC-5**: Section headers ("Strong Buy Signals", "Strong Sell Signals") use
`.premium-section-label` class with sentiment-coloured accent icon.

---

### S4.8: Signal Deep-Dive Drawer

**As a** user,
**I want** clicking a signal row to open a glass-morphic slide-out drawer,
**so that** I can see full signal details without leaving the context.

**AC-1**: Drawer slides in from the right over 300ms using
`transform: translateX(100%)` to `translateX(0)` with `var(--spring-bounce)`.

**AC-2**: Drawer has a `backdrop-filter: blur(24px)` frosted glass surface with
the noise texture overlay (S1.10).

**AC-3**: Drawer displays:
- Asset name and sector in `.text-heading` and `.text-label`
- Signal direction with large emerald/rose sentiment badge
- Horizon forecast table using `premium-thead` and `premium-row`
- Mini sparkline showing recent price action
- Model confidence breakdown as mini stacked bars

**AC-4**: A translucent backdrop overlay (`rgba(10,10,26,0.5)`) covers the main
content, clicking it closes the drawer.

**AC-5**: Drawer contents animate in with staggered fade-up (80ms delay per section)
after the drawer slide finishes, creating a two-phase reveal.

---

### S4.7: Signal Table Number Animation

**As a** user,
**I want** numeric values in the signal table to smoothly animate when updated
via WebSocket,
**so that** changes are visible without jarring flashes.

**AC-1**: When a cell value changes, the number crossfades from old to new over
200ms using CSS transition on opacity.

**AC-2**: Changed cells have a brief (1s) highlight glow:
- Positive change: emerald glow pulse
- Negative change: rose glow pulse

**AC-3**: The highlight fades out over 1s using CSS animation, not JavaScript timers.

**AC-4**: Row-level aurora sweep (already exists) triggers only for the changed row,
not all rows.

---

## E5: Data Management Page

**Current Rating**: 6/10
**Target Rating**: 9/10
**Files**: `src/pages/DataPage.tsx`

The Data page manages price files. Current state has a functional file table but
it lacks depth, hover effects, and visual richness. Section headers are plain.

---

### S5.1: File Table Row Enhancement

**As a** user,
**I want** file table rows to feel like interactive data surfaces,
**so that** browsing price files is a premium experience.

**AC-1**: Rows use `premium-row` class with hover glow, alternating backgrounds,
and `translateY(-0.5px)` lift on hover.

**AC-2**: The "Age" column displays a styled badge instead of plain text:
```css
.age-badge-fresh {
  background: rgba(62,232,165,0.10);
  color: var(--accent-emerald);
  padding: 2px 8px;
  border-radius: 100px;
  font-size: var(--font-2xs);
  font-weight: var(--weight-semibold);
}
.age-badge-stale {
  background: rgba(245,197,66,0.10);
  color: var(--accent-amber);
}
.age-badge-old {
  background: rgba(255,107,138,0.10);
  color: var(--accent-rose);
}
```

**AC-3**: The symbol column uses `var(--text-luminous)` with `font-weight: 600`.

**AC-4**: Row size uses `font-variant-numeric: tabular-nums` for aligned numbers.

**AC-5**: The "Updated" column timestamp uses relative time ("2h ago", "3d ago")
instead of raw date, with `var(--text-muted)` colour.

---

### S5.2: Directory Status Visualisation

**As a** user,
**I want** directory status indicators to be clear visual badges,
**so that** I can instantly see which data directories exist.

**AC-1**: Each directory row shows a status indicator:
- Exists: emerald dot + "Available" badge (`.status-pill.status-pass`)
- Missing: rose dot + "Missing" badge (`.status-pill.status-fail`)

**AC-2**: Directory count (number of files) appears as a subtle secondary metric
in `var(--text-secondary)` next to the name.

**AC-3**: Directory card uses `glass-card hover-lift` with `p-6` padding.

**AC-4**: Section header "Data Directories" uses `.premium-section-label` class
with a `FolderOpen` icon in `var(--accent-violet)`.

---

### S5.3: Data Page Search Enhancement

**As a** user,
**I want** the file search bar to have a premium focus state,
**so that** searching feels responsive and elegant.

**AC-1**: Search input uses `focus-ring` class with violet glow.

**AC-2**: Search container has `backdrop-filter: blur(8px)` for glass depth.

**AC-3**: The search icon transitions colour on focus (muted to violet).

**AC-4**: Results count updates smoothly with a number crossfade animation.

---

### S5.4: Refresh Action Feedback

**As a** user,
**I want** the refresh action to provide rich visual feedback,
**so that** I know the system is working when I trigger a data refresh.

**AC-1**: The refresh button uses `press-spring` active state and `hover-lift`.

**AC-2**: During refresh, the button icon spins with existing `animate-spin` and
the button background pulses with a subtle violet glow.

**AC-3**: Refresh success message appears with a fade-up animation and auto-dismisses
after 3 seconds with a fade-out.

**AC-4**: If refresh fails, an `error-card` styled notification appears with the
error message and a retry button.

---

### S5.5: Data Freshness Dashboard Header

**As a** user,
**I want** a visual dashboard strip at the top of the Data page showing overall
data health,
**so that** I can see freshness status at a glance without scanning rows.

**AC-1**: Three glass-card stat tiles across the top:
- "Fresh" (< 24h): emerald StatCard showing count and percentage
- "Stale" (1-7d): amber StatCard showing count
- "Outdated" (> 7d): rose StatCard showing count

**AC-2**: A progress ring (similar to a donut chart, 80px diameter) shows the
freshness ratio as a gradient arc from emerald to rose.

**AC-3**: All numbers use `useCountUp` animation on mount.

**AC-4**: The progress ring animates its arc from 0 to the actual value over
`var(--duration-reveal)` with `var(--spring-bounce)`.

**AC-5**: The freshness tier thresholds (24h, 7d) are shown as `.text-caption`
legend items below the ring.

---

## E6: Arena Competition Page

**Current Rating**: 7/10
**Target Rating**: 9.5/10
**Files**: `src/pages/ArenaPage.tsx`

The Arena page showcases model competitions. It already uses `premium-thead`,
`premium-row`, and `rank-badge` classes from the prior upgrade. Remaining gaps:
gate colour depth, star map grid visuals, model detail panels, and overall page
entrance choreography.

---

### S6.1: Gate Indicator Depth Enhancement

**As a** user,
**I want** the hard gate indicators (CSS, FEC, Hyv, PIT, vs STD) to have dramatic
pass/fail visualisation,
**so that** the importance of each gate is immediately clear.

**AC-1**: Passing gates display with emerald glow badge:
```css
.gate-pass {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 3px 10px;
  border-radius: 100px;
  background: rgba(62,232,165,0.10);
  color: var(--accent-emerald);
  font-size: var(--font-2xs);
  font-weight: var(--weight-semibold);
  border: 1px solid rgba(62,232,165,0.15);
  box-shadow: 0 0 8px rgba(62,232,165,0.06);
}
```

**AC-2**: Failing gates display with rose glow:
```css
.gate-fail {
  background: rgba(255,107,138,0.10);
  color: var(--accent-rose);
  border: 1px solid rgba(255,107,138,0.15);
  box-shadow: 0 0 8px rgba(255,107,138,0.06);
}
```

**AC-3**: Each gate badge includes a small checkmark (pass) or X (fail) icon
at 10px size inline before the label text.

**AC-4**: Gate values that are close to the threshold (within 10%) display in
amber with a warning glow, indicating borderline status.

**AC-5**: On hover, each gate badge expands a mini tooltip showing the actual
threshold (e.g. "CSS >= 0.65, got 0.71") with `CHART_TOOLTIP_STYLE`. This
makes the gate system self-documenting without cluttering the row.

---

### S6.2: Model Competition Detail Panel

**As a** user,
**I want** expanding a model row to reveal a detailed panel,
**so that** I can see full metrics without navigating away.

**AC-1**: Clicking a model row smoothly expands a detail panel below it using
CSS `max-height` transition with `cubic-bezier(0.16,1,0.3,1)` over 400ms.

**AC-2**: Detail panel uses `glass-card` styling with `p-6`, 1px inset border,
and subtle violet glow background:
```css
.model-detail-panel {
  background: linear-gradient(135deg, rgba(139,92,246,0.03) 0%, transparent 60%);
  border-top: 1px solid rgba(139,92,246,0.06);
}
```

**AC-3**: Detail panel displays all scoring components as a horizontal stat grid:
BIC, CRPS, Hyv, PIT, CSS, FEC, DIG -- each in a mini StatCard-style surface.

**AC-4**: Panel content fades in 100ms after the expand starts to avoid
content appearing before the container is sized.

---

### S6.3: Star Map Grid Elevation

**As a** user,
**I want** the experimental model star map to feel like a premium constellation chart,
**so that** model exploration feels visually rich.

**AC-1**: Grid cells increase from 20x20 to 28x28 pixel minimum with `p-2` padding.

**AC-2**: Each star has a glow ring on hover:
```css
.star-cell:hover .star {
  filter: drop-shadow(0 0 6px rgba(139,92,246,0.4));
  transform: scale(1.3);
  transition: all 200ms cubic-bezier(0.16,1,0.3,1);
}
```

**AC-3**: Stars are coloured by status:
- Champion: `var(--accent-amber)` with gold glow (pulsing)
- Promoted: `var(--accent-emerald)` with steady glow
- Active: `var(--accent-violet)`
- Failed: `var(--text-muted)` at 0.4 opacity

**AC-4**: Connecting lines between related models use subtle gradient strokes with
`stroke-dasharray` animation on mount.

**AC-5**: Hovering a star reveals a tooltip (premium `CHART_TOOLTIP_STYLE`) showing
the model name and final score.

**AC-6**: Constellation lines between related models animate their `stroke-dashoffset`
from full length to 0 over 800ms with staggered delays (50ms per line), creating
the effect of a star map being "drawn" as the page loads. Lines pulse faintly
at 0.02 opacity when idle, brightening to 0.08 on connected-star hover.

---

### S6.4: Arena Scoring Radar Chart

**As a** user,
**I want** a small radar/spider chart for each model's scoring breakdown,
**so that** strengths and weaknesses are visible at a glance.

**AC-1**: Radar chart uses custom SVG (not Recharts) for lightweight rendering.

**AC-2**: Axes: BIC, CRPS, Hyv, PIT, CSS, FEC (6 spokes).

**AC-3**: Area fill uses violet gradient with 0.15 opacity. Stroke uses
`var(--accent-violet)` at 2px.

**AC-4**: Chart background rings at 25%, 50%, 75%, 100% use
`rgba(139,92,246,0.04)` fill with `rgba(139,92,246,0.08)` stroke.

**AC-5**: Chart appears inside the model detail panel (S6.2) alongside the stat grid.

**AC-6**: The radar area fill draws clockwise on mount over 600ms using a clip-path
reveal animation -- the filled polygon sweeps from 12 o'clock around the circle.
This transforms a static chart into a satisfying "radar scan" reveal.

---

### S6.5: Arena Entrance Animation

**As a** user,
**I want** the Arena page to animate in with model rows staggering,
**so that** the competition results feel like a reveal.

**AC-1**: Page elements appear in staggered sequence:
1. PageHeader (0ms)
2. Summary stats (100ms)
3. Table header (200ms)
4. Model rows stagger (250ms + 30ms * index)

**AC-2**: Each row uses `fade-up` from `translateY(8px)` and `opacity: 0`.

**AC-3**: The champion row (rank 1) has a delayed emphasis animation:
after the stagger completes, the row briefly pulses with an amber glow (200ms).

**AC-4**: Animations respect `prefers-reduced-motion`.

**AC-5**: The champion row displays a small crown icon (SVG, 14px) to the left of
the rank badge. On mount, the crown has a subtle gold particle burst effect
(4-6 tiny gold dots that expand and fade over 600ms).

---

## E7: Diagnostics & Calibration Page

**Current Rating**: 6/10
**Target Rating**: 9.5/10
**Files**: `src/pages/DiagnosticsPage.tsx`

The Diagnostics page has 5 tabs (PIT, Models, Matrix, Regimes, Failures). Tabs
themselves are adequate but the content within each tab is utilitarian. Recharts
defaults are unstyled, model comparison tables are flat, and detail panels lack
glass-card depth.

---

### S7.1: PIT Histogram Premium Styling

**As a** user,
**I want** the PIT histogram to use brand colours and glass tooltip,
**so that** calibration quality is presented as premium data art.

**AC-1**: Histogram bars use gradient fill:
`linear-gradient(to top, var(--accent-violet), rgba(139,92,246,0.4))`.

**AC-2**: The uniform reference line uses `var(--accent-amber)` with
`strokeDasharray="6 3"` and a `drop-shadow(0 0 4px rgba(245,197,66,0.3))`.

**AC-3**: Tooltip uses `CHART_TOOLTIP_STYLE` from the shared theme.

**AC-4**: Chart background uses `rgba(139,92,246,0.02)` fill.

**AC-5**: Bars have 2px gap between them and 4px border-radius on top corners.

**AC-6**: Bars grow upward from the baseline on mount over `var(--duration-reveal)`
with `var(--spring-bounce)`. Each bar is staggered by 40ms, creating a cascading
"equaliser" effect. This is the hero animation for the Diagnostics PIT tab.

---

### S7.2: Model Comparison Table Elevation

**As a** user,
**I want** the model comparison table to have premium visual hierarchy,
**so that** comparing models is clear and satisfying.

**AC-1**: Table uses `premium-thead` and `premium-row` classes.

**AC-2**: BMA weight column displays as a mini gradient bar
(0--100% fill using `var(--accent-violet)`) instead of a plain number.

**AC-3**: Best-in-column values have a subtle `var(--accent-emerald)` text colour
and bold weight to indicate they won that metric.

**AC-4**: Model name column uses `var(--text-luminous)` colour.

**AC-5**: Clicking a model row opens a detail panel (similar to Arena S6.2) with
the full parameter set displayed in a two-column grid.

---

### S7.3: Calibration Matrix Heat Styling

**As a** user,
**I want** the calibration matrix to use a premium heat colour scale,
**so that** the matrix is beautiful and informative.

**AC-1**: Matrix cells use a perceptual colour scale:
- 0.00 = deep void (`rgba(10,10,26,0.8)`)
- 0.25 = dark violet (`rgba(139,92,246,0.15)`)
- 0.50 = medium violet (`rgba(139,92,246,0.35)`)
- 0.75 = bright emerald (`rgba(62,232,165,0.35)`)
- 1.00 = full emerald (`rgba(62,232,165,0.6)`)

**AC-2**: Cell hover shows the exact value in a tooltip with `CHART_TOOLTIP_STYLE`.

**AC-3**: Cell hover brightens the cell by 20% using `filter: brightness(1.2)`.

**AC-4**: Axis labels use `.text-caption` typography rotated 45 degrees on the X-axis.

**AC-5**: Matrix container has a `glass-card` wrapper with `p-4`.

**AC-6**: On mount, matrix cells reveal in a staggered diagonal wave (top-left to
bottom-right), each cell fading from `opacity: 0` to its final colour with a
30ms delay per diagonal index. This transforms the matrix from a static grid into
a "scanning" reveal that takes ~800ms total for a 20x20 matrix.

---

### S7.4: Regime Distribution Chart

**As a** user,
**I want** the Regimes tab charts to use regime-specific brand colours,
**so that** each regime is instantly identifiable.

**AC-1**: Bar chart segments use the `regime-*` CSS colours:
- LOW_VOL_TREND: `var(--accent-emerald)`
- HIGH_VOL_TREND: `var(--accent-amber)`
- LOW_VOL_RANGE: `var(--accent-violet)`
- HIGH_VOL_RANGE: `var(--accent-orange)`
- CRISIS_JUMP: `var(--accent-rose)`

**AC-2**: Legend items use coloured dots (8px) with regime names, not Recharts
default boxes.

**AC-3**: Chart tooltip shows "Regime: LOW_VOL_TREND, Count: 42, Pct: 38.2%"
using `CHART_TOOLTIP_STYLE`.

**AC-4**: On regime bar hover, other bars dim to 40% opacity to highlight the
focused regime.

**AC-5**: Below the main regime bar chart, a regime transition timeline shows
the regime sequence over time as a horizontal coloured strip (each regime period
as a coloured block), giving temporal context to the frequency data.

---

### S7.5: Failures Tab Severity Styling

**As a** user,
**I want** calibration failure entries to have severity-appropriate colours,
**so that** critical failures stand out from warnings.

**AC-1**: Failure rows group by severity level:
- `CRITICAL`: Rose left border (3px solid var(--accent-rose)), rose tinted bg
- `WARNING`: Amber left border, amber tinted bg
- `INFO`: Violet left border, violet tinted bg

**AC-2**: Each severity group has a header using `.premium-section-label` with
the severity colour.

**AC-3**: Failure count badges use `status-pill` styling:
- Critical count: `.status-pill.status-fail`
- Warning count: `.status-pill` with amber styling (add `.status-warn` variant)

**AC-4**: Failure detail text uses monospace font for asset names and numeric values.

---

### S7.6: Diagnostics Tab Switching Animation

**As a** user,
**I want** switching between diagnostic tabs to have smooth content transitions,
**so that** navigation between views feels fluid.

**AC-1**: Tab content exits with `opacity: 0` and `translateY(4px)` over 150ms.

**AC-2**: New tab content enters with `opacity: 0` and `translateY(-4px)` to
`opacity: 1` and `translateY(0)` over 250ms.

**AC-3**: Tab indicator (underline) slides horizontally to the new position using
`transform: translateX()` with 250ms spring curve.

**AC-4**: Tab transitions respect `prefers-reduced-motion`.

---

### S7.7: Diagnostics Summary Hero

**As a** user,
**I want** a summary hero strip at the top of the Diagnostics page,
**so that** I see overall calibration health before diving into tabs.

**AC-1**: Three glass-card stat tiles across the top:
- "PIT Pass Rate" with emerald/rose accent: percentage of assets passing PIT
- "Avg BMA Weight" with violet accent: mean BMA weight across models
- "Active Models" with violet accent: count of non-zero-weight models

**AC-2**: Each stat tile uses `useCountUp` animation (S12.2) on mount.

**AC-3**: A small trend arrow (up/down/flat) next to each value indicates
change vs. last tuning run.

**AC-4**: Summary tiles use staggered `anim-fade-up` entrance with 60ms delay.

**AC-5**: If PIT pass rate < 75%, the tile border glows rose with a
`conviction-breathe`-style pulse to signal calibration concern.

---

## E8: Charts & Technical Analysis

**Current Rating**: 7/10
**Target Rating**: 9.5/10
**Files**: `src/pages/ChartsPage.tsx`

The Charts page wraps lightweight-charts with a symbol list sidebar. The chart view
picker is premium but the sidebar list, overlay controls, and responsive behaviour
have gaps.

---

### S8.1: Symbol List Sidebar Premium Rows

**As a** user,
**I want** the symbol list sidebar to feel like a premium navigation surface,
**so that** browsing assets is a satisfying experience.

**AC-1**: Each symbol row uses `premium-row` styling with hover glow.

**AC-2**: Active/selected symbol has a left accent border (3px solid var(--accent-violet))
and a filled background `rgba(139,92,246,0.06)`.

**AC-3**: Symbol ticker uses `var(--text-luminous)` with `font-weight: 600`.
Company name (if shown) uses `var(--text-muted)` below at `font-size: var(--font-2xs)`.

**AC-4**: Rows have a subtle `translateX(2px)` on hover to indicate interactivity.

**AC-5**: The sidebar scrollbar uses custom styling:
```css
.symbol-sidebar::-webkit-scrollbar { width: 4px; }
.symbol-sidebar::-webkit-scrollbar-thumb {
  background: rgba(139,92,246,0.15);
  border-radius: 100px;
}
.symbol-sidebar::-webkit-scrollbar-thumb:hover {
  background: rgba(139,92,246,0.25);
}
```

---

### S8.2: Sector Header Styling

**As a** user,
**I want** sector headers in the symbol list to be visually distinct,
**so that** I can quickly find the right asset group.

**AC-1**: Sector headers use `.premium-section-label` class with sticky positioning
at the top of the scroll container.

**AC-2**: Sector header has a background that blurs the content scrolling behind:
```css
.sector-header {
  position: sticky;
  top: 0;
  backdrop-filter: blur(12px);
  background: rgba(10,10,26,0.85);
  z-index: 10;
  padding: 8px 12px;
  border-bottom: 1px solid rgba(139,92,246,0.06);
}
```

**AC-3**: Sector count badge uses `var(--text-secondary)` colour in parentheses
next to the sector name.

---

### S8.3: Overlay Toggle Premium Controls

**As a** user,
**I want** chart overlay toggles (volume, indicators) to be premium pill buttons,
**so that** enabling technical indicators feels deliberate.

**AC-1**: Overlay toggles use the `.filter-pill` styling from S4.1 with `data-active`
state toggling.

**AC-2**: Active toggles have a subtle glow matching the indicator colour:
- Volume: emerald glow
- Moving averages: violet glow
- Bollinger bands: amber glow
- RSI: orange glow

**AC-3**: Toggle state change is animated with `press-spring` on click.

**AC-4**: Toggle container uses `glass-card` with `p-2` inner padding and `gap-2`
horizontal spacing.

---

### S8.4: Chart Area Premium Polish

**As a** user,
**I want** the chart area to have a premium border treatment and loading state,
**so that** the chart surface feels like a framed piece of data art.

**AC-1**: Chart container has a `glass-card` frame with `border-radius: 16px`
and `overflow: hidden`.

**AC-2**: Chart loading state shows a `skeleton-card` placeholder with shimmer
animation matching the chart dimensions.

**AC-3**: Chart crosshair tooltip uses dark glass styling consistent with
`CHART_TOOLTIP_STYLE` from the shared theme.

**AC-4**: When no symbol is selected, use the `EmptyState` component (S1.6)
with a chart icon and "Select a symbol to view its chart" message.

---

### S8.5: Charts Sidebar Responsive Behaviour

**As a** user,
**I want** the sidebar to collapse on smaller screens,
**so that** the chart gets full width on tablets and narrow windows.

**AC-1**: Below `lg` breakpoint (1024px), the sidebar collapses to a top horizontal
scrollable strip of symbol pills.

**AC-2**: The horizontal strip uses the same `filter-pill` styling as active/inactive
states.

**AC-3**: On collapse, the sidebar width transitions smoothly over 300ms with
`cubic-bezier(0.16,1,0.3,1)`.

**AC-4**: A toggle button (hamburger menu icon) allows manually opening/closing
the sidebar on any screen size.

---

## E9: Heatmap & Sentiment

**Current Rating**: 6/10
**Target Rating**: 9.5/10
**Files**: `src/pages/HeatmapPage.tsx`

The Heatmap page displays asset correlation or signal strength as a grid. The
summary strip is decent. Major gaps: cells are too small, hover effects are
minimal, tooltip lacks entrance animation, and the sentiment strip height is
anemic (5px).

---

### S9.1: Heatmap Cell Size and Hover

**As a** user,
**I want** heatmap cells to be large enough to see and interact with,
**so that** the heatmap is a usable visualisation, not just a colour blob.

**AC-1**: Minimum cell size increased from current to 32px x 32px with
responsive scaling: `min(calc(100% / gridSize), 48px)`.

**AC-2**: Cell hover shows a bright border ring:
```css
.heatmap-cell:hover {
  outline: 2px solid var(--accent-violet);
  outline-offset: -2px;
  filter: brightness(1.2);
  z-index: 10;
  transform: scale(1.05);
  transition: all 150ms cubic-bezier(0.16,1,0.3,1);
}
```

**AC-3**: Cell hover triggers the tooltip to appear (if not already).

**AC-4**: Cell border-radius is 4px for a softer grid appearance.

**AC-5**: Cell colour scale uses the perceptual scale from S7.3 for consistent
colour language across heatmap and calibration matrix.

**AC-6**: On page mount, the cell grid reveals with a staggered wave animation
starting from the centre and expanding outward in concentric rings. Each ring of
cells fades from `opacity: 0` to final opacity with a 25ms delay per ring. The
entire reveal takes ~600ms for a 16x16 grid. This transforms the heatmap from a
static image into a "radar ping" reveal that users will remember.

---

### S9.2: Heatmap Tooltip Premium Entrance

**As a** user,
**I want** the heatmap tooltip to feel like a premium data surface,
**so that** cell inspection is informative and visually rich.

**AC-1**: Tooltip uses `CHART_TOOLTIP_STYLE` (glass background, blur, shadow).

**AC-2**: Tooltip enters with a 100ms fade-in and subtle scale from 0.95 to 1.0:
```css
.heatmap-tooltip-enter {
  animation: tooltip-enter 100ms cubic-bezier(0.16,1,0.3,1) forwards;
}
@keyframes tooltip-enter {
  from { opacity: 0; transform: scale(0.95) translateY(4px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}
```

**AC-3**: Tooltip shows:
- Asset pair (row label x column label) in `var(--text-luminous)`
- Correlation/signal value in `.text-stat-value` size
- A small colour swatch matching the cell colour
- Regime context if available

**AC-4**: Tooltip follows the cursor with a 12px offset and stays within viewport.

---

### S9.3: Sentiment Strip Enhancement

**As a** user,
**I want** the sentiment strip at the top/bottom of the heatmap to be a substantial
visual indicator,
**so that** market sentiment is visible at a glance.

**AC-1**: Strip height increased from 5px to 12px with `border-radius: 6px`.

**AC-2**: Strip uses a multi-stop gradient:
- Full bear: `var(--accent-rose)`
- Neutral: `var(--accent-violet)`
- Full bull: `var(--accent-emerald)`
With the current sentiment position marked by a small triangle pointer (8px).

**AC-3**: Strip has a subtle box-shadow for depth:
`box-shadow: 0 2px 8px rgba(0,0,0,0.2), inset 0 1px 2px rgba(255,255,255,0.04)`.

**AC-4**: The pointer position animates when sentiment changes using 400ms spring.

**AC-5**: A label below the strip shows the exact sentiment value using `.text-caption`
typography.

**AC-6**: At the bear and bull extremes of the gradient, a soft glow pulse
breathes every 4 seconds, drawing attention to whether the current sentiment
is near an extreme. The pulse intensity scales with how close the pointer is
to the extreme (within 10% of either end triggers the pulse).

---

### S9.4: Heatmap Summary Strip Cards

**As a** user,
**I want** summary statistics above the heatmap to be premium stat cards,
**so that** aggregate metrics have the same quality as the rest of the app.

**AC-1**: Summary cards use `glass-card` with `p-4` and `hover-lift`.

**AC-2**: Card metric values use `.text-stat-value` typography.

**AC-3**: Card labels use `.text-label` typography (uppercase, wide tracking).

**AC-4**: Cards displaying positive metrics get a subtle emerald border-left (2px),
negative get rose, neutral get violet.

**AC-5**: Cards have staggered entrance animation with 50ms delay per card.

---

### S9.5: Heatmap Axis Labels

**As a** user,
**I want** the heatmap axis labels to be legible and styled,
**so that** I can identify assets without squinting.

**AC-1**: X-axis labels rotate 45 degrees with `transform-origin: top left`.

**AC-2**: Labels use `font-size: var(--font-2xs)` with `letter-spacing: var(--tracking-wide)`.

**AC-3**: Hovered row/column labels highlight in `var(--text-luminous)` when a cell
in their row or column is hovered (cross-highlight).

**AC-4**: Labels use `var(--text-muted)` colour by default, transitioning to luminous
on hover over 150ms.

---

### S9.6: Heatmap Zone Annotation Layer

**As a** user,
**I want** to see labelled zones on the heatmap (e.g. "High Correlation Cluster",
"Sector Divergence"),
**so that** patterns are contextualised with human-readable annotations.

**AC-1**: An optional `showAnnotations` toggle (default off) in the heatmap controls
enables zone overlays.

**AC-2**: Annotations render as semi-transparent rounded rectangles (`rgba(139,92,246,0.04)`)
over cell clusters with a dashed violet border and a `.text-caption` label
positioned at the top-left corner of the zone.

**AC-3**: Annotations are computed from a simple clustering heuristic:
cells with correlation > 0.7 forming a connected group of 3+ cells.

**AC-4**: Zone rectangles fade in with `anim-fade-in` when the toggle is enabled.

**AC-5**: Hovering a zone annotation dims cells outside the zone to 30% opacity,
focusing visual attention on the cluster.

---

## E10: Tuning Mission Control

**Current Rating**: 6/10
**Target Rating**: 9.5/10
**Files**: `src/pages/TuningPage.tsx`

The Tuning page is a "mission control" for model parameter estimation. The control
panel header is polished but the star map grid cells are small (20x20px), the
running state lacks a dramatic indicator, and the PIT summary bar is flat.

---

### S10.1: Star Map Grid Cell Enhancement

**As a** user,
**I want** star map grid cells to be larger and more expressive,
**so that** the tuning status of each asset is immediately visible.

**AC-1**: Cell size increases from 20x20 to 32x32 minimum, with responsive scaling:
```css
.star-cell {
  width: 32px;
  height: 32px;
  border-radius: 6px;
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 200ms cubic-bezier(0.16,1,0.3,1);
}
```

**AC-2**: Cell status colours:
- Tuned (fresh): `var(--accent-emerald)` with steady glow
- Tuned (stale): `var(--accent-amber)` at 60% opacity
- Running: `var(--accent-violet)` with pulse animation
- Failed: `var(--accent-rose)` with 40% opacity
- Pending: `var(--text-muted)` at 20% opacity

**AC-3**: Running cells have a breathing pulse:
```css
.star-cell-running {
  animation: star-pulse 1.5s ease-in-out infinite;
}
@keyframes star-pulse {
  0%, 100% { box-shadow: 0 0 8px rgba(139,92,246,0.3); transform: scale(1); }
  50% { box-shadow: 0 0 16px rgba(139,92,246,0.5); transform: scale(1.1); }
}
```

**AC-4**: Cell hover shows asset ticker tooltip with `CHART_TOOLTIP_STYLE`.

**AC-5**: Grid has a subtle background grid pattern:
```css
.star-grid {
  background-image: radial-gradient(circle, rgba(139,92,246,0.04) 1px, transparent 1px);
  background-size: 40px 40px;
}
```

**AC-6**: When a cell transitions from "Running" to "Tuned", it plays a success
flash: a brief emerald ring expands outward (scale 1.0 to 2.0, opacity 1.0 to 0)
over 400ms, like a ripple on water. This transforms the grid from a static status
board into a live mission control surface.

---

### S10.2: Mission Control Button Upgrade

**As a** user,
**I want** the "Tune", "Tune All", and "Stop" buttons to feel like premium actions,
**so that** triggering a tuning run is a deliberate, satisfying action.

**AC-1**: Primary action buttons (Tune, Tune All) use solid violet fill:
```css
.btn-primary {
  background: linear-gradient(135deg, var(--accent-violet), rgba(139,92,246,0.8));
  color: white;
  padding: 10px 24px;
  border-radius: 10px;
  font-size: var(--font-sm);
  font-weight: var(--weight-semibold);
  border: 1px solid rgba(139,92,246,0.3);
  box-shadow: 0 4px 16px rgba(139,92,246,0.2);
  transition: all 200ms cubic-bezier(0.16,1,0.3,1);
}
.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 6px 24px rgba(139,92,246,0.3);
}
```

**AC-2**: Destructive buttons (Stop) use rose variant.

**AC-3**: Disabled buttons use `.state-disabled` class.

**AC-4**: All buttons include `press-spring` active state.

**AC-5**: During tuning, the "Tune" button shows a mini spinner icon with pulse effect.

---

### S10.3: PIT Summary Bar Premium Styling

**As a** user,
**I want** the PIT summary bar to display pass/fail calibration quality as a
premium visualisation,
**so that** overall calibration health is clear at a glance.

**AC-1**: PIT bar segments use gradient fills:
- Pass segment: `linear-gradient(90deg, rgba(62,232,165,0.15), rgba(62,232,165,0.3))`
- Fail segment: `linear-gradient(90deg, rgba(255,107,138,0.15), rgba(255,107,138,0.3))`

**AC-2**: Bar height increased to 12px with `border-radius: 6px`.

**AC-3**: Segment widths animate from 0 on mount with 600ms spring transition.

**AC-4**: Percentage labels appear inside segments (if wide enough) or above them,
using `.text-caption` typography with contrasting colour.

**AC-5**: Bar container is a `glass-card` with `p-4`.

---

### S10.4: Retune Log Stagger Animation

**As a** user,
**I want** tuning log entries to appear with staggered animations,
**so that** the tuning process feels like a live mission log.

**AC-1**: New log entries slide in from the left:
```css
@keyframes log-entry {
  from { opacity: 0; transform: translateX(-16px); }
  to   { opacity: 1; transform: translateX(0); }
}
```

**AC-2**: Each entry has a 30ms stagger delay (max 10 visible entries staggered).

**AC-3**: Log entries use monospace font for asset names and model parameters.

**AC-4**: Success entries have a left emerald border (2px), failure entries have
a left rose border, active entries have a pulsing violet left border.

**AC-5**: The log auto-scrolls to the latest entry with smooth scroll behaviour.

---

### S10.5: Tuning Progress Indicator

**As a** user,
**I want** a premium progress indicator during batch tuning,
**so that** I know how far along the process is and the estimated remaining time.

**AC-1**: Progress bar uses a gradient fill with animated shimmer:
```css
.tuning-progress-bar {
  height: 6px;
  border-radius: 3px;
  background: rgba(139,92,246,0.08);
  overflow: hidden;
}
.tuning-progress-fill {
  height: 100%;
  background: linear-gradient(90deg, var(--accent-violet), var(--accent-emerald));
  border-radius: 3px;
  transition: width 300ms cubic-bezier(0.16,1,0.3,1);
  box-shadow: 0 0 8px rgba(139,92,246,0.3);
}
```

**AC-2**: Above the bar: "Tuning 42/100 assets" with `.text-body` typography. Below:
"~3m remaining" with `.text-caption` typography.

**AC-3**: Completed progress bar pulses once with emerald glow to confirm success.

**AC-4**: Progress bar is inside a `glass-card` with `p-4`.

**AC-5**: When batch tuning completes (100%), a celebration sequence fires:
1. Progress bar flashes emerald and emits a pulse wave
2. All star map cells that were tuned play a staggered success ripple (S10.1 AC-6)
3. A large "Tuning Complete" banner fades in with `anim-scale-in` and a gradient
   text effect (`text-gradient-brand`)
4. The banner auto-dismisses after 3 seconds with `anim-fade-out`

This is the most satisfying moment in the app -- a reward for waiting. LumenLingo
celebrates every success; so do we.

---

### S10.6: Tuning Completion Celebration

**As a** user,
**I want** a dramatic visual celebration when a full tuning run completes,
**so that** the wait feels rewarded and the milestone is unmistakable.

**AC-1**: 12-16 small particle dots (3-5px) emanate from the centre of the star
map in random directions, fading over 800ms. Particles use the `--accent-emerald`
and `--accent-violet` colours at random.

**AC-2**: The page header title briefly flashes with `text-gradient-brand` effect
then returns to normal.

**AC-3**: A "session summary" glass-card slides up from the bottom showing:
- Total assets tuned
- PIT pass rate
- Mean BMA weight
- Duration ("Completed in 4m 23s")
using `.text-stat-value` for numbers and `.text-label` for captions.

**AC-4**: The summary card has a gold top border:
`border-top: 2px solid var(--accent-amber)`.

**AC-5**: Summary card auto-dismisses after 5s or on click, with `anim-fade-up`
in reverse (slide down + fade).

---

## E11: Services & Health

**Current Rating**: 7/10
**Target Rating**: 9/10
**Files**: `src/pages/ServicesPage.tsx`

The Services page monitors backend service health. The status hero is polished.
Gaps: service cards lack hover animation, error log is flat, metric
icons lack background badges.

---

### S11.1: Service Card Hover Animation

**As a** user,
**I want** service cards to respond to hover with lift and glow,
**so that** inspecting service status feels interactive.

**AC-1**: Service cards use `glass-card hover-lift` with `p-6`.

**AC-2**: Hover lifts the card `translateY(-3px)` and increases box-shadow to
`var(--shadow-card-hover)`.

**AC-3**: Service status indicator (healthy/unhealthy) uses a coloured dot:
- Healthy: emerald with `ws-pulse` animation (same as WebSocket indicator)
- Unhealthy: rose with no pulse, steady glow
- Unknown: amber with slow pulse

**AC-4**: Service uptime percentage uses `.text-stat-value` typography.

**AC-5**: Cards have staggered entrance animation (50ms delay per card).

**AC-6**: Service cards respond to the 3D perspective tilt from `.hover-lift`
(S1.4 AC-1). The directional tilt makes cards feel like physical status panels
in a real operations centre.

---

### S11.2: Metric Icon Background Badge

**As a** user,
**I want** metric icons on service cards to have a coloured background badge,
**so that** icons have visual weight and are not floating bare.

**AC-1**: Icon container:
```css
.metric-icon-badge {
  display: flex;
  align-items: center;
  justify-content: center;
  width: 36px;
  height: 36px;
  border-radius: 10px;
  background: rgba(139,92,246,0.08);
  color: var(--accent-violet);
}
```

**AC-2**: Different metric types use different accent colours:
- CPU/Memory: violet
- Response time: emerald (fast) / amber (slow) / rose (timeout)
- Error count: rose if > 0, emerald if 0

**AC-3**: Icon inside badge is 18px (not the full badge size).

**AC-4**: On card hover, the icon badge plays a subtle animation specific to
the metric type:
- CPU: icon rotates 180deg
- Response time: icon bounces once
- Error count: icon shakes horizontally (2px x 3 cycles)
- Default: icon scales to 1.1x and back
This micro-detail rewards cursor exploration.

---

### S11.3: Error Log Premium Rows

**As a** user,
**I want** the error log to feel like a premium data surface,
**so that** reviewing errors is clear and structured.

**AC-1**: Error log container uses `glass-card` with `p-4`.

**AC-2**: Each error row uses `premium-row` styling with:
- Timestamp in `var(--text-muted)` monospace
- Error message in `var(--text-primary)`
- Severity badge (CRITICAL/WARNING/INFO) using `status-pill` variants

**AC-3**: Critical errors have a left rose border (3px).

**AC-4**: Error log has a search/filter input at the top with `focus-ring` styling.

**AC-5**: Empty error log (no errors) shows the `EmptyState` component with a
shield-check icon and "No errors recorded" message with emerald accent.

---

### S11.4: Status Hero Entrance Animation

**As a** user,
**I want** the status hero section to animate dramatically on page load,
**so that** the overall system health is presented with impact.

**AC-1**: The main status text ("All Systems Operational" / "Degraded") fades up
with blur-to-sharp animation over 400ms.

**AC-2**: The status icon scales from 0.8 to 1.0 with spring overshoot.

**AC-3**: The status badge background glow pulses once on entrance to draw attention.

**AC-4**: Sub-metrics below the hero (latency, uptime, etc.) stagger in at 80ms
intervals.

---

### S11.5: Uptime Sparkline Timeline

**As a** user,
**I want** each service card to show a 30-day uptime sparkline,
**so that** I can see historical reliability at a glance.

**AC-1**: A mini sparkline (120px wide, 24px tall) appears at the bottom of each
service card showing up/down status over the last 30 days.

**AC-2**: The sparkline uses a step function: up periods are `var(--accent-emerald)`
at full height, down periods are `var(--accent-rose)` at a lower height, creating
a clear binary visual.

**AC-3**: The sparkline line draws left-to-right on card entrance using the
`stroke-dashoffset` animation pattern from S1.8 AC-6.

**AC-4**: Hovering the sparkline shows a tooltip with the specific date and uptime
percentage for that day.

**AC-5**: If no historical data exists, the sparkline area shows a subtle dashed
line at 100% as a placeholder, not an empty void.

---

## E12: Profitability Analytics

**Current Rating**: 5/10 (weakest page)
**Target Rating**: 9.5/10
**Files**: `src/pages/ProfitabilityPage.tsx`

The Profitability page is the weakest in the app. Cards exist but charts use bare
Recharts defaults, numbers are static, tooltips are plain, and the reference line
for targets has no visual emphasis. This page needs the most dramatic transformation.

---

### S12.1: MetricCard Gradient Background

**As a** user,
**I want** profitability metric cards to have subtle gradient backgrounds that
indicate performance,
**so that** positive/negative results are felt before reading the number.

**AC-1**: Positive metric cards get an emerald gradient glow:
```css
.metric-card-positive {
  background: linear-gradient(135deg, rgba(62,232,165,0.04) 0%, transparent 50%);
  border-left: 2px solid rgba(62,232,165,0.3);
}
```

**AC-2**: Negative metric cards get a rose gradient:
```css
.metric-card-negative {
  background: linear-gradient(135deg, rgba(255,107,138,0.04) 0%, transparent 50%);
  border-left: 2px solid rgba(255,107,138,0.3);
}
```

**AC-3**: Neutral metric cards use the violet gradient:
```css
.metric-card-neutral {
  background: linear-gradient(135deg, rgba(139,92,246,0.03) 0%, transparent 50%);
  border-left: 2px solid rgba(139,92,246,0.15);
}
```

**AC-4**: Cards use `glass-card hover-lift` classes with `p-6`.

**AC-5**: Metric title uses `.text-label`, value uses `.text-stat-value`, and the
delta/change indicator uses `.text-caption` with sentiment colour.

**AC-6**: Cards respond to the 3D perspective tilt from `.hover-lift` (S1.4).
The directional tilt combined with the sentiment gradient makes profitability
cards feel like premium financial data surfaces.

---

### S12.2: Number Counting Animation

**As a** user,
**I want** profitability numbers (CAGR, Sharpe, etc.) to count up from zero to
their final value on page load,
**so that** the reveal of performance data is dramatic and engaging.

**AC-1**: Create a `useCountUp` hook:
```typescript
function useCountUp(target: number, duration: number = 800, decimals: number = 2): string
```

**AC-2**: The counting animation uses `requestAnimationFrame` with an ease-out curve
(cubic-bezier output mapped to progress).

**AC-3**: During counting, the number uses `font-variant-numeric: tabular-nums` to
prevent layout shift.

**AC-4**: Counting starts when the card enters the viewport (IntersectionObserver)
not on page mount, so scrolled-into-view cards animate on reveal.

**AC-5**: Numbers that represent percentages, ratios, and currency all format correctly
during the count animation (e.g. "0.00%" counting to "12.45%").

**AC-6**: When a counted number reaches its final value, it flashes briefly with
sentiment colour (emerald for positive, rose for negative) for 300ms, then settles
to its default colour. This "arrival flash" creates a micro-celebration per metric.

---

### S12.3: Recharts Brand Palette Override

**As a** user,
**I want** all profitability charts to use the app's colour palette,
**so that** charts look integrated rather than default.

**AC-1**: Equity curve chart uses `var(--accent-violet)` stroke with gradient fill:
```css
linearGradient(to bottom, rgba(139,92,246,0.15) 0%, rgba(15,15,35,0) 100%)
```

**AC-2**: Drawdown chart uses `var(--accent-rose)` stroke with rose gradient fill.

**AC-3**: All chart `CartesianGrid` uses `CHART_COLORS.grid` from the shared theme.

**AC-4**: All chart `XAxis` and `YAxis` use `CHART_AXIS_STYLE` from the shared theme.

**AC-5**: Area between positive and negative returns shaded with emerald (above zero)
and rose (below zero) using `<defs>` gradient definitions.

**AC-6**: The equity curve line draws left-to-right on mount over 1200ms using the
`useAnimatedLine` hook (S1.8 AC-6). The gradient area fill follows the line
draw, appearing to "paint" the area as the line progresses. This is the hero
animation of the Profitability page.

---

### S12.4: Chart Tooltip Glass-Morphism

**As a** user,
**I want** chart tooltips to use the premium glass-morphic style,
**so that** hovering over charts feels consistent with the rest of the app.

**AC-1**: All Recharts `<Tooltip>` components use `CHART_TOOLTIP_STYLE` from the
shared theme.

**AC-2**: Tooltip content layout:
- Date in `.text-caption` typography at the top
- Main value in `.text-section` typography with accent colour
- Secondary values in `.text-body` typography with labels

**AC-3**: Tooltip has a 100ms fade-in entrance animation.

**AC-4**: Custom tooltip component created as `PremiumTooltip.tsx` and reused across
all chart pages (Profitability, Risk, Diagnostics, Charts).

---

### S12.5: Target Reference Line Glow

**As a** user,
**I want** target/benchmark reference lines on charts to have a glowing visual
emphasis,
**so that** targets are clearly distinguished from actual data.

**AC-1**: Reference lines use dashed stroke (`strokeDasharray="8 4"`) with
`var(--accent-amber)` colour.

**AC-2**: Reference lines have a glow drop-shadow:
```
filter: drop-shadow(0 0 6px rgba(245,197,66,0.3))
```

**AC-3**: Reference line label uses `.text-caption` typography positioned above
the line with a small glass-card background badge:
```css
.reference-label {
  background: rgba(15,15,35,0.85);
  backdrop-filter: blur(8px);
  padding: 2px 8px;
  border-radius: 4px;
  border: 1px solid rgba(245,197,66,0.15);
  font-size: var(--font-2xs);
  color: var(--accent-amber);
}
```

**AC-4**: Multiple reference lines (e.g. target Sharpe, benchmark CAGR) use
different dash patterns and accent colours to differentiate.

---

### S12.6: Equity Curve Full-Width Hero

**As a** user,
**I want** the equity curve chart to span the full page width as a hero element,
**so that** portfolio performance is presented with dramatic visual impact.

**AC-1**: Equity curve chart occupies full width of the content area with a height
of 320px (desktop) / 240px (tablet) / 180px (mobile).

**AC-2**: The chart area is a `glass-card` with extra padding (`p-0` for chart
area, `p-4` for the title strip above).

**AC-3**: Above the chart: a title strip with "Portfolio Equity Curve" in
`.text-heading` and the current total return in `.text-stat-value` with sentiment
colour.

**AC-4**: The chart has a crosshair interaction that shows a vertical line at the
hover position with the date, portfolio value, and drawdown percentage.

**AC-5**: Below the equity curve, a drawdown chart (120px height) mirrors the same
full width, using rose gradient fill. The two charts share the same X-axis zoom.

---

### S12.7: Performance Comparison Overlay

**As a** user,
**I want** to toggle a benchmark overlay (SPY, equal-weight) on the equity curve,
**so that** relative performance is immediately visible.

**AC-1**: A toggle strip below the title (using `.filter-pill` styling from S4.1)
offers: "Portfolio", "vs SPY", "vs Equal-Weight".

**AC-2**: Toggling a benchmark adds a second line to the chart in `var(--accent-amber)`
with a dashed stroke (`strokeDasharray: "6 3"`).

**AC-3**: The benchmark line draws in with the same `stroke-dashoffset` animation.

**AC-4**: The area between portfolio and benchmark is shaded:
- Outperformance: faint emerald fill
- Underperformance: faint rose fill

**AC-5**: A "vs Benchmark" metric card appears showing the alpha, tracking error,
and information ratio with `useCountUp` animation.

---

## E13: Shared Components

**Current Rating**: 8/10
**Target Rating**: 9.5/10
**Files**: Layout.tsx, PageHeader.tsx, LoadingSpinner.tsx, StatCard.tsx, AmbientOrbs.tsx

Shared components are the highest-rated elements but still have colour hardcoding,
missing entrance animations, and opportunities for micro-interaction polish.

---

### S13.1: Layout Sidebar Badge Variable Migration

**As a** user,
**I want** sidebar notification badges to use CSS variables,
**so that** badge colours are centrally controlled.

**AC-1**: Badge background colours use semantic CSS classes:
```css
.badge-count {
  padding: 1px 6px;
  border-radius: 100px;
  font-size: var(--font-2xs);
  font-weight: var(--weight-bold);
  min-width: 18px;
  text-align: center;
}
.badge-count-violet { background: rgba(139,92,246,0.15); color: var(--accent-violet); }
.badge-count-emerald { background: rgba(62,232,165,0.15); color: var(--accent-emerald); }
.badge-count-rose { background: rgba(255,107,138,0.15); color: var(--accent-rose); }
.badge-count-amber { background: rgba(245,197,66,0.15); color: var(--accent-amber); }
```

**AC-2**: Active nav item badge pulses once when count changes (animation trigger
on value update).

**AC-3**: Badge entrance uses `scale(0)` to `scale(1)` spring animation when it
first appears.

**AC-4**: Badge count numbers use the `useCountUp` micro-animation (S12.2) when
the count changes, making badge updates feel alive rather than silently swapping.

---

### S13.2: PageHeader Glow Variable Extraction

**As a** user,
**I want** PageHeader glow and text-shadow styles extracted to CSS classes,
**so that** the component is maintainable and themed.

**AC-1**: Header text shadow uses CSS variable:
```css
.page-title-glow {
  text-shadow: 0 0 40px rgba(139,92,246,0.15);
}
```

**AC-2**: The gradient text effect on the title uses a CSS class:
```css
.text-gradient-brand {
  background: linear-gradient(135deg, var(--text-luminous) 0%, var(--accent-violet) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
}
```

**AC-3**: Subtitle uses `.text-body` typography with `var(--text-secondary)`.

**AC-4**: PageHeader has a fade-up entrance animation (first element, 0ms delay).

**AC-5**: On first visit to a page, the title text renders with a typewriter effect
(characters appearing left-to-right over 400ms). On subsequent SPA navigations,
the title renders instantly. The typewriter uses `opacity` per character, not
`width` clipping, so no layout shift occurs. This creates a premium "introducing"
feel on first encounter.

---

### S13.3: LoadingSpinner Cosmic Upgrade

**As a** user,
**I want** the loading spinner to feel like a cosmic loading experience,
**so that** waiting is visually interesting rather than boring.

**AC-1**: Replace the current spinner with an orbital animation:
```css
.cosmic-loader {
  position: relative;
  width: 48px;
  height: 48px;
}
.cosmic-loader::before,
.cosmic-loader::after {
  content: '';
  position: absolute;
  inset: 0;
  border-radius: 50%;
  border: 2px solid transparent;
}
.cosmic-loader::before {
  border-top-color: var(--accent-violet);
  animation: cosmic-spin 1.2s cubic-bezier(0.5,0,0.5,1) infinite;
}
.cosmic-loader::after {
  border-right-color: var(--accent-emerald);
  animation: cosmic-spin 1.2s cubic-bezier(0.5,0,0.5,1) infinite reverse;
  inset: 4px;
}
@keyframes cosmic-spin {
  to { transform: rotate(360deg); }
}
```

**AC-2**: A small pulsing dot at the centre:
```css
.cosmic-loader .core {
  position: absolute;
  top: 50%; left: 50%;
  width: 6px; height: 6px;
  margin: -3px;
  border-radius: 50%;
  background: var(--accent-violet);
  animation: core-pulse 1.2s ease-in-out infinite;
}
```

**AC-3**: Loading text below uses `.text-caption` typography with a subtle
fade-in/out animation.

**AC-4**: The spinner has size variants: `sm` (24px), `md` (48px), `lg` (72px).

**AC-5**: Around the outer ring, 4-6 tiny cosmic dust particles (2px circles at
0.4 opacity) orbit at varying speeds and distances, creating a miniature
solar system effect. Particles use `--accent-violet` and `--accent-emerald`
alternating. This transforms a loading spinner from a boring wait into a
mesmerising micro-animation that users actually enjoy watching.

---

### S13.4: StatCard Entrance Animation

**As a** user,
**I want** StatCards to animate in with staggered reveals,
**so that** the overview dashboard loading feels dramatic.

**AC-1**: StatCards use the `fade-up` animation class with staggered delays:
```css
.stat-card-enter {
  animation: fade-up 500ms cubic-bezier(0.16,1,0.3,1) both;
}
```

**AC-2**: Stagger delay is calculated from the card index: `delay: index * 60ms`.

**AC-3**: The glow effect on StatCard intensifies briefly (300ms) after the entrance
animation completes, then settles to normal intensity.

**AC-4**: Animations respect `prefers-reduced-motion`.

---

### S13.5: AmbientOrbs Performance Optimisation

**As a** user,
**I want** ambient orbs to be GPU-accelerated and not cause layout thrashing,
**so that** the background animation is smooth on all devices.

**AC-1**: Orbs use `will-change: transform` and `contain: strict` for GPU compositing.

**AC-2**: Orb animation uses only `transform` and `opacity` (no `top`, `left`,
`width`, `height` animations).

**AC-3**: On devices with `prefers-reduced-motion`, orbs are static (no animation)
but still visible with fixed positions.

**AC-4**: Orb count reduces on smaller screens: 3 orbs on desktop, 2 on tablet,
1 on mobile.

**AC-5**: Orbs react to the current page context by shifting their hue. Each page
sets a `data-page-accent` attribute on the layout:
- Overview/Signals: violet (default)
- Risk/Diagnostics: amber tint (mixing warm into the orb colours)
- Arena: emerald tint (competition/champion feel)
- Charts/Heatmap: cool blue tint (analytical feel)
The hue shift transitions over 800ms when navigating between pages, creating a
subtle ambient colour adaptation that most users feel but can't quite identify.

---

### S13.6: Global Micro-Interaction Library

**As a** user,
**I want** a set of reusable micro-interaction CSS classes I can compose,
**so that** adding premium interactions to any element is a one-class addition.

**AC-1**: Define the following utility animation classes:
```css
/* Entrance animations */
.anim-fade-up     { animation: fade-up 500ms cubic-bezier(0.16,1,0.3,1) both; }
.anim-fade-in     { animation: fade-in 300ms ease both; }
.anim-scale-in    { animation: scale-in 300ms cubic-bezier(0.16,1,0.3,1) both; }
.anim-slide-left  { animation: slide-left 400ms cubic-bezier(0.16,1,0.3,1) both; }
.anim-slide-right { animation: slide-right 400ms cubic-bezier(0.16,1,0.3,1) both; }

/* Delay modifiers */
.delay-1 { animation-delay: 50ms; }
.delay-2 { animation-delay: 100ms; }
.delay-3 { animation-delay: 150ms; }
.delay-4 { animation-delay: 200ms; }
.delay-5 { animation-delay: 250ms; }
.delay-6 { animation-delay: 300ms; }
.delay-8 { animation-delay: 400ms; }
.delay-10 { animation-delay: 500ms; }

/* Interactive overlays */
.glow-on-hover  { ... } /* Soft violet glow on hover */
.shine-on-hover { ... } /* Moving light sweep on hover */
```

**AC-2**: Define corresponding `@keyframes` for each animation.

**AC-3**: All animations are disabled under `prefers-reduced-motion`:
```css
@media (prefers-reduced-motion: reduce) {
  .anim-fade-up, .anim-fade-in, .anim-scale-in,
  .anim-slide-left, .anim-slide-right {
    animation: none !important;
    opacity: 1 !important;
    transform: none !important;
  }
}
```

**AC-4**: Document each class name and its visual effect in a comment block at
the top of the animation section in index.css.

**AC-5**: Add advanced animations for premium moments:
```css
/* 3D flip -- for card reveals and mode toggles */
.anim-3d-flip {
  animation: flip-in 600ms var(--spring-bounce) both;
  transform-style: preserve-3d;
}
@keyframes flip-in {
  from { transform: rotateY(90deg); opacity: 0; }
  to { transform: rotateY(0); opacity: 1; }
}

/* Bounce-in -- for celebration elements */
.anim-bounce-in {
  animation: bounce-in 500ms var(--spring-snappy) both;
}
@keyframes bounce-in {
  0% { transform: scale(0); }
  60% { transform: scale(1.15); }
  100% { transform: scale(1); }
}

/* Shine sweep -- for premium card hover accent */
.shine-sweep::after {
  content: '';
  position: absolute;
  top: 0; left: -100%; width: 50%; height: 100%;
  background: linear-gradient(90deg, transparent, rgba(255,255,255,0.04), transparent);
  pointer-events: none;
}
.shine-sweep:hover::after {
  animation: sweep 800ms ease forwards;
}
@keyframes sweep {
  to { left: 150%; }
}
```

**AC-6**: Add a `.ripple` click effect that spawns an expanding circle at the click
position (similar to Material Design ripple but using `--accent-violet` at 0.06
opacity). Suitable for buttons and clickable surfaces:
```css
.ripple {
  position: relative;
  overflow: hidden;
}
.ripple::after {
  content: '';
  position: absolute;
  border-radius: 50%;
  background: rgba(139,92,246,0.06);
  transform: scale(0);
  pointer-events: none;
}
.ripple:active::after {
  animation: ripple-expand 600ms ease forwards;
}
@keyframes ripple-expand {
  to { transform: scale(4); opacity: 0; }
}
```

---

## Implementation Priority

### Phase 1: Foundation (Must-Have)
- S1.1: Typographic Scale
- S1.2: Spacing Scale
- S1.3: Colour Variable Consolidation
- S1.4: Interactive State System
- S1.8: Recharts Theme Configuration
- S13.6: Global Micro-Interaction Library

### Phase 2: Worst Pages First
- E12: Profitability Analytics (currently 5/10)
- E9: Heatmap & Sentiment (currently 6/10)
- E10: Tuning Mission Control (currently 6/10)
- E4: Signals Page (currently 6/10)
- E5: Data Management (currently 6/10)
- E7: Diagnostics & Calibration (currently 6/10)

### Phase 3: Good-to-Great
- E6: Arena Competition (7/10 -> 9.5/10)
- E3: Risk Dashboard (7/10 -> 9.5/10)
- E8: Charts & Technical Analysis (7/10 -> 9.5/10)
- E11: Services & Health (7/10 -> 9/10)

### Phase 4: Polish & Components
- S1.5: Premium Loading Skeletons
- S1.6: Premium Empty States
- S1.7: Premium Error States
- E2: Overview Page (8/10 -> 9.5/10)
- E13: Shared Components (8/10 -> 9.5/10)

---

## Completion Criteria

The UX overhaul is complete when:

1. **Zero hardcoded colours** remain in any `.tsx` file (verified via `grep -rn '#[0-9a-fA-F]\{6\}' src/pages/ src/components/`)
2. **Every page scores 9+** on the premium assessment scale
3. **All interactive elements** have defined hover, active, focus, and disabled states
4. **3D perspective tilt** works on all glass-card surfaces with cursor-tracking
5. **Recharts** across all pages use the shared `chartTheme.ts`
6. **Loading skeletons** replace spinners on every data-fetching page
7. **Empty states** use the `EmptyState` component everywhere
8. **Entrance choreography** exists on every page with staggered reveals
9. **Scroll-triggered data reveals** animate numbers and bars when they enter viewport
10. **Celebration moments** exist for: tuning completion, high-conviction signals, and championship wins
11. **Noise texture overlay** is visible on all glass-card surfaces
12. **Custom scrollbars** are styled throughout (no OS-default scrollbars visible)
13. **Spring physics tokens** are used consistently (no ad-hoc `cubic-bezier` values)
14. **`prefers-reduced-motion`** is respected throughout (all animations, particles, tilt)
15. **Lighthouse performance** stays above 90 (animations are GPU-accelerated, no layout thrashing)
16. **Visual regression tests** pass for all 11 pages at 1280x800 viewport
17. **Every chart line draws** on mount (no static pre-rendered charts)
18. **Font stack** is Inter with tabular-nums for numeric content
