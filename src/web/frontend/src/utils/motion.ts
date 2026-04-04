/**
 * motion.ts -- Spring physics & duration tokens (S1.11)
 *
 * JS-side mirror of the CSS motion custom properties.
 * Use these when animating via JS / Framer Motion / GSAP.
 */

export const SPRING = {
  bounce: { type: 'spring' as const, stiffness: 300, damping: 20 },
  snappy: { type: 'spring' as const, stiffness: 400, damping: 15 },
  gentle: { type: 'spring' as const, stiffness: 200, damping: 25 },
} as const;

export const DURATION = {
  instant: 80,
  fast:    150,
  normal:  250,
  slow:    400,
  reveal:  600,
  epic:    1000,
} as const;

export const EASING = {
  springBounce: [0.16, 1, 0.3, 1],
  springSnappy: [0.34, 1.56, 0.64, 1],
  springGentle: [0.22, 1, 0.36, 1],
  easeOutExpo:  [0.16, 1, 0.3, 1],
  easeInOut:    [0.4, 0, 0.2, 1],
} as const;
