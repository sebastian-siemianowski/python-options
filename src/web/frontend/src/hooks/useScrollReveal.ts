/**
 * useScrollReveal - Scroll-triggered materialization (Story 10.6)
 *
 * Uses IntersectionObserver to add 'is-visible' class when elements
 * enter the viewport. Triggers once per element. Respects
 * prefers-reduced-motion.
 */
import { useEffect, useRef } from 'react';

export function useScrollReveal(staggerMs = 40) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Respect reduced motion preference
    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      container.querySelectorAll('.scroll-reveal').forEach((el) => {
        el.classList.add('is-visible');
      });
      return;
    }

    const observer = new IntersectionObserver(
      (entries) => {
        // Collect all newly visible entries
        const visible = entries.filter((e) => e.isIntersecting && !e.target.getAttribute('data-animated'));
        visible.forEach((entry, i) => {
          const el = entry.target as HTMLElement;
          el.setAttribute('data-animated', 'true');
          // Stagger timing for simultaneous entries
          setTimeout(() => {
            el.classList.add('is-visible');
          }, i * staggerMs);
        });
      },
      { threshold: 0.1 },
    );

    container.querySelectorAll('.scroll-reveal').forEach((el) => {
      // Skip elements above the fold (already visible on load)
      const rect = el.getBoundingClientRect();
      if (rect.top < window.innerHeight) {
        el.classList.add('is-visible');
        (el as HTMLElement).setAttribute('data-animated', 'true');
      } else {
        observer.observe(el);
      }
    });

    return () => observer.disconnect();
  }, [staggerMs]);

  return containerRef;
}
