import { useEffect, useRef, useState } from 'react';

/**
 * useCountUp - Animate a number from 0 to target value.
 * Respects prefers-reduced-motion by showing final value instantly.
 */
export function useCountUp(target: number | string, duration = 800): string {
  const numericTarget = typeof target === 'string' ? parseFloat(target) : target;
  const isNumeric = !isNaN(numericTarget) && isFinite(numericTarget);
  const [value, setValue] = useState(isNumeric ? 0 : target);
  const startRef = useRef<number | null>(null);
  const rafRef = useRef(0);

  useEffect(() => {
    if (!isNumeric) {
      setValue(target);
      return;
    }

    const prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (prefersReducedMotion) {
      setValue(numericTarget);
      return;
    }

    startRef.current = null;
    const isInt = Number.isInteger(numericTarget);

    const animate = (ts: number) => {
      if (startRef.current === null) startRef.current = ts;
      const elapsed = ts - startRef.current;
      const progress = Math.min(elapsed / duration, 1);
      // Ease-out cubic
      const eased = 1 - Math.pow(1 - progress, 3);
      const current = eased * numericTarget;
      setValue(isInt ? Math.round(current) : parseFloat(current.toFixed(1)));
      if (progress < 1) {
        rafRef.current = requestAnimationFrame(animate);
      }
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [numericTarget, isNumeric, duration, target]);

  return String(value);
}
