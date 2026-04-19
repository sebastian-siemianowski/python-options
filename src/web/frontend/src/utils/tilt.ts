/**
 * tilt.ts -- Lightweight cursor-driven 3D tilt for .hover-lift elements.
 *
 * Sets CSS custom properties on each card:
 *   --tilt-x   (rotateX, vertical axis, max +/-2deg)
 *   --tilt-y   (rotateY, horizontal axis, max +/-2deg)
 *   --light-angle (border luminance gradient angle, 0-360deg)
 *
 * Usage: call initTilt() once in your root component / layout.
 * It uses event delegation on the document, so no per-card listeners needed.
 */

const MAX_TILT = 2; // degrees

function handlePointerMove(e: PointerEvent) {
  const target = (e.target as HTMLElement).closest<HTMLElement>('.hover-lift');
  if (!target) return;

  const rect = target.getBoundingClientRect();
  const x = (e.clientX - rect.left) / rect.width;  // 0..1
  const y = (e.clientY - rect.top) / rect.height;   // 0..1

  // Tilt: center = 0, edges = +/-MAX_TILT
  const tiltX = (0.5 - y) * MAX_TILT;  // top edge tilts toward viewer
  const tiltY = (x - 0.5) * MAX_TILT;  // right edge tilts away

  // Light angle follows cursor position (for border luminance gradient)
  const angle = Math.atan2(y - 0.5, x - 0.5) * (180 / Math.PI) + 180;

  target.style.setProperty('--tilt-x', `${tiltX.toFixed(2)}deg`);
  target.style.setProperty('--tilt-y', `${tiltY.toFixed(2)}deg`);
  target.style.setProperty('--light-angle', `${angle.toFixed(0)}deg`);
}

function handlePointerLeave(e: PointerEvent) {
  const el = e.target;
  if (!(el instanceof Element)) return;
  const target = el.closest<HTMLElement>('.hover-lift');
  if (!target) return;

  target.style.removeProperty('--tilt-x');
  target.style.removeProperty('--tilt-y');
  target.style.removeProperty('--light-angle');
}

let initialized = false;

export function initTilt(): () => void {
  if (initialized) return () => {};
  initialized = true;

  document.addEventListener('pointermove', handlePointerMove, { passive: true });
  document.addEventListener('pointerleave', handlePointerLeave, true);

  return () => {
    document.removeEventListener('pointermove', handlePointerMove);
    document.removeEventListener('pointerleave', handlePointerLeave, true);
    initialized = false;
  };
}
