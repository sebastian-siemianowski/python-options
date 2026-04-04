/**
 * AmbientOrbs -- Breathing orbs inspired by lumenlingo's cosmic aesthetic.
 * Soft, slow-moving gradient orbs that create a living background surface.
 * Pure CSS, no JS animation loops. Respects prefers-reduced-motion.
 */
export default function AmbientOrbs() {
  return (
    <div
      className="ambient-orbs-container"
      aria-hidden="true"
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: 0,
        pointerEvents: 'none',
        overflow: 'hidden',
      }}
    >
      {/* Primary violet orb -- top left drift */}
      <div
        className="ambient-orb"
        style={{
          position: 'absolute',
          width: '600px',
          height: '600px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(139,92,246,0.06) 0%, rgba(139,92,246,0.02) 40%, transparent 70%)',
          filter: 'blur(80px)',
          top: '-10%',
          left: '-5%',
          animation: 'orb-float-1 25s ease-in-out infinite',
        }}
      />
      {/* Cyan orb -- bottom right drift */}
      <div
        className="ambient-orb"
        style={{
          position: 'absolute',
          width: '500px',
          height: '500px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(56,217,245,0.04) 0%, rgba(56,217,245,0.015) 40%, transparent 70%)',
          filter: 'blur(80px)',
          bottom: '-10%',
          right: '-5%',
          animation: 'orb-float-2 30s ease-in-out infinite',
        }}
      />
      {/* Fuchsia orb -- center drift */}
      <div
        className="ambient-orb"
        style={{
          position: 'absolute',
          width: '400px',
          height: '400px',
          borderRadius: '50%',
          background: 'radial-gradient(circle, rgba(226,122,245,0.03) 0%, rgba(226,122,245,0.01) 40%, transparent 70%)',
          filter: 'blur(80px)',
          top: '40%',
          left: '30%',
          animation: 'orb-float-3 35s ease-in-out infinite',
        }}
      />
    </div>
  );
}
