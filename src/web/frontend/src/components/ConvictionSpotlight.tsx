/**
 * ConvictionSpotlight -- Dual nebula-glow panels for strongest buys/sells.
 *
 * Emerald aurora on the buy side, rose aurora on the sell side,
 * with rich mini-cards showing ticker, sector, expected return, p_up arc,
 * Kelly bar, and signal age.
 */
import { useNavigate } from 'react-router-dom';
import type { StrongSignalEntry } from '../api';
import { TrendingUp, TrendingDown, Scale } from 'lucide-react';

interface Props {
  strongBuy: StrongSignalEntry[];
  strongSell: StrongSignalEntry[];
}

function MiniArc({ value, color, size = 16 }: { value: number; color: string; size?: number }) {
  const r = (size - 2) / 2;
  const cx = size / 2;
  const cy = size / 2;
  const arcAngle = 270;
  const startAngle = 135;
  const circumference = (2 * Math.PI * r * arcAngle) / 360;
  const filled = circumference * Math.min(Math.max(value, 0), 1);
  const toRad = (a: number) => (a * Math.PI) / 180;
  const startX = cx + r * Math.cos(toRad(startAngle));
  const startY = cy + r * Math.sin(toRad(startAngle));
  const endAngle = startAngle + arcAngle;
  const endX = cx + r * Math.cos(toRad(endAngle));
  const endY = cy + r * Math.sin(toRad(endAngle));
  const largeArc = arcAngle > 180 ? 1 : 0;
  const d = `M ${startX} ${startY} A ${r} ${r} 0 ${largeArc} 1 ${endX} ${endY}`;
  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      <path d={d} fill="none" stroke="var(--violet-10)" strokeWidth="1.5" strokeLinecap="round" />
      <path d={d} fill="none" stroke={color} strokeWidth="1.5" strokeLinecap="round"
        strokeDasharray={`${filled} ${circumference}`}
      />
    </svg>
  );
}

function EmptyState() {
  return (
    <div className="flex flex-col items-center justify-center py-10 gap-3">
      <div className="animate-float-subtle">
        <Scale className="w-10 h-10" style={{ color: 'var(--text-muted, #6b7a90)' }} strokeWidth={1.2} />
      </div>
      <p className="text-sm" style={{ color: 'var(--text-secondary, #94a3b8)' }}>No strong signals today</p>
      <p className="text-[11px]" style={{ color: 'var(--text-muted, #6b7a90)' }}>Markets in equilibrium</p>
      <style>{`
        @keyframes float-subtle {
          0%, 100% { transform: translateY(0); }
          50% { transform: translateY(-2px); }
        }
        .animate-float-subtle { animation: float-subtle 3s ease-in-out infinite; }
      `}</style>
    </div>
  );
}

function SignalCard({ entry, accent, index }: { entry: StrongSignalEntry; accent: 'emerald' | 'rose'; index: number }) {
  const navigate = useNavigate();
  const isEmerald = accent === 'emerald';
  const accentColor = isEmerald ? 'var(--accent-emerald)' : 'var(--accent-rose)';
  const gradientText = isEmerald
    ? 'linear-gradient(135deg, var(--text-luminous) 0%, var(--accent-emerald) 100%)'
    : 'linear-gradient(135deg, var(--text-luminous) 0%, var(--accent-rose) 100%)';

  // Kelly rough estimate from p_up and exp_ret
  const kelly = Math.abs(entry.exp_ret) > 0 ? Math.min(Math.abs(entry.p_up - 0.5) * 2, 0.5) : 0;
  const isUrgent = Math.abs(entry.exp_ret) > 0.08;
  const cardClass = isEmerald ? 'conviction-buy' : 'conviction-sell';
  const urgentClass = isUrgent ? (isEmerald ? 'conviction-urgent' : 'conviction-urgent-sell') : '';
  const fadeClass = `fade-up-delay-${Math.min(index, 4)}`;

  return (
    <div
      className={`${cardClass} ${urgentClass} ${fadeClass} px-4 py-3 rounded-xl cursor-pointer`}
      onClick={() => navigate(`/charts/${entry.symbol}`)}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="flex-1 min-w-0">
          {/* Ticker */}
          <div
            className="text-section"
            style={{
              background: gradientText,
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              color: 'var(--text-luminous)',
            }}
          >
            {entry.symbol}
          </div>
          {/* Sector badge */}
          <span
            className="inline-block px-2 py-0.5 rounded text-[9px] mt-1"
            style={{ background: 'var(--void-active, #1c1845)', color: 'var(--text-muted, #6b7a90)' }}
          >
            {entry.sector}
          </span>
        </div>

        <div className="text-right flex-shrink-0">
          {/* Expected return */}
          <div className="text-stat-value" style={{ color: accentColor }}>
            {entry.exp_ret >= 0 ? '+' : ''}{(entry.exp_ret * 100).toFixed(1)}%
          </div>
          {/* Horizon */}
          <span className="text-caption">
            {entry.horizon}
          </span>
        </div>
      </div>

      {/* Bottom row: p_up arc + kelly bar + momentum */}
      <div className="flex items-center gap-3 mt-2">
        <div className="flex items-center gap-1">
          <MiniArc value={entry.p_up} color={accentColor} size={16} />
          <span className="text-[10px] tabular-nums" style={{ color: 'var(--text-secondary, #94a3b8)' }}>
            {(entry.p_up * 100).toFixed(0)}%
          </span>
        </div>

        <div className="flex items-center gap-1">
          <div className="w-10 h-[3px] rounded-full overflow-hidden"
            style={{ background: 'var(--void-active, #1c1845)' }}>
            <div className="h-full rounded-full"
              style={{
                width: `${Math.min(kelly * 200, 100)}%`,
                background: `linear-gradient(90deg, ${accentColor}66, ${accentColor})`,
              }}
            />
          </div>
          <span className="text-[9px] tabular-nums" style={{ color: 'var(--text-muted, #6b7a90)' }}>
            K {(kelly * 100).toFixed(0)}%
          </span>
        </div>

        {entry.momentum != null && (
          <span className="text-[9px] tabular-nums ml-auto"
            style={{ color: entry.momentum > 0 ? 'var(--accent-emerald)' : entry.momentum < 0 ? 'var(--accent-rose)' : 'var(--text-muted, #6b7a90)' }}>
            {entry.momentum > 0 ? '+' : ''}{Math.round(entry.momentum)}% mom
          </span>
        )}
      </div>
    </div>
  );
}

export default function ConvictionSpotlight({ strongBuy, strongSell }: Props) {
  const hasBuys = strongBuy.length > 0;
  const hasSells = strongSell.length > 0;

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
      {/* Strongest Buys panel */}
      <div
        className="glass-card overflow-hidden relative"
        style={{
          background: 'linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)',
        }}
      >
        {/* Emerald aurora glow */}
        <div className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at 10% 10%, var(--emerald-8) 0%, transparent 60%)',
          }}
        />
        <div className="relative z-10 p-5">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-6 h-6 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(62,232,165,0.1)' }}>
              <TrendingUp className="w-3.5 h-3.5" style={{ color: 'var(--accent-emerald)' }} />
            </div>
            <h3 className="text-[15px] font-semibold" style={{ color: 'var(--accent-emerald)' }}>
              Strongest Buys
            </h3>
          </div>
          {hasBuys ? (
            <div className="space-y-1">
              {strongBuy.slice(0, 5).map((s, i) => (
                <SignalCard key={`${s.symbol}-${i}`} entry={s} accent="emerald" index={i} />
              ))}
            </div>
          ) : (
            <EmptyState />
          )}
        </div>
      </div>

      {/* Strongest Sells panel */}
      <div
        className="glass-card overflow-hidden relative"
        style={{
          background: 'linear-gradient(135deg, #1a0533 0%, #0d1b3e 40%, #0a2540 100%)',
        }}
      >
        {/* Rose aurora glow */}
        <div className="absolute inset-0 pointer-events-none"
          style={{
            background: 'radial-gradient(ellipse at 90% 10%, var(--rose-8) 0%, transparent 60%)',
          }}
        />
        <div className="relative z-10 p-5">
          <div className="flex items-center gap-2 mb-4">
            <div className="w-6 h-6 rounded-lg flex items-center justify-center"
              style={{ background: 'rgba(255,107,138,0.1)' }}>
              <TrendingDown className="w-3.5 h-3.5" style={{ color: 'var(--accent-rose)' }} />
            </div>
            <h3 className="text-[15px] font-semibold" style={{ color: 'var(--accent-rose)' }}>
              Strongest Sells
            </h3>
          </div>
          {hasSells ? (
            <div className="space-y-1">
              {strongSell.slice(0, 5).map((s, i) => (
                <SignalCard key={`${s.symbol}-${i}`} entry={s} accent="rose" index={i} />
              ))}
            </div>
          ) : (
            <EmptyState />
          )}
        </div>
      </div>
    </div>
  );
}
