import type { ReactNode } from 'react';
import { ChevronRight, ExternalLink } from 'lucide-react';
import type { SummaryRow } from '../../../api';
import SignalDetailPanel, { type SignalDetailChartType } from '../../../components/SignalDetailPanel';
import { Sparkline, SparklineLensStateBadge } from '../../../components/Sparkline';
import { signalLabelColor, smaQualityTone } from '../theme';

export type ChartDecisionLens = 'reversal' | 'extremes' | 'pullback';

export const CHART_DECISION_LENS_LS_KEY = 'signals-chart-decision-lens-v1';

export const CHART_DECISION_LENS_OPTIONS: Array<{
  key: ChartDecisionLens;
  label: string;
  shortLabel: string;
  title: string;
}> = [
  {
    key: 'reversal',
    label: 'Reversal zones',
    shortLabel: 'Zones',
    title: 'Current view: BUY and SELL reversal regimes',
  },
  {
    key: 'extremes',
    label: 'Overbought / oversold',
    shortLabel: 'Extremes',
    title: 'Adaptive exhaustion lens: volatility-normalized stretch from equilibrium',
  },
  {
    key: 'pullback',
    label: 'Pullback trend',
    shortLabel: 'Pullback',
    title: 'EMA trend lens: buyable dips in uptrends and sellable bounces in downtrends',
  },
];

export function loadChartDecisionLens(): ChartDecisionLens {
  try {
    const raw = localStorage.getItem(CHART_DECISION_LENS_LS_KEY);
    if (raw === 'extremes' || raw === 'pullback' || raw === 'reversal') return raw;
  } catch { /* ignore unavailable storage */ }
  return 'reversal';
}

interface ChartAssetRowProps {
  row: SummaryRow;
  ticker: string;
  horizons: number[];
  qualityScore: number;
  highlighted?: boolean;
  isExpanded: boolean;
  onToggleExpand: () => void;
  onNavigateChart: () => void;
  detailDefaultChartType?: SignalDetailChartType;
  chartLens?: ChartDecisionLens;
}

const CHART_VIEW_HORIZONS = [1, 3, 7, 30, 90, 180];

export default function ChartAssetRow({
  row,
  ticker,
  horizons,
  qualityScore,
  highlighted,
  isExpanded,
  onToggleExpand,
  onNavigateChart,
  detailDefaultChartType,
  chartLens = 'reversal',
}: ChartAssetRowProps) {
  const label = (row.nearest_label || 'HOLD').toUpperCase();
  const labelColor = signalLabelColor(label);
  const company = row.asset_label.includes('(') ? row.asset_label.split('(')[0].trim() : '';
  const chartHorizons = CHART_VIEW_HORIZONS.filter((h) => horizons.includes(h) || row.horizon_signals[h] || row.horizon_signals[String(h)]);
  const risk = Math.max(0, Math.min(100, row.crash_risk_score ?? 0));
  const riskTone = chartRiskTone(risk);
  const quality = Math.round(qualityScore ?? 50);
  const qualityTone = smaQualityTone(quality);
  const momentumTone = chartMomentumTone(row.momentum_score);
  const momentumText = `${row.momentum_score > 0 ? '+' : ''}${Math.round(row.momentum_score ?? 0)}%`;
  const industryLabel = row.sector || 'Other';

  return (
    <>
      <div
        role="button"
        tabIndex={0}
        onClick={onToggleExpand}
        onKeyDown={(event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            onToggleExpand();
          }
        }}
        className={`group relative overflow-hidden rounded-xl outline-none transition-all duration-200 ${highlighted ? 'aurora-upgrade' : ''}`}
        style={{
          background: isExpanded
            ? `linear-gradient(180deg, ${labelColor}0f, rgba(139,92,246,0.045) 48%, rgba(255,255,255,0.012))`
            : 'linear-gradient(180deg, rgba(255,255,255,0.032), rgba(255,255,255,0.009))',
          border: `1px solid ${isExpanded ? 'rgba(167,139,250,0.42)' : 'rgba(255,255,255,0.06)'}`,
          boxShadow: isExpanded ? `0 14px 42px -28px ${labelColor}, 0 1px 0 rgba(255,255,255,0.05) inset` : '0 1px 0 rgba(255,255,255,0.035) inset',
        }}
      >
        <div aria-hidden className="absolute inset-x-3 top-0 h-px opacity-80" style={{ background: `linear-gradient(90deg, transparent, ${labelColor}66, transparent)` }} />
        <div className="grid w-full min-w-0 gap-2 p-2 xl:grid-cols-[minmax(178px,0.42fr)_minmax(0,2.38fr)_minmax(336px,0.94fr)_30px] xl:items-center 2xl:grid-cols-[minmax(196px,0.40fr)_minmax(0,2.58fr)_minmax(388px,0.98fr)_30px]">
          <div className="flex min-h-[76px] min-w-0 items-center gap-2">
            <span
              className="h-[62px] w-1 flex-shrink-0 rounded-full"
              style={{ background: labelColor, boxShadow: `0 0 15px -4px ${labelColor}` }}
            />
            <div className="min-w-0 flex-1">
              <div className="flex min-w-0 items-center gap-1.5">
                <span className="truncate text-[13.2px] font-bold leading-none tabular-nums text-[var(--text-primary)]">{ticker}</span>
              </div>
              {company && <span className="mt-1 block max-w-[196px] truncate text-[8.7px] leading-tight text-[var(--text-muted)]">{company}</span>}
              <div className="mt-1.5 grid max-w-[190px] grid-cols-3 gap-1">
                <div className="col-span-3">
                  <ChartIdentityChip
                    value={compactChartSignalLabel(label)}
                    color={labelColor}
                    bg={`${labelColor}12`}
                    border={`${labelColor}34`}
                    title={`Current signal: ${label}`}
                  />
                </div>
                <ChartIdentityChip
                  value={momentumText}
                  color={momentumTone.color}
                  bg={momentumTone.bg}
                  border={momentumTone.border}
                  title={`Momentum score: ${momentumText}`}
                />
                <ChartIdentityChip
                  value={`Q ${quality}`}
                  color={qualityTone.color}
                  bg={qualityTone.background}
                  border={qualityTone.border}
                  title={`Business quality: ${quality}`}
                />
                <ChartIdentityChip
                  value={`R ${risk.toFixed(0)}`}
                  color={riskTone.color}
                  bg={riskTone.bg}
                  border={riskTone.border}
                  title={`Crash risk: ${risk.toFixed(0)} (${riskTone.label})`}
                />
              </div>
            </div>
          </div>

          <div
            className="flex h-[76px] min-w-0 items-center rounded-[10px] px-2.5 py-1.5"
            style={{
              background: `linear-gradient(180deg, ${labelColor}10, rgba(255,255,255,0.012))`,
              border: `1px solid ${labelColor}24`,
              boxShadow: `0 0 22px -17px ${labelColor} inset, 0 10px 24px -28px ${labelColor}`,
            }}
          >
            <Sparkline ticker={ticker} width={760} height={60} tail={220} variant={chartLens} fluid />
          </div>

          <div className="flex h-[76px] min-w-0 flex-col">
            <div className="mb-1 flex justify-center">
              <span
                className="max-w-full truncate rounded-full px-3 py-1 text-center text-[8.5px] font-semibold uppercase tracking-[0.12em] backdrop-blur-md"
                style={{
                  color: '#cbd5e1',
                  background: 'linear-gradient(135deg, rgba(15,23,42,0.72), rgba(255,255,255,0.035))',
                  border: '1px solid rgba(255,255,255,0.08)',
                  boxShadow: '0 10px 24px -22px rgba(148,163,184,0.80), inset 0 1px 0 rgba(255,255,255,0.055)',
                }}
                title={industryLabel}
              >
                {industryLabel}
              </span>
            </div>
            <div
              className="flex min-h-0 min-w-0 flex-1 items-stretch gap-1.5 overflow-hidden rounded-[12px] p-1"
              style={{
                background: 'radial-gradient(240px 72px at 0% 0%, rgba(167,139,250,0.08), transparent 58%), linear-gradient(180deg, rgba(255,255,255,0.034), rgba(255,255,255,0.01))',
                border: '1px solid rgba(255,255,255,0.065)',
                boxShadow: 'inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 28px -30px rgba(167,139,250,0.75)',
              }}
            >
              <div className="w-[66px] shrink-0">
                <SparklineLensStateBadge ticker={ticker} tail={220} variant={chartLens} compact tile />
              </div>
              <div className="grid min-w-0 flex-1 grid-cols-[repeat(auto-fit,minmax(58px,1fr))] gap-1">
                {chartHorizons.map((h) => {
                  const sig = row.horizon_signals[h] || row.horizon_signals[String(h)];
                  return (
                    <ChartHorizonMiniTile key={h} label={chartViewHorizonLabel(h)} expRet={sig?.exp_ret} />
                  );
                })}
              </div>
            </div>
          </div>
          <div className="flex items-center justify-end gap-1 xl:flex-col xl:justify-center">
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation();
                onNavigateChart();
              }}
              className="inline-flex h-7 w-7 items-center justify-center rounded-[9px] transition-all hover:brightness-125 active:scale-95"
              style={{
                color: '#c4b5fd',
                background: 'rgba(167,139,250,0.08)',
                border: '1px solid rgba(167,139,250,0.18)',
              }}
              title="Open full chart"
              aria-label={`Open full chart for ${ticker}`}
            >
              <ExternalLink className="h-3.5 w-3.5" />
            </button>
            <ChevronRight
              className="h-3.5 w-3.5 transition-transform duration-200"
              style={{ color: isExpanded ? '#c4b5fd' : 'var(--text-muted)', transform: isExpanded ? 'rotate(90deg)' : 'rotate(0deg)' }}
            />
          </div>
        </div>
      </div>
      {isExpanded && (
        <div className="overflow-hidden rounded-b-xl border-x border-b" style={{ borderColor: 'rgba(167,139,250,0.20)' }}>
          <SignalDetailPanel
            ticker={ticker}
            signal={row.nearest_label}
            momentum={row.momentum_score}
            crashRisk={row.crash_risk_score}
            horizonSignals={row.horizon_signals}
            defaultChartType={detailDefaultChartType ?? 'area'}
            onNavigateChart={onNavigateChart}
          />
        </div>
      )}
    </>
  );
}

function chartViewHorizonLabel(days: number): string {
  if (days === 1) return '1D';
  if (days === 3) return '3D';
  if (days === 7) return '1W';
  if (days === 30) return '1M';
  if (days === 90) return '3M';
  if (days === 180) return '6M';
  return `${days}D`;
}

function compactChartSignalLabel(label: string): string {
  if (label === 'STRONG BUY') return 'STRONG BUY';
  if (label === 'STRONG SELL') return 'STRONG SELL';
  if (label === 'BUY') return 'BUY';
  if (label === 'SELL') return 'SELL';
  return 'HOLD';
}

function chartRiskTone(score: number) {
  const s = Math.max(0, Math.min(100, score));
  if (s < 30) return { label: 'LOW', color: '#34d399', bg: 'rgba(16,185,129,0.10)', border: 'rgba(16,185,129,0.25)' };
  if (s < 55) return { label: 'OK', color: '#facc15', bg: 'rgba(250,204,21,0.105)', border: 'rgba(250,204,21,0.28)' };
  if (s < 75) return { label: 'ELEV', color: '#fb923c', bg: 'rgba(251,146,60,0.115)', border: 'rgba(251,146,60,0.34)' };
  return { label: 'HIGH', color: '#fb7185', bg: 'rgba(244,63,94,0.125)', border: 'rgba(244,63,94,0.38)' };
}

function chartMomentumTone(value: number) {
  const v = value ?? 0;
  if (v > 0) return { color: '#34d399', bg: 'rgba(16,185,129,0.105)', border: 'rgba(16,185,129,0.28)' };
  if (v < -1) return { color: '#fb7185', bg: 'rgba(244,63,94,0.115)', border: 'rgba(244,63,94,0.32)' };
  return { color: 'var(--text-muted)', bg: 'rgba(100,116,139,0.095)', border: 'rgba(100,116,139,0.20)' };
}

function ChartIdentityChip({ value, color, bg, border, title }: {
  value: ReactNode;
  color: string;
  bg: string;
  border: string;
  title?: string;
}) {
  return (
    <span
      className="inline-flex h-[19px] w-full max-w-full items-center justify-center truncate rounded-[7px] px-1.5 text-[8.3px] font-extrabold uppercase tracking-[0.035em] tabular-nums transition-transform duration-150 group-hover:translate-y-[-0.5px]"
      title={title}
      style={{
        color,
        background: `linear-gradient(180deg, ${bg}, rgba(255,255,255,0.012))`,
        border: `1px solid ${border}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.06), 0 6px 14px -12px ${color}`,
      }}
    >
      {value}
    </span>
  );
}

function ChartHorizonMiniTile({ label, expRet }: { label: string; expRet: number | null | undefined }) {
  const pct = expRet == null ? null : expRet * 100;
  const isUp = (pct ?? 0) > 0;
  const isFlat = pct == null || Math.abs(pct) < 0.1;
  const absPct = Math.abs(pct ?? 0);
  const color = isFlat ? 'var(--text-muted)' : isUp ? '#34d399' : '#fb7185';
  const bg = isFlat
    ? 'rgba(100,116,139,0.075)'
    : isUp
      ? `rgba(16,185,129,${Math.min(0.18, 0.075 + absPct / 54)})`
      : `rgba(244,63,94,${Math.min(0.18, 0.075 + absPct / 54)})`;
  const border = isFlat
    ? 'rgba(100,116,139,0.14)'
    : isUp
      ? 'rgba(16,185,129,0.24)'
      : 'rgba(244,63,94,0.26)';
  const value = pct == null
    ? '-'
    : `${pct >= 0 ? '+' : ''}${Math.abs(pct) >= 10 ? pct.toFixed(0) : pct.toFixed(1)}`;

  return (
    <div
      className="relative h-[38px] min-w-0 overflow-hidden rounded-[9px] px-1.5 py-1 text-center"
      title={pct == null ? `${label}: no forecast` : `${label}: ${pct >= 0 ? '+' : ''}${pct.toFixed(2)}%`}
      style={{
        background: `linear-gradient(180deg, ${bg}, rgba(255,255,255,0.012))`,
        border: `1px solid ${border}`,
        boxShadow: `inset 0 1px 0 rgba(255,255,255,0.045), 0 10px 18px -18px ${color}`,
      }}
    >
      <div aria-hidden className="absolute inset-x-2 top-0 h-px" style={{ background: `linear-gradient(90deg, transparent, ${color}66, transparent)` }} />
      <div className="text-[6.6px] font-bold uppercase tracking-[0.09em] text-[var(--text-muted)]">{label}</div>
      <div className="mt-0.5 flex min-w-0 items-baseline justify-center gap-0.5 leading-none" style={{ color }}>
        <span className="min-w-0 truncate text-[9.7px] font-extrabold tabular-nums">{value}</span>
        {pct != null && <span className="text-[6.8px] font-bold opacity-80">%</span>}
      </div>
    </div>
  );
}
