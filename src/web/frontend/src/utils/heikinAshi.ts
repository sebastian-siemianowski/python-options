import type { OHLCVBar } from '../api';

export function toHeikinAshiBars(bars: OHLCVBar[]): OHLCVBar[] {
  let previousOpen: number | null = null;
  let previousClose: number | null = null;

  return bars.map((bar) => {
    const close = (bar.open + bar.high + bar.low + bar.close) / 4;
    const open =
      previousOpen == null || previousClose == null
        ? (bar.open + bar.close) / 2
        : (previousOpen + previousClose) / 2;
    const high = Math.max(bar.high, open, close);
    const low = Math.min(bar.low, open, close);

    previousOpen = open;
    previousClose = close;

    return {
      ...bar,
      open,
      high,
      low,
      close,
    };
  });
}

export function isHeikinAshiUp(bar: OHLCVBar): boolean {
  return bar.close >= bar.open;
}
