"""Test Batch 11: S051-S055 mean reversion strategies."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__) + "/src")

from indicators.registry import get_all_strategies
from indicators.base import get_indicators
from indicators.backtest import backtest_strategy

symbols = ['AAPL', 'NVDA', 'GLD', 'TSLA', 'META']
strats = get_all_strategies()

for sid in [51, 52, 53, 54, 55]:
    info = strats[sid]
    name = info['name']
    fn = info['fn']
    print(f"\n{'='*60}")
    print(f"S{sid:03d} | {name}")
    print(f"{'='*60}")
    for sym in symbols:
        try:
            ind = get_indicators(sym)
            sig = fn(ind)
            close = ind['close']
            bt = backtest_strategy(sig, close)
            mean_abs = sig.abs().mean()
            buy_pct = (sig > 30).mean() * 100
            sell_pct = (sig < -30).mean() * 100
            print(f"  {sym:5s}: trades={bt['n_trades']:3d}  sharpe={bt['sharpe']:+.2f}  "
                  f"cagr={bt['cagr']:+.1%}  maxdd={bt['max_dd']:.1%}  "
                  f"|sig|={mean_abs:.1f}  buy%={buy_pct:.1f}  sell%={sell_pct:.1f}")
        except Exception as e:
            print(f"  {sym:5s}: ERROR - {e}")
