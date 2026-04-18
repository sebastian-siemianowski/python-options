"""
Walk-forward backtester for indicator strategies.
Long-only: buy when signal > buy_threshold, flat otherwise.
Tracks per-asset and aggregate metrics.
"""

import numpy as np
import pandas as pd
from typing import Callable

# Transaction cost in basis points
COST_BPS = 2.0


def backtest_strategy(
    signal: pd.Series,
    close: pd.Series,
    buy_threshold: float = 30.0,
    sell_threshold: float = -30.0,
    cost_bps: float = COST_BPS,
) -> dict:
    """
    Run a single-asset backtest on a signal series.

    Rules:
        - Go long when signal > buy_threshold
        - Go flat when signal < sell_threshold (or below 0 if no explicit sell)
        - Long-only (no shorting)
        - 2 bps transaction cost per trade

    Returns dict of metrics or None if insufficient data.
    """
    valid = signal.notna() & close.notna()
    sig = signal[valid].copy()
    px = close[valid].copy()

    if len(sig) < 60:
        return None

    ret_1 = px.pct_change(1)
    fwd_1 = ret_1.shift(-1)
    fwd_5 = px.pct_change(5).shift(-5)

    # Position: 1 = long, 0 = flat
    pos = pd.Series(0.0, index=sig.index)
    in_trade = False
    for i in range(len(sig)):
        if in_trade:
            if sig.iloc[i] < sell_threshold:
                in_trade = False
                pos.iloc[i] = 0.0
            else:
                pos.iloc[i] = 1.0
        else:
            if sig.iloc[i] > buy_threshold:
                in_trade = True
                pos.iloc[i] = 1.0
            else:
                pos.iloc[i] = 0.0

    # Transaction costs
    trades = pos.diff().abs().fillna(0)
    cost = trades * cost_bps / 10000.0

    # Strategy returns
    strat_ret = (pos * fwd_1 - cost).dropna()
    bh_ret = fwd_1.dropna()

    if len(strat_ret) < 30:
        return None

    # Cumulative performance
    cum = (1 + strat_ret).cumprod()
    bh_cum = (1 + bh_ret).cumprod()

    # CAGR
    n_years = len(strat_ret) / 252
    cagr = (cum.iloc[-1] ** (1 / max(n_years, 0.1)) - 1) * 100 if cum.iloc[-1] > 0 else -99.0
    bh_cagr = (bh_cum.iloc[-1] ** (1 / max(n_years, 0.1)) - 1) * 100 if bh_cum.iloc[-1] > 0 else -99.0

    # Sharpe
    sharpe = (strat_ret.mean() / strat_ret.std()) * np.sqrt(252) if strat_ret.std() > 0 else 0.0
    bh_sharpe = (bh_ret.mean() / bh_ret.std()) * np.sqrt(252) if bh_ret.std() > 0 else 0.0

    # Sortino (downside deviation)
    downside = strat_ret[strat_ret < 0]
    down_std = downside.std() if len(downside) > 5 else strat_ret.std()
    sortino = (strat_ret.mean() / down_std) * np.sqrt(252) if down_std > 0 else 0.0

    # Max drawdown
    peak = cum.cummax()
    dd = ((cum - peak) / peak)
    max_dd = dd.min() * 100

    # Hit rates (buy signals)
    buy_mask = sig > buy_threshold
    sell_mask = sig < sell_threshold
    buy_hit = (fwd_5[buy_mask] > 0).mean() * 100 if buy_mask.sum() > 5 else np.nan
    sell_hit = (fwd_5[sell_mask] < 0).mean() * 100 if sell_mask.sum() > 5 else np.nan

    # Win rate on trades
    trade_rets = strat_ret[pos.shift(1).fillna(0) > 0]
    win_rate = (trade_rets > 0).mean() * 100 if len(trade_rets) > 10 else np.nan

    # Profit factor
    gross_profit = trade_rets[trade_rets > 0].sum()
    gross_loss = abs(trade_rets[trade_rets < 0].sum())
    profit_factor = gross_profit / max(gross_loss, 1e-10)

    # Exposure (% time in market)
    exposure = pos.mean() * 100

    # Number of trades (round trips)
    n_trades = int(trades.sum() / 2)

    return {
        "cagr": round(cagr, 2),
        "bh_cagr": round(bh_cagr, 2),
        "sharpe": round(sharpe, 3),
        "bh_sharpe": round(bh_sharpe, 3),
        "sortino": round(sortino, 3),
        "max_dd": round(max_dd, 2),
        "buy_hit": round(buy_hit, 1) if not np.isnan(buy_hit) else None,
        "sell_hit": round(sell_hit, 1) if not np.isnan(sell_hit) else None,
        "win_rate": round(win_rate, 1) if not np.isnan(win_rate) else None,
        "profit_factor": round(profit_factor, 3),
        "exposure": round(exposure, 1),
        "n_trades": n_trades,
        "total_return": round((cum.iloc[-1] - 1) * 100, 2),
        "bh_return": round((bh_cum.iloc[-1] - 1) * 100, 2),
    }


def evaluate_signal_quality(signal: pd.Series, close: pd.Series) -> dict:
    """Quick signal quality metrics without full backtest."""
    valid = signal.notna() & close.notna()
    sig = signal[valid]
    px = close[valid]
    if len(sig) < 60:
        return {}

    fwd_5 = px.pct_change(5).shift(-5)
    mask = fwd_5.notna()
    s = sig[mask]
    f = fwd_5[mask]

    buy_m = s > 30
    sell_m = s < -30
    buy_avg = f[buy_m].mean() * 100 if buy_m.sum() > 5 else np.nan
    sell_avg = f[sell_m].mean() * 100 if sell_m.sum() > 5 else np.nan

    return {
        "buy_avg_5d": round(buy_avg, 3) if not np.isnan(buy_avg) else None,
        "sell_avg_5d": round(sell_avg, 3) if not np.isnan(sell_avg) else None,
        "signal_mean": round(s.mean(), 2),
        "signal_std": round(s.std(), 2),
        "pct_buy": round((s > 30).mean() * 100, 1),
        "pct_sell": round((s < -30).mean() * 100, 1),
    }
