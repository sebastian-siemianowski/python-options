"""
Backtest-related helper utilities extracted from options.py to reduce code volume
and improve reuse. Keeps options.py leaner.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from bs_utils import bsm_call_price


def approximate_backtest_option_10x(ticker, candidate_row, hist, r=0.01):
    """Vectorized approximation: evaluate every eligible start day in parallel.
    Moved out of options.py for maintainability.
    """
    # Import lazily to avoid heavy imports at module load in other contexts
    import numpy as _np
    import pandas as _pd

    strike = float(candidate_row['strike'])
    dte = int(candidate_row['dte'])
    if dte <= 0:
        return pd.DataFrame(), {}

    # Ensure required columns
    df = hist.copy()
    if 'rv21' not in df.columns:
        df['ret'] = df['Close'].pct_change()
        df['rv21'] = df['ret'].rolling(21).std() * np.sqrt(252)
        df['rv21'] = df['rv21'].fillna(method='bfill').fillna(df['rv21'].median())

    n = len(df)
    start_idx = 252  # warmup for volatility
    end_idx = n - dte
    if end_idx - start_idx <= 0:
        return pd.DataFrame(), {}

    idx = np.arange(start_idx, end_idx, dtype=int)
    buy_dates = df['Date'].to_numpy()[idx]
    S_buy = df['Close'].to_numpy()[idx].astype(float)
    sigmas = df['rv21'].to_numpy()[idx].astype(float)
    T_buy = max(1/252.0, dte/252.0)

    # Model entry premium via vectorized BSM
    prices = bsm_call_price(S_buy, strike, T_buy, r, sigmas)
    # Filter non-positive premiums
    valid = prices > 0
    if not np.any(valid):
        return pd.DataFrame(), {}

    idx_v = idx[valid]
    buy_dates_v = buy_dates[valid]
    S_buy_v = S_buy[valid]
    prices_v = prices[valid]

    # Required expiry threshold for 10x
    thresh_v = strike + 10.0 * prices_v
    # Actual expiry prices
    S_exp_all = df['Close'].to_numpy().astype(float)
    S_exp_v = S_exp_all[idx_v + dte]

    hit_v = (S_exp_v >= thresh_v).astype(int)
    payoff_v = np.maximum(S_exp_v - strike, 0.0)
    ret_x_v = np.divide(payoff_v, prices_v, out=np.zeros_like(payoff_v), where=prices_v>0)

    df_res = pd.DataFrame({
        'buy_date': buy_dates_v,
        'S_buy': S_buy_v,
        'price_model': prices_v,
        'S_exp': S_exp_v,
        'hit_10x': hit_v,
        'ret_x': ret_x_v,
    })

    hits = int(hit_v.sum())
    tries = int(len(df_res))
    hit_rate = float(hits / tries) if tries > 0 else 0.0
    avg_return_x = float(np.mean(ret_x_v)) if tries > 0 else 0.0
    metrics = {'tries': tries, 'hits': hits, 'hit_rate': hit_rate, 'avg_return_x': avg_return_x}
    return df_res, metrics
