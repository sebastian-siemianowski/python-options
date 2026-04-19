"""PnL attribution by horizon from walk-forward records.

Extracted from signals.py - Story 8.5.
Contains compute_pnl_attribution() and related constants.
"""

import os
from typing import Dict, List, Optional

import numpy as np

import sys as _sys
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _SRC_DIR not in _sys.path:
    _sys.path.insert(0, _SRC_DIR)

# ─── Story 2.5: Profit-and-Loss Attribution by Horizon ──────────────────
PNL_BUY_THRESHOLD = 0.55    # P(up) above this -> BUY
PNL_SELL_THRESHOLD = 0.45   # P(up) below this -> SELL
PNL_DEFAULT_NOTIONAL = 1e6  # 1M PLN default notional


def compute_pnl_attribution(
    wf_records: list,
    notional: float = PNL_DEFAULT_NOTIONAL,
    horizons: Optional[List[int]] = None,
) -> Dict[int, Dict[str, float]]:
    """
    Compute per-horizon P&L attribution from walk-forward records.

    For each forecast:
        - If forecast_p_up > 0.55:  BUY  -> pnl = notional * (exp(realized_ret) - 1)
        - If forecast_p_up < 0.45:  SELL -> pnl = notional * (1 - exp(realized_ret))
        - Otherwise:                HOLD -> pnl = 0

    Returns per-horizon dict:
        {horizon: {cumulative_pnl, n_trades, hit_rate, sharpe, mean_pnl, std_pnl}}

    Args:
        wf_records: List of WalkForwardRecord from Story 2.1.
        notional: Trade notional (default 1M).
        horizons: Horizons to include (default all found in records).

    Returns:
        {horizon: {cumulative_pnl, n_trades, hit_rate, sharpe, mean_pnl, std_pnl}}
    """
    if horizons is None:
        horizons = sorted(set(r.horizon for r in wf_records))

    result: Dict[int, Dict[str, float]] = {}
    for H in horizons:
        h_recs = [r for r in wf_records if r.horizon == H]
        if not h_recs:
            result[H] = {
                "cumulative_pnl": 0.0, "n_trades": 0, "hit_rate": 0.0,
                "sharpe": 0.0, "mean_pnl": 0.0, "std_pnl": 0.0,
            }
            continue

        pnl_list = []
        hits = 0
        n_trades = 0
        for r in h_recs:
            if r.forecast_p_up > PNL_BUY_THRESHOLD:
                pnl = notional * (np.exp(r.realized_ret) - 1.0)
                n_trades += 1
                if r.realized_ret > 0:
                    hits += 1
            elif r.forecast_p_up < PNL_SELL_THRESHOLD:
                pnl = notional * (1.0 - np.exp(r.realized_ret))
                n_trades += 1
                if r.realized_ret < 0:
                    hits += 1
            else:
                pnl = 0.0
            pnl_list.append(pnl)

        pnl_arr = np.array(pnl_list)
        cumulative = float(np.sum(pnl_arr))
        mean_pnl = float(np.mean(pnl_arr))
        std_pnl = float(np.std(pnl_arr, ddof=1)) if len(pnl_arr) > 1 else 0.0
        hit_rate = hits / n_trades if n_trades > 0 else 0.0

        # Sharpe: mean(pnl) / std(pnl) * sqrt(252/H), annualized
        if std_pnl > 0 and H > 0:
            sharpe = (mean_pnl / std_pnl) * np.sqrt(252.0 / H)
        else:
            sharpe = 0.0

        result[H] = {
            "cumulative_pnl": cumulative,
            "n_trades": n_trades,
            "hit_rate": hit_rate,
            "sharpe": float(sharpe),
            "mean_pnl": mean_pnl,
            "std_pnl": std_pnl,
        }
    return result
